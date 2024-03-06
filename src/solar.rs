use std::io::Write;
use std::path::Path;

use anyhow::{Error, Result};
use candle_core::{DType, Device, IndexOp};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama2_c::{Cache, Config};
use candle_transformers::models::quantized_llama2_c::QLlama;
use candle_transformers::quantized_var_builder;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::utils;

// Define a struct to represent the configuration
#[derive(Debug, Deserialize, Serialize)]
pub struct SolarConfig {
    pub _name_or_path: String,
    pub architectures: Vec<String>,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub initializer_range: f64,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub model_type: String,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub pad_token_id: usize,
    pub pretraining_tp: usize,
    pub rms_norm_eps: f64,
    pub rope_scaling: Option<f64>,
    pub rope_theta: f64,
    pub tie_word_embeddings: bool,
    pub torch_dtype: String,
    pub transformers_version: String,
    pub use_cache: bool,
    pub vocab_size: usize,
}

impl SolarConfig {
    pub fn default() -> Self {
        Self {
            _name_or_path: "Edentns/DataVortexS-10.7B-dpo-v1.6".to_string(),
            architectures: vec!["LlamaForCausalLM".to_string()],
            attention_bias: false,
            attention_dropout: 0.0,
            bos_token_id: 1,
            eos_token_id: 32000,
            hidden_act: "silu".to_string(),
            hidden_size: 4096,
            initializer_range: 0.02,
            intermediate_size: 14336,
            max_position_embeddings: 4096,
            model_type: "llama".to_string(),
            num_attention_heads: 32,
            num_hidden_layers: 48,
            num_key_value_heads: 8,
            pad_token_id: 2,
            pretraining_tp: 1,
            rms_norm_eps: 1e-05,
            rope_scaling: None,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            torch_dtype: "float16".to_string(),
            transformers_version: "4.36.2".to_string(),
            use_cache: true,
            vocab_size: 48000,
        }
    }

    pub fn load(file_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(file_path)?;
        let reader = std::io::BufReader::new(file);

        let config = serde_json::from_reader(reader)?;

        Ok(config)
    }
}

/// Loads a model from the given path and prepares it for use.
///
/// This function takes in a `model_path` which is the path to the parent directory of the model's .safetensors file.
/// It then constructs the paths to the quantized model and the tokenizer using this `model_path`.
/// The quantized model and the tokenizer are loaded into memory and prepared for use.
///
/// # Arguments
///
/// * `model_path` - A string slice that holds the path to the parent directory of the model's .safetensors file.
/// * `device` - The device where the model will be loaded.
///
/// # Returns
///
/// * `Result<()>` - This function returns a `Result` that indicates whether the model was successfully loaded.
///
/// # Errors
///
/// This function will return an error if the model or the tokenizer could not be loaded from the given `model_path`.
pub fn load_model(model_path: &str, quantized_model_name: &str, device: &Device) -> Result<(QLlama, Tokenizer, Cache)> {
    println!("--Start to load a quantized model..");
    let START_TIME = std::time::Instant::now();

    let tokenizer_path = Path::new(model_path).join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;

    let quantized_model_path = Path::new(model_path).join(quantized_model_name);
    let qvb = quantized_var_builder::VarBuilder::from_gguf(quantized_model_path, &Device::Cpu)?;

    let (_vocab_size, dim) = qvb.get_no_shape("model.embed_tokens.weight")?.shape().dims2()?;

    let solar_config = SolarConfig::default();
    let config = Config {
        dim: dim,                                      // transformer dimension
        hidden_dim: solar_config.intermediate_size,    // for ffn layers
        n_layers: solar_config.num_hidden_layers,      // number of layers
        n_heads: solar_config.num_attention_heads,     // number of query heads
        n_kv_heads: solar_config.num_key_value_heads, // number of key/value heads (can be < query heads because of multiquery)
        vocab_size: _vocab_size,                      // vocabulary size, usually 256 (byte-level)
        seq_len: solar_config.max_position_embeddings, // max sequence length
        norm_eps: solar_config.rms_norm_eps,
    };

    let fake_vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
    let cache = Cache::new(false, &config, fake_vb)?;
    let model = QLlama::load(qvb, config)?;

    println!("--model loaded in {:.3?} sec.", START_TIME.elapsed());
    Ok((model, tokenizer, cache))
}

fn format_size(size_in_bytes: usize) -> String {
    if size_in_bytes < 1_000 {
        format!("{}B", size_in_bytes)
    } else if size_in_bytes < 1_000_000 {
        format!("{:.2}KB", size_in_bytes as f64 / 1e3)
    } else if size_in_bytes < 1_000_000_000 {
        format!("{:.2}MB", size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", size_in_bytes as f64 / 1e9)
    }
}

pub fn generate(
    prompt: String,
    tokenizer: Tokenizer,
    model: QLlama,
    max_seq_len: usize,
    cache: &mut Cache,
    device: &Device,
) -> Result<()> {
    let SEED: u64 = 299792458;
    let repeat_last_n: usize = 32;
    let repeat_penalty: f32 = 1.2;
    let temperature = 1.5;
    let top_p = 0.95;
    let mut logits_processor = LogitsProcessor::new(SEED, Some(temperature), Some(top_p));
    let mut index_pos = 0;
    let mut response: Vec<String> = vec![];

    let mut tokenizer_stream = utils::TokenOutputStream::new(tokenizer.clone());

    let mut tokens = tokenizer
        .encode(prompt.clone(), true)
        .map_err(|_| Error::msg("failed to parse tokens"))?
        .get_ids()
        .to_vec();

    let eos_token = match tokenizer_stream.get_token("<|im_end|>") {
        Some(token) => token,
        None => anyhow::bail!("cannot find the endoftext token"),
    };

    let start_gen = std::time::Instant::now();
    for index in 0..max_seq_len {
        let (context_size, context_index) = {
            if index > 0 {
                (1, index_pos)
            } else {
                (tokens.len(), 0)
            }
        };

        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = candle_core::Tensor::new(ctxt, device)?.unsqueeze(0)?;
        let logits = model.forward(&input, context_index, cache)?;
        let logits = logits.i((0, logits.dim(1)? - 1))?;
        let logits = if tokens.is_empty() {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(&logits, repeat_penalty, &tokens[start_at..])?
        };
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);

        if next_token == eos_token {
            break;
        }

        if let Some(t) = tokenizer_stream.next_token(next_token)? {
            print!("{t}");
            response.push(t);
            std::io::stdout().flush()?;
        }
    }
    if let Some(rest) = tokenizer_stream.decode_rest()? {
        print!("{rest}");
    }
    let dt = start_gen.elapsed();
    println!(
        "\n{} tokens generated ({:.2} token/s)\n",
        tokens.len(),
        tokens.len() as f64 / dt.as_secs_f64(),
    );
    Ok(())
}
