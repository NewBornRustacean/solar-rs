use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{Device, Result};
use clap::ValueEnum;
use rayon::prelude::*;

#[derive(ValueEnum, Debug, Clone)]
pub enum QuantizationMode {
    /// The default quantization includes all 2d tensors, except the output tensor which always
    /// uses Q6_K.
    Llama,
}

impl QuantizationMode {
    pub fn quantize(&self, name: &str, tensor: QTensor, dtype: GgmlDType) -> Result<QTensor> {
        match self {
            Self::Llama => {
                // Same behavior as the llama.cpp quantization.
                let should_quantize = name.ends_with(".weight") && tensor.rank() == 2;
                if should_quantize {
                    let tensor = tensor.dequantize(&Device::Cpu)?;
                    if name == "output.weight" {
                        QTensor::quantize(&tensor, GgmlDType::Q6K)
                    } else {
                        QTensor::quantize(&tensor, dtype)
                    }
                } else {
                    Ok(tensor)
                }
            },
        }
    }
}

#[derive(ValueEnum, Debug, Clone)]
pub enum Quantization {
    #[value(name = "q4_0")]
    Q4_0,
    #[value(name = "q4_1")]
    Q4_1,
    #[value(name = "q5_0")]
    Q5_0,
    #[value(name = "q5_1")]
    Q5_1,
    #[value(name = "q8_0")]
    Q8_0,
    #[value(name = "q8_1")]
    Q8_1,
    Q2k,
    Q3k,
    Q4k,
    Q5k,
    Q6k,
    Q8k,
    F16,
    F32,
}

impl Quantization {
    fn dtype(&self) -> GgmlDType {
        match self {
            Quantization::Q4_0 => GgmlDType::Q4_0,
            Quantization::Q4_1 => GgmlDType::Q4_1,
            Quantization::Q5_0 => GgmlDType::Q5_0,
            Quantization::Q5_1 => GgmlDType::Q5_1,
            Quantization::Q8_0 => GgmlDType::Q8_0,
            Quantization::Q8_1 => GgmlDType::Q8_1,
            Quantization::Q2k => GgmlDType::Q2K,
            Quantization::Q3k => GgmlDType::Q3K,
            Quantization::Q4k => GgmlDType::Q4K,
            Quantization::Q5k => GgmlDType::Q5K,
            Quantization::Q6k => GgmlDType::Q6K,
            Quantization::Q8k => GgmlDType::Q8K,
            Quantization::F16 => GgmlDType::F16,
            Quantization::F32 => GgmlDType::F32,
        }
    }
}

#[derive(ValueEnum, Debug, Clone)]
pub enum Format {
    Safetensors,
    Npz,
    Ggml,
    Gguf,
    Pth,
    Pickle,
}

impl Format {
    pub fn infer<P: AsRef<std::path::Path>>(p: P) -> Option<Self> {
        p.as_ref().extension().and_then(|e| e.to_str()).and_then(|e| match e {
            // We don't infer any format for .bin as it can be used for ggml/gguf or pytorch.
            "safetensors" | "safetensor" => Some(Self::Safetensors),
            "npz" => Some(Self::Npz),
            "pth" | "pt" => Some(Self::Pth),
            "ggml" => Some(Self::Ggml),
            "gguf" => Some(Self::Gguf),
            _ => None,
        })
    }
}

pub fn display_tensors(
    file: &std::path::PathBuf,
    format: Option<Format>,
    verbose: bool,
    device: &Device,
) -> Result<()> {
    let format = match format {
        Some(format) => format,
        None => match Format::infer(file) {
            Some(format) => format,
            None => {
                println!("{file:?}: cannot infer format from file extension, use the --format flag");
                return Ok(());
            },
        },
    };
    match format {
        Format::Npz => {
            let tensors = candle_core::npy::NpzTensors::new(file)?;
            let mut names = tensors.names();
            names.sort();
            for name in names {
                let shape_dtype = match tensors.get_shape_and_dtype(name) {
                    Ok((shape, dtype)) => format!("[{shape:?}; {dtype:?}]"),
                    Err(err) => err.to_string(),
                };
                println!("{name}: {shape_dtype}")
            }
        },
        Format::Safetensors => {
            let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(file)? };
            let mut tensors = tensors.tensors();
            tensors.sort_by(|a, b| a.0.cmp(&b.0));
            for (name, view) in tensors.iter() {
                let dtype = view.dtype();
                let dtype = match candle_core::DType::try_from(dtype) {
                    Ok(dtype) => format!("{dtype:?}"),
                    Err(_) => format!("{dtype:?}"),
                };
                let shape = view.shape();
                println!("{name}: [{shape:?}; {dtype}]")
            }
        },
        Format::Pth => {
            let mut tensors = candle_core::pickle::read_pth_tensor_info(file, verbose, None)?;
            tensors.sort_by(|a, b| a.name.cmp(&b.name));
            for tensor_info in tensors.iter() {
                println!("{}: [{:?}; {:?}]", tensor_info.name, tensor_info.layout.shape(), tensor_info.dtype,);
                if verbose {
                    println!("    {:?}", tensor_info);
                }
            }
        },

        Format::Pickle => {
            let file = std::fs::File::open(file)?;
            let mut reader = std::io::BufReader::new(file);
            let mut stack = candle_core::pickle::Stack::empty();
            stack.read_loop(&mut reader)?;
            for (i, obj) in stack.stack().iter().enumerate() {
                println!("{i} {obj:?}");
            }
        },
        Format::Ggml => {
            let mut file = std::fs::File::open(file)?;
            let content = candle_core::quantized::ggml_file::Content::read(&mut file, device)?;
            let mut tensors = content.tensors.into_iter().collect::<Vec<_>>();
            tensors.sort_by(|a, b| a.0.cmp(&b.0));
            for (name, qtensor) in tensors.iter() {
                println!("{name}: [{:?}; {:?}]", qtensor.shape(), qtensor.dtype());
            }
        },
        Format::Gguf => {
            let mut file = std::fs::File::open(file)?;
            let content = gguf_file::Content::read(&mut file)?;
            if verbose {
                let mut metadata = content.metadata.into_iter().collect::<Vec<_>>();
                metadata.sort_by(|a, b| a.0.cmp(&b.0));
                println!("metadata entries ({})", metadata.len());
                for (key, value) in metadata.iter() {
                    println!("  {key}: {value:?}");
                }
            }
            let mut tensors = content.tensor_infos.into_iter().collect::<Vec<_>>();
            tensors.sort_by(|a, b| a.0.cmp(&b.0));
            for (name, info) in tensors.iter() {
                println!("{name}: [{:?}; {:?}]", info.shape, info.ggml_dtype);
            }
        },
    }
    Ok(())
}

pub fn safetensors_to_gguf(
    in_files: &[std::path::PathBuf],
    out_file: std::path::PathBuf,
    q: Quantization,
) -> Result<()> {
    let mut out_file = std::fs::File::create(out_file)?;
    let mut tensors = std::collections::HashMap::new();
    for in_file in in_files.iter() {
        let in_tensors = candle_core::safetensors::load(in_file, &Device::Cpu)?;
        tensors.extend(in_tensors)
    }
    println!("tensors: {}", tensors.len());

    let dtype = q.dtype();
    let block_size = dtype.block_size();

    let qtensors = tensors
        .into_par_iter()
        .map(|(name, tensor)| {
            let should_quantize = tensor.rank() == 2 && tensor.dim(1)? % block_size == 0;
            println!("  quantizing {name} {tensor:?} {should_quantize}");
            let tensor = if should_quantize {
                QTensor::quantize(&tensor, dtype)?
            } else {
                QTensor::quantize(&tensor, GgmlDType::F32)?
            };
            Ok((name, tensor))
        })
        .collect::<Result<Vec<_>>>()?;
    let qtensors = qtensors.iter().map(|(k, v)| (k.as_str(), v)).collect::<Vec<_>>();

    gguf_file::write(&mut out_file, &[], &qtensors)?;

    Ok(())
}

fn run_dequantize(in_file: std::path::PathBuf, out_file: std::path::PathBuf, device: &Device) -> Result<()> {
    let mut in_file = std::fs::File::open(in_file)?;
    let content = gguf_file::Content::read(&mut in_file)?;
    let mut tensors = std::collections::HashMap::new();
    for (tensor_name, _) in content.tensor_infos.iter() {
        let tensor = content.tensor(&mut in_file, tensor_name, device)?;
        let tensor = tensor.dequantize(device)?;
        tensors.insert(tensor_name.to_string(), tensor);
    }
    candle_core::safetensors::save(&tensors, out_file)?;
    Ok(())
}

pub fn get_files_with_extension(dir: &std::path::Path, extension: &str) -> Vec<std::path::PathBuf> {
    let mut result = Vec::new();

    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        if ext == extension {
                            result.push(path);
                        }
                    }
                }
            }
        }
    }

    result
}

/// original source from: https://github.com/huggingface/candle/blob/main/candle-examples/src/token_output_stream.rs#L5
/// This is a wrapper around a tokenizer to ensure that tokens can be returned to the user in a
/// streaming way rather than having to wait for the full decoding.
pub struct TokenOutputStream {
    tokenizer: tokenizers::Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}
impl TokenOutputStream {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn into_inner(self) -> tokenizers::Tokenizer {
        self.tokenizer
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => Ok(str),
            Err(err) => candle_core::bail!("cannot decode: {err}"),
        }
    }

    // https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_rest(&self) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_all(&self) -> Result<String> {
        self.decode(&self.tokens)
    }

    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token_s).copied()
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}
