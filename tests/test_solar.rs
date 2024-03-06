use anyhow::Result;
use candle_core::Device;
use solar_rs::solar;

#[test]
#[cfg(not(feature = "exclude_from_ci"))]
fn test_load_model_generate() -> Result<()> {
    let device = Device::Cpu;
    let model_path = "resources/DataVortexS-10.7B-dpo-v1.6";
    let (model, tokenizer) =
        solar::load_model(model_path, "solar-datavortexs-10.7b-dpo-v1.6-quantized-q4_1.gguf", &device)?;
    // let max_seq_len:usize = 100;
    //
    // let prompt = r###"<|im_start|>user 대한민국의 수도는 어디야?<|im_end|>"###.to_string();
    // solar::generate(prompt, tokenizer, model, max_seq_len, &device);

    Ok(())
}
