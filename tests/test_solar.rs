use candle_core::Device;
use solar_rs::solar;
use anyhow::Result;

#[test]
fn test_load_model_forward()-> Result<()>{
    let model_path = "resources/DataVortexS-10.7B-dpo-v1.6";
    let (model, tokenizer) = solar::load_model(
        model_path,
        "solar-datavortexs-10.7b-dpo-v1.6-quantized-q4_1.gguf",
          &Device::Cpu
    )?;
    Ok(())
}