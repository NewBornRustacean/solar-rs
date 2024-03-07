use std::fs;
use std::io::Write;
use std::path::Path;

use anyhow::Result;
use candle_core::Device;
use tempfile::tempdir;

use solar_rs::utils::{
    display_tensors, get_files_with_extension, safetensors_to_gguf, Format, Quantization, QuantizationMode,
};

#[test]
fn test_get_files_with_extension() {
    // Create a temporary directory
    let temp_dir = tempdir().expect("Failed to create temporary directory");

    // Create files with different extensions
    let file_names = ["file1.safetensors", "file2.txt", "file3.safetensors"];
    for &file_name in &file_names {
        let mut file = fs::File::create(temp_dir.path().join(file_name)).expect("Failed to create file");
        writeln!(file, "Test data").expect("Failed to write to file");
    }

    // Call the function under test
    let files = get_files_with_extension(temp_dir.path(), "safetensors");

    // Assert that the function returned the correct number of files
    assert_eq!(files.len(), 2);

    // Assert that the function returned the correct file paths
    let expected_paths = [
        temp_dir.path().join("file1.safetensors"),
        temp_dir.path().join("file3.safetensors"),
    ];
    for path in &expected_paths {
        assert!(files.contains(path));
    }
}

#[test]
#[ignore]
#[cfg(not(feature = "exclude_from_ci"))]
fn test_safetensors_to_gguf() -> Result<()> {
    let resource_path = Path::new("D:/models/DataVortexS-10.7B-dpo-v1.11");
    let in_files = get_files_with_extension(resource_path, "safetensors");
    let out_file = resource_path.join("solar-datavortexs-10.7b-dpo-v1.11-quantized-q4_0.gguf");
    safetensors_to_gguf(&in_files, out_file.to_path_buf(), Quantization::Q4_0)?;
    Ok(())
}

#[test]
#[cfg(not(feature = "exclude_from_ci"))]
fn test_display_tensors() -> Result<()> {
    let resource_path = Path::new(
        "resources/DataVortexS-10.7B-dpo-v1.11/solar-datavortexs-10.7b-dpo-v1.11-quantized-q4_0.gguf",
    );
    display_tensors(&resource_path.to_path_buf(), Option::from(Format::Gguf), true, &Device::Cpu);
    Ok(())
}
