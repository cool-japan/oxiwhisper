use crate::tensor::Tensor;
use half::f16;
use std::collections::HashMap;
use std::io::{BufReader, Read};
use std::path::Path;

/// Whisper model hyperparameters (from GGML header)
#[derive(Debug, Clone)]
pub struct Hparams {
    pub n_vocab: usize,
    pub n_audio_ctx: usize,
    pub n_audio_state: usize,
    pub n_audio_head: usize,
    pub n_audio_layer: usize,
    pub n_text_ctx: usize,
    pub n_text_state: usize,
    pub n_text_head: usize,
    pub n_text_layer: usize,
    pub n_mels: usize,
    pub ftype: i32,
}

/// Token vocabulary entry
#[derive(Debug, Clone)]
pub struct VocabEntry {
    pub text: String,
}

/// All loaded model data
pub struct ModelData {
    pub hparams: Hparams,
    pub mel_filters: Vec<f32>,
    pub vocab: Vec<VocabEntry>,
    pub tensors: HashMap<String, Tensor>,
    /// Quantized tensors kept in their original GGML format (Q4_0 / Q8_0).
    /// Large 2D weight matrices are stored here instead of being dequantized to f32.
    pub quantized_tensors: HashMap<String, crate::quantize::QuantizedTensor>,
}

impl ModelData {
    /// Load a GGML whisper model file
    pub fn load(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("Cannot open model: {e}"))?;
        let mut reader = BufReader::new(file);

        // Read magic
        let magic = read_u32(&mut reader)?;
        if magic != 0x67676D6C {
            return Err(format!(
                "Invalid magic: expected 0x67676D6C (ggml), got 0x{magic:08X}"
            ));
        }

        // Read hyperparameters
        let hparams = Hparams {
            n_vocab: read_i32(&mut reader)? as usize,
            n_audio_ctx: read_i32(&mut reader)? as usize,
            n_audio_state: read_i32(&mut reader)? as usize,
            n_audio_head: read_i32(&mut reader)? as usize,
            n_audio_layer: read_i32(&mut reader)? as usize,
            n_text_ctx: read_i32(&mut reader)? as usize,
            n_text_state: read_i32(&mut reader)? as usize,
            n_text_head: read_i32(&mut reader)? as usize,
            n_text_layer: read_i32(&mut reader)? as usize,
            n_mels: read_i32(&mut reader)? as usize,
            ftype: read_i32(&mut reader)?,
        };

        #[cfg(feature = "timing")]
        eprintln!("Model hparams: {:?}", hparams);

        // Read mel filters
        let n_mel_filters = read_i32(&mut reader)? as usize;
        let n_mel_len = read_i32(&mut reader)? as usize;
        let mel_total = n_mel_filters * n_mel_len;
        let mut mel_filters = vec![0.0f32; mel_total];
        read_f32_slice(&mut reader, &mut mel_filters)?;

        #[cfg(feature = "timing")]
        eprintln!(
            "Mel filters: {} x {} = {} values",
            n_mel_filters, n_mel_len, mel_total
        );

        // Read vocabulary
        let n_vocab = read_i32(&mut reader)? as usize;
        let mut vocab = Vec::with_capacity(n_vocab);
        for _ in 0..n_vocab {
            let len = read_i32(&mut reader)? as usize;
            let mut buf = vec![0u8; len];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("read vocab: {e}"))?;
            vocab.push(VocabEntry {
                text: String::from_utf8_lossy(&buf).into_owned(),
            });
        }

        #[cfg(feature = "timing")]
        eprintln!("Vocabulary: {} tokens", vocab.len());

        // Read tensors
        let mut tensors = HashMap::new();
        let mut quantized_tensors = HashMap::new();
        #[cfg(feature = "timing")]
        let mut tensor_count = 0usize;
        #[cfg(feature = "timing")]
        let mut quant_count = 0usize;

        while let Some(v) = try_read_i32(&mut reader) {
            let n_dims = v as usize;

            let name_len = read_i32(&mut reader)? as usize;
            let dtype = read_i32(&mut reader)?;

            let mut dims = vec![0i32; n_dims];
            for d in dims.iter_mut() {
                *d = read_i32(&mut reader)?;
            }

            let mut name_buf = vec![0u8; name_len];
            reader
                .read_exact(&mut name_buf)
                .map_err(|e| format!("read tensor name: {e}"))?;
            let name = String::from_utf8_lossy(&name_buf)
                .trim_end_matches('\0')
                .to_string();

            // Calculate total elements
            let shape: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
            let n_elements: usize = shape.iter().product();

            // Determine if this is a large 2D weight suitable for quantized storage
            let is_large_2d_weight = n_dims == 2 && n_elements > 1024;

            // Read data
            match dtype {
                0 => {
                    // f32
                    let mut f32_data = vec![0.0f32; n_elements];
                    read_f32_slice(&mut reader, &mut f32_data)?;
                    let tensor = Tensor::from_vec(f32_data, &shape);
                    tensors.insert(name.clone(), tensor);
                }
                1 => {
                    // f16
                    let mut raw = vec![0u16; n_elements];
                    let byte_slice = unsafe {
                        std::slice::from_raw_parts_mut(raw.as_mut_ptr() as *mut u8, n_elements * 2)
                    };
                    reader
                        .read_exact(byte_slice)
                        .map_err(|e| format!("read f16 data: {e}"))?;
                    let f32_data: Vec<f32> = raw
                        .iter()
                        .map(|&bits| f16::from_bits(bits).to_f32())
                        .collect();
                    let tensor = Tensor::from_vec(f32_data, &shape);
                    tensors.insert(name.clone(), tensor);
                }
                2 => {
                    // Q4_0: 4-bit quantized
                    use crate::quantize::{Q4_0_BLOCK_BYTES, Q4_0_BLOCK_SIZE};
                    let n_blocks = n_elements / Q4_0_BLOCK_SIZE;
                    let n_bytes = n_blocks * Q4_0_BLOCK_BYTES;
                    let mut raw = vec![0u8; n_bytes];
                    reader
                        .read_exact(&mut raw)
                        .map_err(|e| format!("read Q4_0 data: {e}"))?;

                    if is_large_2d_weight {
                        // Store in quantized format to save memory
                        quantized_tensors.insert(
                            name.clone(),
                            crate::quantize::QuantizedTensor {
                                raw,
                                shape,
                                qtype: crate::quantize::QuantType::Q4_0,
                            },
                        );
                        #[cfg(feature = "timing")]
                        {
                            quant_count += 1;
                        }
                    } else {
                        // Small tensor: dequantize to f32
                        let f32_data = crate::quantize::dequantize_q4_0(&raw, n_elements);
                        let tensor = Tensor::from_vec(f32_data, &shape);
                        tensors.insert(name.clone(), tensor);
                    }
                }
                3 => {
                    // Q8_0: 8-bit quantized
                    use crate::quantize::{Q8_0_BLOCK_BYTES, Q8_0_BLOCK_SIZE};
                    let n_blocks = n_elements / Q8_0_BLOCK_SIZE;
                    let n_bytes = n_blocks * Q8_0_BLOCK_BYTES;
                    let mut raw = vec![0u8; n_bytes];
                    reader
                        .read_exact(&mut raw)
                        .map_err(|e| format!("read Q8_0 data: {e}"))?;

                    if is_large_2d_weight {
                        // Store in quantized format to save memory
                        quantized_tensors.insert(
                            name.clone(),
                            crate::quantize::QuantizedTensor {
                                raw,
                                shape,
                                qtype: crate::quantize::QuantType::Q8_0,
                            },
                        );
                        #[cfg(feature = "timing")]
                        {
                            quant_count += 1;
                        }
                    } else {
                        // Small tensor: dequantize to f32
                        let f32_data = crate::quantize::dequantize_q8_0(&raw, n_elements);
                        let tensor = Tensor::from_vec(f32_data, &shape);
                        tensors.insert(name.clone(), tensor);
                    }
                }
                6 => {
                    // Q5_0: 5-bit quantized
                    use crate::quantize::{Q5_0_BLOCK_BYTES, Q5_0_BLOCK_SIZE};
                    let n_blocks = n_elements / Q5_0_BLOCK_SIZE;
                    let n_bytes = n_blocks * Q5_0_BLOCK_BYTES;
                    let mut raw = vec![0u8; n_bytes];
                    reader
                        .read_exact(&mut raw)
                        .map_err(|e| format!("read Q5_0 data: {e}"))?;

                    if is_large_2d_weight {
                        // Store in quantized format to save memory
                        quantized_tensors.insert(
                            name.clone(),
                            crate::quantize::QuantizedTensor {
                                raw,
                                shape,
                                qtype: crate::quantize::QuantType::Q5_0,
                            },
                        );
                        #[cfg(feature = "timing")]
                        {
                            quant_count += 1;
                        }
                    } else {
                        // Small tensor: dequantize to f32
                        let f32_data = crate::quantize::dequantize_q5_0(&raw, n_elements);
                        let tensor = Tensor::from_vec(f32_data, &shape);
                        tensors.insert(name.clone(), tensor);
                    }
                }
                _ => {
                    return Err(format!("Unsupported tensor dtype: {dtype} for {name}"));
                }
            }

            #[cfg(feature = "timing")]
            {
                tensor_count += 1;
            }
        }

        #[cfg(feature = "timing")]
        eprintln!(
            "Loaded {} tensors ({} quantized)",
            tensor_count, quant_count
        );

        Ok(ModelData {
            hparams,
            mel_filters,
            vocab,
            tensors,
            quantized_tensors,
        })
    }

    pub fn get(&self, name: &str) -> Result<&Tensor, String> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("Missing tensor: {name}"))
    }

    pub fn try_get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Look up a quantized tensor by name.
    pub fn get_quantized(&self, name: &str) -> Option<&crate::quantize::QuantizedTensor> {
        self.quantized_tensors.get(name)
    }

    /// Alias for `get_quantized`.
    pub fn try_get_quantized(&self, name: &str) -> Option<&crate::quantize::QuantizedTensor> {
        self.quantized_tensors.get(name)
    }
}

fn read_i32<R: Read>(reader: &mut R) -> Result<i32, String> {
    let mut buf = [0u8; 4];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("read_i32: {e}"))?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("read_u32: {e}"))?;
    Ok(u32::from_le_bytes(buf))
}

fn try_read_i32<R: Read>(reader: &mut R) -> Option<i32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf).ok()?;
    Some(i32::from_le_bytes(buf))
}

fn read_f32_slice<R: Read>(reader: &mut R, dst: &mut [f32]) -> Result<(), String> {
    let byte_slice =
        unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, dst.len() * 4) };
    reader
        .read_exact(byte_slice)
        .map_err(|e| format!("read_f32_slice: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::generate_synthetic_model;

    fn load_test_model() -> ModelData {
        let path = generate_synthetic_model();
        let model = ModelData::load(&path).expect("failed to load synthetic model");
        let _ = std::fs::remove_file(&path);
        model
    }

    #[test]
    fn test_get_missing_tensor() {
        let model = load_test_model();
        let result = model.get("nonexistent_tensor_name");
        assert!(
            result.is_err(),
            "get() should return Err for missing tensor"
        );
        let err_msg = result.expect_err("expected error");
        assert!(
            err_msg.contains("Missing tensor"),
            "error message should mention 'Missing tensor', got: {err_msg}"
        );
    }

    #[test]
    fn test_try_get_missing_returns_none() {
        let model = load_test_model();
        assert!(
            model.try_get("nonexistent").is_none(),
            "try_get() should return None for missing tensor"
        );
    }

    #[test]
    fn test_get_quantized_missing_returns_none() {
        let model = load_test_model();
        assert!(
            model.get_quantized("nonexistent").is_none(),
            "get_quantized() should return None for missing tensor"
        );
    }

    #[test]
    fn test_load_truncated_file() {
        let path = generate_synthetic_model();
        let full_data = std::fs::read(&path).expect("read full model");
        let _ = std::fs::remove_file(&path);

        let truncated_path =
            std::env::temp_dir().join(format!("oxiwhisper_truncated_{}.bin", std::process::id()));
        std::fs::write(&truncated_path, &full_data[..10]).expect("write truncated file");

        let result = ModelData::load(&truncated_path);
        let _ = std::fs::remove_file(&truncated_path);
        assert!(
            result.is_err(),
            "loading a truncated file should return an error"
        );
    }

    #[test]
    fn test_load_wrong_magic() {
        let wrong_magic_path =
            std::env::temp_dir().join(format!("oxiwhisper_wrong_magic_{}.bin", std::process::id()));
        // Write 4 bytes with wrong magic, followed by enough data to avoid short-read before magic check
        let mut data = vec![0u8; 128];
        // Wrong magic: 0xDEADBEEF instead of 0x67676D6C
        data[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
        std::fs::write(&wrong_magic_path, &data).expect("write wrong magic file");

        let result = ModelData::load(&wrong_magic_path);
        let _ = std::fs::remove_file(&wrong_magic_path);
        assert!(result.is_err(), "wrong magic should return an error");
        let err_msg = match result {
            Err(e) => e,
            Ok(_) => panic!("expected error for wrong magic"),
        };
        assert!(
            err_msg.contains("Invalid magic"),
            "error should mention invalid magic, got: {err_msg}"
        );
    }

    #[test]
    fn test_model_hparams_consistency() {
        let model = load_test_model();
        let hp = &model.hparams;
        assert!(hp.n_vocab > 0, "n_vocab should be positive");
        assert!(hp.n_mels > 0, "n_mels should be positive");
        assert!(hp.n_audio_ctx > 0, "n_audio_ctx should be positive");
        assert!(hp.n_audio_state > 0, "n_audio_state should be positive");
        assert!(hp.n_audio_head > 0, "n_audio_head should be positive");
        assert!(hp.n_audio_layer > 0, "n_audio_layer should be positive");
        assert!(hp.n_text_ctx > 0, "n_text_ctx should be positive");
        assert!(hp.n_text_state > 0, "n_text_state should be positive");
        assert!(hp.n_text_head > 0, "n_text_head should be positive");
        assert!(hp.n_text_layer > 0, "n_text_layer should be positive");
        // Verify audio_state is divisible by audio_head
        assert_eq!(
            hp.n_audio_state % hp.n_audio_head,
            0,
            "n_audio_state should be divisible by n_audio_head"
        );
        assert_eq!(
            hp.n_text_state % hp.n_text_head,
            0,
            "n_text_state should be divisible by n_text_head"
        );
    }
}
