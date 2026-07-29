use crate::tensor::Tensor;
use half::f16;
use std::collections::HashMap;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

/// Whisper model hyperparameters (from GGML header)
#[derive(Debug, Clone)]
pub struct Hparams {
    /// Vocabulary size (number of tokens).
    pub n_vocab: usize,
    /// Audio encoder context length (maximum encoder sequence length in frames).
    pub n_audio_ctx: usize,
    /// Hidden state size of the audio encoder.
    pub n_audio_state: usize,
    /// Number of attention heads in the audio encoder.
    pub n_audio_head: usize,
    /// Number of transformer layers in the audio encoder.
    pub n_audio_layer: usize,
    /// Text decoder context length (maximum decoder sequence length in tokens).
    pub n_text_ctx: usize,
    /// Hidden state size of the text decoder.
    pub n_text_state: usize,
    /// Number of attention heads in the text decoder.
    pub n_text_head: usize,
    /// Number of transformer layers in the text decoder.
    pub n_text_layer: usize,
    /// Number of mel filterbank channels.
    pub n_mels: usize,
    /// GGML weight type flag (0 = f32, 1 = f16, 2+ = quantized).
    pub ftype: i32,
}

/// Token vocabulary entry.
///
/// The token payload is stored as **raw bytes**, exactly as the model file
/// spells it. Whisper uses GPT-2 byte-level BPE, so a single multi-byte
/// character is frequently split across several tokens: the kanji `渋`
/// (`U+6E0B`, UTF-8 `E6 B8 8B`) is tokens `162`, `116`, `233` in the standard
/// multilingual vocabulary, and none of those three is valid UTF-8 on its own.
///
/// Converting each entry to a `String` at load time therefore destroys the
/// vocabulary (1476 of 50257 entries in `ggml-tiny.bin` become `U+FFFD`).
/// Decoding must concatenate **bytes** and run the UTF-8 conversion exactly
/// once over the joined buffer — see [`crate::tokenizer::decode`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VocabEntry {
    /// Raw token bytes as stored in the model file.
    pub bytes: Vec<u8>,
}

impl VocabEntry {
    /// Build an entry from raw model-file bytes.
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    /// Build an entry from text that is already known to be valid UTF-8
    /// (special tokens, ONNX/GGUF vocab strings, tests).
    pub fn from_text(text: impl AsRef<str>) -> Self {
        Self {
            bytes: text.as_ref().as_bytes().to_vec(),
        }
    }

    /// Raw bytes of this token.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Lossy UTF-8 view of this **single** entry.
    ///
    /// Only meaningful for entries that are complete characters (ASCII, special
    /// tokens, whole CJK tokens). Never use this to build a transcript — join
    /// the bytes first, then decode once.
    pub fn text(&self) -> std::borrow::Cow<'_, str> {
        String::from_utf8_lossy(&self.bytes)
    }

    /// `true` when this entry is a Whisper special token, i.e. it starts with
    /// the `<|` marker (`<|startoftranscript|>`, `<|en|>`, …).
    pub fn is_special(&self) -> bool {
        self.bytes.starts_with(b"<|")
    }

    /// `true` when the entry carries no bytes at all.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

/// All loaded model data
#[derive(Debug)]
pub struct ModelData {
    /// Whisper model hyperparameters (dimensions, layer counts, etc.).
    pub hparams: Hparams,
    /// Mel filterbank coefficients `[n_mels, n_fft/2+1]`, row-major.
    pub mel_filters: Vec<f32>,
    /// Token vocabulary; index is the token ID.
    pub vocab: Vec<VocabEntry>,
    /// Unquantized f32 tensors indexed by name.
    pub tensors: HashMap<String, Tensor>,
    /// Quantized tensors kept in their original GGML block format
    /// (Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0).
    /// Large 2D weight matrices are stored here instead of being dequantized to f32.
    pub quantized_tensors: HashMap<String, crate::quantize::QuantizedTensor>,
}

impl ModelData {
    /// Load a whisper model file (GGML or GGUF format).
    pub fn load(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("Cannot open model: {e}"))?;
        let mut reader = BufReader::new(file);
        Self::load_from_reader(&mut reader)
    }

    /// Load a whisper model by memory-mapping the file.
    ///
    /// Equivalent to [`Self::load`] but the file is mapped into the process
    /// address space during parsing rather than read into a userspace buffer.
    /// On large models with cold OS page cache this can reduce peak RSS and
    /// exploit kernel read-ahead. Tensor data is still copied into owned
    /// buffers — the mmap is dropped after parsing.
    ///
    /// # Safety considerations
    /// If another process truncates or replaces the file while it is mapped,
    /// the process may receive SIGBUS. The file must not be modified during
    /// the call.
    pub fn load_mmap(path: &Path) -> Result<Self, String> {
        use std::io::Cursor;
        let file = std::fs::File::open(path).map_err(|e| format!("Cannot open model: {e}"))?;
        // SAFETY: We read the file sequentially and do not mutate it.
        // If the file is externally truncated, SIGBUS may result — see doc comment.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("mmap failed: {e}"))?;
        #[cfg(unix)]
        let _ = mmap.advise(memmap2::Advice::Sequential);
        let mut cursor = Cursor::new(&mmap[..]);
        Self::load_from_reader(&mut cursor)
    }

    /// Parse a whisper model from any `Read + Seek` source.
    ///
    /// Reads the 4-byte magic, then dispatches to the appropriate format
    /// parser: GGML (`0x67676D6C`) or GGUF (`GGUF` / `[0x47,0x47,0x55,0x46]`).
    fn load_from_reader<R: Read + Seek>(reader: &mut R) -> Result<Self, String> {
        let mut magic_bytes = [0u8; 4];
        reader
            .read_exact(&mut magic_bytes)
            .map_err(|e| format!("read magic: {e}"))?;

        match magic_bytes {
            // GGML magic 0x67676D6C stored little-endian: bytes are [0x6C, 0x6D, 0x67, 0x67]
            [0x6C, 0x6D, 0x67, 0x67] => load_ggml_from_reader(reader),
            // GGUF magic: ASCII 'G','G','U','F'
            [0x47, 0x47, 0x55, 0x46] => {
                crate::gguf::parse::load_gguf_from_reader(reader).map_err(|e| format!("{e}"))
            }
            other => Err(format!("Invalid magic: unknown format {:02x?}", other)),
        }
    }

    /// Look up an f32 tensor by name, returning an error if not found.
    pub fn get(&self, name: &str) -> Result<&Tensor, String> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("Missing tensor: {name}"))
    }

    /// Look up an f32 tensor by name, returning `None` if not found.
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

/// Parse the body of a GGML-format whisper model (magic already consumed).
fn load_ggml_from_reader<R: Read>(reader: &mut R) -> Result<ModelData, String> {
    // Read hyperparameters
    let hparams = Hparams {
        n_vocab: read_i32(reader)? as usize,
        n_audio_ctx: read_i32(reader)? as usize,
        n_audio_state: read_i32(reader)? as usize,
        n_audio_head: read_i32(reader)? as usize,
        n_audio_layer: read_i32(reader)? as usize,
        n_text_ctx: read_i32(reader)? as usize,
        n_text_state: read_i32(reader)? as usize,
        n_text_head: read_i32(reader)? as usize,
        n_text_layer: read_i32(reader)? as usize,
        n_mels: read_i32(reader)? as usize,
        ftype: read_i32(reader)?,
    };

    #[cfg(feature = "timing")]
    eprintln!("Model hparams: {:?}", hparams);

    // Read mel filters
    let n_mel_filters = read_i32(reader)? as usize;
    let n_mel_len = read_i32(reader)? as usize;
    let mel_total = n_mel_filters * n_mel_len;
    let mut mel_filters = vec![0.0f32; mel_total];
    read_f32_slice(reader, &mut mel_filters)?;

    #[cfg(feature = "timing")]
    eprintln!(
        "Mel filters: {} x {} = {} values",
        n_mel_filters, n_mel_len, mel_total
    );

    // Read vocabulary
    let n_vocab = read_i32(reader)? as usize;
    let mut vocab = Vec::with_capacity(n_vocab);
    for _ in 0..n_vocab {
        let len = read_i32(reader)? as usize;
        let mut buf = vec![0u8; len];
        reader
            .read_exact(&mut buf)
            .map_err(|e| format!("read vocab: {e}"))?;
        // Store the bytes verbatim. GPT-2 byte-level BPE splits multi-byte
        // characters across tokens, so a per-entry UTF-8 conversion here would
        // irrecoverably replace those fragments with U+FFFD.
        vocab.push(VocabEntry::from_bytes(buf));
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

    while let Some(v) = try_read_i32(reader) {
        let n_dims = v as usize;

        let name_len = read_i32(reader)? as usize;
        let dtype = read_i32(reader)?;

        let mut dims = vec![0i32; n_dims];
        for d in dims.iter_mut() {
            *d = read_i32(reader)?;
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

        // Read data based on dtype.
        //
        // The legacy GGML whisper container stores the *same* `ggml_type`
        // discriminants as GGUF:
        //   0 = F32, 1 = F16, 2 = Q4_0, 3 = Q4_1, 6 = Q5_0, 7 = Q5_1, 8 = Q8_0
        // (4/5 are the removed Q4_2/Q4_3, 9 = Q8_1, 10+ = K-quants).
        match dtype {
            0 => {
                // f32
                let mut f32_data = vec![0.0f32; n_elements];
                read_f32_slice(reader, &mut f32_data)?;
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
            other => {
                let qtype =
                    crate::quantize::QuantType::from_ggml_type(other as u32).ok_or_else(|| {
                        format!(
                            "Unsupported tensor dtype: {dtype} for {name}; \
                             supported GGML types are F32(0), F16(1), Q4_0(2), Q4_1(3), \
                             Q5_0(6), Q5_1(7) and Q8_0(8)"
                        )
                    })?;
                let loaded = read_ggml_quantized(reader, qtype, &name, shape, is_large_2d_weight)?;
                match loaded {
                    LoadedTensor::Float(t) => {
                        tensors.insert(name.clone(), t);
                    }
                    LoadedTensor::Quantized(q) => {
                        quantized_tensors.insert(name.clone(), q);
                        #[cfg(feature = "timing")]
                        {
                            quant_count += 1;
                        }
                    }
                }
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

/// Outcome of reading one block-quantized GGML tensor body.
enum LoadedTensor {
    /// Dequantized to f32 (small tensors and non-2-D tensors).
    Float(Tensor),
    /// Kept in its native quantized layout (large 2-D weight matrices).
    Quantized(crate::quantize::QuantizedTensor),
}

/// Read the body of a block-quantized GGML tensor.
///
/// Large 2-D weight matrices (`keep_quantized`) are retained in their native
/// block layout so the GEMV kernels can consume them directly; everything else
/// is dequantized to f32 up front.
fn read_ggml_quantized<R: Read>(
    reader: &mut R,
    qtype: crate::quantize::QuantType,
    name: &str,
    shape: Vec<usize>,
    keep_quantized: bool,
) -> Result<LoadedTensor, String> {
    let n_elements: usize = shape.iter().product();
    let block_size = qtype.block_size();
    if block_size == 0 || !n_elements.is_multiple_of(block_size) {
        return Err(format!(
            "Tensor {name} has {n_elements} elements, not a multiple of the {} block size {block_size}",
            qtype.name()
        ));
    }
    let n_blocks = n_elements / block_size;
    let n_bytes = n_blocks
        .checked_mul(qtype.block_bytes())
        .ok_or_else(|| format!("Tensor {name} byte size overflows usize"))?;

    let mut raw = vec![0u8; n_bytes];
    reader
        .read_exact(&mut raw)
        .map_err(|e| format!("read {} data for {name}: {e}", qtype.name()))?;

    if keep_quantized {
        Ok(LoadedTensor::Quantized(crate::quantize::QuantizedTensor {
            raw,
            shape,
            qtype,
        }))
    } else {
        let f32_data = crate::quantize::dequantize(&raw, n_elements, qtype);
        Ok(LoadedTensor::Float(Tensor::from_vec(f32_data, &shape)))
    }
}

fn read_i32<R: Read>(reader: &mut R) -> Result<i32, String> {
    let mut buf = [0u8; 4];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("read_i32: {e}"))?;
    Ok(i32::from_le_bytes(buf))
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

    #[test]
    fn test_load_mmap_smoke() {
        let path = generate_synthetic_model();
        let result = ModelData::load_mmap(&path);
        let _ = std::fs::remove_file(&path);
        assert!(
            result.is_ok(),
            "load_mmap should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_load_mmap_equivalence() {
        let path_read = generate_synthetic_model();
        // Write a second identical model file so both loads see the same content
        let path_mmap = generate_synthetic_model();

        let via_read = ModelData::load(&path_read).expect("load via read");
        let via_mmap = ModelData::load_mmap(&path_mmap).expect("load via mmap");

        let _ = std::fs::remove_file(&path_read);
        let _ = std::fs::remove_file(&path_mmap);

        assert_eq!(
            via_read.hparams.n_vocab, via_mmap.hparams.n_vocab,
            "n_vocab mismatch"
        );
        assert_eq!(
            via_read.hparams.n_mels, via_mmap.hparams.n_mels,
            "n_mels mismatch"
        );
        assert_eq!(
            via_read.vocab.len(),
            via_mmap.vocab.len(),
            "vocab len mismatch"
        );
        assert_eq!(
            via_read.tensors.len(),
            via_mmap.tensors.len(),
            "tensor count mismatch"
        );

        // Compare tensor data by shape and element count (values are deterministic per name)
        for (name, t_read) in &via_read.tensors {
            let t_mmap = via_mmap
                .tensors
                .get(name)
                .unwrap_or_else(|| panic!("tensor {name} missing in mmap load"));
            assert_eq!(
                t_read.data.len(),
                t_mmap.data.len(),
                "tensor {name} length mismatch"
            );
            for (i, (&r, &m)) in t_read.data.iter().zip(t_mmap.data.iter()).enumerate() {
                assert_eq!(r.to_bits(), m.to_bits(), "tensor {name} data[{i}] mismatch");
            }
        }
    }

    #[test]
    fn test_load_mmap_truncated_file() {
        let path = std::env::temp_dir().join(format!(
            "oxiwhisper_test_mmap_trunc_{}.bin",
            std::process::id()
        ));
        std::fs::write(&path, b"tinydata").expect("write truncated file");
        let result = ModelData::load_mmap(&path);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err(), "truncated file must fail");
    }

    #[test]
    fn test_load_mmap_wrong_magic() {
        let path = std::env::temp_dir().join(format!(
            "oxiwhisper_test_mmap_magic_{}.bin",
            std::process::id()
        ));
        // Write 64 bytes with wrong magic bytes
        std::fs::write(&path, [0xDEu8; 64]).expect("write bad magic file");
        let result = ModelData::load_mmap(&path);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err(), "wrong magic must fail");
    }

    // ── GGUF tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_load_synthetic_gguf() {
        use crate::test_utils::{SyntheticSpec, generate_synthetic_gguf};
        use std::io::Cursor;

        let bytes = generate_synthetic_gguf(&SyntheticSpec::default());
        let mut cursor = Cursor::new(bytes);
        let model = ModelData::load_from_reader(&mut cursor).expect("load synthetic GGUF");
        assert!(model.hparams.n_vocab > 0, "n_vocab should be positive");
        assert!(
            model.hparams.n_audio_layer > 0,
            "n_audio_layer should be positive"
        );
        assert!(model.hparams.n_mels > 0, "n_mels should be positive");
    }

    #[test]
    fn test_ggml_gguf_equivalence() {
        use crate::test_utils::{SyntheticSpec, generate_synthetic_ggml, generate_synthetic_gguf};
        use std::io::Cursor;

        let spec = SyntheticSpec::default();
        let ggml_bytes = generate_synthetic_ggml(&spec);
        let gguf_bytes = generate_synthetic_gguf(&spec);

        let mut c_ggml = Cursor::new(ggml_bytes);
        let mut c_gguf = Cursor::new(gguf_bytes);

        let ggml_model = ModelData::load_from_reader(&mut c_ggml).expect("load GGML");
        let gguf_model = ModelData::load_from_reader(&mut c_gguf).expect("load GGUF");

        assert_eq!(
            ggml_model.hparams.n_vocab, gguf_model.hparams.n_vocab,
            "n_vocab mismatch"
        );
        assert_eq!(
            ggml_model.hparams.n_audio_layer, gguf_model.hparams.n_audio_layer,
            "n_audio_layer mismatch"
        );
        assert_eq!(
            ggml_model.hparams.n_audio_state, gguf_model.hparams.n_audio_state,
            "n_audio_state mismatch"
        );
        assert_eq!(
            ggml_model.hparams.n_mels, gguf_model.hparams.n_mels,
            "n_mels mismatch"
        );
        assert_eq!(
            ggml_model.hparams.n_text_layer, gguf_model.hparams.n_text_layer,
            "n_text_layer mismatch"
        );
        assert_eq!(
            ggml_model.vocab.len(),
            gguf_model.vocab.len(),
            "vocab length mismatch"
        );

        // Verify all tensors present in the GGML load appear in the GGUF load
        // with identical element counts AND identical f32 values.
        //
        // Both generators write F16 for weight tensors and F32 for the rest —
        // so both loaders perform the same F16→F32 conversion, producing
        // bit-exact f32 values.
        for (name, ggml_tensor) in &ggml_model.tensors {
            let gguf_tensor = gguf_model
                .tensors
                .get(name)
                .unwrap_or_else(|| panic!("tensor '{name}' missing from GGUF load"));
            assert_eq!(
                ggml_tensor.data.len(),
                gguf_tensor.data.len(),
                "tensor '{name}' element count differs between GGML and GGUF"
            );
            for (i, (&g, &u)) in ggml_tensor
                .data
                .iter()
                .zip(gguf_tensor.data.iter())
                .enumerate()
            {
                assert_eq!(
                    g.to_bits(),
                    u.to_bits(),
                    "tensor '{name}' data[{i}] differs: GGML={g} GGUF={u}"
                );
            }
        }
        // Also verify the total tensor count matches (no extra tensors in either).
        assert_eq!(
            ggml_model.tensors.len(),
            gguf_model.tensors.len(),
            "total plain tensor count differs between GGML and GGUF"
        );
    }

    #[test]
    fn test_load_mmap_gguf() {
        use crate::test_utils::{SyntheticSpec, generate_synthetic_gguf};

        let bytes = generate_synthetic_gguf(&SyntheticSpec::default());
        let path = std::env::temp_dir().join(format!(
            "oxiwhisper_test_mmap_gguf_{}.bin",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).expect("write GGUF temp file");
        let result = ModelData::load_mmap(&path);
        let _ = std::fs::remove_file(&path);
        let model = result.expect("load_mmap GGUF should succeed");
        assert!(model.hparams.n_vocab > 0, "n_vocab should be positive");
    }

    #[test]
    fn test_load_gguf_unsupported_dtype() {
        use crate::test_utils::{SyntheticSpec, generate_synthetic_gguf};
        use std::io::Cursor;

        // Generate a valid GGUF, then patch one tensor's dtype field to Q8_1
        // (`ggml_type` 9) — a real ggml type that this crate deliberately does
        // not implement, so it must be rejected with a clear message rather
        // than silently mis-decoded.
        //
        // `encoder.conv1.weight` is a 3-dim F16 tensor (GgmlType::F16 = 1).
        // Its tensor-info layout (after the u64 name-length prefix + name bytes):
        //   n_dims:    u32  (4 bytes) = 3
        //   dim[0]:    u64  (8 bytes)
        //   dim[1]:    u64  (8 bytes)
        //   dim[2]:    u64  (8 bytes)
        //   ggml_type: u32  (4 bytes) ← patch this from F16=1 to Q8_1=9
        //   offset:    u64  (8 bytes)
        //
        // We scan for the name bytes and jump `4 + 3*8` bytes past the end to reach ggml_type.
        let spec = SyntheticSpec::default();
        let mut bytes = generate_synthetic_gguf(&spec);

        let tensor_name = b"encoder.conv1.weight";
        // After name bytes: n_dims(4) + 3×dim(24) = 28 bytes before ggml_type
        let offset_from_name_end = 4 + 3 * 8usize;
        // F16 ggml_type stored as LE u32 = [0x01, 0x00, 0x00, 0x00]
        let ggml_type_f16_le = [1u8, 0, 0, 0];
        // Q8_1 ggml_type stored as LE u32 = [0x09, 0x00, 0x00, 0x00]
        let q8_1_le = [9u8, 0, 0, 0];

        let mut patched = false;
        let search_end = bytes
            .len()
            .saturating_sub(tensor_name.len() + offset_from_name_end + 4);
        for i in 0..search_end {
            if bytes[i..i + tensor_name.len()] == *tensor_name {
                let dtype_pos = i + tensor_name.len() + offset_from_name_end;
                if dtype_pos + 4 <= bytes.len()
                    && bytes[dtype_pos..dtype_pos + 4] == ggml_type_f16_le
                {
                    bytes[dtype_pos..dtype_pos + 4].copy_from_slice(&q8_1_le);
                    patched = true;
                    break;
                }
            }
        }
        assert!(
            patched,
            "failed to patch dtype in GGUF; test infrastructure issue"
        );

        let mut cursor = Cursor::new(bytes);
        let result = ModelData::load_from_reader(&mut cursor);
        assert!(result.is_err(), "Q8_1 should be rejected as unsupported");
        let err = result.expect_err("expected error");
        let err_lower = err.to_lowercase();
        assert!(
            err_lower.contains("q8_1") || err_lower.contains("unsupported"),
            "error should mention Q8_1 or unsupported, got: {err}"
        );
    }

    #[test]
    fn test_gguf_malformed_magic() {
        use std::io::Cursor;

        let bad = [0x00u8, 0x01, 0x02, 0x03];
        let mut cursor = Cursor::new(&bad[..]);
        let result = ModelData::load_from_reader(&mut cursor);
        assert!(result.is_err(), "bad magic should fail");
        let err = result.expect_err("expected error");
        assert!(
            err.to_lowercase().contains("invalid magic") || err.contains("magic"),
            "error should mention magic, got: {err}"
        );
    }

    #[test]
    fn test_gguf_alignment_math() {
        use crate::gguf::spec::align_offset;
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(31, 32), 32);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(33, 32), 64);
        assert_eq!(align_offset(0, 64), 0);
        assert_eq!(align_offset(64, 64), 64);
        assert_eq!(align_offset(65, 64), 128);
    }
}
