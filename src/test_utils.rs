//! Synthetic model generators for integration tests.
//!
//! Produces minimal valid GGML and GGUF Whisper model binaries with deterministic
//! weights. Models will not produce meaningful transcriptions but will exercise
//! the full load path without crashing.

use std::path::PathBuf;

// ── SyntheticSpec ────────────────────────────────────────────────────────────

/// Hyperparameters for a synthetic test model.
#[derive(Debug, Clone)]
pub struct SyntheticSpec {
    /// Vocabulary size used when writing the model header.
    pub n_vocab: usize,
    /// Audio encoder context length.
    pub n_audio_ctx: usize,
    /// Audio encoder hidden state dimension.
    pub n_audio_state: usize,
    /// Number of attention heads in the audio encoder.
    pub n_audio_head: usize,
    /// Number of transformer layers in the audio encoder.
    pub n_audio_layer: usize,
    /// Text decoder context length.
    pub n_text_ctx: usize,
    /// Text decoder hidden state dimension.
    pub n_text_state: usize,
    /// Number of attention heads in the text decoder.
    pub n_text_head: usize,
    /// Number of transformer layers in the text decoder.
    pub n_text_layer: usize,
    /// Number of mel filterbank channels.
    pub n_mels: usize,
    /// GGML weight type flag (0 = f32).
    pub ftype: i32,
}

impl Default for SyntheticSpec {
    fn default() -> Self {
        Self {
            n_vocab: 51865,
            n_audio_ctx: 1500,
            n_audio_state: 384,
            n_audio_head: 6,
            n_audio_layer: 4,
            n_text_ctx: 448,
            n_text_state: 384,
            n_text_head: 6,
            n_text_layer: 4,
            n_mels: 80,
            ftype: 1, // f16 weights
        }
    }
}

impl SyntheticSpec {
    /// Inner dimension of feed-forward layers (4× model dim).
    pub fn n_ff(&self) -> usize {
        self.n_audio_state * 4
    }

    /// Number of FFT bins used by the mel filter bank.
    pub fn n_fft_bins() -> usize {
        201 // WHISPER_N_FFT / 2 + 1
    }
}

// ── Tensor descriptors ───────────────────────────────────────────────────────

/// Describes a single tensor in the synthetic model.
///
/// The `is_f16` flag mirrors the dtype choice made by the real GGML whisper
/// converter: projection weights and embedding tables are stored as F16;
/// biases, layer-norm weights, and positional embeddings are stored as F32.
struct TensorDesc {
    name: String,
    shape: Vec<usize>,
    /// `true` → write as F16 (dtype 1); `false` → write as F32 (dtype 0).
    is_f16: bool,
}

impl TensorDesc {
    fn n_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Byte size of the on-disk representation (F16 = 2 bytes, F32 = 4 bytes).
    fn byte_size(&self) -> usize {
        let bytes_per_elem = if self.is_f16 { 2 } else { 4 };
        self.n_elements() * bytes_per_elem
    }
}

fn f16_desc(name: impl Into<String>, shape: Vec<usize>) -> TensorDesc {
    TensorDesc {
        name: name.into(),
        shape,
        is_f16: true,
    }
}

fn f32_desc(name: impl Into<String>, shape: Vec<usize>) -> TensorDesc {
    TensorDesc {
        name: name.into(),
        shape,
        is_f16: false,
    }
}

/// Enumerate all tensor descriptors for the given spec.
///
/// The dtype (`is_f16`) matches what the reference whisper.cpp GGML converter
/// produces: projection / embedding weights → F16, everything else → F32.
fn collect_tensor_descriptors(spec: &SyntheticSpec) -> Vec<TensorDesc> {
    let sa = spec.n_audio_state;
    let st = spec.n_text_state;
    let ff = spec.n_ff();
    let n_mels = spec.n_mels;

    let mut descs: Vec<TensorDesc> = vec![
        f16_desc("encoder.conv1.weight", vec![3, n_mels, sa]),
        f32_desc("encoder.conv1.bias", vec![sa]),
        f16_desc("encoder.conv2.weight", vec![3, sa, sa]),
        f32_desc("encoder.conv2.bias", vec![sa]),
        // positional embeddings are F32 in reference converter
        f32_desc("encoder.positional_embedding", vec![spec.n_audio_ctx, sa]),
    ];

    for i in 0..spec.n_audio_layer {
        let p = format!("encoder.blocks.{i}");
        descs.push(f32_desc(format!("{p}.attn_ln.weight"), vec![sa]));
        descs.push(f32_desc(format!("{p}.attn_ln.bias"), vec![sa]));
        descs.push(f16_desc(format!("{p}.attn.query.weight"), vec![sa, sa]));
        descs.push(f32_desc(format!("{p}.attn.query.bias"), vec![sa]));
        descs.push(f16_desc(format!("{p}.attn.key.weight"), vec![sa, sa]));
        descs.push(f16_desc(format!("{p}.attn.value.weight"), vec![sa, sa]));
        descs.push(f32_desc(format!("{p}.attn.value.bias"), vec![sa]));
        descs.push(f16_desc(format!("{p}.attn.out.weight"), vec![sa, sa]));
        descs.push(f32_desc(format!("{p}.attn.out.bias"), vec![sa]));
        descs.push(f32_desc(format!("{p}.mlp_ln.weight"), vec![sa]));
        descs.push(f32_desc(format!("{p}.mlp_ln.bias"), vec![sa]));
        descs.push(f16_desc(format!("{p}.mlp.0.weight"), vec![sa, ff]));
        descs.push(f32_desc(format!("{p}.mlp.0.bias"), vec![ff]));
        descs.push(f16_desc(format!("{p}.mlp.2.weight"), vec![ff, sa]));
        descs.push(f32_desc(format!("{p}.mlp.2.bias"), vec![sa]));
    }

    descs.push(f32_desc("encoder.ln_post.weight", vec![sa]));
    descs.push(f32_desc("encoder.ln_post.bias", vec![sa]));

    // Decoder
    descs.push(f16_desc(
        "decoder.token_embedding.weight",
        vec![st, spec.n_vocab],
    ));
    // positional embedding is F32
    descs.push(f32_desc(
        "decoder.positional_embedding",
        vec![st, spec.n_text_ctx],
    ));

    for i in 0..spec.n_text_layer {
        let p = format!("decoder.blocks.{i}");
        descs.push(f32_desc(format!("{p}.attn_ln.weight"), vec![st]));
        descs.push(f32_desc(format!("{p}.attn_ln.bias"), vec![st]));
        descs.push(f16_desc(format!("{p}.attn.query.weight"), vec![st, st]));
        descs.push(f32_desc(format!("{p}.attn.query.bias"), vec![st]));
        descs.push(f16_desc(format!("{p}.attn.key.weight"), vec![st, st]));
        descs.push(f16_desc(format!("{p}.attn.value.weight"), vec![st, st]));
        descs.push(f32_desc(format!("{p}.attn.value.bias"), vec![st]));
        descs.push(f16_desc(format!("{p}.attn.out.weight"), vec![st, st]));
        descs.push(f32_desc(format!("{p}.attn.out.bias"), vec![st]));
        descs.push(f32_desc(format!("{p}.cross_attn_ln.weight"), vec![st]));
        descs.push(f32_desc(format!("{p}.cross_attn_ln.bias"), vec![st]));
        descs.push(f16_desc(
            format!("{p}.cross_attn.query.weight"),
            vec![st, st],
        ));
        descs.push(f32_desc(format!("{p}.cross_attn.query.bias"), vec![st]));
        descs.push(f16_desc(format!("{p}.cross_attn.key.weight"), vec![st, st]));
        descs.push(f16_desc(
            format!("{p}.cross_attn.value.weight"),
            vec![st, st],
        ));
        descs.push(f32_desc(format!("{p}.cross_attn.value.bias"), vec![st]));
        descs.push(f16_desc(format!("{p}.cross_attn.out.weight"), vec![st, st]));
        descs.push(f32_desc(format!("{p}.cross_attn.out.bias"), vec![st]));
        descs.push(f32_desc(format!("{p}.mlp_ln.weight"), vec![st]));
        descs.push(f32_desc(format!("{p}.mlp_ln.bias"), vec![st]));
        descs.push(f16_desc(format!("{p}.mlp.0.weight"), vec![st, ff]));
        descs.push(f32_desc(format!("{p}.mlp.0.bias"), vec![ff]));
        descs.push(f16_desc(format!("{p}.mlp.2.weight"), vec![ff, st]));
        descs.push(f32_desc(format!("{p}.mlp.2.bias"), vec![st]));
    }

    descs.push(f32_desc("decoder.ln.weight", vec![st]));
    descs.push(f32_desc("decoder.ln.bias", vec![st]));

    descs
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Generate a minimal valid GGML Whisper model file with deterministic weights.
///
/// Returns the path to the temporary file. The caller is responsible for
/// deleting it when the test completes.
pub fn generate_synthetic_model() -> PathBuf {
    write_to_temp_file(
        "oxiwhisper_test_model",
        generate_synthetic_ggml(&SyntheticSpec::default()),
    )
}

/// Generate a valid GGML binary for the given spec.
pub fn generate_synthetic_ggml(spec: &SyntheticSpec) -> Vec<u8> {
    let mut buf = Vec::with_capacity(1 << 20);
    write_ggml_header(&mut buf, spec);
    write_ggml_mel_filters(&mut buf, spec);
    write_ggml_vocab(&mut buf, spec);

    // Write all tensors using the shared descriptor list.
    for desc in collect_tensor_descriptors(spec) {
        if desc.is_f16 {
            write_ggml_tensor_f16(&mut buf, &desc.name, &desc.shape);
        } else {
            write_ggml_tensor_f32(&mut buf, &desc.name, &desc.shape);
        }
    }

    buf
}

/// Generate a valid GGUF v3 binary for the given spec.
///
/// Weight tensors are written as F16 to match the GGML generator; biases and
/// layernorm weights are written as F32.  Both loaders produce identical f32
/// values after conversion, enabling byte-exact equivalence tests.
pub fn generate_synthetic_gguf(spec: &SyntheticSpec) -> Vec<u8> {
    let mut buf = Vec::with_capacity(1 << 20);

    let tensor_descs = collect_tensor_descriptors(spec);

    // ── GGUF magic + version ──────────────────────────────────────────────────
    buf.extend_from_slice(b"GGUF");
    write_u32_le(&mut buf, 3); // version

    // ── tensor_count + metadata_kv_count ─────────────────────────────────────
    write_u64_le(&mut buf, tensor_descs.len() as u64);
    let kv_entries = build_kv_entries(spec);
    write_u64_le(&mut buf, kv_entries.len() as u64);

    // ── KV metadata ───────────────────────────────────────────────────────────
    for (key, value_bytes) in &kv_entries {
        write_gguf_string(&mut buf, key);
        buf.extend_from_slice(value_bytes);
    }

    // ── Tensor infos ──────────────────────────────────────────────────────────
    // Compute per-tensor byte offsets (aligned to 32 bytes).
    let alignment: u64 = 32;
    let mut current_offset: u64 = 0;
    let mut tensor_offsets: Vec<u64> = Vec::with_capacity(tensor_descs.len());
    for desc in &tensor_descs {
        tensor_offsets.push(current_offset);
        let n_bytes = desc.byte_size() as u64;
        current_offset = align_up(current_offset + n_bytes, alignment);
    }

    for (desc, &offset) in tensor_descs.iter().zip(tensor_offsets.iter()) {
        write_gguf_string(&mut buf, &desc.name);
        write_u32_le(&mut buf, desc.shape.len() as u32); // n_dims
        for &d in &desc.shape {
            write_u64_le(&mut buf, d as u64);
        }
        // GgmlType: F32 = 0, F16 = 1
        let ggml_type: u32 = if desc.is_f16 { 1 } else { 0 };
        write_u32_le(&mut buf, ggml_type);
        write_u64_le(&mut buf, offset);
    }

    // ── Align to data section ─────────────────────────────────────────────────
    let header_end = buf.len() as u64;
    let aligned_start = align_up(header_end, alignment);
    let pad_bytes = (aligned_start - header_end) as usize;
    buf.extend(std::iter::repeat_n(0u8, pad_bytes));

    // ── Tensor data ───────────────────────────────────────────────────────────
    let data_section_start = buf.len() as u64;
    for (idx, desc) in tensor_descs.iter().enumerate() {
        let n_elements = desc.n_elements();
        let name_hash = simple_hash(&desc.name);

        if desc.is_f16 {
            for i in 0..n_elements {
                let v = deterministic_value(name_hash, i);
                let h = half::f16::from_f32(v);
                buf.extend_from_slice(&h.to_le_bytes());
            }
        } else {
            for i in 0..n_elements {
                let v = deterministic_value(name_hash, i);
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }

        // Pad to the start of the next tensor's aligned offset.
        // The absolute position of the next tensor's data is:
        //   data_section_start + tensor_offsets[idx + 1]  (or end-of-last if final tensor)
        let current_abs = buf.len() as u64;
        let next_tensor_abs = if idx + 1 < tensor_offsets.len() {
            data_section_start + tensor_offsets[idx + 1]
        } else {
            align_up(current_abs, alignment)
        };
        let pad = (next_tensor_abs.saturating_sub(current_abs)) as usize;
        buf.extend(std::iter::repeat_n(0u8, pad));
    }

    buf
}

// ── File helpers ─────────────────────────────────────────────────────────────

/// Write `bytes` to a unique temp file and return its path.
fn write_to_temp_file(prefix: &str, bytes: Vec<u8>) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "{}_{pid}_{id}.bin",
        prefix,
        pid = std::process::id(),
    ));
    std::fs::write(&path, &bytes).expect("write temp model file");
    path
}

/// Write a GGML model to a temp file and return the path.
pub fn write_ggml_to_file(spec: &SyntheticSpec) -> PathBuf {
    write_to_temp_file("oxiwhisper_test_ggml", generate_synthetic_ggml(spec))
}

/// Write a GGUF model to a temp file and return the path.
pub fn write_gguf_to_file(spec: &SyntheticSpec) -> PathBuf {
    write_to_temp_file("oxiwhisper_test_gguf", generate_synthetic_gguf(spec))
}

// ── GGML generation internals ────────────────────────────────────────────────

fn write_ggml_header(buf: &mut Vec<u8>, spec: &SyntheticSpec) {
    const GGML_MAGIC: u32 = 0x67676D6C;
    write_u32_le(buf, GGML_MAGIC);
    write_i32_le(buf, spec.n_vocab as i32);
    write_i32_le(buf, spec.n_audio_ctx as i32);
    write_i32_le(buf, spec.n_audio_state as i32);
    write_i32_le(buf, spec.n_audio_head as i32);
    write_i32_le(buf, spec.n_audio_layer as i32);
    write_i32_le(buf, spec.n_text_ctx as i32);
    write_i32_le(buf, spec.n_text_state as i32);
    write_i32_le(buf, spec.n_text_head as i32);
    write_i32_le(buf, spec.n_text_layer as i32);
    write_i32_le(buf, spec.n_mels as i32);
    write_i32_le(buf, spec.ftype);
}

fn write_ggml_mel_filters(buf: &mut Vec<u8>, spec: &SyntheticSpec) {
    let n_fft_bins = SyntheticSpec::n_fft_bins();
    write_i32_le(buf, spec.n_mels as i32);
    write_i32_le(buf, n_fft_bins as i32);

    let total = spec.n_mels * n_fft_bins;
    for i in 0..total {
        let mel_idx = i / n_fft_bins;
        let bin_idx = i % n_fft_bins;
        let center = (mel_idx as f32 + 0.5) * n_fft_bins as f32 / spec.n_mels as f32;
        let dist = (bin_idx as f32 - center).abs();
        let width = n_fft_bins as f32 / spec.n_mels as f32;
        let val = if dist < width {
            (1.0 - dist / width) * 0.01
        } else {
            0.0
        };
        buf.extend_from_slice(&val.to_le_bytes());
    }
}

fn write_ggml_vocab(buf: &mut Vec<u8>, spec: &SyntheticSpec) {
    write_i32_le(buf, spec.n_vocab as i32);
    for i in 0..spec.n_vocab {
        let token = format!("<|{i}|>");
        let bytes = token.as_bytes();
        write_i32_le(buf, bytes.len() as i32);
        buf.extend_from_slice(bytes);
    }
}

fn write_ggml_tensor_f32(buf: &mut Vec<u8>, name: &str, shape: &[usize]) {
    write_ggml_tensor_header(buf, name, shape, 0);
    let n_elements: usize = shape.iter().product();
    let name_hash = simple_hash(name);
    for i in 0..n_elements {
        let v = deterministic_value(name_hash, i);
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

fn write_ggml_tensor_f16(buf: &mut Vec<u8>, name: &str, shape: &[usize]) {
    write_ggml_tensor_header(buf, name, shape, 1);
    let n_elements: usize = shape.iter().product();
    let name_hash = simple_hash(name);
    for i in 0..n_elements {
        let v = deterministic_value(name_hash, i);
        let h = half::f16::from_f32(v);
        buf.extend_from_slice(&h.to_le_bytes());
    }
}

fn write_ggml_tensor_header(buf: &mut Vec<u8>, name: &str, shape: &[usize], dtype: i32) {
    write_i32_le(buf, shape.len() as i32);
    write_i32_le(buf, name.len() as i32);
    write_i32_le(buf, dtype);
    for &d in shape {
        write_i32_le(buf, d as i32);
    }
    buf.extend_from_slice(name.as_bytes());
}

// ── GGUF generation internals ────────────────────────────────────────────────

/// Build KV metadata entries as `(key, raw_value_bytes)` pairs.
///
/// The raw bytes include the value_type u32 followed by the value payload,
/// matching the GGUF binary format.
fn build_kv_entries(spec: &SyntheticSpec) -> Vec<(String, Vec<u8>)> {
    let mut entries: Vec<(String, Vec<u8>)> = Vec::new();

    // Helper closures
    let kv_u32 = |key: &str, val: u32| -> (String, Vec<u8>) {
        let mut v = Vec::with_capacity(8);
        v.extend_from_slice(&4u32.to_le_bytes()); // GgufValueType::U32 = 4
        v.extend_from_slice(&val.to_le_bytes());
        (key.to_string(), v)
    };

    let kv_str = |key: &str, val: &str| -> (String, Vec<u8>) {
        let bytes = val.as_bytes();
        let mut v = Vec::with_capacity(8 + 8 + bytes.len());
        v.extend_from_slice(&8u32.to_le_bytes()); // GgufValueType::String = 8
        v.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        v.extend_from_slice(bytes);
        (key.to_string(), v)
    };

    // General metadata
    entries.push(kv_str("general.architecture", "whisper"));
    entries.push(kv_str("general.name", "oxiwhisper-synthetic"));

    // Whisper hyperparameters (using GGUF spec key names)
    entries.push(kv_u32("whisper.vocab_size", spec.n_vocab as u32));
    entries.push(kv_u32(
        "whisper.encoder.context_length",
        spec.n_audio_ctx as u32,
    ));
    entries.push(kv_u32(
        "whisper.encoder.embedding_length",
        spec.n_audio_state as u32,
    ));
    entries.push(kv_u32(
        "whisper.encoder.attention.head_count",
        spec.n_audio_head as u32,
    ));
    entries.push(kv_u32(
        "whisper.encoder.block_count",
        spec.n_audio_layer as u32,
    ));
    entries.push(kv_u32("whisper.encoder.mels_count", spec.n_mels as u32));
    entries.push(kv_u32(
        "whisper.decoder.context_length",
        spec.n_text_ctx as u32,
    ));
    entries.push(kv_u32(
        "whisper.decoder.embedding_length",
        spec.n_text_state as u32,
    ));
    entries.push(kv_u32(
        "whisper.decoder.attention.head_count",
        spec.n_text_head as u32,
    ));
    entries.push(kv_u32(
        "whisper.decoder.block_count",
        spec.n_text_layer as u32,
    ));

    // Tokenizer tokens array
    {
        let mut v = Vec::new();
        v.extend_from_slice(&9u32.to_le_bytes()); // GgufValueType::Array = 9
        // array: elem_type = String (8), count = n_vocab
        v.extend_from_slice(&8u32.to_le_bytes()); // elem type = String
        v.extend_from_slice(&(spec.n_vocab as u64).to_le_bytes());
        for i in 0..spec.n_vocab {
            let token = format!("<|{i}|>");
            let bytes = token.as_bytes();
            v.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            v.extend_from_slice(bytes);
        }
        entries.push(("tokenizer.ggml.tokens".to_string(), v));
    }

    entries
}

/// Write a GGUF string: u64 length + UTF-8 bytes.
fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(bytes);
}

fn align_up(v: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return v;
    }
    let r = v % alignment;
    if r == 0 { v } else { v + (alignment - r) }
}

// ── Low-level helpers ─────────────────────────────────────────────────────────

fn write_i32_le(buf: &mut Vec<u8>, v: i32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_u32_le(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_u64_le(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// Produce a deterministic value in `[-0.01, 0.01]` from a name hash and index.
fn deterministic_value(name_hash: u64, index: usize) -> f32 {
    let mixed = name_hash
        .wrapping_mul(2654435761)
        .wrapping_add(index as u64);
    let frac = ((mixed & 0xFFFF) as f32) / 65535.0;
    (frac - 0.5) * 0.02
}

/// Simple FNV-1a string hash.
fn simple_hash(s: &str) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}
