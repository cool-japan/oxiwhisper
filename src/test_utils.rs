//! Synthetic model generators for integration tests.
//!
//! Produces minimal valid GGML and GGUF Whisper model binaries with deterministic
//! weights that exercise the full load path without crashing.
//!
//! # Designed decoding behaviour
//!
//! Unlike a purely random synthetic model (which drives the decoder into a
//! degenerate constant-token loop and yields empty text / empty segments), the
//! weights here are *crafted* so the greedy decoder walks a fixed, deterministic
//! token chain that produces real text and properly paired timestamps.
//!
//! The construction relies on three facts about the decoder
//! (`crate::decoder::forward`):
//! 1. the residual stream at step `t` is `token_embedding[cur] + positional_embedding[t]`
//!    once the attention / MLP **output projections** are zeroed (so the blocks add
//!    nothing to the residual);
//! 2. the final layer norm is made a pure normalisation (`weight = 1`, `bias = 0`);
//! 3. logits are `hidden @ token_embedding^T` (the embedding table is tied).
//!
//! Each designed token `v` is given a **one-hot** embedding row `S · e_{dim(v)}`
//! (distinct dimension per token), so `logits[v] ≈ S · hidden[dim(v)]`. The
//! positional embedding at position `t` is set to `P · e_{dim(target)}` with
//! `P > S`, so the positional term dominates the argmax and steers step `t`
//! onto the designed `target` token — while satisfying OpenAI's
//! `apply_timestamp_rules` masking (see [`crate::decode_utils`]).
//!
//! With timestamps enabled the emitted sequence (excluding the prompt) is:
//! `<|0.00|> hello <|1.00|> <|1.00|> world <|2.00|> <|endoftext|>`, which
//! [`crate::tokenizer::parse_segments`] turns into two segments
//! `(0.00–1.00, "hello")` and `(1.00–2.00, "world")`.
//!
//! That shape is dictated by OpenAI's `ApplyTimestampRules` (see
//! `crate::decode_utils::apply_timestamp_rules`): the first sampled token must
//! be a timestamp; a timestamp whose predecessor is a timestamp *or is absent*
//! must be followed by text; and a timestamp preceded by text closes a segment,
//! so only another timestamp or `<|endoftext|>` may follow — hence the doubled
//! `<|1.00|>` **between** segments and the single `<|0.00|>` at the start.
//!
//! See `design_override` and `DESIGN_TOKENS` for the concrete weights.

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
                let v = tensor_element_value(&desc.name, name_hash, i, &desc.shape);
                let h = half::f16::from_f32(v);
                buf.extend_from_slice(&h.to_le_bytes());
            }
        } else {
            for i in 0..n_elements {
                let v = tensor_element_value(&desc.name, name_hash, i, &desc.shape);
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
        let token = designed_vocab_text(i);
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
        let v = tensor_element_value(name, name_hash, i, shape);
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

fn write_ggml_tensor_f16(buf: &mut Vec<u8>, name: &str, shape: &[usize]) {
    write_ggml_tensor_header(buf, name, shape, 1);
    let n_elements: usize = shape.iter().product();
    let name_hash = simple_hash(name);
    for i in 0..n_elements {
        let v = tensor_element_value(name, name_hash, i, shape);
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
            let token = designed_vocab_text(i);
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

// ── Designed decoding weights ────────────────────────────────────────────────

/// One-hot magnitude `S` written into a designed token's embedding row.
const DESIGN_EMB_SCALE: f32 = 1.0;

/// Positional steering magnitude `P`. Must exceed [`DESIGN_EMB_SCALE`] so the
/// positional term at step `t` beats the previous token's embedding term and
/// the argmax lands on the designed target rather than repeating the last token.
const DESIGN_POS_SCALE: f32 = 4.0;

/// Whisper `<|endoftext|>` token id (stop token).
const TOK_EOT: u32 = 50256;
/// Timestamp token for 0.00s (`TIMESTAMP_BEGIN`).
const TOK_TS0: u32 = 50364;
/// Timestamp token for 1.00s (`TIMESTAMP_BEGIN + 50`).
const TOK_TS1: u32 = 50414;
/// Timestamp token for 2.00s (`TIMESTAMP_BEGIN + 100`).
const TOK_TS2: u32 = 50464;
/// Text token id emitted as the first word ("hello").
const TOK_HELLO: u32 = 1000;
/// Text token id emitted as the second word (" world").
const TOK_WORLD: u32 = 2000;

/// Designed `(token_id, embedding_dimension)` pairs. Each dimension is unique so
/// the one-hot rows stay orthogonal and `logits[token] ≈ S · hidden[dim]`.
const DESIGN_TOKENS: [(u32, usize); 6] = [
    (TOK_TS0, 0),
    (TOK_HELLO, 1),
    (TOK_TS1, 2),
    (TOK_WORLD, 3),
    (TOK_TS2, 4),
    (TOK_EOT, 5),
];

/// Designed `(decoder position, steering dimension)` pairs.
///
/// Position `p` steers the token generated *from* that position onto the token
/// owning that dimension. Positions 2–8 cover the greedy walk for a
/// timestamps-enabled prompt (`[sot, lang, transcribe]`, length 3, so the first
/// generated token is driven by `positional_embedding[2]`). A timestamps-
/// disabled prompt is one token longer, which simply shifts the same table by
/// one and yields `hello world` after timestamp stripping.
const DESIGN_POSITIONS: [(usize, usize); 7] = [
    (2, 0), // <|0.00|>  (initial position: a timestamp is forced)
    (3, 1), // hello     (forced to be text — the opening timestamp has no predecessor)
    (4, 2), // <|1.00|>  (closes the first segment)
    (5, 2), // <|1.00|>  (pair-closing timestamp forced by the lone-timestamp rule)
    (6, 3), // world
    (7, 4), // <|2.00|>  (closes the second segment)
    (8, 5), // <|endoftext|> (legal after a lone timestamp — EOT is never masked)
];

/// Return the designed embedding dimension for a token id, if it is a designed token.
fn designed_token_dim(token: u32) -> Option<usize> {
    DESIGN_TOKENS
        .iter()
        .find(|&&(t, _)| t == token)
        .map(|&(_, d)| d)
}

/// Return the designed steering dimension for a decoder position, if steered.
fn designed_pos_dim(pos: usize) -> Option<usize> {
    DESIGN_POSITIONS
        .iter()
        .find(|&&(p, _)| p == pos)
        .map(|&(_, d)| d)
}

/// `true` for the decoder block output projections (weight and bias) that are
/// zeroed to neutralise the transformer blocks so the residual stream carries
/// only `token_embedding + positional_embedding`.
fn is_neutralised_tensor(name: &str) -> bool {
    if !name.starts_with("decoder.blocks.") {
        return false;
    }
    name.ends_with(".attn.out.weight")
        || name.ends_with(".attn.out.bias")
        || name.ends_with(".cross_attn.out.weight")
        || name.ends_with(".cross_attn.out.bias")
        || name.ends_with(".mlp.2.weight")
        || name.ends_with(".mlp.2.bias")
}

/// Override the default random weight for a tensor element when the designed
/// decoding construction requires a specific value.
///
/// `shape` is the tensor's shape; for the embedding tables the inner (row)
/// dimension is `shape[0]` (GGML `ne[0]`, the fastest-varying axis), matching
/// how `crate::decoder::forward` indexes them (`data[index * n_state + dim]`).
///
/// Returns `None` to fall back to [`deterministic_value`].
fn design_override(name: &str, index: usize, shape: &[usize]) -> Option<f32> {
    // Neutralise the transformer block output projections.
    if is_neutralised_tensor(name) {
        return Some(0.0);
    }
    // Make the final layer norm a pure normalisation.
    if name == "decoder.ln.weight" {
        return Some(1.0);
    }
    if name == "decoder.ln.bias" {
        return Some(0.0);
    }
    // One-hot embedding rows for the designed tokens.
    if name == "decoder.token_embedding.weight" {
        let n_state = shape[0];
        let token = (index / n_state) as u32;
        let dim = index % n_state;
        if let Some(target_dim) = designed_token_dim(token) {
            return Some(if dim == target_dim {
                DESIGN_EMB_SCALE
            } else {
                0.0
            });
        }
        return None;
    }
    // Positional steering vectors for the designed positions.
    if name == "decoder.positional_embedding" {
        let n_state = shape[0];
        let pos = index / n_state;
        let dim = index % n_state;
        if let Some(target_dim) = designed_pos_dim(pos) {
            return Some(if dim == target_dim {
                DESIGN_POS_SCALE
            } else {
                0.0
            });
        }
        return None;
    }
    None
}

/// Value written for a single tensor element: the designed override if present,
/// otherwise the default deterministic pseudo-random value.
fn tensor_element_value(name: &str, name_hash: u64, index: usize, shape: &[usize]) -> f32 {
    design_override(name, index, shape).unwrap_or_else(|| deterministic_value(name_hash, index))
}

/// Text for a vocabulary entry. Designed text tokens carry real words so the
/// decoded transcript is non-empty; every other id is a self-describing
/// placeholder that [`crate::tokenizer::decode`] elides as a special token.
fn designed_vocab_text(i: usize) -> String {
    match i as u32 {
        TOK_HELLO => "hello".to_string(),
        TOK_WORLD => " world".to_string(),
        _ => format!("<|{i}|>"),
    }
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

// ── Multi-speaker diarization fixtures ────────────────────────────────────────
//
// A deterministic multi-speaker AUDIO mixer plus its exact ground-truth RTTM,
// for the diarization evaluation tests (Batch F2). Everything here is a pure,
// closed-form function of the arguments — no randomness, no clock — so the
// generated waveform and reference are byte-for-byte reproducible.

/// Number of round-robin rounds produced by [`synthetic_multispeaker`]: every
/// speaker takes exactly this many turns, so the mixture contains
/// `num_speakers * MULTISPEAKER_ROUNDS` turns in speaker order
/// `0, 1, …, k-1, 0, 1, …`.
#[cfg(feature = "diarization")]
const MULTISPEAKER_ROUNDS: usize = 2;

/// One sample of speaker `speaker`'s *phonation-style* voice at time `t_sec`.
///
/// This is **not** real speech: it is a deterministic voiced-vowel proxy — a
/// glottal fundamental `f0` plus three formant sinusoids, all at
/// speaker-dependent frequencies. Because each speaker owns a distinct
/// `(f0, F1, F2, F3)` tuple, two speakers occupy visibly different spectral
/// bands, so their log-mel spectrograms (and hence any content- or
/// speaker-discriminative embedding) differ. The amplitudes are fixed so the
/// waveform RMS sits far above the VAD energy threshold (a turn is always
/// detected as speech).
#[cfg(feature = "diarization")]
fn speaker_voice_sample(speaker: usize, t_sec: f32) -> f32 {
    let k = speaker as f32;
    // Distinct fundamental per speaker (≈110 Hz, +35 Hz per speaker index).
    let f0 = 110.0 + 35.0 * k;
    // Three formant resonances, each shifted per speaker so the spectral
    // envelope — not just the pitch — is speaker-specific.
    let formants = [520.0 + 130.0 * k, 1500.0 + 260.0 * k, 2600.0 + 320.0 * k];
    let formant_amps = [0.32f32, 0.20, 0.12];

    let two_pi = 2.0 * std::f32::consts::PI;
    let mut s = 0.55 * (two_pi * f0 * t_sec).sin();
    for (&freq, &amp) in formants.iter().zip(formant_amps.iter()) {
        s += amp * (two_pi * freq * t_sec).sin();
    }
    // Peak |s| ≤ 1.19, scaled to ≤ ~0.6 so the signal stays well inside [-1, 1]
    // while its RMS (~0.25) dominates the default 0.01 VAD energy threshold.
    0.5 * s
}

/// Build a deterministic multi-speaker "conversation" and its exact
/// ground-truth RTTM turns.
///
/// The waveform is `num_speakers * MULTISPEAKER_ROUNDS` back-to-back turns of
/// `turn_seconds` each, cycling through the speakers in round-robin order. Turn
/// `i` is spoken by speaker `i % num_speakers` using a distinct voiced profile
/// (a glottal fundamental plus three formant sinusoids at speaker-dependent
/// frequencies), and there is **no** silence between turns — the only acoustic
/// event at a turn boundary is the speaker (spectral-content) change, which is
/// exactly what a diarizer must detect.
///
/// Returns `(audio, reference)` where:
/// * `audio` is mono `f32` PCM at `sample_rate` Hz, length
///   `num_speakers * MULTISPEAKER_ROUNDS * round(turn_seconds * sample_rate)`;
/// * `reference` is the exact ground truth as a `Vec` of
///   [`RttmSegment`](crate::diarize::metrics::RttmSegment), one entry per turn,
///   with speaker name `"speaker_{k}"` and `start`/`end` derived from the turn's
///   **sample** boundaries (so the RTTM times line up bit-for-bit with the audio
///   rather than accumulating `f32` rounding drift).
///
/// # Honesty note (this is a proxy, not speech)
///
/// Each "speaker" here is a fixed sum of sinusoids, so the speaker identity is
/// carried entirely by raw spectral content. A low DER against this fixture
/// therefore does **not** demonstrate real speaker-embedding quality — even a
/// speaker-*invariant* content encoder can separate pure tones at different
/// frequencies. Genuine speaker-discrimination validation needs a pretrained
/// ECAPA-TDNN / x-vector checkpoint and a labelled speech corpus, which is out
/// of scope for this in-crate fixture (see the Batch F2 note in
/// `tests/diarization_synthetic.rs`).
///
/// Degenerate arguments (`num_speakers == 0`, `sample_rate == 0`, or a
/// `turn_seconds` that rounds to zero samples) yield an empty waveform and empty
/// reference rather than panicking.
#[cfg(feature = "diarization")]
pub fn synthetic_multispeaker(
    num_speakers: usize,
    turn_seconds: f32,
    sample_rate: usize,
) -> (Vec<f32>, Vec<crate::diarize::metrics::RttmSegment>) {
    use crate::diarize::metrics::RttmSegment;

    if num_speakers == 0 || sample_rate == 0 {
        return (Vec::new(), Vec::new());
    }
    let turn_samples = (turn_seconds * sample_rate as f32).round() as usize;
    if turn_samples == 0 {
        return (Vec::new(), Vec::new());
    }

    let total_turns = num_speakers * MULTISPEAKER_ROUNDS;
    let mut audio = Vec::with_capacity(turn_samples * total_turns);
    let mut reference = Vec::with_capacity(total_turns);

    for turn in 0..total_turns {
        let speaker = turn % num_speakers;
        let turn_start = turn * turn_samples;
        for i in 0..turn_samples {
            let global = turn_start + i;
            let t_sec = global as f32 / sample_rate as f32;
            audio.push(speaker_voice_sample(speaker, t_sec));
        }
        // Times are taken from the sample-index boundaries so RTTM and audio
        // agree exactly (no drift from re-multiplying turn_seconds in f32).
        reference.push(RttmSegment {
            speaker: format!("speaker_{speaker}"),
            start: turn_start as f32 / sample_rate as f32,
            end: (turn_start + turn_samples) as f32 / sample_rate as f32,
        });
    }

    (audio, reference)
}

#[cfg(all(test, feature = "diarization"))]
mod multispeaker_tests {
    use super::*;

    /// Root-mean-square energy of a slice, in the same units the VAD uses.
    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = samples.iter().map(|&x| (x as f64) * (x as f64)).sum();
        (sum_sq / samples.len() as f64).sqrt() as f32
    }

    #[test]
    fn test_multispeaker_layout_and_ground_truth_are_exact() {
        let sr = 16_000usize;
        let turn_seconds = 2.0f32;
        let (audio, reference) = synthetic_multispeaker(2, turn_seconds, sr);

        // 2 speakers × MULTISPEAKER_ROUNDS(=2) round-robin turns.
        let turn_samples = (turn_seconds * sr as f32).round() as usize;
        assert_eq!(turn_samples, 32_000, "1 turn = 2.0s @ 16 kHz");
        assert_eq!(reference.len(), 2 * MULTISPEAKER_ROUNDS);
        assert_eq!(
            audio.len(),
            turn_samples * 2 * MULTISPEAKER_ROUNDS,
            "audio length must be exactly total_turns × turn_samples"
        );

        // Round-robin speaker order and sample-aligned boundaries.
        let expected_speakers = ["speaker_0", "speaker_1", "speaker_0", "speaker_1"];
        for (i, seg) in reference.iter().enumerate() {
            assert_eq!(seg.speaker, expected_speakers[i], "turn {i} speaker");
            let start = (i * turn_samples) as f32 / sr as f32;
            let end = ((i + 1) * turn_samples) as f32 / sr as f32;
            assert_eq!(seg.start, start, "turn {i} start is sample-aligned");
            assert_eq!(seg.end, end, "turn {i} end is sample-aligned");
        }
        // Contiguous, gap-free timeline (each turn begins where the last ended).
        for pair in reference.windows(2) {
            assert_eq!(pair[0].end, pair[1].start, "turns must be back-to-back");
        }
    }

    #[test]
    fn test_multispeaker_turns_are_audible_and_speaker_distinct() {
        let sr = 16_000usize;
        let (audio, reference) = synthetic_multispeaker(2, 2.0, sr);
        let turn_samples = (2.0f32 * sr as f32).round() as usize;

        // Every turn's RMS clears the default VAD energy threshold (0.01) by a
        // wide margin, so the whole mixture is detected as speech.
        for i in 0..reference.len() {
            let start = i * turn_samples;
            let block = &audio[start..start + turn_samples];
            assert!(
                rms(block) > 0.05,
                "turn {i} RMS {} must clear the VAD threshold",
                rms(block)
            );
        }

        // The two speakers must actually differ acoustically: the sample blocks
        // of speaker_0 (turn 0) and speaker_1 (turn 1) are not equal, and their
        // per-block mean-absolute difference is non-trivial.
        let s0 = &audio[0..turn_samples];
        let s1 = &audio[turn_samples..2 * turn_samples];
        let mean_abs_diff: f64 = s0
            .iter()
            .zip(s1.iter())
            .map(|(&a, &b)| (a - b).abs() as f64)
            .sum::<f64>()
            / turn_samples as f64;
        assert!(
            mean_abs_diff > 0.1,
            "distinct speaker profiles must produce distinct waveforms, got {mean_abs_diff}"
        );
    }

    #[test]
    fn test_multispeaker_three_speakers_round_robin() {
        let sr = 16_000usize;
        let (_audio, reference) = synthetic_multispeaker(3, 1.5, sr);
        assert_eq!(reference.len(), 3 * MULTISPEAKER_ROUNDS);
        let names: Vec<&str> = reference.iter().map(|s| s.speaker.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "speaker_0",
                "speaker_1",
                "speaker_2",
                "speaker_0",
                "speaker_1",
                "speaker_2",
            ],
            "3 speakers must cycle in round-robin order across rounds"
        );
    }

    #[test]
    fn test_multispeaker_degenerate_args_yield_empty() {
        assert_eq!(
            synthetic_multispeaker(0, 1.0, 16_000),
            (Vec::new(), Vec::new())
        );
        assert_eq!(synthetic_multispeaker(2, 1.0, 0), (Vec::new(), Vec::new()));
        // turn_seconds that rounds to zero samples.
        assert_eq!(
            synthetic_multispeaker(2, 0.000_01, 16_000),
            (Vec::new(), Vec::new())
        );
    }
}
