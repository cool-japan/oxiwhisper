//! Synthetic GGML model generator for integration tests.
//!
//! Produces a minimal valid GGML Whisper model binary with deterministic
//! weights. The model will not produce meaningful transcriptions but will
//! exercise the full inference pipeline without crashing.

use std::io::{BufWriter, Write};
use std::path::PathBuf;

// ── Hyperparameters for the synthetic tiny model ────────────────────────────

const N_VOCAB: usize = 51865;
const N_AUDIO_CTX: usize = 1500;
const N_AUDIO_STATE: usize = 384;
const N_AUDIO_HEAD: usize = 6;
const N_AUDIO_LAYER: usize = 4;
const N_TEXT_CTX: usize = 448;
const N_TEXT_STATE: usize = 384;
const N_TEXT_HEAD: usize = 6;
const N_TEXT_LAYER: usize = 4;
const N_MELS: usize = 80;
const FTYPE: i32 = 1; // f16 weights

/// Inner dimension of feed-forward layers (4x model dim for Whisper tiny).
const N_FF: usize = N_AUDIO_STATE * 4; // 1536

/// GGML magic number expected by the loader.
const GGML_MAGIC: u32 = 0x67676D6C;

/// Number of FFT bins used by the mel filter bank (WHISPER_N_FFT / 2 + 1).
const N_FFT_BINS: usize = 201;

/// Type alias for the buffered writer used throughout.
type W = BufWriter<std::fs::File>;

/// Generate a minimal valid GGML Whisper model file with deterministic weights.
///
/// Returns the path to the temporary file. The caller is responsible for
/// deleting it (use [`TempFileCleanup`] in tests).
pub fn generate_synthetic_model() -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir();
    let path = dir.join(format!(
        "oxiwhisper_test_model_{}_{}.bin",
        std::process::id(),
        id
    ));
    let file = std::fs::File::create(&path).expect("create temp model file");
    let mut f = BufWriter::with_capacity(1 << 20, file); // 1 MB buffer

    write_header(&mut f);
    write_mel_filters(&mut f);
    write_vocab(&mut f);
    write_encoder_tensors(&mut f);
    write_decoder_tensors(&mut f);
    f.flush().expect("flush model file");

    path
}

// ── Binary writing helpers ──────────────────────────────────────────────────

fn write_i32(f: &mut W, v: i32) {
    f.write_all(&v.to_le_bytes()).expect("write i32");
}

fn write_u32(f: &mut W, v: u32) {
    f.write_all(&v.to_le_bytes()).expect("write u32");
}

fn write_header(f: &mut W) {
    write_u32(f, GGML_MAGIC);

    write_i32(f, N_VOCAB as i32);
    write_i32(f, N_AUDIO_CTX as i32);
    write_i32(f, N_AUDIO_STATE as i32);
    write_i32(f, N_AUDIO_HEAD as i32);
    write_i32(f, N_AUDIO_LAYER as i32);
    write_i32(f, N_TEXT_CTX as i32);
    write_i32(f, N_TEXT_STATE as i32);
    write_i32(f, N_TEXT_HEAD as i32);
    write_i32(f, N_TEXT_LAYER as i32);
    write_i32(f, N_MELS as i32);
    write_i32(f, FTYPE);
}

fn write_mel_filters(f: &mut W) {
    write_i32(f, N_MELS as i32);
    write_i32(f, N_FFT_BINS as i32);

    // Build the entire filter bank in memory then write once.
    let total = N_MELS * N_FFT_BINS;
    let mut buf = Vec::with_capacity(total * 4);
    for i in 0..total {
        let mel_idx = i / N_FFT_BINS;
        let bin_idx = i % N_FFT_BINS;
        let center = (mel_idx as f32 + 0.5) * N_FFT_BINS as f32 / N_MELS as f32;
        let dist = (bin_idx as f32 - center).abs();
        let width = N_FFT_BINS as f32 / N_MELS as f32;
        let val = if dist < width {
            (1.0 - dist / width) * 0.01
        } else {
            0.0
        };
        buf.extend_from_slice(&val.to_le_bytes());
    }
    f.write_all(&buf).expect("write mel filters");
}

fn write_vocab(f: &mut W) {
    write_i32(f, N_VOCAB as i32);
    // Pre-allocate a buffer for all vocab entries to minimize I/O calls.
    let mut buf = Vec::with_capacity(N_VOCAB * 16);
    for i in 0..N_VOCAB {
        let token = format!("<|{i}|>");
        let bytes = token.as_bytes();
        buf.extend_from_slice(&(bytes.len() as i32).to_le_bytes());
        buf.extend_from_slice(bytes);
    }
    f.write_all(&buf).expect("write vocab");
}

// ── Tensor writing ──────────────────────────────────────────────────────────

/// Write a tensor in f32 format (dtype 0) with deterministic small values.
fn write_tensor_f32(f: &mut W, name: &str, shape: &[usize]) {
    let n_dims = shape.len();
    let n_elements: usize = shape.iter().product();

    write_i32(f, n_dims as i32);
    write_i32(f, name.len() as i32);
    write_i32(f, 0); // dtype = f32

    for &dim in shape {
        write_i32(f, dim as i32);
    }
    f.write_all(name.as_bytes()).expect("write tensor name");

    // Build data buffer in memory then write in one call.
    let name_hash = simple_hash(name);
    let mut buf = vec![0u8; n_elements * 4];
    for i in 0..n_elements {
        let v = deterministic_value(name_hash, i);
        buf[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
    }
    f.write_all(&buf).expect("write f32 data");
}

/// Write a tensor in f16 format (dtype 1) with deterministic small values.
fn write_tensor_f16(f: &mut W, name: &str, shape: &[usize]) {
    let n_dims = shape.len();
    let n_elements: usize = shape.iter().product();

    write_i32(f, n_dims as i32);
    write_i32(f, name.len() as i32);
    write_i32(f, 1); // dtype = f16

    for &dim in shape {
        write_i32(f, dim as i32);
    }
    f.write_all(name.as_bytes()).expect("write tensor name");

    // Build data buffer in memory then write in one call.
    let name_hash = simple_hash(name);
    let mut buf = vec![0u8; n_elements * 2];
    for i in 0..n_elements {
        let v = deterministic_value(name_hash, i);
        let h = half::f16::from_f32(v);
        buf[i * 2..(i + 1) * 2].copy_from_slice(&h.to_le_bytes());
    }
    f.write_all(&buf).expect("write f16 data");
}

/// Produce a deterministic value in [-0.01, 0.01] from a name hash and index.
fn deterministic_value(name_hash: u64, index: usize) -> f32 {
    // Mix bits to get a pseudo-random but deterministic value.
    let mixed = name_hash
        .wrapping_mul(2654435761)
        .wrapping_add(index as u64);
    let frac = ((mixed & 0xFFFF) as f32) / 65535.0; // [0, 1]
    (frac - 0.5) * 0.02 // [-0.01, 0.01]
}

/// Simple string hash (FNV-1a variant).
fn simple_hash(s: &str) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

// ── Encoder tensors ─────────────────────────────────────────────────────────

fn write_encoder_tensors(f: &mut W) {
    // Conv1: weight [kernel=3, in_ch=80, out_ch=384], bias [384]
    write_tensor_f16(f, "encoder.conv1.weight", &[3, N_MELS, N_AUDIO_STATE]);
    write_tensor_f32(f, "encoder.conv1.bias", &[N_AUDIO_STATE]);

    // Conv2: weight [kernel=3, in_ch=384, out_ch=384], bias [384]
    write_tensor_f16(
        f,
        "encoder.conv2.weight",
        &[3, N_AUDIO_STATE, N_AUDIO_STATE],
    );
    write_tensor_f32(f, "encoder.conv2.bias", &[N_AUDIO_STATE]);

    // Positional embedding: [n_audio_ctx, n_audio_state]
    write_tensor_f32(
        f,
        "encoder.positional_embedding",
        &[N_AUDIO_CTX, N_AUDIO_STATE],
    );

    // Transformer blocks
    for i in 0..N_AUDIO_LAYER {
        let pfx = format!("encoder.blocks.{i}");
        write_encoder_block(f, &pfx);
    }

    // Post layer norm
    write_tensor_f32(f, "encoder.ln_post.weight", &[N_AUDIO_STATE]);
    write_tensor_f32(f, "encoder.ln_post.bias", &[N_AUDIO_STATE]);
}

fn write_encoder_block(f: &mut W, pfx: &str) {
    let s = N_AUDIO_STATE;
    let ff = N_FF;

    // Self-attention layer norm
    write_tensor_f32(f, &format!("{pfx}.attn_ln.weight"), &[s]);
    write_tensor_f32(f, &format!("{pfx}.attn_ln.bias"), &[s]);

    // Self-attention Q/K/V/out weights and biases
    // Weight shape in GGML: [in_features, out_features]
    write_tensor_f16(f, &format!("{pfx}.attn.query.weight"), &[s, s]);
    write_tensor_f32(f, &format!("{pfx}.attn.query.bias"), &[s]);
    write_tensor_f16(f, &format!("{pfx}.attn.key.weight"), &[s, s]);
    // Note: key has no bias in Whisper
    write_tensor_f16(f, &format!("{pfx}.attn.value.weight"), &[s, s]);
    write_tensor_f32(f, &format!("{pfx}.attn.value.bias"), &[s]);
    write_tensor_f16(f, &format!("{pfx}.attn.out.weight"), &[s, s]);
    write_tensor_f32(f, &format!("{pfx}.attn.out.bias"), &[s]);

    // MLP layer norm
    write_tensor_f32(f, &format!("{pfx}.mlp_ln.weight"), &[s]);
    write_tensor_f32(f, &format!("{pfx}.mlp_ln.bias"), &[s]);

    // MLP FC1: [n_audio_state, n_ff]
    write_tensor_f16(f, &format!("{pfx}.mlp.0.weight"), &[s, ff]);
    write_tensor_f32(f, &format!("{pfx}.mlp.0.bias"), &[ff]);

    // MLP FC2: [n_ff, n_audio_state]
    write_tensor_f16(f, &format!("{pfx}.mlp.2.weight"), &[ff, s]);
    write_tensor_f32(f, &format!("{pfx}.mlp.2.bias"), &[s]);
}

// ── Decoder tensors ─────────────────────────────────────────────────────────

fn write_decoder_tensors(f: &mut W) {
    // Token embedding: GGML shape [n_text_state, n_vocab]
    // Data layout: [n_vocab, n_text_state] (row per token)
    write_tensor_f16(
        f,
        "decoder.token_embedding.weight",
        &[N_TEXT_STATE, N_VOCAB],
    );

    // Positional embedding: GGML shape [n_text_state, n_text_ctx]
    write_tensor_f32(
        f,
        "decoder.positional_embedding",
        &[N_TEXT_STATE, N_TEXT_CTX],
    );

    // Transformer blocks
    for i in 0..N_TEXT_LAYER {
        let pfx = format!("decoder.blocks.{i}");
        write_decoder_block(f, &pfx);
    }

    // Final layer norm
    write_tensor_f32(f, "decoder.ln.weight", &[N_TEXT_STATE]);
    write_tensor_f32(f, "decoder.ln.bias", &[N_TEXT_STATE]);
}

fn write_decoder_block(f: &mut W, pfx: &str) {
    let s = N_TEXT_STATE;
    let ff = N_FF;

    // Self-attention layer norm
    write_tensor_f32(f, &format!("{pfx}.attn_ln.weight"), &[s]);
    write_tensor_f32(f, &format!("{pfx}.attn_ln.bias"), &[s]);

    // Self-attention Q/K/V/out
    write_tensor_f16(f, &format!("{pfx}.attn.query.weight"), &[s, s]);
    write_tensor_f32(f, &format!("{pfx}.attn.query.bias"), &[s]);
    write_tensor_f16(f, &format!("{pfx}.attn.key.weight"), &[s, s]);
    // key has no bias
    write_tensor_f16(f, &format!("{pfx}.attn.value.weight"), &[s, s]);
    write_tensor_f32(f, &format!("{pfx}.attn.value.bias"), &[s]);
    write_tensor_f16(f, &format!("{pfx}.attn.out.weight"), &[s, s]);
    write_tensor_f32(f, &format!("{pfx}.attn.out.bias"), &[s]);

    // Cross-attention layer norm
    write_tensor_f32(f, &format!("{pfx}.cross_attn_ln.weight"), &[s]);
    write_tensor_f32(f, &format!("{pfx}.cross_attn_ln.bias"), &[s]);

    // Cross-attention Q/K/V/out
    write_tensor_f16(f, &format!("{pfx}.cross_attn.query.weight"), &[s, s]);
    write_tensor_f32(f, &format!("{pfx}.cross_attn.query.bias"), &[s]);
    write_tensor_f16(f, &format!("{pfx}.cross_attn.key.weight"), &[s, s]);
    // cross-attn key has no bias
    write_tensor_f16(f, &format!("{pfx}.cross_attn.value.weight"), &[s, s]);
    write_tensor_f32(f, &format!("{pfx}.cross_attn.value.bias"), &[s]);
    write_tensor_f16(f, &format!("{pfx}.cross_attn.out.weight"), &[s, s]);
    write_tensor_f32(f, &format!("{pfx}.cross_attn.out.bias"), &[s]);

    // MLP layer norm
    write_tensor_f32(f, &format!("{pfx}.mlp_ln.weight"), &[s]);
    write_tensor_f32(f, &format!("{pfx}.mlp_ln.bias"), &[s]);

    // MLP FC1: [n_text_state, n_ff]
    write_tensor_f16(f, &format!("{pfx}.mlp.0.weight"), &[s, ff]);
    write_tensor_f32(f, &format!("{pfx}.mlp.0.bias"), &[ff]);

    // MLP FC2: [n_ff, n_text_state]
    write_tensor_f16(f, &format!("{pfx}.mlp.2.weight"), &[ff, s]);
    write_tensor_f32(f, &format!("{pfx}.mlp.2.bias"), &[s]);
}
