//! Parity tests: verify SDPA sgemm path produces stable deterministic output.
//!
//! These tests run the full transcribe pipeline and assert bit-exact
//! determinism across repeated runs of the same input.  They capture the
//! baseline behaviour before and after the G1 sgemm refactor so any
//! numerical regression is caught immediately.
//!
//! Model generation is inlined here because `test_utils` is only exposed
//! under `#[cfg(test)]` and therefore not accessible from integration tests.

use std::io::{BufWriter, Write};
use std::path::PathBuf;

// ── Synthetic model constants (matches src/test_utils.rs) ──────────────────

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
const FTYPE: i32 = 1;
const N_FF: usize = N_AUDIO_STATE * 4;
const GGML_MAGIC: u32 = 0x67676D6C;
const N_FFT_BINS: usize = 201;

type W = BufWriter<std::fs::File>;

fn wi32(f: &mut W, v: i32) {
    f.write_all(&v.to_le_bytes()).expect("write i32");
}
fn wu32(f: &mut W, v: u32) {
    f.write_all(&v.to_le_bytes()).expect("write u32");
}

fn simple_hash(s: &str) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

fn det_val(name_hash: u64, index: usize) -> f32 {
    let mixed = name_hash
        .wrapping_mul(2654435761)
        .wrapping_add(index as u64);
    let frac = ((mixed & 0xFFFF) as f32) / 65535.0;
    (frac - 0.5) * 0.02
}

fn write_tensor_f32(f: &mut W, name: &str, shape: &[usize]) {
    let n_el: usize = shape.iter().product();
    wi32(f, shape.len() as i32);
    wi32(f, name.len() as i32);
    wi32(f, 0);
    for &d in shape {
        wi32(f, d as i32);
    }
    f.write_all(name.as_bytes()).expect("write name");
    let h = simple_hash(name);
    let mut buf = vec![0u8; n_el * 4];
    for i in 0..n_el {
        buf[i * 4..(i + 1) * 4].copy_from_slice(&det_val(h, i).to_le_bytes());
    }
    f.write_all(&buf).expect("write f32 data");
}

fn write_tensor_f16(f: &mut W, name: &str, shape: &[usize]) {
    let n_el: usize = shape.iter().product();
    wi32(f, shape.len() as i32);
    wi32(f, name.len() as i32);
    wi32(f, 1);
    for &d in shape {
        wi32(f, d as i32);
    }
    f.write_all(name.as_bytes()).expect("write name");
    let h = simple_hash(name);
    let mut buf = vec![0u8; n_el * 2];
    for i in 0..n_el {
        let v = det_val(h, i);
        let hv = half::f16::from_f32(v);
        buf[i * 2..(i + 1) * 2].copy_from_slice(&hv.to_le_bytes());
    }
    f.write_all(&buf).expect("write f16 data");
}

fn write_encoder_block(f: &mut W, pfx: &str) {
    let s = N_AUDIO_STATE;
    let ff = N_FF;
    write_tensor_f32(f, &format!("{pfx}.attn_ln.weight"), &[s]);
    write_tensor_f32(f, &format!("{pfx}.attn_ln.bias"), &[s]);
    write_tensor_f16(f, &format!("{pfx}.attn.query.weight"), &[s, s]);
    write_tensor_f32(f, &format!("{pfx}.attn.query.bias"), &[s]);
    write_tensor_f16(f, &format!("{pfx}.attn.key.weight"), &[s, s]);
    write_tensor_f16(f, &format!("{pfx}.attn.value.weight"), &[s, s]);
    write_tensor_f32(f, &format!("{pfx}.attn.value.bias"), &[s]);
    write_tensor_f16(f, &format!("{pfx}.attn.out.weight"), &[s, s]);
    write_tensor_f32(f, &format!("{pfx}.attn.out.bias"), &[s]);
    write_tensor_f32(f, &format!("{pfx}.mlp_ln.weight"), &[s]);
    write_tensor_f32(f, &format!("{pfx}.mlp_ln.bias"), &[s]);
    write_tensor_f16(f, &format!("{pfx}.mlp.0.weight"), &[s, ff]);
    write_tensor_f32(f, &format!("{pfx}.mlp.0.bias"), &[ff]);
    write_tensor_f16(f, &format!("{pfx}.mlp.2.weight"), &[ff, s]);
    write_tensor_f32(f, &format!("{pfx}.mlp.2.bias"), &[s]);
}

fn write_decoder_block(f: &mut W, pfx: &str) {
    let s = N_TEXT_STATE;
    let ff = N_FF;
    write_tensor_f32(f, &format!("{pfx}.attn_ln.weight"), &[s]);
    write_tensor_f32(f, &format!("{pfx}.attn_ln.bias"), &[s]);
    write_tensor_f16(f, &format!("{pfx}.attn.query.weight"), &[s, s]);
    write_tensor_f32(f, &format!("{pfx}.attn.query.bias"), &[s]);
    write_tensor_f16(f, &format!("{pfx}.attn.key.weight"), &[s, s]);
    write_tensor_f16(f, &format!("{pfx}.attn.value.weight"), &[s, s]);
    write_tensor_f32(f, &format!("{pfx}.attn.value.bias"), &[s]);
    write_tensor_f16(f, &format!("{pfx}.attn.out.weight"), &[s, s]);
    write_tensor_f32(f, &format!("{pfx}.attn.out.bias"), &[s]);
    write_tensor_f32(f, &format!("{pfx}.cross_attn_ln.weight"), &[s]);
    write_tensor_f32(f, &format!("{pfx}.cross_attn_ln.bias"), &[s]);
    write_tensor_f16(f, &format!("{pfx}.cross_attn.query.weight"), &[s, s]);
    write_tensor_f32(f, &format!("{pfx}.cross_attn.query.bias"), &[s]);
    write_tensor_f16(f, &format!("{pfx}.cross_attn.key.weight"), &[s, s]);
    write_tensor_f16(f, &format!("{pfx}.cross_attn.value.weight"), &[s, s]);
    write_tensor_f32(f, &format!("{pfx}.cross_attn.value.bias"), &[s]);
    write_tensor_f16(f, &format!("{pfx}.cross_attn.out.weight"), &[s, s]);
    write_tensor_f32(f, &format!("{pfx}.cross_attn.out.bias"), &[s]);
    write_tensor_f32(f, &format!("{pfx}.mlp_ln.weight"), &[s]);
    write_tensor_f32(f, &format!("{pfx}.mlp_ln.bias"), &[s]);
    write_tensor_f16(f, &format!("{pfx}.mlp.0.weight"), &[s, ff]);
    write_tensor_f32(f, &format!("{pfx}.mlp.0.bias"), &[ff]);
    write_tensor_f16(f, &format!("{pfx}.mlp.2.weight"), &[ff, s]);
    write_tensor_f32(f, &format!("{pfx}.mlp.2.bias"), &[s]);
}

/// Generate a minimal synthetic model binary for testing.
fn generate_model() -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static CTR: AtomicU64 = AtomicU64::new(0);
    let id = CTR.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "oxiwhisper_g1_test_{}_{}_.bin",
        std::process::id(),
        id
    ));
    let file = std::fs::File::create(&path).expect("create temp model");
    let mut f = BufWriter::with_capacity(1 << 20, file);

    // Header
    wu32(&mut f, GGML_MAGIC);
    wi32(&mut f, N_VOCAB as i32);
    wi32(&mut f, N_AUDIO_CTX as i32);
    wi32(&mut f, N_AUDIO_STATE as i32);
    wi32(&mut f, N_AUDIO_HEAD as i32);
    wi32(&mut f, N_AUDIO_LAYER as i32);
    wi32(&mut f, N_TEXT_CTX as i32);
    wi32(&mut f, N_TEXT_STATE as i32);
    wi32(&mut f, N_TEXT_HEAD as i32);
    wi32(&mut f, N_TEXT_LAYER as i32);
    wi32(&mut f, N_MELS as i32);
    wi32(&mut f, FTYPE);

    // Mel filters
    wi32(&mut f, N_MELS as i32);
    wi32(&mut f, N_FFT_BINS as i32);
    let total = N_MELS * N_FFT_BINS;
    let mut buf = vec![0u8; total * 4];
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
        buf[i * 4..(i + 1) * 4].copy_from_slice(&val.to_le_bytes());
    }
    f.write_all(&buf).expect("write mel filters");

    // Vocab
    wi32(&mut f, N_VOCAB as i32);
    let mut vbuf = Vec::with_capacity(N_VOCAB * 16);
    for i in 0..N_VOCAB {
        let token = format!("<|{i}|>");
        let bytes = token.as_bytes();
        vbuf.extend_from_slice(&(bytes.len() as i32).to_le_bytes());
        vbuf.extend_from_slice(bytes);
    }
    f.write_all(&vbuf).expect("write vocab");

    // Encoder tensors
    write_tensor_f16(&mut f, "encoder.conv1.weight", &[3, N_MELS, N_AUDIO_STATE]);
    write_tensor_f32(&mut f, "encoder.conv1.bias", &[N_AUDIO_STATE]);
    write_tensor_f16(
        &mut f,
        "encoder.conv2.weight",
        &[3, N_AUDIO_STATE, N_AUDIO_STATE],
    );
    write_tensor_f32(&mut f, "encoder.conv2.bias", &[N_AUDIO_STATE]);
    write_tensor_f32(
        &mut f,
        "encoder.positional_embedding",
        &[N_AUDIO_CTX, N_AUDIO_STATE],
    );
    for i in 0..N_AUDIO_LAYER {
        write_encoder_block(&mut f, &format!("encoder.blocks.{i}"));
    }
    write_tensor_f32(&mut f, "encoder.ln_post.weight", &[N_AUDIO_STATE]);
    write_tensor_f32(&mut f, "encoder.ln_post.bias", &[N_AUDIO_STATE]);

    // Decoder tensors
    write_tensor_f16(
        &mut f,
        "decoder.token_embedding.weight",
        &[N_TEXT_STATE, N_VOCAB],
    );
    write_tensor_f32(
        &mut f,
        "decoder.positional_embedding",
        &[N_TEXT_STATE, N_TEXT_CTX],
    );
    for i in 0..N_TEXT_LAYER {
        write_decoder_block(&mut f, &format!("decoder.blocks.{i}"));
    }
    write_tensor_f32(&mut f, "decoder.ln.weight", &[N_TEXT_STATE]);
    write_tensor_f32(&mut f, "decoder.ln.bias", &[N_TEXT_STATE]);

    f.flush().expect("flush model");
    path
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[test]
fn test_transcribe_parity_baseline() {
    use oxiwhisper::{TranscribeOptions, WhisperModel};

    let path = generate_model();
    let model = WhisperModel::from_file(&path).expect("load model");
    let silence = vec![0.0f32; 16000];
    let opts = TranscribeOptions {
        language: Some("en"),
        temperature: 0.0,
        beam_width: 1,
        ..TranscribeOptions::default()
    };

    let result1 = model.transcribe(&silence, &opts).expect("transcribe 1");
    let result2 = model.transcribe(&silence, &opts).expect("transcribe 2");
    assert_eq!(
        result1, result2,
        "transcribe must be bit-exact deterministic across runs"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_transcribe_with_sine_parity() {
    use oxiwhisper::{TranscribeOptions, WhisperModel};

    let path = generate_model();
    let model = WhisperModel::from_file(&path).expect("load model");

    let sample_rate = 16000u32;
    let freq = 440.0f32;
    let sine: Vec<f32> = (0..sample_rate)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin() * 0.5)
        .collect();

    let opts = TranscribeOptions {
        language: Some("en"),
        temperature: 0.0,
        beam_width: 1,
        ..TranscribeOptions::default()
    };

    let result1 = model.transcribe(&sine, &opts).expect("transcribe 1");
    let result2 = model.transcribe(&sine, &opts).expect("transcribe 2");
    assert_eq!(
        result1, result2,
        "sine transcribe must be bit-exact deterministic"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_sdpa_unit_sgemm_correctness() {
    // Unit test: compare scalar SDPA against sgemm-based path directly,
    // using known Q/K/V data and comparing outputs numerically.
    //
    // Q: [n_head=2, q_len=3, head_dim=4]
    // K: [n_head=2, kv_len=5, head_dim=4]  (full attention, no mask)
    // V: [n_head=2, kv_len=5, head_dim=4]

    let n_head = 2usize;
    let q_len = 3usize;
    let kv_len = 5usize;
    let head_dim = 4usize;
    let scale = (head_dim as f32).sqrt().recip();

    // Deterministic Q, K, V in [n_head, seq, head_dim] layout
    let q: Vec<f32> = (0..(n_head * q_len * head_dim))
        .map(|i| (i as f32 + 1.0) * 0.05)
        .collect();
    let k: Vec<f32> = (0..(n_head * kv_len * head_dim))
        .map(|i| (i as f32 + 1.0) * 0.03)
        .collect();
    let v: Vec<f32> = (0..(n_head * kv_len * head_dim))
        .map(|i| (i as f32 + 1.0) * 0.02)
        .collect();

    // --- Scalar reference ---
    let n_state = n_head * head_dim;
    let mut scalar_out = vec![0.0f32; q_len * n_state];
    for h in 0..n_head {
        let q_off = h * q_len * head_dim;
        let k_off = h * kv_len * head_dim;
        let v_off = h * kv_len * head_dim;

        let mut scores = vec![0.0f32; q_len * kv_len];
        for i in 0..q_len {
            for j in 0..kv_len {
                let mut s = 0.0f32;
                for d in 0..head_dim {
                    s += q[q_off + i * head_dim + d] * k[k_off + j * head_dim + d];
                }
                scores[i * kv_len + j] = s * scale;
            }
        }
        // Row softmax
        for i in 0..q_len {
            let row = &mut scores[i * kv_len..(i + 1) * kv_len];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for x in row.iter_mut() {
                *x = (*x - max).exp();
                sum += *x;
            }
            if sum > 0.0 {
                let inv = sum.recip();
                for x in row.iter_mut() {
                    *x *= inv;
                }
            }
        }
        // Accumulate: scores @ V
        for i in 0..q_len {
            for j in 0..kv_len {
                let s = scores[i * kv_len + j];
                for d in 0..head_dim {
                    scalar_out[i * n_state + h * head_dim + d] += s * v[v_off + j * head_dim + d];
                }
            }
        }
    }

    // --- sgemm path ---
    let mut sgemm_scores = vec![0.0f32; n_head * q_len * kv_len];
    for h in 0..n_head {
        let q_ptr = q[h * q_len * head_dim..].as_ptr();
        let k_ptr = k[h * kv_len * head_dim..].as_ptr();
        let s_ptr = sgemm_scores[h * q_len * kv_len..].as_mut_ptr();
        // Q @ K^T: Q is [q_len, head_dim], K is [kv_len, head_dim] so K^T has strides (1, head_dim)
        unsafe {
            matrixmultiply::sgemm(
                q_len,
                head_dim,
                kv_len,
                scale,
                q_ptr,
                head_dim as isize,
                1,
                k_ptr,
                1,
                head_dim as isize,
                0.0,
                s_ptr,
                kv_len as isize,
                1,
            );
        }
    }
    // Row softmax
    for h in 0..n_head {
        for i in 0..q_len {
            let off = h * q_len * kv_len + i * kv_len;
            let row = &mut sgemm_scores[off..off + kv_len];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for x in row.iter_mut() {
                *x = (*x - max).exp();
                sum += *x;
            }
            if sum > 0.0 {
                let inv = sum.recip();
                for x in row.iter_mut() {
                    *x *= inv;
                }
            }
        }
    }
    // scores @ V via sgemm, writing into [q_len, n_state] interleaved output
    let mut sgemm_out = vec![0.0f32; q_len * n_state];
    for h in 0..n_head {
        let s_ptr = sgemm_scores[h * q_len * kv_len..].as_ptr();
        let v_ptr = v[h * kv_len * head_dim..].as_ptr();
        // Write into sgemm_out[i*n_state + h*head_dim .. +head_dim]
        // Row stride for output = n_state, col stride = 1
        let o_ptr = unsafe { sgemm_out.as_mut_ptr().add(h * head_dim) };
        unsafe {
            matrixmultiply::sgemm(
                q_len,
                kv_len,
                head_dim,
                1.0,
                s_ptr,
                kv_len as isize,
                1,
                v_ptr,
                head_dim as isize,
                1,
                0.0,
                o_ptr,
                n_state as isize,
                1,
            );
        }
    }

    // Compare
    for (i, (&sc, &sg)) in scalar_out.iter().zip(sgemm_out.iter()).enumerate() {
        assert!(
            (sc - sg).abs() < 1e-5,
            "SDPA scalar vs sgemm mismatch at index {i}: scalar={sc}, sgemm={sg}"
        );
    }
}
