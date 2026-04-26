//! Verifies that successive forward passes produce identical results.
//!
//! If the `scores_scratch` buffer retains stale values from a previous call
//! and is not properly zeroed/overwritten before use, outputs across runs
//! would differ.  This test catches that class of bug.
//!
//! Model generation is inlined because `test_utils` is `#[cfg(test)]` only
//! and not accessible from integration tests.

use std::io::{BufWriter, Write};
use std::path::PathBuf;

// ── Synthetic model constants ─────────────────────────────────────────────

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

fn generate_model() -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static CTR: AtomicU64 = AtomicU64::new(0);
    let id = CTR.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "oxiwhisper_scratch_test_{}_{}_.bin",
        std::process::id(),
        id
    ));
    let file = std::fs::File::create(&path).expect("create temp model");
    let mut f = BufWriter::with_capacity(1 << 20, file);

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

    wi32(&mut f, N_VOCAB as i32);
    let mut vbuf = Vec::with_capacity(N_VOCAB * 16);
    for i in 0..N_VOCAB {
        let token = format!("<|{i}|>");
        let bytes = token.as_bytes();
        vbuf.extend_from_slice(&(bytes.len() as i32).to_le_bytes());
        vbuf.extend_from_slice(bytes);
    }
    f.write_all(&vbuf).expect("write vocab");

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
fn test_successive_transcribe_identical() {
    use oxiwhisper::{TranscribeOptions, WhisperModel};

    let path = generate_model();
    let model = WhisperModel::from_file(&path).expect("load model");
    let audio = vec![0.0f32; 16000];
    let opts = TranscribeOptions {
        language: Some("en"),
        temperature: 0.0,
        beam_width: 1,
        ..TranscribeOptions::default()
    };

    // Run 3 times — all must be bit-exact identical if scratch is properly handled
    let r1 = model.transcribe(&audio, &opts).expect("r1");
    let r2 = model.transcribe(&audio, &opts).expect("r2");
    let r3 = model.transcribe(&audio, &opts).expect("r3");
    assert_eq!(r1, r2, "run 1 vs 2 must be identical");
    assert_eq!(r2, r3, "run 2 vs 3 must be identical");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_successive_transcribe_different_inputs() {
    use oxiwhisper::{TranscribeOptions, WhisperModel};

    let path = generate_model();
    let model = WhisperModel::from_file(&path).expect("load model");

    let opts = TranscribeOptions {
        language: Some("en"),
        temperature: 0.0,
        beam_width: 1,
        ..TranscribeOptions::default()
    };

    // Alternate between silence and sine wave — each must be self-consistent
    let silence = vec![0.0f32; 16000];
    let sine: Vec<f32> = (0..16000u32)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
        .collect();

    let s1 = model.transcribe(&silence, &opts).expect("silence 1");
    let t1 = model.transcribe(&sine, &opts).expect("sine 1");
    let s2 = model.transcribe(&silence, &opts).expect("silence 2");
    let t2 = model.transcribe(&sine, &opts).expect("sine 2");

    assert_eq!(s1, s2, "silence: run 1 vs 2 must be identical");
    assert_eq!(t1, t2, "sine: run 1 vs 2 must be identical");

    let _ = std::fs::remove_file(&path);
}
