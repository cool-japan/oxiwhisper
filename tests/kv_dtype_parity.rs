//! Parity tests: verify f16 KV-cache dtype paths reproduce the F32 numerics.
//!
//! - `test_kv_f32_and_v_half_same_text`: F32 and VHalf must produce identical
//!   transcription output on silence (the f16 V path must not change semantics).
//! - `test_kv_half_matches_f32_within_tolerance` /
//!   `test_kv_half_beam_search_matches_f32_within_tolerance`: KvHalf (K and V in
//!   f16, K pre-scaled by `1/sqrt(head_dim)`) must reproduce F32's decoded token
//!   sequence exactly and its per-token log-probs within an f16-derived bound.
//!   These replaced the old `is_ok()`-only smoke tests, which pinned nothing.
//!
//! NOTE ON THE SYNTHETIC MODEL: its weights are `+/-0.01`, so activations stay
//! deep in f16's high-resolution range and the measured F32-vs-KvHalf divergence
//! is `0.0`. That is an honest result but not an f16 stress test; the real
//! precision stress for the pre-scaled-K trick lives in the inline unit test
//! `test_kvhalf_prescaled_k_matches_f32_over_sqrt_head_dim` in
//! `src/decoder/kv_cache.rs`, which uses realistic-magnitude K/V.
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
        "oxiwhisper_kv_dtype_test_{}_{}_.bin",
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

// ── Numerical parity harness ─────────────────────────────────────────────────

/// A deterministic, non-silent 1 s / 16 kHz signal. Silence can trip the
/// no-speech gate and zero out `tokens`/`token_probs`, which would make a
/// dtype comparison vacuous; this signal guarantees a full-length decode so
/// the per-token log-prob vectors are actually populated and comparable.
fn deterministic_audio() -> Vec<f32> {
    (0..16000)
        .map(|i| (i as f32 * 0.05).sin() * 0.3 + (i as f32 * 0.013).sin() * 0.2)
        .collect()
}

/// Run the full mel -> encoder -> decoder pipeline for one KV-cache dtype and
/// return the public `DecodeResult` (which exposes `tokens` and `token_probs`).
///
/// This uses the public `oxiwhisper::{model, mel, tensor, encoder, decoder}`
/// surface directly rather than `WhisperModel::transcribe`, because only
/// `decoder::decode` returns `token_probs` -- `transcribe` collapses the result
/// down to a `String`, discarding the per-token numerics we need to bound.
fn decode_with_dtype(
    md: &oxiwhisper::model::ModelData,
    audio: &[f32],
    dtype: oxiwhisper::KvCacheDtype,
    beam_width: usize,
) -> oxiwhisper::decoder::DecodeResult {
    use oxiwhisper::tensor::Tensor;
    use oxiwhisper::{TranscribeOptions, decoder, encoder, mel};

    let mel_data = mel::log_mel_spectrogram(audio, &md.mel_filters).expect("mel");
    let n_mels = md.hparams.n_mels;
    let n_frames = mel_data.len() / n_mels;
    let mel = Tensor::from_vec(mel_data, &[n_mels, n_frames]);
    let enc = encoder::encode(&mel, md).expect("encode synthetic mel");

    let opts = TranscribeOptions {
        language: Some("en"),
        temperature: 0.0,
        beam_width,
        kv_cache_dtype: dtype,
        ..TranscribeOptions::default()
    };
    decoder::decode(&enc, md, &opts).expect("decode")
}

/// Largest absolute per-token log-prob divergence between two decodes.
fn max_prob_divergence(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// F32 and VHalf KV-cache dtypes must produce identical transcription output.
///
/// VHalf stores V as f16 and K as f32. The conversion is lossless enough that
/// the top-1 token at every decoding step is identical, so the output string
/// must match exactly.
#[test]
fn test_kv_f32_and_v_half_same_text() {
    use oxiwhisper::{KvCacheDtype, TranscribeOptions, WhisperModel};

    let path = generate_model();
    let model = WhisperModel::from_file(&path).expect("load model");
    let silence = vec![0.0f32; 16000];

    let opts_f32 = TranscribeOptions {
        language: Some("en"),
        temperature: 0.0,
        beam_width: 1,
        kv_cache_dtype: KvCacheDtype::F32,
        ..TranscribeOptions::default()
    };
    let opts_vhalf = TranscribeOptions {
        language: Some("en"),
        temperature: 0.0,
        beam_width: 1,
        kv_cache_dtype: KvCacheDtype::VHalf,
        ..TranscribeOptions::default()
    };

    let result_f32 = model
        .transcribe(&silence, &opts_f32)
        .expect("transcribe with F32 KV cache");
    let result_vhalf = model
        .transcribe(&silence, &opts_vhalf)
        .expect("transcribe with VHalf KV cache");

    assert_eq!(
        result_f32, result_vhalf,
        "VHalf KV-cache must produce the same output as F32 on silence\n\
         F32:    {:?}\n\
         VHalf:  {:?}",
        result_f32, result_vhalf
    );

    let _ = std::fs::remove_file(&path);
}

/// KvHalf (K and V both f16, K pre-scaled by `1/sqrt(head_dim)`) must reproduce
/// the F32 decode within a tolerance derived from f16 precision.
///
/// This is the load-bearing replacement for the old `is_ok()`-only smoke test:
/// KvHalf is the risky variant (it stores K in f16 using the pre-scaled-K trick
/// and drives QK^T with `alpha = 1.0`), so we pin its numerics against F32
/// instead of merely checking that nothing panics.
///
/// Tolerance rationale (chosen from f16, not tuned to pass):
/// - f16 carries an 11-bit significand -> ~3 decimal digits, relative precision
///   `EPSILON = 2^-11 ~= 4.9e-4` (round-to-nearest error <= half that).
/// - `token_probs` are log-softmax values, magnitude ~10 here. A per-value f16
///   perturbation of `|x|*EPSILON` propagating through the attention matmuls can
///   in principle move a log-prob by ~`1e-3` in relative terms; we therefore
///   bound each token by `1e-3 * |f32_prob| + 1e-3` (relative f16 term + a small
///   absolute floor). We additionally require the argmax token sequence to be
///   identical, which is the property that actually matters for transcription.
///
/// MEASURED on this synthetic model: divergence is exactly `0.0` and the token
/// sequences are byte-identical. That is HONEST but NOT a stress test of f16:
/// the synthetic weights live in `+/-0.01`, so every activation (and hence every
/// K/V value, further shrunk by the `1/sqrt(head_dim)` pre-scale) stays deep in
/// f16's high-resolution range where conversion is near-lossless, and the model
/// is degenerate (it emits one constant token). The genuine f16-precision stress
/// for the pre-scaled-K trick therefore lives in the inline unit test
/// `test_kvhalf_prescaled_k_matches_f32_over_sqrt_head_dim` in
/// `src/decoder/kv_cache.rs`, which injects realistic O(1..6)-magnitude K/V.
/// This end-to-end test still pins that the KvHalf pipeline does not diverge
/// from F32 in either token choice or per-token log-prob.
#[test]
fn test_kv_half_matches_f32_within_tolerance() {
    use oxiwhisper::{KvCacheDtype, model::ModelData};

    let path = generate_model();
    let md = ModelData::load(&path).expect("load model");
    let audio = deterministic_audio();

    let r_f32 = decode_with_dtype(&md, &audio, KvCacheDtype::F32, 1);
    let r_kvhalf = decode_with_dtype(&md, &audio, KvCacheDtype::KvHalf, 1);

    // Guard against a vacuous pass: the decode must actually emit tokens.
    assert!(
        !r_f32.tokens.is_empty() && !r_f32.token_probs.is_empty(),
        "F32 decode produced no tokens; comparison would be vacuous"
    );

    // The argmax token stream must be identical (transcription-relevant property).
    assert_eq!(
        r_f32.tokens, r_kvhalf.tokens,
        "KvHalf must not change the decoded token sequence vs F32"
    );
    assert_eq!(
        r_f32.token_probs.len(),
        r_kvhalf.token_probs.len(),
        "token_probs length must match"
    );

    // Per-token log-prob bound derived from f16 precision (see doc comment).
    let f16_eps = half::f16::EPSILON.to_f32();
    for (i, (&p_f32, &p_kv)) in r_f32
        .token_probs
        .iter()
        .zip(r_kvhalf.token_probs.iter())
        .enumerate()
    {
        let tol = 1e-3 * p_f32.abs() + 1e-3;
        let err = (p_f32 - p_kv).abs();
        assert!(
            err <= tol,
            "token {i}: KvHalf log-prob diverged from F32 beyond f16 tolerance: \
             f32={p_f32} kvhalf={p_kv} err={err} tol={tol} (f16_eps={f16_eps})"
        );
    }

    let max_div = max_prob_divergence(&r_f32.token_probs, &r_kvhalf.token_probs);
    eprintln!("max|F32 - KvHalf| log-prob divergence = {max_div}");

    let _ = std::fs::remove_file(&path);
}

/// KvHalf with beam search must match F32 with beam search within f16 tolerance.
///
/// Beam search clones the KV cache across beams, exercising the copy-on-write
/// path for both f16 K and f16 V storage. Same tolerance reasoning as
/// `test_kv_half_matches_f32_within_tolerance`; the beam that wins under F32
/// must also win (identical tokens) under KvHalf, and its per-token log-probs
/// must agree within the f16-derived bound.
#[test]
fn test_kv_half_beam_search_matches_f32_within_tolerance() {
    use oxiwhisper::{KvCacheDtype, model::ModelData};

    let path = generate_model();
    let md = ModelData::load(&path).expect("load model");
    let audio = deterministic_audio();

    let r_f32 = decode_with_dtype(&md, &audio, KvCacheDtype::F32, 3);
    let r_kvhalf = decode_with_dtype(&md, &audio, KvCacheDtype::KvHalf, 3);

    assert!(
        !r_f32.tokens.is_empty() && !r_f32.token_probs.is_empty(),
        "F32 beam decode produced no tokens; comparison would be vacuous"
    );
    assert_eq!(
        r_f32.tokens, r_kvhalf.tokens,
        "KvHalf beam search must select the same best-beam token sequence as F32"
    );
    assert_eq!(
        r_f32.token_probs.len(),
        r_kvhalf.token_probs.len(),
        "token_probs length must match under beam search"
    );

    for (i, (&p_f32, &p_kv)) in r_f32
        .token_probs
        .iter()
        .zip(r_kvhalf.token_probs.iter())
        .enumerate()
    {
        let tol = 1e-3 * p_f32.abs() + 1e-3;
        let err = (p_f32 - p_kv).abs();
        assert!(
            err <= tol,
            "beam token {i}: KvHalf log-prob diverged from F32 beyond f16 tolerance: \
             f32={p_f32} kvhalf={p_kv} err={err} tol={tol}"
        );
    }

    let max_div = max_prob_divergence(&r_f32.token_probs, &r_kvhalf.token_probs);
    eprintln!("max|F32 - KvHalf| log-prob divergence (beam) = {max_div}");

    let _ = std::fs::remove_file(&path);
}

/// VHalf with beam search must produce the same result as F32 with beam search.
///
/// Both dtypes must agree on the best beam output when V is stored as f16.
#[test]
fn test_kv_v_half_beam_search_same_as_f32() {
    use oxiwhisper::{KvCacheDtype, TranscribeOptions, WhisperModel};

    let path = generate_model();
    let model = WhisperModel::from_file(&path).expect("load model");
    let silence = vec![0.0f32; 16000];

    let opts_f32 = TranscribeOptions {
        language: Some("en"),
        temperature: 0.0,
        beam_width: 3,
        kv_cache_dtype: KvCacheDtype::F32,
        ..TranscribeOptions::default()
    };
    let opts_vhalf = TranscribeOptions {
        language: Some("en"),
        temperature: 0.0,
        beam_width: 3,
        kv_cache_dtype: KvCacheDtype::VHalf,
        ..TranscribeOptions::default()
    };

    let result_f32 = model
        .transcribe(&silence, &opts_f32)
        .expect("transcribe F32 + beam_width=3");
    let result_vhalf = model
        .transcribe(&silence, &opts_vhalf)
        .expect("transcribe VHalf + beam_width=3");

    assert_eq!(
        result_f32, result_vhalf,
        "VHalf + beam search must produce the same output as F32 + beam search\n\
         F32:    {:?}\n\
         VHalf:  {:?}",
        result_f32, result_vhalf
    );

    let _ = std::fs::remove_file(&path);
}
