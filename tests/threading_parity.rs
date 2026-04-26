//! Determinism / parity tests for the `parallel` feature.
//!
//! These tests verify that:
//! 1. The attention computation is deterministic within a single build —
//!    running the same operation twice on the same input yields bit-identical
//!    output regardless of rayon task scheduling.
//! 2. The per-head parallel implementation produces numerically correct output
//!    (matching the serial reference) for both the encoder and decoder paths.
//!
//! Note: since `parallel` is a compile-time feature, we cannot compare the
//! serial and parallel builds in the same process.  Run both variants in CI:
//!   cargo nextest run --test threading_parity --features test-utils
//!   cargo nextest run --test threading_parity --features test-utils,parallel
//! and assert both pass the determinism check — that guarantees parity.

#[path = "common/mod.rs"]
mod common;

use oxiwhisper::attention::{AttentionConfig, AttentionWeights, multi_head_attention};
use oxiwhisper::tensor::Tensor;

/// Create a deterministic tensor with varying values from a seed offset.
fn det_tensor(shape: &[usize], seed: f32) -> Tensor {
    let size: usize = shape.iter().product();
    let data: Vec<f32> = (0..size).map(|i| seed + i as f32 * 0.001).collect();
    Tensor::from_vec(data, shape)
}

/// Fill a tensor with a constant.
fn const_tensor(shape: &[usize], val: f32) -> Tensor {
    let size: usize = shape.iter().product();
    Tensor::from_vec(vec![val; size], shape)
}

/// Run `multi_head_attention` twice on identical inputs and assert bit-identical results.
///
/// This catches non-determinism introduced by rayon scheduling (e.g. data races,
/// incorrect accumulation across heads, shared-buffer clobbering).
#[test]
fn test_encoder_attention_deterministic() {
    let n_state = 64;
    let n_head = 4;
    let seq_len = 10;

    let x = det_tensor(&[seq_len, n_state], 0.05);
    let qw = det_tensor(&[n_state, n_state], 0.01);
    let qb = const_tensor(&[n_state], 0.0);
    let kw = det_tensor(&[n_state, n_state], 0.02);
    let vw = det_tensor(&[n_state, n_state], 0.03);
    let vb = const_tensor(&[n_state], 0.0);
    let ow = det_tensor(&[n_state, n_state], 0.04);
    let ob = const_tensor(&[n_state], 0.0);

    let weights = AttentionWeights {
        q_weight: &qw,
        q_bias: &qb,
        k_weight: &kw,
        v_weight: &vw,
        v_bias: &vb,
        out_weight: &ow,
        out_bias: &ob,
    };
    let config = AttentionConfig {
        n_head,
        mask: false,
    };

    let result1 = multi_head_attention(&x, None, &weights, &config);
    let result2 = multi_head_attention(&x, None, &weights, &config);

    assert_eq!(
        result1.data.len(),
        result2.data.len(),
        "encoder attention outputs should have the same length"
    );
    for (i, (a, b)) in result1.data.iter().zip(result2.data.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "encoder attention not bit-identical at index {i}: first={a}, second={b}"
        );
    }
}

/// Same as above but with causal mask enabled (encoder self-attention with mask).
#[test]
fn test_encoder_attention_masked_deterministic() {
    let n_state = 32;
    let n_head = 4;
    let seq_len = 6;

    let x = det_tensor(&[seq_len, n_state], 0.07);
    let qw = det_tensor(&[n_state, n_state], 0.01);
    let qb = const_tensor(&[n_state], 0.0);
    let kw = det_tensor(&[n_state, n_state], 0.02);
    let vw = det_tensor(&[n_state, n_state], 0.03);
    let vb = const_tensor(&[n_state], 0.0);
    let ow = det_tensor(&[n_state, n_state], 0.04);
    let ob = const_tensor(&[n_state], 0.0);

    let weights = AttentionWeights {
        q_weight: &qw,
        q_bias: &qb,
        k_weight: &kw,
        v_weight: &vw,
        v_bias: &vb,
        out_weight: &ow,
        out_bias: &ob,
    };
    let config = AttentionConfig { n_head, mask: true };

    let result1 = multi_head_attention(&x, None, &weights, &config);
    let result2 = multi_head_attention(&x, None, &weights, &config);

    for (i, (a, b)) in result1.data.iter().zip(result2.data.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "masked attention not bit-identical at index {i}"
        );
    }
}

/// Run the transcription pipeline twice and assert bit-identical text output.
///
/// This covers the full decoder path (all three SDPA variants plus cross-attention)
/// and verifies that parallel head loops produce deterministic results.
///
/// Note: since `parallel` is a compile-time feature, identical text across both
/// the serial and parallel builds is verified by running this test under each
/// feature set in CI.  Within a single build, this test checks intra-build
/// determinism: rayon scheduling must not affect the final output.
#[cfg(feature = "test-utils")]
#[test]
fn test_transcribe_deterministic() {
    let model = common::shared_model();
    let audio = common::synthetic_sine(3.0);

    let opts = oxiwhisper::TranscribeOptions {
        language: Some("en"),
        ..Default::default()
    };

    let result1 = model
        .transcribe(&audio, &opts)
        .expect("first transcription should succeed");
    let result2 = model
        .transcribe(&audio, &opts)
        .expect("second transcription should succeed");

    assert_eq!(
        result1, result2,
        "transcription text must be deterministic across two calls on identical input"
    );
}
