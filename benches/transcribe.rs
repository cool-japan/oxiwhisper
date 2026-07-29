//! Criterion benchmarks for oxiwhisper inference pipeline components.
//!
//! # Regression gate
//!
//! Criterion's own `--baseline` comparison *reports* a regression (a red
//! "Performance has regressed." line) but does **not** make `cargo bench`
//! exit non-zero, so it cannot fail a CI job by itself. `scripts/bench_gate.sh`
//! closes that gap: it parses `target/criterion/**/estimates.json` after a run
//! and exits non-zero when the new mean is more than a threshold (default 10%)
//! slower than the saved baseline mean.
//!
//! Save a baseline (do this on `master` / before a perf-sensitive change):
//! ```text
//! cargo bench --bench transcribe -- --save-baseline main
//! ```
//!
//! Compare a later run against it and print the regression table (does not
//! fail the process by itself):
//! ```text
//! cargo bench --bench transcribe -- --baseline main
//! ```
//!
//! Compare AND fail the process on a >10% regression:
//! ```text
//! cargo bench --bench transcribe -- --baseline main
//! scripts/bench_gate.sh main 10
//! ```
//!
//! `scripts/bench_gate.sh` reads `target/criterion/<group>/<id>/new/estimates.json`
//! (this run) and `target/criterion/<group>/<id>/main/estimates.json` (the
//! saved baseline) and compares `estimates.mean.point_estimate` for every
//! benchmark id present in both. See that script for full usage.
//!
//! # Coverage / reachability notes
//!
//! Decoder self-attention (`scaled_dot_product_cached`, `SdpaScratch`,
//! `CachedSdpaConfig` in `src/decoder/sdpa.rs`) is `pub(crate)` and therefore
//! **unreachable** from this external `benches/` crate — visibility is not
//! changed here. `bench_attention_encoder_scale` below exercises
//! `oxiwhisper::attention::multi_head_attention` instead: it is public, and
//! its per-head QK^T / scores@V matmuls use the identical rayon-gated sgemm
//! pattern (`#[cfg(feature = "parallel")]` + `par_chunks_mut`) as the
//! decoder's cached SDPA, so a regression in that shared kernel machinery
//! shows up here even though the decoder-only code path itself cannot be
//! called directly. `bench_transcribe_e2e` additionally drives the *whole*
//! pipeline (encoder + decoder + KV cache) through the public
//! `WhisperModel::transcribe` API on the `test-utils` synthetic model, which
//! does reach the decoder SDPA code internally, just not as a directly
//! addressable microbenchmark.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::f32::consts::PI;
use std::hint::black_box;
use std::time::Duration;

/// Generate a sine wave at the given frequency.
fn sine_wave(freq_hz: f32, sample_rate: usize, n_samples: usize) -> Vec<f32> {
    (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * PI * freq_hz * t).sin() * 0.3
        })
        .collect()
}

/// Generate a deterministic pseudo-random f32 tensor in `[-1, 1]` from a
/// simple sine-based sequence (matches the deterministic-data style already
/// used by the other benchmarks in this file, so no extra RNG setup or seed
/// bookkeeping is needed just to fill weight matrices).
fn deterministic_tensor(shape: &[usize], phase: f32) -> oxiwhisper::tensor::Tensor {
    let len: usize = shape.iter().product();
    let data: Vec<f32> = (0..len)
        .map(|i| (i as f32 * 0.0173 + phase).sin() * 0.9)
        .collect();
    oxiwhisper::tensor::Tensor::from_vec(data, shape)
}

/// Benchmark mel spectrogram computation at different audio lengths.
fn bench_mel_spectrogram(c: &mut Criterion) {
    let mut group = c.benchmark_group("mel_spectrogram");

    let mel_filters = oxiwhisper::mel_filters::generate_mel_filters();

    for duration_secs in [1, 5, 10, 30] {
        let n_samples = 16000 * duration_secs;
        let audio = sine_wave(440.0, 16000, n_samples);

        group.bench_with_input(
            BenchmarkId::new("duration", format!("{duration_secs}s")),
            &audio,
            |b, audio| {
                b.iter(|| {
                    black_box(
                        oxiwhisper::mel::log_mel_spectrogram(audio, &mel_filters)
                            .expect("mel filter bank must be well formed"),
                    )
                });
            },
        );
    }

    group.finish();
}

/// Benchmark f32 dot product at different vector sizes.
fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");

    for size in [64, 256, 512, 1024, 4096] {
        let a: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin()).collect();
        let b: Vec<f32> = (0..size).map(|i| (i as f32 * 0.002).cos()).collect();

        group.bench_with_input(
            BenchmarkId::new("size", size),
            &(a.clone(), b.clone()),
            |bench, (a, b)| {
                bench.iter(|| black_box(oxiwhisper::linear::dot(a, b)));
            },
        );
    }

    group.finish();
}

/// Benchmark quantized dot products (Q4_0, Q5_0, Q8_0; scalar and SIMD-dispatched).
fn bench_quantized_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantized_dot");

    let n = 1024; // 1024 elements = 32 blocks
    let data_f32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
    let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.02).cos()).collect();

    // Q8_0
    if let Ok(q8_data) = oxiwhisper::quantize::quantize_to_q8_0(&data_f32) {
        group.bench_function("q8_0_scalar", |b| {
            b.iter(|| black_box(oxiwhisper::quantize::dot_q8_0(&input, &q8_data, n)));
        });

        group.bench_function("q8_0_fast", |b| {
            b.iter(|| black_box(oxiwhisper::quantize::dot_q8_0_fast(&input, &q8_data, n)));
        });
    }

    // Q5_0
    if let Ok(q5_data) = oxiwhisper::quantize::quantize_to_q5_0(&data_f32) {
        group.bench_function("q5_0_scalar", |b| {
            b.iter(|| black_box(oxiwhisper::quantize::dot_q5_0(&input, &q5_data, n)));
        });

        group.bench_function("q5_0_fast", |b| {
            b.iter(|| black_box(oxiwhisper::quantize::dot_q5_0_fast(&input, &q5_data, n)));
        });
    }

    // Q4_0
    if let Ok(q4_data) = oxiwhisper::quantize::quantize_to_q4_0(&data_f32) {
        group.bench_function("q4_0_scalar", |b| {
            b.iter(|| black_box(oxiwhisper::quantize::dot_q4_0(&input, &q4_data, n)));
        });

        group.bench_function("q4_0_fast", |b| {
            b.iter(|| black_box(oxiwhisper::quantize::dot_q4_0_fast(&input, &q4_data, n)));
        });
    }

    group.finish();
}

/// Benchmark linear layer (sgemm vs GEMV paths).
fn bench_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear");

    let in_f = 512;
    let out_f = 512;
    let weight = oxiwhisper::tensor::Tensor::from_vec(
        (0..in_f * out_f)
            .map(|i| (i as f32 * 0.001).sin())
            .collect(),
        &[in_f, out_f],
    );

    // GEMV path (batch=1)
    let input_1 = oxiwhisper::tensor::Tensor::from_vec(
        (0..in_f).map(|i| (i as f32 * 0.002).cos()).collect(),
        &[1, in_f],
    );

    group.bench_function("gemv_512x512", |b| {
        b.iter(|| black_box(oxiwhisper::linear::linear(&input_1, &weight, None)));
    });

    // SGEMM path (batch=8)
    let input_8 = oxiwhisper::tensor::Tensor::from_vec(
        (0..8 * in_f).map(|i| (i as f32 * 0.002).cos()).collect(),
        &[8, in_f],
    );

    group.bench_function("sgemm_8x512x512", |b| {
        b.iter(|| black_box(oxiwhisper::linear::linear(&input_8, &weight, None)));
    });

    group.finish();
}

/// Benchmark tensor operations.
fn bench_tensor_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_ops");

    let n = 1500 * 512; // typical encoder output size
    let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();
    let t = oxiwhisper::tensor::Tensor::from_vec(data, &[1500, 512]);

    group.bench_function("gelu_inplace", |b| {
        b.iter(|| {
            let mut t = t.clone();
            t.gelu_inplace();
            black_box(&t);
        });
    });

    let gamma = oxiwhisper::tensor::Tensor::from_vec(vec![1.0; 512], &[512]);
    let beta = oxiwhisper::tensor::Tensor::from_vec(vec![0.0; 512], &[512]);

    group.bench_function("layer_norm_inplace", |b| {
        b.iter(|| {
            let mut t = t.clone();
            t.layer_norm_inplace(&gamma, &beta, 1e-5);
            black_box(&t);
        });
    });

    group.finish();
}

/// Benchmark `multi_head_attention` at `seq_len = 1500`, the real Whisper
/// encoder context length. This is the highest-regression-risk shape for the
/// per-head sgemm / rayon kernels (see the module doc comment above for why
/// this stands in for the unreachable `pub(crate)` decoder SDPA path).
///
/// `n_state=1280, n_head=20` (the `large`-sized configuration) is expensive
/// at this sequence length, so the group uses a reduced sample size /
/// measurement time to keep total runtime bounded — see the tuning notes
/// inline below.
fn bench_attention_encoder_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_encoder_scale");
    // Full seq_len=1500 self-attention is heavy (O(seq^2 * n_head * head_dim)
    // for the QK^T / scores@V matmuls); trade sample count for a bounded
    // wall-clock budget instead of criterion's default 100 samples.
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(8));

    struct HeadConfig {
        label: &'static str,
        n_head: usize,
        head_dim: usize,
    }

    let configs = [
        HeadConfig {
            label: "tiny_n_head6_hd64",
            n_head: 6,
            head_dim: 64,
        },
        HeadConfig {
            label: "small_n_head12_hd64",
            n_head: 12,
            head_dim: 64,
        },
        HeadConfig {
            label: "large_n_head20_hd64",
            n_head: 20,
            head_dim: 64,
        },
    ];

    let seq_len = 1500usize;

    for cfg in &configs {
        let n_state = cfg.n_head * cfg.head_dim;
        let x = deterministic_tensor(&[seq_len, n_state], 0.0);
        let q_weight = deterministic_tensor(&[n_state, n_state], 0.1);
        let q_bias = deterministic_tensor(&[n_state], 0.2);
        let k_weight = deterministic_tensor(&[n_state, n_state], 0.3);
        let v_weight = deterministic_tensor(&[n_state, n_state], 0.4);
        let v_bias = deterministic_tensor(&[n_state], 0.5);
        let out_weight = deterministic_tensor(&[n_state, n_state], 0.6);
        let out_bias = deterministic_tensor(&[n_state], 0.7);

        let weights = oxiwhisper::attention::AttentionWeights {
            q_weight: &q_weight,
            q_bias: &q_bias,
            k_weight: &k_weight,
            v_weight: &v_weight,
            v_bias: &v_bias,
            out_weight: &out_weight,
            out_bias: &out_bias,
        };
        // Encoder self-attention is bidirectional (unmasked), matching the
        // real Whisper encoder block, not the causal decoder self-attention.
        let attn_cfg = oxiwhisper::attention::AttentionConfig {
            n_head: cfg.n_head,
            mask: false,
        };

        group.bench_with_input(
            BenchmarkId::new("seq1500", cfg.label),
            &(x, weights, attn_cfg),
            |b, (x, weights, attn_cfg)| {
                b.iter(|| {
                    black_box(oxiwhisper::attention::multi_head_attention(
                        x, None, weights, attn_cfg,
                    ))
                });
            },
        );
    }

    group.finish();
}

/// End-to-end `WhisperModel::transcribe` on the `test-utils` synthetic model.
///
/// Only compiled when the `test-utils` feature is enabled (the synthetic
/// model generator lives behind that feature and is not otherwise reachable
/// from an external bench crate). Run with:
/// ```text
/// cargo bench --bench transcribe --features test-utils -- transcribe_e2e
/// ```
/// This is the only path in this file that reaches the decoder self-attention
/// (`src/decoder/sdpa.rs`) code, even though it cannot be microbenchmarked
/// directly (see the module doc comment). It is inherently slow — a full
/// greedy decode loop per iteration — so the sample size is cut well below
/// criterion's default of 100.
#[cfg(feature = "test-utils")]
fn bench_transcribe_e2e(c: &mut Criterion) {
    let model_path = oxiwhisper::test_utils::generate_synthetic_model();
    let model = oxiwhisper::WhisperModel::from_file(&model_path)
        .expect("load synthetic WhisperModel for benchmarking");
    let audio = sine_wave(440.0, 16000, 16000); // 1s @ 16kHz

    let mut group = c.benchmark_group("transcribe_e2e");
    // A full greedy decode over the synthetic model's designed token chain is
    // orders of magnitude slower than the microbenchmarks above; keep the
    // sample size small so the whole suite still finishes in minutes.
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("synthetic_1s", |b| {
        b.iter(|| {
            black_box(
                model
                    .transcribe(&audio, &oxiwhisper::TranscribeOptions::default())
                    .expect("synthetic transcribe must succeed"),
            )
        });
    });

    group.finish();

    let _ = std::fs::remove_file(&model_path);
}

#[cfg(not(feature = "test-utils"))]
fn bench_transcribe_e2e(_c: &mut Criterion) {
    // No-op without `test-utils`: the synthetic model generator
    // (`oxiwhisper::test_utils`) is not compiled in, so there is nothing
    // reachable to benchmark here. Kept as a same-named stub (rather than
    // conditionally omitted from `criterion_group!` below) so this file
    // compiles and lints cleanly both with and without the feature, without
    // touching `Cargo.toml`.
}

criterion_group!(
    benches,
    bench_mel_spectrogram,
    bench_dot_product,
    bench_quantized_dot,
    bench_linear,
    bench_tensor_ops,
    bench_attention_encoder_scale,
    bench_transcribe_e2e,
);
criterion_main!(benches);
