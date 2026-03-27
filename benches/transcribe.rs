//! Criterion benchmarks for oxiwhisper inference pipeline components.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use std::f32::consts::PI;

/// Generate a sine wave at the given frequency.
fn sine_wave(freq_hz: f32, sample_rate: usize, n_samples: usize) -> Vec<f32> {
    (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * PI * freq_hz * t).sin() * 0.3
        })
        .collect()
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
                b.iter(|| black_box(oxiwhisper::mel::log_mel_spectrogram(audio, &mel_filters)));
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

/// Benchmark quantized dot products (Q4_0 and Q8_0).
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

criterion_group!(
    benches,
    bench_mel_spectrogram,
    bench_dot_product,
    bench_quantized_dot,
    bench_linear,
    bench_tensor_ops,
);
criterion_main!(benches);
