// Component-level micro-benchmark for oxiwhisper internals.
//
// Run with:  cargo run --example bench2 --release
//
// Benchmarks individual components:
//   - Tensor ops: matmul, batched_matmul, softmax, gelu, layer_norm
//   - FFT: power_spectrum at various sizes
//   - Mel spectrogram computation
//   - Linear layer forward pass
//   - Conv1d forward pass

use std::hint::black_box;
use std::time::Instant;

use oxiwhisper::fft::power_spectrum;
use oxiwhisper::linear::{conv1d, linear};
use oxiwhisper::mel::{WHISPER_N_FFT, WHISPER_N_MELS, log_mel_spectrogram};
use oxiwhisper::tensor::Tensor;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Fill a vector with a deterministic pseudo-random pattern (no external deps).
fn pseudo_random_vec(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_add(1);
    (0..len)
        .map(|_| {
            // xorshift32
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            // map to roughly [-1, 1]
            (state as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Create a Tensor filled with pseudo-random data.
fn random_tensor(shape: &[usize], seed: u32) -> Tensor {
    let size: usize = shape.iter().product();
    let data = pseudo_random_vec(size, seed);
    Tensor::from_vec(data, shape)
}

/// Run a benchmark closure `iters` times and report timing statistics.
fn bench<F: FnMut()>(label: &str, iters: usize, mut f: F) {
    // Warm-up run (not counted)
    f();

    let mut times_us = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        f();
        let elapsed = start.elapsed();
        times_us.push(elapsed.as_micros() as f64);
    }

    let total: f64 = times_us.iter().sum();
    let mean = total / iters as f64;
    let min = times_us.iter().copied().fold(f64::INFINITY, f64::min);
    let max = times_us.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if mean >= 1_000_000.0 {
        println!(
            "  {:<42}  {:>8.2} s   (min {:.2} s, max {:.2} s, {} iters)",
            label,
            mean / 1_000_000.0,
            min / 1_000_000.0,
            max / 1_000_000.0,
            iters,
        );
    } else if mean >= 1_000.0 {
        println!(
            "  {:<42}  {:>8.2} ms  (min {:.2} ms, max {:.2} ms, {} iters)",
            label,
            mean / 1_000.0,
            min / 1_000.0,
            max / 1_000.0,
            iters,
        );
    } else {
        println!(
            "  {:<42}  {:>8.2} us  (min {:.2} us, max {:.2} us, {} iters)",
            label, mean, min, max, iters,
        );
    }
}

// ---------------------------------------------------------------------------
// Benchmark groups
// ---------------------------------------------------------------------------

fn bench_matmul() {
    println!("\n=== Tensor matmul (2D) ===");

    let configs: &[(usize, usize, usize, usize)] = &[
        // (M, K, N, iters)
        (64, 64, 64, 50),
        (128, 128, 128, 30),
        (256, 256, 256, 20),
        (512, 512, 512, 10),
        (1, 384, 384, 50),    // single-row (decoder-like)
        (80, 384, 512, 20),   // encoder-like
        (1500, 384, 384, 10), // full encoder seq
    ];

    for &(m, k, n, iters) in configs {
        let a = random_tensor(&[m, k], 42);
        let b = random_tensor(&[k, n], 137);
        let label = format!("matmul [{m}x{k}] x [{k}x{n}]");
        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        bench(&label, iters, || {
            let _ = black_box(a.matmul(black_box(&b)));
        });
        // Print throughput for the largest sizes
        if m * k * n >= 128 * 128 * 128 {
            // Re-time for throughput estimate
            let start = Instant::now();
            for _ in 0..iters {
                let _ = black_box(a.matmul(black_box(&b)));
            }
            let elapsed_s = start.elapsed().as_secs_f64();
            let gflops = (flops * iters as f64) / elapsed_s / 1e9;
            println!("    -> throughput: {gflops:.2} GFLOP/s");
        }
    }
}

fn bench_batched_matmul() {
    println!("\n=== Tensor batched_matmul ===");

    let configs: &[(usize, usize, usize, usize, usize)] = &[
        // (batch, M, K, N, iters)
        (6, 80, 64, 80, 20),    // small multi-head attention shape
        (6, 1500, 64, 1500, 5), // encoder self-attention QK^T
        (6, 1500, 1500, 64, 5), // encoder self-attention attn@V
        (1, 256, 256, 256, 20),
    ];

    for &(batch, m, k, n, iters) in configs {
        let a = random_tensor(&[batch, m, k], 42);
        let b = random_tensor(&[batch, k, n], 137);
        let label = format!("batched_matmul [{batch}x{m}x{k}] x [{batch}x{k}x{n}]");
        bench(&label, iters, || {
            let _ = black_box(a.batched_matmul(black_box(&b)));
        });
    }
}

fn bench_softmax() {
    println!("\n=== Tensor softmax ===");

    let configs: &[(usize, usize, usize)] = &[
        // (rows, cols, iters)
        (6 * 1500, 1500, 10), // encoder self-attention scores
        (6, 1500, 50),        // decoder cross-attention scores (single step)
        (1, 51865, 30),       // vocabulary softmax
    ];

    for &(rows, cols, iters) in configs {
        let t = random_tensor(&[rows, cols], 42);
        let label = format!("softmax [{rows}x{cols}]");
        bench(&label, iters, || {
            let _ = black_box(t.softmax());
        });
    }
}

fn bench_gelu() {
    println!("\n=== Tensor gelu ===");

    let configs: &[(usize, usize)] = &[
        // (numel, iters)
        (1500 * 1536, 20), // encoder FFN intermediate
        (1536, 50),        // decoder FFN intermediate (single step)
    ];

    for &(numel, iters) in configs {
        let t = random_tensor(&[numel], 42);
        let label = format!("gelu [{numel}]");
        bench(&label, iters, || {
            let _ = black_box(t.gelu());
        });
    }
}

fn bench_layer_norm() {
    println!("\n=== Tensor layer_norm ===");

    let configs: &[(&[usize], usize)] = &[
        // (shape, iters)
        (&[1500, 384], 30), // encoder sequence
        (&[1, 384], 50),    // single decoder step
        (&[80, 512], 30),   // larger dim
    ];

    for &(shape, iters) in configs {
        let dim = shape[shape.len() - 1];
        let t = random_tensor(shape, 42);
        let w = random_tensor(&[dim], 100);
        let b = random_tensor(&[dim], 200);
        let label = format!(
            "layer_norm [{}]",
            shape
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("x")
        );
        bench(&label, iters, || {
            let _ = black_box(t.layer_norm(black_box(&w), black_box(&b), 1e-5));
        });
    }
}

fn bench_power_spectrum() {
    println!("\n=== FFT power_spectrum ===");

    let configs: &[(usize, usize)] = &[
        // (n_fft, iters)
        (256, 50),
        (400, 50), // Whisper default
        (512, 50),
        (1024, 30),
        (2048, 20),
        (4096, 10),
    ];

    for &(n_fft, iters) in configs {
        let input = pseudo_random_vec(n_fft, 42);
        let label = format!("power_spectrum (n_fft={n_fft})");
        bench(&label, iters, || {
            let _ = black_box(power_spectrum(black_box(&input), n_fft));
        });
    }
}

fn bench_mel_spectrogram() {
    println!("\n=== Mel spectrogram ===");

    // Number of mel filter bins for Whisper
    let n_bins = WHISPER_N_FFT / 2 + 1;
    let mel_filters = pseudo_random_vec(WHISPER_N_MELS * n_bins, 99);

    let configs: &[(f32, usize)] = &[
        // (duration_seconds, iters)
        (1.0, 20),
        (5.0, 10),
        (10.0, 5),
        (30.0, 3), // full Whisper chunk
    ];

    for &(duration_s, iters) in configs {
        let n_samples = (16000.0 * duration_s) as usize;
        let audio = pseudo_random_vec(n_samples, 42);
        let label = format!("log_mel_spectrogram ({duration_s:.0}s audio, {n_samples} samples)");
        bench(&label, iters, || {
            let _ = black_box(log_mel_spectrogram(
                black_box(&audio),
                black_box(&mel_filters),
            ));
        });
    }
}

fn bench_linear_layer() {
    println!("\n=== Linear layer forward ===");

    let configs: &[(usize, usize, usize, bool, usize)] = &[
        // (batch, in_f, out_f, use_bias, iters)
        (1, 384, 384, true, 50),    // decoder single-step
        (1, 384, 1536, true, 30),   // decoder FFN up-proj
        (1500, 384, 384, true, 10), // encoder full-seq proj
        (1500, 384, 1536, true, 5), // encoder FFN up-proj
        (1500, 1536, 384, true, 5), // encoder FFN down-proj
        (1, 384, 384, false, 50),   // no-bias variant
    ];

    for &(batch, in_f, out_f, use_bias, iters) in configs {
        let input = random_tensor(&[batch, in_f], 42);
        let weight = random_tensor(&[in_f, out_f], 137);
        let bias_tensor = random_tensor(&[out_f], 200);
        let bias: Option<&Tensor> = if use_bias { Some(&bias_tensor) } else { None };
        let bias_tag = if use_bias { "+bias" } else { "" };
        let label = format!("linear [{batch}x{in_f}] -> [{batch}x{out_f}]{bias_tag}");
        let flops = 2.0 * batch as f64 * in_f as f64 * out_f as f64;
        bench(&label, iters, || {
            let _ = black_box(linear(black_box(&input), black_box(&weight), bias));
        });
        if batch * in_f * out_f >= 384 * 384 {
            let start = Instant::now();
            for _ in 0..iters {
                let _ = black_box(linear(black_box(&input), black_box(&weight), bias));
            }
            let elapsed_s = start.elapsed().as_secs_f64();
            let gflops = (flops * iters as f64) / elapsed_s / 1e9;
            println!("    -> throughput: {gflops:.2} GFLOP/s");
        }
    }
}

fn bench_conv1d() {
    println!("\n=== Conv1d forward ===");

    type Conv1dConfig = (usize, usize, usize, usize, usize, usize, usize, usize);
    let configs: &[Conv1dConfig] = &[
        // (batch, in_ch, out_ch, in_len, kernel, stride, padding, iters)
        (1, 80, 384, 3000, 3, 1, 1, 10),  // Whisper encoder conv1
        (1, 384, 384, 3000, 3, 2, 1, 10), // Whisper encoder conv2 (stride=2)
        (1, 80, 384, 1500, 3, 1, 1, 15),  // Half-length variant
        (1, 1, 64, 16000, 3, 1, 1, 5),    // Narrow single-channel
    ];

    for &(batch, in_ch, out_ch, in_len, kernel, stride, padding, iters) in configs {
        let input = random_tensor(&[batch, in_ch, in_len], 42);
        let weight = random_tensor(&[kernel, in_ch, out_ch], 137);
        let bias = random_tensor(&[out_ch], 200);
        let out_len = (in_len + 2 * padding).saturating_sub(kernel) / stride + 1;
        let label = format!(
            "conv1d [{batch}x{in_ch}x{in_len}] k={kernel} s={stride} -> [{batch}x{out_ch}x{out_len}]"
        );
        bench(&label, iters, || {
            let _ = black_box(conv1d(
                black_box(&input),
                black_box(&weight),
                black_box(&bias),
                stride,
                padding,
            ));
        });
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("oxiwhisper component micro-benchmarks");
    println!("======================================");
    println!(
        "Platform: {} {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );

    bench_matmul();
    bench_batched_matmul();
    bench_softmax();
    bench_gelu();
    bench_layer_norm();
    bench_power_spectrum();
    bench_mel_spectrogram();
    bench_linear_layer();
    bench_conv1d();

    println!("\nDone.");
}
