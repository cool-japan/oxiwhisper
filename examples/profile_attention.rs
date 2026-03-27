//! Matmul profiling benchmark for oxiwhisper.
//!
//! Profiles QK^T attention score computation at encoder-scale sequence lengths.
//! Compares the tiled 32x32 matmul in `tensor.rs` against `matrixmultiply::sgemm`.
//!
//! Usage:
//!   cargo run --example profile_attention --release

use std::time::{Duration, Instant};

use oxiwhisper::tensor::Tensor;
use rand::RngExt as _;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Fill a Vec<f32> with random values in [-1, 1].
fn random_vec(len: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    (0..len)
        .map(|_| rng.random_range(-1.0f32..1.0f32))
        .collect()
}

/// Compute QK^T for one head using `Tensor::matmul` (tiled 32x32 path).
///
/// q_slice: [seq_len, head_dim]  (row-major)
/// k_slice: [seq_len, head_dim]  (row-major) -- transposed internally
///
/// Returns: [seq_len, seq_len]
fn tiled_qkt(q_slice: &[f32], k_slice: &[f32], seq_len: usize, head_dim: usize) -> Vec<f32> {
    let q = Tensor::from_vec(q_slice.to_vec(), &[seq_len, head_dim]);
    let kt = Tensor::from_vec(k_slice.to_vec(), &[seq_len, head_dim]).transpose_2d();
    let out = q.matmul(&kt);
    out.data
}

/// Compute QK^T for one head using `matrixmultiply::sgemm`.
///
/// q_slice: [seq_len, head_dim]  row-major
/// k_slice: [seq_len, head_dim]  row-major
///
/// sgemm computes C = alpha * A * B + beta * C.
/// We want Q @ K^T where Q is (M, K) and K^T is (K, N).
///   M = seq_len, K = head_dim, N = seq_len
///   A = Q  row-major: rsa = head_dim, csa = 1
///   B = K  interpreted as K^T in col-major: rsb = 1, csb = head_dim
///   C = out row-major: rsc = seq_len, csc = 1
fn sgemm_qkt(q_slice: &[f32], k_slice: &[f32], seq_len: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * seq_len];
    unsafe {
        matrixmultiply::sgemm(
            seq_len,
            head_dim,
            seq_len,
            1.0,
            q_slice.as_ptr(),
            head_dim as isize,
            1,
            k_slice.as_ptr(),
            1,
            head_dim as isize,
            0.0,
            out.as_mut_ptr(),
            seq_len as isize,
            1,
        );
    }
    out
}

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

struct BenchResult {
    avg: Duration,
    best: Duration,
    gflops: f64,
}

/// Run a closure `iters` times with warm-up, then measure `iters` timed runs.
fn bench<F: FnMut()>(mut f: F, iters: usize, flops_per_call: f64) -> BenchResult {
    // Warm-up: 2 iterations
    for _ in 0..2 {
        f();
    }

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        times.push(t0.elapsed());
    }

    let total: Duration = times.iter().sum();
    let avg = total / iters as u32;
    let best = times.iter().copied().min().unwrap_or(Duration::ZERO);

    let avg_secs = avg.as_secs_f64();
    let gflops = if avg_secs > 0.0 {
        flops_per_call / avg_secs / 1e9
    } else {
        0.0
    };

    BenchResult { avg, best, gflops }
}

fn format_duration(d: Duration) -> String {
    let us = d.as_micros();
    if us < 1000 {
        format!("{us} us")
    } else {
        let ms = d.as_secs_f64() * 1e3;
        format!("{ms:.2} ms")
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let n_head: usize = 6;
    let head_dim: usize = 64;
    let seq_lens: &[usize] = &[150, 750, 1500];
    let iters = 10;

    println!("oxiwhisper matmul profiling: QK^T attention scores");
    println!("===================================================");
    println!("n_head={n_head}, head_dim={head_dim}, iters={iters}");
    println!();

    for &seq_len in seq_lens {
        // Generate random Q and K data: [n_head, seq_len, head_dim]
        let q_data = random_vec(n_head * seq_len * head_dim);
        let k_data = random_vec(n_head * seq_len * head_dim);

        // FLOPs for one head's QK^T: seq_len * seq_len * (2 * head_dim - 1)
        // Approximate: 2 * seq_len * seq_len * head_dim
        let flops_per_head = 2.0 * (seq_len as f64) * (seq_len as f64) * (head_dim as f64);
        let flops_total = flops_per_head * (n_head as f64);

        // --- Tiled matmul path ---
        let tiled_result = bench(
            || {
                for h in 0..n_head {
                    let off = h * seq_len * head_dim;
                    let q_slice = &q_data[off..off + seq_len * head_dim];
                    let k_slice = &k_data[off..off + seq_len * head_dim];
                    let _ = tiled_qkt(q_slice, k_slice, seq_len, head_dim);
                }
            },
            iters,
            flops_total,
        );

        // --- sgemm path ---
        let sgemm_result = bench(
            || {
                for h in 0..n_head {
                    let off = h * seq_len * head_dim;
                    let q_slice = &q_data[off..off + seq_len * head_dim];
                    let k_slice = &k_data[off..off + seq_len * head_dim];
                    let _ = sgemm_qkt(q_slice, k_slice, seq_len, head_dim);
                }
            },
            iters,
            flops_total,
        );

        // --- Correctness check (first head only) ---
        let q0 = &q_data[..seq_len * head_dim];
        let k0 = &k_data[..seq_len * head_dim];
        let tiled_out = tiled_qkt(q0, k0, seq_len, head_dim);
        let sgemm_out = sgemm_qkt(q0, k0, seq_len, head_dim);
        let max_diff = tiled_out
            .iter()
            .zip(sgemm_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        let speedup = if sgemm_result.avg.as_nanos() > 0 {
            tiled_result.avg.as_secs_f64() / sgemm_result.avg.as_secs_f64()
        } else {
            f64::NAN
        };

        println!("seq_len={seq_len}, n_head={n_head}, head_dim={head_dim}");
        println!(
            "  tiled_matmul: avg {}, best {}, {:.1} GFLOPS",
            format_duration(tiled_result.avg),
            format_duration(tiled_result.best),
            tiled_result.gflops,
        );
        println!(
            "  sgemm:        avg {}, best {}, {:.1} GFLOPS",
            format_duration(sgemm_result.avg),
            format_duration(sgemm_result.best),
            sgemm_result.gflops,
        );
        println!("  speedup:      {speedup:.2}x (tiled_avg / sgemm_avg, >1 means sgemm is faster)");
        println!("  max_diff:     {max_diff:.6} (numerical agreement)");
        println!();
    }
}
