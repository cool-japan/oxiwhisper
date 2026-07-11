//! Per-head threading profile for oxiwhisper's optional `parallel` feature.
//!
//! Profiles [`oxiwhisper::attention::multi_head_attention`], the encoder-side
//! multi-head attention entry point whose QK^T and scores@V matmuls are
//! computed per-head in `src/attention.rs`. That per-head loop is gated by
//! `#[cfg(feature = "parallel")]`: with the feature on, heads are processed
//! concurrently via `rayon::par_chunks_mut`; with it off, the same loop runs
//! serially on the calling thread. The decoder's cached SDPA kernels in
//! `src/decoder/sdpa.rs` use the identical per-head rayon pattern, but every
//! item there (`SdpaScratch`, `scaled_dot_product_cached`, ...) is
//! `pub(crate)` and therefore unreachable from an external example binary —
//! this example exercises the public `attention.rs` path instead, which is
//! representative of the same parallelism mechanism.
//!
//! # IMPORTANT: this binary alone cannot show a serial-vs-parallel speedup
//!
//! `parallel` is a *compile-time* Cargo feature. A single compiled binary
//! only ever contains one of the two code paths (see the `#[cfg(...)]`
//! blocks above), so there is no way to measure both within one process.
//! This example prints which build it is, the thread count in use, and
//! min/avg/max timings for that build only. To actually compare serial vs.
//! parallel performance, run it **twice** and compare the printed numbers
//! by hand:
//!
//! ```text
//! cargo run --release --example profile_threading
//! cargo run --release --example profile_threading --features parallel
//! ```
//!
//! No fabricated "speedup" number is printed — only what was measured in
//! the current build.
//!
//! Usage:
//!   cargo run --release --example profile_threading
//!   cargo run --release --example profile_threading --features parallel

use std::time::{Duration, Instant};

use oxiwhisper::attention::{AttentionConfig, AttentionWeights, multi_head_attention};
use oxiwhisper::tensor::Tensor;
use rand::RngExt as _;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a tensor of the given shape filled with random values in `[-1, 1]`.
fn random_tensor(shape: &[usize]) -> Tensor {
    let mut rng = rand::rng();
    let len: usize = shape.iter().product();
    let data: Vec<f32> = (0..len)
        .map(|_| rng.random_range(-1.0f32..1.0f32))
        .collect();
    Tensor::from_vec(data, shape)
}

/// One attention-layer configuration to profile.
struct HeadConfig {
    /// Human-readable label (approximates a Whisper model size).
    label: &'static str,
    /// Number of attention heads.
    n_head: usize,
    /// Per-head dimension.
    head_dim: usize,
}

/// Wall-clock statistics from a timed benchmark run.
struct BenchStats {
    min: Duration,
    avg: Duration,
    max: Duration,
    gflops_avg: f64,
}

/// Run `f` `iters` times (after a short warm-up) and report min/avg/max
/// wall-clock time, plus an approximate average GFLOPS figure.
fn bench<F: FnMut()>(mut f: F, iters: usize, flops_per_call: f64) -> BenchStats {
    // Warm-up: 2 iterations, discarded (JIT-free in Rust, but this still
    // warms allocator caches and CPU frequency scaling).
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
    let min = times.iter().copied().min().unwrap_or(Duration::ZERO);
    let max = times.iter().copied().max().unwrap_or(Duration::ZERO);

    let avg_secs = avg.as_secs_f64();
    let gflops_avg = if avg_secs > 0.0 {
        flops_per_call / avg_secs / 1e9
    } else {
        0.0
    };

    BenchStats {
        min,
        avg,
        max,
        gflops_avg,
    }
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

/// Approximate total FLOPs for one `multi_head_attention` self-attention
/// call: 4 linear projections (Q, K, V, out) plus the per-head QK^T and
/// scores@V matmuls. This mixes parallel-eligible work (the per-head
/// matmuls, when `parallel` is enabled) with strictly serial work (linear
/// projections, causal masking, softmax, and the final transpose/stitch),
/// so the wall-clock numbers below reflect the *whole* layer, not just the
/// parallel portion — consistent with Amdahl's law.
fn approx_flops(seq_len: usize, n_state: usize, n_head: usize, head_dim: usize) -> f64 {
    let seq = seq_len as f64;
    let state = n_state as f64;
    let heads = n_head as f64;
    let hd = head_dim as f64;

    let flops_proj = 4.0 * 2.0 * seq * state * state;
    let flops_qkt = heads * 2.0 * seq * seq * hd;
    let flops_sv = heads * 2.0 * seq * seq * hd;

    flops_proj + flops_qkt + flops_sv
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("oxiwhisper per-head threading profile");
    println!("======================================");

    #[cfg(feature = "parallel")]
    {
        println!("build:                 `parallel` feature ENABLED");
        println!("                        (src/attention.rs and src/decoder/sdpa.rs per-head");
        println!("                        loops run via rayon::par_iter/par_chunks_mut)");
    }
    #[cfg(not(feature = "parallel"))]
    {
        println!("build:                 `parallel` feature DISABLED");
        println!("                        (per-head loops run serially on the calling thread)");
    }

    let logical_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!("OS-reported logical CPUs: {logical_cpus}");

    #[cfg(feature = "parallel")]
    println!(
        "rayon global thread pool size: {}",
        rayon::current_num_threads()
    );
    #[cfg(not(feature = "parallel"))]
    println!("rayon global thread pool size: n/a (parallel feature disabled)");

    println!();
    println!("NOTE: `parallel` is a compile-time Cargo feature, so this single binary");
    println!("only exercises ONE of the two code paths (see #[cfg(feature = \"parallel\")]");
    println!("in src/attention.rs / src/decoder/sdpa.rs). To compare serial vs. parallel");
    println!("performance, run this example TWICE and compare the numbers by hand:");
    println!("  cargo run --release --example profile_threading");
    println!("  cargo run --release --example profile_threading --features parallel");
    println!();

    let configs = [
        HeadConfig {
            label: "tiny-like  (n_head=6,  head_dim=64, n_state=384)",
            n_head: 6,
            head_dim: 64,
        },
        HeadConfig {
            label: "small-like (n_head=12, head_dim=64, n_state=768)",
            n_head: 12,
            head_dim: 64,
        },
        HeadConfig {
            label: "large-like (n_head=20, head_dim=64, n_state=1280)",
            n_head: 20,
            head_dim: 64,
        },
    ];
    let seq_lens: &[usize] = &[150, 750];
    let iters = 5;

    println!("iters={iters} (+2 discarded warm-up runs) per (config, seq_len) pair");
    println!();

    for cfg in &configs {
        let n_state = cfg.n_head * cfg.head_dim;
        for &seq_len in seq_lens {
            let x = random_tensor(&[seq_len, n_state]);
            let q_weight = random_tensor(&[n_state, n_state]);
            let q_bias = random_tensor(&[n_state]);
            let k_weight = random_tensor(&[n_state, n_state]);
            let v_weight = random_tensor(&[n_state, n_state]);
            let v_bias = random_tensor(&[n_state]);
            let out_weight = random_tensor(&[n_state, n_state]);
            let out_bias = random_tensor(&[n_state]);

            let weights = AttentionWeights {
                q_weight: &q_weight,
                q_bias: &q_bias,
                k_weight: &k_weight,
                v_weight: &v_weight,
                v_bias: &v_bias,
                out_weight: &out_weight,
                out_bias: &out_bias,
            };
            let attn_cfg = AttentionConfig {
                n_head: cfg.n_head,
                mask: true,
            };

            let flops = approx_flops(seq_len, n_state, cfg.n_head, cfg.head_dim);
            let stats = bench(
                || {
                    let _ = multi_head_attention(&x, None, &weights, &attn_cfg);
                },
                iters,
                flops,
            );

            println!("{}, seq_len={seq_len}", cfg.label);
            println!(
                "  min={}  avg={}  max={}  ~{:.1} GFLOPS (avg)",
                format_duration(stats.min),
                format_duration(stats.avg),
                format_duration(stats.max),
                stats.gflops_avg,
            );
            println!();
        }
    }

    println!("Done. Re-run with/without --features parallel and diff the `avg` lines above.");
}
