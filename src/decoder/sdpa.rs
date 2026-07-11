//! Scaled dot-product attention (SDPA) kernels for the Whisper decoder.
//!
//! Provides scratch buffer types (`HeadScratch`, `SdpaScratch`) and the
//! attention computation functions used by both self-attention (with a KV
//! cache) and cross-attention (with pre-built flat KV matrices).

use super::kv_cache::LayerKVCache;

/// Per-head scratch buffers for a single attention head.
///
/// Each head gets its own independent scratch so that parallel head loops
/// (when the `parallel` feature is enabled) cannot race on shared state.
///
/// * `scores`    — QK^T result: `[q_len * kv_len]`
/// * `k_dequant` — K dequantization (f16→f32), populated only for KvHalf storage
/// * `v_dequant` — V dequantization (f16→f32), populated only for VHalf/KvHalf storage
/// * `out` — head-major per-head output: `[q_len * head_dim]`.
///   Written by each head's sgemm call, then stitched into the
///   interleaved output buffer by a serial pass.
#[derive(Default)]
pub(crate) struct HeadScratch {
    /// QK^T score matrix for this head: shape `[q_len, kv_len]`.
    pub(crate) scores: Vec<f32>,
    /// K dequantization scratch (f16 -> f32), used only for KvHalf storage.
    pub(crate) k_dequant: Vec<f32>,
    /// V dequantization scratch (f16 -> f32), used only for VHalf/KvHalf storage.
    pub(crate) v_dequant: Vec<f32>,
    /// Head-major output buffer for this head: shape `[q_len, head_dim]`.
    pub(crate) out: Vec<f32>,
}

/// Scratch buffers for SDPA — pre-allocated to avoid per-call heap allocation.
///
/// Holds one `HeadScratch` per attention head.  Grown lazily on first use when
/// `n_head` is not yet known at construction time.  The `forward()` function
/// allocates a single `SdpaScratch` per decode call and reuses it across all
/// layers and both self/cross attention paths.
pub(crate) struct SdpaScratch {
    /// One scratch entry per head.  Empty until first use; grown to `n_head`
    /// on demand inside each sdpa function.
    pub(crate) heads: Vec<HeadScratch>,
}

impl SdpaScratch {
    /// Create an empty scratch container (no heap allocation until first use).
    pub(crate) fn new() -> Self {
        Self { heads: Vec::new() }
    }

    /// Ensure `self.heads` has at least `n` entries, appending defaults as needed.
    pub(crate) fn ensure_heads(&mut self, n: usize) {
        if self.heads.len() < n {
            self.heads.resize_with(n, HeadScratch::default);
        }
    }
}

/// Configuration for the cached scaled dot-product attention.
pub(crate) struct CachedSdpaConfig {
    /// Number of attention heads.
    pub(crate) n_head: usize,
    /// Query sequence length (number of new tokens).
    pub(crate) q_len: usize,
    /// Total KV sequence length (past + q_len).
    pub(crate) kv_len: usize,
    /// Dimension of each head.
    pub(crate) head_dim: usize,
    /// Number of tokens already in the KV cache before this forward pass.
    pub(crate) past_len: usize,
    /// Whether to apply a causal (lower-triangular) attention mask.
    pub(crate) causal_mask: bool,
}

/// Scaled dot-product attention using the pre-allocated KV cache.
///
/// Accepts a mutable `SdpaScratch` to avoid per-call heap allocation inside
/// the head loop.  The score buffer is grown as needed via `Vec::resize`.
/// K/V dequantization buffers are also grown as needed when f16 is in use.
pub(crate) fn scaled_dot_product_cached(
    q: &[f32],
    kv: &LayerKVCache,
    cfg: &CachedSdpaConfig,
    scratch: &mut SdpaScratch,
) -> Vec<f32> {
    // When K is pre-scaled (KvHalf mode), alpha=1.0 avoids double-scaling.
    let scale = if kv.k_prescaled {
        1.0f32
    } else {
        (cfg.head_dim as f32).sqrt().recip()
    };
    let n_state = cfg.n_head * cfg.head_dim;
    let mut out = vec![0.0f32; cfg.q_len * n_state];

    if !cfg.causal_mask {
        sdpa_cached_full(q, kv, cfg, scale, n_state, scratch, &mut out);
    } else if cfg.q_len == 1 {
        sdpa_cached_single(q, kv, cfg, scale, scratch, &mut out);
    } else {
        sdpa_cached_prefill(q, kv, cfg, scale, n_state, scratch, &mut out);
    }

    out
}

/// Full attention (no causal mask) using the KV cache.
///
/// Uses `matrixmultiply::sgemm` for both QK^T and scores@V matrix products.
/// Each head writes into its per-head `HeadScratch::out` buffer (head-major layout),
/// and the results are stitched into the interleaved output in a serial pass.
/// This layout ensures parallel head loops (when `parallel` feature is on) access
/// only disjoint memory, letting the borrow checker verify safety.
fn sdpa_cached_full(
    q: &[f32],
    kv: &LayerKVCache,
    cfg: &CachedSdpaConfig,
    scale: f32,
    n_state: usize,
    scratch: &mut SdpaScratch,
    out: &mut [f32],
) {
    scratch.ensure_heads(cfg.n_head);
    let needed_scores = cfg.q_len * cfg.kv_len;
    let needed_out = cfg.q_len * cfg.head_dim;

    // Prepare per-head scratch sizes and materialise KV into them (must be serial:
    // KvStorage uses shared Arc and the borrow checker cannot split it across threads).
    for h in 0..cfg.n_head {
        let hs = &mut scratch.heads[h];
        if hs.scores.len() < needed_scores {
            hs.scores.resize(needed_scores, 0.0);
        }
        if hs.out.len() < needed_out {
            hs.out.resize(needed_out, 0.0);
        }
        // Materialise K and V into per-head scratch buffers.
        kv.materialize_k_head(h, &mut hs.k_dequant);
        kv.materialize_v_head(h, &mut hs.v_dequant);
    }

    // Per-head computation (parallel when feature = "parallel").
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        scratch.heads[..cfg.n_head]
            .par_iter_mut()
            .enumerate()
            .for_each(|(h, hs)| {
                sdpa_full_one_head(q, hs, cfg, scale, h);
            });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for h in 0..cfg.n_head {
            let hs = &mut scratch.heads[h];
            sdpa_full_one_head(q, hs, cfg, scale, h);
        }
    }

    // Serial stitch: head-major per-head out -> interleaved output [q_len, n_state].
    for h in 0..cfg.n_head {
        let hs = &scratch.heads[h];
        for i in 0..cfg.q_len {
            let dst =
                &mut out[i * n_state + h * cfg.head_dim..i * n_state + (h + 1) * cfg.head_dim];
            let src = &hs.out[i * cfg.head_dim..(i + 1) * cfg.head_dim];
            dst.copy_from_slice(src);
        }
    }
}

/// Compute the full-attention product for a single head, writing into `hs.out`.
///
/// Input layout (head-major, produced by `to_head_first`):
/// * Q: `q[h * q_len * head_dim ..]`  shape `[q_len, head_dim]`
/// * K: `hs.k_dequant`                shape `[kv_len, head_dim]`
/// * V: `hs.v_dequant`                shape `[kv_len, head_dim]`
///
/// Output: `hs.out` shape `[q_len, head_dim]` (head-major, contiguous).
#[inline]
fn sdpa_full_one_head(
    q: &[f32],
    hs: &mut HeadScratch,
    cfg: &CachedSdpaConfig,
    scale: f32,
    h: usize,
) {
    let q_off = h * cfg.q_len * cfg.head_dim;
    let needed = cfg.q_len * cfg.kv_len;
    let scores = &mut hs.scores[..needed];

    // QK^T: Q[q_len, head_dim] x K^T[head_dim, kv_len] = scores[q_len, kv_len]
    // K stored as [kv_len, head_dim]; K^T achieved by swapping strides.
    unsafe {
        matrixmultiply::sgemm(
            cfg.q_len,
            cfg.head_dim,
            cfg.kv_len,
            scale,
            q[q_off..].as_ptr(),
            cfg.head_dim as isize,
            1,
            hs.k_dequant.as_ptr(),
            1,
            cfg.head_dim as isize,
            0.0,
            scores.as_mut_ptr(),
            cfg.kv_len as isize,
            1,
        );
    }

    softmax_rows(scores, cfg.q_len, cfg.kv_len);

    // scores @ V: [q_len, kv_len] x [kv_len, head_dim] -> out[q_len, head_dim] (head-major)
    let needed_out = cfg.q_len * cfg.head_dim;
    let out_slice = &mut hs.out[..needed_out];
    unsafe {
        matrixmultiply::sgemm(
            cfg.q_len,
            cfg.kv_len,
            cfg.head_dim,
            1.0,
            scores.as_ptr(),
            cfg.kv_len as isize,
            1,
            hs.v_dequant.as_ptr(),
            cfg.head_dim as isize,
            1,
            0.0,
            out_slice.as_mut_ptr(),
            cfg.head_dim as isize,
            1,
        );
    }
}

/// Optimised single-query attention (q_len == 1) with causal mask.
///
/// Uses `matrixmultiply::sgemm` for both QK^T (M=1) and score@V (M=1).
/// With q_len==1 the output is a single row of `n_state` elements, so each
/// head writes `head_dim` contiguous values starting at `out[h * head_dim]`.
/// The per-head `HeadScratch::out` buffer is used for uniformity (same
/// stitch path as the multi-token functions), giving each head fully
/// independent scratch even when running in parallel.
fn sdpa_cached_single(
    q: &[f32],
    kv: &LayerKVCache,
    cfg: &CachedSdpaConfig,
    scale: f32,
    scratch: &mut SdpaScratch,
    out: &mut [f32],
) {
    // q_len == 1 is a precondition (enforced by the caller).
    let valid_len = cfg.past_len + 1;
    scratch.ensure_heads(cfg.n_head);

    // Serial KV materialisation into per-head scratch.
    for h in 0..cfg.n_head {
        let hs = &mut scratch.heads[h];
        if hs.scores.len() < valid_len {
            hs.scores.resize(valid_len, 0.0);
        }
        if hs.out.len() < cfg.head_dim {
            hs.out.resize(cfg.head_dim, 0.0);
        }
        kv.materialize_k_head(h, &mut hs.k_dequant);
        kv.materialize_v_head(h, &mut hs.v_dequant);
    }

    // Per-head computation (parallel when feature = "parallel").
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        scratch.heads[..cfg.n_head]
            .par_iter_mut()
            .enumerate()
            .for_each(|(h, hs)| {
                sdpa_single_one_head(q, hs, cfg, scale, valid_len, h);
            });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for h in 0..cfg.n_head {
            let hs = &mut scratch.heads[h];
            sdpa_single_one_head(q, hs, cfg, scale, valid_len, h);
        }
    }

    // Serial stitch: with q_len==1 each head's out is head_dim values going
    // into the flat single-row output at out[h*head_dim..(h+1)*head_dim].
    for h in 0..cfg.n_head {
        let hs = &scratch.heads[h];
        out[h * cfg.head_dim..h * cfg.head_dim + cfg.head_dim]
            .copy_from_slice(&hs.out[..cfg.head_dim]);
    }
}

/// Compute the single-query attention product for one head.
#[inline]
fn sdpa_single_one_head(
    q: &[f32],
    hs: &mut HeadScratch,
    cfg: &CachedSdpaConfig,
    scale: f32,
    valid_len: usize,
    h: usize,
) {
    let q_off = h * cfg.head_dim;
    let scores = &mut hs.scores[..valid_len];

    // QK^T: Q[1, head_dim] x K^T[head_dim, valid_len] = scores[1, valid_len]
    // K stored as [valid_len, head_dim]; K^T via strides (1, head_dim)
    unsafe {
        matrixmultiply::sgemm(
            1,
            cfg.head_dim,
            valid_len,
            scale,
            q[q_off..].as_ptr(),
            cfg.head_dim as isize,
            1,
            hs.k_dequant.as_ptr(),
            1,
            cfg.head_dim as isize,
            0.0,
            scores.as_mut_ptr(),
            valid_len as isize,
            1,
        );
    }

    softmax_rows(scores, 1, valid_len);

    // scores @ V: [1, valid_len] x [valid_len, head_dim] -> out[1, head_dim]
    unsafe {
        matrixmultiply::sgemm(
            1,
            valid_len,
            cfg.head_dim,
            1.0,
            scores.as_ptr(),
            valid_len as isize,
            1,
            hs.v_dequant.as_ptr(),
            cfg.head_dim as isize,
            1,
            0.0,
            hs.out.as_mut_ptr(),
            cfg.head_dim as isize,
            1,
        );
    }
}

/// Optimised prefill attention (q_len > 1) with causal mask.
///
/// Uses a single `sgemm` call to compute the full `[q_len, kv_len]` score
/// matrix, then masks the upper triangle to `-inf` before row-softmax, and
/// uses a second `sgemm` for `scores @ V`.
///
/// Each head writes into its per-head `HeadScratch::out` buffer (head-major),
/// and the results are stitched into the interleaved output in a serial pass.
fn sdpa_cached_prefill(
    q: &[f32],
    kv: &LayerKVCache,
    cfg: &CachedSdpaConfig,
    scale: f32,
    n_state: usize,
    scratch: &mut SdpaScratch,
    out: &mut [f32],
) {
    scratch.ensure_heads(cfg.n_head);
    let needed_scores = cfg.q_len * cfg.kv_len;
    let needed_out = cfg.q_len * cfg.head_dim;

    // Serial KV materialisation.
    for h in 0..cfg.n_head {
        let hs = &mut scratch.heads[h];
        if hs.scores.len() < needed_scores {
            hs.scores.resize(needed_scores, 0.0);
        }
        if hs.out.len() < needed_out {
            hs.out.resize(needed_out, 0.0);
        }
        kv.materialize_k_head(h, &mut hs.k_dequant);
        kv.materialize_v_head(h, &mut hs.v_dequant);
    }

    // Per-head computation.
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        scratch.heads[..cfg.n_head]
            .par_iter_mut()
            .enumerate()
            .for_each(|(h, hs)| {
                sdpa_prefill_one_head(q, hs, cfg, scale, h);
            });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for h in 0..cfg.n_head {
            let hs = &mut scratch.heads[h];
            sdpa_prefill_one_head(q, hs, cfg, scale, h);
        }
    }

    // Serial stitch into interleaved output.
    for h in 0..cfg.n_head {
        let hs = &scratch.heads[h];
        for i in 0..cfg.q_len {
            let dst =
                &mut out[i * n_state + h * cfg.head_dim..i * n_state + (h + 1) * cfg.head_dim];
            let src = &hs.out[i * cfg.head_dim..(i + 1) * cfg.head_dim];
            dst.copy_from_slice(src);
        }
    }
}

/// Compute the prefill (multi-query, causal-masked) attention product for one head.
#[inline]
fn sdpa_prefill_one_head(
    q: &[f32],
    hs: &mut HeadScratch,
    cfg: &CachedSdpaConfig,
    scale: f32,
    h: usize,
) {
    let q_off = h * cfg.q_len * cfg.head_dim;
    let needed = cfg.q_len * cfg.kv_len;
    let scores = &mut hs.scores[..needed];

    // Full QK^T for all q positions at once.
    unsafe {
        matrixmultiply::sgemm(
            cfg.q_len,
            cfg.head_dim,
            cfg.kv_len,
            scale,
            q[q_off..].as_ptr(),
            cfg.head_dim as isize,
            1,
            hs.k_dequant.as_ptr(),
            1,
            cfg.head_dim as isize,
            0.0,
            scores.as_mut_ptr(),
            cfg.kv_len as isize,
            1,
        );
    }

    // Causal mask: query i can only attend to positions 0..=(past_len + i).
    for i in 0..cfg.q_len {
        let valid = cfg.past_len + i + 1;
        for j in valid..cfg.kv_len {
            scores[i * cfg.kv_len + j] = f32::NEG_INFINITY;
        }
    }

    softmax_rows(scores, cfg.q_len, cfg.kv_len);

    // scores @ V -> hs.out (head-major [q_len, head_dim]).
    let needed_out = cfg.q_len * cfg.head_dim;
    let out_slice = &mut hs.out[..needed_out];
    unsafe {
        matrixmultiply::sgemm(
            cfg.q_len,
            cfg.kv_len,
            cfg.head_dim,
            1.0,
            scores.as_ptr(),
            cfg.kv_len as isize,
            1,
            hs.v_dequant.as_ptr(),
            cfg.head_dim as isize,
            1,
            0.0,
            out_slice.as_mut_ptr(),
            cfg.head_dim as isize,
            1,
        );
    }
}

/// Scaled dot-product attention for cross-attention (flat head-first K,V, no causal mask).
///
/// Uses `matrixmultiply::sgemm` for both QK^T and scores@V.
/// Each head writes into a head-major per-head buffer, then a serial stitch
/// copies the results into the interleaved output `[q_len, n_state]`.
///
/// `capture` is an optional out-parameter of shape `[q_len * kv_len]`.  When
/// `Some`, the post-softmax attention matrix is averaged over all heads and
/// written to the slice.  The reduction runs AFTER the parallel/serial section
/// completes — no `&mut` crosses the rayon closure boundary, so there is no
/// aliasing hazard.  The `None` path is byte-for-byte identical to the
/// pre-capture code (no extra allocation, no branch overhead in the hot loop).
#[allow(clippy::too_many_arguments)]
pub(crate) fn scaled_dot_product_flat(
    q: &[f32], // [n_head, q_len, head_dim]
    k: &[f32], // [n_head, kv_len, head_dim]  (pre-built, tightly packed)
    v: &[f32],
    n_head: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    capture: Option<&mut [f32]>, // NEW: [q_len * kv_len] mean-over-heads sink
) -> Vec<f32> {
    let scale = (head_dim as f32).sqrt().recip();
    let n_state = n_head * head_dim;
    let mut out = vec![0.0f32; q_len * n_state];

    // Allocate per-head score and output buffers.
    let mut head_scores: Vec<Vec<f32>> =
        (0..n_head).map(|_| vec![0.0f32; q_len * kv_len]).collect();
    let mut head_out: Vec<Vec<f32>> = (0..n_head)
        .map(|_| vec![0.0f32; q_len * head_dim])
        .collect();

    // Per-head computation.
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        head_scores
            .par_iter_mut()
            .zip(head_out.par_iter_mut())
            .enumerate()
            .for_each(|(h, (scores, h_out))| {
                sdpa_flat_one_head(q, k, v, scores, h_out, h, q_len, kv_len, head_dim, scale);
            });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for h in 0..n_head {
            sdpa_flat_one_head(
                q,
                k,
                v,
                &mut head_scores[h],
                &mut head_out[h],
                h,
                q_len,
                kv_len,
                head_dim,
                scale,
            );
        }
    }

    // Serial stitch into interleaved output.
    for h in 0..n_head {
        for i in 0..q_len {
            let dst = &mut out[i * n_state + h * head_dim..i * n_state + (h + 1) * head_dim];
            let src = &head_out[h][i * head_dim..(i + 1) * head_dim];
            dst.copy_from_slice(src);
        }
    }

    // Optionally capture the post-softmax head-mean attention matrix.
    // Runs AFTER the parallel section — no &mut crosses the rayon closure.
    if let Some(dst) = capture {
        debug_assert_eq!(dst.len(), q_len * kv_len);
        let inv_h = (n_head as f32).recip();
        for v in dst.iter_mut() {
            *v = 0.0;
        }
        for sh in head_scores.iter().take(n_head) {
            for (d, &s) in dst.iter_mut().zip(sh.iter()) {
                *d += s;
            }
        }
        for v in dst.iter_mut() {
            *v *= inv_h;
        }
    }

    out
}

/// Compute the cross-attention product for one head (flat KV, no causal mask).
#[inline]
#[allow(clippy::too_many_arguments)]
fn sdpa_flat_one_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    scores: &mut [f32],
    h_out: &mut [f32],
    h: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
) {
    let q_off = h * q_len * head_dim;
    let kv_off = h * kv_len * head_dim;

    // QK^T: Q[q_len, head_dim] x K^T[head_dim, kv_len] = scores[q_len, kv_len]
    // K stored [kv_len, head_dim]; K^T via strides (1, head_dim).
    unsafe {
        matrixmultiply::sgemm(
            q_len,
            head_dim,
            kv_len,
            scale,
            q[q_off..].as_ptr(),
            head_dim as isize,
            1,
            k[kv_off..].as_ptr(),
            1,
            head_dim as isize,
            0.0,
            scores.as_mut_ptr(),
            kv_len as isize,
            1,
        );
    }

    softmax_rows(scores, q_len, kv_len);

    // scores @ V -> h_out [q_len, head_dim] (head-major, contiguous).
    unsafe {
        matrixmultiply::sgemm(
            q_len,
            kv_len,
            head_dim,
            1.0,
            scores.as_ptr(),
            kv_len as isize,
            1,
            v[kv_off..].as_ptr(),
            head_dim as isize,
            1,
            0.0,
            h_out.as_mut_ptr(),
            head_dim as isize,
            1,
        );
    }
}

#[cfg(test)]
#[path = "sdpa_tests.rs"]
mod tests;

/// Apply row-wise softmax in-place to a `[rows, cols]` score matrix.
pub(crate) fn softmax_rows(scores: &mut [f32], rows: usize, cols: usize) {
    for i in 0..rows {
        let row = &mut scores[i * cols..(i + 1) * cols];
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
