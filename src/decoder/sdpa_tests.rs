//! Inline tests for the decoder SDPA kernels.
//!
//! This is a `#[path]` child module of `sdpa.rs`, so it can reach the parent
//! module's private items (`sdpa_cached_single`, `sdpa_cached_prefill`, …) and
//! `pub(crate)` API (`scaled_dot_product_cached`, `scaled_dot_product_flat`,
//! `softmax_rows`, `SdpaScratch`, `CachedSdpaConfig`).  An integration test
//! under `tests/` physically cannot — none of these symbols are `pub`.
//!
//! Strategy: an INDEPENDENT naive scalar reference SDPA is derived directly
//! from the definition of attention (triple-nested QK^T, causal mask, row
//! softmax, scores@V) and the real crate functions are asserted to match it.
//! The reference is NOT copied from the production code — it uses plain scalar
//! loops rather than `matrixmultiply::sgemm`, so the two implementations share
//! no arithmetic and a bug in either surfaces as a mismatch.
//!
//! ## Tolerances
//!
//! * `F32_ABS_TOL = 1e-5` — f32 path.  The reference accumulates a QK^T dot
//!   product and a scores@V dot product in scalar order, while production uses
//!   blocked `sgemm` with a different summation order (and possibly FMA).  For
//!   the small head dimensions used here (`head_dim <= 8`) with inputs in
//!   `[-1, 1]`, the relative error between two f32 summation orders is on the
//!   order of `head_dim * f32::EPSILON ≈ 8 * 1.2e-7 ≈ 1e-6`; `1e-5` gives a
//!   comfortable safety margin without hiding real errors.
//!
//! * `VHALF_ABS_TOL = 5e-3` — V stored as f16 (K still f32).  The scores are
//!   computed in f32, so `sum_j p[i,j] = 1` exactly (to f32).  The output error
//!   is `sum_j p[i,j] * (f16(V)-V)`, bounded by `max_j |V[j]| * 2^-11`.  With
//!   `|V| <= 1` this is `~4.9e-4`; `5e-3` covers that plus f32 `sgemm` noise
//!   with a 10x margin.
//!
//! * `KVHALF_ABS_TOL = 2e-2` — K and V both f16 (K pre-scaled by
//!   `1/sqrt(head_dim)`).  K rounding perturbs the pre-softmax logits by
//!   `sum_d Q[d] * (f16(K_scaled)-K_scaled)`, bounded by
//!   `head_dim * |Q| * |K|/sqrt(head_dim) * 2^-11`.  With `head_dim = 8`,
//!   `|Q|,|K| <= 1` that is `~sqrt(8) * 4.9e-4 ≈ 1.4e-3` of logit error, which
//!   the softmax turns into a comparable probability perturbation, then the
//!   f16-V output error compounds on top.  `2e-2` is an honest bound with
//!   margin — it is NOT the f32 tolerance quietly loosened.

use super::*;
use crate::types::KvCacheDtype;

const F32_ABS_TOL: f32 = 1e-5;
const VHALF_ABS_TOL: f32 = 5e-3;
const KVHALF_ABS_TOL: f32 = 2e-2;

// ── Deterministic input generation (no rand, no time, no filesystem) ─────────

/// Deterministic pseudo-value in `[-1, 1]`, keyed by a salt and three indices.
///
/// Uses a small integer mix folded through `sin`, so different tensors (via
/// `salt`) and different `(head, seq, dim)` positions get well-spread values
/// without any external randomness.
fn gen_val(salt: usize, h: usize, s: usize, d: usize) -> f32 {
    let key = salt
        .wrapping_mul(1009)
        .wrapping_add(h.wrapping_mul(9176))
        .wrapping_add(s.wrapping_mul(263))
        .wrapping_add(d.wrapping_mul(31))
        .wrapping_add(7);
    ((key as f32) * 0.017f32).sin()
}

/// Build a head-major `[n_head, seq, head_dim]` tensor of deterministic values.
fn make_head_major(salt: usize, n_head: usize, seq: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_head * seq * head_dim];
    for h in 0..n_head {
        for s in 0..seq {
            for d in 0..head_dim {
                out[h * seq * head_dim + s * head_dim + d] = gen_val(salt, h, s, d);
            }
        }
    }
    out
}

/// Convert a head-major `[n_head, seq, head_dim]` tensor to the interleaved
/// row-major `[seq, n_state]` layout that `LayerKVCache::append` consumes.
fn interleave(head_major: &[f32], n_head: usize, seq: usize, head_dim: usize) -> Vec<f32> {
    let n_state = n_head * head_dim;
    let mut out = vec![0.0f32; seq * n_state];
    for h in 0..n_head {
        for s in 0..seq {
            for d in 0..head_dim {
                out[s * n_state + h * head_dim + d] =
                    head_major[h * seq * head_dim + s * head_dim + d];
            }
        }
    }
    out
}

// ── Independent naive scalar reference ───────────────────────────────────────

/// Output of the reference: the interleaved attention output plus the
/// head-mean post-softmax attention matrix (mirrors the `capture` argument of
/// `scaled_dot_product_flat`).
struct Reference {
    /// Interleaved attention output `[q_len, n_state]`.
    out: Vec<f32>,
    /// Head-mean post-softmax attention `[q_len, kv_len]`.
    attn_mean: Vec<f32>,
}

/// Independent naive scalar SDPA, derived from the definition of attention.
///
/// * `q_hm`: head-major query `[n_head, q_len, head_dim]`.
/// * `k_hm`, `v_hm`: head-major key/value `[n_head, kv_len, head_dim]`.
///
/// For each head `h` and query `i`:
///   `logit[j] = (1/sqrt(head_dim)) * sum_d q[h,i,d] * k[h,j,d]`
///   causal mask (when `causal`): `logit[j] = -inf` for `j > past_len + i`
///   `p[j] = softmax_j(logit)`
///   `out[i, h, d] = sum_j p[j] * v[h,j,d]`
///
/// `attn_mean[i,j] = (1/n_head) * sum_h p_h[i,j]`.
#[allow(clippy::too_many_arguments)]
fn naive_sdpa(
    q_hm: &[f32],
    k_hm: &[f32],
    v_hm: &[f32],
    n_head: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    past_len: usize,
    causal: bool,
) -> Reference {
    let n_state = n_head * head_dim;
    let scale = (head_dim as f32).sqrt().recip();
    let mut out = vec![0.0f32; q_len * n_state];
    let mut attn_mean = vec![0.0f32; q_len * kv_len];
    let inv_h = (n_head as f32).recip();

    for h in 0..n_head {
        let q_base = h * q_len * head_dim;
        let kv_base = h * kv_len * head_dim;
        for i in 0..q_len {
            // Pre-softmax logits for this (head, query).
            let mut logits = vec![0.0f32; kv_len];
            for j in 0..kv_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_hm[q_base + i * head_dim + d] * k_hm[kv_base + j * head_dim + d];
                }
                logits[j] = scale * dot;
            }
            // Causal mask: query i (global position past_len + i) may attend to
            // keys 0..=past_len+i only.
            if causal {
                let valid = past_len + i + 1;
                for logit in logits.iter_mut().skip(valid.min(kv_len)) {
                    *logit = f32::NEG_INFINITY;
                }
            }
            // Numerically-stable row softmax.
            let maxv = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = vec![0.0f32; kv_len];
            let mut sum = 0.0f32;
            for j in 0..kv_len {
                let e = (logits[j] - maxv).exp();
                probs[j] = e;
                sum += e;
            }
            if sum > 0.0 {
                let inv = sum.recip();
                for p in probs.iter_mut() {
                    *p *= inv;
                }
            }
            // Weighted sum of V, and head-mean attention accumulation.
            for j in 0..kv_len {
                attn_mean[i * kv_len + j] += probs[j] * inv_h;
                let pj = probs[j];
                for d in 0..head_dim {
                    out[i * n_state + h * head_dim + d] += pj * v_hm[kv_base + j * head_dim + d];
                }
            }
        }
    }

    Reference { out, attn_mean }
}

// ── Assertion helpers ────────────────────────────────────────────────────────

/// Assert `got[i]` is within `abs_tol + rel_tol*|want[i]|` of `want[i]`.
fn assert_close_tol(got: &[f32], want: &[f32], abs_tol: f32, rel_tol: f32, ctx: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{ctx}: length mismatch (got {}, want {})",
        got.len(),
        want.len()
    );
    for (idx, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let allowed = abs_tol + rel_tol * w.abs();
        let diff = (g - w).abs();
        assert!(
            diff <= allowed,
            "{ctx}: element {idx} diverged: got {g}, want {w}, diff {diff} > allowed {allowed}"
        );
    }
}

/// Absolute-tolerance closeness (rel_tol = 0).
fn assert_close(got: &[f32], want: &[f32], abs_tol: f32, ctx: &str) {
    assert_close_tol(got, want, abs_tol, 0.0, ctx);
}

/// Build an F32 KV cache and populate it head-major from interleaved input.
fn build_cache(
    dtype: KvCacheDtype,
    n_head: usize,
    head_dim: usize,
    kv_len: usize,
    k_hm: &[f32],
    v_hm: &[f32],
) -> LayerKVCache {
    // Capacity slightly larger than kv_len to exercise the per-head stride.
    let capacity = kv_len + 3;
    let mut cache = LayerKVCache::new_with_dtype(n_head, head_dim, capacity, dtype);
    let k_il = interleave(k_hm, n_head, kv_len, head_dim);
    let v_il = interleave(v_hm, n_head, kv_len, head_dim);
    cache.append(&k_il, &v_il, kv_len);
    cache
}

// ── softmax_rows ─────────────────────────────────────────────────────────────

#[test]
fn softmax_rows_basic_matches_definition() {
    // Two independent rows; check each is a valid probability distribution and
    // matches exp(x-max)/sum computed independently.
    let cols = 4;
    let mut scores = vec![0.5f32, -1.0, 2.0, 0.0, 3.0, 3.0, 1.0, -2.0];
    let original = scores.clone();
    softmax_rows(&mut scores, 2, cols);

    for r in 0..2 {
        let row = &original[r * cols..(r + 1) * cols];
        let maxv = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&x| (x - maxv).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let want: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
        assert_close(
            &scores[r * cols..(r + 1) * cols],
            &want,
            1e-6,
            "softmax row",
        );
        let got_sum: f32 = scores[r * cols..(r + 1) * cols].iter().sum();
        assert!(
            (got_sum - 1.0).abs() < 1e-6,
            "row {r} must sum to 1, got {got_sum}"
        );
    }
}

#[test]
fn softmax_rows_handles_neg_infinity() {
    // A row with -inf entries: those positions must become EXACTLY 0.0, and the
    // finite positions must form a valid softmax over just themselves.
    let cols = 4;
    let mut scores = vec![1.0f32, f32::NEG_INFINITY, 2.0, f32::NEG_INFINITY];
    softmax_rows(&mut scores, 1, cols);

    assert_eq!(scores[1], 0.0, "masked position 1 must be exactly 0");
    assert_eq!(scores[3], 0.0, "masked position 3 must be exactly 0");
    let sum: f32 = scores.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "finite positions must sum to 1");
    // ratio p2/p0 == exp(2-1) == e.
    let ratio = scores[2] / scores[0];
    assert!(
        (ratio - std::f32::consts::E).abs() < 1e-5,
        "p2/p0 should equal e, got {ratio}"
    );
}

#[test]
fn softmax_rows_all_masked_row_is_an_unsupported_precondition() {
    // Characterisation test (pins ACTUAL behaviour, not desired output): an
    // all-`-inf` row is a PRECONDITION VIOLATION that `softmax_rows` does not
    // defend against.  With every entry `-inf`, the row max is `-inf`, so
    // `(-inf) - (-inf) = NaN`; `sum` becomes `NaN`, `sum > 0.0` is false, and
    // the `NaN`s survive.  This is documented here so a future refactor knows
    // the contract.
    //
    // Crucially this is UNREACHABLE in production: the causal mask always keeps
    // at least one valid key (query `i` attends to keys `0..=past_len+i`, which
    // always includes key 0), so no attention row is ever fully masked.  The
    // `causal_masked_scores_are_exactly_zero_after_softmax` test asserts that
    // invariant (every masked row still has `valid >= 1` finite entries and
    // sums to 1).
    let cols = 3;
    let mut scores = vec![f32::NEG_INFINITY; cols];
    softmax_rows(&mut scores, 1, cols);
    assert!(
        scores.iter().all(|p| p.is_nan()),
        "an all-masked row is unsupported and yields NaN (never occurs in \
         production because the causal mask always leaves >=1 valid key)"
    );
}

// ── scaled_dot_product_cached: single (q_len == 1), causal ───────────────────

#[test]
fn cached_single_matches_reference() {
    let n_head = 3;
    let head_dim = 8;
    let kv_len = 5;
    let q_len = 1;
    let past_len = kv_len - q_len; // 4

    let q_hm = make_head_major(1, n_head, q_len, head_dim);
    let k_hm = make_head_major(2, n_head, kv_len, head_dim);
    let v_hm = make_head_major(3, n_head, kv_len, head_dim);

    let cache = build_cache(KvCacheDtype::F32, n_head, head_dim, kv_len, &k_hm, &v_hm);
    let cfg = CachedSdpaConfig {
        n_head,
        q_len,
        kv_len,
        head_dim,
        past_len,
        causal_mask: true,
    };
    let mut scratch = SdpaScratch::new();
    let got = scaled_dot_product_cached(&q_hm, &cache, &cfg, &mut scratch);

    let want = naive_sdpa(
        &q_hm, &k_hm, &v_hm, n_head, q_len, kv_len, head_dim, past_len, true,
    );
    assert_close(&got, &want.out, F32_ABS_TOL, "cached_single");
}

// ── scaled_dot_product_cached: prefill (q_len > 1), causal ───────────────────

#[test]
fn cached_prefill_no_past_matches_reference() {
    let n_head = 2;
    let head_dim = 8;
    let q_len = 4;
    let kv_len = 4;
    let past_len = 0;

    let q_hm = make_head_major(11, n_head, q_len, head_dim);
    let k_hm = make_head_major(12, n_head, kv_len, head_dim);
    let v_hm = make_head_major(13, n_head, kv_len, head_dim);

    let cache = build_cache(KvCacheDtype::F32, n_head, head_dim, kv_len, &k_hm, &v_hm);
    let cfg = CachedSdpaConfig {
        n_head,
        q_len,
        kv_len,
        head_dim,
        past_len,
        causal_mask: true,
    };
    let mut scratch = SdpaScratch::new();
    let got = scaled_dot_product_cached(&q_hm, &cache, &cfg, &mut scratch);

    let want = naive_sdpa(
        &q_hm, &k_hm, &v_hm, n_head, q_len, kv_len, head_dim, past_len, true,
    );
    assert_close(&got, &want.out, F32_ABS_TOL, "cached_prefill_no_past");
}

#[test]
fn cached_prefill_with_past_matches_reference() {
    let n_head = 2;
    let head_dim = 8;
    let q_len = 3;
    let past_len = 4;
    let kv_len = past_len + q_len; // 7

    let q_hm = make_head_major(21, n_head, q_len, head_dim);
    let k_hm = make_head_major(22, n_head, kv_len, head_dim);
    let v_hm = make_head_major(23, n_head, kv_len, head_dim);

    let cache = build_cache(KvCacheDtype::F32, n_head, head_dim, kv_len, &k_hm, &v_hm);
    let cfg = CachedSdpaConfig {
        n_head,
        q_len,
        kv_len,
        head_dim,
        past_len,
        causal_mask: true,
    };
    let mut scratch = SdpaScratch::new();
    let got = scaled_dot_product_cached(&q_hm, &cache, &cfg, &mut scratch);

    let want = naive_sdpa(
        &q_hm, &k_hm, &v_hm, n_head, q_len, kv_len, head_dim, past_len, true,
    );
    assert_close(&got, &want.out, F32_ABS_TOL, "cached_prefill_with_past");
}

// ── scaled_dot_product_cached: full (no causal mask) ─────────────────────────

#[test]
fn cached_full_no_mask_matches_reference() {
    let n_head = 3;
    let head_dim = 8;
    let q_len = 3;
    let kv_len = 5;
    let past_len = 0; // irrelevant when causal_mask == false

    let q_hm = make_head_major(31, n_head, q_len, head_dim);
    let k_hm = make_head_major(32, n_head, kv_len, head_dim);
    let v_hm = make_head_major(33, n_head, kv_len, head_dim);

    let cache = build_cache(KvCacheDtype::F32, n_head, head_dim, kv_len, &k_hm, &v_hm);
    let cfg = CachedSdpaConfig {
        n_head,
        q_len,
        kv_len,
        head_dim,
        past_len,
        causal_mask: false,
    };
    let mut scratch = SdpaScratch::new();
    let got = scaled_dot_product_cached(&q_hm, &cache, &cfg, &mut scratch);

    let want = naive_sdpa(
        &q_hm, &k_hm, &v_hm, n_head, q_len, kv_len, head_dim, past_len, false,
    );
    assert_close(&got, &want.out, F32_ABS_TOL, "cached_full");
}

// ── scaled_dot_product_flat (cross-attention, no mask) + capture ─────────────

#[test]
fn flat_matches_reference_and_capture_is_head_mean() {
    let n_head = 3;
    let head_dim = 8;
    let q_len = 4;
    let kv_len = 6; // cross-attention: kv is the encoder length

    let q_hm = make_head_major(41, n_head, q_len, head_dim);
    let k_hm = make_head_major(42, n_head, kv_len, head_dim);
    let v_hm = make_head_major(43, n_head, kv_len, head_dim);

    let want = naive_sdpa(
        &q_hm, &k_hm, &v_hm, n_head, q_len, kv_len, head_dim, 0, false,
    );

    // With capture.
    let mut captured = vec![0.0f32; q_len * kv_len];
    let got_with = scaled_dot_product_flat(
        &q_hm,
        &k_hm,
        &v_hm,
        n_head,
        q_len,
        kv_len,
        head_dim,
        Some(&mut captured),
    );
    assert_close(&got_with, &want.out, F32_ABS_TOL, "flat output (capture)");
    assert_close(
        &captured,
        &want.attn_mean,
        F32_ABS_TOL,
        "flat capture head-mean attention",
    );
    // The captured attention rows are probability distributions (mean of
    // per-head softmaxes), each summing to 1.
    for i in 0..q_len {
        let row_sum: f32 = captured[i * kv_len..(i + 1) * kv_len].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "capture row {i} must sum to 1, got {row_sum}"
        );
    }

    // Without capture: the output must still equal the `capture: Some(...)`
    // run (capture is documented to run after the compute and touch only the
    // sink, not the score/output arithmetic).
    //
    // This is intentionally a TOLERANCE comparison, not `assert_eq!`. The two
    // calls take different code shapes (`capture: None` vs `capture:
    // Some(&mut captured)`), and under a non-FMA-contracting kernel (notably
    // Miri's portable `matrixmultiply` fallback, which is not gated behind
    // `is_x86_feature_detected!` the same way native AVX2/FMA is) the
    // scores@V accumulation can be reordered between the two call shapes.
    // Floating-point addition is not associative, so bit-identical output is
    // not guaranteed across different summation orders even though both are
    // "correct" to within f32 rounding. Observed divergence under Miri's
    // portable kernel is in the 7th significant digit (e.g. -0.19171949 vs
    // -0.19171958), i.e. ~1e-7 relative — consistent with a single
    // reassociated f32 addition, not a real bug. `F32_ABS_TOL` (1e-5) is the
    // same tolerance already used throughout this file for f32-summation-order
    // differences (see the module doc comment), so this does not weaken the
    // check beyond what floating-point semantics require.
    //
    // (A DIFFERENT kind of check — bit-exact equality between two IDENTICAL
    // calls, e.g. run-to-run determinism — is a legitimate and stronger
    // property; see `tests/sdpa_sgemm_parity.rs` and
    // `tests/threading_parity.rs`, which assert exactly that and are correct
    // to do so, since there both call shapes are identical and only
    // scheduling/threading can vary.)
    let got_none =
        scaled_dot_product_flat(&q_hm, &k_hm, &v_hm, n_head, q_len, kv_len, head_dim, None);
    assert_close(
        &got_with,
        &got_none,
        F32_ABS_TOL,
        "capture=None must not change the output vs capture=Some (within f32 summation-order tolerance)",
    );
}

// ── Causal-mask exactness (highest-value test) ───────────────────────────────

/// Sentinel technique: a "future" V row holds a huge value.  If the causal
/// mask leaks even one future key, that query's output explodes far beyond the
/// legitimate reference — the reference comparison (and the magnitude bound)
/// fire.  We drive this through the real `sdpa_cached_prefill` path.
fn run_causal_sentinel(past_len: usize) {
    let n_head = 2;
    let head_dim = 4;
    let q_len = 4;
    let kv_len = past_len + q_len;
    const SENTINEL: f32 = 1000.0;

    // K holds ordinary small values so that, if a future key were NOT masked,
    // it would receive genuine softmax weight (the logits are comparable).
    let q_hm = make_head_major(51, n_head, q_len, head_dim);
    let k_hm = make_head_major(52, n_head, kv_len, head_dim);
    // V: ordinary small values, except the LAST key (index kv_len-1) is a huge
    // sentinel for every head/dim.  The last key is "future" for every query
    // except the final one (whose global position is exactly kv_len-1).
    let mut v_hm = make_head_major(53, n_head, kv_len, head_dim);
    let last = kv_len - 1;
    for h in 0..n_head {
        for d in 0..head_dim {
            v_hm[h * kv_len * head_dim + last * head_dim + d] = SENTINEL;
        }
    }

    let cache = build_cache(KvCacheDtype::F32, n_head, head_dim, kv_len, &k_hm, &v_hm);
    let cfg = CachedSdpaConfig {
        n_head,
        q_len,
        kv_len,
        head_dim,
        past_len,
        causal_mask: true,
    };
    let mut scratch = SdpaScratch::new();
    let got = scaled_dot_product_cached(&q_hm, &cache, &cfg, &mut scratch);

    let want = naive_sdpa(
        &q_hm, &k_hm, &v_hm, n_head, q_len, kv_len, head_dim, past_len, true,
    );

    // Reference comparison detects an off-by-one in EITHER direction (a leak
    // adds sentinel mass; an over-restriction drops a legitimate diagonal
    // term).  Relative tolerance accommodates the legitimately-large final row
    // (which does attend to the sentinel).
    assert_close_tol(
        &got,
        &want.out,
        1e-3,
        1e-4,
        &format!("causal sentinel (past_len={past_len}) vs reference"),
    );

    // Direct magnitude guard: every query strictly before the last must NOT see
    // the sentinel, so its output stays O(1).  A leaked future key would inject
    // ~SENTINEL/kv_len ≈ hundreds — impossible to sneak past this bound.
    let n_state = n_head * head_dim;
    for i in 0..(q_len - 1) {
        for d in 0..n_state {
            let val = got[i * n_state + d].abs();
            assert!(
                val < 10.0,
                "causal leak (past_len={past_len}): query {i} elem {d} = {val} \
                 exceeds bound 10 — a future sentinel key was not masked"
            );
        }
    }
}

#[test]
fn causal_mask_exact_no_past() {
    run_causal_sentinel(0);
}

#[test]
fn causal_mask_exact_with_past() {
    run_causal_sentinel(5);
}

#[test]
fn causal_masked_scores_are_exactly_zero_after_softmax() {
    // Independently reproduce the production mask formula
    // (sdpa.rs: `valid = past_len + i + 1`) on a fresh score matrix, run the
    // real `softmax_rows`, and assert masked positions are EXACTLY 0.0 and each
    // row still sums to 1.
    let q_len = 3;
    let kv_len = 6;
    let past_len = 2;
    let mut scores = vec![0.0f32; q_len * kv_len];
    for i in 0..q_len {
        for j in 0..kv_len {
            scores[i * kv_len + j] = gen_val(61, 0, i, j);
        }
    }
    for i in 0..q_len {
        let valid = past_len + i + 1;
        for j in valid..kv_len {
            scores[i * kv_len + j] = f32::NEG_INFINITY;
        }
    }
    softmax_rows(&mut scores, q_len, kv_len);

    for i in 0..q_len {
        let valid = past_len + i + 1;
        for j in valid..kv_len {
            assert_eq!(
                scores[i * kv_len + j],
                0.0,
                "masked score at ({i},{j}) must be exactly 0 after softmax"
            );
        }
        let row_sum: f32 = scores[i * kv_len..(i + 1) * kv_len].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-6,
            "row {i} must still sum to 1, got {row_sum}"
        );
        // Positions 0..valid must be strictly positive (they attend to the
        // present + past), confirming the mask did not zero a legitimate key.
        for j in 0..valid {
            assert!(
                scores[i * kv_len + j] > 0.0,
                "legitimate key ({i},{j}) was wrongly masked"
            );
        }
    }
}

// ── KvCacheDtype coverage: F32 / VHalf / KvHalf ──────────────────────────────

/// Run the no-mask cached path for a given dtype and compare to the reference
/// built from the ORIGINAL f32 inputs, exercising `materialize_k_head` /
/// `materialize_v_head` (f16 dequant) and, for `KvHalf`, the `k_prescaled`
/// (`alpha = 1.0`) code path.
fn run_dtype_full(dtype: KvCacheDtype, tol: f32) {
    let n_head = 2;
    let head_dim = 8;
    let q_len = 2;
    let kv_len = 4;

    let q_hm = make_head_major(71, n_head, q_len, head_dim);
    let k_hm = make_head_major(72, n_head, kv_len, head_dim);
    let v_hm = make_head_major(73, n_head, kv_len, head_dim);

    let cache = build_cache(dtype, n_head, head_dim, kv_len, &k_hm, &v_hm);
    // Sanity: the cache actually stores what the dtype promises.
    match dtype {
        KvCacheDtype::F32 => {
            assert!(!cache.k_prescaled);
            assert!(!cache.k_is_f16());
            assert!(!cache.v_is_f16());
        }
        KvCacheDtype::VHalf => {
            assert!(!cache.k_prescaled);
            assert!(!cache.k_is_f16());
            assert!(cache.v_is_f16());
        }
        KvCacheDtype::KvHalf => {
            assert!(cache.k_prescaled, "KvHalf must pre-scale K");
            assert!(cache.k_is_f16());
            assert!(cache.v_is_f16());
        }
    }

    let cfg = CachedSdpaConfig {
        n_head,
        q_len,
        kv_len,
        head_dim,
        past_len: 0,
        causal_mask: false,
    };
    let mut scratch = SdpaScratch::new();
    let got = scaled_dot_product_cached(&q_hm, &cache, &cfg, &mut scratch);

    let want = naive_sdpa(
        &q_hm, &k_hm, &v_hm, n_head, q_len, kv_len, head_dim, 0, false,
    );
    assert_close(&got, &want.out, tol, "dtype full");
}

#[test]
fn dtype_f32_full_matches_reference() {
    run_dtype_full(KvCacheDtype::F32, F32_ABS_TOL);
}

#[test]
fn dtype_vhalf_full_matches_reference() {
    run_dtype_full(KvCacheDtype::VHalf, VHALF_ABS_TOL);
}

#[test]
fn dtype_kvhalf_full_matches_reference() {
    run_dtype_full(KvCacheDtype::KvHalf, KVHALF_ABS_TOL);
}

#[test]
fn dtype_kvhalf_causal_prefill_matches_reference() {
    // Exercise the pre-scaled K path TOGETHER with the causal mask and a
    // non-zero past, so `k_prescaled` (alpha=1.0) and the prefill masking are
    // covered simultaneously.
    let n_head = 2;
    let head_dim = 8;
    let q_len = 3;
    let past_len = 3;
    let kv_len = past_len + q_len; // 6

    let q_hm = make_head_major(81, n_head, q_len, head_dim);
    let k_hm = make_head_major(82, n_head, kv_len, head_dim);
    let v_hm = make_head_major(83, n_head, kv_len, head_dim);

    let cache = build_cache(KvCacheDtype::KvHalf, n_head, head_dim, kv_len, &k_hm, &v_hm);
    assert!(cache.k_prescaled);
    let cfg = CachedSdpaConfig {
        n_head,
        q_len,
        kv_len,
        head_dim,
        past_len,
        causal_mask: true,
    };
    let mut scratch = SdpaScratch::new();
    let got = scaled_dot_product_cached(&q_hm, &cache, &cfg, &mut scratch);

    let want = naive_sdpa(
        &q_hm, &k_hm, &v_hm, n_head, q_len, kv_len, head_dim, past_len, true,
    );
    assert_close(&got, &want.out, KVHALF_ABS_TOL, "kvhalf causal prefill");
}
