// SPDX-License-Identifier: Apache-2.0
// Copyright (c) COOLJAPAN OU (Team Kitasan)

//! Speaker clustering + speaker-count estimation.
//!
//! This module groups per-window speaker embeddings into speaker clusters and,
//! when the caller does not fix the count, estimates the number of speakers.
//! Two strategies are provided:
//!
//! * **AHC** — average-linkage agglomerative hierarchical clustering on cosine
//!   distance (`1 - cos`). Implemented inline as a real Lance-Williams update
//!   with deterministic tie-breaking (smallest index pair wins). The
//!   `ClusteringMethod::Ahc { threshold }` knob is a cosine-**distance**
//!   stopping threshold (matching this `1 - cos` metric), so merging halts once
//!   the closest remaining pair is farther than `threshold`. Its default `0.5`
//!   corresponds to merging clusters whose cosine similarity is at least `0.5`
//!   (`1 - cos <= 0.5 <=> cos >= 0.5`).
//! * **Spectral** — clustering on the symmetric normalized graph Laplacian
//!   `L_sym = I - D^{-1/2} A D^{-1/2}` built from the cosine affinity matrix.
//!   The eigenpairs are obtained with an inline cyclic Jacobi eigenvalue
//!   solver for real symmetric matrices; the speaker count (when not fixed) is
//!   estimated with the eigengap heuristic, and the row-normalised leading
//!   eigenvectors are clustered with a deterministic k-means++ (fixed,
//!   data-derived seed — no `rand`).
//!
//! Embeddings are assumed **L2-normalised**, so the cosine similarity between
//! two embeddings equals their dot product.

use crate::diarize::ClusteringMethod;
use crate::types::OxiWhisperError;
use std::cmp::Ordering;
use std::collections::HashMap;

/// Cluster per-window speaker embeddings into speaker groups.
///
/// * `embeddings` — one L2-normalised embedding vector per analysis window.
/// * `method` — [`ClusteringMethod::Ahc`] or [`ClusteringMethod::Spectral`].
/// * `num_speakers` — exact speaker count if known; otherwise it is estimated
///   inside `[min_speakers, max_speakers]`.
/// * `min_speakers` / `max_speakers` — inclusive bounds on the speaker count.
///
/// Returns `(labels, k)` where `labels[i]` is the cluster index in `0..k` of
/// window `i`, and `k` is the number of clusters discovered. Empty input maps
/// to `(vec![], 0)` and a single embedding to `(vec![0], 1)`.
pub fn cluster_speakers(
    embeddings: &[Vec<f32>],
    method: &ClusteringMethod,
    num_speakers: Option<usize>,
    min_speakers: usize,
    max_speakers: usize,
) -> Result<(Vec<usize>, usize), OxiWhisperError> {
    let n = embeddings.len();
    if n == 0 {
        return Ok((Vec::new(), 0));
    }
    if n == 1 {
        return Ok((vec![0], 1));
    }

    // Validate embedding geometry: a non-zero, uniform dimension is required so
    // that dot-product cosine similarity is well defined across all pairs.
    let dim = embeddings[0].len();
    if dim == 0 {
        return Err(OxiWhisperError::ConfigError(
            "cluster_speakers: embeddings must have a non-zero dimension".into(),
        ));
    }
    if embeddings.iter().any(|e| e.len() != dim) {
        return Err(OxiWhisperError::ConfigError(
            "cluster_speakers: all embeddings must share the same dimension".into(),
        ));
    }

    // Effective speaker-count bounds: at least one cluster, never more clusters
    // than there are embeddings, and `hi >= lo`.
    let lo = min_speakers.max(1).min(n);
    let hi = max_speakers.max(1).max(lo).min(n);

    let assignment = match method {
        ClusteringMethod::Ahc { threshold } => {
            ahc_cluster(embeddings, *threshold, num_speakers, lo, hi)
        }
        ClusteringMethod::Spectral => spectral_cluster(embeddings, num_speakers, lo, hi),
    };

    Ok(compact_labels(&assignment))
}

/// Cosine similarity of two L2-normalised embeddings (their dot product),
/// accumulated in `f64` for numerical stability.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum()
}

/// Squared Euclidean distance between two real vectors.
fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Compact an arbitrary cluster-representative labelling into dense labels
/// `0..k` in order of first appearance, returning `(labels, k)`.
fn compact_labels(assignment: &[usize]) -> (Vec<usize>, usize) {
    let mut remap: HashMap<usize, usize> = HashMap::new();
    let mut next = 0usize;
    let mut labels = Vec::with_capacity(assignment.len());
    for &rep in assignment {
        match remap.get(&rep) {
            Some(&label) => labels.push(label),
            None => {
                remap.insert(rep, next);
                labels.push(next);
                next += 1;
            }
        }
    }
    (labels, next)
}

// ---------------------------------------------------------------------------
// Agglomerative hierarchical clustering (average linkage, cosine distance)
// ---------------------------------------------------------------------------

/// Average-linkage AHC on cosine distance (`1 - cos`).
///
/// With a fixed `num_speakers` the closest clusters are merged until exactly
/// that many remain (the target being clamped into `[lo, hi]`). Otherwise the
/// count is first forced down to `hi`, then the closest pair is merged while
/// its average-linkage distance is `<= threshold`, never dropping below `lo`.
///
/// Returns, for every input point, the representative index of its final
/// cluster (not yet compacted).
fn ahc_cluster(
    embeddings: &[Vec<f32>],
    threshold: f32,
    num_speakers: Option<usize>,
    lo: usize,
    hi: usize,
) -> Vec<usize> {
    let n = embeddings.len();
    let threshold = f64::from(threshold);

    // Pairwise cosine-distance matrix.
    let mut dist = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = 1.0 - cosine_similarity(&embeddings[i], &embeddings[j]);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }

    let mut active = vec![true; n];
    let mut size = vec![1usize; n];
    let mut assignment: Vec<usize> = (0..n).collect();
    let mut count = n;

    let target = num_speakers.map(|t| t.max(1).clamp(lo, hi));

    loop {
        // Stopping conditions.
        match target {
            Some(t) => {
                if count <= t {
                    break;
                }
            }
            None => {
                if count <= lo {
                    break;
                }
            }
        }

        let (bi, bj, dmin) = closest_pair(&dist, &active, n);

        // In threshold mode, once we are at or below the upper bound, honour the
        // distance threshold; above the upper bound we force merges regardless.
        if target.is_none() && count <= hi && dmin > threshold {
            break;
        }

        merge_clusters(bi, bj, &mut dist, &mut active, &mut size, &mut assignment);
        count -= 1;
    }

    assignment
}

/// Find the closest pair of active clusters. Ties are broken deterministically
/// towards the lexicographically smallest `(i, j)` via a strict `<` comparison.
fn closest_pair(dist: &[Vec<f64>], active: &[bool], n: usize) -> (usize, usize, f64) {
    let mut best = (0usize, 0usize, f64::INFINITY);
    for i in 0..n {
        if !active[i] {
            continue;
        }
        for j in (i + 1)..n {
            if !active[j] {
                continue;
            }
            if dist[i][j] < best.2 {
                best = (i, j, dist[i][j]);
            }
        }
    }
    best
}

/// Merge cluster `j` into cluster `i` (`i < j`) using the Lance-Williams update
/// for average linkage (UPGMA): `d(ij, c) = (|i| d(i,c) + |j| d(j,c)) / |ij|`.
fn merge_clusters(
    i: usize,
    j: usize,
    dist: &mut [Vec<f64>],
    active: &mut [bool],
    size: &mut [usize],
    assignment: &mut [usize],
) {
    let si = size[i] as f64;
    let sj = size[j] as f64;
    let denom = si + sj;
    let n = active.len();
    for c in 0..n {
        if c == i || c == j || !active[c] {
            continue;
        }
        let new_d = (si * dist[i][c] + sj * dist[j][c]) / denom;
        dist[i][c] = new_d;
        dist[c][i] = new_d;
    }
    size[i] += size[j];
    active[j] = false;
    for slot in assignment.iter_mut() {
        if *slot == j {
            *slot = i;
        }
    }
}

// ---------------------------------------------------------------------------
// Spectral clustering
// ---------------------------------------------------------------------------

/// Spectral clustering on the symmetric normalised Laplacian.
///
/// Builds the cosine affinity matrix (negatives clamped to zero, diagonal
/// zeroed), forms `L_sym = I - D^{-1/2} A D^{-1/2}`, computes its eigenpairs
/// with the inline Jacobi solver, estimates `k` with the eigengap heuristic
/// (unless fixed), row-normalises the `k` leading eigenvectors, and clusters
/// the rows with deterministic k-means.
///
/// Returns a per-point cluster labelling (not yet compacted).
fn spectral_cluster(
    embeddings: &[Vec<f32>],
    num_speakers: Option<usize>,
    lo: usize,
    hi: usize,
) -> Vec<usize> {
    let n = embeddings.len();

    // Cosine affinity, non-negative, zero diagonal.
    let mut affinity = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let s = cosine_similarity(&embeddings[i], &embeddings[j]).max(0.0);
            affinity[i][j] = s;
            affinity[j][i] = s;
        }
    }

    // Inverse square-root degrees; isolated nodes contribute nothing.
    let mut d_inv_sqrt = vec![0.0f64; n];
    for i in 0..n {
        let degree: f64 = affinity[i].iter().sum();
        d_inv_sqrt[i] = if degree > 0.0 {
            1.0 / degree.sqrt()
        } else {
            0.0
        };
    }

    // Symmetric normalised Laplacian.
    let mut laplacian = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            let normalised = d_inv_sqrt[i] * affinity[i][j] * d_inv_sqrt[j];
            laplacian[i][j] = if i == j {
                1.0 - normalised
            } else {
                -normalised
            };
        }
    }

    let (eigenvalues, eigenvectors) = jacobi_eigen(laplacian);

    // Ascending eigenvalue order, ties broken by original index for stability.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| {
        eigenvalues[x]
            .partial_cmp(&eigenvalues[y])
            .unwrap_or(Ordering::Equal)
            .then(x.cmp(&y))
    });
    let sorted_values: Vec<f64> = order.iter().map(|&idx| eigenvalues[idx]).collect();

    let k = match num_speakers {
        Some(t) => t.max(1).clamp(lo, hi),
        None => estimate_k_eigengap(&sorted_values, lo, hi),
    };

    if k <= 1 {
        return vec![0; n];
    }

    // Row-normalised matrix of the `k` leading eigenvectors.
    let mut spectral_rows = vec![vec![0.0f64; k]; n];
    for col in 0..k {
        let source = order[col];
        for (row, dst) in spectral_rows.iter_mut().enumerate() {
            dst[col] = eigenvectors[row][source];
        }
    }
    for row in spectral_rows.iter_mut() {
        let norm = row.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for value in row.iter_mut() {
                *value /= norm;
            }
        }
    }

    kmeans(&spectral_rows, k)
}

/// Estimate the number of clusters via the eigengap heuristic: pick the split
/// `k` in `[lo, hi]` that maximises `lambda[k] - lambda[k-1]` on the ascending
/// eigenvalues. Ties are broken towards the smaller `k`.
fn estimate_k_eigengap(sorted_values: &[f64], lo: usize, hi: usize) -> usize {
    if lo >= hi {
        return lo;
    }
    let n = sorted_values.len();
    let mut best_k = lo;
    let mut best_gap = f64::NEG_INFINITY;
    for k in lo..=hi {
        // A split at `k` requires a `(k+1)`-th eigenvalue to gap against.
        let gap = if k < n {
            sorted_values[k] - sorted_values[k - 1]
        } else {
            f64::NEG_INFINITY
        };
        if gap > best_gap {
            best_gap = gap;
            best_k = k;
        }
    }
    best_k
}

// ---------------------------------------------------------------------------
// Cyclic Jacobi eigenvalue solver for real symmetric matrices
// ---------------------------------------------------------------------------

/// Compute all eigenvalues and eigenvectors of a real symmetric matrix using
/// the cyclic Jacobi rotation method.
///
/// The input `matrix` is consumed and rotated in place towards a diagonal form.
/// Returns `(eigenvalues, eigenvectors)` where `eigenvalues[i]` corresponds to
/// column `i` of `eigenvectors` (i.e. `eigenvectors[row][i]` is the `row`-th
/// component of the `i`-th eigenvector). Eigenpairs are **not** sorted.
fn jacobi_eigen(mut matrix: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = matrix.len();
    let mut vectors = vec![vec![0.0f64; n]; n];
    for (i, row) in vectors.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    if n <= 1 {
        let eigenvalues = (0..n).map(|i| matrix[i][i]).collect();
        return (eigenvalues, vectors);
    }

    const MAX_SWEEPS: usize = 100;
    for _ in 0..MAX_SWEEPS {
        // Sum of squares of the strictly-upper off-diagonal entries.
        let mut off_diagonal = 0.0;
        for (p, row) in matrix.iter().enumerate() {
            for &value in row.iter().skip(p + 1) {
                off_diagonal += value * value;
            }
        }
        if off_diagonal <= 1e-30 {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let a_pq = matrix[p][q];
                if a_pq.abs() <= 1e-300 {
                    continue;
                }
                let a_pp = matrix[p][p];
                let a_qq = matrix[q][q];

                // Rotation angle that annihilates the (p, q) entry.
                let theta = (a_qq - a_pp) / (2.0 * a_pq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                // Rotate the remaining rows/columns. Only columns/rows `p` and
                // `q` change; snapshot them, compute the rotated columns, and
                // write both the columns and their symmetric row counterparts.
                let col_p: Vec<f64> = matrix.iter().map(|row| row[p]).collect();
                let col_q: Vec<f64> = matrix.iter().map(|row| row[q]).collect();
                let new_col_p: Vec<f64> = col_p
                    .iter()
                    .zip(col_q.iter())
                    .map(|(&a, &b)| c * a - s * b)
                    .collect();
                let new_col_q: Vec<f64> = col_p
                    .iter()
                    .zip(col_q.iter())
                    .map(|(&a, &b)| s * a + c * b)
                    .collect();
                for (i, row) in matrix.iter_mut().enumerate() {
                    if i != p && i != q {
                        row[p] = new_col_p[i];
                        row[q] = new_col_q[i];
                    }
                }
                for (i, slot) in matrix[p].iter_mut().enumerate() {
                    if i != p && i != q {
                        *slot = new_col_p[i];
                    }
                }
                for (i, slot) in matrix[q].iter_mut().enumerate() {
                    if i != p && i != q {
                        *slot = new_col_q[i];
                    }
                }

                // Update the pivot 2x2 block; the off-diagonal is annihilated.
                matrix[p][p] = c * c * a_pp - 2.0 * s * c * a_pq + s * s * a_qq;
                matrix[q][q] = s * s * a_pp + 2.0 * s * c * a_pq + c * c * a_qq;
                matrix[p][q] = 0.0;
                matrix[q][p] = 0.0;

                // Accumulate the rotation into the eigenvector matrix.
                for row in vectors.iter_mut() {
                    let v_ip = row[p];
                    let v_iq = row[q];
                    row[p] = c * v_ip - s * v_iq;
                    row[q] = s * v_ip + c * v_iq;
                }
            }
        }
    }

    let eigenvalues = (0..n).map(|i| matrix[i][i]).collect();
    (eigenvalues, vectors)
}

// ---------------------------------------------------------------------------
// Deterministic k-means (k-means++ init, data-derived fixed seed)
// ---------------------------------------------------------------------------

/// One-shot 64-bit avalanche mix (SplitMix64 finaliser), used to fold data into
/// a seed and to advance the deterministic PRNG stream.
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Advance a SplitMix64 state and return the next 64-bit value.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    mix64(*state)
}

/// Uniform `f64` in `[0, 1)` drawn from the deterministic stream.
fn next_unit_f64(state: &mut u64) -> f64 {
    let bits = splitmix64(state) >> 11; // top 53 bits
    (bits as f64) / ((1u64 << 53) as f64)
}

/// Point (by index) whose nearest current center is farthest away; ties break
/// towards the smallest index (strict `>`). Used for seeding fallbacks and for
/// empty-cluster reinitialisation.
fn farthest_point(points: &[Vec<f64>], centers: &[Vec<f64>]) -> usize {
    let mut best_index = 0usize;
    let mut best_distance = f64::NEG_INFINITY;
    for (p, point) in points.iter().enumerate() {
        let mut nearest = f64::INFINITY;
        for center in centers {
            let d = squared_distance(point, center);
            if d < nearest {
                nearest = d;
            }
        }
        if nearest > best_distance {
            best_distance = nearest;
            best_index = p;
        }
    }
    best_index
}

/// Deterministic k-means clustering of `points` into `k` groups.
///
/// Initialisation is k-means++ driven by a SplitMix64 stream whose seed is
/// folded from the point data (fully reproducible, no `rand`). Lloyd iterations
/// then refine the assignment; empty clusters are reseeded to the farthest
/// point. Assignment ties break towards the smallest center index.
fn kmeans(points: &[Vec<f64>], k: usize) -> Vec<usize> {
    let n = points.len();
    if k <= 1 {
        return vec![0; n];
    }
    let dim = points[0].len();

    // Fold the point data into a reproducible seed.
    let mut seed = 0x243F_6A88_85A3_08D3u64;
    for point in points {
        for &value in point {
            seed = seed.rotate_left(7) ^ mix64(value.to_bits());
        }
    }
    let mut state = seed;

    // k-means++ initialisation.
    let mut centers: Vec<Vec<f64>> = Vec::with_capacity(k);
    let first = (splitmix64(&mut state) % (n as u64)) as usize;
    centers.push(points[first].clone());

    let mut squared = vec![0.0f64; n];
    while centers.len() < k {
        let mut total = 0.0;
        for (p, point) in points.iter().enumerate() {
            let mut nearest = f64::INFINITY;
            for center in &centers {
                let d = squared_distance(point, center);
                if d < nearest {
                    nearest = d;
                }
            }
            squared[p] = nearest;
            total += nearest;
        }

        let chosen = if total <= 1e-18 {
            // All points coincide with an existing center; fall back to a
            // deterministic farthest-point pick.
            farthest_point(points, &centers)
        } else {
            let threshold = next_unit_f64(&mut state) * total;
            let mut accumulated = 0.0;
            let mut index = n - 1;
            for (p, &d) in squared.iter().enumerate() {
                accumulated += d;
                if accumulated >= threshold {
                    index = p;
                    break;
                }
            }
            index
        };
        centers.push(points[chosen].clone());
    }

    // Lloyd iterations.
    let mut labels = vec![0usize; n];
    const MAX_ITERS: usize = 300;
    for _ in 0..MAX_ITERS {
        let mut changed = false;

        for (p, point) in points.iter().enumerate() {
            let mut best_label = 0usize;
            let mut best_distance = f64::INFINITY;
            for (ci, center) in centers.iter().enumerate() {
                let d = squared_distance(point, center);
                if d < best_distance {
                    best_distance = d;
                    best_label = ci;
                }
            }
            if labels[p] != best_label {
                labels[p] = best_label;
                changed = true;
            }
        }

        let mut sums = vec![vec![0.0f64; dim]; k];
        let mut counts = vec![0usize; k];
        for (p, point) in points.iter().enumerate() {
            let label = labels[p];
            counts[label] += 1;
            for (acc, &value) in sums[label].iter_mut().zip(point.iter()) {
                *acc += value;
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f64;
                for (center_value, &sum) in centers[c].iter_mut().zip(sums[c].iter()) {
                    *center_value = sum * inv;
                }
            } else {
                // Reseed an empty cluster to the farthest point.
                let far = farthest_point(points, &centers);
                centers[c] = points[far].clone();
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIM: usize = 12;

    /// L2-normalise a vector in place.
    fn l2_normalize(v: &mut [f32]) {
        let norm = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }

    /// Build `num_groups` well-separated clusters of L2-normalised vectors.
    ///
    /// Each group points along a distinct coordinate axis (`0..num_groups`)
    /// with small, deterministic jitter on two shared high axes. Within-group
    /// cosine similarity is ~0.99; cross-group is ~0. Returns the embeddings
    /// and the ground-truth group index of each.
    fn planted_groups(num_groups: usize, per_group: usize) -> (Vec<Vec<f32>>, Vec<usize>) {
        assert!((1..=DIM - 2).contains(&num_groups));
        let jitter_a = DIM - 1;
        let jitter_b = DIM - 2;
        let mut embeddings = Vec::new();
        let mut truth = Vec::new();
        for g in 0..num_groups {
            for i in 0..per_group {
                let mut v = vec![0.0f32; DIM];
                v[g] = 1.0;
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                v[jitter_a] += 0.03 * sign * (i as f32 + 1.0);
                v[jitter_b] += 0.02 * sign;
                l2_normalize(&mut v);
                embeddings.push(v);
                truth.push(g);
            }
        }
        (embeddings, truth)
    }

    /// True iff `pred` and `truth` induce the same partition (equal up to a
    /// relabelling), i.e. there is a bijection between predicted labels and
    /// ground-truth group indices.
    fn same_partition(pred: &[usize], truth: &[usize]) -> bool {
        if pred.len() != truth.len() {
            return false;
        }
        let mut truth_to_pred: HashMap<usize, usize> = HashMap::new();
        let mut pred_to_truth: HashMap<usize, usize> = HashMap::new();
        for (&t, &p) in truth.iter().zip(pred.iter()) {
            if let Some(&existing) = truth_to_pred.get(&t) {
                if existing != p {
                    return false;
                }
            } else {
                truth_to_pred.insert(t, p);
            }
            if let Some(&existing) = pred_to_truth.get(&p) {
                if existing != t {
                    return false;
                }
            } else {
                pred_to_truth.insert(p, t);
            }
        }
        true
    }

    /// One standard-normal sample via Box-Muller, driven by the module's own
    /// inline SplitMix64 stream ([`next_unit_f64`]). This is the same
    /// deterministic-PRNG pattern the production `kmeans` seeding uses — no
    /// `rand`, no clock — so the generated blobs are byte-reproducible.
    fn next_gaussian(state: &mut u64) -> f64 {
        // Clamp the first uniform away from 0 so `ln` never sees 0.
        let u1 = next_unit_f64(state).max(1e-12);
        let u2 = next_unit_f64(state);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Generate `k` well-separated Gaussian blobs of `per_group` points each.
    ///
    /// Center `g` sits at `CENTER_SCALE · e_g` (a distinct coordinate axis per
    /// group, so the centers are mutually orthogonal and far apart in a
    /// `k + 2`-dimensional space). Every coordinate then receives independent
    /// `N(0, sigma²)` jitter from the inline Box-Muller stream, and each point is
    /// **L2-normalized** exactly as the clustering pipeline expects its input.
    ///
    /// With `CENTER_SCALE` large relative to `sigma`, within-group cosine
    /// similarity is ≈1 and cross-group ≈0, so the planted partition is exactly
    /// recoverable — the recovery assertions below are therefore real, not
    /// vacuous. Returns `(embeddings, truth)` with `truth[i]` the planted group.
    fn gaussian_blobs(
        k: usize,
        per_group: usize,
        sigma: f64,
        seed: u64,
    ) -> (Vec<Vec<f32>>, Vec<usize>) {
        const CENTER_SCALE: f64 = 8.0;
        let dim = k + 2; // two extra axes carry only noise, never a center
        let mut state = seed;
        let mut embeddings = Vec::with_capacity(k * per_group);
        let mut truth = Vec::with_capacity(k * per_group);
        for g in 0..k {
            for _ in 0..per_group {
                let mut v = vec![0.0f32; dim];
                for (axis, slot) in v.iter_mut().enumerate() {
                    let center = if axis == g { CENTER_SCALE } else { 0.0 };
                    *slot = (center + sigma * next_gaussian(&mut state)) as f32;
                }
                let norm = v
                    .iter()
                    .map(|&x| f64::from(x) * f64::from(x))
                    .sum::<f64>()
                    .sqrt();
                assert!(norm > 0.0, "blob point must have a positive norm");
                for x in v.iter_mut() {
                    *x = (f64::from(*x) / norm) as f32;
                }
                embeddings.push(v);
                truth.push(g);
            }
        }
        (embeddings, truth)
    }

    /// True iff `pred` and `truth` induce the same partition up to a label
    /// permutation, with the permutation resolved by the crate's **own**
    /// Hungarian solver ([`crate::diarize::metrics::hungarian`]) on the negated
    /// `k × k` confusion matrix. Both label vectors must already use dense labels
    /// `0..k`.
    ///
    /// The optimal one-to-one label map maximises agreement; the partitions are
    /// equal iff *every* point lands on that map (matched count == `n`). Any
    /// merge/split error leaves at least one point off the optimal map, so this
    /// returns `false` — the assertion cannot pass on broken clustering.
    fn same_partition_via_hungarian(pred: &[usize], truth: &[usize], k: usize) -> bool {
        assert_eq!(pred.len(), truth.len(), "label vectors must align");
        let mut confusion = vec![vec![0i64; k]; k];
        for (&p, &t) in pred.iter().zip(truth.iter()) {
            if p >= k || t >= k {
                return false; // a label outside 0..k cannot be a k-way partition
            }
            confusion[p][t] += 1;
        }
        // Hungarian minimises cost; feed -count to turn it into max-agreement.
        let cost: Vec<Vec<f64>> = confusion
            .iter()
            .map(|row| row.iter().map(|&c| -(c as f64)).collect())
            .collect();
        let assign = crate::diarize::metrics::hungarian(&cost);
        let mut matched = 0i64;
        for (p, &t) in assign.iter().enumerate() {
            if t < k {
                matched += confusion[p][t];
            }
        }
        matched as usize == pred.len()
    }

    #[test]
    fn test_empty_input_returns_zero_clusters() {
        let embeddings: Vec<Vec<f32>> = Vec::new();
        let (labels, k) = cluster_speakers(&embeddings, &ClusteringMethod::default(), None, 1, 10)
            .expect("empty input must succeed");
        assert!(labels.is_empty());
        assert_eq!(k, 0);
    }

    #[test]
    fn test_single_embedding_returns_one_cluster() {
        let embeddings = vec![vec![1.0f32, 0.0, 0.0]];
        let (labels, k) = cluster_speakers(&embeddings, &ClusteringMethod::default(), None, 1, 10)
            .expect("single embedding must succeed");
        assert_eq!(labels, vec![0]);
        assert_eq!(k, 1);
    }

    #[test]
    fn test_dimension_mismatch_is_config_error() {
        let embeddings = vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0, 0.0]];
        let result = cluster_speakers(&embeddings, &ClusteringMethod::default(), None, 1, 10);
        assert!(matches!(result, Err(OxiWhisperError::ConfigError(_))));
    }

    #[test]
    fn test_ahc_fixed_two_speakers_recovers_grouping() {
        let (embeddings, truth) = planted_groups(2, 4);
        let (labels, k) = cluster_speakers(
            &embeddings,
            &ClusteringMethod::Ahc { threshold: 0.5 },
            Some(2),
            1,
            10,
        )
        .expect("ahc must succeed");
        assert_eq!(k, 2);
        assert!(same_partition(&labels, &truth));
    }

    #[test]
    fn test_ahc_fixed_three_speakers_recovers_grouping() {
        let (embeddings, truth) = planted_groups(3, 4);
        let (labels, k) = cluster_speakers(
            &embeddings,
            &ClusteringMethod::Ahc { threshold: 0.5 },
            Some(3),
            1,
            10,
        )
        .expect("ahc must succeed");
        assert_eq!(k, 3);
        assert!(same_partition(&labels, &truth));
    }

    #[test]
    fn test_ahc_auto_estimates_two_speakers() {
        let (embeddings, truth) = planted_groups(2, 5);
        let (labels, k) = cluster_speakers(
            &embeddings,
            &ClusteringMethod::Ahc { threshold: 0.5 },
            None,
            1,
            10,
        )
        .expect("ahc must succeed");
        assert_eq!(k, 2, "eigen-free AHC must recover the planted count");
        assert!(same_partition(&labels, &truth));
    }

    #[test]
    fn test_ahc_auto_estimates_three_speakers() {
        let (embeddings, truth) = planted_groups(3, 5);
        let (labels, k) = cluster_speakers(
            &embeddings,
            &ClusteringMethod::Ahc { threshold: 0.5 },
            None,
            1,
            10,
        )
        .expect("ahc must succeed");
        assert_eq!(k, 3);
        assert!(same_partition(&labels, &truth));
    }

    #[test]
    fn test_ahc_min_speakers_clamp_prevents_over_merge() {
        // A large threshold would merge the two well-separated groups into a
        // single cluster; min_speakers=2 must keep them apart.
        let (embeddings, truth) = planted_groups(2, 4);
        let (labels, k) = cluster_speakers(
            &embeddings,
            &ClusteringMethod::Ahc { threshold: 1.9 },
            None,
            2,
            10,
        )
        .expect("ahc must succeed");
        assert_eq!(k, 2, "min_speakers must clamp k up from 1");
        assert!(same_partition(&labels, &truth));

        // Without the clamp (min_speakers=1) the same threshold collapses to 1.
        let (_, k_unclamped) = cluster_speakers(
            &embeddings,
            &ClusteringMethod::Ahc { threshold: 1.9 },
            None,
            1,
            10,
        )
        .expect("ahc must succeed");
        assert_eq!(
            k_unclamped, 1,
            "large threshold must otherwise merge to one"
        );
    }

    #[test]
    fn test_ahc_max_speakers_clamp_forces_merge() {
        // Four separable groups, but max_speakers=2 forces a merge down to two.
        let (embeddings, _) = planted_groups(4, 3);
        let (_, k) = cluster_speakers(
            &embeddings,
            &ClusteringMethod::Ahc { threshold: 0.5 },
            None,
            1,
            2,
        )
        .expect("ahc must succeed");
        assert_eq!(k, 2, "max_speakers must clamp k down");

        // With a permissive bound the same data yields all four groups.
        let (_, k_free) = cluster_speakers(
            &embeddings,
            &ClusteringMethod::Ahc { threshold: 0.5 },
            None,
            1,
            10,
        )
        .expect("ahc must succeed");
        assert_eq!(k_free, 4, "unclamped AHC must recover all four groups");
    }

    #[test]
    fn test_spectral_fixed_two_speakers_recovers_grouping() {
        let (embeddings, truth) = planted_groups(2, 4);
        let (labels, k) =
            cluster_speakers(&embeddings, &ClusteringMethod::Spectral, Some(2), 1, 10)
                .expect("spectral must succeed");
        assert_eq!(k, 2);
        assert!(same_partition(&labels, &truth));
    }

    #[test]
    fn test_spectral_fixed_three_speakers_recovers_grouping() {
        let (embeddings, truth) = planted_groups(3, 4);
        let (labels, k) =
            cluster_speakers(&embeddings, &ClusteringMethod::Spectral, Some(3), 1, 10)
                .expect("spectral must succeed");
        assert_eq!(k, 3);
        assert!(same_partition(&labels, &truth));
    }

    #[test]
    fn test_spectral_auto_eigengap_estimates_two_speakers() {
        let (embeddings, truth) = planted_groups(2, 5);
        let (labels, k) = cluster_speakers(&embeddings, &ClusteringMethod::Spectral, None, 1, 10)
            .expect("spectral must succeed");
        assert_eq!(k, 2, "eigengap must recover the planted count");
        assert!(same_partition(&labels, &truth));
    }

    #[test]
    fn test_spectral_auto_eigengap_estimates_three_speakers() {
        let (embeddings, truth) = planted_groups(3, 5);
        let (labels, k) = cluster_speakers(&embeddings, &ClusteringMethod::Spectral, None, 1, 10)
            .expect("spectral must succeed");
        assert_eq!(k, 3, "eigengap must recover the planted count");
        assert!(same_partition(&labels, &truth));
    }

    #[test]
    fn test_fixed_num_speakers_is_honored_over_structure() {
        // Three planted groups but the caller demands exactly two speakers.
        let (embeddings, _) = planted_groups(3, 4);
        for method in [
            ClusteringMethod::Ahc { threshold: 0.5 },
            ClusteringMethod::Spectral,
        ] {
            let (labels, k) = cluster_speakers(&embeddings, &method, Some(2), 1, 10)
                .expect("clustering must succeed");
            assert_eq!(k, 2, "fixed num_speakers must be honored for {method:?}");
            let distinct: std::collections::BTreeSet<usize> = labels.iter().copied().collect();
            assert_eq!(distinct.len(), 2, "exactly two labels must be used");
        }
    }

    // ── Separable Gaussian-blob recovery (SplitMix64 + Box-Muller) ──────────
    //
    // These exercise the clustering directly on embeddings where exact recovery
    // IS achievable, and resolve the arbitrary cluster-label permutation with
    // the crate's own Hungarian solver on a confusion matrix. Each assertion
    // fails outright if clustering (or the estimator) were broken.

    #[test]
    fn test_ahc_recovers_gaussian_blobs_via_hungarian() {
        // Three tight, orthogonally-centred blobs; AHC with the true count must
        // reproduce the planted partition exactly (up to relabelling).
        let (embeddings, truth) = gaussian_blobs(3, 6, 0.12, 0x1234_5678_9ABC_DEF0);
        let (labels, k) = cluster_speakers(
            &embeddings,
            &ClusteringMethod::Ahc { threshold: 0.5 },
            Some(3),
            1,
            10,
        )
        .expect("ahc must succeed");
        assert_eq!(k, 3, "AHC must return exactly the planted cluster count");
        assert!(
            same_partition_via_hungarian(&labels, &truth, 3),
            "AHC must recover the planted partition (Hungarian-resolved), got {labels:?}"
        );
    }

    #[test]
    fn test_spectral_recovers_gaussian_blobs_via_hungarian() {
        // The same blobs must be recovered by the spectral backend as well.
        let (embeddings, truth) = gaussian_blobs(3, 6, 0.12, 0x1234_5678_9ABC_DEF0);
        let (labels, k) =
            cluster_speakers(&embeddings, &ClusteringMethod::Spectral, Some(3), 1, 10)
                .expect("spectral must succeed");
        assert_eq!(
            k, 3,
            "spectral must return exactly the planted cluster count"
        );
        assert!(
            same_partition_via_hungarian(&labels, &truth, 3),
            "spectral must recover the planted partition (Hungarian-resolved), got {labels:?}"
        );
    }

    #[test]
    fn test_estimators_recover_planted_k_on_gaussian_blobs() {
        // num_speakers = None with min < k < max: both the eigen-free AHC
        // estimator and the spectral eigengap estimator must discover k = 4 and
        // reproduce the planted partition. The `1` and `8` passed as the
        // min/max bounds below strictly bracket the planted k = 4, so recovering
        // 4 is a genuine estimate, not a clamp to a boundary.
        let (embeddings, truth) = gaussian_blobs(4, 6, 0.10, 0x0F1E_2D3C_4B5A_6978);

        let (ahc_labels, ahc_k) = cluster_speakers(
            &embeddings,
            &ClusteringMethod::Ahc { threshold: 0.5 },
            None,
            1,
            8,
        )
        .expect("ahc estimator must succeed");
        assert_eq!(ahc_k, 4, "AHC estimator must recover the planted k = 4");
        assert!(
            same_partition_via_hungarian(&ahc_labels, &truth, 4),
            "AHC estimator partition must match the planting, got {ahc_labels:?}"
        );

        let (spec_labels, spec_k) =
            cluster_speakers(&embeddings, &ClusteringMethod::Spectral, None, 1, 8)
                .expect("spectral estimator must succeed");
        assert_eq!(
            spec_k, 4,
            "spectral eigengap estimator must recover the planted k = 4"
        );
        assert!(
            same_partition_via_hungarian(&spec_labels, &truth, 4),
            "spectral estimator partition must match the planting, got {spec_labels:?}"
        );
    }

    #[test]
    fn test_jacobi_eigen_matches_known_spectrum() {
        // Symmetric matrix with a known spectrum: diag(2,3) plus off-diagonal 1
        // has eigenvalues (5 +/- sqrt(5)) / 2.
        let matrix = vec![vec![2.0f64, 1.0], vec![1.0, 3.0]];
        let (mut values, _vectors) = jacobi_eigen(matrix);
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let expected_low = (5.0 - 5.0f64.sqrt()) / 2.0;
        let expected_high = (5.0 + 5.0f64.sqrt()) / 2.0;
        assert!((values[0] - expected_low).abs() < 1e-9);
        assert!((values[1] - expected_high).abs() < 1e-9);
    }
}
