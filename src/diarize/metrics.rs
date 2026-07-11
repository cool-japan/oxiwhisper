// SPDX-License-Identifier: Apache-2.0
// Copyright (c) COOLJAPAN OU (Team Kitasan)

//! Diarization scoring: NIST RTTM parsing, the Hungarian (Kuhn–Munkres)
//! optimal assignment, and the two standard "who spoke when" error metrics —
//! **DER** (Diarization Error Rate) and **JER** (Jaccard Error Rate).
//!
//! # Why these three pieces live together
//!
//! A diarizer emits *anonymous* speaker labels ([`SpeakerId`](crate::diarize::SpeakerId)):
//! label `0` in the hypothesis is not required to be the same real person as
//! label `0` in the reference. Both DER and JER therefore begin by finding the
//! **optimal one-to-one relabelling** of hypothesis speakers onto reference
//! speakers that maximises the total time the two agree. That relabelling is an
//! assignment problem, solved here in `O(n^3)` by [`hungarian`]. The reference
//! side is read from a NIST RTTM document via [`parse_rttm`]; the hypothesis
//! side is this crate's own [`SpeakerSegment`]
//! stream (which E2 can itself serialise back to RTTM via
//! [`rttm_string`](crate::diarize::format::rttm_string), so the two sides are
//! symmetric and round-trippable).
//!
//! # DER (md-eval formulation)
//!
//! The timeline is cut at the sorted union of every reference and hypothesis
//! boundary into *elementary intervals* over which the set of active reference
//! speakers `R` and active (relabelled) hypothesis speakers `H` are both
//! constant. For an interval of duration `d`, with `n_ref = |R|`,
//! `n_sys = |H|` and `n_correct` = the number of reference speakers whose
//! mapped hypothesis partner is also active in the interval:
//!
//! * **missed speech**       `d * max(0, n_ref - n_sys)` — reference speakers
//!   with no hypothesis speaker to account for them;
//! * **false-alarm speech**  `d * max(0, n_sys - n_ref)` — hypothesis speakers
//!   in excess of the reference;
//! * **speaker confusion**   `d * (min(n_ref, n_sys) - n_correct)` — the pairs
//!   that *could* have matched but were assigned the wrong label.
//!
//! `total` is the scored reference speaker time, `d * n_ref` summed over scored
//! intervals, and `der = (miss + false_alarm + confusion) / total`.
//!
//! Two scoring options ([`DerOptions`]) mirror `md-eval`:
//! * **collar** — a `+/-collar` no-score zone around every *reference* boundary
//!   (default `0.25 s`), forgiving annotation imprecision at speaker changes;
//! * **skip_overlap** — drop intervals where more than one reference speaker is
//!   active, scoring only single-speaker regions.
//!
//! The optimal label mapping itself is computed over the **entire** support
//! (it is collar- and overlap-independent); the collar and overlap options only
//! decide which intervals are *scored*.
//!
//! # JER (Jaccard formulation)
//!
//! After the same optimal mapping, [`jer`] averages, over reference speakers,
//! the per-speaker Jaccard error `1 - overlap / union` between a reference
//! speaker and its mapped hypothesis speaker. Unlike DER it is collar- and
//! overlap-agnostic and weights every speaker equally regardless of how much
//! they talk.

use crate::diarize::SpeakerSegment;
use crate::types::OxiWhisperError;
use std::collections::HashMap;

// ===========================================================================
// RTTM parsing
// ===========================================================================

/// A single reference speaker turn parsed from an RTTM `SPEAKER` line.
///
/// This is the *reference* counterpart of a
/// [`SpeakerSegment`]: it carries an arbitrary
/// speaker *name* (`String`, as written in the RTTM file — e.g. `speaker_0`,
/// `spk1`, `Alice`) rather than an opaque numeric
/// [`SpeakerId`](crate::diarize::SpeakerId), because reference annotations use
/// human-meaningful labels. `end` is stored (not duration) so that overlap
/// arithmetic in [`der`]/[`jer`] never has to re-derive it.
#[derive(Debug, Clone, PartialEq)]
pub struct RttmSegment {
    /// Speaker name exactly as it appears in field 8 (0-indexed field 7) of the
    /// RTTM line.
    pub speaker: String,
    /// Turn start time, in seconds (RTTM `tbeg`).
    pub start: f32,
    /// Turn end time, in seconds (`tbeg + tdur`).
    pub end: f32,
}

/// Parse a NIST RTTM document into its `SPEAKER` turns.
///
/// The RTTM line grammar this reads is the 10-field form emitted by
/// [`write_rttm`](crate::diarize::format::write_rttm) and expected by
/// `md-eval` / `pyannote.metrics`:
///
/// ```text
/// SPEAKER <file> <chan> <tbeg> <tdur> <ortho> <stype> <name> <conf> <slat>
///    0       1      2      3      4       5       6       7      8      9
/// ```
///
/// For every whitespace-tokenised line:
/// * blank lines and comment lines (first non-space character `;` or `#`) are
///   skipped;
/// * lines whose first token is not `SPEAKER` are skipped (RTTM allows other
///   record types such as `LEXEME` / `NON-SPEECH` which this scorer ignores);
/// * for a `SPEAKER` line, field 3 (`tbeg`) and field 4 (`tdur`) are parsed as
///   `f32` and field 7 (`name`) is taken verbatim; the segment's `end` is
///   `tbeg + tdur`.
///
/// # Errors
///
/// Returns [`OxiWhisperError::ConfigError`] — never panics and never silently
/// drops a malformed record — when a `SPEAKER` line has fewer than 8 tokens
/// (so field 7, the name, is missing) or when `tbeg`/`tdur` do not parse as a
/// finite `f32`.
pub fn parse_rttm(text: &str) -> Result<Vec<RttmSegment>, OxiWhisperError> {
    let mut out = Vec::new();
    for (line_no, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.first().copied() != Some("SPEAKER") {
            continue;
        }
        // Need at least tokens 0..=7 so that field 7 (the speaker name) exists.
        if tokens.len() < 8 {
            return Err(OxiWhisperError::ConfigError(format!(
                "RTTM line {}: SPEAKER record has {} tokens, need >= 8",
                line_no + 1,
                tokens.len()
            )));
        }
        let start = parse_field(tokens[3], "tbeg", line_no)?;
        let dur = parse_field(tokens[4], "tdur", line_no)?;
        out.push(RttmSegment {
            speaker: tokens[7].to_string(),
            start,
            end: start + dur,
        });
    }
    Ok(out)
}

/// Parse one numeric RTTM field, attaching the field name and 1-based line
/// number to any error.
fn parse_field(tok: &str, field: &str, line_no: usize) -> Result<f32, OxiWhisperError> {
    tok.parse::<f32>().map_err(|e| {
        OxiWhisperError::ConfigError(format!(
            "RTTM line {}: field '{}' = {:?} is not a valid f32: {}",
            line_no + 1,
            field,
            tok,
            e
        ))
    })
}

// ===========================================================================
// Hungarian (Kuhn–Munkres) optimal assignment — O(n^3)
// ===========================================================================

/// Solve the **minimum-cost** assignment problem with the Kuhn–Munkres
/// (Hungarian) algorithm in `O(n^3)`.
///
/// Given a `cost[i][j]` matrix, returns a vector `assign` of length
/// `cost.len()` (one entry per row) where `assign[i]` is the column assigned to
/// row `i`, chosen so that the total cost `sum_i cost[i][assign[i]]` is the
/// global minimum over all one-to-one row→column assignments. Costs may be
/// negative — the DER/JER mapping feeds `-overlap` here to turn "maximise
/// agreement" into a minimisation.
///
/// # Padding convention (rectangular inputs)
///
/// The classical algorithm needs a square matrix, so a rectangular input is
/// padded to `n = max(rows, cols)` with **zero-cost dummy** rows/columns:
///
/// * If there are **more rows than columns**, the extra dummy columns absorb the
///   surplus rows. The output still has one entry per real row, but a returned
///   column index `>= C` (where `C` is the original column count, i.e. the
///   longest input row) means "row `i` was matched to a dummy column" — it has
///   **no real assignment**. Callers must treat such an index as *unassigned*.
/// * If there are **more columns than rows**, the dummy rows soak up the surplus
///   columns; those dummy rows are solved internally and never appear in the
///   returned vector, so every returned index is a real column.
///
/// Because dummy entries cost `0` and every real DER/JER cost is `<= 0`, the
/// solver always prefers a real (overlapping) match over a dummy whenever any
/// positive overlap exists.
///
/// An empty matrix (no rows) returns an empty vector. Ragged rows are tolerated:
/// missing entries are read as `0.0`.
///
/// This is a genuine primal–dual shortest-augmenting-path implementation (the
/// `u`/`v` potential method), **not** a greedy nearest-neighbour heuristic and
/// **not** a factorial brute force.
pub fn hungarian(cost: &[Vec<f64>]) -> Vec<usize> {
    let nrows = cost.len();
    if nrows == 0 {
        return Vec::new();
    }
    let ncols = cost.iter().map(Vec::len).max().unwrap_or(0);
    let n = nrows.max(ncols);

    // Pad to an n x n square with zero-cost dummies (ragged rows read as 0.0).
    let mut a = vec![vec![0.0f64; n]; n];
    for (i, arow) in a.iter_mut().enumerate() {
        for (j, cell) in arow.iter_mut().enumerate() {
            *cell = cost.get(i).and_then(|r| r.get(j)).copied().unwrap_or(0.0);
        }
    }

    let full = solve_square(&a);
    full.into_iter().take(nrows).collect()
}

/// Kuhn–Munkres on an `n x n` matrix using the `u`/`v` potential method with a
/// per-row shortest-augmenting-path search. Returns `ans[i]` = column matched to
/// row `i`. All state is 1-indexed internally (column `0` is the virtual source
/// used by the augmenting path); the public [`hungarian`] wrapper handles
/// rectangular padding around this core.
fn solve_square(a: &[Vec<f64>]) -> Vec<usize> {
    let n = a.len();
    let inf = f64::INFINITY;
    // Potentials u (rows) and v (columns); p[j] = row currently matched to
    // column j (0 = unmatched); way[j] = predecessor column on the augmenting
    // path used for reconstruction.
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; n + 1];
    let mut p = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![inf; n + 1];
        let mut used = vec![false; n + 1];

        // Grow a shortest augmenting path from row i until it reaches a free
        // column (p[j0] == 0).
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0usize;
            for j in 1..=n {
                if !used[j] {
                    let cur = a[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            // Re-weight potentials so the found edge becomes tight.
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }

        // Flip the augmenting path, matching row i.
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut ans = vec![0usize; n];
    for j in 1..=n {
        if p[j] != 0 {
            ans[p[j] - 1] = j - 1;
        }
    }
    ans
}

// ===========================================================================
// DER / JER
// ===========================================================================

/// Scoring options for [`der`] (and, for API symmetry, [`jer`], which ignores
/// them).
#[derive(Debug, Clone, PartialEq)]
pub struct DerOptions {
    /// Half-width, in seconds, of the no-score collar placed around every
    /// **reference** segment boundary. Intervals whose midpoint lies within
    /// `+/-collar` of any reference boundary are excluded from scoring. `0.0`
    /// disables the collar. Default `0.25`.
    pub collar: f32,
    /// When `true`, intervals in which more than one reference speaker is active
    /// (overlapped speech) are excluded from scoring. Default `false`.
    pub skip_overlap: bool,
}

impl Default for DerOptions {
    fn default() -> Self {
        Self {
            collar: 0.25,
            skip_overlap: false,
        }
    }
}

/// The decomposed result of a [`der`] computation.
///
/// All durations are in seconds and all fields are `f64`. `der` is the ratio
/// `(miss + false_alarm + confusion) / total`; the three error terms and the
/// `total` scored reference time are exposed separately so callers can report
/// the breakdown (missed vs. false-alarm vs. confusion) that DER alone hides.
#[derive(Debug, Clone, PartialEq)]
pub struct DerReport {
    /// Diarization Error Rate: `(miss + false_alarm + confusion) / total`.
    /// `0.0` when there is no scored reference speech.
    pub der: f64,
    /// Missed-speech time: scored reference speaker time with too few
    /// hypothesis speakers to cover it.
    pub miss: f64,
    /// False-alarm time: scored hypothesis speaker time in excess of the
    /// reference.
    pub false_alarm: f64,
    /// Speaker-confusion time: matched-count deficit `min(n_ref, n_sys) -
    /// n_correct` integrated over scored intervals.
    pub confusion: f64,
    /// Total scored reference speaker time (the DER denominator).
    pub total: f64,
}

/// A speaker in the alignment, indexed 0..n on each side.
type Sid = usize;

/// One elementary interval of the timeline over which the active reference and
/// hypothesis speaker sets are both constant.
struct Interval {
    dur: f32,
    mid: f32,
    active_ref: Vec<Sid>,
    active_hyp: Vec<Sid>,
}

/// The shared substrate of [`der`] and [`jer`]: the elementary-interval
/// timeline, the co-occurrence matrix, per-speaker total durations, and the
/// optimal hypothesis→reference label mapping.
struct Alignment {
    intervals: Vec<Interval>,
    /// Sorted, de-duplicated reference boundaries (for collar exclusion).
    ref_boundaries: Vec<f32>,
    /// `cooccur[h][r]` = total time hypothesis speaker `h` and reference
    /// speaker `r` are simultaneously active.
    cooccur: Vec<Vec<f64>>,
    /// Total active time of each reference speaker (self-overlap merged).
    ref_dur: Vec<f64>,
    /// Total active time of each hypothesis speaker (self-overlap merged).
    hyp_dur: Vec<f64>,
    /// `ref_to_hyp[r]` = the hypothesis speaker mapped onto reference speaker
    /// `r`, or `None` if `r` has no partner (more reference than hypothesis
    /// speakers).
    ref_to_hyp: Vec<Option<Sid>>,
    n_ref: usize,
}

/// True iff `t` lies in `[start, end)`.
fn contains(start: f32, end: f32, t: f32) -> bool {
    start <= t && t < end
}

/// Build the elementary-interval timeline and the optimal speaker mapping.
///
/// `collar` only affects the *granularity* of the interval grid (extra cut
/// points at every reference boundary `+/-collar`); it does not change the
/// co-occurrence sums or the mapping, which are subdivision-invariant. Callers
/// that do not need collar-aligned intervals (e.g. [`jer`]) pass `0.0`.
fn build_alignment(
    reference: &[RttmSegment],
    hypothesis: &[SpeakerSegment],
    collar: f32,
) -> Alignment {
    // First-appearance-ordered speaker indices (deterministic; HashMap is used
    // for lookup only, never iterated).
    let mut ref_index: HashMap<&str, Sid> = HashMap::new();
    for s in reference {
        let next = ref_index.len();
        ref_index.entry(s.speaker.as_str()).or_insert(next);
    }
    let n_ref = ref_index.len();

    let mut hyp_index: HashMap<u32, Sid> = HashMap::new();
    for s in hypothesis {
        let next = hyp_index.len();
        hyp_index.entry(s.speaker.0).or_insert(next);
    }
    let n_hyp = hyp_index.len();

    // Collect timeline cut points and (separately) the reference boundaries the
    // collar keys off.
    let mut bounds: Vec<f32> = Vec::new();
    let mut ref_boundaries: Vec<f32> = Vec::new();
    for s in reference {
        bounds.push(s.start);
        bounds.push(s.end);
        ref_boundaries.push(s.start);
        ref_boundaries.push(s.end);
        if collar > 0.0 {
            bounds.push(s.start - collar);
            bounds.push(s.start + collar);
            bounds.push(s.end - collar);
            bounds.push(s.end + collar);
        }
    }
    for s in hypothesis {
        bounds.push(s.start);
        bounds.push(s.end);
    }
    bounds.sort_by(f32::total_cmp);
    bounds.dedup();
    ref_boundaries.sort_by(f32::total_cmp);
    ref_boundaries.dedup();

    let mut intervals: Vec<Interval> = Vec::new();
    let mut cooccur = vec![vec![0.0f64; n_ref]; n_hyp];
    let mut ref_dur = vec![0.0f64; n_ref];
    let mut hyp_dur = vec![0.0f64; n_hyp];

    for w in bounds.windows(2) {
        let t0 = w[0];
        let t1 = w[1];
        let d = t1 - t0;
        if d <= 0.0 {
            continue;
        }
        let mid = t0 + d * 0.5;

        let mut active_ref: Vec<Sid> = Vec::new();
        for s in reference {
            if contains(s.start, s.end, mid)
                && let Some(&ri) = ref_index.get(s.speaker.as_str())
                && !active_ref.contains(&ri)
            {
                active_ref.push(ri);
            }
        }
        let mut active_hyp: Vec<Sid> = Vec::new();
        for s in hypothesis {
            if contains(s.start, s.end, mid)
                && let Some(&hi) = hyp_index.get(&s.speaker.0)
                && !active_hyp.contains(&hi)
            {
                active_hyp.push(hi);
            }
        }

        let dd = d as f64;
        for &hi in &active_hyp {
            for &ri in &active_ref {
                cooccur[hi][ri] += dd;
            }
        }
        for &ri in &active_ref {
            ref_dur[ri] += dd;
        }
        for &hi in &active_hyp {
            hyp_dur[hi] += dd;
        }

        intervals.push(Interval {
            dur: d,
            mid,
            active_ref,
            active_hyp,
        });
    }

    // Optimal mapping: minimise -overlap == maximise co-occurrence.
    let cost: Vec<Vec<f64>> = cooccur
        .iter()
        .map(|row| row.iter().map(|&c| -c).collect())
        .collect();
    let assign = hungarian(&cost);
    let mut ref_to_hyp: Vec<Option<Sid>> = vec![None; n_ref];
    for (h, &col) in assign.iter().enumerate() {
        if col < n_ref {
            // col is a real reference speaker (not a dummy padding column).
            ref_to_hyp[col] = Some(h);
        }
    }

    Alignment {
        intervals,
        ref_boundaries,
        cooccur,
        ref_dur,
        hyp_dur,
        ref_to_hyp,
        n_ref,
    }
}

/// Compute the Diarization Error Rate (and its miss/false-alarm/confusion
/// decomposition) of `hypothesis` against `reference`.
///
/// See the [module documentation](self) for the full algorithm. In brief: the
/// hypothesis labels are optimally re-mapped onto the reference labels
/// (maximising agreement, via [`hungarian`] on a `-overlap` cost matrix), the
/// timeline is diced into elementary intervals, the [`collar`](DerOptions::collar)
/// and [`skip_overlap`](DerOptions::skip_overlap) options select which intervals
/// are scored, and each scored interval of duration `d` contributes:
///
/// * `miss        += d * max(0, n_ref - n_sys)`
/// * `false_alarm += d * max(0, n_sys - n_ref)`
/// * `confusion   += d * (min(n_ref, n_sys) - n_correct)`
/// * `total       += d * n_ref`
///
/// where `n_ref`/`n_sys` are the active reference/hypothesis speaker counts and
/// `n_correct` is the number of active reference speakers whose mapped
/// hypothesis partner is also active.
///
/// When there is no scored reference speech (`total == 0`) the report is
/// all-zero (`der = 0.0`): DER is undefined without a reference-time
/// denominator, and this crate reports `0.0` rather than a NaN or a spuriously
/// large ratio.
pub fn der(
    reference: &[RttmSegment],
    hypothesis: &[SpeakerSegment],
    opts: &DerOptions,
) -> DerReport {
    let al = build_alignment(reference, hypothesis, opts.collar);

    let mut miss = 0.0f64;
    let mut false_alarm = 0.0f64;
    let mut confusion = 0.0f64;
    let mut total = 0.0f64;

    for iv in &al.intervals {
        // Collar: drop intervals within +/-collar of any reference boundary.
        if opts.collar > 0.0
            && al
                .ref_boundaries
                .iter()
                .any(|&b| (iv.mid - b).abs() < opts.collar)
        {
            continue;
        }
        let n_ref = iv.active_ref.len();
        if opts.skip_overlap && n_ref > 1 {
            continue;
        }
        let n_sys = iv.active_hyp.len();
        let d = iv.dur as f64;

        let mut n_correct = 0usize;
        for &r in &iv.active_ref {
            if let Some(h) = al.ref_to_hyp[r]
                && iv.active_hyp.contains(&h)
            {
                n_correct += 1;
            }
        }

        miss += d * (n_ref.saturating_sub(n_sys) as f64);
        false_alarm += d * (n_sys.saturating_sub(n_ref) as f64);
        confusion += d * (n_ref.min(n_sys).saturating_sub(n_correct) as f64);
        total += d * n_ref as f64;
    }

    if total == 0.0 {
        return DerReport {
            der: 0.0,
            miss: 0.0,
            false_alarm: 0.0,
            confusion: 0.0,
            total: 0.0,
        };
    }

    DerReport {
        der: (miss + false_alarm + confusion) / total,
        miss,
        false_alarm,
        confusion,
        total,
    }
}

/// Compute the Jaccard Error Rate of `hypothesis` against `reference`.
///
/// After the same optimal speaker mapping as [`der`], for each reference
/// speaker `s` the per-speaker Jaccard error is
/// `1 - overlap(s, mapped_hyp(s)) / union(s, mapped_hyp(s))`, where `overlap`
/// is their simultaneous-active time and `union = dur(s) + dur(mapped) -
/// overlap` (each side's own self-overlap already merged). A reference speaker
/// with **no** mapped hypothesis partner contributes the maximal error `1.0`.
/// The result is the unweighted mean over reference speakers, so every speaker
/// counts equally regardless of talk time.
///
/// Unlike [`der`], JER is **collar- and overlap-agnostic** — `opts` is accepted
/// only for API symmetry and does not influence the result. With no reference
/// speakers the mean is undefined and `0.0` is returned.
pub fn jer(reference: &[RttmSegment], hypothesis: &[SpeakerSegment], opts: &DerOptions) -> f64 {
    // JER ignores the collar / skip_overlap knobs by construction; `opts` is
    // part of the signature only for symmetry with `der`.
    let _ = opts;
    let al = build_alignment(reference, hypothesis, 0.0);
    if al.n_ref == 0 {
        return 0.0;
    }

    let mut sum = 0.0f64;
    for r in 0..al.n_ref {
        let je = match al.ref_to_hyp[r] {
            Some(h) => {
                let overlap = al.cooccur[h][r];
                let union = al.ref_dur[r] + al.hyp_dur[h] - overlap;
                if union > 0.0 {
                    1.0 - overlap / union
                } else {
                    1.0
                }
            }
            None => 1.0,
        };
        sum += je;
    }
    sum / al.n_ref as f64
}

// ===========================================================================
// Tests — every assertion is a hand-computed exact value (pencil work in the
// comments). None of these pass on empty / degenerate input.
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diarize::format::rttm_string;
    use crate::diarize::{DiarizeResult, SpeakerId, SpeakerSegment};

    /// Assert two f64 values agree to 1e-6, printing both on failure.
    fn close(actual: f64, expected: f64, what: &str) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "{what}: got {actual}, expected {expected}"
        );
    }

    fn hyp(id: u32, start: f32, end: f32) -> SpeakerSegment {
        SpeakerSegment {
            speaker: SpeakerId(id),
            start,
            end,
        }
    }

    fn rttm(name: &str, start: f32, end: f32) -> RttmSegment {
        RttmSegment {
            speaker: name.to_string(),
            start,
            end,
        }
    }

    /// Total cost of an assignment, for verifying optimality.
    fn assignment_cost(cost: &[Vec<f64>], assign: &[usize]) -> f64 {
        assign.iter().enumerate().map(|(i, &j)| cost[i][j]).sum()
    }

    // ── Hungarian ───────────────────────────────────────────────────────

    #[test]
    fn test_hungarian_greedy_would_miss_3x3() {
        // cost[i][j] = (i+1)*(j+1):
        //   [1 2 3]
        //   [2 4 6]
        //   [3 6 9]
        // A global-greedy solver picks the smallest cell (0,0)=1 first, then is
        // forced into (1,1)=4 and (2,2)=9 for a total of 1+4+9 = 14.
        // By the rearrangement inequality the minimum sum of products pairs the
        // sequences in OPPOSITE order: row0->col2, row1->col1, row2->col0 =
        // 1*3 + 2*2 + 3*1 = 3+4+3 = 10 (unique; next best is 11). Greedy misses.
        let cost = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![3.0, 6.0, 9.0],
        ];
        let assign = hungarian(&cost);
        assert_eq!(assign, vec![2, 1, 0], "optimal permutation");
        close(assignment_cost(&cost, &assign), 10.0, "optimal cost");

        // Sanity: the greedy answer really is worse, so this matrix genuinely
        // separates the two strategies.
        close(assignment_cost(&cost, &[0, 1, 2]), 14.0, "greedy cost");
    }

    #[test]
    fn test_hungarian_rectangular_more_rows_than_cols() {
        // 3 rows, 2 cols -> padded with one zero-cost dummy column (index 2).
        //   r0: [1 2]   r1: [3 4]   r2: [0 5]
        // Enumerating the 3! ways to place rows on {col0, col1, dummy(0)}:
        //   r0->1, r1->dummy, r2->0  => 2 + 0 + 0 = 2   (unique minimum)
        // so r1 has NO real assignment (mapped to dummy column 2 >= ncols=2).
        let cost = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![0.0, 5.0]];
        let assign = hungarian(&cost);
        assert_eq!(assign, vec![1, 2, 0], "assignment incl. dummy column");
        // Real cost counts only the two real matches (r0->1=2, r2->0=0) = 2.
        let ncols = 2;
        assert_eq!(assign[0], 1);
        assert!(assign[1] >= ncols, "row 1 is unassigned (dummy column)");
        assert_eq!(assign[2], 0);
    }

    // ── DER ─────────────────────────────────────────────────────────────
    //
    // Shared hand timeline for the DER tests:
    //   reference:  A = [0, 2],  B = [2, 4]      (no ref overlap; ref speech 4s)
    //   hypothesis: 0 = [0.5, 2.5], 1 = [3.0, 4.5]
    //
    // Co-occurrence (full support):
    //         A     B
    //   h0   1.5   0.5
    //   h1   0.0   1.0
    // Optimal mapping minimises -overlap: h0->A (1.5) + h1->B (1.0) = 2.5
    //   beats h0->B (0.5) + h1->A (0.0) = 0.5, so map h0->A, h1->B.

    fn der_ref() -> Vec<RttmSegment> {
        vec![rttm("A", 0.0, 2.0), rttm("B", 2.0, 4.0)]
    }
    fn der_hyp() -> Vec<SpeakerSegment> {
        vec![hyp(0, 0.5, 2.5), hyp(1, 3.0, 4.5)]
    }

    #[test]
    fn test_der_hand_timeline_no_collar() {
        // Elementary intervals over boundaries {0, 0.5, 2, 2.5, 3, 4, 4.5}:
        //   [0.0,0.5] d0.5  R={A} H={}      -> miss 0.5                total 0.5
        //   [0.5,2.0] d1.5  R={A} H={h0=A}  -> correct                 total 1.5
        //   [2.0,2.5] d0.5  R={B} H={h0=A}  -> confusion 0.5           total 0.5
        //   [2.5,3.0] d0.5  R={B} H={}      -> miss 0.5                total 0.5
        //   [3.0,4.0] d1.0  R={B} H={h1=B}  -> correct                 total 1.0
        //   [4.0,4.5] d0.5  R={}  H={h1}    -> false_alarm 0.5         total 0.0
        // miss=1.0  false_alarm=0.5  confusion=0.5  total=4.0
        // der = (1.0+0.5+0.5)/4.0 = 0.5
        let opts = DerOptions {
            collar: 0.0,
            skip_overlap: false,
        };
        let r = der(&der_ref(), &der_hyp(), &opts);
        close(r.miss, 1.0, "miss");
        close(r.false_alarm, 0.5, "false_alarm");
        close(r.confusion, 0.5, "confusion");
        close(r.total, 4.0, "total");
        close(r.der, 0.5, "der");
    }

    #[test]
    fn test_der_collar_changes_scored_region() {
        // Same timeline, collar = 0.25. No-score zones (±0.25) around ref
        // boundaries {0, 2, 4}: (-0.25,0.25), (1.75,2.25), (3.75,4.25).
        // Surviving scored intervals:
        //   [0.25,0.5]  d0.25  R={A} H={}     -> miss 0.25       total 0.25
        //   [0.5,1.75]  d1.25  R={A} H={h0=A} -> correct         total 1.25
        //   [2.25,2.5]  d0.25  R={B} H={h0=A} -> confusion 0.25  total 0.25
        //   [2.5,3.0]   d0.5   R={B} H={}     -> miss 0.5        total 0.5
        //   [3.0,3.75]  d0.75  R={B} H={h1=B} -> correct         total 0.75
        //   [4.25,4.5]  d0.25  R={}  H={h1}   -> false_alarm 0.25 total 0.0
        // miss=0.75  false_alarm=0.25  confusion=0.25  total=3.0
        // der = 1.25 / 3.0 = 0.41666...
        let no_collar = der(
            &der_ref(),
            &der_hyp(),
            &DerOptions {
                collar: 0.0,
                skip_overlap: false,
            },
        );
        let collar = der(
            &der_ref(),
            &der_hyp(),
            &DerOptions {
                collar: 0.25,
                skip_overlap: false,
            },
        );

        // The collar demonstrably changes the outcome.
        close(no_collar.total, 4.0, "no-collar total");
        close(collar.total, 3.0, "collar total");
        assert!(
            (no_collar.total - collar.total).abs() > 1e-6,
            "collar must change the scored total"
        );

        close(collar.miss, 0.75, "collar miss");
        close(collar.false_alarm, 0.25, "collar false_alarm");
        close(collar.confusion, 0.25, "collar confusion");
        close(collar.der, 1.25 / 3.0, "collar der");
    }

    #[test]
    fn test_der_perfect_hypothesis_is_zero() {
        // Hypothesis identical to the reference partition -> zero error.
        //   ref A=[0,2] B=[2,4]; hyp 0=[0,2] 1=[2,4]. Map 0->A, 1->B.
        //   two intervals, both correct: der 0, total = ref speech = 4.
        let reference = vec![rttm("A", 0.0, 2.0), rttm("B", 2.0, 4.0)];
        let hypothesis = vec![hyp(0, 0.0, 2.0), hyp(1, 2.0, 4.0)];
        let r = der(
            &reference,
            &hypothesis,
            &DerOptions {
                collar: 0.0,
                skip_overlap: false,
            },
        );
        close(r.der, 0.0, "perfect der");
        close(r.miss, 0.0, "perfect miss");
        close(r.false_alarm, 0.0, "perfect false_alarm");
        close(r.confusion, 0.0, "perfect confusion");
        close(r.total, 4.0, "perfect total");
    }

    #[test]
    fn test_der_skip_overlap_excludes_overlap_region() {
        // reference with genuine overlap: A=[0,2], B=[1,3] (both active on [1,2]).
        // hypothesis: single speaker 0=[0,2.5].
        //   cooccur h0-A = 2.0 (on [0,2)), h0-B = 1.5 (on [1,2.5)).
        //   map h0->A (cost -2.0 beats -1.5); ref_to_hyp: A->h0, B->None.
        // Boundaries {0,1,2,2.5,3}:
        //   [0,1]   d1.0  R={A}   H={h0=A} -> correct                total 1.0
        //   [1,2]   d1.0  R={A,B} H={h0}   -> miss 1 (n_ref2,n_sys1) total 2.0   [OVERLAP]
        //   [2,2.5] d0.5  R={B}   H={h0}   -> confusion 0.5          total 0.5
        //   [2.5,3] d0.5  R={B}   H={}     -> miss 0.5               total 0.5
        // Without skip_overlap: miss=1.5 fa=0 conf=0.5 total=4.0
        // With skip_overlap the [1,2] interval is dropped:
        //   miss=0.5 fa=0 conf=0.5 total=2.0
        let reference = vec![rttm("A", 0.0, 2.0), rttm("B", 1.0, 3.0)];
        let hypothesis = vec![hyp(0, 0.0, 2.5)];

        let full = der(
            &reference,
            &hypothesis,
            &DerOptions {
                collar: 0.0,
                skip_overlap: false,
            },
        );
        close(full.miss, 1.5, "full miss");
        close(full.false_alarm, 0.0, "full false_alarm");
        close(full.confusion, 0.5, "full confusion");
        close(full.total, 4.0, "full total");

        let skip = der(
            &reference,
            &hypothesis,
            &DerOptions {
                collar: 0.0,
                skip_overlap: true,
            },
        );
        close(skip.miss, 0.5, "skip miss");
        close(skip.false_alarm, 0.0, "skip false_alarm");
        close(skip.confusion, 0.5, "skip confusion");
        close(skip.total, 2.0, "skip total");

        // skip_overlap genuinely removed the overlap interval's contribution.
        assert!(
            (full.total - skip.total).abs() > 1e-6 && (full.miss - skip.miss).abs() > 1e-6,
            "skip_overlap must change the scored totals"
        );
    }

    // ── JER ─────────────────────────────────────────────────────────────

    #[test]
    fn test_jer_hand_timeline() {
        // reference:  A = [0, 4] (dur 4),  B = [4, 6] (dur 2)
        // hypothesis: 0 = [0, 3] (dur 3),  1 = [3, 7] (dur 4)
        // cooccur:  (h0,A)=3  (h0,B)=0  (h1,A)=1  (h1,B)=2
        // optimal map: h0->A (3) + h1->B (2) = 5 beats h0->B(0)+h1->A(1)=1.
        // JER(A) = 1 - overlap/union = 1 - 3 / (4 + 3 - 3) = 1 - 3/4 = 0.25
        // JER(B) = 1 - 2 / (2 + 4 - 2)               = 1 - 2/4 = 0.50
        // mean   = (0.25 + 0.50) / 2 = 0.375
        let reference = vec![rttm("A", 0.0, 4.0), rttm("B", 4.0, 6.0)];
        let hypothesis = vec![hyp(0, 0.0, 3.0), hyp(1, 3.0, 7.0)];
        let value = jer(&reference, &hypothesis, &DerOptions::default());
        close(value, 0.375, "jer");
    }

    #[test]
    fn test_jer_unmapped_reference_speaker_is_full_error() {
        // Two ref speakers, one hyp speaker: B cannot be mapped -> JER(B)=1.0.
        // ref A=[0,2] B=[2,4]; hyp 0=[0,2] (maps to A, JER(A)=0).
        // mean = (0.0 + 1.0)/2 = 0.5
        let reference = vec![rttm("A", 0.0, 2.0), rttm("B", 2.0, 4.0)];
        let hypothesis = vec![hyp(0, 0.0, 2.0)];
        let value = jer(&reference, &hypothesis, &DerOptions::default());
        close(value, 0.5, "jer with unmapped speaker");
    }

    // ── parse_rttm ──────────────────────────────────────────────────────

    #[test]
    fn test_parse_rttm_round_trips_rttm_string() {
        // Build a DiarizeResult, serialise via E2's rttm_string, parse it back,
        // and assert the exact RttmSegment vector. All times are exact f32
        // (multiples of 0.25) so start/end compare bit-for-bit.
        let result = DiarizeResult {
            segments: vec![
                SpeakerSegment {
                    speaker: SpeakerId(0),
                    start: 0.0,
                    end: 1.5,
                },
                SpeakerSegment {
                    speaker: SpeakerId(1),
                    start: 1.5,
                    end: 3.25,
                },
                SpeakerSegment {
                    speaker: SpeakerId(0),
                    start: 3.25,
                    end: 4.0,
                },
            ],
            num_speakers: 2,
        };
        let text = rttm_string(&result, "utt");
        let parsed = parse_rttm(&text).expect("round-trip parse of well-formed RTTM");
        assert_eq!(
            parsed,
            vec![
                rttm("speaker_0", 0.0, 1.5),
                rttm("speaker_1", 1.5, 3.25),
                rttm("speaker_0", 3.25, 4.0),
            ]
        );
    }

    #[test]
    fn test_parse_rttm_skips_comments_blanks_and_other_records() {
        let text = "\
;; this is a comment\n\
# another comment\n\
\n\
LEXEME utt 1 0.0 0.5 <NA> <NA> hello <NA> <NA>\n\
SPEAKER utt 1 0.000 1.500 <NA> <NA> spkA <NA> <NA>\n\
   \n\
SPEAKER utt 1 1.500 0.500 <NA> <NA> spkB <NA> <NA>\n";
        let parsed = parse_rttm(text).expect("parse mixed document");
        assert_eq!(
            parsed,
            vec![rttm("spkA", 0.0, 1.5), rttm("spkB", 1.5, 2.0)],
            "only the two SPEAKER lines survive"
        );
    }

    #[test]
    fn test_parse_rttm_errors_on_non_numeric_field() {
        let text = "SPEAKER utt 1 zero 1.500 <NA> <NA> spkA <NA> <NA>\n";
        let err = parse_rttm(text).expect_err("non-numeric tbeg must error");
        match err {
            OxiWhisperError::ConfigError(msg) => {
                assert!(
                    msg.contains("tbeg"),
                    "error names the offending field: {msg}"
                );
            }
            other => panic!("expected ConfigError, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_rttm_errors_on_too_few_tokens() {
        // Only 5 tokens: field 7 (name) is missing.
        let text = "SPEAKER utt 1 0.0 1.5\n";
        let err = parse_rttm(text).expect_err("truncated SPEAKER line must error");
        assert!(matches!(err, OxiWhisperError::ConfigError(_)));
    }

    #[test]
    fn test_parse_rttm_empty_input_is_empty_vec() {
        assert_eq!(parse_rttm("").expect("empty parses"), Vec::new());
    }
}
