#![cfg(all(feature = "diarization", feature = "test-utils"))]

//! End-to-end diarization on a deterministic synthetic multi-speaker mixture.
//!
//! This wires the whole pipeline together — the [`synthetic_multispeaker`]
//! phonation-proxy mixer (with its exact ground-truth RTTM), a loadable
//! synthetic [`WhisperModel`], the built-in [`WhisperEncoderEmbedder`] baseline
//! reached via [`WhisperModel::diarize`], and the [`der`] scorer — and pins the
//! **measured** Diarization Error Rate.
//!
//! # Honesty note (Batch F2) — why this is a *characterization* test
//!
//! The baseline embedder mean-pools Whisper **encoder** features, and Whisper's
//! encoder is trained to be largely speaker-*invariant* (it encodes phonetic
//! content for ASR). On this fixture the "speakers" are distinct sums of
//! sinusoids, so a low DER here would reflect raw spectral-content separation,
//! **not** genuine speaker-embedding quality. The measured behaviour is: the
//! pipeline keeps both requested speakers (`num_speakers = Some(2)`) but
//! mislocates the turn boundaries, giving `der = 0.265625` — far from the
//! sub-0.1 a production system reaches.
//! We therefore assert the recovered count and pin the measured DER band tightly
//! (to catch regressions) rather than assert a vacuous bound to look green.
//!
//! **Real** DER validation — a genuine sub-τ speaker-discrimination result —
//! requires a pretrained ECAPA-TDNN / x-vector ONNX checkpoint fed through
//! [`WhisperModel::diarize_with_embedder`] and a labelled speech corpus, both of
//! which are out of scope for this in-crate fixture (tracked as the remaining
//! Batch F2 work).
//!
//! [`WhisperEncoderEmbedder`]: oxiwhisper::WhisperEncoderEmbedder
//! [`WhisperModel::diarize`]: oxiwhisper::WhisperModel::diarize
//! [`WhisperModel::diarize_with_embedder`]: oxiwhisper::WhisperModel::diarize_with_embedder

use oxiwhisper::test_utils::{generate_synthetic_model, synthetic_multispeaker};
use oxiwhisper::{DerOptions, DiarizeOptions, SpeakerId, SpeakerSegment, WhisperModel, der};

/// Deletes the temp model file when the test scope ends.
struct TempModel(std::path::PathBuf);
impl Drop for TempModel {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Build the two-speaker fixture, its RTTM, and a loaded synthetic model.
fn setup() -> (
    Vec<f32>,
    Vec<oxiwhisper::RttmSegment>,
    WhisperModel,
    TempModel,
) {
    const SR: usize = 16_000;
    let (audio, reference) = synthetic_multispeaker(2, 2.5, SR);
    // Fixture sanity: 2 speakers × 2 rounds = 4 back-to-back 2.5 s turns = 10 s.
    assert_eq!(reference.len(), 4, "expected 4 round-robin turns");
    assert_eq!(audio.len(), 4 * (2.5 * SR as f32).round() as usize);
    let path = generate_synthetic_model();
    let cleanup = TempModel(path.clone());
    let model = WhisperModel::from_file(&path).expect("load synthetic WhisperModel");
    (audio, reference, model, cleanup)
}

#[test]
fn test_baseline_diarize_two_speakers_measured_der() {
    let (audio, reference, model, _cleanup) = setup();

    let opts = DiarizeOptions {
        num_speakers: Some(2),
        min_speakers: 1,
        max_speakers: 5,
        ..DiarizeOptions::default()
    };
    let result = model
        .diarize(&audio, &opts)
        .expect("diarize the synthetic mixture");
    let report = der(&reference, &result.segments, &DerOptions::default());

    // `num_speakers: Some(2)` was requested, so count discovery is NOT under test
    // here; this asserts the pipeline PRESERVES both requested speakers through
    // resegmentation — a reseg bug that dropped a speaker would make this 1.
    assert_eq!(
        result.num_speakers, 2,
        "pipeline must preserve both requested speakers through resegmentation"
    );
    // There is real scored reference speech: 10 s of single-speaker reference
    // minus the ±0.25 s collar around the 5 interior/edge boundaries = 8.0 s.
    // (Guards against the v0.10 trap of a DER that is vacuously 0 on empty data.)
    assert!(
        (report.total - 8.0).abs() < 1e-6,
        "scored reference speech must be 8.0 s, got {}",
        report.total
    );
    assert!(report.der.is_finite(), "der must be finite");

    // Characterization pin (see the module-level Batch F2 note): the measured
    // decomposition on this deterministic fixture is
    //   miss = 0, false_alarm = 0, confusion = 2.125 s  ->  der = 2.125/8 = 0.265625.
    // The confusion is entirely a mislocated turn boundary (the baseline lumps
    // ~7.9 s under one speaker), which is exactly the speaker-invariant weakness
    // this test documents. The tolerances are far below the ~0.09 DER step a
    // single flipped window would cause, so any real regression trips them while
    // exact f64 ratio arithmetic keeps the pin stable.
    assert!(
        report.miss.abs() < 1e-6,
        "measured miss must be 0, got {}",
        report.miss
    );
    assert!(
        report.false_alarm.abs() < 1e-6,
        "measured false_alarm must be 0, got {}",
        report.false_alarm
    );
    assert!(
        (report.confusion - 2.125).abs() < 1e-4,
        "measured confusion must be ~2.125 s, got {}",
        report.confusion
    );
    assert!(
        (report.der - 0.265625).abs() < 1e-4,
        "characterization: speaker-invariant baseline, DER measured ~0.265625, \
         see Batch F2 note; got {}",
        report.der
    );
    // Non-vacuous framing: the baseline is genuinely imperfect (real confusion is
    // scored) yet not catastrophic — so this exercises the error path, not a
    // trivially-perfect or trivially-broken one.
    assert!(
        report.der > 0.0 && report.der < 0.5,
        "baseline DER must be a real, non-trivial error in (0, 0.5), got {}",
        report.der
    );
}

#[test]
fn test_oracle_hypothesis_scores_zero_der() {
    // Sibling to the characterization test: an ORACLE hypothesis equal to the
    // ground truth must score DER = 0. This proves (a) the fixture's RTTM is
    // internally consistent and scorable, and (b) the 0.265625 above is a real
    // pipeline error, not an artefact of a broken scorer — if `der` were broken
    // this would not be 0, and if the reference were malformed the oracle could
    // not reach 0.
    let (_audio, reference, _model, _cleanup) = setup();

    let oracle: Vec<SpeakerSegment> = reference
        .iter()
        .map(|r| {
            let id: u32 = r
                .speaker
                .strip_prefix("speaker_")
                .and_then(|s| s.parse().ok())
                .expect("fixture labels are 'speaker_<k>'");
            SpeakerSegment {
                speaker: SpeakerId(id),
                start: r.start,
                end: r.end,
            }
        })
        .collect();

    let report = der(&reference, &oracle, &DerOptions::default());
    assert!(
        (report.total - 8.0).abs() < 1e-6,
        "oracle must score the same 8.0 s of reference speech, got {}",
        report.total
    );
    assert!(
        report.der.abs() < 1e-6,
        "an oracle hypothesis must score DER = 0, got {}",
        report.der
    );
}
