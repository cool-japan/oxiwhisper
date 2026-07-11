//! Smoke tests for the `threading` module and the `parallel` feature's
//! effect on the end-to-end transcription pipeline.
//!
//! What this file actually verifies:
//! 1. `test_set_thread_count_pool_lifecycle` — the *real* contract of
//!    `threading::set_thread_count`: it must succeed on the first call in a
//!    process, and (only with the `parallel` feature) must return a
//!    descriptive `Err` on the second call because rayon's global pool can
//!    only be built once. See the harness note on that test for why the
//!    whole lifecycle must live in a single `#[test]` fn.
//! 2. `test_transcribe_end_to_end_on_synthetic_mel` — the PERF1/TODO.md-
//!    promised check that the transcription pipeline (mel spectrogram ->
//!    encoder -> decoder, including the rayon-parallelized per-head
//!    attention loops when `parallel` is enabled) runs to completion on a
//!    synthetic model and produces a structurally valid, deterministic
//!    result — not merely `Ok(())` with the value thrown away.
//!
//! Run with: cargo nextest run --test threading_smoke --features test-utils
//!           cargo nextest run --test threading_smoke --features test-utils,parallel
//! Both must pass.

#[path = "common/mod.rs"]
mod common;

/// Exercises the complete `set_thread_count` contract in a single test.
///
/// # Why this must be ONE test function, not several
///
/// `cargo nextest` spawns a fresh process per test, so every test that calls
/// `set_thread_count` would see rayon's global pool in its pristine,
/// uninitialised state — "the first call always succeeds" would hold no
/// matter which test made it. `cargo test`, by contrast, runs every test in
/// this binary in one shared process, so whichever test happens to call
/// `set_thread_count` first "wins" the global pool, and any other test that
/// also calls it would non-deterministically observe the "already
/// initialised" `Err` depending on test execution order. Splitting the
/// "first call" and "second call" contracts across separate `#[test]` fns
/// would therefore be correct under nextest but flaky/order-dependent under
/// `cargo test`.
///
/// Owning the entire lifecycle (first call, then second call, both asserted
/// inline) in a single test sidesteps the ordering problem entirely: this
/// function is the only place in this file that touches
/// `threading::set_thread_count`, so it is always the first *and* second
/// caller in whichever process it runs in, under both harnesses.
#[test]
fn test_set_thread_count_pool_lifecycle() {
    // First call in this process. With `parallel`, this is the call that
    // builds rayon's global thread pool and must succeed. Without
    // `parallel`, `set_thread_count` is an unconditional no-op and must
    // also succeed.
    let first = oxiwhisper::threading::set_thread_count(2);
    assert!(
        first.is_ok(),
        "first call to set_thread_count must succeed: {first:?}"
    );

    // Second call in this process, with a different thread count.
    let second = oxiwhisper::threading::set_thread_count(4);

    #[cfg(feature = "parallel")]
    {
        // rayon only allows `build_global()` to succeed once per process;
        // this call must return a descriptive Err, not panic and not
        // silently succeed (which would mean the requested thread count
        // was silently ignored).
        let err = second.expect_err(
            "second call to set_thread_count must fail once rayon's global pool \
             is already initialised",
        );
        assert!(
            err.contains("Failed to set thread count"),
            "error message should explain what failed: {err}"
        );
    }

    #[cfg(not(feature = "parallel"))]
    {
        // Without `parallel` there is no global pool to protect, so every
        // call -- including the second -- must return Ok(()).
        assert!(
            second.is_ok(),
            "set_thread_count must remain a no-op returning Ok on every call \
             without the `parallel` feature: {second:?}"
        );
    }
}

/// End-to-end smoke test: the transcription pipeline (mel spectrogram ->
/// encoder -> decoder, including per-head attention loops that run through
/// rayon when the `parallel` feature is enabled) must run to completion on
/// a synthetic model and produce a structurally valid, deterministic
/// result.
///
/// This is the "parallel build runs end-to-end on synthetic mel" test that
/// `TODO.md` claimed this file already contained. Unlike
/// `tests/threading_parity.rs` (which pins bit-identical *attention*
/// output), this test drives the full public `transcribe_segmented` API —
/// mel, encoder, decoder, and segment/timestamp parsing — and checks real
/// properties of the result rather than discarding it.
///
/// Note: this test does **not** call `threading::set_thread_count`, so it
/// never interacts with the global-pool lifecycle covered by
/// `test_set_thread_count_pool_lifecycle` above, under either harness.
#[cfg(feature = "test-utils")]
#[test]
fn test_transcribe_end_to_end_on_synthetic_mel() {
    let model = common::shared_model();
    let audio = common::synthetic_sine(3.0);
    let audio_duration_secs = audio.len() as f32 / 16_000.0;

    let opts = oxiwhisper::TranscribeOptions {
        // `None` exercises the auto language-detection path through the
        // encoder in addition to the decoder/segment-parsing path.
        language: None,
        timestamps: true,
        ..Default::default()
    };

    let result1 = model
        .transcribe_segmented(&audio, &opts)
        .expect("end-to-end transcribe_segmented on synthetic mel must succeed");
    let result2 = model
        .transcribe_segmented(&audio, &opts)
        .expect("end-to-end transcribe_segmented on synthetic mel must succeed (2nd run)");

    // The synthetic model's weights and the input audio are both fixed, so
    // two runs through the (possibly rayon-parallelized) mel -> encoder ->
    // decoder pipeline must be bit-for-bit identical. A data race or an
    // off-by-one in the per-head parallel attention loop -- the exact class
    // of bug this smoke test exists to catch -- would make this comparison
    // fail or flake.
    assert_eq!(
        result1, result2,
        "transcribe_segmented must be deterministic across repeated runs on \
         identical input"
    );

    // Auto language-detection was requested (`language: None`); the decoder
    // must always resolve *some* language code in that case, never leave it
    // unresolved.
    let detected = result1
        .language
        .as_deref()
        .expect("language must be auto-detected when opts.language is None");
    assert!(
        detected.len() >= 2 && detected.chars().all(|c| c.is_ascii_lowercase()),
        "detected language code should look like a BCP-47 code, got {detected:?}"
    );

    // Non-vacuity guard: the designed synthetic model emits paired timestamp
    // tokens with real text under `timestamps: true`, so the segment list must
    // be non-empty — otherwise the per-segment invariant loop and the ordering
    // check below would pass without examining anything.
    assert!(
        !result1.segments.is_empty(),
        "timestamps=true must produce at least one segment on the designed model"
    );

    // Structural invariants that must hold for every emitted segment,
    // regardless of what text the untrained synthetic model happens to
    // decode.
    for segment in &result1.segments {
        assert!(
            segment.start.is_finite() && segment.start >= 0.0,
            "segment start must be a finite, non-negative time: {segment:?}"
        );
        assert!(
            segment.end.is_finite() && segment.end >= segment.start,
            "segment end must be finite and not precede its start: {segment:?}"
        );
        assert!(
            segment.end <= audio_duration_secs + 1.0,
            "segment end must not run past the input audio duration (+1s slack \
             for coarse timestamp-token bucketing): {segment:?}"
        );
        assert!(
            segment.confidence.is_finite(),
            "segment confidence must be a finite log-probability: {segment:?}"
        );
    }
    assert!(
        result1
            .segments
            .windows(2)
            .all(|pair| pair[0].start <= pair[1].start),
        "segments must be ordered by non-decreasing start time: {:?}",
        result1.segments
    );

    // Sanity bound against runaway/corrupted output (mirrors the bound used
    // by `whisper_model::tests::test_decoder_roundtrip`).
    assert!(
        result1.text.len() <= 10_000,
        "output text suspiciously long: {} bytes",
        result1.text.len()
    );
}
