//! End-to-end integration test against a **real** Whisper checkpoint.
//!
//! The synthetic fixtures in `src/test_utils.rs` exercise the plumbing but not
//! the numerics: their weights are designed so the greedy walk is predetermined.
//! This test runs the actual `ggml-tiny.bin` graph over real speech and asserts
//! on the transcript, the segment timeline and the word timings — the three
//! things the synthetic model cannot check.
//!
//! # Running it
//!
//! The test is skipped (and prints why) unless `OXIWHISPER_TEST_MODEL` points at
//! a Whisper checkpoint:
//!
//! ```text
//! mkdir -p models
//! curl -L -o models/ggml-tiny.bin \
//!   https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin
//! OXIWHISPER_TEST_MODEL=models/ggml-tiny.bin cargo nextest run --all-features real_model
//! ```
//!
//! `models/` is gitignored; the 77 MB checkpoint is never committed.
//!
//! The audio is `samples/jfk.wav` (the whisper.cpp sample, ~11 s of 16 kHz mono
//! speech), zero-padded to the full 30 s Whisper window as every caller must.

use std::path::{Path, PathBuf};

use oxiwhisper::{TranscribeOptions, WhisperModel, audio};

/// Environment variable holding the path to a real GGML/GGUF Whisper model.
const MODEL_ENV: &str = "OXIWHISPER_TEST_MODEL";

/// Whisper's fixed analysis window.
const WINDOW_SAMPLES: usize = 16_000 * 30;

/// Resolve the model path from the environment, or `None` when the test should
/// be skipped.
fn model_path() -> Option<PathBuf> {
    let raw = std::env::var_os(MODEL_ENV)?;
    let path = PathBuf::from(raw);
    if path.as_os_str().is_empty() {
        return None;
    }
    assert!(
        path.is_file(),
        "{MODEL_ENV} points at {}, which is not a readable file",
        path.display()
    );
    Some(path)
}

/// Absolute path to `samples/jfk.wav` inside this crate.
fn sample_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("samples/jfk.wav")
}

/// Load the sample and return `(padded_to_30s, true_duration_seconds)`.
fn load_padded_sample() -> (Vec<f32>, f32) {
    let path = sample_path();
    let pcm =
        audio::load_wav(&path).unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display()));
    assert!(
        !pcm.is_empty(),
        "{} decoded to zero samples",
        path.display()
    );
    let duration = pcm.len() as f32 / 16_000.0;
    assert!(
        (9.0..13.0).contains(&duration),
        "samples/jfk.wav should be ~11 s, got {duration:.2} s"
    );

    let mut padded = pcm;
    padded.resize(WINDOW_SAMPLES, 0.0);
    (padded, duration)
}

fn skip(reason: &str) {
    eprintln!("SKIP: {reason} (set {MODEL_ENV} to a real Whisper checkpoint)");
}

#[test]
fn test_real_model_transcribes_jfk_with_segments() {
    let Some(model_path) = model_path() else {
        skip("no real model supplied");
        return;
    };

    let (audio_30s, duration) = load_padded_sample();
    let model = WhisperModel::from_file(&model_path).expect("load real Whisper model");

    let opts = TranscribeOptions {
        language: Some("en"),
        timestamps: true,
        ..TranscribeOptions::default()
    };
    let result = model
        .transcribe_segmented(&audio_30s, &opts)
        .expect("transcription must succeed on real audio");

    // (a) A non-empty transcript containing plausible English words.
    let text = result.text.trim().to_lowercase();
    eprintln!("transcript: {:?}", result.text);
    assert!(!text.is_empty(), "transcript must not be empty");
    assert!(
        !text.contains('\u{FFFD}'),
        "transcript must not contain U+FFFD replacement characters: {text:?}"
    );
    // The clip is JFK's "ask not what your country can do for you…".
    let expected_words = ["ask", "country", "you"];
    for word in expected_words {
        assert!(
            text.contains(word),
            "expected the word {word:?} in the transcript, got {text:?}"
        );
    }

    // (b) At least one timed segment.
    assert!(
        !result.segments.is_empty(),
        "timestamps were requested, so at least one segment must be produced"
    );
    for seg in &result.segments {
        assert!(
            seg.start <= seg.end,
            "segment start must not exceed end: {seg:?}"
        );
        assert!(
            seg.start >= 0.0,
            "segment start must be non-negative: {seg:?}"
        );
        assert!(
            !seg.text.trim().is_empty(),
            "segments must carry text: {seg:?}"
        );
    }
    // Segments must be ordered.
    for pair in result.segments.windows(2) {
        assert!(
            pair[0].start <= pair[1].start,
            "segment starts must be non-decreasing: {:?} then {:?}",
            pair[0],
            pair[1]
        );
    }

    // (c) The final segment must end close to the real end of speech, not at
    //     the end of the 30 s zero padding and not at half the true time.
    let last_end = result
        .segments
        .last()
        .map(|s| s.end)
        .expect("checked non-empty above");
    eprintln!("last segment end: {last_end:.2}s (audio is {duration:.2}s)");
    assert!(
        (last_end - duration).abs() <= 1.5,
        "last segment ends at {last_end:.2}s but the audio is {duration:.2}s long \
         (tolerance 1.5s); segments: {:?}",
        result.segments
    );
}

#[test]
fn test_real_model_word_timestamps_span_the_clip() {
    let Some(model_path) = model_path() else {
        skip("no real model supplied");
        return;
    };

    let (audio_30s, duration) = load_padded_sample();
    let model = WhisperModel::from_file(&model_path).expect("load real Whisper model");

    let opts = TranscribeOptions {
        language: Some("en"),
        timestamps: true,
        word_timestamps: true,
        ..TranscribeOptions::default()
    };
    let transcript = model
        .transcribe_words(&audio_30s, &opts)
        .expect("word-level transcription must succeed");

    assert!(
        !transcript.words.is_empty(),
        "word timestamps were requested but no words were produced"
    );
    eprintln!(
        "words: {}",
        transcript
            .words
            .iter()
            .map(|w| format!("{}@{:.2}-{:.2}", w.word, w.start, w.end))
            .collect::<Vec<_>>()
            .join(" ")
    );

    // Text must be space-separated for English, not run together.
    assert!(
        transcript.text.contains(' '),
        "English word transcript must be space separated, got {:?}",
        transcript.text
    );

    for w in &transcript.words {
        assert!(w.start <= w.end, "word start must not exceed end: {w:?}");
        assert!(w.start >= 0.0, "word start must be non-negative: {w:?}");
        assert!(
            w.end <= 30.5,
            "word end must stay inside the 30 s window: {w:?}"
        );
    }
    for pair in transcript.words.windows(2) {
        assert!(
            pair[0].start <= pair[1].start,
            "word starts must be non-decreasing: {:?} then {:?}",
            pair[0],
            pair[1]
        );
    }

    // (d) The final word must land after 9 s. The DTW column stride is one
    //     *encoder* frame = 320 samples = 20 ms; using the 160-sample mel hop
    //     halved every timestamp and put the last word of this clip at 5.51 s.
    let last_end = transcript
        .words
        .last()
        .map(|w| w.end)
        .expect("checked non-empty above");
    eprintln!("last word ends at {last_end:.2}s (audio is {duration:.2}s)");
    assert!(
        last_end > 9.0,
        "last word ends at {last_end:.2}s; expected > 9.0s for a {duration:.2}s clip \
         (a value near half the audio length means the DTW hop is wrong)"
    );
}

/// Environment variable holding the path to a **quantized** GGML checkpoint
/// (`ggml-tiny.en-q8_0.bin`, `ggml-tiny-q5_1.bin`, …).
const QUANT_MODEL_ENV: &str = "OXIWHISPER_TEST_MODEL_QUANT";

#[test]
fn test_real_quantized_model_loads_and_transcribes() {
    let Some(raw) = std::env::var_os(QUANT_MODEL_ENV) else {
        eprintln!(
            "SKIP: no quantized model supplied (set {QUANT_MODEL_ENV} to e.g. \
             models/ggml-tiny.en-q8_0.bin or models/ggml-tiny-q5_1.bin)"
        );
        return;
    };
    let path = PathBuf::from(raw);
    assert!(
        path.is_file(),
        "{QUANT_MODEL_ENV} points at {}, which is not a readable file",
        path.display()
    );

    // Regression: the legacy GGML loader mapped `ggml_type` 3 to Q8_0 (3 is
    // Q4_1) and rejected 7 (Q5_1) and 8 (Q8_0) outright, so every quantized
    // whisper.cpp checkpoint failed with "Unsupported tensor dtype".
    let model = WhisperModel::from_file(&path)
        .unwrap_or_else(|e| panic!("quantized model {} must load: {e}", path.display()));

    let stats = model.model_stats();
    assert!(
        stats.quantized_params > 0,
        "a quantized checkpoint must retain quantized weights, stats={stats:?}"
    );

    let (audio_30s, _duration) = load_padded_sample();
    let opts = TranscribeOptions {
        language: Some("en"),
        timestamps: true,
        ..TranscribeOptions::default()
    };
    let result = model
        .transcribe_segmented(&audio_30s, &opts)
        .expect("quantized transcription must succeed");
    let text = result.text.trim().to_lowercase();
    eprintln!(
        "quantized transcript ({}): {:?}",
        path.display(),
        result.text
    );
    assert!(!text.is_empty(), "quantized transcript must not be empty");
    assert!(
        text.contains("country"),
        "expected the word \"country\" in the quantized transcript, got {text:?}"
    );
}

#[test]
fn test_real_model_vocabulary_preserves_multibyte_bytes() {
    let Some(model_path) = model_path() else {
        skip("no real model supplied");
        return;
    };

    use oxiwhisper::model::ModelData;
    use oxiwhisper::tokenizer;

    let md = ModelData::load(&model_path).expect("load real Whisper model");

    // The multilingual GGML vocabulary stores 50257 GPT-2 byte-level BPE
    // entries, 1476 of which are *fragments* of multi-byte characters and are
    // therefore not valid UTF-8 on their own. Loading them through
    // `String::from_utf8_lossy` replaced each with U+FFFD and made every kanji,
    // kana and hangul token unrecoverable.
    let fragments = md
        .vocab
        .iter()
        .filter(|e| std::str::from_utf8(e.as_bytes()).is_err())
        .count();
    eprintln!(
        "vocab: {} entries, {fragments} non-UTF-8 byte fragments",
        md.vocab.len()
    );
    assert!(
        fragments > 1000,
        "a real multilingual vocabulary must retain its raw byte fragments, found {fragments}"
    );
    // Tokens 162 / 116 / 233 hold the bytes E6 B8 8B — the kanji 渋 (U+6E0B).
    assert_eq!(md.vocab[162].as_bytes(), [0xE6]);
    assert_eq!(md.vocab[116].as_bytes(), [0xB8]);
    assert_eq!(md.vocab[233].as_bytes(), [0x8B]);
    assert_eq!(
        tokenizer::decode(&[162, 116, 233], &md.vocab),
        "\u{6E0B}",
        "the three byte-fallback tokens must decode to a single kanji"
    );

    // And in a segment context, with timestamps bracketing the text.
    let segments = tokenizer::parse_segments(&[50364, 162, 116, 233, 50414], &md.vocab);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].2, "\u{6E0B}");
}
