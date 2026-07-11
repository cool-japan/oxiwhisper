# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - Unreleased

## [0.1.2] - 2026-07-11

### Added
- **Speaker diarization** (new `diarization` feature, off by default): answers *who spoke when* via `WhisperModel::diarize` / `diarize_with_embedder`, and *who spoke what* via `WhisperModel::transcribe_with_speakers` / `transcribe_with_speakers_using_embedder`. Offline embedding-clustering pipeline: energy VAD -> uniform sub-segmentation -> per-window speaker embedding -> cosine-affinity clustering (agglomerative or spectral, with speaker-count estimation) -> resegmentation -> fusion with word timestamps. `SpeakerEmbedder` trait with two backends: `WhisperEncoderEmbedder` (baseline) and `EcapaOnnx` (ECAPA-TDNN / x-vector via `oxionnx`, behind the `onnx` feature, user-supplied checkpoint). NIST **RTTM** export (`write_rttm`, `rttm_string`) and `[SPEAKER_k]`-labeled transcripts; **DER/JER** evaluation with Hungarian speaker mapping (`der`, `jer`, `parse_rttm`); `examples/diarize.rs` CLI. Clustering and the symmetric eigensolver are implemented **inline** (no `ndarray`/`nalgebra`). **Honest limitation:** the built-in `WhisperEncoderEmbedder` is a low-accuracy baseline — Whisper's encoder is trained to be speaker-*invariant* — provided for tests and no-model demos only; production accuracy requires an external pretrained ECAPA-TDNN / x-vector ONNX model, and overlapped speech is attributed to a single speaker.
- **Word-level timestamps** (`WhisperModel::transcribe_words`, `WordTimedTranscript`,
  `WordSegment`): set `word_timestamps: true` on `TranscribeOptions` (or call
  `transcribe_words`) to receive per-word start/end times aligned via cross-attention DTW;
  the existing fully-tested `dtw.rs` (`align_tokens_dp_dtw`, `build_word_segments`) is now
  wired into the public API; greedy and temperature-sampling paths supported; beam search
  (`beam_width > 1`) returns `ConfigError`
- **`TranscribeOptions::word_timestamps`** (default `false`) — opt-in flag; zero overhead when
  disabled (no cross-attention buffers allocated)
- **`TranscribeOptions::no_speech_threshold`** (default `0.6`) — combined OpenAI-style silence
  gate: if `no_speech_prob > no_speech_threshold` AND `avg_logprob < logprob_threshold`, the
  segment is returned empty (silence detected); `no_speech_prob` now captured from the raw
  prefill logits (before any suppression) for all three samplers
- **`TranscribeOptions::suppress_blank`** (default `true`) — suppress the leading-space "blank"
  token and EOT at decode step 0 (matches OpenAI's `SuppressBlank`); prevents transcripts from
  opening with whitespace or terminating immediately
- **`ApplyTimestampRules`** applied automatically when `timestamps == true`: (a) suppress
  `<|notimestamps|>` always; (b) force text after a complete timestamp pair, force another
  timestamp after a lone one, monotonic floor; (c) force timestamp when timestamp probability
  mass dominates best text token — full OpenAI parity, applied inside all three samplers
- **`DecodeResult::cross_attention`** — flat `[n_tokens * enc_len]` head- and layer-averaged
  cross-attention matrix; `None` by default (zero overhead when not capturing)
- **`DecodeResult::enc_len`** — encoder frame count (required for DTW)
- **`DecodeResult::no_speech_prob`** — `<|nospeech|>` probability at the first decoded position
- **Translation task** (`Task { Transcribe, Translate }` enum + `TranscribeOptions::task` field):
  set `task: Task::Translate` to decode any-language audio into English via the Whisper
  `<|translate|>` (50358) token; default `Task::Transcribe` is a no-op for existing callers
- **Temperature fallback decoding** (`fallback_temperatures: &[f32]` + `logprob_threshold: f32`
  fields on `TranscribeOptions`): OpenAI-style robustness — when a decode attempt has low average
  log-probability or degenerate char-entropy, the decoder retries at the next temperature in the
  schedule; the first acceptable result is returned, or the last attempt if none qualify; an empty
  schedule (default) preserves the existing single-dispatch behaviour with zero overhead
- **Progress callbacks** for long-audio transcription:
  `transcribe_long_with_progress`, `transcribe_long_segmented_with_progress`, and
  `transcribe_long_with_vad_with_progress` each accept `FnMut(chunk_index: usize, total: usize)`;
  the original methods now delegate via a no-op closure, preserving their signatures

### Changed
- **Symphonia 0.6 API migration**: updated `decode_with_symphonia` in `src/audio.rs` to the
  Symphonia 0.6 API; `SampleBuffer` replaced by `GenericAudioBufferRef::copy_to_vec_interleaved`,
  `CODEC_TYPE_NULL`/`DecoderOptions` replaced by `CodecParameters::Audio`/`AudioDecoderOptions`,
  `Probe::format()` replaced by `Probe::probe()`, `CodecRegistry::make()` replaced by
  `make_audio_decoder()`, `Hint` import path corrected to `formats::probe::Hint`, and
  `next_packet()` now handles `Ok(None)` for end-of-stream; zero API changes for callers
- **Behavior change (OpenAI parity defaults ON)**: `suppress_blank=true` and
  `no_speech_threshold=0.6` are active by default. Existing callers that previously relied on
  the leading-space token or EOT being selectable at step 0 should set `suppress_blank=false`.
  The `ApplyTimestampRules` filter is applied whenever `timestamps=true` (no opt-out needed —
  it only activates timestamp-related invariants and has no effect when timestamps are disabled)
- **Sampler return arity**: `decode_greedy`, `decode_sample`, `decode_beam` now return
  `(tokens, probs, no_speech_prob)` 3-tuple (internal API only; no public API change)
- **Removed crude no-speech checks**: the argmax-equals-no_speech early-return in greedy and
  beam was deleted in favour of the proper `no_speech_prob` gate applied post-decode

### Fixed
- **Beam search decoding (`beam_width > 1`) could return an empty transcription for audible
  speech** when timestamps were enabled: if a stop/EOT token appeared among the top-k seed
  candidates, it was seeded as an already-`done` beam with zero tokens, and that beam's
  normalized score (`score / 1`) could unfairly outscore every real hypothesis in the final beam
  comparison, producing empty output for audible input. Seeding now over-samples the top-k
  candidates and filters out stop tokens before seeding, falling back to empty output only when
  every top candidate is genuinely a stop token (true silence); pinned by a new regression test
  (`src/beam_search.rs`, `decode_beam`)
- **Malformed or truncated GGUF model files could panic (integer overflow) or attempt an
  unbounded allocation** instead of failing cleanly: tensor element counts and byte sizes are now
  computed with checked arithmetic, and each tensor's declared data range is validated against
  the actual file length before its buffer is allocated, so a corrupted or adversarially-crafted
  `.gguf` file is now rejected with `OxiWhisperError::InvalidModel` instead of crashing or
  attempting a multi-gigabyte allocation; an out-of-range `general.alignment` value no longer
  panics either (`src/gguf/parse.rs`, `src/gguf/spec.rs`)

## [0.1.1] - 2026-04-26

### Added
- **GGUF format support**: `WhisperModel::from_file()` and `from_file_mmap()` auto-detect magic bytes and transparently accept both legacy GGML (`ggml-*.bin`) and modern GGUF (`*.gguf`) model files; no API change required (`src/model.rs`)
- **`parallel` feature** (optional, not in defaults): per-head parallelism in decoder SDPA loops and encoder attention via rayon; enable with `features = ["parallel"]`; `threading::set_thread_count(n)` helper configures the rayon global pool; disabled by default to keep WASM and single-threaded builds unaffected (`src/threading.rs`, `src/decoder/sdpa.rs`, `src/encoder.rs`)
- `WhisperModel::from_file_mmap()` — memory-mapped GGML model loading via `memmap2`; lower peak RSS for large models (`src/model.rs`, `src/lib.rs`)
- `align_tokens_monotonic_peak()` — renamed from `align_tokens_dtw()`; deprecated shim preserves SemVer for 0.1.x callers (`src/dtw.rs`)
- `load_audio()` — magic-byte auto-detecting audio loader; FLAC/OGG/MP3/AAC/Opus support via `audio-flac`/`audio-ogg`/`audio-mp3`/`audio-aac`/`audio-opus` features (`src/audio.rs`)
- `KvCacheDtype { F32, VHalf, KvHalf }` — optional f16 KV-cache storage for ~25–50% memory savings (`src/decoder.rs`, `src/types.rs`)
- Integration tests directory `tests/` with 5 binaries exercising full public API; new `test-utils` feature gates the synthetic model generator
- `quantize.rs` refactored into `src/quantize/` directory (7 modules, each <500 lines); all public API preserved

### Changed
- `align_tokens_dtw` is no longer deprecated; it now implements true Sakoe-Chiba-banded dynamic programming DTW with traceback, replacing the previous monotonic-peak approximation; timestamps produced are smoother and more accurate for noisy attention matrices (semantic change, `src/dtw.rs`)
- Word-timestamp feature graduated from Alpha to Stable; algorithm correctly documented as monotonic-peak alignment (not DP-DTW)
- Decoder SDPA hot-path migrated from scalar triple-loops to `matrixmultiply::sgemm`; encoder attention scratch allocations hoisted out of head loops

### Fixed
- `parse_json_string` now correctly decodes UTF-16 surrogate pairs (emoji, Mathematical Alphanumeric Symbols, CJK Extension B) from `tokenizer.json`; previously, lone high surrogates were silently dropped; a lone `\uD800` now returns `Err` instead of being ignored (`src/tokenizer.rs`)

### Known Issues
- When the `onnx` feature is enabled, `Cargo.lock` contains both `oxifft 0.2.0` (transitive via `oxionnx-ops 0.1.2`) and `oxifft 0.3.0` (direct dependency). This is a transient state until `oxionnx-ops` releases a version that upgrades to `oxifft 0.3+`. The duplicate has zero impact when the `onnx` feature is disabled (the default). Track: https://github.com/cool-japan/oxionnx

## [0.1.0] - 2026-03-27

### Added

#### Core Inference
- Pure Rust Whisper inference engine with zero C/C++ dependencies
- GGML model format loading with Q4_0, Q5_0, and Q8_0 quantized weight support
- ONNX model loading via optional `onnx` feature (oxionnx integration)
- OxiFFT-powered mel spectrogram computation with pre-computed Hann window
- Full encoder-decoder transformer pipeline with KV cache

#### Decoding
- Greedy decoding, beam search (configurable width), and temperature sampling
- Top-k and top-p (nucleus) filtering
- Language auto-detection (99 languages)
- Timestamp token support with segment-level timing
- Word-level timestamps via DTW cross-attention alignment (`dtw` module)
- Initial prompt conditioning for domain-specific vocabulary
- Suppress tokens to block specific token IDs
- No-repeat-ngram penalty to prevent hallucination loops
- Compression ratio filtering for hallucination detection
- Previous context conditioning for cross-chunk coherence

#### Performance
- SIMD-accelerated GEMV kernels: AVX2 (x86_64) and NEON (aarch64)
- SIMD-accelerated quantized dot products for Q4_0, Q5_0, Q8_0
- `matrixmultiply::sgemm` for attention QK^T and scores@V (stride-based K^T)
- Arc copy-on-write KV cache for beam search (~4.5GB allocation savings)
- Zero-copy tensor reshape (`reshape_inplace`)
- In-place operations wired to encoder/decoder/attention hot paths
- WASM simd128 feature path for WebAssembly targets
- Buffer reuse allocator (`InferenceBuffer`)

#### API
- `WhisperModel::transcribe()`, `transcribe_segmented()`, `transcribe_timed()`
- `WhisperModel::transcribe_long()`, `transcribe_long_segmented()` for audio > 30s
- `WhisperModel::transcribe_long_with_vad()` with custom VAD configuration
- `WhisperModel::transcribe_batch()` for multiple audio clips
- `WhisperModel::transcribe_to_srt()`, `transcribe_to_vtt()` subtitle export
- `WhisperModel::stream()` returning `StreamTranscriber` for real-time processing
- `WhisperModel::encoder_output()` for embedding extraction
- `WhisperModel::mel_spectrogram()` for audio analysis
- `WhisperModel::model_stats()` for memory/parameter statistics
- `TranscribeOptions` with beam_width, temperature, top_k, top_p, timestamps,
  initial_prompt, suppress_tokens, no_repeat_ngram_size, compression_ratio_threshold,
  previous_tokens
- Input validation with `ConfigError` and `AudioFormatError` error types
- Token-level confidence (`token_probs`) and segment confidence scores
- Thread-safe `WhisperModel` (Send + Sync)

#### Audio
- Pure Rust WAV parser (PCM 8/16/24/32-bit, IEEE float)
- Multi-channel downmix to mono with linear resampling to 16 kHz
- Voice Activity Detection (RMS energy-based) with adaptive thresholding
- VAD-aware audio chunking for long transcriptions

#### Tooling
- Model quantization tools: `quantize_to_q4_0()`, `quantize_to_q5_0()`, `quantize_to_q8_0()`
- SRT and WebVTT subtitle export (`subtitle` module)
- Optional `serde` feature for JSON serialization of results
- Criterion benchmarks for mel, encoder, decoder, dot products
- 10 examples: transcribe, streaming, batch, bench, profile_attention, etc.

#### Quality
- 278 tests across 25 modules
- Zero clippy warnings, zero doc warnings
- All files under 2000 lines
- No unwrap() in production code
