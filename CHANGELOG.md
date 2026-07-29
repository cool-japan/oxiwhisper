# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - Unreleased

### Fixed
- **Japanese / CJK transcripts were mojibake.** The GGML loader decoded every
  vocabulary entry with `String::from_utf8_lossy` at load time. Whisper uses
  GPT-2 *byte-level* BPE, so a multi-byte character is routinely spelled by
  several tokens that are not valid UTF-8 on their own — 1476 of the 50257
  entries in `ggml-tiny.bin` were replaced by `U+FFFD`, making all kanji, kana
  and hangul unrecoverable. `VocabEntry` now stores **raw bytes**
  (`VocabEntry::bytes`, with `from_bytes` / `from_text` / `as_bytes` / `text()`
  / `is_special()` accessors), and `tokenizer::decode` / `parse_segments`
  concatenate bytes and run a single lossy UTF-8 conversion over the joined
  buffer. New `tokenizer::decode_bytes` / `decode_bytes_into` expose the
  byte-exact result. Mirrored in the GGUF and ONNX vocabulary builders.
  Regression test: BPE tokens `[162, 116, 233]` (bytes `E6 B8 8B`) now decode
  to the single kanji `渋` (U+6E0B).
- **Segment timestamps were inverted, producing empty transcripts.**
  `decode_utils::apply_timestamp_rules` treated a missing penultimate token as
  "was not a timestamp", so it forced a *second* timestamp immediately after the
  opening one instead of forcing text; it also masked the whole `[0, ts_begin)`
  range — including `<|endoftext|>` — when closing a segment, and never applied
  the strict-increase rule that prevents zero-length segments. 30 s-padded audio
  therefore decoded to nothing. The filter is now a faithful port of OpenAI's
  `ApplyTimestampRules`: the first sampled token must be a timestamp; a
  timestamp whose predecessor is a timestamp *or absent* must be followed by
  text; a timestamp preceded by text may only be followed by another timestamp
  or `<|endoftext|>`; the monotonic floor is inclusive only for the pair-closing
  partner. The synthetic fixture in `test_utils.rs`, which codified the wrong
  rule, was regenerated accordingly.
- **Word timestamps were reported at half their true value.**
  `word_timestamps` used the 160-sample mel hop as the DTW column stride, but a
  cross-attention column is one *encoder* frame — 320 samples / 20 ms, because
  the encoder's second convolution has stride 2. The last word of an 11.0 s clip
  came out at 5.51 s. The DTW is additionally restricted to the encoder frames
  up to the decoded closing timestamp (OpenAI's `num_frames // 2` slice), so the
  final word is no longer dragged out to the end of the 30 s zero padding.
- **`WordTimedTranscript::text` ran words together** (`"helloworld"`). Words are
  now re-joined with a space, except across boundaries that touch a script
  written without inter-word spaces (CJK, kana, Thai, fullwidth forms).
- **Word splitting collapsed Japanese into a single word.** The splitter keyed
  purely on ASCII leading spaces. `dtw::build_word_segments_bytes` (new,
  byte-based; `build_word_segments` is kept as a `&[String]` wrapper) now splits
  on UTF-8 character boundaries when no token carries a leading space, and never
  splits inside a character.
- **The mel front-end warped every frequency by 1.28x.** The 400-sample analysis
  window was zero-padded to 512 before the FFT while the filter bank still
  assumed 40 Hz bins, so a 1 kHz tone landed in the 1254 Hz band. The transform
  now runs on the exact 400-point window (`oxifft::rfft` handles non-power-of-two
  sizes), adds the `center=True` 200-sample reflect pre-pad, and pads the output
  to Whisper's fixed 3000-frame window — numerically identical to zero-padding
  the audio to 30 s. New numeric test asserts a 1 kHz tone peaks in the mel band
  centred near 1026 Hz.
- **The legacy GGML dtype table was wrong, so no quantized checkpoint loaded.**
  Dtype 3 was mapped to Q8_0 (3 is Q4_1) and 7/8 were rejected outright, so
  `ggml-tiny-q8_0.bin`, `ggml-tiny-q5_1.bin` and friends all failed with
  "Unsupported tensor dtype". The GGML and GGUF loaders now share one table
  (`QuantType::from_ggml_type`): 2 = Q4_0, 3 = Q4_1, 6 = Q5_0, 7 = Q5_1,
  8 = Q8_0; anything else (Q8_1, the K-quants, …) is rejected with a message
  naming the type.
- **`mel.rs` panicked on non-80-mel models.** An `assert_eq!(n_mels, 80)`
  reachable from the public `transcribe()` aborted the process for `large-v3`
  (128 mels). The channel count is now derived from the filter bank
  (`mel::n_mels_from_filters`) and a mis-shaped bank returns
  `OxiWhisperError::InvalidModel`.
- **GGUF models with a non-standard mel count silently transcribed silence.**
  `gguf::whisper::resolve_mel_filters` substituted an **all-zero** filter bank
  when it could not generate one; it now returns `InvalidModel`.

### Added
- **Q4_1 and Q5_1 quantization kernels** — block dequantizers
  (`dequantize_q4_1`, `dequantize_q5_1`, plus the `dequantize(data, n, qtype)`
  dispatcher), dot products (`dot_q4_1`, `dot_q5_1` and their `_fast` shims) and
  quantizers (`quantize_block_q4_1`, `quantize_to_q4_1`, `quantize_block_q5_1`,
  `quantize_to_q5_1`). `quantize_tensor` covers all five types.
- **Quantized token-embedding support.** `decoder.token_embedding.weight` is the
  largest tensor in the model and is kept in block form by every quantized
  checkpoint, which made the decoder fail with
  `Missing tensor: decoder.token_embedding.weight`. The embedding lookup now
  gathers rows from either an f32 or a quantized table.
- **`mel::log_mel_spectrogram_unpadded`** — the same front-end without the
  30 s padding, for analysis paths (embedding extraction, diarization) where the
  full window would be pure overhead. `WhisperModel::encoder_output` uses it.
- **`mel::WHISPER_N_FRAMES`** (3000) and **`mel::n_mels_from_filters`**.
- **Real-model integration test** (`tests/real_model.rs`), skipped cleanly unless
  `OXIWHISPER_TEST_MODEL` points at a checkpoint (and
  `OXIWHISPER_TEST_MODEL_QUANT` for the quantized variant). Transcribes
  `samples/jfk.wav` zero-padded to 30 s and asserts a plausible English
  transcript, at least one ordered segment, a final segment end within 1.5 s of
  the true 11 s duration, a last word after 9 s, and byte-exact multi-token
  kanji decoding against the real vocabulary.
- **`samples/jfk.wav`** — the whisper.cpp sample clip used by that test.

### Changed
- **Breaking:** `VocabEntry { text: String }` is now `VocabEntry { bytes: Vec<u8> }`.
  Use `VocabEntry::from_text(..)` to construct and `entry.text()` (a
  `Cow<'_, str>`) for a lossy per-entry view.
- **Breaking:** `mel::log_mel_spectrogram` returns
  `Result<Vec<f32>, OxiWhisperError>` and always emits the full 3000-frame
  window. `WhisperModel::mel_spectrogram` returns
  `Result<tensor::Tensor, OxiWhisperError>` for the same reason.
- **Breaking:** `mel::n_frames_for_samples` now returns `n_samples / 160`
  (clamped to `1..=3000`), matching `torch.stft(center=True)` with the trailing
  frame dropped, instead of `ceil(n / 160) + 1`.

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
