# Changelog

All notable changes to this project will be documented in this file.

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
