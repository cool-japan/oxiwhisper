# Changelog

All notable changes to this project will be documented in this file.

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
