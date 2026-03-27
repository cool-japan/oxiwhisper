# oxiwhisper — TODO / Roadmap

Pure Rust Whisper inference engine. Zero C/C++ deps. GGML model format.

---

## Performance

- [x] **Q4_0 / Q8_0 quantized weight loading** — dequantize-on-the-fly in matmul kernel via
      `linear_quantized()` and `linear_auto()` dispatch; 2-4x memory reduction for quantized models
- [x] **Buffer reuse allocator** — `InferenceBuffer` struct with `transcribe_with_buffer()`;
      in-place tensor ops (`gelu_inplace`, `softmax_inplace`, `layer_norm_inplace`, `add_inplace`)
- [x] **WASM `simd128` feature path** — `#[cfg(target_feature="simd128")]` for elementwise
      ops (GELU, softmax, layer norm, add, add_bias); auto-detected at compile time
- [x] **Profile encoder attention at seq=1500** — `examples/profile_attention.rs` benchmarks
      tiled matmul vs sgemm; result: sgemm is ~14-17x faster (105 vs 6.5 GFLOPS)

---

## Decoder Quality

- [x] **Beam search** — configurable width via `TranscribeOptions::beam_width`;
      keep top-k hypotheses per step, merge at EOS
- [x] **Temperature sampling** — `top_k` and `top_p` (nucleus) sampling for diversity and
      hallucination reduction; configurable via `TranscribeOptions`
- [x] **Timestamp token support** — emit `<|t.xx|>` tokens as word-level alignment;
      return `Vec<Segment { text, start, end }>` via `transcribe_segmented()` and `parse_segments()`
- [x] **Language auto-detection** — run first pass with no language token, argmax over the
      99 language logits to identify source language automatically when `language: None`

---

## API & Ergonomics

- [x] **`TranscribeOptions` struct** — consolidate all decoding parameters
- [x] **Streaming / chunked API** — `StreamTranscriber` with `push_audio()`, `next_segment()`,
      `finish()` for real-time partial results on long audio
- [x] **Batch API** — `transcribe_batch()` processes multiple audio clips independently
- [x] **Structured error enum** — `OxiWhisperError { Io, InvalidModel, ShapeMismatch, InferenceFailed }`
- [x] **`log` crate feature flag** — timing prints gated behind `#[cfg(feature = "timing")]`
- [x] **Timed transcription** — `transcribe_timed()` returns per-phase breakdown (mel, encoder, decoder)

---

## Testing

- [x] **Encoder integration test** — synthetic GGML model generator (`test_utils.rs`),
      load model, transcribe 1s silence, verify no crash and correct model info
- [x] **Decoder round-trip test** — synthetic model, transcribe 440 Hz sine wave,
      verify segmented output structure is valid
- [x] **Silent audio test** — mel spectrogram silence validation, VAD no-speech detection,
      split point edge cases
- [x] **Benchmark example** — `examples/bench.rs` with wall-clock RTF reporting,
      per-phase timing breakdown, multi-iteration min/avg/max

---

## crates.io Publish Prep

- [x] Add `README.md` with quick-start example and supported model list
- [x] Add `keywords`, `categories`, `repository`, `documentation` to `Cargo.toml`
- [x] `cargo doc --no-deps` passes with zero warnings
- [x] `cargo publish --dry-run` succeeds

---

## v0.2 — Performance Deep Dive

- [x] **OxiFFT integration** — replaced custom radix-2 FFT (`fft.rs`) with OxiFFT;
      eliminates 138M redundant trig calls per inference via twiddle factor caching
- [x] **Mel spectrogram optimization** — pre-compute Hann window once (saves 1.2M trig calls),
      pre-allocate FFT buffer outside per-frame loop (saves 12.3MB alloc/dealloc per 30s)
- [x] **Zero-copy tensor reshape** — `reshape_inplace()` avoids cloning data vector;
      2-5MB saved per reshape in attention hot paths
- [x] **In-place ops wiring** — use existing `gelu_inplace`, `softmax_inplace`, `add_inplace`,
      `layer_norm_inplace` in encoder/decoder/attention forward passes
- [x] **Beam search COW KV cache** — `Arc<Vec<f32>>` copy-on-write for `LayerKVCache`;
      clone becomes Arc ref bump, `Arc::make_mut()` on append; ~4.5GB alloc savings for beam search

---

## v0.2 — API Enhancements

- [x] **SRT/VTT subtitle export** — `subtitle.rs` with `to_srt()` and `to_vtt()`;
      convenience methods `transcribe_to_srt()`, `transcribe_to_vtt()`
- [x] **Token-level confidence** — `token_probs: Vec<f32>` in `DecodeResult`;
      collect log-prob of chosen token at each decode step (greedy, beam, sampling)
- [x] **Model quantization tools** — `quantize_to_q4_0()`, `quantize_to_q8_0()` in quantize.rs;
      block-wise scale computation, `quantize_tensor()` utility
- [x] **Input validation** — validate `TranscribeOptions` (beam_width >= 1, temperature >= 0,
      0 < top_p <= 1); return `ConfigError` instead of silent fallback
- [x] **Error enrichment** — add `ConfigError(String)`, `AudioFormatError(String)` to
      `OxiWhisperError`; convert `linear_auto` panic to `Result`

---

## v0.2 — Testing & Robustness

- [x] **Encoder unit tests** — shape test, no-NaN test, short audio edge case
- [x] **Model loading robustness** — truncated file, wrong magic bytes, missing tensor tests
- [x] **KV cache unit tests** — test `LayerKVCache` new/append/k_head/v_head with known data;
      verify Arc COW behavior (clone + append only clones on write)

---

## v0.3 — Advanced Decoding Quality

- [x] **Initial prompt support** — `initial_prompt: Option<&str>` in `TranscribeOptions`;
      tokenize and prepend to decoder prompt for domain-specific vocabulary hints
- [x] **Suppress tokens** — `suppress_tokens: Option<&[u32]>` to block specific tokens;
      set logits to NEG_INFINITY before argmax/sampling
- [x] **No-repeat-ngram penalty** — `no_repeat_ngram_size: usize` prevents repeated n-grams;
      eliminates "the the the..." hallucination patterns
- [x] **Compression ratio filtering** — character-level entropy to detect low-entropy hallucinated output;
      marks suspect segments with `is_hallucination: bool`

---

## v0.3 — Performance

- [x] **sgemm attention kernels** — replaced manual triple-nested loops in attention QK^T and
      scores@V with `matrixmultiply::sgemm`; eliminated `transpose_last_two` via stride args
- [x] **AVX2 GEMV kernel** — `#[cfg(target_arch = "x86_64")]` explicit `_mm256_fmadd_ps` dot product
      for batch=1 decoder steps
- [x] **NEON GEMV kernel** — `#[cfg(target_arch = "aarch64")]` `vfmaq_f32` dot product for ARM
- [x] **Quantized SIMD dot products** — AVX2/NEON variants of `dot_q8_0` with dispatch via `dot_q8_0_fast`

---

## v0.3 — Code Quality

- [x] **Split decoder.rs** — extracted beam search into `beam_search.rs`, helpers into `decode_utils.rs`;
      decoder.rs reduced from ~1,350 to ~1,006 lines
- [x] **PartialEq on result types** — derive `PartialEq` on `Segment`, `TranscribeResult`, `TranscribeTiming`
- [x] **Streaming example** — `examples/streaming.rs` demonstrating `StreamTranscriber` API
- [x] **Batch example** — `examples/batch_transcribe.rs` demonstrating `transcribe_batch()`

---

## v0.4 — API & Ecosystem

- [x] **Encoder output API** — `WhisperModel::encoder_output()` returns `[seq_len, d_model]` tensor;
      enables embedding extraction, similarity search, fine-tuning pipelines
- [x] **Mel spectrogram API** — `WhisperModel::mel_spectrogram()` returns reusable `[n_mels, n_frames]`;
      enables audio analysis and visualization
- [x] **Model stats API** — `model_stats()` returns `ModelStats` with param counts, memory, quantization info
- [x] **Serde feature** — optional `serde` feature for JSON serialization of `TranscribeResult`,
      `Segment`, `ModelInfo`, `ModelStats`; `to_json()` convenience function
- [x] **Simple transcribe example** — `examples/transcribe.rs` CLI for single-file transcription
      with `--srt`, `--vtt`, `--timestamps` output options
- [x] **Thread-safety documentation** — Send/Sync compile-time assertions for WhisperModel,
      TranscribeOptions, TranscribeResult

---

## v0.4 — Testing & Robustness

- [x] **Beam search tests** — 8 tests: struct creation, clone independence, done propagation,
      normalized score, integration test with synthetic model
- [x] **Attention tests expansion** — 6 new tests: sgemm QK^T correctness, causal mask,
      scale factor, cross-attention lengths, auto matches regular
- [x] **Decode utils tests** — 8 new tests: numerical stability, uniform log_softmax,
      top_k ordering, argmax single element, token_log_prob out of range
- [x] **Q4_0 SIMD kernel** — AVX2/NEON variant of `dot_q4_0` with nibble unpacking;
      dispatch via `dot_q4_0_fast`

---

## v0.5 — Code Quality

- [x] **Refactor lib.rs** — extracted types into `types.rs`, streaming into `stream.rs`,
      hallucination detection into `hallucination.rs`; lib.rs reduced from ~1,988 to ~1,309 lines

---

## v0.5 — Feature Parity

- [x] **Word-level timestamps (DTW)** — `dtw.rs` module with `align_tokens_dtw()` and
      `build_word_segments()`; `WordSegment` struct with per-word start/end/confidence
- [x] **Q5_0 quantization** — 5-bit format (22 bytes/block); dequantize, quantize, dot product,
      AVX2/NEON SIMD kernels; model loading for GGML dtype 6
- [x] **Previous context conditioning** — `previous_tokens` in `TranscribeOptions`;
      cross-chunk coherence via `initial_prompt` chaining in `transcribe_long`

---

## v0.5 — Infrastructure & Quality

- [x] **Criterion benchmarks** — `benches/transcribe.rs` with mel spectrogram, dot product,
      quantized dot, linear layer, and tensor ops benchmark groups
- [x] **Adaptive VAD** — noise floor estimation (10th percentile RMS); adaptive threshold
      via `noise_margin`; `transcribe_long_with_vad()` with custom `VadConfig`
