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

---

## v0.6 — Cleanup Observations

- [x] **A1 — Cargo.toml target ordering (transcribe example/bench colocation)** (planned 2026-04-25)
  - **Goal:** Single canonically-positioned `[[example]] name = "transcribe"` block colocated with other example blocks; `cargo build --examples` and `cargo bench --no-run` both work; no apparent duplication in manifest.
  - **Design:** The misplaced `[[example]] name = "transcribe"` block (currently after `[dev-dependencies]` and `[[bench]]`) moves to top of the example list. Final order: `transcribe`, `bench`, `bench2`, `check_shapes`, `check_shapes2`, `test_voice`, `profile_attention`, `onnx_transcribe`, `streaming`, `batch_transcribe`, then `[dev-dependencies]`, then `[[bench]] transcribe` at end. No semantic change.
  - **Files:** `Cargo.toml`
  - **Prerequisites:** None
  - **Tests:** `cargo build --examples --features onnx`; `cargo bench --no-run`; `cargo metadata` target set-equal before/after
  - **Risk:** None substantive

- [x] **A2 — Document `oxifft 0.2.0/0.3.0` transitive duplication** (planned 2026-04-25)
  - **Goal:** Duplication documented as known-acceptable in `CHANGELOG.md` with a tracking comment in `Cargo.toml`. Cannot resolve upstream in this cycle (oxionnx 0.1.2 is the latest; `oxionnx-ops 0.1.2` pins `oxifft ^0.2.0`).
  - **Design:** Verify with `cargo tree --duplicates --features onnx`; add a "Known Issues" entry to CHANGELOG.md under v0.1.1; add tracking comment next to the `oxionnx` line in Cargo.toml.
  - **Files:** `Cargo.toml`, `CHANGELOG.md`
  - **Prerequisites:** None
  - **Tests:** `cargo tree --duplicates --features onnx` shows oxifft 0.2.0; `cargo tree --duplicates --no-default-features` shows no oxifft duplicate
  - **Risk:** Low; comment includes tracking link

- [x] **A3 — Preemptively split `quantize.rs` (1986L → 7 modules, each <500L)** (planned 2026-04-25)
  - **Goal:** `src/quantize.rs` becomes thin re-export facade (~80L) plus `src/quantize/*.rs` modules. Public API unchanged — every existing import path still works. All 278 tests still pass. New `tests/quantize_api_stability.rs` catches visibility regressions.
  - **Design:** Operation-axis split: `types.rs` (~100L), `dequant.rs` (~110L), `quant.rs` (~270L), `dot_scalar.rs` (~85L), `dot_simd_x86.rs` (~265L, cfg x86_64), `dot_simd_neon.rs` (~140L, cfg aarch64), `dot_dispatch.rs` (~80L), `tests.rs` (split into ≤500L chunks). Facade re-exports all public symbols. Hand-split preferred over auto-splitrs for this arch-cfg-gated layout.
  - **Files:** Replace `src/quantize.rs` (1986L) with facade + `src/quantize/*.rs`; add `tests/quantize_api_stability.rs`
  - **Prerequisites:** splitrs already installed; no new deps
  - **Tests:** `cargo nextest run --all-features`; cross-arch builds (`aarch64-apple-darwin`, `wasm32-unknown-unknown --no-default-features`); clippy; `wc -l` max <1500L per file
  - **Risk:** Test sectioning may break shared helpers — keep helpers in single `tests.rs`

---

## v0.6 — Roadmap

- [x] **B1 — `WhisperModel::from_file_mmap()` via `memmap2`** (planned 2026-04-25)
  - **Goal:** New constructor `from_file_mmap(path: &Path) -> Result<Self, OxiWhisperError>` parses GGML via mmap. Same `WhisperModel` shape returned. Lower peak RSS on large models. Existing `from_file()` unchanged.
  - **Design:** Streaming-mmap (option b): map file as `&[u8]`, run existing parser via `Cursor<&[u8]>`. Tensor data still copies into owned Vecs. Refactor `ModelData::load()` into `load_from_reader<R: Read>()` + 4-line wrapper. Add `load_mmap()` using `memmap2::Mmap::map()` + `Advice::Sequential` on Unix. Add `WhisperModel::from_file_mmap()` in `lib.rs`. Document SIGBUS risk. No new error variant needed (mmap failures → `InvalidModel`).
  - **Files:** `Cargo.toml` (`memmap2 = "0.9"`), `src/model.rs`, `src/lib.rs`, `README.md`, `CHANGELOG.md`
  - **Prerequisites:** Add `memmap2 = "0.9"` to `Cargo.toml`; `cargo update`
  - **Tests:** `test_load_mmap_smoke`, `test_load_mmap_equivalence` (bit-equal tensors vs `load()`), `test_load_mmap_truncated_file`, `test_load_mmap_wrong_magic`, `test_from_file_mmap_public_api`; env-gated real GGML test
  - **Risk:** Windows divergence (gate `mmap.advise()` on `cfg(unix)`); single new `unsafe` (tight SAFETY comment); memmap2 is pure-Rust-policy compatible

- [x] **C1 — DTW word-timestamps: rename + redoc + 9 new tests → Stable badge** (planned 2026-04-25)
  - **Goal:** `src/dtw.rs` ships with ≥12 tests; module doc accurately describes algorithm; function-name corrected; README badge updated to Stable.
  - **Design:** Current `align_tokens_dtw()` is argmax-per-row + monotonic clamp — NOT DP-DTW. Rename to `align_tokens_monotonic_peak()`; add `#[deprecated]` shim forwarding `align_tokens_dtw()` (preserves SemVer for 0.1.x). Rewrite module doc. Document determinism guarantee and `WordSegment.confidence` as mean log-probability (≤ 0.0). README: "Word timestamps (DTW) | Alpha | 6" → "Word timestamps (monotonic peak) | Stable | 15+".
  - **Files:** `src/dtw.rs`, `README.md`
  - **Prerequisites:** Rename + deprecation shim first; `WordSegment.confidence` field doc before calibration tests
  - **Tests:** `test_align_tokens_all_attention_on_final_frame`, `test_align_tokens_determinism_same_inputs_same_outputs`, `test_align_tokens_argmax_tiebreak_is_documented`, `test_build_word_segments_start_le_end`, `test_build_word_segments_monotonic_starts`, `test_build_word_segments_punctuation_attaches_to_previous_word`, `test_align_tokens_with_synthetic_cross_attention_matrix`, `test_build_word_segments_confidence_clean_alignment`, `test_build_word_segments_confidence_noisy_alignment`, `test_align_tokens_handles_zero_frames_per_token_gracefully`
  - **Risk:** Rename is breaking — deprecated shim preserves 0.1.x callers; removal in 0.2.0

- [x] **D1 — Tokenizer hardening: 7 decode tests + 2 onnx-loader tests** (planned 2026-04-25)
  - **Goal:** `src/tokenizer.rs` documents actual guarantees (vocab pass-through, not BPE merging); 7 new decode tests for non-ASCII / special / OOV / punctuation; `onnx_loader::parse_tokenizer_json` gains 2 Unicode-escape tests.
  - **Design:** The module is pure vocab pass-through (no encode, no merge table); GPT-2 byte-decoding happens at GGML load time. Reframe as vocab-passthrough fidelity tests. Add module-level doc. Test `parse_tokenizer_json` for `\uXXXX` escape handling — may discover a bug (file as issue if so).
  - **Files:** `src/tokenizer.rs` (module doc + 7 tests), `src/onnx_loader.rs` (2 tests near line 1108)
  - **Prerequisites:** Module doc lands first
  - **Tests:** `test_decode_cjk_passthrough`, `test_decode_emoji_with_zwj_sequences`, `test_decode_zero_width_characters`, `test_decode_whisper_prefix_space_handling`, `test_decode_special_tokens_round_trip`, `test_decode_out_of_vocab_id_no_panic`, `test_decode_mixed_ascii_cjk_emoji_punctuation`; `test_parse_tokenizer_json_handles_unicode_escapes`, `test_parse_tokenizer_json_rejects_unpaired_surrogate`
  - **Risk:** `\uXXXX` escape test may reveal bug in `parse_json_string` — treat as finding, not blocker

- [x] **E1 — `tests/` integration directory (5 binaries + shared fixture)** (planned 2026-04-25)
  - **Goal:** `tests/` exists with 5 integration binaries exercising full public API. Shared fixture via `tests/common/mod.rs` with `OnceLock`-cached synthetic model.
  - **Design:** Change `lib.rs` `#[cfg(test)]` to `#[cfg(any(test, feature = "test-utils"))]` (the `any()` form is load-bearing — `feature = "test-utils"` alone breaks 14 existing inline call sites). Add `test-utils = []` feature. Shared `tests/common/mod.rs` exports `shared_model()`, `synthetic_sine()`, `silence()`, `sine_with_gaps()`. 5 binaries: `integration_synthetic.rs`, `integration_streaming.rs`, `integration_batch.rs`, `integration_segmented.rs`, `integration_long.rs`.
  - **Files:** `Cargo.toml`, `src/lib.rs:40-41`, `tests/common/mod.rs`, 5 new `tests/integration_*.rs`
  - **Prerequisites:** `test-utils = []` feature first; cfg gate change second; verify both `cargo test` (plain) and `cargo test --features test-utils` compile; then shared fixture; then 5 binaries
  - **Tests:** pipeline smoke, StreamTranscriber push/finish, batch (3 clips, invalid opts, mixed lengths), segmented consistency, 60s VAD long-form
  - **Risk:** `cargo test` (no flag) breaks if `any()` cfg is wrong — explicit two-mode verification step

- [x] **F1 — Pure-Rust audio format expansion (FLAC/OGG/MP3/AAC/Opus behind features)** (planned 2026-04-25)
  - **Goal:** `load_audio(path) -> Result<Vec<f32>, OxiWhisperError>` auto-detects container by magic bytes; returns 16 kHz mono f32 PCM. Format-specific entries gated. WAV path unchanged.
  - **Design:** `symphonia 0.5` (pure Rust, `default-features = false`) for FLAC/OGG/MP3/AAC; `opus-decoder 0.1.1` + `ogg 0.9` for Opus. API stays `Vec<f32>` (not a new AudioInput struct — would break examples). Features: `audio-flac`, `audio-ogg`, `audio-mp3`, `audio-aac`, `audio-opus`, `audio-all`. Refactor `audio.rs` to expose `pub(crate) fn downmix_to_mono` / `resample_linear`. Magic-byte dispatcher. Committed test fixtures ≤50 KB each.
  - **Files:** `src/audio.rs` (~+300L), `Cargo.toml`, `tests/audio_formats.rs`, `tests/fixtures/sample_440hz.{flac,ogg,opus}`
  - **Prerequisites:** Expose `pub(crate)` helpers first; add deps + features; generate fixtures with ffmpeg; write dispatcher; wire backends one feature at a time
  - **Tests:** `flac_decoded_matches_wav`, `ogg_vorbis_decoded_length_correct`, `opus_decoded_length_correct`, `auto_detect_returns_correct_decoder`, `unknown_magic_returns_format_error`, `multi_channel_downmix_via_ogg`, `resample_44100_to_16000`
  - **Risk:** Dep bloat mitigated by per-format gates; Opus surround families rejected with AudioFormatError; symphonia errors wrapped via From

- [x] **G1 — Decoder SDPA scalar → sgemm + encoder scratch reuse** (planned 2026-04-25)
  - **Goal:** `decoder::scaled_dot_product_cached` (single/prefill/full) and `scaled_dot_product_flat` use `matrixmultiply::sgemm`. Encoder attention allocates scratch once per layer (not per head). All tests pass; parity tests pin numerical equivalence. Target ≥1.5× speedup on autoregressive hot path.
  - **Design:** NOTE: `attention.rs` already uses sgemm — the hot path is decoder SDPA which is scalar. Sub-paths: `sdpa_cached_single` → gemv-shaped sgemm (M=1); `sdpa_cached_prefill` → single sgemm + upper-triangle `-inf` mask + row-softmax; `sdpa_cached_full` → sgemm pair. Add `pub(crate) struct SdpaScratch { scores: Vec<f32>, attn_out: Vec<f32> }` passed by `&mut`. Hoist encoder `scores`/`attn_out` allocation out of head loop.
  - **Files:** `src/decoder.rs`, `src/attention.rs`, `benches/transcribe.rs`, `tests/sdpa_sgemm_parity.rs`, `tests/attention_scratch_reuse.rs`, `tests/causal_mask_prefill_correctness.rs`
  - **Prerequisites:** Capture scalar reference outputs in `tests/sdpa_sgemm_parity.rs` BEFORE rewriting; add micro-bench baseline
  - **Tests:** `sdpa_sgemm_parity.rs` (tolerance 1e-5), `attention_scratch_reuse.rs`, `causal_mask_prefill_correctness.rs`
  - **Risk:** Numerical drift from sgemm vs scalar (FMA fusion) — 1e-5 tolerance; M=1 underperformance possible but unlikely

- [x] **G2 — KV-cache f16 storage (V-only default, K+V opt-in)** (planned 2026-04-25)
  - **Goal:** Cut decoder-side KV memory by ~25% (VHalf) or ~50% (KvHalf) with RTF impact ≤±5%. Default `F32` = no behavioral change for existing callers. New `TranscribeOptions::kv_cache_dtype: KvCacheDtype` field.
  - **Design:** `KvCacheDtype { F32, VHalf, KvHalf }` enum in `types.rs`. Internal `KvStorage { F32(Arc<Vec<f32>>), F16(Arc<Vec<half::f16>>) }` enum in `decoder.rs`. `LayerKVCache` holds `k: KvStorage, v: KvStorage`. Two access patterns: `k_head_borrow()` (zero-copy F32) and `k_head_into(h, scratch)` (dequant F16). `SdpaScratch` gains `k_dequant`/`v_dequant` buffers (reused). Pre-scaled K trick for `KvHalf` (store `k / sqrt(head_dim)`, use `α=1.0` for QKᵀ). `half = "2.7"` already a dep.
  - **Files:** `src/decoder.rs`, `src/types.rs`, `src/beam_search.rs`, `tests/kv_dtype_parity.rs`, `tests/kv_dtype_memory.rs`
  - **Prerequisites:** G1 must land first (SdpaScratch lives in sgemm paths)
  - **Tests:** `kv_cache_f16_v_roundtrip`, `kv_cache_f16_k_roundtrip`, `kv_cache_cow_clone_works_with_f16`, `kv_dtype_parity.rs` (3 dtype variants), `kv_dtype_memory.rs`
  - **Risk:** F16 dynamic range overflow on K → mitigated by pre-scaled K; default F32 preserves backward compat

## v0.7 — Bug Fix

- [x] BUG1 — `parse_json_string`: UTF-16 surrogate-pair handling for non-BMP code points
  - **Goal:** `😀` correctly decodes to U+1F600; a lone `\uD800` returns `Err` instead of being silently dropped; tokenizer.json files containing emojis, mathematical-alphanumeric symbols, or CJK Extension B characters now load correctly. The existing test `test_parse_tokenizer_json_rejects_unpaired_surrogate` is rewritten from documenting-broken-behavior to asserting-fixed.
  - **Design:** Inside the `b'u'` arm of `parse_json_string` (`src/onnx_loader.rs:720-734`), after parsing the 4-hex `code_point: u32`, branch on three ranges: (1) `0xD800..=0xDBFF` (high surrogate) — require lookahead bytes `\uXXXX` at `i+5..i+11`, parse low surrogate, require `0xDC00..=0xDFFF`, combine as `0x10000 + (high - 0xD800)*0x400 + (low - 0xDC00)`, push via `char::from_u32` → `encode_utf8`, advance `i += 10`; (2) `0xDC00..=0xDFFF` (lone low surrogate) — return `Err`; (3) other — tighten silent `if let Some` to `.ok_or_else()?`. All branches use `?`; no `unwrap()`.
  - **Files:** `src/onnx_loader.rs` (~30 LoC delta inside `parse_json_string`; rewrite of existing test; 5 new tests), `CHANGELOG.md` (v0.1.1 entry)
  - **Prerequisites:** None
  - **Tests:** (1) supplementary-plane decode: `"😀"`, `"𝐀"`, `"𠀀"` roundtrip; (2) lone high surrogate → Err; (3) lone low surrogate → Err; (4) high surrogate followed by non-low → Err; (5) truncated after high surrogate → Err; (6) update existing test to assert `is_err()`; (7) regression guard for `test_json_string_escapes` and `test_parse_tokenizer_json_handles_unicode_escapes`
  - **Risk:** Off-by-one in lookahead bounds — mitigated by explicit `i+10 < len` precondition + `bytes.get(...)`. Index-advance arithmetic verified by table walkthrough.

## v0.7 — Roadmap

- [x] GGUF1 — Add GGUF format support side-by-side with GGML
  - **Goal:** Magic-byte autodetection in `ModelData::load_from_reader` so existing `WhisperModel::from_file()` / `from_file_mmap()` transparently accept both legacy GGML (`0x67676D6C`) and modern GGUF (`GGUF` = `0x46554747` LE). Public API unchanged. New private module `src/gguf/` does spec parsing; populates the same `Hparams` / `mel_filters` / `vocab` / `tensors` / `quantized_tensors` containers.
  - **Design:** Dispatcher `load_from_reader<R: Read + Seek>` reads 4-byte magic, branches to `load_ggml_from_reader<R: Read>` (existing logic extracted) or `load_gguf_from_reader<R: Read + Seek>` (new). GGUF stages: (1) header: `magic/version/tensor_count/metadata_kv_count`; reject version != 3; (2) metadata KV loop: 13-type `GgufValueType` enum (U8=0..F64=12), strings as `u64 len + UTF-8`, arrays as `type u32 + u64 len + elements`; store in `HashMap<String, GgufValue>`; (3) tensor info loop: `name/n_dims/dims[n_dims]/dtype/offset`; (4) alignment: read `general.alignment` (default 32), pad to alignment, record `tensor_data_base`; (5) tensor data: `seek(Start(tensor_data_base + info.offset))` for each tensor. Dtype map: 0→F32, 1→F16, 2→Q4_0, 6→Q5_0, 8→Q8_0 reuse existing readers; all others → `InvalidModel("unsupported GGUF dtype N")`. Whisper KV-key resolver uses candidate-lists per hparam field. Mel-filter three-tier probe: tensor named `mel_filters` → KV array `whisper.mel_filters` → regenerate from `n_mels`.
  - **Files:** New `src/gguf/mod.rs` (~40L), `src/gguf/spec.rs` (~180L), `src/gguf/parse.rs` (~320L), `src/gguf/whisper.rs` (~140L); modify `src/model.rs` (extract GGML body, add dispatcher, +Seek bound, ~40 net), `src/lib.rs` (`pub mod gguf;`), `src/test_utils.rs` (`SyntheticSpec` + `generate_synthetic_gguf`)
  - **Prerequisites:** GGUF1.0 — research real ggml-tiny.gguf KV keys from whisper.cpp convert script; GGUF1.1 — extract GGML body; GGUF1.2 — refactor test_utils to SyntheticSpec; GGUF1.3–1.6 — spec types, parse, whisper, wire dispatcher
  - **Tests:** alignment math at edge offsets; KV roundtrip for all 13 value types; malformed magic; truncated header; absurd tensor_count rejected; non-LE version rejected; key-resolver picks first present candidate; `test_load_synthetic_gguf`; `test_ggml_gguf_equivalence` (bitwise-identical tensors + hparams); `test_load_mmap_gguf`; `test_load_gguf_unsupported_dtype`; env-gated `test_real_gguf_load` / `test_real_gguf_transcribe`
  - **Risk:** Whisper.cpp KV-key drift (candidate-list resolver), mel-filter location uncertainty (three-tier probe), dtype coverage gap (explicit InvalidModel), tensor offset misalignment (alignment math unit tests), BE GGUF v3 (explicit reject)

- [x] DTW1 — True DP-DTW with Sakoe-Chiba band + traceback
  - **Goal:** Add `pub fn align_tokens_dp_dtw(attention_weights: &[f32], n_tokens: usize, n_frames: usize, hop_length: usize, sample_rate: usize, band_width: Option<usize>) -> Vec<(f32, f32)>` to `src/dtw.rs`. Un-deprecate `align_tokens_dtw` and rebind it to forward to the new genuine DP implementation. Remove the "planned for 0.2" disclaimer. CHANGELOG notes semantic change.
  - **Design:** (1) Softmax-normalize each token row (subtract row-max, exp/sum) to get probabilities; (2) local cost `c[i,j] = -ln(p[i,j].max(1e-22))` clamped to `[0.0, 50.0]`; (3) default band = `band_width.unwrap_or(max(10, n_frames / 4))`; (4) Sakoe-Chiba: cell `(i,j)` in-band iff `(j*n_tokens).abs_diff(i*n_frames) <= band_width*n_tokens`; (5) DP: `C[i,j] = c[i,j] + min(C[i-1,j-1], C[i-1,j], C[i,j-1])` with rolling two-row buffer (O(n_frames) working set) + `Vec<u8>` predecessor matrix (0=diag, 1=up, 2=left) sized `n_tokens * n_frames`; (6) traceback from `(n-1,m-1)` to `(0,0)`, group consecutive same-i frames to derive per-token span, convert to seconds via `frame_to_time`; (7) edge cases: 0 tokens/frames → empty Vec; overconstrained → fallback to monotonic_peak; final cell infinite → retry with full band; all-zero → diagonal path. No `unwrap()`; all Vec accesses use `.get()` or pre-bounded indices.
  - **Files:** `src/dtw.rs` (~250 LoC), `README.md` (algorithm section), `CHANGELOG.md` (v0.1.1 entry)
  - **Prerequisites:** None (purely additive)
  - **Tests:** (1) cost-matrix shape 4×12; (2) traceback ends at (0,0) starts at (n-1,m-1); (3) timestamps strictly monotonic non-decreasing; (4) auto-widen when band=1; (5) parity vs monotonic_peak on clean diagonal-attention fixture; (6) divergence vs monotonic_peak on noisy attention (DP smoother); (7) bit-exact determinism; (8) overconstrained fallback; (9) band_width=Some(0) auto-widens; (10) all-zero attention → no NaN, finite times; (11) align_tokens_dtw alias equals align_tokens_dp_dtw with default band
  - **Risk:** Memory — predecessor matrix `O(n_tokens * n_frames)` bytes (~670 KB for 448×1500), acceptable; documented in rustdoc. Numerical — -ln(0) mitigated by 1e-22 floor + 50.0 clamp. Behavioral change in `align_tokens_dtw` — surfaced in CHANGELOG as documented semantic improvement.

- [x] PERF1 — Per-head decoder threading via rayon (feature-gated `parallel`)
  - **Goal:** Add opt-in feature `parallel = ["dep:rayon"]` (NOT in default-features). Decoder SDPA per-head loops at `decoder.rs:920/991/1065/1143/1218` and encoder per-head loops at `attention.rs:85/114/130/158` parallelize across heads when feature enabled. Default build bit-identical to today; WASM `--no-default-features` still compiles. Bench shows ≥1.3× speedup on Whisper-base (n_head=8) at beam=5 on multi-core.
  - **Design:** Refactor `SdpaScratch` from one shared workspace to `Vec<HeadScratch>` indexed by `h` (mandatory to prevent races). New `src/threading.rs` with `crate::par::{par_for_each, install_pool}` shim: with feature on → `into_par_iter().for_each(...)`; with feature off → `(0..n).for_each(...)`. Apply at all 9 head-loop sites. Sampler/softmax-final/beam-merge/layer-loop stay serial. `oxiwhisper::threading::set_thread_count(n)` wraps `rayon::ThreadPoolBuilder`. Amdahl ceiling for base (n_head=8, head-loop ≈60% decode): 1/(0.4 + 0.6/8) ≈ 2.1×.
  - **Files:** `Cargo.toml` (rayon 1.x optional + `parallel` feature), `src/decoder.rs` (5 head-loops + SdpaScratch shape), `src/attention.rs` (4 encoder head-loops), `src/threading.rs` (new), `src/lib.rs` (`pub mod threading`), `benches/transcribe.rs` (parallel/serial groups), `examples/profile_threading.rs` (new), `README.md` (feature note)
  - **Prerequisites:** Capture single-thread baseline BEFORE wiring rayon. SdpaScratch per-head Vec<HeadScratch> refactor must land in same commit as parallel iteration.
  - **Tests:** `tests/threading_parity.rs` — bit-equal logits within 1e-5 over fixed seed beam-5 decode; `tests/threading_smoke.rs` — parallel build runs end-to-end on synthetic mel; WASM smoke in CI matrix; bench delta ≥1.3× gate on Whisper-base
  - **Risk:** Data races on SdpaScratch (mitigated by per-head Vec<HeadScratch>); tiny-model regression where rayon overhead > head work (Amdahl-doc note + bench gate); WASM compile breakage (feature gate); thread-pool oversubscription (set_thread_count wrapper)

- [x] CQ1 — Split `decoder.rs` (1598L) and `lib.rs` (1580L); enforce `missing_docs` on `lib.rs`
  - **Goal:** `decoder.rs` becomes ~50L facade re-exporting from `src/decoder/{sdpa.rs, kv_cache.rs, forward.rs, sampler.rs}` (each <600L). `lib.rs` slims to ~400L by extracting impl methods into `src/whisper_model.rs`. Add `#![warn(missing_docs)]` lint; backfill ~95+ pub items — NO `#[allow(missing_docs)]` escape hatch.
  - **Design:** `decoder.rs` split seams: `KvStorage`/`LayerKVCache` (lines 34-322) → `kv_cache.rs`; `SdpaScratch`/SDPA fns (823-1228) → `sdpa.rs` (PERF1's per-head Vec lives here); `pub fn decode`/build_prompt/ForwardCtx (323-522) → `forward.rs`; `decode_greedy`/`decode_sample` (525-822) → `sampler.rs`. Old `decoder.rs` → `pub use self::{kv_cache::*, sdpa::*, forward::*, sampler::*}`. `lib.rs` split: extract every `impl WhisperModel` method into `src/whisper_model.rs`; `lib.rs` retains `//!`, `pub mod`, `pub use`, bare struct, lint. Run `splitrs --dry-run` first. `missing_docs` backfill: one concise rustdoc-line per item across all `src/*.rs`.
  - **Files:** `src/decoder.rs` (→ facade), `src/decoder/{kv_cache,sdpa,forward,sampler}.rs` (new), `src/lib.rs` (slim + lint), `src/whisper_model.rs` (new), `.splitrs.toml` (new), all `src/*.rs` for doc backfill
  - **Prerequisites:** PERF1 must land FIRST (SdpaScratch shape change conflicts with splitting sdpa.rs; `pub mod threading` needs to be in lib.rs before CQ1 slims it). Strict order: PERF1 → CQ1.
  - **Tests:** `tests/api_stability_v07.rs` — every pre-split pub re-importable from `oxiwhisper::*` and `oxiwhisper::decoder::*`; all 358+ existing tests pass unchanged; `cargo doc --all-features --no-deps` zero warnings; `cargo clippy --all-features --all-targets -- -D warnings` clean; `wc -l` confirms no file >1500L
  - **Risk:** splitrs mis-resolving crate-private imports (hand review + clippy + dry-run); missing_docs backfill across ~95+ items tedious (dedicated doc pass); ordering conflict with PERF1 (strict serial sequencing); `pub use *` glob conflict if two submodules export same name (audit during split)
