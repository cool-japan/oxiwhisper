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

---

## v0.8 — Robustness & Whisper Parity (ships as 0.1.2)

- [x] **A — Translation task** — `Task { Transcribe, Translate }` enum + `task` field in `TranscribeOptions`; `build_prompt` takes explicit `task_token: u32`; default `Task::Transcribe` is a no-op for all existing callers; `SpecialTokens.translate` (50358) was already present
- [x] **B — Temperature fallback decoding** — `fallback_temperatures: &[f32]` + `logprob_threshold: f32` in `TranscribeOptions`; `decode()` loops over temperatures, accepts first result that passes `is_likely_hallucination` + avg-log-prob gate; empty schedule (default) preserves single-dispatch at zero cost
- [x] **C — Progress callbacks** — `transcribe_long_with_progress`, `transcribe_long_segmented_with_progress`, `transcribe_long_with_vad_with_progress` variants taking `FnMut(usize, usize)`; original methods delegate via no-op closure; signatures unchanged


---

## v0.9 — Word Timestamps & OpenAI Parity (ships as 0.1.2)

- [x] **1A — Cross-attention capture in `scaled_dot_product_flat`** — added `capture: Option<&mut [f32]>` param; post-softmax head-mean reduction runs AFTER the rayon section (no `&mut` crosses the closure); zero overhead when `None`
- [x] **1B — Upper-half alignment layer heuristic** — `is_alignment_layer(layer, n_layer) -> bool` in `cross_attn_capture.rs`; captures only `layer >= n_layer/2` to get the cleanest monotonic attention signal
- [x] **1C — `CrossAttnCapture` accumulator** — new `src/decoder/cross_attn_capture.rs`; `layer_sink()` + `commit_layer()` + `finish()` accumulate head- and layer-averaged attention with O(1) scratch reuse; 5 tests
- [x] **1D — Thread capture through `forward()` and samplers** — `forward()` gains `mut capture: Option<&mut CrossAttnCapture>`; layer loop conditionally captures alignment layers; `decode_greedy`/`decode_sample` gain `capture_output: Option<&mut Vec<f32>>`; prefill pushes last row, incremental steps push per-token rows only on acceptance
- [x] **1E — `DecodeResult` extended** — `cross_attention: Option<Vec<f32>>`, `enc_len: usize`, `no_speech_prob: f32` added; silence gate + capture assembly wired in `decode()`
- [x] **1F — Public word-timestamps API** — new `src/word_timestamps.rs`; `WordTimedTranscript { text, words, language, no_speech_prob }`; `build_word_timed_transcript` wires `align_tokens_dp_dtw` + `build_word_segments`; `transcribe_words_impl` free function; `WhisperModel::transcribe_words()` delegate; `pub use dtw::WordSegment` and `pub use word_timestamps::WordTimedTranscript` in `lib.rs`
- [x] **2A — `decode_utils` helpers** — `token_prob`, `apply_suppress_blank`, `apply_timestamp_rules` (full OpenAI 4-rule logic, NaN-safe); `DecodeConstraints` extended with `timestamp_rules`, `suppress_blank`, `blank_token`; 16 unit tests
- [x] **2B — Wire into all three samplers** — `decode_greedy`, `decode_sample`, `decode_beam` apply `apply_suppress_blank` (step 0 only) and `apply_timestamp_rules` (every step); crude no-speech argmax checks deleted
- [x] **2C — `no_speech_prob` extraction + silence gate** — captured from raw prefill logits BEFORE suppression in all three samplers; return 3-tuple `(tokens, probs, no_speech_prob)`; silence gate in `decode()` uses combined threshold
- [x] **2D — `TranscribeOptions` options** — `word_timestamps: bool` (default `false`), `no_speech_threshold: f32` (default `0.6`), `suppress_blank: bool` (default `true`); exhaustive test literal in `whisper_model.rs` updated; `validate_options` rejects out-of-range `no_speech_threshold`
- [x] **Lone-timestamp semantics corrected** — OpenAI parity: after a lone timestamp ALL text tokens including EOT are suppressed (not just non-EOT); forces the model to emit a closing timestamp
- [x] **Version bump** — Cargo.toml 0.1.1 → 0.1.2

---

## v0.10 — Test-Integrity Repair (audit 2026-07-10)

Audit finding: v0.6–v0.9 shipped their **features** but not several of their claimed **test
deliverables**. The suite is green (436 pass, 0 clippy warnings) because the tests that were
supposed to pin the risky code do not exercise it. Specifically:

- `tests/sdpa_sgemm_parity.rs::test_sdpa_unit_sgemm_correctness` hand-rolls *both* a scalar
  reference and an sgemm call inside the test file and compares them to each other. It never
  calls into `oxiwhisper`. Deleting `src/decoder/sdpa.rs` would not fail it.
- `src/decoder/sdpa.rs` (659 L, 8 `unsafe` sgemm blocks, the decoder hot path) has **zero**
  inline tests. The prefill causal mask (`sdpa.rs:453-459`) is untested — an off-by-one there
  leaks future tokens and still produces plausible text.
- `tests/threading_smoke.rs` — both tests assert nothing (`let _result = ...;`).
- `KvHalf` is only asserted `is_ok()`; its pre-scaled-K numerical trick has no parity bound.
- Claimed-but-never-written: `tests/causal_mask_prefill_correctness.rs`, `tests/kv_dtype_memory.rs`,
  `examples/profile_threading.rs`, `.splitrs.toml`.
- `cargo nextest` does not run doctests; `src/lib.rs` Quick Start is ```` ```ignore ```` and is
  therefore never compile-checked (and appears to be wrong).

Decisions taken this session: `rand` stays as a direct dependency, no SciRS2 policy doc (user).

### Batch 1 — repair ✅ complete (454 tests pass, 0 skipped; doctests 3/3, 0 ignored; clippy clean)

Outcomes and corrections discovered while doing the work:

- **T1 mutation-tested its own test.** Injecting an off-by-one at `sdpa.rs:455` (`past_len + i + 1`
  → `+ 2`) makes the new causal-mask tests fail loudly; reverted. The test genuinely detects the leak.
- **T1 latent finding (unfixed, unreachable):** `softmax_rows` returns `NaN`, not zeros, for a
  fully-masked (all `-inf`) row, because `(-inf) - (-inf) = NaN`. Not reachable in production —
  the causal mask always leaves ≥1 valid key (query `i` always attends to key 0). Pinned as a
  characterisation test, not silently "fixed".
- **T6 corrected the audit.** `WhisperModel::transcribe` returns `Result<String, _>`, so
  `println!("{text}")` was always valid. The Quick Start's only real defect was a bare `?` with no
  enclosing function returning `Result` — invisible because ```` ```ignore ```` never compiled it.
- **T5 verified `.splitrs.toml` is a real supported schema** (read `splitrs-0.3.4/src/config.rs`,
  validated with `--dry-run`) rather than inventing a plausible-looking config the tool ignores.
- **T5 measured a real parallel speedup:** seq_len=750, n_head=20 → 297.65 ms serial vs 221.76 ms
  with `--features parallel` (5 threads) ≈ 1.34×, consistent with PERF1's ≥1.3× claim.
- **T2 measured F32-vs-VHalf and F32-vs-KvHalf divergence = 0.0**, and reported honestly that the
  synthetic model is too well-conditioned for this to be a real f16 stress test. The genuine
  pre-scaled-K precision check is the new unit test in `kv_cache.rs`
  (`max_k_err ≈ 1.95e-4`, within the f16 half-ULP bound). No bug in the prescale path.

**Escalated finding — the synthetic model is degenerate.** T2, T4 and T6 independently observed
that `test_utils`' synthetic model emits a single constant token (51700, ×224), yielding empty
`text` and zero `segments`. T6 confirmed `test_long_segmented_monotonic_timestamps`'s
`for i in 1..segs.len()` loop has always executed **zero** iterations. An unknown number of
integration tests therefore pass vacuously. This is a deeper defect than the four missing files
and is now Batch 2's first task.

- [x] **T1 [opus] — Real inline tests for `src/decoder/sdpa.rs` vs a scalar reference**
  - Add `#[cfg(test)] #[path = "sdpa_tests.rs"] mod tests;` + new `src/decoder/sdpa_tests.rs`
    (keeps `sdpa.rs` well under the 2000 L limit) that calls the **actual** crate functions:
    `scaled_dot_product_cached` (single / prefill / full), `scaled_dot_product_flat`, `softmax_rows`.
  - Prefill causal-mask exactness: `past_len = 0` and `past_len > 0`; assert masked positions
    contribute exactly zero; off-by-one guards at `valid = past_len + i + 1`.
  - f16 KV dequant parity (`materialize_k_head` / `materialize_v_head`) and the `k_prescaled`
    `alpha = 1.0` path.
  - **Files:** `src/decoder/sdpa.rs`, `src/decoder/sdpa_tests.rs` (new)

- [x] **T2 [opus] — `KvHalf` / `VHalf` numerical parity + KV memory assertions**
  - Upgrade `test_kv_kv_half_does_not_panic` from `is_ok()` to a real numerical bound
    (compare `token_probs` against `F32` within tolerance). Same for the beam-search variant.
  - Fold the never-written `tests/kv_dtype_memory.rs` into inline tests in `kv_cache.rs`
    (memory introspection needs `pub(crate)` access); assert `KvStorage` byte sizes per dtype.
  - **Files:** `tests/kv_dtype_parity.rs`, `src/decoder/kv_cache.rs`

- [x] **T3 [sonnet] — Remove the vacuous SDPA unit test**
  - Delete `test_sdpa_unit_sgemm_correctness` (superseded by T1's real inline tests; note why in
    a comment). Keep and keep-honest the two end-to-end `transcribe` parity tests.
  - **Files:** `tests/sdpa_sgemm_parity.rs`

- [x] **T4 [sonnet] — Make `threading_smoke.rs` assert something**
  - Both tests currently discard their result. Assert the real `set_thread_count` contract, and
    add the PERF1-claimed "parallel build runs end-to-end on synthetic mel" test.
  - **Files:** `tests/threading_smoke.rs`

- [x] **T5 [sonnet] — Write the missing `examples/profile_threading.rs` + `.splitrs.toml`**
  - PERF1 claimed both. Add the `[[example]]` block to `Cargo.toml`.
  - **Files:** `examples/profile_threading.rs` (new), `Cargo.toml`, `.splitrs.toml` (new)

- [x] **T6 [sonnet] — Un-ignore the long-form test; fix the uncompiled headline doctest**
  - Remove `#[ignore]` from `tests/integration_long.rs:9` (no `#[ignore]` policy). If runtime
    exceeds ~90 s, shrink the fixture but keep it above the 30 s chunk threshold so it still
    exercises the multi-encoder-pass path. Do **not** re-add `#[ignore]`.
  - `src/lib.rs` Quick Start: ```` ```ignore ```` → ```` ```no_run ```` and make it actually compile.
  - **Files:** `tests/integration_long.rs`, `src/lib.rs`

### Batch 2 — hardening

- [x] **H0 [opus] — Vacuous-test audit + non-degenerate synthetic model** *(escalated; highest value)*
      The synthetic model emits one constant token, so `text` is empty and `segments` is empty.
      Every `for i in 1..segs.len()` / `for seg in &segs` assertion in the integration suite is
      therefore a no-op. Enumerate every test that can pass on empty input; then either craft the
      synthetic weights so the decoder emits a **designed** token sequence (including paired
      timestamp tokens, so `parse_segments` actually yields segments), or — if that is not
      principled — add explicit non-vacuity guards and document which tests are structure-only.
      Never assert on empty collections and call it coverage.
      **Files:** `src/test_utils.rs`, `tests/*.rs`
- [x] **H1 [opus] — Property tests for the GGUF parser + `parse_json_string`**: malformed headers,
      absurd `tensor_count`, truncated KV, bad alignment, surrogate pairs — must return `Err`, never
      panic. Owns the `proptest` dev-dependency.
      **Files:** `Cargo.toml`, `src/gguf/parse.rs`, `src/onnx_loader.rs`, `src/model.rs`
- [x] **H2 [opus] — Miri over the `unsafe` blocks.** Assess feasibility honestly: `matrixmultiply::sgemm`
      and the AVX2/NEON intrinsics may be un-Miri-able. Cover what can be covered; report what cannot.
      Investigation only — report, do not paper over.
- [x] **H3 [sonnet] — Criterion regression gate** wired to the existing `benches/transcribe.rs`.
      **Files:** `benches/transcribe.rs`

---

## v0.11 — Speaker Diarization (ships as 0.1.2)

**Goal:** answer *"who spoke when"* (diarization) and *"who spoke what"* (speaker-attributed
transcript). oxiwhisper today has none of this — no `diariz`/`speaker` symbol exists anywhere in
`src/`. This section adds it as an **offline embedding-clustering pipeline** (the pyannote-2.x
lineage), chosen over end-to-end neural diarization (EEND) because that would require *training*,
whereas oxiwhisper is an inference-only crate and already has an ONNX backend (`oxionnx`) to host a
pretrained embedding model.

**Pipeline** (each stage below is a batch): speech activity → uniform sub-segmentation →
per-window speaker embedding → affinity + clustering (with speaker-count estimation) →
resegmentation → fusion with word timestamps. Existing bricks it builds on: `vad::detect_speech`
(`src/vad.rs`, energy VAD → `Vec<SpeechSegment>`), `encoder_output()` (`src/whisper_model.rs:744`),
`transcribe_words()` + `WordSegment`/`WordTimedTranscript` (`src/word_timestamps.rs`, `src/dtw.rs`).

**HONEST CONSTRAINT — the embedding model is external.** Discriminating speakers needs a
*speaker-verification* embedding (x-vector / ECAPA-TDNN), which is a **separate pretrained model**
oxiwhisper cannot train and does not ship. The plan loads one via `oxionnx`. Critically: Whisper's
own encoder is trained to be **speaker-*invariant*** (it encodes phonetic content for ASR), so
`encoder_output()` embeddings cluster only weakly by speaker — the Whisper-encoder path below is a
zero-extra-model **baseline/smoke** route only, and MUST be documented as low-accuracy, never
presented as production diarization. Shipping it as "diarization" unqualified would be exactly the
kind of plausible-but-fake capability the v0.10 audit was about.

**Ecosystem policy:** clustering + eigendecomposition come from `scirs2-cluster` / `scirs2-linalg`
(no `ndarray`/`nalgebra` per SCIRS2 policy); the ONNX embedder rides the existing `onnx` feature.
The whole subsystem sits behind a new `diarization` Cargo feature so the default build is unchanged.

### Batch D — foundation

- [x] **D1 [opus] — Public types + `diarization` feature gate.** New `src/diarize/mod.rs`.
      `SpeakerId(pub u32)`; `SpeakerSegment { speaker: SpeakerId, start: f32, end: f32 }`;
      `DiarizeResult { segments: Vec<SpeakerSegment>, num_speakers: usize }`;
      `DiarizeOptions { num_speakers: Option<usize>, min_speakers: usize, max_speakers: usize,
      window_s: f32 /*≈1.5*/, hop_s: f32 /*≈0.75*/, min_duration_s: f32, clustering: ClusteringMethod,
      vad: VadConfig }` with a documented `Default`; `DiarizeOptions::validate()` (ranges, `min≤max`,
      `hop_s≤window_s`, positive windows) mirroring `validate_options` in `types.rs`. `serde` derives
      behind the existing `serde` feature on the **output** types only (`SpeakerId`/`SpeakerSegment`/
      `DiarizeResult`); `DiarizeOptions`/`ClusteringMethod` stay non-serde like `TranscribeOptions`
      (they hold a non-serde `VadConfig`). Gate the module + re-exports in `lib.rs`.
      **Files:** `src/diarize/mod.rs` (new), `src/lib.rs`, `Cargo.toml`
      **✅ done (green):** build feature on/off, +serde, clippy `-D warnings`, 14/14 diarize tests pass;
      adversarial review `verified_green`/`faithful_to_spec`. **Deviation:** `diarization = []` (empty)
      — the `scirs2-cluster`/`scirs2-linalg` deps were deferred to D4 (where clustering uses them) to
      keep this feature free of unused deps. **Carry-forward (fix in D-scaffold):** the AHC-threshold
      rejection test only covers `0.0`; add a `NaN`/`Inf` case to exercise the `is_finite()` branch.

- [x] **D2 [sonnet] — Uniform sub-segmentation over VAD speech regions.** `src/diarize/segment.rs`:
      `fn window_speech(regions: &[SpeechSegment], sample_rate: usize, window_s, hop_s) ->
      Vec<SpeechSegment>` — tile each VAD region into overlapping fixed windows, keep a trailing
      short window only if ≥ `min_duration_s`, never emit a window crossing a silence gap. Pure,
      deterministic, no model. This is where boundary resolution vs embedding-SNR trades off — document
      the trade-off in the module header.
      **Files:** `src/diarize/segment.rs` (new)

- [x] **D3 [opus] — `SpeakerEmbedder` trait + two backends.** `src/diarize/embed.rs`:
      `trait SpeakerEmbedder { fn dim(&self) -> usize; fn embed(&self, audio: &[f32], sample_rate:
      usize) -> Result<Vec<f32>, OxiWhisperError>; }` returning an **L2-normalized** vector.
    - `EcapaOnnx` (behind `onnx`) — load a pretrained ECAPA-TDNN / x-vector ONNX via `oxionnx`; the
      **real** embedder. Document the expected input (16 kHz mono, fbank/mel per the model card) and
      that the `.onnx` is user-supplied (licensing: WeSpeaker/SpeechBrain weights are not ours to vend).
    - `WhisperEncoderEmbedder` — mean-pool (or attentive-pool) `encoder_output()` over the window;
      the **baseline**. Doc-comment MUST state the speaker-invariance caveat above and that it exists
      for tests / no-model demos only.
      **Files:** `src/diarize/embed.rs` (new), `src/whisper_model.rs`

- [x] **D4 [opus] — Clustering + speaker-count estimation.** `src/diarize/cluster.rs`:
      `enum ClusteringMethod { Ahc { threshold: f32 }, Spectral }`. Build a cosine-affinity matrix over
      the L2-normalized embeddings, then:
    - **AHC** — agglomerative average/ward linkage via `scirs2-cluster`; cut at `threshold`, or at the
      largest dendrogram gap when `num_speakers` is `None`, bounded to `min..=max`.
    - **Spectral** — normalized Laplacian, leading eigenvectors via `scirs2-linalg`, **eigengap
      heuristic** for `k` when unknown (bounded), then k-means on the embedded rows.
      Fixed `num_speakers` short-circuits the estimator. Emits a per-window label vector.
      D4 also **owns the dep wiring deferred from D1**: update `Cargo.toml` to `diarization =
      ["dep:scirs2-cluster", "dep:scirs2-linalg"]` (verify the crates/versions on crates.io first; if
      `scirs2-cluster`'s API doesn't fit, implement average-linkage AHC **inline** — real, not stubbed —
      and file Spectral as a follow-up rather than faking it).
      **Files:** `src/diarize/cluster.rs` (new), `Cargo.toml`

> **✅ Round 2 (D2·D3·D4) done — verified green (468/468 tests, clippy clean feature on/off + onnx; adversarial review `verified_green`/`faithful_to_spec`).**
> - **D2** `window_speech` — gap-safe tiling, 7 exact-boundary tests. Public: `diarize::segment::window_speech`.
> - **D3** `SpeakerEmbedder` trait + `WhisperEncoderEmbedder` (baseline, carries the speaker-invariance caveat) + `EcapaOnnx` (`#[cfg(feature="onnx")]`, **real** log-mel front-end + oxionnx run). Added `WhisperModel::d_model()`. Honest caveat: stock WeSpeaker/SpeechBrain exports need a Kaldi-fbank (natural-log + CMN) front-end to match — see D3 followups; a `KaldiFbank` option is a future enhancement.
> - **D4** `cluster_speakers` — **real inline** average-linkage AHC + spectral (inline cyclic **Jacobi** eigensolver, eigengap `k`, deterministic k-means++/SplitMix64, no `rand`). Threshold is cosine-**distance** (1−cos), range [0,2] (docs reconciled from a review-caught affinity/distance mismatch). **Deviation kept:** `diarization = []` — clustering is fully inline, `scirs2-cluster`/`scirs2-linalg` NOT added (opus judged `eigh` would pull `ndarray` against the crate's minimal-dep policy). Roadmap D4's "add scirs2 deps" is therefore intentionally not done.
> - ⚠️ **`src/diarize/` is untracked in git** — `git add src/diarize/` when committing D-series work.
> - Carry-forward for D5: the `src/diarize/mod.rs` header + `src/lib.rs` "type foundation only" comment are now stale (D2–D4 implemented) — fix during D5.

- [x] **D5 [opus] — Resegmentation → `DiarizeResult`.** `src/diarize/reseg.rs`: turn per-window
      labels into contiguous `SpeakerSegment`s — merge adjacent same-speaker windows, resolve
      overlap-region label conflicts by majority/embedding-distance, drop < `min_duration_s` flickers,
      convert sample indices → seconds. `WhisperModel::diarize(&self, audio, &DiarizeOptions) ->
      Result<DiarizeResult>` wiring D2→D3→D4→D5. `pub use` the types in `lib.rs`.
      **Files:** `src/diarize/reseg.rs` (new), `src/whisper_model.rs`, `src/lib.rs`
      **✅ done (green):** `resegment()` (overlap-midpoint boundaries, monotonic/sorted/non-overlapping,
      flicker absorption); `WhisperModel::diarize()` (built-in Whisper baseline embedder, doc'd as
      low-accuracy) + `diarize_with_embedder(&dyn SpeakerEmbedder)` (real path for `EcapaOnnx`).
      `num_speakers` derived from final segments. Honest tests: a **planted-embedder** integration test
      drives the full VAD→segment→embed→cluster→reseg path and recovers 2 speakers (boundary ~4.125 s vs
      4.0 s midpoint); a real-`diarize()` smoke test asserts structure only (DER deferred to F2).

---

> ## 🏁 Batch D — Speaker Diarization FOUNDATION COMPLETE (2026-07-10)
> **All of D1–D5 landed and independently verified: 480/480 tests pass (58 in `diarize`), clippy +
> rustdoc clean under `-D warnings` (feature on/off, ±onnx).** Diarization is now usable end-to-end:
> ```rust
> let opts = DiarizeOptions::default();
> let result = model.diarize(&audio, &opts)?;                 // built-in baseline (low-accuracy)
> let result = model.diarize_with_embedder(&audio, &opts, &ecapa)?;  // real ECAPA path
> ```
> **Deliberate deviations from the original roadmap (all honest, all documented):**
> - `diarization = []` — clustering + the symmetric eigensolver are **fully inline**; `scirs2-cluster`/
>   `scirs2-linalg` were NOT added (their `eigh` pulls `ndarray` against the crate's minimal-dep policy).
> - `EcapaOnnx` uses a Whisper-convention log-mel front-end; stock WeSpeaker/SpeechBrain checkpoints
>   need a Kaldi-fbank (natural-log + CMN) front-end to match — a future enhancement (see D3 followups).
> - **No accuracy/DER is asserted anywhere** — the Whisper baseline is speaker-invariant and can't
>   honestly validate accuracy; real DER validation is Batch **F2** (needs an ECAPA checkpoint + labelled corpus).
>
> **⚠️ ACTION FOR WHOEVER COMMITS:** `src/diarize/` is **untracked in git** — run `git add src/diarize/`
> (and stage the edits to `src/whisper_model.rs`, `src/lib.rs`, `Cargo.toml`) so the D-series work is included.
>
> **Next session:** Batch E (ASR fusion — "who spoke what"), then Batch F (RTTM, DER/JER metrics, F2 real eval).

### Batch E — speaker-attributed transcription (fusion with ASR)

> ## 🏁 Batch E — SPEAKER-ATTRIBUTED TRANSCRIPTION COMPLETE (2026-07-11)
> **E1 + E2 landed and independently verified: full gate GREEN — 557/557 tests pass (0 skipped),
> clippy `-D warnings` clean (`--all-features`), doctests 3/3, and `RUSTDOCFLAGS="-D warnings" cargo doc`
> clean for default / `diarization`-only / `--all-features`.** oxiwhisper now answers "who spoke what":
> ```rust
> let st = model.transcribe_with_speakers(&audio, &t_opts, &d_opts)?;          // baseline (low-accuracy)
> let st = model.transcribe_with_speakers_using_embedder(&audio, &t_opts, &d_opts, &ecapa)?; // real path
> println!("{}", oxiwhisper::labeled_transcript(&st));                          // [SPEAKER_0] ...
> oxiwhisper::write_rttm(&diar, "utt", &mut std::io::stdout())?;               // NIST RTTM
> ```
> **Bonus fix:** the E1 pass also closed a **pre-existing** doc break — three `EcapaOnnx` intra-doc links
> (embed.rs, whisper_model.rs) failed `cargo doc --features diarization` (without `onnx`); now cfg-gated
> so the crate docs clean under every feature combination.
> **⚠️ whisper_model.rs is now 1918 lines** (< 2000, but approaching the refactor threshold) — extracting
> the diarize/attribute wrappers into a `whisper_model/` submodule is a near-term follow-up.

- [x] **E1 [opus] — Fuse diarization with word timestamps.** `src/diarize/attribute.rs`:
      `SpeakerTurn { speaker: SpeakerId, text: String, start: f32, end: f32, words: Vec<WordSegment> }`;
      `SpeakerTranscript { turns: Vec<SpeakerTurn>, num_speakers: usize, language: Option<String> }`.
      Assign each `WordSegment` (from `transcribe_words()`) to the `SpeakerSegment` covering its
      **midpoint** (fallback: nearest by gap); collapse consecutive same-speaker words into turns.
      `WhisperModel::transcribe_with_speakers(&self, audio, &TranscribeOptions, &DiarizeOptions) ->
      Result<SpeakerTranscript>` — runs ASR and diarization then fuses. Note the honest limitation:
      one speaker per word, so overlapped speech is mis-attributed (→ Batch G).
      **Files:** `src/diarize/attribute.rs` (new), `src/whisper_model.rs`, `src/lib.rs`
      **✅ done (green):** `attribute_words(words, segments, language) -> SpeakerTranscript` — midpoint
      `0.5*(start+end)`, half-open `[start,end)` containment, distance-to-interval nearest-gap fallback
      (`(gap,start)`→lowest-index tie-break), consecutive same-speaker collapse, `text = concat+trim`,
      `num_speakers = distinct-in-turns` (may be ≤ `DiarizeResult.num_speakers`), empty-segments never
      fabricates a speaker. Thin `transcribe_with_speakers` (baseline) + `transcribe_with_speakers_using_embedder`
      (real) share a private `fuse_*`; both clone `TranscribeOptions` to force `word_timestamps=true`. 9
      non-vacuous tests (exact ids/bounds/text/serde round-trip). Honest overlap limitation documented.

- [x] **E2 [sonnet] — RTTM + speaker-labeled output formats.** Add
      `src/diarize/format.rs` with NIST **RTTM** export for `DiarizeResult` and a `[SPEAKER_k]`-prefixed
      transcript for `SpeakerTranscript`. RTTM is the lingua franca that makes Batch F's DER checkable.
      **Files:** `src/diarize/format.rs` (new), ~~`examples/`~~ (example deferred to F3)
      **✅ done (green):** `write_rttm<W: Write>` + `rttm_string` (9-field NIST lines, chan 1, 3-decimal
      tbeg/tdur, `duration=(end-start).max(0)`, `speaker_{id}`, `<NA>` fills) and `labeled_transcript` /
      `labeled_transcript_timed`. 12 **byte-exact** tests (incl. negative-span clamp, f32 `{:.3}` rounding
      verified against standalone rustc). No `.unwrap()`; `rttm_string` uses justified `.expect` on
      infallible in-memory writes. `examples/diarize.rs` intentionally left to **F3** (not created here).

### Batch F — evaluation (do this BEFORE claiming any accuracy number)

> ## 🏁 Batch F — EVALUATION COMPLETE (2026-07-11)
> **F1 + F2 + F3 landed, each independently adversarial-reviewed, and reconciled/re-verified together:
> full gate GREEN (orchestrator-verified)** — `cargo nextest run --all-features` → **579/579 tests pass,
> 0 skipped** (518 under `--features diarization` alone)
> (includes `diarize::metrics::tests` DER/JER + Hungarian-mapping unit tests, `diarize::cluster::tests`
> AHC/spectral exact-partition-recovery tests, `test_utils::multispeaker_tests` synthetic-fixture tests,
> and `tests/diarization_synthetic.rs` end-to-end mixture tests). `cargo clippy --all-targets -D warnings`
> clean under `--features diarization`, `diarization,serde`, **and** `diarization,onnx`.
> `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` clean under `--features diarization` **and**
> `diarization,onnx`. oxiwhisper can now honestly measure its own diarization accuracy:
> ```rust
> use oxiwhisper::diarize::metrics::{der, jer, parse_rttm, DerOptions};
> let reference = parse_rttm(rttm_text)?;                      // NIST RTTM -> Vec<RttmSegment>
> let report = der(&reference, &hypothesis.segments, &DerOptions::default()); // Hungarian-mapped DerReport
> let jaccard_error = jer(&reference, &hypothesis.segments, &DerOptions::default());
> println!("DER {:.3}  JER {:.3}", report.der, jaccard_error);
> ```
> and `examples/diarize.rs` gives a CLI over the whole pipeline (`--speakers`, `--clustering`,
> `--embedder whisper|ecapa:<path>`, `--rttm`), documented in README's "Speaker Diarization" section.
> **D-bump was intentionally NOT part of this run** (deferred — see banner above Batch E).

- [x] **F1 [opus] — DER/JER metrics + Hungarian speaker mapping.** `src/diarize/metrics.rs`:
      RTTM reference parser; optimal reference↔hypothesis speaker assignment via the Hungarian
      algorithm; **Diarization Error Rate** = (miss + false-alarm + confusion)/total-speech with a
      configurable collar (default 0.25 s) and overlap-skip toggle; **Jaccard Error Rate**. No stubbed
      constants — every term computed from the segment timelines.
      **Files:** `src/diarize/metrics.rs` (new)

- [x] **F2 [opus] — Non-vacuous synthetic multi-speaker fixtures + real assertions.**
      *(Heeds the v0.10 lesson: the synthetic model emitting one constant token made whole suites pass
      on empty data. Do NOT repeat it here.)* Build a deterministic 2- and 3-speaker mixer in
      `src/test_utils.rs` — concatenate/overlay distinct synthetic sources (different formant/pitch
      profiles) with **known** ground-truth boundaries → RTTM. Then:
    - `cluster.rs` unit tests: separable Gaussian blobs with known labels → AHC and spectral recover
      the exact label partition (assert cluster count **and** per-point assignment, not `is_ok()`).
    - eigengap/dendrogram estimator returns the planted `k`.
    - Hungarian mapping + DER unit test against a hand-computed tiny timeline (DER known by pencil).
    - end-to-end `diarize()` on the synthetic mixture with `WhisperEncoderEmbedder`: assert
      `num_speakers == 2` and `DER < τ`. If the speaker-invariant baseline can't clear a real τ, say so
      in the test comment and pin it as characterization — never assert a vacuous bound to look green.
      **Files:** `src/test_utils.rs`, `src/diarize/*.rs`, `tests/diarization_synthetic.rs` (new)

- [x] **F3 [sonnet] — `examples/diarize.rs` + docs.** CLI: audio in → RTTM / speaker-labeled
      transcript out, `--speakers N`, `--embedder ecapa:<path>|whisper`, `--clustering ahc|spectral`.
      README section documenting the external-model requirement and the baseline caveat verbatim.
      **Files:** `examples/diarize.rs` (new), `README.md`, `Cargo.toml` (`[[example]]`, `required-features`)

- [x] **D-bump — Version aligned to 0.1.2; CHANGELOG consolidated.** (2026-07-11) Corrected from
      the earlier "0.1.5" assumption: only `v0.1.0`/`v0.1.1` were ever tagged, so the three
      never-released CHANGELOG sections (0.1.2 / 0.1.3 / 0.1.4, all dated 2026-06-10) were collapsed
      into a single `## [0.1.2] - 2026-07-11` covering everything since v0.1.1 (symphonia 0.6 +
      translation + word-timestamps + **speaker diarization**, with an honest low-accuracy caveat),
      and `Cargo.toml` set 0.1.4 → **0.1.2** to match the branch name and the next crates.io publish.
      Implement (sonnet) → adversarial-verify (opus) pass: clean; clippy `--all-features` green as
      `oxiwhisper v0.1.2`. **Files:** `Cargo.toml`, `CHANGELOG.md`

### Batch G — overlap & quality (research; v0.12+, do NOT promise in 0.1.2)

- [ ] **G1 [opus] — Overlap-aware diarization.** The embedding-clustering pipeline assigns exactly
      one speaker per frame and **structurally cannot** represent overlapped speech. Real handling needs
      an overlap-detection head or an EEND-style permutation-invariant model. Scope honestly: either an
      ONNX overlap-detector that splits ambiguous frames, or document that overlap is out of scope and
      report its contribution to DER separately. Research + written findings first — no silent partial.
- [ ] **G2 [sonnet] — Neural VAD backend** (ONNX Silero-class) as an optional `SpeechActivity` trait
      impl alongside the energy `vad.rs`, for noisy-audio robustness.
- [ ] **G3 [opus] — VBx resegmentation** (Variational-Bayes HMM over x-vectors) as a higher-accuracy
      alternative to AHC/spectral cuts, if `scirs2-linalg` supplies the needed primitives.
