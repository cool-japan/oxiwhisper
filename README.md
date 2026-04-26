# OxiWhisper

Pure Rust Whisper speech-to-text inference engine. Zero C/C++ dependencies.

> 17,101 LoC | 399 tests | 24 modules | 10 examples | Apache-2.0

## Status

| Component | Status | Tests |
|-----------|--------|-------|
| Core inference (encoder/decoder) | Stable | 399 passing |
| Quantized inference (Q4_0/Q5_0/Q8_0) | Stable | 40+ |
| SIMD kernels (AVX2/NEON/WASM) | Stable | 15+ |
| Streaming API | Stable | 8+ |
| Word timestamps (monotonic peak) | Stable | 15+ |
| ONNX model loading | Stable | 13 |

## Features

### Inference
- GGML and GGUF model loading (`ggml-tiny.bin`, `ggml-tiny.gguf`, etc.). Both formats are supported; the loader auto-detects the format from the file magic bytes — just change the path, no code change needed.
- Q4_0, Q5_0, and Q8_0 quantized inference with dequantize-on-the-fly GEMV
- SIMD-accelerated dot products: AVX2+FMA (x86_64), NEON (aarch64), simd128 (WASM)
- `matrixmultiply::sgemm` for attention QK^T and scores@V with stride-based transpose
- Arc copy-on-write KV cache for beam search
- Zero-copy tensor reshape, in-place activations (GELU, softmax, layer norm)
- Pre-allocated inference buffers for latency-sensitive applications

### Decoding
- Greedy decoding, beam search (configurable width), temperature sampling
- Top-k and nucleus (top-p) filtering
- Automatic language detection (99 languages)
- Timestamp segments with start/end times and per-segment confidence
- Token-level log-probabilities
- Initial prompt conditioning for domain-specific vocabulary
- Suppress tokens to block specific token IDs
- No-repeat-ngram penalty to prevent hallucination loops
- Compression ratio filtering for hallucination detection
- Previous context conditioning for cross-chunk coherence

### Audio & Analysis
- Pure Rust WAV loader (PCM 8/16/24/32-bit, IEEE float, multi-channel)
- Automatic resampling to 16 kHz mono
- Voice Activity Detection with adaptive noise floor thresholding
- VAD-aware chunking for long audio at silence boundaries
- Word-level timestamps via cross-attention alignment. `align_tokens_dp_dtw` is the canonical function — a Sakoe-Chiba-banded DP-DTW with traceback. `align_tokens_dtw` is an alias with default band width. `align_tokens_monotonic_peak` is the fast-path alternative (argmax-per-row, no DP).
- Log-mel spectrogram computation using OxiFFT

### API
- `from_file(path)` — load a GGML or GGUF model from a file path (format auto-detected)
- `from_file_mmap(path)` — memory-mapped model loading; lower peak RSS for medium/large models
- `transcribe()`, `transcribe_segmented()`, `transcribe_timed()`
- `transcribe_long()`, `transcribe_long_segmented()`, `transcribe_long_with_vad()`
- `transcribe_batch()` for multiple audio clips
- `transcribe_to_srt()`, `transcribe_to_vtt()` subtitle export
- `stream()` returning `StreamTranscriber` for real-time processing
- `encoder_output()` for embedding extraction
- `mel_spectrogram()` for audio analysis
- `model_stats()` for memory/parameter statistics
- `oxiwhisper::threading::set_thread_count(n)` — configure the rayon thread-pool size when the `parallel` feature is enabled
- Optional `serde` feature for JSON serialization via `to_json()`

## Quick Start

```rust
use oxiwhisper::{WhisperModel, TranscribeOptions};
use std::path::Path;

fn main() -> Result<(), oxiwhisper::OxiWhisperError> {
    let model = WhisperModel::from_file(Path::new("ggml-tiny.bin"))?;
    let audio = oxiwhisper::audio::load_wav(Path::new("audio.wav"))?;
    let text = model.transcribe(&audio, &TranscribeOptions::default())?;
    println!("{text}");
    Ok(())
}
```

## Supported Models

| Model  | Parameters | Size (f32) | Size (Q4_0) | Size (Q5_0) |
|--------|-----------|------------|-------------|-------------|
| tiny   | 39M       | ~150 MB    | ~40 MB      | ~48 MB      |
| base   | 74M       | ~290 MB    | ~80 MB      | ~95 MB      |
| small  | 244M      | ~950 MB    | ~250 MB     | ~300 MB     |
| medium | 769M      | ~3.0 GB    | ~800 MB     | ~950 MB     |
| large  | 1.5B      | ~6.0 GB    | ~1.5 GB     | ~1.8 GB     |

## Segmented Transcription

Get segment-level output with timestamps and confidence:

```rust
use oxiwhisper::{WhisperModel, TranscribeOptions};
use std::path::Path;

fn main() -> Result<(), oxiwhisper::OxiWhisperError> {
    let model = WhisperModel::from_file(Path::new("ggml-tiny.bin"))?;
    let audio = oxiwhisper::audio::load_wav(Path::new("audio.wav"))?;
    let opts = TranscribeOptions {
        timestamps: true,
        ..TranscribeOptions::default()
    };
    let result = model.transcribe_segmented(&audio, &opts)?;
    for seg in &result.segments {
        println!("[{:.1}s - {:.1}s] {} (conf: {:.3})", seg.start, seg.end, seg.text, seg.confidence);
    }
    Ok(())
}
```

## Streaming API

Process audio incrementally with `StreamTranscriber`:

```rust
use oxiwhisper::{WhisperModel, TranscribeOptions};
use std::path::Path;

fn main() -> Result<(), oxiwhisper::OxiWhisperError> {
    let model = WhisperModel::from_file(Path::new("ggml-tiny.bin"))?;
    let mut stream = model.stream(TranscribeOptions::default());

    // Feed audio in arbitrary-sized chunks
    stream.push_audio(&[0.0f32; 8000]);
    stream.push_audio(&[0.0f32; 8000]);

    // Process available 30-second segments
    while let Some(seg) = stream.next_segment() {
        let seg = seg?;
        println!("[{:.1}s - {:.1}s] {}", seg.start, seg.end, seg.text);
    }

    // Flush remaining audio
    let result = stream.finish()?;
    println!("{}", result.text);
    Ok(())
}
```

## Subtitle Export

Generate SRT or WebVTT subtitles directly:

```rust
use oxiwhisper::{WhisperModel, TranscribeOptions};
use std::path::Path;

fn main() -> Result<(), oxiwhisper::OxiWhisperError> {
    let model = WhisperModel::from_file(Path::new("ggml-tiny.bin"))?;
    let audio = oxiwhisper::audio::load_wav(Path::new("audio.wav"))?;
    let srt = model.transcribe_to_srt(&audio, &TranscribeOptions::default())?;
    println!("{srt}");
    Ok(())
}
```

## Advanced Options

```rust
use oxiwhisper::TranscribeOptions;

let opts = TranscribeOptions {
    language: Some("ja"),           // Force Japanese (None = auto-detect)
    beam_width: 5,                  // Beam search with width 5
    temperature: 0.0,               // Deterministic (>0 enables sampling)
    top_k: 0,                       // Disabled (0 = all tokens eligible)
    top_p: 1.0,                     // Disabled (1.0 = no nucleus filtering)
    timestamps: true,               // Enable segment timestamps
    initial_prompt: Some("Hello"),  // Condition on domain vocabulary
    suppress_tokens: None,          // Block specific token IDs
    no_repeat_ngram_size: 3,        // Prevent 3-gram repetition
    compression_ratio_threshold: 2.4, // Hallucination detection
    previous_tokens: None,          // Cross-chunk context
};
```

## Feature Flags

| Feature      | Description                                                                                         | Default |
|--------------|-----------------------------------------------------------------------------------------------------|---------|
| `timing`     | Print per-phase timing diagnostics to stderr                                                        | off     |
| `onnx`       | Enable ONNX model loading via `oxionnx`                                                             | off     |
| `serde`      | JSON serialization for `TranscribeResult`, etc.                                                     | off     |
| `parallel`   | Per-head parallelism in encoder/decoder attention via rayon. WASM-safe (not in default features). Use `oxiwhisper::threading::set_thread_count(n)` to configure the pool size. | off |
| `audio-flac` | FLAC audio decoding via symphonia                                                                   | off     |
| `audio-ogg`  | Ogg/Vorbis audio decoding via symphonia                                                             | off     |
| `audio-mp3`  | MP3 audio decoding via symphonia                                                                    | off     |
| `audio-aac`  | AAC/M4A audio decoding via symphonia                                                                | off     |
| `audio-opus` | Opus audio decoding via opus-decoder + ogg                                                          | off     |
| `audio-all`  | Enables all audio format decoders above                                                             | off     |
| `test-utils` | Exposes internal test helpers. For integration testing only; not for normal use.                    | off     |

## Architecture

```
Audio (WAV/f32) ─→ Mel Spectrogram (OxiFFT) ─→ Encoder (Conv + Transformer)
                                                         │
                                                         ▼
Text ←─ Tokenizer ←─ Decoder (Autoregressive + KV Cache + Beam Search)
```

**24 modules**: types, tensor, fft, mel, mel_filters, model, quantize, linear,
attention, encoder, decoder, beam_search, decode_utils, tokenizer, audio, vad,
stream, subtitle, dtw, hallucination, onnx_loader, test_utils, threading, whisper_model

## Examples

| Example | Description |
|---------|-------------|
| `transcribe` | Simple CLI: `cargo run --example transcribe -- model.bin audio.wav` |
| `streaming` | Real-time streaming with `StreamTranscriber` |
| `batch_transcribe` | Multi-file batch transcription |
| `bench` | Performance benchmarking with RTF reporting |
| `profile_attention` | Attention kernel profiling (sgemm vs tiled) |

## License

Apache-2.0

Copyright (c) 2025-2026 COOLJAPAN OU (Team Kitasan)
