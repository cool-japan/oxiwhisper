//! # OxiWhisper
//!
//! Pure Rust Whisper speech-to-text inference engine with zero C/C++ dependencies.
//!
//! OxiWhisper loads GGML-format Whisper models and transcribes audio to text,
//! supporting quantized inference (Q4_0, Q5_0, Q8_0), streaming, beam search,
//! word-level timestamps, and SIMD-accelerated kernels (AVX2, NEON, WASM simd128).
//!
//! ## Quick Start
//!
//! ```ignore
//! use oxiwhisper::{WhisperModel, TranscribeOptions};
//! use std::path::Path;
//!
//! let model = WhisperModel::from_file(Path::new("ggml-tiny.bin"))?;
//! let audio = oxiwhisper::audio::load_wav(Path::new("audio.wav"))?;
//! let text = model.transcribe(&audio, &TranscribeOptions::default())?;
//! println!("{text}");
//! ```

pub mod attention;
pub mod audio;
pub mod beam_search;
pub mod decode_utils;
pub mod decoder;
pub mod dtw;
pub mod encoder;
pub mod fft;
pub mod hallucination;
pub mod linear;
pub mod mel;
pub mod mel_filters;
pub mod model;
#[cfg(feature = "onnx")]
pub mod onnx_loader;
pub mod quantize;
pub mod stream;
pub mod subtitle;
pub mod tensor;
#[cfg(test)]
pub mod test_utils;
pub mod tokenizer;
pub mod types;
pub mod vad;

pub use types::*;

use std::path::Path;
use std::sync::Arc;

use hallucination::{compute_segment_confidences, is_likely_hallucination};
use types::validate_options;

pub struct WhisperModel {
    model_data: Arc<model::ModelData>,
}

impl WhisperModel {
    /// Load a GGML whisper model file (e.g., `ggml-tiny.bin`).
    pub fn from_file(path: &Path) -> Result<Self, OxiWhisperError> {
        #[cfg(feature = "timing")]
        eprintln!("Loading whisper model from: {}", path.display());

        let model_data = model::ModelData::load(path).map_err(OxiWhisperError::InvalidModel)?;

        #[cfg(feature = "timing")]
        eprintln!(
            "Model loaded: {} vocab, {} audio layers, {} text layers, d_model={}",
            model_data.hparams.n_vocab,
            model_data.hparams.n_audio_layer,
            model_data.hparams.n_text_layer,
            model_data.hparams.n_audio_state,
        );

        Ok(Self {
            model_data: Arc::new(model_data),
        })
    }

    /// Load a Whisper model from an ONNX file.
    ///
    /// Supports both single-file ONNX models and split encoder/decoder models.
    /// Automatically detects HuggingFace Optimum and OpenAI naming conventions.
    ///
    /// # Arguments
    /// * `config` - Configuration specifying model path(s) and optional vocabulary file.
    ///
    /// # Example
    /// ```no_run
    /// use oxiwhisper::{WhisperModel, onnx_loader::OnnxModelConfig};
    /// use std::path::PathBuf;
    ///
    /// let config = OnnxModelConfig {
    ///     model_path: PathBuf::from("whisper-tiny.onnx"),
    ///     decoder_path: None,
    ///     vocab_path: Some(PathBuf::from("vocab.txt")),
    /// };
    /// let model = WhisperModel::from_onnx(&config).unwrap();
    /// ```
    #[cfg(feature = "onnx")]
    pub fn from_onnx(config: &onnx_loader::OnnxModelConfig) -> Result<Self, OxiWhisperError> {
        #[cfg(feature = "timing")]
        eprintln!(
            "Loading ONNX whisper model from: {}",
            config.model_path.display()
        );

        let model_data = onnx_loader::load_onnx(config).map_err(OxiWhisperError::InvalidModel)?;

        #[cfg(feature = "timing")]
        eprintln!(
            "ONNX model loaded: {} vocab, {} audio layers, {} text layers, d_model={}",
            model_data.hparams.n_vocab,
            model_data.hparams.n_audio_layer,
            model_data.hparams.n_text_layer,
            model_data.hparams.n_audio_state,
        );

        Ok(Self {
            model_data: Arc::new(model_data),
        })
    }

    /// Return metadata about the loaded model (dimensions, layer counts, etc.).
    pub fn info(&self) -> ModelInfo {
        let hp = &self.model_data.hparams;
        ModelInfo {
            n_vocab: hp.n_vocab,
            n_audio_layers: hp.n_audio_layer,
            n_text_layers: hp.n_text_layer,
            d_model: hp.n_audio_state,
            n_mels: hp.n_mels,
            n_audio_heads: hp.n_audio_head,
            n_text_heads: hp.n_text_head,
            n_audio_ctx: hp.n_audio_ctx,
            n_text_ctx: hp.n_text_ctx,
        }
    }

    /// Create a pre-allocated inference buffer sized for this model.
    ///
    /// The returned buffer can be reused across multiple calls to
    /// [`transcribe_with_buffer`](Self::transcribe_with_buffer) to reduce
    /// allocation overhead in latency-sensitive applications.
    pub fn create_buffer(&self) -> InferenceBuffer {
        let hp = &self.model_data.hparams;
        let mel_capacity = hp.n_mels * 3000; // ~30 s of audio
        InferenceBuffer {
            mel_buf: Vec::with_capacity(mel_capacity),
        }
    }

    /// Transcribe using pre-allocated buffers to reduce allocation overhead.
    ///
    /// Semantically identical to [`transcribe`](Self::transcribe), but reuses
    /// the mel-spectrogram scratch buffer instead of allocating a new one.
    pub fn transcribe_with_buffer(
        &self,
        audio: &[f32],
        opts: &TranscribeOptions<'_>,
        buffer: &mut InferenceBuffer,
    ) -> Result<String, OxiWhisperError> {
        validate_options(opts)?;
        if audio.is_empty() {
            return Err(OxiWhisperError::InferenceFailed("Empty audio".into()));
        }

        let md = &self.model_data;
        #[cfg(feature = "timing")]
        let t0 = std::time::Instant::now();

        // Reuse mel_buf: clear and refill
        buffer.mel_buf.clear();
        let mel_fresh = mel::log_mel_spectrogram(audio, &md.mel_filters);
        buffer.mel_buf.extend_from_slice(&mel_fresh);

        let n_mels = md.hparams.n_mels;
        let n_frames = buffer.mel_buf.len() / n_mels;
        let mel =
            tensor::Tensor::from_vec(std::mem::take(&mut buffer.mel_buf), &[n_mels, n_frames]);
        #[cfg(feature = "timing")]
        eprintln!(
            "mel: {}ms  ({} frames for {}ms audio)",
            t0.elapsed().as_millis(),
            n_frames,
            audio.len() * 1000 / 16000
        );

        #[cfg(feature = "timing")]
        let t1 = std::time::Instant::now();
        let encoded = encoder::encode(&mel, md).map_err(OxiWhisperError::InvalidModel)?;
        #[cfg(feature = "timing")]
        eprintln!(
            "encoder: {}ms  (enc_len={})",
            t1.elapsed().as_millis(),
            encoded.shape[0]
        );

        // Return the mel buffer for reuse
        buffer.mel_buf = mel.data;

        #[cfg(feature = "timing")]
        let t2 = std::time::Instant::now();
        let decode_result =
            decoder::decode(&encoded, md, opts).map_err(OxiWhisperError::InferenceFailed)?;
        #[cfg(feature = "timing")]
        eprintln!(
            "decoder: {}ms  ({} tokens)",
            t2.elapsed().as_millis(),
            decode_result.tokens.len()
        );

        let text = tokenizer::decode(&decode_result.tokens, &md.vocab);
        #[cfg(feature = "timing")]
        eprintln!("total: {}ms", t0.elapsed().as_millis());

        Ok(text)
    }

    /// Transcribe 16 kHz mono f32 PCM audio to text.
    pub fn transcribe(
        &self,
        audio: &[f32],
        opts: &TranscribeOptions<'_>,
    ) -> Result<String, OxiWhisperError> {
        validate_options(opts)?;
        if audio.is_empty() {
            return Err(OxiWhisperError::InferenceFailed("Empty audio".into()));
        }

        let md = &self.model_data;
        #[cfg(feature = "timing")]
        let t0 = std::time::Instant::now();

        let mel_data = mel::log_mel_spectrogram(audio, &md.mel_filters);
        let n_mels = md.hparams.n_mels;
        let n_frames = mel_data.len() / n_mels;
        let mel = tensor::Tensor::from_vec(mel_data, &[n_mels, n_frames]);
        #[cfg(feature = "timing")]
        eprintln!(
            "mel: {}ms  ({} frames for {}ms audio)",
            t0.elapsed().as_millis(),
            n_frames,
            audio.len() * 1000 / 16000
        );

        #[cfg(feature = "timing")]
        let t1 = std::time::Instant::now();
        let encoded = encoder::encode(&mel, md).map_err(OxiWhisperError::InvalidModel)?;
        #[cfg(feature = "timing")]
        eprintln!(
            "encoder: {}ms  (enc_len={})",
            t1.elapsed().as_millis(),
            encoded.shape[0]
        );

        #[cfg(feature = "timing")]
        let t2 = std::time::Instant::now();
        let decode_result =
            decoder::decode(&encoded, md, opts).map_err(OxiWhisperError::InferenceFailed)?;
        #[cfg(feature = "timing")]
        eprintln!(
            "decoder: {}ms  ({} tokens)",
            t2.elapsed().as_millis(),
            decode_result.tokens.len()
        );

        let text = tokenizer::decode(&decode_result.tokens, &md.vocab);
        #[cfg(feature = "timing")]
        eprintln!("total: {}ms", t0.elapsed().as_millis());

        Ok(text)
    }

    /// Transcribe 16 kHz mono f32 PCM audio to text, returning per-phase timing.
    ///
    /// Identical to [`transcribe`](Self::transcribe) but also reports how long
    /// each phase (mel, encoder, decoder) took.
    pub fn transcribe_timed(
        &self,
        audio: &[f32],
        opts: &TranscribeOptions<'_>,
    ) -> Result<(String, TranscribeTiming), OxiWhisperError> {
        validate_options(opts)?;
        if audio.is_empty() {
            return Err(OxiWhisperError::InferenceFailed("Empty audio".into()));
        }

        let md = &self.model_data;
        let t_total_start = std::time::Instant::now();

        // --- Mel spectrogram ---
        let t_mel_start = std::time::Instant::now();
        let mel_data = mel::log_mel_spectrogram(audio, &md.mel_filters);
        let n_mels = md.hparams.n_mels;
        let n_frames = mel_data.len() / n_mels;
        let mel = tensor::Tensor::from_vec(mel_data, &[n_mels, n_frames]);
        let mel_dur = t_mel_start.elapsed();

        // --- Encoder ---
        let t_enc_start = std::time::Instant::now();
        let encoded = encoder::encode(&mel, md).map_err(OxiWhisperError::InvalidModel)?;
        let enc_dur = t_enc_start.elapsed();

        // --- Decoder ---
        let t_dec_start = std::time::Instant::now();
        let decode_result =
            decoder::decode(&encoded, md, opts).map_err(OxiWhisperError::InferenceFailed)?;
        let text = tokenizer::decode(&decode_result.tokens, &md.vocab);
        let dec_dur = t_dec_start.elapsed();

        let total_dur = t_total_start.elapsed();

        let timing = TranscribeTiming {
            mel: mel_dur,
            encoder: enc_dur,
            decoder: dec_dur,
            total: total_dur,
        };

        Ok((text, timing))
    }

    /// Transcribe audio of arbitrary length by chunking into 30-second segments.
    ///
    /// For audio <= 30s, this is equivalent to [`transcribe()`](Self::transcribe).
    /// For longer audio, splits into overlapping chunks and concatenates results.
    pub fn transcribe_long(
        &self,
        audio: &[f32],
        opts: &TranscribeOptions<'_>,
    ) -> Result<String, OxiWhisperError> {
        validate_options(opts)?;
        if audio.is_empty() {
            return Err(OxiWhisperError::InferenceFailed("Empty audio".into()));
        }

        const CHUNK_SAMPLES: usize = 16000 * 30; // 30 seconds
        const OVERLAP_SAMPLES: usize = 16000; // 1 second overlap
        const STEP_SAMPLES: usize = CHUNK_SAMPLES - OVERLAP_SAMPLES; // 29 seconds step

        if audio.len() <= CHUNK_SAMPLES {
            return self.transcribe(audio, opts);
        }

        // Try VAD-aware splitting first, fall back to fixed chunking
        let split_points = compute_split_points(audio, STEP_SAMPLES, CHUNK_SAMPLES);

        let mut texts: Vec<String> = Vec::new();
        let mut prev_text: Option<String> = None;
        for (chunk_start, chunk_end) in split_points {
            let chunk = &audio[chunk_start..chunk_end];
            let mut chunk_opts = opts.clone();
            // Use previous segment text as initial_prompt for cross-chunk coherence
            if let Some(ref text) = prev_text {
                chunk_opts.initial_prompt = Some(text);
            }
            let text = self.transcribe(chunk, &chunk_opts)?;
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                texts.push(trimmed.to_string());
                prev_text = Some(trimmed.to_string());
            } else {
                prev_text = None;
            }
        }

        Ok(texts.join(" "))
    }

    /// Transcribe long audio with segment-level output.
    ///
    /// For audio <= 30s, this is equivalent to
    /// [`transcribe_segmented()`](Self::transcribe_segmented).
    /// For longer audio, splits into overlapping chunks and concatenates results,
    /// adjusting segment timestamps to reflect their position in the original audio.
    pub fn transcribe_long_segmented(
        &self,
        audio: &[f32],
        opts: &TranscribeOptions<'_>,
    ) -> Result<TranscribeResult, OxiWhisperError> {
        validate_options(opts)?;
        if audio.is_empty() {
            return Err(OxiWhisperError::InferenceFailed("Empty audio".into()));
        }

        const CHUNK_SAMPLES: usize = 16000 * 30;
        const OVERLAP_SAMPLES: usize = 16000;
        const STEP_SAMPLES: usize = CHUNK_SAMPLES - OVERLAP_SAMPLES;

        if audio.len() <= CHUNK_SAMPLES {
            return self.transcribe_segmented(audio, opts);
        }

        let split_points = compute_split_points(audio, STEP_SAMPLES, CHUNK_SAMPLES);

        let mut all_texts: Vec<String> = Vec::new();
        let mut all_segments: Vec<Segment> = Vec::new();
        let mut detected_language: Option<String> = None;
        let mut prev_text: Option<String> = None;

        for (chunk_start, chunk_end) in &split_points {
            let chunk = &audio[*chunk_start..*chunk_end];
            let chunk_offset_seconds = *chunk_start as f32 / 16000.0;

            let mut chunk_opts = opts.clone();
            // Use previous segment text as initial_prompt for cross-chunk coherence
            if let Some(ref text) = prev_text {
                chunk_opts.initial_prompt = Some(text);
            }
            let result = self.transcribe_segmented(chunk, &chunk_opts)?;

            let trimmed = result.text.trim();
            if !trimmed.is_empty() {
                all_texts.push(trimmed.to_string());
            }

            // Offset segment timestamps by the chunk's position in the original audio
            for seg in result.segments {
                all_segments.push(Segment {
                    text: seg.text,
                    start: seg.start + chunk_offset_seconds,
                    end: seg.end + chunk_offset_seconds,
                    confidence: seg.confidence,
                    is_hallucination: seg.is_hallucination,
                });
            }

            // Save last segment text for cross-chunk context conditioning
            let trimmed_text = result.text.trim();
            prev_text = if trimmed_text.is_empty() {
                None
            } else {
                Some(trimmed_text.to_string())
            };

            // Use the first chunk's detected language
            if detected_language.is_none() {
                detected_language = result.language;
            }
        }

        Ok(TranscribeResult {
            text: all_texts.join(" "),
            segments: all_segments,
            language: detected_language,
        })
    }

    /// Transcribe long audio with custom VAD configuration.
    ///
    /// Similar to [`transcribe_long_segmented`](Self::transcribe_long_segmented) but
    /// allows fine-tuning the voice activity detection parameters used for chunk
    /// splitting. This is useful when working with noisy audio or when the default
    /// energy threshold produces sub-optimal split points.
    ///
    /// When `vad_config.adaptive_threshold` is `true`, the energy threshold is
    /// automatically estimated from the audio's noise floor, ignoring the fixed
    /// `energy_threshold` value.
    pub fn transcribe_long_with_vad(
        &self,
        audio: &[f32],
        opts: &TranscribeOptions<'_>,
        vad_config: &vad::VadConfig,
    ) -> Result<TranscribeResult, OxiWhisperError> {
        validate_options(opts)?;
        if audio.is_empty() {
            return Err(OxiWhisperError::InferenceFailed("Empty audio".into()));
        }

        const CHUNK_SAMPLES: usize = 16000 * 30;
        const OVERLAP_SAMPLES: usize = 16000;
        const STEP_SAMPLES: usize = CHUNK_SAMPLES - OVERLAP_SAMPLES;

        if audio.len() <= CHUNK_SAMPLES {
            return self.transcribe_segmented(audio, opts);
        }

        let split_points =
            compute_split_points_with_vad(audio, STEP_SAMPLES, CHUNK_SAMPLES, vad_config);

        let mut all_texts: Vec<String> = Vec::new();
        let mut all_segments: Vec<Segment> = Vec::new();
        let mut detected_language: Option<String> = None;
        let mut prev_text: Option<String> = None;

        for (chunk_start, chunk_end) in &split_points {
            let chunk = &audio[*chunk_start..*chunk_end];
            let chunk_offset_seconds = *chunk_start as f32 / 16000.0;

            let mut chunk_opts = opts.clone();
            if let Some(ref text) = prev_text {
                chunk_opts.initial_prompt = Some(text);
            }
            let result = self.transcribe_segmented(chunk, &chunk_opts)?;

            let trimmed = result.text.trim();
            if !trimmed.is_empty() {
                all_texts.push(trimmed.to_string());
            }

            for seg in result.segments {
                all_segments.push(Segment {
                    text: seg.text,
                    start: seg.start + chunk_offset_seconds,
                    end: seg.end + chunk_offset_seconds,
                    confidence: seg.confidence,
                    is_hallucination: seg.is_hallucination,
                });
            }

            let trimmed_text = result.text.trim();
            prev_text = if trimmed_text.is_empty() {
                None
            } else {
                Some(trimmed_text.to_string())
            };

            if detected_language.is_none() {
                detected_language = result.language;
            }
        }

        Ok(TranscribeResult {
            text: all_texts.join(" "),
            segments: all_segments,
            language: detected_language,
        })
    }

    /// Transcribe 16 kHz mono f32 PCM audio and return a [`TranscribeResult`]
    /// containing the full text, timed segments, and detected language.
    ///
    /// When `opts.timestamps` is `true`, the result includes [`Segment`]s with
    /// start/end times. When `false`, the `segments` field will be empty.
    pub fn transcribe_segmented(
        &self,
        audio: &[f32],
        opts: &TranscribeOptions<'_>,
    ) -> Result<TranscribeResult, OxiWhisperError> {
        validate_options(opts)?;
        if audio.is_empty() {
            return Err(OxiWhisperError::InferenceFailed("Empty audio".into()));
        }

        let md = &self.model_data;

        #[cfg(feature = "timing")]
        let t0 = std::time::Instant::now();

        let mel_data = mel::log_mel_spectrogram(audio, &md.mel_filters);
        let n_mels = md.hparams.n_mels;
        let n_frames = mel_data.len() / n_mels;
        let mel = tensor::Tensor::from_vec(mel_data, &[n_mels, n_frames]);

        #[cfg(feature = "timing")]
        eprintln!(
            "mel: {}ms  ({} frames for {}ms audio)",
            t0.elapsed().as_millis(),
            n_frames,
            audio.len() * 1000 / 16000
        );

        #[cfg(feature = "timing")]
        let t1 = std::time::Instant::now();
        let encoded = encoder::encode(&mel, md).map_err(OxiWhisperError::InvalidModel)?;
        #[cfg(feature = "timing")]
        eprintln!(
            "encoder: {}ms  (enc_len={})",
            t1.elapsed().as_millis(),
            encoded.shape[0]
        );

        #[cfg(feature = "timing")]
        let t2 = std::time::Instant::now();
        let decode_result =
            decoder::decode(&encoded, md, opts).map_err(OxiWhisperError::InferenceFailed)?;
        #[cfg(feature = "timing")]
        eprintln!(
            "decoder: {}ms  ({} tokens)",
            t2.elapsed().as_millis(),
            decode_result.tokens.len()
        );

        // Always produce the plain text (timestamps stripped).
        let text = tokenizer::decode(&decode_result.tokens, &md.vocab);

        // Parse segments if timestamps were requested.
        let segments = if opts.timestamps {
            let parsed = tokenizer::parse_segments(&decode_result.tokens, &md.vocab);
            let confidences = compute_segment_confidences(
                &decode_result.tokens,
                &decode_result.token_probs,
                parsed.len(),
            );
            parsed
                .into_iter()
                .zip(confidences)
                .map(|((start, end, seg_text), conf)| {
                    let is_hall =
                        is_likely_hallucination(&seg_text, opts.compression_ratio_threshold);
                    Segment {
                        text: seg_text,
                        start,
                        end,
                        confidence: conf,
                        is_hallucination: is_hall,
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        #[cfg(feature = "timing")]
        eprintln!("total: {}ms", t0.elapsed().as_millis());

        Ok(TranscribeResult {
            text,
            segments,
            language: decode_result.detected_language,
        })
    }

    /// Transcribe multiple audio clips sequentially.
    ///
    /// Returns one result per input audio. Each audio is processed independently.
    /// Errors in individual items do not affect others.
    pub fn transcribe_batch(
        &self,
        audios: &[&[f32]],
        opts: &TranscribeOptions<'_>,
    ) -> Vec<Result<TranscribeResult, OxiWhisperError>> {
        // Validate once; on failure, return the same error for every input.
        if let Err(e) = validate_options(opts) {
            let msg = format!("{e}");
            return audios
                .iter()
                .map(|_| Err(OxiWhisperError::ConfigError(msg.clone())))
                .collect();
        }
        audios
            .iter()
            .map(|audio| self.transcribe_segmented(audio, opts))
            .collect()
    }

    /// Transcribe audio and return SRT-formatted subtitles.
    /// Automatically enables timestamps.
    pub fn transcribe_to_srt(
        &self,
        audio: &[f32],
        opts: &TranscribeOptions<'_>,
    ) -> Result<String, OxiWhisperError> {
        let mut opts = opts.clone();
        opts.timestamps = true;
        let result = self.transcribe_segmented(audio, &opts)?;
        Ok(subtitle::to_srt(&result.segments))
    }

    /// Transcribe audio and return WebVTT-formatted subtitles.
    /// Automatically enables timestamps.
    pub fn transcribe_to_vtt(
        &self,
        audio: &[f32],
        opts: &TranscribeOptions<'_>,
    ) -> Result<String, OxiWhisperError> {
        let mut opts = opts.clone();
        opts.timestamps = true;
        let result = self.transcribe_segmented(audio, &opts)?;
        Ok(subtitle::to_vtt(&result.segments))
    }

    /// Create a streaming transcriber that accumulates audio and processes
    /// 30-second chunks incrementally.
    pub fn stream(&self, opts: TranscribeOptions<'static>) -> stream::StreamTranscriber<'_> {
        stream::StreamTranscriber::new(self, opts)
    }

    /// Extract the encoder's output representation for the given audio.
    ///
    /// Returns a tensor of shape `[seq_len, n_audio_state]` where `seq_len = n_frames / 2`.
    /// Useful for embedding extraction, similarity search, or downstream tasks.
    pub fn encoder_output(&self, audio: &[f32]) -> Result<tensor::Tensor, OxiWhisperError> {
        if audio.is_empty() {
            return Err(OxiWhisperError::InferenceFailed("Empty audio".into()));
        }
        let md = &self.model_data;
        let mel_data = mel::log_mel_spectrogram(audio, &md.mel_filters);
        let n_mels = md.hparams.n_mels;
        let n_frames = mel_data.len() / n_mels;
        let mel_tensor = tensor::Tensor::from_vec(mel_data, &[n_mels, n_frames]);
        encoder::encode(&mel_tensor, md).map_err(OxiWhisperError::InvalidModel)
    }

    /// Compute the log-mel spectrogram for the given audio.
    ///
    /// Returns a tensor of shape `[n_mels, n_frames]`. Useful for audio analysis,
    /// visualization, or reusing the mel computation across multiple inference runs.
    pub fn mel_spectrogram(&self, audio: &[f32]) -> tensor::Tensor {
        let md = &self.model_data;
        let mel_data = mel::log_mel_spectrogram(audio, &md.mel_filters);
        let n_mels = md.hparams.n_mels;
        let n_frames = mel_data.len() / n_mels;
        tensor::Tensor::from_vec(mel_data, &[n_mels, n_frames])
    }

    /// Return statistics about the loaded model.
    pub fn model_stats(&self) -> ModelStats {
        let mut total = 0usize;
        let mut quantized = 0usize;
        let mut float32 = 0usize;
        let mut mem = 0usize;

        for tensor in self.model_data.tensors.values() {
            let n = tensor.numel();
            float32 += n;
            total += n;
            mem += n * 4;
        }
        for qt in self.model_data.quantized_tensors.values() {
            let n = qt.numel();
            quantized += n;
            total += n;
            mem += qt.raw.len();
        }

        ModelStats {
            total_params: total,
            quantized_params: quantized,
            float32_params: float32,
            estimated_memory_bytes: mem,
        }
    }
}

// ---------------------------------------------------------------------------
// Chunking helpers for long-audio transcription
// ---------------------------------------------------------------------------

/// Sample rate assumed by Whisper (16 kHz).
const SAMPLE_RATE: usize = 16000;

/// Compute split points for long audio, with optional VAD-aware boundary adjustment.
///
/// Returns a list of `(start_sample, end_sample)` ranges covering the entire audio.
/// Each range is at most `chunk_samples` long. The function tries to place split
/// boundaries at silence gaps detected by VAD; if no suitable gap is found near a
/// boundary it falls back to the fixed-step position.
fn compute_split_points(
    audio: &[f32],
    step_samples: usize,
    chunk_samples: usize,
) -> Vec<(usize, usize)> {
    let vad_config = vad::VadConfig {
        energy_threshold: 0.01,
        min_speech_ms: 100,
        min_silence_ms: 100,
        frame_size_ms: 30,
        ..vad::VadConfig::default()
    };
    compute_split_points_with_vad(audio, step_samples, chunk_samples, &vad_config)
}

/// Compute split points for long audio using a custom [`vad::VadConfig`].
///
/// Returns a list of `(start_sample, end_sample)` ranges covering the entire audio.
/// Each range is at most `chunk_samples` long. The function tries to place split
/// boundaries at silence gaps detected by VAD; if no suitable gap is found near a
/// boundary it falls back to the fixed-step position.
fn compute_split_points_with_vad(
    audio: &[f32],
    step_samples: usize,
    chunk_samples: usize,
    vad_config: &vad::VadConfig,
) -> Vec<(usize, usize)> {
    if audio.is_empty() || step_samples == 0 || chunk_samples == 0 {
        return Vec::new();
    }

    if audio.len() <= chunk_samples {
        return vec![(0, audio.len())];
    }

    // Run VAD to find silence gaps
    let speech_segments = vad::detect_speech(audio, SAMPLE_RATE, vad_config);

    // Build a list of silence gaps from speech segments
    let silence_gaps = build_silence_gaps(&speech_segments, audio.len());

    let mut points: Vec<(usize, usize)> = Vec::new();
    let mut pos: usize = 0;

    // Search window: +/-2 seconds around the nominal boundary
    let search_window = SAMPLE_RATE * 2;

    while pos < audio.len() {
        let nominal_end = (pos + chunk_samples).min(audio.len());

        if nominal_end >= audio.len() {
            // Last chunk covers the rest
            points.push((pos, audio.len()));
            break;
        }

        // The nominal next-chunk start (before overlap adjustment)
        let nominal_boundary = pos + step_samples;

        // Try to find a silence gap near the nominal boundary
        let adjusted_boundary = find_silence_near(nominal_boundary, search_window, &silence_gaps)
            .unwrap_or(nominal_boundary);

        // Clamp adjusted_boundary: must advance beyond pos, but not beyond
        // pos + chunk_samples (so one chunk can cover the gap).
        let next_pos = adjusted_boundary.max(pos + 1).min(pos + chunk_samples);

        // The chunk covers [pos, chunk_end) -- must reach next_pos for seamless coverage
        let chunk_end = next_pos.min(audio.len());
        points.push((pos, chunk_end));

        pos = next_pos;
    }

    points
}

/// A silence gap (start_sample, end_sample).
struct SilenceGap {
    start: usize,
    end: usize,
}

impl SilenceGap {
    fn center(&self) -> usize {
        (self.start + self.end) / 2
    }
}

/// Build silence gaps from speech segments. Gaps are the regions between
/// consecutive speech segments, plus leading/trailing silence.
fn build_silence_gaps(speech_segments: &[vad::SpeechSegment], audio_len: usize) -> Vec<SilenceGap> {
    let mut gaps = Vec::new();

    if speech_segments.is_empty() {
        // Entire audio is silence
        if audio_len > 0 {
            gaps.push(SilenceGap {
                start: 0,
                end: audio_len,
            });
        }
        return gaps;
    }

    // Leading silence
    if speech_segments[0].start > 0 {
        gaps.push(SilenceGap {
            start: 0,
            end: speech_segments[0].start,
        });
    }

    // Gaps between speech segments
    for pair in speech_segments.windows(2) {
        if pair[1].start > pair[0].end {
            gaps.push(SilenceGap {
                start: pair[0].end,
                end: pair[1].start,
            });
        }
    }

    // Trailing silence
    if let Some(last) = speech_segments.last()
        && last.end < audio_len
    {
        gaps.push(SilenceGap {
            start: last.end,
            end: audio_len,
        });
    }

    gaps
}

/// Find the center of the silence gap closest to `target` within `+/-window` samples.
/// Returns `None` if no gap overlaps the search window.
fn find_silence_near(target: usize, window: usize, gaps: &[SilenceGap]) -> Option<usize> {
    let search_start = target.saturating_sub(window);
    let search_end = target + window;

    let mut best: Option<(usize, usize)> = None; // (distance, center)

    for gap in gaps {
        // Check if the gap overlaps the search window
        if gap.end <= search_start || gap.start >= search_end {
            continue;
        }

        // Use the point in the gap closest to target, clamped to the search window.
        // This avoids returning a distant center when a huge gap spans the window.
        let clamped = gap
            .center()
            .max(search_start)
            .min(search_end)
            .max(gap.start)
            .min(gap.end);
        let dist = clamped.abs_diff(target);

        match best {
            Some((best_dist, _)) if dist < best_dist => {
                best = Some((dist, clamped));
            }
            None => {
                best = Some((dist, clamped));
            }
            _ => {}
        }
    }

    best.map(|(_, point)| point)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_split_points_short_audio() {
        let audio = vec![0.0f32; 16000 * 10];
        let chunk = 16000 * 30;
        let step = chunk - 16000;
        let points = compute_split_points(&audio, step, chunk);
        assert_eq!(points.len(), 1);
        assert_eq!(points[0], (0, audio.len()));
    }

    #[test]
    fn test_compute_split_points_exact_chunk() {
        let audio = vec![0.0f32; 16000 * 30];
        let chunk = 16000 * 30;
        let step = chunk - 16000;
        let points = compute_split_points(&audio, step, chunk);
        assert_eq!(points.len(), 1);
        assert_eq!(points[0], (0, audio.len()));
    }

    #[test]
    fn test_compute_split_points_long_silence() {
        let audio = vec![0.0f32; 16000 * 90];
        let chunk = 16000 * 30;
        let step = chunk - 16000;
        let points = compute_split_points(&audio, step, chunk);
        assert!(points.len() >= 2, "expected multiple chunks for 90s audio");
        assert_eq!(points[0].0, 0);
        assert_eq!(points.last().map(|p| p.1), Some(audio.len()));
        for (s, e) in &points {
            assert!(e - s <= chunk, "chunk too long: {} > {}", e - s, chunk);
        }
    }

    #[test]
    fn test_compute_split_points_covers_all_audio() {
        let audio = vec![0.0f32; 16000 * 65];
        let chunk = 16000 * 30;
        let step = chunk - 16000;
        let points = compute_split_points(&audio, step, chunk);
        let mut covered = vec![false; audio.len()];
        for (s, e) in &points {
            for item in covered.iter_mut().take(*e).skip(*s) {
                *item = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "some samples not covered");
    }

    #[test]
    fn test_compute_split_points_empty_audio() {
        let points = compute_split_points(&[], 16000 * 29, 16000 * 30);
        assert!(points.is_empty());
    }

    #[test]
    fn test_build_silence_gaps_no_speech() {
        let gaps = build_silence_gaps(&[], 48000);
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].start, 0);
        assert_eq!(gaps[0].end, 48000);
    }

    #[test]
    fn test_build_silence_gaps_with_speech() {
        let speech = vec![
            vad::SpeechSegment {
                start: 1000,
                end: 5000,
            },
            vad::SpeechSegment {
                start: 8000,
                end: 12000,
            },
        ];
        let gaps = build_silence_gaps(&speech, 16000);
        assert_eq!(gaps.len(), 3);
        assert_eq!(gaps[0].start, 0);
        assert_eq!(gaps[0].end, 1000);
        assert_eq!(gaps[1].start, 5000);
        assert_eq!(gaps[1].end, 8000);
        assert_eq!(gaps[2].start, 12000);
        assert_eq!(gaps[2].end, 16000);
    }

    #[test]
    fn test_find_silence_near_basic() {
        let gaps = vec![
            SilenceGap {
                start: 100,
                end: 200,
            },
            SilenceGap {
                start: 5000,
                end: 6000,
            },
            SilenceGap {
                start: 10000,
                end: 11000,
            },
        ];
        let result = find_silence_near(5500, 2000, &gaps);
        assert_eq!(result, Some(5500));
        let result = find_silence_near(50000, 100, &gaps);
        assert!(result.is_none());
    }

    #[test]
    fn test_find_silence_near_chooses_closest() {
        let gaps = vec![
            SilenceGap {
                start: 900,
                end: 1100,
            },
            SilenceGap {
                start: 1400,
                end: 1600,
            },
        ];
        let result = find_silence_near(1200, 500, &gaps);
        assert_eq!(result, Some(1000));
    }

    #[test]
    fn test_inference_buffer_creation() {
        let buf = InferenceBuffer {
            mel_buf: Vec::with_capacity(80 * 3000),
        };
        assert_eq!(buf.mel_buf.capacity(), 80 * 3000);
        assert!(buf.mel_buf.is_empty());
    }

    #[test]
    fn test_inference_buffer_reuse() {
        let mut buf = InferenceBuffer {
            mel_buf: Vec::with_capacity(80 * 3000),
        };
        buf.mel_buf.extend(std::iter::repeat_n(1.0f32, 80 * 1500));
        assert_eq!(buf.mel_buf.len(), 80 * 1500);
        let cap_before = buf.mel_buf.capacity();
        buf.mel_buf.clear();
        buf.mel_buf.extend(std::iter::repeat_n(2.0f32, 80 * 1000));
        assert_eq!(buf.mel_buf.len(), 80 * 1000);
        assert!(buf.mel_buf.capacity() >= cap_before);
    }

    #[test]
    fn test_compute_split_points_zero_step() {
        let audio = vec![0.0f32; 16000 * 60];
        let points = compute_split_points(&audio, 0, 16000 * 30);
        assert!(points.is_empty());
    }

    #[test]
    fn test_compute_split_points_zero_chunk() {
        let audio = vec![0.0f32; 16000 * 60];
        let points = compute_split_points(&audio, 16000 * 29, 0);
        assert!(points.is_empty());
    }

    #[test]
    fn test_compute_split_points_step_equals_chunk() {
        let audio = vec![0.0f32; 16000 * 90];
        let chunk = 16000 * 30;
        let points = compute_split_points(&audio, chunk, chunk);
        assert!(points.len() >= 2);
        assert_eq!(points[0].0, 0);
        assert_eq!(
            points.last().map(|p| p.1),
            Some(audio.len()),
            "last chunk must reach audio end"
        );
    }

    #[test]
    fn test_compute_split_points_single_sample_over_chunk() {
        let chunk = 16000 * 30;
        let audio = vec![0.0f32; chunk + 1];
        let step = chunk - 16000;
        let points = compute_split_points(&audio, step, chunk);
        assert!(points.len() >= 2, "should require at least 2 chunks");
        assert_eq!(points[0].0, 0);
        assert_eq!(points.last().map(|p| p.1), Some(audio.len()));
    }

    #[test]
    fn test_mel_silence_values_near_minimum() {
        let audio = vec![0.0f32; 16000];
        let n_bins = mel::WHISPER_N_FFT / 2 + 1;
        let mel_filters = vec![1.0f32 / n_bins as f32; mel::WHISPER_N_MELS * n_bins];
        let result = mel::log_mel_spectrogram(&audio, &mel_filters);
        for (i, &v) in result.iter().enumerate() {
            assert!(v.is_finite(), "non-finite at index {i}: {v}");
        }
        let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
        assert!(mean < 0.5, "silence mel mean should be low, got {mean}");
    }

    #[test]
    fn test_batch_empty_list_returns_empty() {
        let audios: &[&[f32]] = &[];
        let results: Vec<Result<(), String>> = audios.iter().map(|_audio| Ok(())).collect();
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_maps_independently() {
        let audios: Vec<&[f32]> = vec![&[1.0, 2.0], &[], &[3.0, 4.0, 5.0]];
        let results: Vec<Result<usize, &str>> = audios
            .iter()
            .map(|audio| {
                if audio.is_empty() {
                    Err("empty")
                } else {
                    Ok(audio.len())
                }
            })
            .collect();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].as_ref().copied(), Ok(2));
        assert!(results[1].is_err());
        assert_eq!(results[2].as_ref().copied(), Ok(3));
    }

    // -----------------------------------------------------------------------
    // Integration tests with synthetic GGML model
    // -----------------------------------------------------------------------

    struct TempFileCleanup<'a>(&'a std::path::Path);
    impl Drop for TempFileCleanup<'_> {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(self.0);
        }
    }

    #[test]
    fn test_encoder_integration() {
        let model_path = test_utils::generate_synthetic_model();
        let _cleanup = TempFileCleanup(&model_path);
        let model = WhisperModel::from_file(&model_path).expect("load synthetic model");
        let info = model.info();
        assert_eq!(info.n_mels, 80);
        assert_eq!(info.d_model, 384);
        assert_eq!(info.n_audio_heads, 6);
        assert_eq!(info.n_audio_layers, 4);
        assert_eq!(info.n_text_layers, 4);
        assert_eq!(info.n_text_heads, 6);
        assert_eq!(info.n_vocab, 51865);
        let silence = vec![0.0f32; 16000];
        let result = model.transcribe(&silence, &TranscribeOptions::default());
        assert!(
            result.is_ok(),
            "transcribe on silence should not crash: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_decoder_roundtrip() {
        let model_path = test_utils::generate_synthetic_model();
        let _cleanup = TempFileCleanup(&model_path);
        let model = WhisperModel::from_file(&model_path).expect("load synthetic model");
        let audio: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();
        let opts = TranscribeOptions {
            language: Some("en"),
            ..TranscribeOptions::default()
        };
        let result = model.transcribe_segmented(&audio, &opts);
        assert!(
            result.is_ok(),
            "transcribe_segmented should not crash: {:?}",
            result.err()
        );
        let result = result.expect("already checked");
        assert!(result.text.len() <= 10000, "output text suspiciously long");
    }

    // -----------------------------------------------------------------------
    // Validation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_beam_width_zero_returns_config_error() {
        let opts = TranscribeOptions {
            beam_width: 0,
            ..TranscribeOptions::default()
        };
        let err = validate_options(&opts);
        assert!(err.is_err());
        let msg = format!("{}", err.expect_err("already checked"));
        assert!(
            msg.contains("beam_width"),
            "error should mention beam_width: {msg}"
        );
    }

    #[test]
    fn test_validate_negative_temperature_returns_config_error() {
        let opts = TranscribeOptions {
            temperature: -0.5,
            ..TranscribeOptions::default()
        };
        let err = validate_options(&opts);
        assert!(err.is_err());
        let msg = format!("{}", err.expect_err("already checked"));
        assert!(
            msg.contains("temperature"),
            "error should mention temperature: {msg}"
        );
    }

    #[test]
    fn test_validate_top_p_zero_returns_config_error() {
        let opts = TranscribeOptions {
            top_p: 0.0,
            ..TranscribeOptions::default()
        };
        let err = validate_options(&opts);
        assert!(err.is_err());
        let msg = format!("{}", err.expect_err("already checked"));
        assert!(msg.contains("top_p"), "error should mention top_p: {msg}");
    }

    #[test]
    fn test_validate_top_p_above_one_returns_config_error() {
        let opts = TranscribeOptions {
            top_p: 1.5,
            ..TranscribeOptions::default()
        };
        let err = validate_options(&opts);
        assert!(err.is_err());
        let msg = format!("{}", err.expect_err("already checked"));
        assert!(msg.contains("top_p"), "error should mention top_p: {msg}");
    }

    #[test]
    fn test_validate_valid_options_pass() {
        let opts = TranscribeOptions::default();
        assert!(validate_options(&opts).is_ok());
        let opts_custom = TranscribeOptions {
            beam_width: 5,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            language: Some("en"),
            timestamps: true,
            initial_prompt: None,
            suppress_tokens: None,
            no_repeat_ngram_size: 3,
            compression_ratio_threshold: 2.4,
            previous_tokens: None,
        };
        assert!(validate_options(&opts_custom).is_ok());
    }

    #[test]
    fn test_validate_edge_cases() {
        let opts = TranscribeOptions {
            temperature: 0.0,
            ..TranscribeOptions::default()
        };
        assert!(validate_options(&opts).is_ok());
        let opts = TranscribeOptions {
            top_p: 1.0,
            ..TranscribeOptions::default()
        };
        assert!(validate_options(&opts).is_ok());
        let opts = TranscribeOptions {
            beam_width: 1,
            ..TranscribeOptions::default()
        };
        assert!(validate_options(&opts).is_ok());
    }

    #[test]
    fn test_error_display_new_variants() {
        let err = OxiWhisperError::ConfigError("test config".into());
        assert_eq!(format!("{err}"), "Config error: test config");
        let err = OxiWhisperError::AudioFormatError("bad format".into());
        assert_eq!(format!("{err}"), "Audio format error: bad format");
    }

    #[test]
    fn test_options_with_initial_prompt() {
        let opts = TranscribeOptions {
            initial_prompt: Some("technical vocabulary"),
            ..TranscribeOptions::default()
        };
        assert_eq!(opts.initial_prompt, Some("technical vocabulary"));
        assert!(opts.suppress_tokens.is_none());
        assert!(validate_options(&opts).is_ok());
    }

    #[test]
    fn test_options_with_suppress_tokens() {
        let suppress = [1u32, 2, 3];
        let opts = TranscribeOptions {
            suppress_tokens: Some(&suppress),
            ..TranscribeOptions::default()
        };
        assert_eq!(opts.suppress_tokens, Some(&suppress[..]));
        assert!(opts.initial_prompt.is_none());
        assert!(validate_options(&opts).is_ok());
    }

    #[test]
    fn test_options_default_has_none_for_new_fields() {
        let opts = TranscribeOptions::default();
        assert!(opts.initial_prompt.is_none());
        assert!(opts.suppress_tokens.is_none());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serialize_transcribe_result() {
        let result = TranscribeResult {
            text: "hello world".to_string(),
            segments: vec![Segment {
                text: "hello world".to_string(),
                start: 0.0,
                end: 2.5,
                confidence: -0.5,
                is_hallucination: false,
            }],
            language: Some("en".to_string()),
        };
        let json = to_json(&result).expect("serialization should succeed");
        assert!(json.contains("hello world"));
        assert!(json.contains("\"start\""));
    }

    // -----------------------------------------------------------------------
    // ModelStats / encoder_output / mel_spectrogram API tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mel_spectrogram_shape() {
        let model_path = test_utils::generate_synthetic_model();
        let _cleanup = TempFileCleanup(&model_path);
        let model = WhisperModel::from_file(&model_path).expect("load synthetic model");
        let audio: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();
        let mel_tensor = model.mel_spectrogram(&audio);
        assert_eq!(mel_tensor.shape.len(), 2, "mel should be 2D");
        assert_eq!(mel_tensor.shape[0], 80, "first dim should be n_mels=80");
        let n_frames = mel_tensor.shape[1];
        assert!(n_frames > 0, "should have at least 1 frame");
        assert_eq!(
            mel_tensor.data.len(),
            80 * n_frames,
            "data length should match shape product"
        );
    }

    #[test]
    fn test_mel_spectrogram_no_nan() {
        let model_path = test_utils::generate_synthetic_model();
        let _cleanup = TempFileCleanup(&model_path);
        let model = WhisperModel::from_file(&model_path).expect("load synthetic model");
        let audio: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();
        let mel_tensor = model.mel_spectrogram(&audio);
        for (i, &v) in mel_tensor.data.iter().enumerate() {
            assert!(
                v.is_finite(),
                "mel spectrogram value at index {i} is not finite: {v}"
            );
        }
    }

    #[test]
    fn test_model_stats() {
        let model_path = test_utils::generate_synthetic_model();
        let _cleanup = TempFileCleanup(&model_path);
        let model = WhisperModel::from_file(&model_path).expect("load synthetic model");
        let stats = model.model_stats();
        assert!(stats.total_params > 0, "total_params should be > 0");
        assert!(
            stats.estimated_memory_bytes > 0,
            "estimated_memory_bytes should be > 0"
        );
        assert_eq!(
            stats.total_params,
            stats.float32_params + stats.quantized_params,
            "total should equal float32 + quantized"
        );
        assert!(
            stats.estimated_memory_bytes >= stats.float32_params * 4,
            "memory should be >= 4 * float32_params"
        );
    }

    #[test]
    fn test_encoder_output_shape() {
        let model_path = test_utils::generate_synthetic_model();
        let _cleanup = TempFileCleanup(&model_path);
        let model = WhisperModel::from_file(&model_path).expect("load synthetic model");
        let info = model.info();
        let audio: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();
        let encoded = model
            .encoder_output(&audio)
            .expect("encoder_output should succeed");
        assert_eq!(encoded.shape.len(), 2, "encoder output should be 2D");
        assert_eq!(
            encoded.shape[1], info.d_model,
            "second dim should be d_model (n_audio_state)"
        );
        assert!(encoded.shape[0] > 0, "sequence length should be > 0");
        for (i, &v) in encoded.data.iter().enumerate() {
            assert!(
                v.is_finite(),
                "encoder output value at index {i} is not finite: {v}"
            );
        }
    }
}

#[cfg(test)]
mod thread_safety_tests {
    use super::*;

    fn _assert_send<T: Send>() {}
    fn _assert_sync<T: Sync>() {}

    #[test]
    fn test_whisper_model_is_send_and_sync() {
        _assert_send::<WhisperModel>();
        _assert_sync::<WhisperModel>();
    }

    #[test]
    fn test_transcribe_options_is_send() {
        _assert_send::<TranscribeOptions<'static>>();
    }

    #[test]
    fn test_transcribe_result_is_send() {
        _assert_send::<TranscribeResult>();
        _assert_sync::<TranscribeResult>();
    }

    #[test]
    fn test_transcribe_long_with_vad_config() {
        // Verify the method signature compiles and accepts a custom VadConfig.
        // We use a tiny model for this test — the actual transcription quality
        // is not the focus here; we just confirm the API works end-to-end.
        let model_path = std::path::Path::new("ggml-tiny.bin");
        if !model_path.exists() {
            // Skip if model file is not available in CI/local
            return;
        }

        let model =
            WhisperModel::from_file(model_path).expect("failed to load model for vad config test");

        // 2 seconds of silence — should produce empty or minimal output
        let audio = vec![0.0f32; 16000 * 2];
        let opts = TranscribeOptions {
            language: Some("en"),
            ..TranscribeOptions::default()
        };

        let vad_config = vad::VadConfig {
            adaptive_threshold: true,
            noise_margin: 5.0,
            energy_threshold: 0.01,
            min_speech_ms: 250,
            min_silence_ms: 300,
            frame_size_ms: 30,
        };

        let result = model.transcribe_long_with_vad(&audio, &opts, &vad_config);
        assert!(
            result.is_ok(),
            "transcribe_long_with_vad should not fail: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_compute_split_points_with_custom_vad() {
        // Verify compute_split_points_with_vad works with adaptive threshold
        let audio = vec![0.0f32; 16000 * 90]; // 90 seconds of silence
        let chunk = 16000 * 30;
        let step = chunk - 16000;

        let vad_config = vad::VadConfig {
            adaptive_threshold: true,
            noise_margin: 3.0,
            energy_threshold: 0.01,
            min_speech_ms: 100,
            min_silence_ms: 100,
            frame_size_ms: 30,
        };

        let points = compute_split_points_with_vad(&audio, step, chunk, &vad_config);
        assert!(
            points.len() >= 2,
            "expected multiple chunks for 90s audio, got {}",
            points.len()
        );
        assert_eq!(points[0].0, 0);
        assert_eq!(
            points.last().map(|p| p.1),
            Some(audio.len()),
            "last chunk should end at audio length"
        );
    }
}
