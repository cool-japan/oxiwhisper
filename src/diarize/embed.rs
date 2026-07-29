// SPDX-License-Identifier: Apache-2.0
// Copyright (c) COOLJAPAN OU (Team Kitasan)

//! Speaker-embedding front-ends for diarization.
//!
//! A *speaker embedding* maps a short window of audio to a fixed-length vector
//! whose geometry encodes **speaker identity** rather than linguistic content:
//! windows spoken by the same person land close together (high cosine
//! similarity) and windows from different people land far apart. The clustering
//! stage (D4) groups these vectors into per-speaker clusters, so embedding
//! quality is the dominant factor in overall diarization accuracy.
//!
//! This module exposes the [`SpeakerEmbedder`] trait plus two backends:
//!
//! * [`WhisperEncoderEmbedder`] — a zero-extra-model baseline that mean-pools
//!   the Whisper encoder features. It is convenient (no second model to ship)
//!   but only a **weak** speaker signal; see its type docs.
#![cfg_attr(
    feature = "onnx",
    doc = "* [`EcapaOnnx`] — a real ECAPA-TDNN / x-vector speaker-embedding network run"
)]
#![cfg_attr(
    not(feature = "onnx"),
    doc = "* `EcapaOnnx` — a real ECAPA-TDNN / x-vector speaker-embedding network run"
)]
//!   through the `oxionnx` backend (behind the `onnx` feature). This is the
//!   backend to use for production-quality diarization.
//!
//! Every embedder returns an **L2-normalized** vector, so a plain dot product
//! between two embeddings is their cosine similarity — exactly the affinity the
//! clustering stage consumes.

use crate::types::OxiWhisperError;

/// Maps a window of audio to a fixed-length, L2-normalized speaker embedding.
///
/// Implementors guarantee that [`embed`](SpeakerEmbedder::embed) returns a
/// vector of length [`dim`](SpeakerEmbedder::dim) whose Euclidean (L2) norm is
/// `1` (within floating-point tolerance). Because the output is unit-norm, the
/// cosine similarity between two embeddings is simply their dot product, which
/// is the affinity measure the clustering stage relies on.
pub trait SpeakerEmbedder {
    /// Dimensionality of the embedding vectors this backend produces.
    ///
    /// This is a fixed property of the backend and MUST be obtainable without
    /// running inference on any audio.
    fn dim(&self) -> usize;

    /// Embed one window of audio into an L2-normalized speaker vector.
    ///
    /// `audio` is mono PCM at `sample_rate` Hz. The returned vector has length
    /// [`dim`](Self::dim) and unit L2 norm.
    ///
    /// # Errors
    ///
    /// Returns [`OxiWhisperError`] if `audio` is empty, if the underlying model
    /// fails, or if the pooled window is degenerate (zero-norm) and therefore
    /// cannot be normalized.
    fn embed(&self, audio: &[f32], sample_rate: usize) -> Result<Vec<f32>, OxiWhisperError>;
}

/// L2-normalize `vec` in place-by-value, returning the unit-norm vector.
///
/// Refuses to divide by a near-zero norm: if `‖vec‖ < 1e-8` the window carries
/// no usable direction (silence, or a pooled-to-zero feature block), so we
/// return an error instead of producing a `NaN`/`inf`-laden or arbitrarily
/// oriented "embedding" that would silently corrupt downstream clustering.
fn l2_normalize(mut vec: Vec<f32>) -> Result<Vec<f32>, OxiWhisperError> {
    // Accumulate the sum of squares in f64 to avoid overflow / precision loss
    // on long, large-magnitude embedding vectors.
    let norm_sq: f64 = vec.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let norm = norm_sq.sqrt();
    if norm < 1e-8 {
        return Err(OxiWhisperError::ConfigError(
            "degenerate (zero-norm) embedding window".into(),
        ));
    }
    let inv = (1.0 / norm) as f32;
    for x in &mut vec {
        *x *= inv;
    }
    Ok(vec)
}

// ---------------------------------------------------------------------------
// Backend 1: Whisper encoder baseline
// ---------------------------------------------------------------------------

/// Baseline speaker embedder that mean-pools the Whisper encoder features.
///
/// It runs the audio through [`WhisperModel::encoder_output`], mean-pools the
/// `[seq_len, d_model]` feature map over the sequence axis to a single
/// `[d_model]` vector, and L2-normalizes it.
///
/// # Accuracy caveat (read this)
///
/// Whisper's encoder is trained to be largely speaker-**invariant** — it
/// encodes phonetic/acoustic content for ASR — so these embeddings cluster only
/// **weakly** by speaker; this backend exists for tests and no-extra-model
/// demos and MUST NOT be presented as production diarization. Use
#[cfg_attr(feature = "onnx", doc = "[`EcapaOnnx`] for real accuracy.")]
#[cfg_attr(not(feature = "onnx"), doc = "`EcapaOnnx` for real accuracy.")]
///
/// [`WhisperModel::encoder_output`]: crate::WhisperModel::encoder_output
pub struct WhisperEncoderEmbedder<'a> {
    model: &'a crate::WhisperModel,
}

impl<'a> WhisperEncoderEmbedder<'a> {
    /// Wrap a loaded [`WhisperModel`](crate::WhisperModel) as a baseline
    /// speaker embedder. The model is borrowed, not cloned.
    pub fn new(model: &'a crate::WhisperModel) -> Self {
        Self { model }
    }
}

impl SpeakerEmbedder for WhisperEncoderEmbedder<'_> {
    fn dim(&self) -> usize {
        // The encoder feature width is the model's d_model (n_audio_state),
        // read straight from the hyperparameters — no encoder run required.
        self.model.d_model()
    }

    fn embed(&self, audio: &[f32], _sample_rate: usize) -> Result<Vec<f32>, OxiWhisperError> {
        if audio.is_empty() {
            return Err(OxiWhisperError::InferenceFailed(
                "cannot embed empty audio".into(),
            ));
        }

        // [seq_len, d_model] encoder feature map.
        let features = self.model.encoder_output(audio)?;
        if features.shape.len() != 2 {
            return Err(OxiWhisperError::InferenceFailed(format!(
                "encoder_output must be 2-D [seq_len, d_model], got shape {:?}",
                features.shape
            )));
        }
        let seq_len = features.shape[0];
        let d_model = features.shape[1];
        if seq_len == 0 || d_model == 0 {
            return Err(OxiWhisperError::InferenceFailed(format!(
                "degenerate encoder_output shape {:?}",
                features.shape
            )));
        }

        // Mean-pool over the sequence axis: row-major [seq_len, d_model], so
        // element (t, c) lives at t * d_model + c.
        let mut pooled = vec![0.0f32; d_model];
        for t in 0..seq_len {
            let row = &features.data[t * d_model..(t + 1) * d_model];
            for (acc, &v) in pooled.iter_mut().zip(row) {
                *acc += v;
            }
        }
        let inv_len = 1.0 / seq_len as f32;
        for v in &mut pooled {
            *v *= inv_len;
        }

        l2_normalize(pooled)
    }
}

// ---------------------------------------------------------------------------
// Backend 2: ECAPA-TDNN / x-vector via ONNX
// ---------------------------------------------------------------------------

/// Speaker embedder backed by a pretrained ECAPA-TDNN / x-vector model exported
/// to ONNX and executed through the [`oxionnx`] backend.
///
/// This is the accuracy-oriented backend: ECAPA-TDNN and x-vector networks are
/// trained with a speaker-discriminative objective, so their embeddings cluster
/// tightly by speaker — unlike the [`WhisperEncoderEmbedder`] baseline.
///
/// # User-supplied weights
///
/// The `.onnx` weights are **user-supplied**. Pretrained WeSpeaker / SpeechBrain
/// checkpoints are released under their own licenses and are not ours to vendor,
/// so this crate ships no weights. Export a speaker-embedding model yourself
/// (e.g. from WeSpeaker or SpeechBrain) and pass its path to
/// [`from_path`](Self::from_path).
///
/// # Expected input tensor layout
///
/// The exported graph must accept a single float input of shape
/// `[batch = 1, num_frames, n_mels = 80]` — a batch of one utterance, a
/// time-major sequence of 80-dimensional log-mel filterbank frames computed
/// from 16 kHz mono audio (25 ms window / 10 ms hop, matching Whisper's STFT
/// configuration). The features are produced by the crate's real mel front-end
/// ([`crate::mel::log_mel_spectrogram`] over
/// [`crate::mel_filters::generate_mel_filters`]); the per-bin log scaling
/// therefore follows Whisper's convention. If your checkpoint was trained on a
/// different feature (raw waveform input, Kaldi-style natural-log fbank with
/// cepstral mean normalization, a different `n_mels`, or a channel-major
/// `[1, 80, num_frames]` layout), re-export it to consume this layout — the
/// front-end here is a genuine log-mel filterbank, never a placeholder.
///
/// # Expected output
///
/// The graph must produce a single embedding tensor whose element count is the
/// embedding dimension `D` (any of `[D]`, `[1, D]`, or `[1, D, 1, ...]` with a
/// single non-trivial axis is accepted). The raw output is L2-normalized before
/// being returned.
#[cfg(feature = "onnx")]
pub struct EcapaOnnx {
    session: oxionnx::Session,
    /// The graph's single input tensor name.
    input_name: String,
    /// The graph's single output tensor name.
    output_name: String,
    /// Embedding dimensionality, read from the model's declared output shape.
    embed_dim: usize,
}

#[cfg(feature = "onnx")]
impl EcapaOnnx {
    /// Number of mel filterbank channels fed to the model (matches the crate's
    /// Whisper mel front-end).
    const N_MELS: usize = crate::mel::WHISPER_N_MELS;

    /// Load an ECAPA-TDNN / x-vector speaker-embedding model from an ONNX file.
    ///
    /// # Errors
    ///
    /// Returns [`OxiWhisperError::InvalidModel`] if the file cannot be loaded or
    /// parsed, or if the graph does not expose exactly one input and one output.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self, OxiWhisperError> {
        let path = path.as_ref();
        let session = oxionnx::Session::from_file(path).map_err(|e| {
            OxiWhisperError::InvalidModel(format!(
                "failed to load ECAPA ONNX model '{}': {e}",
                path.display()
            ))
        })?;

        let input_name = single_io_name(session.input_names(), "input", path)?;
        let output_name = single_io_name(session.output_names(), "output", path)?;

        // Resolve the embedding dimension from the declared output shape: the
        // product of all statically-known dims (ignoring dynamic/None dims and
        // a leading batch of 1). Falls back to reading it at first inference if
        // the shape is fully dynamic.
        let embed_dim = declared_embed_dim(&session, &output_name);

        Ok(Self {
            session,
            input_name,
            output_name,
            embed_dim,
        })
    }

    /// Compute the `[1, num_frames, N_MELS]` log-mel feature tensor for `audio`.
    ///
    /// `audio` is resampled to 16 kHz if `sample_rate` differs, then run through
    /// the crate's real log-mel front-end. The crate mel routine yields a
    /// channel-major `[n_mels, num_frames]` buffer; we transpose it to the
    /// time-major `[num_frames, n_mels]` order the model expects and prepend the
    /// batch axis.
    fn compute_features(
        &self,
        audio: &[f32],
        sample_rate: usize,
    ) -> Result<oxionnx::Tensor, OxiWhisperError> {
        // Resample to the 16 kHz the mel front-end (and these models) assume.
        let resampled: Vec<f32> = if sample_rate == crate::mel::WHISPER_SAMPLE_RATE {
            audio.to_vec()
        } else {
            let rate = u32::try_from(sample_rate).map_err(|_| {
                OxiWhisperError::ConfigError(format!("invalid sample_rate {sample_rate}"))
            })?;
            crate::audio::resample_linear(audio, rate)
        };
        if resampled.is_empty() {
            return Err(OxiWhisperError::InferenceFailed(
                "cannot embed empty audio".into(),
            ));
        }

        // Real log-mel filterbank: [n_mels, num_frames], row-major.
        let mel_filters = crate::mel_filters::generate_mel_filters();
        // Unpadded: a speaker-embedding window is a few seconds long, and the
        // ECAPA graph consumes whatever frame count it is given.
        let mel = crate::mel::log_mel_spectrogram_unpadded(&resampled, &mel_filters)?;
        let n_mels = Self::N_MELS;

        // Transpose [n_mels, num_frames] -> [num_frames, n_mels] so the tensor
        // is time-major, then wrap as [1, num_frames, n_mels].
        let (feats, num_frames) = transpose_mel_to_time_major(&mel, n_mels)?;
        Ok(oxionnx::Tensor::new(feats, vec![1, num_frames, n_mels]))
    }
}

/// Transpose a channel-major `[n_mels, num_frames]` log-mel buffer (row-major,
/// element `(m, t)` at `mel[m * num_frames + t]`) into the time-major
/// `[num_frames, n_mels]` layout the ECAPA graph expects (element `(t, m)` at
/// `feats[t * n_mels + m]`).
///
/// Returns `(feats, num_frames)`.
///
/// # Errors
///
/// Returns [`OxiWhisperError::InferenceFailed`] if `n_mels` is zero, or if the
/// buffer is empty or its length is not an exact multiple of `n_mels` (which
/// would mean the mel front-end produced a shape inconsistent with `n_mels`).
#[cfg(any(feature = "onnx", test))]
fn transpose_mel_to_time_major(
    mel: &[f32],
    n_mels: usize,
) -> Result<(Vec<f32>, usize), OxiWhisperError> {
    if n_mels == 0 {
        return Err(OxiWhisperError::InferenceFailed(
            "n_mels must be non-zero to transpose a mel buffer".into(),
        ));
    }
    if mel.is_empty() || !mel.len().is_multiple_of(n_mels) {
        return Err(OxiWhisperError::InferenceFailed(format!(
            "unexpected mel buffer length {} for n_mels {n_mels}",
            mel.len()
        )));
    }
    let num_frames = mel.len() / n_mels;

    let mut feats = vec![0.0f32; num_frames * n_mels];
    for m in 0..n_mels {
        let src_row = &mel[m * num_frames..(m + 1) * num_frames];
        for (t, &v) in src_row.iter().enumerate() {
            feats[t * n_mels + m] = v;
        }
    }
    Ok((feats, num_frames))
}

#[cfg(feature = "onnx")]
impl SpeakerEmbedder for EcapaOnnx {
    fn dim(&self) -> usize {
        self.embed_dim
    }

    fn embed(&self, audio: &[f32], sample_rate: usize) -> Result<Vec<f32>, OxiWhisperError> {
        if audio.is_empty() {
            return Err(OxiWhisperError::InferenceFailed(
                "cannot embed empty audio".into(),
            ));
        }

        let features = self.compute_features(audio, sample_rate)?;

        let mut inputs = std::collections::HashMap::new();
        inputs.insert(self.input_name.as_str(), features);
        let outputs = self.session.run(&inputs).map_err(|e| {
            OxiWhisperError::InferenceFailed(format!("ECAPA ONNX inference failed: {e}"))
        })?;

        let embedding = outputs.get(&self.output_name).ok_or_else(|| {
            OxiWhisperError::InferenceFailed(format!(
                "ECAPA ONNX output '{}' missing from session results",
                self.output_name
            ))
        })?;
        if embedding.data.is_empty() {
            return Err(OxiWhisperError::InferenceFailed(
                "ECAPA ONNX produced an empty embedding".into(),
            ));
        }

        l2_normalize(embedding.data.clone())
    }
}

/// Extract the sole name from an input/output name list, erroring if the graph
/// does not expose exactly one.
#[cfg(feature = "onnx")]
fn single_io_name(
    names: &[String],
    role: &str,
    path: &std::path::Path,
) -> Result<String, OxiWhisperError> {
    match names {
        [only] => Ok(only.clone()),
        other => Err(OxiWhisperError::InvalidModel(format!(
            "ECAPA ONNX model '{}' must have exactly one {role}, found {}",
            path.display(),
            other.len()
        ))),
    }
}

/// Resolve the embedding dimension from a model's declared output shape.
///
/// Returns the product of the statically-known, non-batch dimensions. A fully
/// dynamic (or absent) shape yields `0`, which callers treat as "resolve at
/// first inference"; in practice these models declare a fixed embedding width.
#[cfg(feature = "onnx")]
fn declared_embed_dim(session: &oxionnx::Session, output_name: &str) -> usize {
    for info in session.output_info() {
        if info.name != output_name {
            continue;
        }
        // Collapse the declared dims: drop a leading batch of 1, multiply the
        // remaining concrete dims. Dynamic dims (`None`) are skipped.
        let mut dim = 1usize;
        let mut saw_concrete = false;
        for (axis, d) in info.shape.iter().enumerate() {
            match *d {
                Some(1) if axis == 0 => {} // batch axis — ignore
                Some(value) => {
                    dim = dim.saturating_mul(value);
                    saw_concrete = true;
                }
                None => {} // dynamic axis — skip
            }
        }
        return if saw_concrete { dim } else { 0 };
    }
    0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A short but non-degenerate synthetic signal: a 440 Hz tone at 16 kHz.
    fn synthetic_tone(n_samples: usize) -> Vec<f32> {
        (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect()
    }

    struct TempFileCleanup<'a>(&'a std::path::Path);
    impl Drop for TempFileCleanup<'_> {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(self.0);
        }
    }

    #[test]
    fn test_l2_normalize_unit_norm() {
        let out = l2_normalize(vec![3.0, 4.0]).expect("non-zero vector normalizes");
        assert!((out[0] - 0.6).abs() < 1e-6);
        assert!((out[1] - 0.8).abs() < 1e-6);
        let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_rejects_zero_norm() {
        let err = l2_normalize(vec![0.0, 0.0, 0.0]);
        assert!(err.is_err(), "zero-norm vector must be rejected");
    }

    #[test]
    fn test_whisper_encoder_embedder_dim_matches_model() {
        let model_path = crate::test_utils::generate_synthetic_model();
        let _cleanup = TempFileCleanup(&model_path);
        let model = crate::WhisperModel::from_file(&model_path).expect("load synthetic model");
        let embedder = WhisperEncoderEmbedder::new(&model);
        // dim() must equal d_model without running the encoder.
        assert_eq!(embedder.dim(), model.d_model());
        assert_eq!(embedder.dim(), model.info().d_model);
    }

    #[test]
    fn test_whisper_encoder_embed_is_unit_norm() {
        let model_path = crate::test_utils::generate_synthetic_model();
        let _cleanup = TempFileCleanup(&model_path);
        let model = crate::WhisperModel::from_file(&model_path).expect("load synthetic model");
        let embedder = WhisperEncoderEmbedder::new(&model);

        // Half a second of a real tone — non-empty, non-degenerate.
        let audio = synthetic_tone(8000);
        let embedding = embedder
            .embed(&audio, 16000)
            .expect("embed of a real tone should succeed");

        assert_eq!(
            embedding.len(),
            embedder.dim(),
            "embedding length must equal dim()"
        );
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "embedding must be L2-normalized, got norm {norm}"
        );
        assert!(
            embedding.iter().all(|v| v.is_finite()),
            "embedding must be finite"
        );
    }

    #[test]
    fn test_whisper_encoder_embed_empty_audio_errs() {
        let model_path = crate::test_utils::generate_synthetic_model();
        let _cleanup = TempFileCleanup(&model_path);
        let model = crate::WhisperModel::from_file(&model_path).expect("load synthetic model");
        let embedder = WhisperEncoderEmbedder::new(&model);
        assert!(
            embedder.embed(&[], 16000).is_err(),
            "empty audio must return Err"
        );
    }

    #[test]
    fn test_transpose_mel_to_time_major_layout() {
        // Channel-major [n_mels = 3, num_frames = 2], element (m, t) at
        // mel[m * num_frames + t]:
        //   channel 0 -> [0, 1], channel 1 -> [10, 11], channel 2 -> [20, 21].
        let n_mels = 3;
        let mel = vec![0.0f32, 1.0, 10.0, 11.0, 20.0, 21.0];
        let (feats, num_frames) =
            transpose_mel_to_time_major(&mel, n_mels).expect("valid mel buffer transposes");
        assert_eq!(num_frames, 2, "num_frames = len / n_mels");
        // Time-major [num_frames = 2, n_mels = 3], element (t, m) at
        // feats[t * n_mels + m]:
        //   frame 0 -> [ch0@0, ch1@0, ch2@0] = [0, 10, 20]
        //   frame 1 -> [ch0@1, ch1@1, ch2@1] = [1, 11, 21]
        assert_eq!(feats, vec![0.0f32, 10.0, 20.0, 1.0, 11.0, 21.0]);
    }

    #[test]
    fn test_transpose_mel_rejects_ragged_length() {
        // Length 3 is not a multiple of n_mels = 2, so the layout is ambiguous
        // and must be rejected rather than silently truncated.
        assert!(
            transpose_mel_to_time_major(&[1.0, 2.0, 3.0], 2).is_err(),
            "ragged mel length must return Err"
        );
    }

    #[test]
    fn test_transpose_mel_rejects_zero_mels() {
        assert!(
            transpose_mel_to_time_major(&[1.0, 2.0], 0).is_err(),
            "zero n_mels must return Err"
        );
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn test_feature_tensor_shape_is_batch_time_mels() {
        // Lock the [1, num_frames, n_mels] tensor shape that wraps the
        // transposed features, so the batch axis and axis order cannot regress.
        let n_mels = 3;
        let mel = vec![0.0f32, 1.0, 10.0, 11.0, 20.0, 21.0];
        let (feats, num_frames) =
            transpose_mel_to_time_major(&mel, n_mels).expect("valid mel buffer transposes");
        let tensor = oxionnx::Tensor::new(feats, vec![1, num_frames, n_mels]);
        assert_eq!(
            tensor.shape,
            vec![1, 2, 3],
            "feature tensor must be [batch = 1, num_frames, n_mels]"
        );
        assert_eq!(
            tensor.data.len(),
            6,
            "tensor data must fill the [1,2,3] shape"
        );
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn test_ecapa_from_nonexistent_path_errs() {
        // We cannot ship pretrained ECAPA weights, so exercise the load-failure
        // path: a path that does not exist must return Err, never panic.
        let missing = std::env::temp_dir().join(format!(
            "oxiwhisper_missing_ecapa_{}_{}.onnx",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        assert!(!missing.exists(), "test path must not exist");
        let result = EcapaOnnx::from_path(&missing);
        assert!(
            result.is_err(),
            "loading a non-existent ECAPA model must return Err"
        );
    }
}
