// SPDX-License-Identifier: Apache-2.0
// Copyright (c) COOLJAPAN OU (Team Kitasan)

//! Speaker diarization ("who spoke when").
//!
//! This module defines the diarization **public types** and drives the
//! end-to-end pipeline. The full pipeline runs the following stages:
//!
//! 1. **VAD** — voice activity detection to isolate speech regions.
//! 2. **Uniform sub-segmentation** — slice speech into overlapping windows
//!    ([`window_speech`](crate::diarize::segment::window_speech)).
//! 3. **Speaker embedding** — map each window to a speaker embedding vector
//!    ([`SpeakerEmbedder`](crate::diarize::embed::SpeakerEmbedder)).
//! 4. **Clustering** — group embeddings into speaker clusters
//!    ([`cluster_speakers`](crate::diarize::cluster::cluster_speakers)).
//! 5. **Resegmentation** — turn per-window labels into contiguous, sorted,
//!    non-overlapping speaker segments
//!    ([`resegment`](crate::diarize::reseg::resegment)).
//! 6. **ASR fusion** — attribute transcribed words to speaker turns
//!    ([`attribute_words`](crate::diarize::attribute::attribute_words)).
//!
//! Stages D2–D4 (segmentation, embedding, clustering) and D5 (resegmentation
//! plus the end-to-end [`WhisperModel::diarize`](crate::WhisperModel::diarize)
//! entry point) are implemented. Speaker-accuracy / DER validation against the
//! baseline and ECAPA embedders is deferred to Batch F (F2).

use crate::types::OxiWhisperError;
use crate::vad::VadConfig;

pub mod attribute;
pub mod cluster;
pub mod embed;
pub mod format;
pub mod metrics;
pub mod reseg;
pub mod segment;

/// Opaque speaker cluster label.
///
/// This is only meaningful within a single [`DiarizeResult`]: it identifies a
/// cluster, not a real-world identity. Labels are **not** stable across runs —
/// the same speaker may receive a different `SpeakerId` on a different input or
/// even a re-run of the same input.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SpeakerId(pub u32);

/// A contiguous span of audio attributed to a single speaker.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerSegment {
    /// Cluster label of the speaker for this span.
    pub speaker: SpeakerId,
    /// Start time of the span, in seconds.
    pub start: f32,
    /// End time of the span, in seconds.
    pub end: f32,
}

impl SpeakerSegment {
    /// Duration of the span in seconds, clamped to a non-negative value.
    pub fn duration(&self) -> f32 {
        (self.end - self.start).max(0.0)
    }
}

/// Result of running diarization over an audio input.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct DiarizeResult {
    /// Speaker-attributed spans, in chronological order.
    pub segments: Vec<SpeakerSegment>,
    /// Number of distinct speakers discovered.
    pub num_speakers: usize,
}

/// Clustering strategy used to group speaker embeddings.
#[derive(Debug, Clone, PartialEq)]
pub enum ClusteringMethod {
    /// Agglomerative hierarchical clustering: repeatedly merge the closest
    /// clusters (smallest average-linkage cosine **distance** `1 - cos`) until
    /// the nearest remaining pair is farther apart than `threshold`.
    Ahc {
        /// Cosine-**distance** (`1 - cos`) stopping threshold: merging continues
        /// while the closest cluster pair is within this average-linkage
        /// distance and stops once the nearest pair is farther than `threshold`.
        ///
        /// Because it is a distance (not an affinity), it ranges over `[0, 2]`:
        /// `0` = identical direction, `1` = orthogonal, `2` = antipodal. The
        /// default `0.5` therefore merges clusters whose cosine similarity is at
        /// least `0.5` (`1 - cos <= 0.5 <=> cos >= 0.5`). Larger values merge
        /// more aggressively (fewer speakers); smaller values split more.
        threshold: f32,
    },
    /// Spectral clustering on the normalized graph Laplacian, using the
    /// eigengap heuristic to estimate the number of speakers.
    Spectral,
}

impl Default for ClusteringMethod {
    fn default() -> Self {
        ClusteringMethod::Ahc { threshold: 0.5 }
    }
}

/// Options controlling the diarization pipeline.
#[derive(Debug, Clone)]
pub struct DiarizeOptions {
    /// Exact number of speakers, if known. `None` estimates it automatically
    /// within `[min_speakers, max_speakers]`.
    pub num_speakers: Option<usize>,
    /// Lower bound on the estimated speaker count (inclusive).
    pub min_speakers: usize,
    /// Upper bound on the estimated speaker count (inclusive).
    pub max_speakers: usize,
    /// Sub-segmentation window length, in seconds.
    pub window_s: f32,
    /// Sub-segmentation hop (stride between windows), in seconds.
    pub hop_s: f32,
    /// Minimum speaker-segment duration, in seconds; shorter spans are merged
    /// or discarded during resegmentation.
    pub min_duration_s: f32,
    /// Clustering strategy used to group speaker embeddings.
    pub clustering: ClusteringMethod,
    /// Voice-activity-detection configuration for the speech-isolation stage.
    pub vad: VadConfig,
}

impl Default for DiarizeOptions {
    fn default() -> Self {
        Self {
            num_speakers: None,
            min_speakers: 1,
            max_speakers: 10,
            window_s: 1.5,
            hop_s: 0.75,
            min_duration_s: 0.5,
            clustering: ClusteringMethod::default(),
            vad: VadConfig::default(),
        }
    }
}

impl DiarizeOptions {
    /// Validate diarization options before running the pipeline.
    pub fn validate(&self) -> Result<(), OxiWhisperError> {
        if self.window_s <= 0.0 {
            return Err(OxiWhisperError::ConfigError(
                "window_s must be > 0.0".into(),
            ));
        }
        if self.hop_s <= 0.0 {
            return Err(OxiWhisperError::ConfigError("hop_s must be > 0.0".into()));
        }
        if self.hop_s > self.window_s {
            return Err(OxiWhisperError::ConfigError(
                "hop_s must be <= window_s".into(),
            ));
        }
        if self.min_duration_s < 0.0 {
            return Err(OxiWhisperError::ConfigError(
                "min_duration_s must be >= 0.0".into(),
            ));
        }
        if self.max_speakers < 1 {
            return Err(OxiWhisperError::ConfigError(
                "max_speakers must be >= 1".into(),
            ));
        }
        if self.min_speakers < 1 {
            return Err(OxiWhisperError::ConfigError(
                "min_speakers must be >= 1".into(),
            ));
        }
        if self.min_speakers > self.max_speakers {
            return Err(OxiWhisperError::ConfigError(
                "min_speakers must be <= max_speakers".into(),
            ));
        }
        if let Some(n) = self.num_speakers {
            if n < 1 {
                return Err(OxiWhisperError::ConfigError(
                    "num_speakers must be >= 1".into(),
                ));
            }
            if n < self.min_speakers || n > self.max_speakers {
                return Err(OxiWhisperError::ConfigError(
                    "num_speakers must be within [min_speakers, max_speakers]".into(),
                ));
            }
        }
        if let ClusteringMethod::Ahc { threshold } = self.clustering
            && !(threshold.is_finite() && threshold > 0.0)
        {
            return Err(OxiWhisperError::ConfigError(
                "ahc threshold must be finite and > 0.0".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_options_validate_ok() {
        assert!(DiarizeOptions::default().validate().is_ok());
    }

    #[test]
    fn test_reject_window_s_not_positive() {
        // window_s <= 0.0 is the first invariant checked and fires here.
        let opts = DiarizeOptions {
            window_s: 0.0,
            hop_s: 0.0,
            ..DiarizeOptions::default()
        };
        assert!(opts.validate().is_err());
    }

    #[test]
    fn test_reject_hop_s_not_positive() {
        let opts = DiarizeOptions {
            hop_s: 0.0,
            ..DiarizeOptions::default()
        };
        assert!(opts.validate().is_err());
    }

    #[test]
    fn test_reject_hop_s_exceeds_window_s() {
        // Both positive so only the hop_s <= window_s check can fire.
        let opts = DiarizeOptions {
            window_s: 1.0,
            hop_s: 2.0,
            ..DiarizeOptions::default()
        };
        assert!(opts.validate().is_err());
    }

    #[test]
    fn test_reject_negative_min_duration_s() {
        let opts = DiarizeOptions {
            min_duration_s: -0.1,
            ..DiarizeOptions::default()
        };
        assert!(opts.validate().is_err());
    }

    #[test]
    fn test_reject_max_speakers_zero() {
        // max_speakers == 0 also forces min_speakers (1) > max_speakers, so
        // this asserts a genuinely-invalid config rather than isolating a
        // single check (structurally un-isolable given check ordering).
        let opts = DiarizeOptions {
            max_speakers: 0,
            ..DiarizeOptions::default()
        };
        assert!(opts.validate().is_err());
    }

    #[test]
    fn test_reject_min_speakers_zero() {
        // max_speakers stays 10, so min_speakers <= max_speakers holds and the
        // min_speakers >= 1 check is the sole failure. Still noted here: this
        // asserts a genuinely-invalid config rather than isolating a single
        // check (structurally un-isolable given check ordering).
        let opts = DiarizeOptions {
            min_speakers: 0,
            ..DiarizeOptions::default()
        };
        assert!(opts.validate().is_err());
    }

    #[test]
    fn test_reject_min_speakers_gt_max_speakers() {
        // This asserts a genuinely-invalid config rather than isolating a
        // single check (structurally un-isolable given check ordering).
        let opts = DiarizeOptions {
            min_speakers: 5,
            max_speakers: 3,
            num_speakers: None,
            ..DiarizeOptions::default()
        };
        assert!(opts.validate().is_err());
    }

    #[test]
    fn test_reject_num_speakers_zero() {
        // min_speakers is 1 by default, so n = 0 trips the `n < 1` check.
        // This asserts a genuinely-invalid config rather than isolating a
        // single check (structurally un-isolable given check ordering).
        let opts = DiarizeOptions {
            num_speakers: Some(0),
            ..DiarizeOptions::default()
        };
        assert!(opts.validate().is_err());
    }

    #[test]
    fn test_reject_num_speakers_out_of_range() {
        // This asserts a genuinely-invalid config rather than isolating a
        // single check (structurally un-isolable given check ordering).
        let opts = DiarizeOptions {
            min_speakers: 1,
            max_speakers: 4,
            num_speakers: Some(5),
            ..DiarizeOptions::default()
        };
        assert!(opts.validate().is_err());
    }

    #[test]
    fn test_reject_ahc_threshold_not_finite_positive() {
        let opts = DiarizeOptions {
            clustering: ClusteringMethod::Ahc { threshold: 0.0 },
            ..DiarizeOptions::default()
        };
        assert!(opts.validate().is_err());
    }

    #[test]
    fn test_reject_ahc_threshold_nan() {
        // Exercises the `threshold.is_finite()` branch specifically, which
        // threshold=0.0 (caught by the `> 0.0` half) does not reach.
        let opts = DiarizeOptions {
            clustering: ClusteringMethod::Ahc {
                threshold: f32::NAN,
            },
            ..DiarizeOptions::default()
        };
        assert!(opts.validate().is_err());
    }

    #[test]
    fn test_reject_ahc_threshold_infinite() {
        // Exercises the `threshold.is_finite()` branch specifically, which
        // threshold=0.0 (caught by the `> 0.0` half) does not reach.
        let opts = DiarizeOptions {
            clustering: ClusteringMethod::Ahc {
                threshold: f32::INFINITY,
            },
            ..DiarizeOptions::default()
        };
        assert!(opts.validate().is_err());
    }

    #[test]
    fn test_speaker_segment_duration() {
        let seg = SpeakerSegment {
            speaker: SpeakerId(0),
            start: 1.0,
            end: 3.5,
        };
        assert_eq!(seg.duration(), 2.5);
    }

    #[test]
    fn test_speaker_segment_duration_clamps_negative() {
        let seg = SpeakerSegment {
            speaker: SpeakerId(1),
            start: 5.0,
            end: 2.0,
        };
        assert_eq!(seg.duration(), 0.0);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_diarize_result_serde_round_trip() {
        let result = DiarizeResult {
            segments: vec![
                SpeakerSegment {
                    speaker: SpeakerId(0),
                    start: 0.0,
                    end: 1.5,
                },
                SpeakerSegment {
                    speaker: SpeakerId(1),
                    start: 1.5,
                    end: 3.0,
                },
            ],
            num_speakers: 2,
        };
        let json = serde_json::to_string(&result).expect("serialize DiarizeResult");
        let decoded: DiarizeResult =
            serde_json::from_str(&json).expect("deserialize DiarizeResult");
        assert_eq!(result, decoded);
    }
}
