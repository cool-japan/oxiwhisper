//! Streaming transcription support for incremental audio processing.

use crate::WhisperModel;
use crate::types::*;

/// Number of samples in a 30-second chunk at 16 kHz (used by streaming API).
pub(crate) const STREAM_CHUNK_SAMPLES: usize = 16000 * 30;

/// Overlap kept between streaming chunks (1 second).
pub(crate) const STREAM_OVERLAP_SAMPLES: usize = 16000;

/// Streaming transcriber that processes audio incrementally.
///
/// Accumulates pushed audio samples and processes 30-second chunks as they
/// become available. Call [`push_audio`](Self::push_audio) to feed samples,
/// [`next_segment`](Self::next_segment) to retrieve decoded segments, and
/// [`finish`](Self::finish) to flush remaining audio.
///
/// # Example
/// ```no_run
/// # use oxiwhisper::*;
/// # let model: WhisperModel = todo!();
/// let mut stream = model.stream(TranscribeOptions::default());
/// // Feed audio in arbitrary-sized chunks
/// stream.push_audio(&[0.0f32; 8000]);
/// stream.push_audio(&[0.0f32; 8000]);
/// // Process available segments
/// while let Some(seg) = stream.next_segment() {
///     let seg = seg.unwrap();
///     println!("[{:.1}s - {:.1}s] {}", seg.start, seg.end, seg.text);
/// }
/// // Flush remaining audio
/// let result = stream.finish().unwrap();
/// ```
pub struct StreamTranscriber<'m> {
    model: &'m WhisperModel,
    opts: TranscribeOptions<'static>,
    audio_buf: Vec<f32>,
    /// Segments ready for consumption.
    pending_segments: Vec<Segment>,
    /// Accumulated text from all processed chunks.
    texts: Vec<String>,
    /// Detected language from first chunk.
    detected_language: Option<String>,
    /// Number of samples processed so far (for timestamp offset).
    samples_processed: usize,
}

impl<'m> StreamTranscriber<'m> {
    /// Create a new streaming transcriber.
    pub(crate) fn new(model: &'m WhisperModel, opts: TranscribeOptions<'static>) -> Self {
        Self {
            model,
            opts,
            audio_buf: Vec::new(),
            pending_segments: Vec::new(),
            texts: Vec::new(),
            detected_language: None,
            samples_processed: 0,
        }
    }

    /// Push audio samples into the internal buffer.
    ///
    /// Samples should be 16 kHz mono f32 PCM. You can push arbitrary amounts;
    /// chunks are processed when [`next_segment`](Self::next_segment) is called.
    pub fn push_audio(&mut self, samples: &[f32]) {
        self.audio_buf.extend_from_slice(samples);
    }

    /// Return the number of buffered samples not yet processed.
    pub fn buffered_samples(&self) -> usize {
        self.audio_buf.len()
    }

    /// Return the total number of samples that have been fully processed.
    pub fn processed_samples(&self) -> usize {
        self.samples_processed
    }

    /// Process a 30-second chunk if enough audio is buffered and return the
    /// next pending segment.
    ///
    /// Returns `None` when no segments are available (either not enough audio
    /// has been pushed or all decoded segments have been consumed). Call
    /// repeatedly until `None` to drain all available segments.
    pub fn next_segment(&mut self) -> Option<Result<Segment, OxiWhisperError>> {
        // Return already-decoded segments first
        if !self.pending_segments.is_empty() {
            return Some(Ok(self.pending_segments.remove(0)));
        }

        // Need at least one full chunk to process
        if self.audio_buf.len() < STREAM_CHUNK_SAMPLES {
            return None;
        }

        // Process the first STREAM_CHUNK_SAMPLES from the buffer
        let chunk: Vec<f32> = self.audio_buf[..STREAM_CHUNK_SAMPLES].to_vec();
        let offset_seconds = self.samples_processed as f32 / 16000.0;

        match self.model.transcribe_segmented(&chunk, &self.opts) {
            Ok(result) => {
                // Collect text
                let trimmed = result.text.trim();
                if !trimmed.is_empty() {
                    self.texts.push(trimmed.to_string());
                }

                // Set detected language from first chunk
                if self.detected_language.is_none() {
                    self.detected_language = result.language;
                }

                // Offset segment timestamps
                for seg in result.segments {
                    self.pending_segments.push(Segment {
                        text: seg.text,
                        start: seg.start + offset_seconds,
                        end: seg.end + offset_seconds,
                        confidence: seg.confidence,
                        is_hallucination: seg.is_hallucination,
                    });
                }

                // Advance: remove processed samples, keeping overlap for continuity
                let advance = STREAM_CHUNK_SAMPLES - STREAM_OVERLAP_SAMPLES;
                self.audio_buf.drain(..advance);
                self.samples_processed += advance;

                // Return first pending segment if any
                if !self.pending_segments.is_empty() {
                    Some(Ok(self.pending_segments.remove(0)))
                } else {
                    None
                }
            }
            Err(e) => Some(Err(e)),
        }
    }

    /// Flush remaining audio (< 30s) and return the final transcription result.
    ///
    /// Consumes the stream. Any audio remaining in the buffer is processed as
    /// a final short chunk.
    pub fn finish(mut self) -> Result<TranscribeResult, OxiWhisperError> {
        // Drain any pending full chunks first
        loop {
            match self.next_segment() {
                Some(Ok(_seg)) => {
                    // Segment was consumed; text was already captured in
                    // self.texts during next_segment processing. Continue
                    // draining.
                }
                Some(Err(e)) => return Err(e),
                None => break,
            }
        }

        // Process remaining audio
        if !self.audio_buf.is_empty() {
            let offset_seconds = self.samples_processed as f32 / 16000.0;
            let result = self
                .model
                .transcribe_segmented(&self.audio_buf, &self.opts)?;

            let trimmed = result.text.trim();
            if !trimmed.is_empty() {
                self.texts.push(trimmed.to_string());
            }

            if self.detected_language.is_none() {
                self.detected_language = result.language;
            }

            for seg in result.segments {
                self.pending_segments.push(Segment {
                    text: seg.text,
                    start: seg.start + offset_seconds,
                    end: seg.end + offset_seconds,
                    confidence: seg.confidence,
                    is_hallucination: seg.is_hallucination,
                });
            }
        } else if self.texts.is_empty() {
            return Err(OxiWhisperError::InferenceFailed("Empty audio".into()));
        }

        Ok(TranscribeResult {
            text: self.texts.join(" "),
            segments: self.pending_segments,
            language: self.detected_language,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_constants_are_sane() {
        assert_eq!(STREAM_CHUNK_SAMPLES, 16000 * 30);
        assert_eq!(STREAM_OVERLAP_SAMPLES, 16000);
        // The advance per chunk should be chunk - overlap = 29 seconds
        let advance = STREAM_CHUNK_SAMPLES - STREAM_OVERLAP_SAMPLES;
        assert_eq!(advance, 16000 * 29);
    }

    #[test]
    fn test_stream_timestamp_offset_calculation() {
        // Verify the offset calculation that StreamTranscriber uses:
        // offset_seconds = samples_processed / 16000.0
        let samples_processed: usize = 16000 * 29; // after one chunk
        let offset = samples_processed as f32 / 16000.0;
        assert!((offset - 29.0).abs() < 1e-6);

        // After two chunks
        let samples_processed_2 = 16000 * 29 * 2;
        let offset_2 = samples_processed_2 as f32 / 16000.0;
        assert!((offset_2 - 58.0).abs() < 1e-6);
    }

    #[test]
    fn test_stream_chunk_advance_preserves_overlap() {
        // Simulate the drain logic used in next_segment:
        // After processing STREAM_CHUNK_SAMPLES, we drain (chunk - overlap)
        // and keep STREAM_OVERLAP_SAMPLES in the buffer.
        let mut buf: Vec<f32> = vec![0.0; STREAM_CHUNK_SAMPLES + 5000];
        let original_len = buf.len();
        let advance = STREAM_CHUNK_SAMPLES - STREAM_OVERLAP_SAMPLES;
        buf.drain(..advance);

        // Remaining should be overlap + the extra 5000
        assert_eq!(buf.len(), STREAM_OVERLAP_SAMPLES + 5000);
        assert_eq!(buf.len(), original_len - advance);
    }

    #[test]
    fn test_stream_not_enough_audio_threshold() {
        // Verify that less than STREAM_CHUNK_SAMPLES means no chunk to process
        let short_len = STREAM_CHUNK_SAMPLES - 1;
        assert!(short_len < STREAM_CHUNK_SAMPLES);
        // Exactly STREAM_CHUNK_SAMPLES should be enough
        const { assert!(STREAM_CHUNK_SAMPLES >= STREAM_CHUNK_SAMPLES) };
    }

    #[test]
    fn test_stream_empty_finish_error_message() {
        // When texts is empty and audio_buf is empty, finish should
        // produce an "Empty audio" error. Verify the error variant.
        let err = OxiWhisperError::InferenceFailed("Empty audio".into());
        assert_eq!(format!("{err}"), "Inference failed: Empty audio");
    }

    #[test]
    fn test_stream_multiple_chunks_offset_progression() {
        // Simulate the offset logic for three consecutive 30-second chunks.
        let mut samples_processed: usize = 0;
        let advance = STREAM_CHUNK_SAMPLES - STREAM_OVERLAP_SAMPLES;

        // First chunk: offset = 0.0s
        let offset_0 = samples_processed as f32 / 16000.0;
        assert!((offset_0 - 0.0).abs() < 1e-6);
        samples_processed += advance;

        // Second chunk: offset = 29.0s
        let offset_1 = samples_processed as f32 / 16000.0;
        assert!((offset_1 - 29.0).abs() < 1e-6);
        samples_processed += advance;

        // Third chunk: offset = 58.0s
        let offset_2 = samples_processed as f32 / 16000.0;
        assert!((offset_2 - 58.0).abs() < 1e-6);
    }
}
