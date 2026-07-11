// SPDX-License-Identifier: Apache-2.0
// Copyright (c) COOLJAPAN OU (Team Kitasan)

//! Speaker-attributed transcription: fuse "who spoke when" with "what was said".
//!
//! [`attribute_words`] takes the per-word timings produced by
//! [`WhisperModel::transcribe_words`](crate::WhisperModel::transcribe_words) and
//! the speaker timeline produced by
//! [`WhisperModel::diarize`](crate::WhisperModel::diarize) and merges them into a
//! [`SpeakerTranscript`]: a chronological list of [`SpeakerTurn`]s, each carrying
//! the contiguous run of words a single speaker uttered.
//!
//! # Fusion rule
//!
//! Each [`WordSegment`] is attributed to exactly one speaker by its temporal
//! **midpoint** `0.5 * (start + end)`:
//!
//! * If some [`SpeakerSegment`]'s half-open span `[start, end)` contains the
//!   midpoint, the word is assigned to that speaker.
//! * Otherwise the word falls in a gap of — or past the ends of — the diarized
//!   timeline and is attributed to the **nearest** segment, measured as the
//!   distance from the midpoint to the closed interval `[start, end]` (zero when
//!   inside). Ties are broken toward the earliest segment (lowest `start`, then
//!   lowest index) so the result is fully deterministic.
//!
//! Consecutive words that resolve to the same [`SpeakerId`] are then collapsed
//! into a single [`SpeakerTurn`].
//!
//! # Honest limitation — overlapped speech
//!
//! This fusion assigns **exactly one speaker per word**. When two people talk at
//! once, the diarized timeline still names a single speaker for that instant, so
//! every overlapping word is credited to whichever speaker owns its midpoint and
//! the other talker is dropped. Proper overlap-aware attribution (word-level
//! multi-speaker labels) is deferred to Batch G. Do **not** read a
//! [`SpeakerTranscript`] as a faithful record of simultaneous speech.

use super::{SpeakerId, SpeakerSegment};
use crate::WordSegment;

/// One contiguous run of words spoken by a single speaker.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerTurn {
    /// Cluster label of the speaker who uttered this turn.
    pub speaker: SpeakerId,
    /// Trimmed text of the turn (the grouped words joined in order).
    pub text: String,
    /// Start time of the first word in the turn, in seconds.
    pub start: f32,
    /// End time of the last word in the turn, in seconds.
    pub end: f32,
    /// The raw per-word segments that make up this turn, in chronological order.
    pub words: Vec<WordSegment>,
}

/// A full transcript partitioned into speaker turns.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerTranscript {
    /// Speaker turns in chronological order.
    pub turns: Vec<SpeakerTurn>,
    /// Number of **distinct** speakers that actually received at least one word.
    ///
    /// This counts the speakers present in [`turns`](Self::turns), which can be
    /// **fewer** than
    /// [`DiarizeResult::num_speakers`](crate::diarize::DiarizeResult::num_speakers):
    /// a clustered speaker whose spans never won a word midpoint contributes no
    /// turn and is therefore not counted here.
    pub num_speakers: usize,
    /// Detected or specified BCP-47 language code, carried through verbatim from
    /// the source transcript.
    pub language: Option<String>,
}

/// Fuse per-word timings with a speaker timeline into a [`SpeakerTranscript`].
///
/// See the [module documentation](self) for the full attribution rule and its
/// overlapped-speech limitation. This function is pure and deterministic: the
/// same inputs always produce the same output, and it never allocates a speaker
/// that did not receive a word.
///
/// # Parameters
///
/// * `words` — per-word segments in chronological order, typically
///   [`WordTimedTranscript::words`](crate::WordTimedTranscript::words).
/// * `segments` — the diarized speaker timeline, typically
///   [`DiarizeResult::segments`](crate::diarize::DiarizeResult::segments). It
///   need not be sorted or non-overlapping; attribution is defined for any set.
/// * `language` — carried through verbatim into
///   [`SpeakerTranscript::language`].
///
/// # Empty timeline
///
/// If `segments` is empty there is no speaker to attribute words to, so this
/// returns `SpeakerTranscript { turns: vec![], num_speakers: 0, language }` — it
/// never fabricates a speaker, even when `words` is non-empty.
pub fn attribute_words(
    words: &[WordSegment],
    segments: &[SpeakerSegment],
    language: Option<String>,
) -> SpeakerTranscript {
    // No speaker timeline → no attribution. Never invent a speaker.
    if segments.is_empty() {
        return SpeakerTranscript {
            turns: Vec::new(),
            num_speakers: 0,
            language,
        };
    }

    // Assign each word to a speaker by its midpoint and collapse consecutive
    // same-speaker words into turns as we go.
    let mut turns: Vec<SpeakerTurn> = Vec::new();
    for word in words {
        let mid = 0.5 * (word.start + word.end);
        // `assign_segment_index` yields `None` only for an empty timeline, which
        // the guard above already excluded. Should a future change route a word
        // here with nothing to assign, skip it rather than fabricate a speaker.
        let Some(idx) = assign_segment_index(mid, segments) else {
            continue;
        };
        let speaker = segments[idx].speaker;

        match turns.last_mut() {
            Some(turn) if turn.speaker == speaker => {
                turn.end = word.end;
                turn.words.push(word.clone());
            }
            _ => {
                turns.push(SpeakerTurn {
                    speaker,
                    text: String::new(),
                    start: word.start,
                    end: word.end,
                    words: vec![word.clone()],
                });
            }
        }
    }

    // Finalize turn text: concatenate the grouped words (which already carry
    // Whisper's leading-space convention) and trim the whole turn once.
    for turn in &mut turns {
        let mut text = String::new();
        for word in &turn.words {
            text.push_str(&word.word);
        }
        turn.text = text.trim().to_string();
    }

    // Count distinct speakers that actually received words (may be fewer than
    // the diarizer's cluster count — see `SpeakerTranscript::num_speakers`).
    let num_speakers = turns
        .iter()
        .map(|turn| turn.speaker)
        .collect::<std::collections::BTreeSet<_>>()
        .len();

    SpeakerTranscript {
        turns,
        num_speakers,
        language,
    }
}

/// Distance from `mid` to the closed interval `[seg.start, seg.end]`, or zero
/// when `mid` lies inside it.
fn gap_to_segment(mid: f32, seg: &SpeakerSegment) -> f32 {
    if mid < seg.start {
        seg.start - mid
    } else if mid > seg.end {
        mid - seg.end
    } else {
        0.0
    }
}

/// Index of the segment that `mid` is attributed to (see the module docs), or
/// `None` when `segments` is empty.
///
/// A half-open `[start, end)` containment wins outright; otherwise the nearest
/// segment by [`gap_to_segment`] is chosen, tie-broken toward the earliest
/// segment (lowest `start`, then lowest index).
fn assign_segment_index(mid: f32, segments: &[SpeakerSegment]) -> Option<usize> {
    // Stage 1: half-open [start, end) containment.
    let mut covering: Option<usize> = None;
    for (i, seg) in segments.iter().enumerate() {
        if seg.start <= mid && mid < seg.end {
            let better = match covering {
                None => true,
                Some(j) => seg.start < segments[j].start,
            };
            if better {
                covering = Some(i);
            }
        }
    }
    if covering.is_some() {
        return covering;
    }

    // Stage 2: nearest by gap to the closed interval, tie-broken to earliest.
    // Comparing the `(gap, start)` key via `partial_cmp` keeps the tie-break
    // float-`==`-free; iterating forward and replacing only on strictly-less
    // preserves the lowest-index tie-break among equal keys.
    let mut best: Option<(usize, f32, f32)> = None; // (index, gap, start)
    for (i, seg) in segments.iter().enumerate() {
        let gap = gap_to_segment(mid, seg);
        let replace = match best {
            None => true,
            Some((_, best_gap, best_start)) => {
                (gap, seg.start).partial_cmp(&(best_gap, best_start))
                    == Some(std::cmp::Ordering::Less)
            }
        };
        if replace {
            best = Some((i, gap, seg.start));
        }
    }
    best.map(|(i, _, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(text: &str, start: f32, end: f32) -> WordSegment {
        WordSegment {
            word: text.to_string(),
            start,
            end,
            confidence: -0.1,
        }
    }

    fn seg(id: u32, start: f32, end: f32) -> SpeakerSegment {
        SpeakerSegment {
            speaker: SpeakerId(id),
            start,
            end,
        }
    }

    #[test]
    fn test_midpoint_inside_assignment() {
        // mid = 0.75 lies inside seg 0's [0.0, 2.0).
        let words = [word("hello", 0.5, 1.0)];
        let segments = [seg(0, 0.0, 2.0), seg(1, 2.0, 4.0)];
        let t = attribute_words(&words, &segments, Some("en".to_string()));

        assert_eq!(t.num_speakers, 1);
        assert_eq!(t.language, Some("en".to_string()));
        assert_eq!(t.turns.len(), 1);
        let turn = &t.turns[0];
        assert_eq!(turn.speaker, SpeakerId(0));
        assert_eq!(turn.text, "hello");
        assert_eq!(turn.start, 0.5);
        assert_eq!(turn.end, 1.0);
        assert_eq!(turn.words, vec![word("hello", 0.5, 1.0)]);
    }

    #[test]
    fn test_gap_fallback_to_nearest() {
        // mid = 4.5 is covered by no segment: gap to seg 0 is 3.5, gap to seg 1
        // is 0.5, so it falls to the nearer seg 1.
        let words = [word("hi", 4.25, 4.75)];
        let segments = [seg(0, 0.0, 1.0), seg(1, 5.0, 6.0)];
        let t = attribute_words(&words, &segments, None);

        assert_eq!(t.num_speakers, 1);
        assert_eq!(t.turns.len(), 1);
        assert_eq!(t.turns[0].speaker, SpeakerId(1));
        assert_eq!(t.turns[0].text, "hi");
        assert_eq!(t.turns[0].start, 4.25);
        assert_eq!(t.turns[0].end, 4.75);
    }

    #[test]
    fn test_gap_fallback_tie_breaks_to_earliest() {
        // mid = 3.0 sits equidistant (gap 2.0) between seg 0 [0,1] and seg 1
        // [5,6]; the tie breaks to the earliest segment (lowest start) → seg 0.
        let words = [word("mid", 2.75, 3.25)];
        let segments = [seg(0, 0.0, 1.0), seg(1, 5.0, 6.0)];
        let t = attribute_words(&words, &segments, None);

        assert_eq!(t.num_speakers, 1);
        assert_eq!(t.turns.len(), 1);
        assert_eq!(t.turns[0].speaker, SpeakerId(0));
    }

    #[test]
    fn test_half_open_boundary_assigns_to_later_segment() {
        // mid = 1.0 sits exactly on the shared edge; half-open [start, end)
        // excludes seg 0's end and includes seg 1's start, so it goes to seg 1.
        let words = [word("x", 0.75, 1.25)];
        let segments = [seg(0, 0.0, 1.0), seg(1, 1.0, 2.0)];
        let t = attribute_words(&words, &segments, None);

        assert_eq!(t.num_speakers, 1);
        assert_eq!(t.turns.len(), 1);
        assert_eq!(t.turns[0].speaker, SpeakerId(1));
    }

    #[test]
    fn test_consecutive_collapse_two_speakers() {
        // Two words in seg 0, three in seg 1: exactly two turns with correct
        // boundaries and trimmed, space-joined text.
        let segments = [seg(0, 0.0, 2.0), seg(1, 2.0, 4.0)];
        let words = [
            word("Hello", 0.0, 0.5),
            word(" world", 0.5, 1.0),
            word(" how", 2.0, 2.5),
            word(" are", 2.5, 3.0),
            word(" you", 3.0, 3.5),
        ];
        let t = attribute_words(&words, &segments, Some("en".to_string()));

        assert_eq!(t.num_speakers, 2);
        assert_eq!(t.turns.len(), 2);

        let first = &t.turns[0];
        assert_eq!(first.speaker, SpeakerId(0));
        assert_eq!(first.text, "Hello world");
        assert_eq!(first.start, 0.0);
        assert_eq!(first.end, 1.0);
        assert_eq!(first.words.len(), 2);

        let second = &t.turns[1];
        assert_eq!(second.speaker, SpeakerId(1));
        assert_eq!(second.text, "how are you");
        assert_eq!(second.start, 2.0);
        assert_eq!(second.end, 3.5);
        assert_eq!(second.words.len(), 3);
    }

    #[test]
    fn test_single_speaker_whole_utterance() {
        let segments = [seg(0, 0.0, 5.0)];
        let words = [
            word("One", 0.0, 1.0),
            word(" two", 1.0, 2.0),
            word(" three", 2.0, 3.0),
        ];
        let t = attribute_words(&words, &segments, Some("en".to_string()));

        assert_eq!(t.num_speakers, 1);
        assert_eq!(t.turns.len(), 1);
        let turn = &t.turns[0];
        assert_eq!(turn.speaker, SpeakerId(0));
        assert_eq!(turn.text, "One two three");
        assert_eq!(turn.start, 0.0);
        assert_eq!(turn.end, 3.0);
        assert_eq!(turn.words.len(), 3);
    }

    #[test]
    fn test_empty_segments_no_fabricated_speaker() {
        let words = [word("a", 0.0, 1.0), word(" b", 1.0, 2.0)];
        let segments: [SpeakerSegment; 0] = [];
        let t = attribute_words(&words, &segments, Some("ja".to_string()));

        assert!(t.turns.is_empty());
        assert_eq!(t.num_speakers, 0);
        assert_eq!(t.language, Some("ja".to_string()));
    }

    #[test]
    fn test_num_speakers_excludes_wordless_cluster() {
        // Three clustered speakers, but only 0 and 2 win any word midpoint, so
        // num_speakers is 2 (< the diarizer's 3-speaker timeline).
        let segments = [seg(0, 0.0, 1.0), seg(1, 1.0, 2.0), seg(2, 2.0, 3.0)];
        let words = [word("a", 0.0, 1.0), word(" c", 2.0, 3.0)];
        let t = attribute_words(&words, &segments, None);

        assert_eq!(t.num_speakers, 2);
        assert_eq!(t.turns.len(), 2);
        assert_eq!(t.turns[0].speaker, SpeakerId(0));
        assert_eq!(t.turns[0].text, "a");
        assert_eq!(t.turns[1].speaker, SpeakerId(2));
        assert_eq!(t.turns[1].text, "c");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_speaker_transcript_serde_round_trip() {
        let segments = [seg(0, 0.0, 2.0), seg(1, 2.0, 4.0)];
        let words = [
            word("Hello", 0.0, 0.5),
            word(" there", 0.5, 1.0),
            word(" friend", 2.0, 2.5),
        ];
        let original = attribute_words(&words, &segments, Some("en".to_string()));

        let json = serde_json::to_string(&original).expect("serialize SpeakerTranscript");
        let decoded: SpeakerTranscript =
            serde_json::from_str(&json).expect("deserialize SpeakerTranscript");
        assert_eq!(original, decoded);
    }
}
