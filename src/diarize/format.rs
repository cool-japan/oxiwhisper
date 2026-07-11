// SPDX-License-Identifier: Apache-2.0
// Copyright (c) COOLJAPAN OU (Team Kitasan)

//! Speaker-labeled output formats: NIST RTTM export and human-readable
//! speaker-tagged transcripts.
//!
//! RTTM ("Rich Transcription Time Marked") is the de-facto interchange format
//! for diarization output: it is what reference toolkits (`pyannote.metrics`,
//! `dscore`, NIST's own `md-eval`) expect on both the hypothesis and reference
//! side of a DER/JER computation, which is why [`write_rttm`] and
//! [`rttm_string`] exist ahead of the DER/JER metrics work — they are the
//! shared substrate that makes that scoring checkable at all.
//!
//! [`labeled_transcript`] and [`labeled_transcript_timed`] instead target a
//! human reader: they render a [`SpeakerTranscript`] (produced by
//! [`attribute_words`](crate::diarize::attribute::attribute_words)) as a
//! `[SPEAKER_k]`-prefixed line per turn.

use super::DiarizeResult;
use super::attribute::SpeakerTranscript;

/// Write `result` as NIST RTTM `SPEAKER` lines to `w`, one line per segment.
///
/// Each line has the standard 10 whitespace-separated tokens (the `SPEAKER`
/// type marker plus 9 fields):
///
/// ```text
/// SPEAKER <uri> 1 <tbeg> <tdur> <NA> <NA> speaker_<id> <NA> <NA>
/// ```
///
/// * `<uri>` is the `uri` parameter verbatim (the conventional RTTM file/session
///   identifier) — it is not escaped or validated, so callers must supply a
///   value that contains no whitespace or newlines if the output is meant to
///   round-trip through RTTM parsers.
/// * Channel is always `1`: this crate does not track a channel dimension.
/// * `<tbeg>` is [`SpeakerSegment::start`](crate::diarize::SpeakerSegment::start),
///   fixed to 3 decimal places.
/// * `<tdur>` is [`SpeakerSegment::duration`](crate::diarize::SpeakerSegment::duration)
///   (`(end - start).max(0.0)`), fixed to 3 decimal places — never negative
///   even if a malformed segment has `end < start`.
/// * `<ortho>`, `<stype>`, `<conf>`, `<slat>` are always the RTTM `<NA>`
///   placeholder: this crate does not produce orthography variants, subtype
///   tags, confidence scores, or signal-lookahead metadata.
/// * `<name>` is `speaker_<id>` where `<id>` is the
///   [`SpeakerId`](crate::diarize::SpeakerId)'s inner `u32`.
///
/// Segments are emitted in the order they appear in `result.segments` (no
/// implicit sorting): callers that need chronological RTTM should ensure
/// `result.segments` is already sorted, e.g. via the resegmentation stage's
/// output ordering.
///
/// # Errors
///
/// Propagates any [`std::io::Error`] from writing to `w`.
pub fn write_rttm<W: std::io::Write>(
    result: &DiarizeResult,
    uri: &str,
    w: &mut W,
) -> std::io::Result<()> {
    for seg in &result.segments {
        let tbeg = seg.start;
        let tdur = seg.duration();
        let sid = seg.speaker.0;
        writeln!(
            w,
            "SPEAKER {uri} 1 {tbeg:.3} {tdur:.3} <NA> <NA> speaker_{sid} <NA> <NA>"
        )?;
    }
    Ok(())
}

/// Render `result` as an RTTM document and return it as a `String`.
///
/// Convenience wrapper around [`write_rttm`] for callers that want the text
/// in memory (e.g. to embed in a JSON payload) rather than to stream it to a
/// file or socket. See [`write_rttm`] for the exact line format.
///
/// This cannot fail: writing to an in-memory `Vec<u8>` never produces an
/// I/O error, and every byte [`write_rttm`] emits (ASCII digits, `.`,
/// space, `<`, `>`, `_`, plus the caller-supplied `uri`, which is already a
/// valid `&str`) is valid UTF-8.
pub fn rttm_string(result: &DiarizeResult, uri: &str) -> String {
    let mut buf = Vec::new();
    write_rttm(result, uri, &mut buf).expect("writing RTTM to an in-memory Vec<u8> is infallible");
    String::from_utf8(buf).expect("write_rttm only ever emits valid UTF-8 given a valid &str uri")
}

/// Render `transcript` as one `[SPEAKER_<id>] <text>` line per turn, joined
/// by `\n` with no trailing newline.
///
/// `<text>` is the turn's already-trimmed [`SpeakerTurn::text`](crate::diarize::attribute::SpeakerTurn::text)
/// verbatim. An empty transcript (no turns) renders as the empty string.
pub fn labeled_transcript(transcript: &SpeakerTranscript) -> String {
    transcript
        .turns
        .iter()
        .map(|turn| {
            let sid = turn.speaker.0;
            let text = &turn.text;
            format!("[SPEAKER_{sid}] {text}")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Render `transcript` as one `[<start> -> <end>] [SPEAKER_<id>] <text>` line
/// per turn, joined by `\n` with no trailing newline.
///
/// `<start>` and `<end>` are the turn's
/// [`SpeakerTurn::start`](crate::diarize::attribute::SpeakerTurn::start) and
/// [`SpeakerTurn::end`](crate::diarize::attribute::SpeakerTurn::end), fixed to
/// 2 decimal places. An empty transcript (no turns) renders as the empty
/// string.
pub fn labeled_transcript_timed(transcript: &SpeakerTranscript) -> String {
    transcript
        .turns
        .iter()
        .map(|turn| {
            let start = turn.start;
            let end = turn.end;
            let sid = turn.speaker.0;
            let text = &turn.text;
            format!("[{start:.2} -> {end:.2}] [SPEAKER_{sid}] {text}")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WordSegment;
    use crate::diarize::attribute::SpeakerTurn;
    use crate::diarize::{SpeakerId, SpeakerSegment};

    fn seg(id: u32, start: f32, end: f32) -> SpeakerSegment {
        SpeakerSegment {
            speaker: SpeakerId(id),
            start,
            end,
        }
    }

    fn word(text: &str, start: f32, end: f32) -> WordSegment {
        WordSegment {
            word: text.to_string(),
            start,
            end,
            confidence: -0.1,
        }
    }

    // ── RTTM: write_rttm / rttm_string ──────────────────────────────────

    #[test]
    fn rttm_empty_result_is_empty_string() {
        let result = DiarizeResult {
            segments: vec![],
            num_speakers: 0,
        };
        assert_eq!(rttm_string(&result, "meeting"), "");

        let mut buf = Vec::new();
        write_rttm(&result, "meeting", &mut buf).expect("write to Vec<u8> cannot fail");
        assert_eq!(buf, b"");
    }

    #[test]
    fn rttm_single_segment_exact_line() {
        let result = DiarizeResult {
            segments: vec![seg(0, 0.0, 1.5)],
            num_speakers: 1,
        };
        let expected = "SPEAKER meeting 1 0.000 1.500 <NA> <NA> speaker_0 <NA> <NA>\n";
        assert_eq!(rttm_string(&result, "meeting"), expected);
    }

    #[test]
    fn rttm_multiple_segments_exact_lines() {
        let result = DiarizeResult {
            segments: vec![seg(0, 0.0, 1.5), seg(1, 1.5, 3.25), seg(0, 3.25, 4.0)],
            num_speakers: 2,
        };
        let expected = "\
SPEAKER call42 1 0.000 1.500 <NA> <NA> speaker_0 <NA> <NA>\n\
SPEAKER call42 1 1.500 1.750 <NA> <NA> speaker_1 <NA> <NA>\n\
SPEAKER call42 1 3.250 0.750 <NA> <NA> speaker_0 <NA> <NA>\n";
        assert_eq!(rttm_string(&result, "call42"), expected);
    }

    #[test]
    fn rttm_negative_span_clamps_duration_to_zero() {
        // A malformed segment with end < start must still round-trip: tbeg is
        // emitted verbatim, but tdur clamps via SpeakerSegment::duration().
        let result = DiarizeResult {
            segments: vec![seg(3, 5.0, 4.5)],
            num_speakers: 1,
        };
        let expected = "SPEAKER x 1 5.000 0.000 <NA> <NA> speaker_3 <NA> <NA>\n";
        assert_eq!(rttm_string(&result, "x"), expected);
    }

    #[test]
    fn rttm_rounds_third_decimal_half_to_even_or_up_consistently_with_format() {
        // 0.1234 rounds to 3 decimals via Rust's `{:.3}` formatting (round
        // half to even on the underlying grisu/dragon formatter); pin the
        // exact digits so any formatting regression is caught byte-for-byte.
        let result = DiarizeResult {
            segments: vec![seg(7, 0.1234, 0.5)],
            num_speakers: 1,
        };
        let expected = "SPEAKER u 1 0.123 0.377 <NA> <NA> speaker_7 <NA> <NA>\n";
        assert_eq!(rttm_string(&result, "u"), expected);
    }

    #[test]
    fn write_rttm_matches_rttm_string() {
        let result = DiarizeResult {
            segments: vec![seg(0, 0.0, 2.0), seg(1, 2.0, 5.0)],
            num_speakers: 2,
        };
        let mut buf = Vec::new();
        write_rttm(&result, "sess", &mut buf).expect("write to Vec<u8> cannot fail");
        let from_writer = String::from_utf8(buf).expect("ASCII-only RTTM output is valid UTF-8");
        assert_eq!(from_writer, rttm_string(&result, "sess"));
    }

    #[test]
    fn write_rttm_to_temp_file_round_trips_exact_bytes() {
        let path = std::env::temp_dir().join(format!(
            "oxiwhisper_e2_rttm_test_{}_{:?}.rttm",
            std::process::id(),
            std::thread::current().id()
        ));
        let result = DiarizeResult {
            segments: vec![seg(0, 0.0, 1.0), seg(1, 1.0, 2.5)],
            num_speakers: 2,
        };
        {
            let mut file = std::fs::File::create(&path).expect("create temp RTTM file");
            write_rttm(&result, "filetest", &mut file).expect("write RTTM to temp file");
        }
        let contents = std::fs::read_to_string(&path).expect("read back temp RTTM file");
        let _ = std::fs::remove_file(&path);

        let expected = "\
SPEAKER filetest 1 0.000 1.000 <NA> <NA> speaker_0 <NA> <NA>\n\
SPEAKER filetest 1 1.000 1.500 <NA> <NA> speaker_1 <NA> <NA>\n";
        assert_eq!(contents, expected);
    }

    // ── labeled_transcript / labeled_transcript_timed ───────────────────

    fn turn(id: u32, text: &str, start: f32, end: f32, words: Vec<WordSegment>) -> SpeakerTurn {
        SpeakerTurn {
            speaker: SpeakerId(id),
            text: text.to_string(),
            start,
            end,
            words,
        }
    }

    #[test]
    fn labeled_transcript_empty_turns_is_empty_string() {
        let transcript = SpeakerTranscript {
            turns: vec![],
            num_speakers: 0,
            language: None,
        };
        assert_eq!(labeled_transcript(&transcript), "");
        assert_eq!(labeled_transcript_timed(&transcript), "");
    }

    #[test]
    fn labeled_transcript_single_turn_exact() {
        let transcript = SpeakerTranscript {
            turns: vec![turn(
                0,
                "hello there",
                0.5,
                1.25,
                vec![word("hello there", 0.5, 1.25)],
            )],
            num_speakers: 1,
            language: Some("en".to_string()),
        };
        assert_eq!(labeled_transcript(&transcript), "[SPEAKER_0] hello there");
    }

    #[test]
    fn labeled_transcript_multiple_turns_joined_by_newline() {
        let transcript = SpeakerTranscript {
            turns: vec![
                turn(0, "hi", 0.0, 1.0, vec![word("hi", 0.0, 1.0)]),
                turn(
                    1,
                    "hello back",
                    1.0,
                    2.5,
                    vec![word("hello back", 1.0, 2.5)],
                ),
                turn(0, "great", 2.5, 3.0, vec![word("great", 2.5, 3.0)]),
            ],
            num_speakers: 2,
            language: None,
        };
        let expected = "[SPEAKER_0] hi\n[SPEAKER_1] hello back\n[SPEAKER_0] great";
        assert_eq!(labeled_transcript(&transcript), expected);
    }

    #[test]
    fn labeled_transcript_timed_single_turn_exact() {
        let transcript = SpeakerTranscript {
            turns: vec![turn(
                2,
                "testing",
                0.1,
                0.987,
                vec![word("testing", 0.1, 0.987)],
            )],
            num_speakers: 1,
            language: None,
        };
        assert_eq!(
            labeled_transcript_timed(&transcript),
            "[0.10 -> 0.99] [SPEAKER_2] testing"
        );
    }

    #[test]
    fn labeled_transcript_timed_multiple_turns_joined_by_newline() {
        let transcript = SpeakerTranscript {
            turns: vec![
                turn(0, "first", 0.0, 1.5, vec![word("first", 0.0, 1.5)]),
                turn(1, "second", 1.5, 3.0, vec![word("second", 1.5, 3.0)]),
            ],
            num_speakers: 2,
            language: Some("ja".to_string()),
        };
        let expected = "[0.00 -> 1.50] [SPEAKER_0] first\n[1.50 -> 3.00] [SPEAKER_1] second";
        assert_eq!(labeled_transcript_timed(&transcript), expected);
    }
}
