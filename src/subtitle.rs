//! Subtitle export: SRT and WebVTT formats.
//!
//! Converts [`Segment`] slices (produced by
//! [`WhisperModel::transcribe_segmented`](crate::WhisperModel::transcribe_segmented))
//! into industry-standard subtitle strings ready for file output.

use crate::Segment;

/// Format segments as SRT subtitles.
///
/// SRT format:
/// ```text
/// 1
/// 00:00:00,000 --> 00:00:02,500
/// Hello world
///
/// 2
/// 00:00:02,500 --> 00:00:05,000
/// Next line
/// ```
pub fn to_srt(segments: &[Segment]) -> String {
    let mut out = String::new();
    for (i, seg) in segments.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(&format!("{}\n", i + 1));
        out.push_str(&format!(
            "{} --> {}\n",
            format_srt_time(seg.start),
            format_srt_time(seg.end)
        ));
        out.push_str(&seg.text);
        out.push('\n');
    }
    out
}

/// Format segments as WebVTT subtitles.
///
/// VTT format:
/// ```text
/// WEBVTT
///
/// 00:00:00.000 --> 00:00:02.500
/// Hello world
///
/// 00:00:02.500 --> 00:00:05.000
/// Next line
/// ```
pub fn to_vtt(segments: &[Segment]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for (i, seg) in segments.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(&format!(
            "{} --> {}\n",
            format_vtt_time(seg.start),
            format_vtt_time(seg.end)
        ));
        out.push_str(&seg.text);
        out.push('\n');
    }
    out
}

fn format_srt_time(seconds: f32) -> String {
    let total_ms = (seconds * 1000.0) as u64;
    let h = total_ms / 3_600_000;
    let m = (total_ms % 3_600_000) / 60_000;
    let s = (total_ms % 60_000) / 1000;
    let ms = total_ms % 1000;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

fn format_vtt_time(seconds: f32) -> String {
    let total_ms = (seconds * 1000.0) as u64;
    let h = total_ms / 3_600_000;
    let m = (total_ms % 3_600_000) / 60_000;
    let s = (total_ms % 60_000) / 1000;
    let ms = total_ms % 1000;
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_segment(text: &str, start: f32, end: f32) -> Segment {
        Segment {
            text: text.to_string(),
            start,
            end,
            confidence: 0.0,
            is_hallucination: false,
        }
    }

    // ── SRT tests ──────────────────────────────────────────────────────

    #[test]
    fn srt_empty_segments() {
        let result = to_srt(&[]);
        assert_eq!(result, "");
    }

    #[test]
    fn srt_single_segment() {
        let segments = [make_segment("Hello world", 0.0, 2.5)];
        let result = to_srt(&segments);
        let expected = "1\n00:00:00,000 --> 00:00:02,500\nHello world\n";
        assert_eq!(result, expected);
    }

    #[test]
    fn srt_multiple_segments() {
        let segments = [
            make_segment("First line", 0.0, 2.5),
            make_segment("Second line", 2.5, 5.0),
            make_segment("Third line", 5.0, 8.0),
        ];
        let result = to_srt(&segments);
        let expected = "\
1\n00:00:00,000 --> 00:00:02,500\nFirst line\n\n\
2\n00:00:02,500 --> 00:00:05,000\nSecond line\n\n\
3\n00:00:05,000 --> 00:00:08,000\nThird line\n";
        assert_eq!(result, expected);
    }

    #[test]
    fn srt_time_with_hours() {
        let segments = [make_segment("Late segment", 3661.5, 3665.0)];
        let result = to_srt(&segments);
        assert!(result.contains("01:01:01,500 --> 01:01:05,000"));
    }

    #[test]
    fn srt_exact_seconds() {
        let segments = [make_segment("Exact", 60.0, 120.0)];
        let result = to_srt(&segments);
        assert!(result.contains("00:01:00,000 --> 00:02:00,000"));
    }

    // ── VTT tests ──────────────────────────────────────────────────────

    #[test]
    fn vtt_empty_segments() {
        let result = to_vtt(&[]);
        assert_eq!(result, "WEBVTT\n\n");
    }

    #[test]
    fn vtt_single_segment() {
        let segments = [make_segment("Hello world", 0.0, 2.5)];
        let result = to_vtt(&segments);
        let expected = "WEBVTT\n\n00:00:00.000 --> 00:00:02.500\nHello world\n";
        assert_eq!(result, expected);
    }

    #[test]
    fn vtt_multiple_segments() {
        let segments = [
            make_segment("First line", 0.0, 2.5),
            make_segment("Second line", 2.5, 5.0),
        ];
        let result = to_vtt(&segments);
        let expected = "WEBVTT\n\n\
00:00:00.000 --> 00:00:02.500\nFirst line\n\n\
00:00:02.500 --> 00:00:05.000\nSecond line\n";
        assert_eq!(result, expected);
    }

    #[test]
    fn vtt_time_with_hours() {
        let segments = [make_segment("Late", 7200.0, 7201.5)];
        let result = to_vtt(&segments);
        assert!(result.contains("02:00:00.000 --> 02:00:01.500"));
    }

    #[test]
    fn vtt_uses_dot_separator() {
        // VTT uses '.' not ',' for millisecond separator
        let segments = [make_segment("Test", 1.234, 5.678)];
        let result = to_vtt(&segments);
        assert!(result.contains('.'));
        assert!(!result.contains(','));
    }

    // ── Time formatting edge cases ─────────────────────────────────────

    #[test]
    fn srt_time_zero() {
        assert_eq!(format_srt_time(0.0), "00:00:00,000");
    }

    #[test]
    fn vtt_time_zero() {
        assert_eq!(format_vtt_time(0.0), "00:00:00.000");
    }

    #[test]
    fn srt_time_sub_second() {
        assert_eq!(format_srt_time(0.123), "00:00:00,123");
    }

    #[test]
    fn vtt_time_sub_second() {
        assert_eq!(format_vtt_time(0.123), "00:00:00.123");
    }

    #[test]
    fn srt_time_large_hours() {
        // 10 hours, 30 minutes, 15.5 seconds (use exact f32-representable value)
        assert_eq!(format_srt_time(37815.5), "10:30:15,500");
    }

    #[test]
    fn vtt_time_large_hours() {
        assert_eq!(format_vtt_time(37815.5), "10:30:15.500");
    }
}
