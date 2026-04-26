//! Pure vocab pass-through decoding for Whisper's BPE token vocabulary.
//!
//! This module performs **pure vocab pass-through decoding** — there is no
//! `encode()` function and no merge table. Whisper's GPT-2 byte-level decoding
//! is performed once at GGML load time (see `model::ModelData::load`), so by
//! the time `decode` runs, every `VocabEntry::text` is already valid UTF-8.
//! oxiwhisper is inference-only.
//!
//! # Elision rules
//! - **Special tokens** whose `text` begins with `"<|"` are silently dropped.
//! - **Out-of-range token IDs** (`id >= vocab.len()`) are silently dropped.
//!
//! Neither elision produces an error — this is intentional for robustness
//! against malformed decoder output.

use crate::model::VocabEntry;

/// First timestamp token ID in Whisper vocabulary.
pub const TIMESTAMP_BEGIN: u32 = 50364;

/// Each timestamp token represents 20ms (0.02 seconds).
pub const TIMESTAMP_RESOLUTION: f32 = 0.02;

/// Special token IDs for Whisper
pub struct SpecialTokens {
    /// `<|startoftranscript|>` — begins the decoder sequence.
    pub sot: u32,
    /// `<|endoftext|>` — signals the end of the decoder output.
    pub eot: u32,
    /// `<|transcribe|>` — task token selecting transcription mode.
    pub transcribe: u32,
    /// `<|translate|>` — task token selecting translation mode.
    pub translate: u32,
    /// `<|notimestamps|>` — suppresses timestamp tokens in the output.
    pub no_timestamps: u32,
    /// `<|nospeech|>` — emitted when no speech is detected in the audio.
    pub no_speech: u32,
    /// `<|startofprev|>` — precedes previous-segment context tokens.
    pub sot_prev: u32,
}

impl SpecialTokens {
    /// Initialise special token IDs for a Whisper multilingual vocabulary.
    ///
    /// The `n_vocab` argument is accepted for forward compatibility but not
    /// currently used — the token IDs are fixed for all Whisper models.
    pub fn new(_n_vocab: usize) -> Self {
        // Whisper special tokens are at the end of the vocabulary
        // For multilingual models:
        // <|endoftext|> = 50256
        // <|startoftranscript|> = 50258
        // language tokens at 50259..50358
        // <|translate|> = 50358
        // <|transcribe|> = 50359
        // <|startoflm|> = 50360
        // <|startofprev|> = 50361
        // <|nospeech|> = 50362
        // <|notimestamps|> = 50363
        // timestamp tokens at 50364..
        Self {
            eot: 50256,
            sot: 50258,
            translate: 50358,
            transcribe: 50359,
            no_speech: 50362,
            no_timestamps: 50363,
            sot_prev: 50361,
        }
    }

    /// Returns `true` if `token` is a timestamp token (ID >= `TIMESTAMP_BEGIN`).
    pub fn is_timestamp(token: u32) -> bool {
        token >= TIMESTAMP_BEGIN
    }

    /// Convert a timestamp token ID to seconds.
    ///
    /// Returns `0.0` for tokens below `TIMESTAMP_BEGIN`.
    pub fn timestamp_seconds(token: u32) -> f32 {
        if token < TIMESTAMP_BEGIN {
            return 0.0;
        }
        (token - TIMESTAMP_BEGIN) as f32 * TIMESTAMP_RESOLUTION
    }

    /// Get language token ID for a language code
    pub fn language_token(&self, lang: &str) -> u32 {
        let languages = [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar",
            "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu",
            "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa",
            "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn",
            "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
            "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
            "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw",
            "su",
        ];

        for (i, &l) in languages.iter().enumerate() {
            if l == lang {
                return 50259 + i as u32;
            }
        }

        // Default to English
        50259
    }
}

/// Decode Whisper token IDs to a UTF-8 string.
///
/// GGML stores each token's text as raw UTF-8 (already decoded from GPT-2 byte encoding).
/// We simply concatenate the text of all non-special tokens.
pub fn decode(token_ids: &[u32], vocab: &[VocabEntry]) -> String {
    let mut result = String::with_capacity(token_ids.len() * 3);

    for &id in token_ids {
        let idx = id as usize;
        if idx >= vocab.len() {
            continue;
        }
        let text = &vocab[idx].text;
        if text.starts_with("<|") {
            continue;
        }
        result.push_str(text);
    }

    result
}

/// Parse a sequence of token IDs (which may contain timestamp tokens) into text segments.
///
/// Each segment has a start time (seconds), end time (seconds), and text.
/// Timestamp tokens bracket text: `<|0.00|> Hello world <|2.00|>`.
///
/// If no timestamp tokens are found, returns a single segment with `(0.0, 0.0, full_text)`.
pub fn parse_segments(token_ids: &[u32], vocab: &[VocabEntry]) -> Vec<(f32, f32, String)> {
    let mut segments: Vec<(f32, f32, String)> = Vec::new();
    let mut current_start: Option<f32> = None;
    let mut current_text = String::new();
    let mut found_any_timestamp = false;

    for &id in token_ids {
        if SpecialTokens::is_timestamp(id) {
            found_any_timestamp = true;
            let time = SpecialTokens::timestamp_seconds(id);

            match current_start {
                None => {
                    // Opening timestamp -- start a new segment
                    current_start = Some(time);
                    current_text.clear();
                }
                Some(start) => {
                    // Closing timestamp -- finalize segment
                    let trimmed = current_text.trim().to_string();
                    if !trimmed.is_empty() {
                        segments.push((start, time, trimmed));
                    }
                    // This closing timestamp may also be the opening of the next segment
                    current_start = Some(time);
                    current_text.clear();
                }
            }
        } else {
            // Regular text token
            let idx = id as usize;
            if idx < vocab.len() {
                let text = &vocab[idx].text;
                if !text.starts_with("<|") {
                    current_text.push_str(text);
                }
            }
        }
    }

    // If no timestamps found at all, return the full decoded text as one segment
    if !found_any_timestamp {
        let full_text = decode(token_ids, vocab);
        let trimmed = full_text.trim().to_string();
        if !trimmed.is_empty() {
            return vec![(0.0, 0.0, trimmed)];
        }
        return Vec::new();
    }

    // If there is leftover text after the last timestamp, include it
    if let Some(start) = current_start {
        let trimmed = current_text.trim().to_string();
        if !trimmed.is_empty() {
            segments.push((start, 0.0, trimmed));
        }
    }

    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(text: &str) -> VocabEntry {
        VocabEntry {
            text: text.to_string(),
        }
    }

    #[test]
    fn test_decode_cjk_passthrough() {
        // CJK text should pass through byte-exact
        let vocab = vec![
            make_entry("日本語"),
            make_entry("テスト"),
            make_entry("한국어"),
            make_entry("中文"),
        ];
        let result = decode(&[0, 1, 2, 3], &vocab);
        assert_eq!(result, "日本語テスト한국어中文");
        assert_eq!(result.len(), 33, "UTF-8 byte count");
    }

    #[test]
    fn test_decode_emoji_with_zwj_sequences() {
        // ZWJ joiners must be preserved byte-exact
        let family = "👨\u{200D}👩\u{200D}👧"; // family emoji via ZWJ
        let vocab = vec![make_entry(family)];
        let result = decode(&[0], &vocab);
        assert_eq!(result, family);
    }

    #[test]
    fn test_decode_zero_width_characters() {
        // BOM (U+FEFF) must be preserved, not stripped
        let with_bom = "\u{FEFF}Hello";
        let vocab = vec![make_entry(with_bom)];
        let result = decode(&[0], &vocab);
        assert!(result.starts_with('\u{FEFF}'), "BOM must be preserved");
    }

    #[test]
    fn test_decode_whisper_prefix_space_handling() {
        // Whisper sentencepiece-style leading spaces are preserved
        let vocab = vec![make_entry(" Hello"), make_entry(" world")];
        let result = decode(&[0, 1], &vocab);
        assert_eq!(result, " Hello world");
    }

    #[test]
    fn test_decode_special_tokens_round_trip() {
        // Special tokens beginning with "<|" are elided; regular tokens pass through
        let vocab = vec![
            make_entry("<|startoftranscript|>"),
            make_entry("<|en|>"),
            make_entry("<|transcribe|>"),
            make_entry("<|notimestamps|>"),
            make_entry("Hello"),
        ];
        let result = decode(&[0, 1, 2, 3, 4], &vocab);
        assert_eq!(result, "Hello", "all special tokens must be elided");
    }

    #[test]
    fn test_decode_out_of_vocab_id_no_panic() {
        // Out-of-range IDs must be silently dropped, not panic
        let vocab = vec![make_entry("a"), make_entry("b"), make_entry("c")];
        let result = decode(&[0, 99999, 1, u32::MAX, 2], &vocab);
        assert_eq!(result, "abc", "only valid IDs should appear");
    }

    #[test]
    fn test_decode_mixed_ascii_cjk_emoji_punctuation() {
        // Combined stress test
        let vocab = vec![
            make_entry(" Hello"),
            make_entry(" 世界"),
            make_entry("!"),
            make_entry(" 🎉"),
            make_entry(" test."),
        ];
        let result = decode(&[0, 1, 2, 3, 4], &vocab);
        assert_eq!(result, " Hello 世界! 🎉 test.");
    }

    #[test]
    fn test_decode_space() {
        // GGML vocab stores actual decoded text; space token has text " "
        let vocab = vec![VocabEntry {
            text: " Hello".to_string(),
        }];
        assert_eq!(decode(&[0], &vocab), " Hello");
    }

    #[test]
    fn test_decode_japanese() {
        let vocab = vec![VocabEntry {
            text: "はい".to_string(),
        }];
        assert_eq!(decode(&[0], &vocab), "はい");
    }

    #[test]
    fn test_decode_skips_special() {
        let vocab = vec![
            VocabEntry {
                text: "<|startoftranscript|>".to_string(),
            },
            VocabEntry {
                text: "Hello".to_string(),
            },
        ];
        assert_eq!(decode(&[0, 1], &vocab), "Hello");
    }

    #[test]
    fn test_decode_out_of_range() {
        let vocab = vec![VocabEntry {
            text: "hi".to_string(),
        }];
        assert_eq!(decode(&[0, 9999], &vocab), "hi");
    }

    #[test]
    fn test_is_timestamp() {
        assert!(!SpecialTokens::is_timestamp(0));
        assert!(!SpecialTokens::is_timestamp(50256)); // EOT
        assert!(!SpecialTokens::is_timestamp(50363)); // no_timestamps
        assert!(SpecialTokens::is_timestamp(50364)); // TIMESTAMP_BEGIN (0.00s)
        assert!(SpecialTokens::is_timestamp(50365)); // 0.02s
        assert!(SpecialTokens::is_timestamp(51864)); // 30.00s
        assert!(SpecialTokens::is_timestamp(u32::MAX));
    }

    #[test]
    fn test_timestamp_seconds() {
        // Token 50364 = 0.00s
        let secs = SpecialTokens::timestamp_seconds(50364);
        assert!((secs - 0.0).abs() < 1e-6);

        // Token 50365 = 0.02s
        let secs = SpecialTokens::timestamp_seconds(50365);
        assert!((secs - 0.02).abs() < 1e-6);

        // Token 50414 = 1.00s  (50364 + 50 = 50414, 50 * 0.02 = 1.0)
        let secs = SpecialTokens::timestamp_seconds(50414);
        assert!((secs - 1.0).abs() < 1e-6);

        // Token below TIMESTAMP_BEGIN returns 0.0
        let secs = SpecialTokens::timestamp_seconds(100);
        assert!((secs - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_segments_with_timestamps() {
        // Build a minimal vocab: 0="Hello", 1=" world"
        let vocab = vec![
            VocabEntry {
                text: "Hello".to_string(),
            },
            VocabEntry {
                text: " world".to_string(),
            },
        ];
        // Timestamp tokens: 50364 = 0.00s, 50464 = 2.00s (50364+100, 100*0.02=2.0)
        let ts_start: u32 = 50364; // 0.00s
        let ts_end: u32 = 50464; // 2.00s
        let tokens = vec![ts_start, 0, 1, ts_end];

        let segments = parse_segments(&tokens, &vocab);
        assert_eq!(segments.len(), 1);
        assert!((segments[0].0 - 0.0).abs() < 1e-6); // start
        assert!((segments[0].1 - 2.0).abs() < 1e-6); // end
        assert_eq!(segments[0].2, "Hello world");
    }

    #[test]
    fn test_parse_segments_multiple() {
        let vocab = vec![
            VocabEntry {
                text: "Hello".to_string(),
            }, // 0
            VocabEntry {
                text: " there".to_string(),
            }, // 1
        ];
        // Two segments: <|0.00|> Hello <|1.00|> there <|2.00|>
        let ts0: u32 = 50364; // 0.00s
        let ts1: u32 = 50414; // 1.00s
        let ts2: u32 = 50464; // 2.00s
        let tokens = vec![ts0, 0, ts1, 1, ts2];

        let segments = parse_segments(&tokens, &vocab);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].2, "Hello");
        assert!((segments[0].0 - 0.0).abs() < 1e-6);
        assert!((segments[0].1 - 1.0).abs() < 1e-6);
        assert_eq!(segments[1].2, "there");
        assert!((segments[1].0 - 1.0).abs() < 1e-6);
        assert!((segments[1].1 - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_segments_no_timestamps() {
        let vocab = vec![
            VocabEntry {
                text: "Hello".to_string(),
            },
            VocabEntry {
                text: " world".to_string(),
            },
        ];
        let tokens = vec![0, 1];
        let segments = parse_segments(&tokens, &vocab);
        assert_eq!(segments.len(), 1);
        assert!((segments[0].0 - 0.0).abs() < 1e-6);
        assert!((segments[0].1 - 0.0).abs() < 1e-6);
        assert_eq!(segments[0].2, "Hello world");
    }

    #[test]
    fn test_parse_segments_empty() {
        let vocab: Vec<VocabEntry> = Vec::new();
        let tokens: Vec<u32> = Vec::new();
        let segments = parse_segments(&tokens, &vocab);
        assert!(segments.is_empty());
    }
}
