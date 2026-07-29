//! Pure vocab pass-through decoding for Whisper's BPE token vocabulary.
//!
//! This module performs **pure vocab pass-through decoding** — there is no
//! `encode()` function and no merge table. oxiwhisper is inference-only.
//!
//! # Byte-level decoding
//!
//! Whisper uses GPT-2 byte-level BPE: a single multi-byte character is often
//! spelled by several tokens, none of which is valid UTF-8 on its own (the
//! kanji `渋` = `U+6E0B` = `E6 B8 8B` is tokens `162`, `116`, `233`). Every
//! function here therefore concatenates the **raw bytes** of the selected
//! entries and performs a single lossy UTF-8 conversion over the joined
//! buffer. Converting per entry would replace 1476 of the 50257 entries in the
//! standard multilingual vocabulary with `U+FFFD`.
//!
//! # Elision rules
//! - **Special tokens** whose bytes begin with `"<|"` are silently dropped.
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

/// Append the raw bytes of every non-special, in-range token to `out`.
///
/// This is the shared primitive behind [`decode`] and [`parse_segments`]: it
/// never converts to UTF-8, so multi-token characters survive intact.
pub fn decode_bytes_into(token_ids: &[u32], vocab: &[VocabEntry], out: &mut Vec<u8>) {
    for &id in token_ids {
        let Some(entry) = vocab.get(id as usize) else {
            continue;
        };
        if entry.is_special() {
            continue;
        }
        out.extend_from_slice(entry.as_bytes());
    }
}

/// Decode Whisper token IDs to their raw byte sequence.
///
/// The result is the byte-exact concatenation of every non-special token,
/// before any UTF-8 interpretation. Useful when a caller needs to splice
/// decoder outputs together (streaming, segment stitching) without risking a
/// mid-character UTF-8 conversion.
pub fn decode_bytes(token_ids: &[u32], vocab: &[VocabEntry]) -> Vec<u8> {
    let mut out = Vec::with_capacity(token_ids.len() * 3);
    decode_bytes_into(token_ids, vocab, &mut out);
    out
}

/// Decode Whisper token IDs to a UTF-8 string.
///
/// The bytes of all non-special tokens are concatenated first and converted
/// **once** with [`String::from_utf8_lossy`]. Doing the conversion per token
/// would corrupt every character whose UTF-8 encoding spans more than one BPE
/// token (all kanji, kana, hangul, emoji, …).
pub fn decode(token_ids: &[u32], vocab: &[VocabEntry]) -> String {
    let bytes = decode_bytes(token_ids, vocab);
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Convert a raw byte buffer to a trimmed `String` with a single lossy UTF-8
/// conversion.
fn finish_text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).trim().to_string()
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
    // Text is accumulated as raw bytes and converted once per segment so that
    // characters spanning several BPE tokens survive.
    let mut current_bytes: Vec<u8> = Vec::new();
    let mut found_any_timestamp = false;

    for &id in token_ids {
        if SpecialTokens::is_timestamp(id) {
            found_any_timestamp = true;
            let time = SpecialTokens::timestamp_seconds(id);

            match current_start {
                None => {
                    // Opening timestamp -- start a new segment
                    current_start = Some(time);
                    current_bytes.clear();
                }
                Some(start) => {
                    // Closing timestamp -- finalize segment
                    let text = finish_text(&current_bytes);
                    if !text.is_empty() {
                        segments.push((start, time, text));
                    }
                    // This closing timestamp may also be the opening of the next segment
                    current_start = Some(time);
                    current_bytes.clear();
                }
            }
        } else {
            // Regular text token
            if let Some(entry) = vocab.get(id as usize)
                && !entry.is_special()
            {
                current_bytes.extend_from_slice(entry.as_bytes());
            }
        }
    }

    // If no timestamps found at all, return the full decoded text as one segment
    if !found_any_timestamp {
        let full = decode_bytes(token_ids, vocab);
        let text = finish_text(&full);
        if !text.is_empty() {
            return vec![(0.0, 0.0, text)];
        }
        return Vec::new();
    }

    // If there is leftover text after the last timestamp, include it
    if let Some(start) = current_start {
        let text = finish_text(&current_bytes);
        if !text.is_empty() {
            segments.push((start, 0.0, text));
        }
    }

    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(text: &str) -> VocabEntry {
        VocabEntry::from_text(text)
    }

    /// Build a vocabulary in which the token ids used by the real GGML
    /// multilingual vocabulary for single raw bytes carry exactly that byte.
    ///
    /// Verified against `ggml-tiny.bin`: ids 162/116/233 hold `0xE6`, `0xB8`
    /// and `0x8B`, the three bytes of `渋` (U+6E0B).
    fn byte_fallback_vocab() -> Vec<VocabEntry> {
        let mut vocab = vec![VocabEntry::from_bytes(Vec::new()); 256];
        vocab[162] = VocabEntry::from_bytes(vec![0xE6]);
        vocab[116] = VocabEntry::from_bytes(vec![0xB8]);
        vocab[233] = VocabEntry::from_bytes(vec![0x8B]);
        vocab
    }

    #[test]
    fn test_decode_multi_token_kanji_from_raw_bytes() {
        // Regression: the vocabulary used to be converted with
        // `String::from_utf8_lossy` per entry at load time, which turned each
        // of these three byte fragments into U+FFFD and made every kanji in
        // the transcript unrecoverable.
        let vocab = byte_fallback_vocab();
        let decoded = decode(&[162, 116, 233], &vocab);
        assert_eq!(decoded, "\u{6E0B}", "BPE bytes E6 B8 8B must decode to 渋");
        assert_eq!(decoded.chars().count(), 1);
        assert!(
            !decoded.contains('\u{FFFD}'),
            "no replacement characters may appear, got {decoded:?}"
        );
    }

    #[test]
    fn test_decode_bytes_is_byte_exact() {
        let vocab = byte_fallback_vocab();
        assert_eq!(
            decode_bytes(&[162, 116, 233], &vocab),
            vec![0xE6, 0xB8, 0x8B]
        );
    }

    #[test]
    fn test_parse_segments_multi_token_kanji() {
        // <|0.00|> 渋 <|1.00|> — the segment text must be the joined kanji.
        let vocab = byte_fallback_vocab();
        let tokens = vec![50364u32, 162, 116, 233, 50414];
        let segments = parse_segments(&tokens, &vocab);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].2, "\u{6E0B}");
    }

    #[test]
    fn test_decode_japanese_sentence_split_across_byte_tokens() {
        // こんにちは spelled entirely as single-byte fallback tokens.
        let text = "こんにちは";
        let mut vocab = Vec::new();
        let mut ids = Vec::new();
        for (i, b) in text.as_bytes().iter().enumerate() {
            vocab.push(VocabEntry::from_bytes(vec![*b]));
            ids.push(i as u32);
        }
        assert_eq!(decode(&ids, &vocab), text);
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
        let vocab = vec![VocabEntry::from_text(" Hello")];
        assert_eq!(decode(&[0], &vocab), " Hello");
    }

    #[test]
    fn test_decode_japanese() {
        let vocab = vec![VocabEntry::from_text("はい")];
        assert_eq!(decode(&[0], &vocab), "はい");
    }

    #[test]
    fn test_decode_skips_special() {
        let vocab = vec![
            VocabEntry::from_text("<|startoftranscript|>"),
            VocabEntry::from_text("Hello"),
        ];
        assert_eq!(decode(&[0, 1], &vocab), "Hello");
    }

    #[test]
    fn test_decode_out_of_range() {
        let vocab = vec![VocabEntry::from_text("hi")];
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
            VocabEntry::from_text("Hello"),
            VocabEntry::from_text(" world"),
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
            VocabEntry::from_text("Hello"),  // 0
            VocabEntry::from_text(" there"), // 1
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
            VocabEntry::from_text("Hello"),
            VocabEntry::from_text(" world"),
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
