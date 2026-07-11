//! Pure Rust WAV file loader for oxiwhisper.
//!
//! Parses RIFF/WAVE format headers, supports multiple PCM formats,
//! handles multi-channel audio (downmix to mono), and resamples to 16kHz.

use crate::OxiWhisperError;
use std::path::Path;

/// Target sample rate for Whisper inference.
const TARGET_SAMPLE_RATE: u32 = 16000;

/// Audio format tag for integer PCM.
const WAVE_FORMAT_PCM: u16 = 1;

/// Audio format tag for IEEE 754 floating-point PCM.
const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;

/// Parsed fmt chunk fields.
struct FmtChunk {
    audio_format: u16,
    num_channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
}

// ---------------------------------------------------------------------------
// Little-endian readers
// ---------------------------------------------------------------------------

fn read_u16_le(data: &[u8], offset: usize) -> Result<u16, OxiWhisperError> {
    if offset + 2 > data.len() {
        return Err(OxiWhisperError::InvalidModel(
            "Unexpected end of WAV data reading u16".into(),
        ));
    }
    Ok(u16::from_le_bytes([data[offset], data[offset + 1]]))
}

fn read_u32_le(data: &[u8], offset: usize) -> Result<u32, OxiWhisperError> {
    if offset + 4 > data.len() {
        return Err(OxiWhisperError::InvalidModel(
            "Unexpected end of WAV data reading u32".into(),
        ));
    }
    Ok(u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

fn read_f32_le(data: &[u8], offset: usize) -> Result<f32, OxiWhisperError> {
    if offset + 4 > data.len() {
        return Err(OxiWhisperError::InvalidModel(
            "Unexpected end of WAV data reading f32".into(),
        ));
    }
    Ok(f32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

fn read_i24_le(data: &[u8], offset: usize) -> Result<i32, OxiWhisperError> {
    if offset + 3 > data.len() {
        return Err(OxiWhisperError::InvalidModel(
            "Unexpected end of WAV data reading i24".into(),
        ));
    }
    // Sign-extend 24-bit to 32-bit
    let low = data[offset] as i32;
    let mid = data[offset + 1] as i32;
    let high = data[offset + 2] as i32;
    let val = low | (mid << 8) | (high << 16);
    // If the sign bit (bit 23) is set, extend it
    if val & 0x0080_0000 != 0 {
        Ok(val | (0xFF << 24))
    } else {
        Ok(val)
    }
}

fn read_i32_le(data: &[u8], offset: usize) -> Result<i32, OxiWhisperError> {
    if offset + 4 > data.len() {
        return Err(OxiWhisperError::InvalidModel(
            "Unexpected end of WAV data reading i32".into(),
        ));
    }
    Ok(i32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

// ---------------------------------------------------------------------------
// Four-byte tag helpers
// ---------------------------------------------------------------------------

fn tag_matches(data: &[u8], offset: usize, expected: &[u8; 4]) -> bool {
    if offset + 4 > data.len() {
        return false;
    }
    &data[offset..offset + 4] == expected
}

// ---------------------------------------------------------------------------
// Chunk parsing
// ---------------------------------------------------------------------------

fn parse_fmt_chunk(
    data: &[u8],
    offset: usize,
    chunk_size: u32,
) -> Result<FmtChunk, OxiWhisperError> {
    if (chunk_size as usize) < 16 {
        return Err(OxiWhisperError::InvalidModel("fmt chunk too small".into()));
    }
    let audio_format = read_u16_le(data, offset)?;
    let num_channels = read_u16_le(data, offset + 2)?;
    let sample_rate = read_u32_le(data, offset + 4)?;
    // byte_rate at offset+8 (skip)
    // block_align at offset+12 (skip)
    let bits_per_sample = read_u16_le(data, offset + 14)?;

    if num_channels == 0 {
        return Err(OxiWhisperError::InvalidModel(
            "WAV has zero channels".into(),
        ));
    }
    if sample_rate == 0 {
        return Err(OxiWhisperError::InvalidModel(
            "WAV has zero sample rate".into(),
        ));
    }

    match audio_format {
        WAVE_FORMAT_PCM => {
            if !matches!(bits_per_sample, 8 | 16 | 24 | 32) {
                return Err(OxiWhisperError::InvalidModel(format!(
                    "Unsupported PCM bit depth: {bits_per_sample}"
                )));
            }
        }
        WAVE_FORMAT_IEEE_FLOAT => {
            if bits_per_sample != 32 {
                return Err(OxiWhisperError::InvalidModel(format!(
                    "Unsupported IEEE float bit depth: {bits_per_sample}"
                )));
            }
        }
        _ => {
            return Err(OxiWhisperError::InvalidModel(format!(
                "Unsupported audio format tag: {audio_format}"
            )));
        }
    }

    Ok(FmtChunk {
        audio_format,
        num_channels,
        sample_rate,
        bits_per_sample,
    })
}

/// Decode raw sample bytes into f32 values normalised to [-1.0, 1.0].
fn decode_samples(
    data: &[u8],
    offset: usize,
    data_size: usize,
    fmt: &FmtChunk,
) -> Result<Vec<f32>, OxiWhisperError> {
    let bytes_per_sample = (fmt.bits_per_sample as usize).div_ceil(8);
    let frame_size = bytes_per_sample * fmt.num_channels as usize;

    if frame_size == 0 {
        return Err(OxiWhisperError::InvalidModel("Frame size is zero".into()));
    }

    let num_frames = data_size / frame_size;
    let total_samples = num_frames * fmt.num_channels as usize;
    let mut samples = Vec::with_capacity(total_samples);

    let end = offset + num_frames * frame_size;
    if end > data.len() {
        return Err(OxiWhisperError::InvalidModel(
            "Data chunk extends past end of file".into(),
        ));
    }

    let mut pos = offset;
    for _ in 0..total_samples {
        let sample = match (fmt.audio_format, fmt.bits_per_sample) {
            (WAVE_FORMAT_PCM, 8) => {
                // 8-bit PCM is unsigned, centre at 128
                if pos >= data.len() {
                    return Err(OxiWhisperError::InvalidModel(
                        "Unexpected end of sample data".into(),
                    ));
                }
                let val = data[pos] as f32;
                pos += 1;
                (val - 128.0) / 128.0
            }
            (WAVE_FORMAT_PCM, 16) => {
                let val = read_u16_le(data, pos)? as i16;
                pos += 2;
                val as f32 / 32768.0
            }
            (WAVE_FORMAT_PCM, 24) => {
                let val = read_i24_le(data, pos)?;
                pos += 3;
                val as f32 / 8_388_608.0
            }
            (WAVE_FORMAT_PCM, 32) => {
                let val = read_i32_le(data, pos)?;
                pos += 4;
                val as f32 / 2_147_483_648.0
            }
            (WAVE_FORMAT_IEEE_FLOAT, 32) => {
                let val = read_f32_le(data, pos)?;
                pos += 4;
                val
            }
            _ => {
                return Err(OxiWhisperError::InvalidModel(format!(
                    "Unsupported format/depth combination: fmt={}, bits={}",
                    fmt.audio_format, fmt.bits_per_sample
                )));
            }
        };
        samples.push(sample);
    }

    Ok(samples)
}

/// Downmix interleaved multi-channel samples to mono by averaging channels.
pub(crate) fn downmix_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let ch = channels as usize;
    let num_frames = samples.len() / ch;
    let mut mono = Vec::with_capacity(num_frames);
    let inv_ch = 1.0 / channels as f32;
    for frame in 0..num_frames {
        let base = frame * ch;
        let mut sum = 0.0_f32;
        for c in 0..ch {
            sum += samples[base + c];
        }
        mono.push(sum * inv_ch);
    }
    mono
}

/// Resample mono audio from `source_rate` to `TARGET_SAMPLE_RATE` using
/// linear interpolation.
pub(crate) fn resample_linear(samples: &[f32], source_rate: u32) -> Vec<f32> {
    if source_rate == TARGET_SAMPLE_RATE || samples.is_empty() {
        return samples.to_vec();
    }

    let ratio = source_rate as f64 / TARGET_SAMPLE_RATE as f64;
    let output_len = ((samples.len() as f64) / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);
    let last_idx = if samples.is_empty() {
        0
    } else {
        samples.len() - 1
    };

    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos as usize;
        let frac = (src_pos - idx as f64) as f32;

        if idx >= last_idx {
            output.push(samples[last_idx]);
        } else {
            let a = samples[idx];
            let b = samples[idx + 1];
            output.push(a * (1.0 - frac) + b * frac);
        }
    }

    output
}

/// Parse a RIFF/WAVE byte buffer and return normalised f32 mono samples at
/// 16 kHz.
fn parse_wav(data: &[u8]) -> Result<Vec<f32>, OxiWhisperError> {
    // Minimum RIFF header: 12 bytes
    if data.len() < 12 {
        return Err(OxiWhisperError::InvalidModel(
            "File too small to be a WAV".into(),
        ));
    }

    if !tag_matches(data, 0, b"RIFF") {
        return Err(OxiWhisperError::InvalidModel("Missing RIFF header".into()));
    }
    if !tag_matches(data, 8, b"WAVE") {
        return Err(OxiWhisperError::InvalidModel("Missing WAVE tag".into()));
    }

    let mut fmt: Option<FmtChunk> = None;
    let mut pos = 12_usize; // past RIFF header

    loop {
        if pos + 8 > data.len() {
            break;
        }

        let chunk_size = read_u32_le(data, pos + 4)? as usize;
        let chunk_data_start = pos + 8;

        if tag_matches(data, pos, b"fmt ") {
            fmt = Some(parse_fmt_chunk(data, chunk_data_start, chunk_size as u32)?);
        } else if tag_matches(data, pos, b"data") {
            let fmt = fmt.as_ref().ok_or_else(|| {
                OxiWhisperError::InvalidModel("data chunk before fmt chunk".into())
            })?;

            let available = data.len().saturating_sub(chunk_data_start);
            let actual_data_size = chunk_size.min(available);

            let interleaved = decode_samples(data, chunk_data_start, actual_data_size, fmt)?;
            let mono = downmix_to_mono(&interleaved, fmt.num_channels);
            let resampled = resample_linear(&mono, fmt.sample_rate);
            return Ok(resampled);
        }

        // Advance past this chunk. Chunks are word-aligned (pad byte if odd size).
        let padded = (chunk_size + 1) & !1;
        pos = chunk_data_start + padded;
    }

    Err(OxiWhisperError::InvalidModel(
        "No data chunk found in WAV".into(),
    ))
}

// ===========================================================================
// Public API
// ===========================================================================

/// Load a WAV file from disk and return 16 kHz mono f32 PCM audio normalised
/// to [-1.0, 1.0].
pub fn load_wav(path: &Path) -> Result<Vec<f32>, OxiWhisperError> {
    let data = std::fs::read(path)?;
    parse_wav(&data)
}

/// Load WAV audio from an in-memory byte slice and return 16 kHz mono f32
/// PCM audio normalised to [-1.0, 1.0].
pub fn load_wav_from_bytes(data: &[u8]) -> Result<Vec<f32>, OxiWhisperError> {
    parse_wav(data)
}

/// Convert raw PCM f32 samples (possibly multi-channel and/or a different
/// sample rate) into 16 kHz mono suitable for Whisper inference.
pub fn load_raw_pcm(samples: &[f32], sample_rate: u32, channels: u16) -> Vec<f32> {
    let mono = downmix_to_mono(samples, channels);
    resample_linear(&mono, sample_rate)
}

/// Auto-detect audio format by magic bytes and decode to 16 kHz mono f32 PCM.
///
/// Supported formats depend on enabled feature flags:
/// - WAV: always supported
/// - FLAC: requires `audio-flac` feature
/// - OGG Vorbis: requires `audio-ogg` feature
/// - MP3: requires `audio-mp3` feature
/// - AAC/M4A: requires `audio-aac` feature
/// - Ogg Opus: requires `audio-opus` feature
pub fn load_audio(path: &std::path::Path) -> Result<Vec<f32>, OxiWhisperError> {
    use std::io::Read;

    let mut magic = [0u8; 16];
    let n = {
        let mut f = std::fs::File::open(path).map_err(OxiWhisperError::Io)?;
        f.read(&mut magic).map_err(OxiWhisperError::Io)?
    };
    if n < 4 {
        return Err(OxiWhisperError::AudioFormatError(
            "file too short to determine format".to_string(),
        ));
    }

    // WAV: RIFF....WAVE
    if magic.starts_with(b"RIFF") {
        return load_wav(path);
    }

    // FLAC: fLaC
    #[cfg(feature = "audio-flac")]
    if magic.starts_with(b"fLaC") {
        return load_flac(path);
    }

    // OGG container: OggS — check if it's Opus or Vorbis
    if magic.starts_with(b"OggS") {
        let bytes = std::fs::read(path).map_err(OxiWhisperError::Io)?;
        let scan_end = bytes.len().min(100);

        if bytes[..scan_end].windows(8).any(|w| w == b"OpusHead") {
            #[cfg(feature = "audio-opus")]
            return load_ogg_opus(&bytes);
            #[cfg(not(feature = "audio-opus"))]
            return Err(OxiWhisperError::AudioFormatError(
                "Ogg Opus detected but `audio-opus` feature not enabled".to_string(),
            ));
        }

        #[cfg(feature = "audio-ogg")]
        return load_ogg_vorbis(path);
        #[cfg(not(feature = "audio-ogg"))]
        return Err(OxiWhisperError::AudioFormatError(
            "OGG Vorbis detected but `audio-ogg` feature not enabled".to_string(),
        ));
    }

    // MP3: ID3 tag or sync frame 0xFF 0xFB / 0xFF 0xFA / 0xFF 0xF3
    if magic.starts_with(b"ID3") || (magic[0] == 0xFF && (magic[1] & 0xE0 == 0xE0)) {
        #[cfg(feature = "audio-mp3")]
        return load_mp3(path);
        #[cfg(not(feature = "audio-mp3"))]
        return Err(OxiWhisperError::AudioFormatError(
            "MP3 detected but `audio-mp3` feature not enabled".to_string(),
        ));
    }

    // AAC/M4A: ftyp box at bytes 4..8
    if n >= 8 && &magic[4..8] == b"ftyp" {
        #[cfg(feature = "audio-aac")]
        return load_aac(path);
        #[cfg(not(feature = "audio-aac"))]
        return Err(OxiWhisperError::AudioFormatError(
            "AAC/M4A detected but `audio-aac` feature not enabled".to_string(),
        ));
    }

    Err(OxiWhisperError::AudioFormatError(format!(
        "unrecognized audio format (magic: {:02x?})",
        &magic[..4]
    )))
}

// ===========================================================================
// Symphonia backend (FLAC, OGG Vorbis, MP3, AAC)
// ===========================================================================

#[cfg(any(
    feature = "audio-flac",
    feature = "audio-ogg",
    feature = "audio-mp3",
    feature = "audio-aac"
))]
fn decode_with_symphonia(path: &std::path::Path) -> Result<Vec<f32>, OxiWhisperError> {
    use symphonia::core::codecs::CodecParameters;
    use symphonia::core::codecs::audio::AudioDecoderOptions;
    use symphonia::core::errors::Error as SymphoniaError;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::formats::{FormatOptions, TrackType};
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;

    let src = std::fs::File::open(path).map_err(OxiWhisperError::Io)?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());
    let hint = Hint::new();
    let dec_opts = AudioDecoderOptions::default();

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| OxiWhisperError::AudioFormatError(format!("symphonia probe: {e}")))?;

    // Extract all track data in a block so the &Track borrow on `format` ends
    // before the decode loop mutably borrows it via next_packet().
    let (track_id, sample_rate, channels, audio_params) = {
        let track = format
            .first_track(TrackType::Audio)
            .ok_or_else(|| OxiWhisperError::AudioFormatError("no audio track found".to_string()))?;

        let audio_params = match &track.codec_params {
            Some(CodecParameters::Audio(p)) => p.clone(),
            _ => {
                return Err(OxiWhisperError::AudioFormatError(
                    "track has no audio codec parameters".to_string(),
                ));
            }
        };

        let sample_rate = audio_params
            .sample_rate
            .ok_or_else(|| OxiWhisperError::AudioFormatError("unknown sample rate".to_string()))?;
        let channels = audio_params
            .channels
            .as_ref()
            .map(|c| c.count())
            .unwrap_or(1);

        (track.id, sample_rate, channels, audio_params)
    };

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&audio_params, &dec_opts)
        .map_err(|e| OxiWhisperError::AudioFormatError(format!("symphonia decoder: {e}")))?;

    let mut all_samples: Vec<f32> = Vec::new();
    let mut chunk: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(SymphoniaError::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => continue,
            Err(e) => {
                return Err(OxiWhisperError::AudioFormatError(format!(
                    "symphonia packet: {e}"
                )));
            }
        };

        if packet.track_id != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(SymphoniaError::IoError(_)) => break,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => {
                return Err(OxiWhisperError::AudioFormatError(format!(
                    "symphonia decode: {e}"
                )));
            }
        };

        // copy_to_vec_interleaved resizes `chunk` then fills it; extend collects all packets.
        decoded.copy_to_vec_interleaved::<f32>(&mut chunk);
        all_samples.extend_from_slice(&chunk);
    }

    let mono = downmix_to_mono(&all_samples, channels as u16);
    let resampled = resample_linear(&mono, sample_rate);
    Ok(resampled)
}

#[cfg(feature = "audio-flac")]
fn load_flac(path: &std::path::Path) -> Result<Vec<f32>, OxiWhisperError> {
    decode_with_symphonia(path)
}

#[cfg(feature = "audio-ogg")]
fn load_ogg_vorbis(path: &std::path::Path) -> Result<Vec<f32>, OxiWhisperError> {
    decode_with_symphonia(path)
}

#[cfg(feature = "audio-mp3")]
fn load_mp3(path: &std::path::Path) -> Result<Vec<f32>, OxiWhisperError> {
    decode_with_symphonia(path)
}

#[cfg(feature = "audio-aac")]
fn load_aac(path: &std::path::Path) -> Result<Vec<f32>, OxiWhisperError> {
    decode_with_symphonia(path)
}

// ===========================================================================
// Ogg Opus backend
// ===========================================================================

#[cfg(feature = "audio-opus")]
fn load_ogg_opus(bytes: &[u8]) -> Result<Vec<f32>, OxiWhisperError> {
    use std::io::Cursor;

    let mut reader = ogg::PacketReader::new(Cursor::new(bytes));

    // First packet: OpusHead
    let head_packet = reader
        .read_packet_expected()
        .map_err(|e| OxiWhisperError::AudioFormatError(format!("ogg read: {e}")))?;

    if !head_packet.data.starts_with(b"OpusHead") {
        return Err(OxiWhisperError::AudioFormatError(
            "not an Ogg Opus file".to_string(),
        ));
    }

    // RFC 7845 §5.1: byte 9 = channel count
    let channels = if head_packet.data.len() > 9 {
        head_packet.data[9] as usize
    } else {
        1
    };

    if channels == 0 || channels > 2 {
        return Err(OxiWhisperError::AudioFormatError(format!(
            "unsupported Opus channel count: {channels} (only 1-2 supported)"
        )));
    }

    // Skip OpusTags packet
    let _ = reader
        .read_packet_expected()
        .map_err(|e| OxiWhisperError::AudioFormatError(format!("ogg tags: {e}")))?;

    let mut decoder = opus_decoder::OpusDecoder::new(48000, channels)
        .map_err(|e| OxiWhisperError::AudioFormatError(format!("opus decoder init: {e}")))?;

    let max_frame_samples = opus_decoder::OpusDecoder::MAX_FRAME_SIZE_48K;
    let mut all_samples: Vec<f32> = Vec::new();
    let mut pcm_buf = vec![0.0f32; max_frame_samples * channels];

    while let Ok(Some(packet)) = reader.read_packet() {
        let n_samples = decoder
            .decode_float(&packet.data, &mut pcm_buf, false)
            .map_err(|e| OxiWhisperError::AudioFormatError(format!("opus decode: {e}")))?;
        all_samples.extend_from_slice(&pcm_buf[..n_samples * channels]);
    }

    // Opus always outputs at 48 kHz; downmix then resample to 16 kHz
    let mono = downmix_to_mono(&all_samples, channels as u16);
    let resampled = resample_linear(&mono, 48000);
    Ok(resampled)
}

// ===========================================================================
// Test helpers – construct WAV byte arrays programmatically
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal WAV file in memory.
    fn build_wav(
        sample_rate: u32,
        num_channels: u16,
        bits_per_sample: u16,
        audio_format: u16,
        raw_samples: &[u8],
    ) -> Vec<u8> {
        let fmt_chunk_size: u32 = 16;
        let data_chunk_size = raw_samples.len() as u32;
        // RIFF size = 4 (WAVE) + 8+fmt_size + 8+data_size
        let riff_size: u32 = 4 + 8 + fmt_chunk_size + 8 + data_chunk_size;

        let byte_rate = sample_rate * num_channels as u32 * (bits_per_sample as u32 / 8);
        let block_align = num_channels * (bits_per_sample / 8);

        let mut buf = Vec::new();
        // RIFF header
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&riff_size.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        // fmt chunk
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&fmt_chunk_size.to_le_bytes());
        buf.extend_from_slice(&audio_format.to_le_bytes());
        buf.extend_from_slice(&num_channels.to_le_bytes());
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&byte_rate.to_le_bytes());
        buf.extend_from_slice(&block_align.to_le_bytes());
        buf.extend_from_slice(&bits_per_sample.to_le_bytes());
        // data chunk
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_chunk_size.to_le_bytes());
        buf.extend_from_slice(raw_samples);
        buf
    }

    fn samples_to_i16_bytes(samples: &[i16]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(samples.len() * 2);
        for &s in samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        bytes
    }

    fn samples_to_f32_bytes(samples: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(samples.len() * 4);
        for &s in samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        bytes
    }

    // -----------------------------------------------------------------------
    // 16-bit PCM WAV
    // -----------------------------------------------------------------------
    #[test]
    fn test_parse_16bit_pcm() {
        let raw: Vec<i16> = vec![0, 16384, 32767, -32768, -1];
        let bytes = samples_to_i16_bytes(&raw);
        let wav = build_wav(16000, 1, 16, WAVE_FORMAT_PCM, &bytes);

        let result = load_wav_from_bytes(&wav);
        assert!(
            result.is_ok(),
            "Failed to parse 16-bit WAV: {:?}",
            result.err()
        );
        let samples = result.expect("already checked");
        assert_eq!(samples.len(), 5);

        // 0 / 32768 = 0.0
        assert!((samples[0]).abs() < 1e-5);
        // 16384 / 32768 = 0.5
        assert!((samples[1] - 0.5).abs() < 1e-4);
        // 32767 / 32768 ≈ 1.0
        assert!((samples[2] - 32767.0 / 32768.0).abs() < 1e-5);
        // -32768 / 32768 = -1.0
        assert!((samples[3] - (-1.0)).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // 32-bit float WAV
    // -----------------------------------------------------------------------
    #[test]
    fn test_parse_32bit_float() {
        let raw: Vec<f32> = vec![0.0, 0.5, 1.0, -1.0, -0.25];
        let bytes = samples_to_f32_bytes(&raw);
        let wav = build_wav(16000, 1, 32, WAVE_FORMAT_IEEE_FLOAT, &bytes);

        let result = load_wav_from_bytes(&wav);
        assert!(result.is_ok());
        let samples = result.expect("already checked");
        assert_eq!(samples.len(), 5);
        assert!((samples[0]).abs() < 1e-7);
        assert!((samples[1] - 0.5).abs() < 1e-7);
        assert!((samples[2] - 1.0).abs() < 1e-7);
        assert!((samples[3] - (-1.0)).abs() < 1e-7);
        assert!((samples[4] - (-0.25)).abs() < 1e-7);
    }

    // -----------------------------------------------------------------------
    // 8-bit unsigned PCM WAV
    // -----------------------------------------------------------------------
    #[test]
    fn test_parse_8bit_pcm() {
        // 128 = silence, 0 = -1.0, 255 ≈ +1.0
        let raw: Vec<u8> = vec![128, 0, 255];
        let wav = build_wav(16000, 1, 8, WAVE_FORMAT_PCM, &raw);

        let result = load_wav_from_bytes(&wav);
        assert!(result.is_ok());
        let samples = result.expect("already checked");
        assert_eq!(samples.len(), 3);
        assert!(samples[0].abs() < 1e-5); // silence
        assert!((samples[1] - (-1.0)).abs() < 1e-5);
        assert!((samples[2] - (127.0 / 128.0)).abs() < 1e-4);
    }

    // -----------------------------------------------------------------------
    // 24-bit signed PCM WAV
    // -----------------------------------------------------------------------
    #[test]
    fn test_parse_24bit_pcm() {
        // Encode a few 24-bit samples manually
        let mut raw = Vec::new();
        // 0
        raw.extend_from_slice(&[0x00, 0x00, 0x00]);
        // max positive: 0x7FFFFF = 8388607
        raw.extend_from_slice(&[0xFF, 0xFF, 0x7F]);
        // max negative: 0x800000 = -8388608
        raw.extend_from_slice(&[0x00, 0x00, 0x80]);

        let wav = build_wav(16000, 1, 24, WAVE_FORMAT_PCM, &raw);
        let result = load_wav_from_bytes(&wav);
        assert!(result.is_ok());
        let samples = result.expect("already checked");
        assert_eq!(samples.len(), 3);
        assert!(samples[0].abs() < 1e-7);
        assert!((samples[1] - (8_388_607.0 / 8_388_608.0)).abs() < 1e-5);
        assert!((samples[2] - (-1.0)).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // 32-bit signed integer PCM WAV
    // -----------------------------------------------------------------------
    #[test]
    fn test_parse_32bit_int_pcm() {
        let mut raw = Vec::new();
        raw.extend_from_slice(&0_i32.to_le_bytes());
        raw.extend_from_slice(&i32::MAX.to_le_bytes());
        raw.extend_from_slice(&i32::MIN.to_le_bytes());

        let wav = build_wav(16000, 1, 32, WAVE_FORMAT_PCM, &raw);
        let result = load_wav_from_bytes(&wav);
        assert!(result.is_ok());
        let samples = result.expect("already checked");
        assert_eq!(samples.len(), 3);
        assert!(samples[0].abs() < 1e-7);
        // i32::MAX / 2^31 ≈ 1.0
        assert!((samples[1] - (i32::MAX as f32 / 2_147_483_648.0)).abs() < 1e-5);
        assert!((samples[2] - (-1.0)).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // Stereo downmix to mono
    // -----------------------------------------------------------------------
    #[test]
    fn test_stereo_downmix() {
        // Stereo 16-bit: L=32767, R=-32768 → average ≈ -0.5/32768
        let raw: Vec<i16> = vec![
            32767, -32768, // frame 0
            0, 0, // frame 1
            16384, 16384, // frame 2
        ];
        let bytes = samples_to_i16_bytes(&raw);
        let wav = build_wav(16000, 2, 16, WAVE_FORMAT_PCM, &bytes);

        let result = load_wav_from_bytes(&wav);
        assert!(result.is_ok());
        let samples = result.expect("already checked");
        assert_eq!(samples.len(), 3); // 3 frames, mono

        // Frame 0: (32767/32768 + (-32768/32768)) / 2 = (-1/32768)/2
        let expected_0 = ((32767.0_f32 / 32768.0) + (-1.0)) / 2.0;
        assert!((samples[0] - expected_0).abs() < 1e-4);

        // Frame 1: 0
        assert!(samples[1].abs() < 1e-5);

        // Frame 2: 0.5
        assert!((samples[2] - 0.5).abs() < 1e-4);
    }

    // -----------------------------------------------------------------------
    // Resampling 48kHz → 16kHz
    // -----------------------------------------------------------------------
    #[test]
    fn test_resample_48k_to_16k() {
        // Generate a simple 48kHz signal: 480 samples = 10 ms
        let source_rate = 48000_u32;
        let num_source = 480_usize;
        let mut raw_f32 = Vec::with_capacity(num_source);
        for i in 0..num_source {
            // A linear ramp from 0 to 1
            raw_f32.push(i as f32 / (num_source - 1) as f32);
        }
        let bytes = samples_to_f32_bytes(&raw_f32);
        let wav = build_wav(source_rate, 1, 32, WAVE_FORMAT_IEEE_FLOAT, &bytes);

        let result = load_wav_from_bytes(&wav);
        assert!(result.is_ok());
        let samples = result.expect("already checked");

        // 480 samples at 48kHz → 160 samples at 16kHz
        assert_eq!(samples.len(), 160);

        // First sample should be ~0.0, last should be ~1.0
        assert!(samples[0].abs() < 0.01);
        assert!((samples[samples.len() - 1] - 1.0).abs() < 0.05);

        // Check monotonicity (ramp should remain monotonically increasing)
        for i in 1..samples.len() {
            assert!(
                samples[i] >= samples[i - 1] - 1e-6,
                "Non-monotonic at index {i}: {} < {}",
                samples[i],
                samples[i - 1]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Already at 16kHz — no resampling
    // -----------------------------------------------------------------------
    #[test]
    fn test_no_resample_when_already_16k() {
        let raw: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let bytes = samples_to_f32_bytes(&raw);
        let wav = build_wav(16000, 1, 32, WAVE_FORMAT_IEEE_FLOAT, &bytes);

        let result = load_wav_from_bytes(&wav);
        assert!(result.is_ok());
        let samples = result.expect("already checked");
        assert_eq!(samples.len(), 4);
        for (a, b) in samples.iter().zip(raw.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    // -----------------------------------------------------------------------
    // Empty/corrupt WAV error handling
    // -----------------------------------------------------------------------
    #[test]
    fn test_empty_data_error() {
        let result = load_wav_from_bytes(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_riff_header() {
        let result = load_wav_from_bytes(b"NOT_A_WAV_FILE_AT_ALL");
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_wave_tag() {
        let mut data = Vec::new();
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&100_u32.to_le_bytes());
        data.extend_from_slice(b"XXXX"); // not WAVE
        let result = load_wav_from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_truncated_wav() {
        // Valid header but truncated before data
        let wav = build_wav(16000, 1, 16, WAVE_FORMAT_PCM, &[]);
        let result = load_wav_from_bytes(&wav);
        // Should succeed with 0 samples (data chunk size is 0)
        assert!(result.is_ok());
        let samples = result.expect("already checked");
        assert!(samples.is_empty());
    }

    #[test]
    fn test_no_data_chunk() {
        // Only a fmt chunk, no data chunk
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&20_u32.to_le_bytes()); // size
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16_u32.to_le_bytes());
        // Minimal valid fmt: PCM, 1ch, 16000Hz, 16bit
        buf.extend_from_slice(&1_u16.to_le_bytes()); // format
        buf.extend_from_slice(&1_u16.to_le_bytes()); // channels
        buf.extend_from_slice(&16000_u32.to_le_bytes()); // sample rate
        buf.extend_from_slice(&32000_u32.to_le_bytes()); // byte rate
        buf.extend_from_slice(&2_u16.to_le_bytes()); // block align
        buf.extend_from_slice(&16_u16.to_le_bytes()); // bits per sample

        let result = load_wav_from_bytes(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_data_before_fmt_error() {
        // data chunk appears before fmt chunk
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&40_u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        // data chunk first
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&4_u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 4]);

        let result = load_wav_from_bytes(&buf);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Unsupported format
    // -----------------------------------------------------------------------
    #[test]
    fn test_unsupported_audio_format() {
        // Build a WAV with format=2 (ADPCM) which we don't support
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&40_u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16_u32.to_le_bytes());
        buf.extend_from_slice(&2_u16.to_le_bytes()); // format=2 (ADPCM)
        buf.extend_from_slice(&1_u16.to_le_bytes());
        buf.extend_from_slice(&16000_u32.to_le_bytes());
        buf.extend_from_slice(&32000_u32.to_le_bytes());
        buf.extend_from_slice(&2_u16.to_le_bytes());
        buf.extend_from_slice(&16_u16.to_le_bytes());

        let result = load_wav_from_bytes(&buf);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Output normalisation bounds
    // -----------------------------------------------------------------------
    #[test]
    fn test_output_normalised_range() {
        // Full-range 16-bit samples
        let raw: Vec<i16> = vec![i16::MIN, i16::MAX, 0, 1000, -1000];
        let bytes = samples_to_i16_bytes(&raw);
        let wav = build_wav(16000, 1, 16, WAVE_FORMAT_PCM, &bytes);

        let result = load_wav_from_bytes(&wav);
        assert!(result.is_ok());
        let samples = result.expect("already checked");
        for &s in &samples {
            assert!((-1.0..=1.0).contains(&s), "Sample {s} outside [-1.0, 1.0]");
        }
    }

    // -----------------------------------------------------------------------
    // load_raw_pcm helper
    // -----------------------------------------------------------------------
    #[test]
    fn test_load_raw_pcm_passthrough() {
        let input = vec![0.1_f32, 0.2, 0.3, 0.4];
        let result = load_raw_pcm(&input, 16000, 1);
        assert_eq!(result.len(), input.len());
        for (a, b) in result.iter().zip(input.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn test_load_raw_pcm_stereo_resample() {
        // Stereo interleaved at 32kHz: L and R identical
        let mut input = Vec::new();
        for i in 0..64 {
            let v = i as f32 / 63.0;
            input.push(v); // L
            input.push(v); // R
        }
        let result = load_raw_pcm(&input, 32000, 2);
        // 64 frames at 32kHz → 32 frames at 16kHz
        assert_eq!(result.len(), 32);
        // Should be monotonically increasing
        for i in 1..result.len() {
            assert!(result[i] >= result[i - 1] - 1e-6);
        }
    }

    // -----------------------------------------------------------------------
    // WAV with unknown chunk before data (should skip it)
    // -----------------------------------------------------------------------
    #[test]
    fn test_skip_unknown_chunks() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        // placeholder for size – fill in later
        let riff_size_pos = buf.len();
        buf.extend_from_slice(&0_u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");

        // fmt chunk
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16_u32.to_le_bytes());
        buf.extend_from_slice(&1_u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&1_u16.to_le_bytes()); // mono
        buf.extend_from_slice(&16000_u32.to_le_bytes());
        buf.extend_from_slice(&32000_u32.to_le_bytes());
        buf.extend_from_slice(&2_u16.to_le_bytes());
        buf.extend_from_slice(&16_u16.to_le_bytes());

        // Unknown "LIST" chunk with 10 bytes of junk
        buf.extend_from_slice(b"LIST");
        buf.extend_from_slice(&10_u32.to_le_bytes());
        buf.extend_from_slice(&[0xAB; 10]);

        // data chunk
        let samples_i16: Vec<i16> = vec![100, 200, 300];
        let sample_bytes = samples_to_i16_bytes(&samples_i16);
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&(sample_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(&sample_bytes);

        // Fix RIFF size
        let riff_size = (buf.len() - 8) as u32;
        buf[riff_size_pos..riff_size_pos + 4].copy_from_slice(&riff_size.to_le_bytes());

        let result = load_wav_from_bytes(&buf);
        assert!(result.is_ok());
        let samples = result.expect("already checked");
        assert_eq!(samples.len(), 3);
        assert!((samples[0] - 100.0 / 32768.0).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // load_wav with a temp file
    // -----------------------------------------------------------------------
    #[test]
    fn test_load_wav_from_file() {
        let raw: Vec<f32> = vec![0.5, -0.5, 0.0];
        let bytes = samples_to_f32_bytes(&raw);
        let wav = build_wav(16000, 1, 32, WAVE_FORMAT_IEEE_FLOAT, &bytes);

        let dir = std::env::temp_dir();
        let path = dir.join("oxiwhisper_test_audio.wav");
        std::fs::write(&path, &wav).expect("failed to write temp WAV");

        let result = load_wav(&path);
        let _ = std::fs::remove_file(&path);

        assert!(result.is_ok());
        let samples = result.expect("already checked");
        assert_eq!(samples.len(), 3);
        assert!((samples[0] - 0.5).abs() < 1e-7);
    }

    // -----------------------------------------------------------------------
    // Four-channel downmix
    // -----------------------------------------------------------------------
    #[test]
    fn test_four_channel_downmix() {
        let raw: Vec<f32> = vec![
            1.0, 0.0, -1.0, 0.0, // frame 0 → avg 0.0
            0.5, 0.5, 0.5, 0.5, // frame 1 → avg 0.5
        ];
        let bytes = samples_to_f32_bytes(&raw);
        let wav = build_wav(16000, 4, 32, WAVE_FORMAT_IEEE_FLOAT, &bytes);

        let result = load_wav_from_bytes(&wav);
        assert!(result.is_ok());
        let samples = result.expect("already checked");
        assert_eq!(samples.len(), 2);
        assert!(samples[0].abs() < 1e-6);
        assert!((samples[1] - 0.5).abs() < 1e-6);
    }
}
