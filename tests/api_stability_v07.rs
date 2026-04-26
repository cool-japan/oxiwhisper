//! Regression test: ensure all pre-split public items remain importable.
//!
//! If a split accidentally removes a public re-export, this test will fail to compile.

use oxiwhisper::{
    InferenceBuffer, KvCacheDtype, ModelInfo, ModelStats, OxiWhisperError, Segment,
    TranscribeOptions, TranscribeResult, TranscribeTiming, WhisperModel,
};

// Verify threading public API
use oxiwhisper::threading::set_thread_count;

#[test]
fn test_public_types_are_importable() {
    // This test passes if it compiles
    let _opts = TranscribeOptions::default();
    let _ = KvCacheDtype::F32;
    let result = set_thread_count(1);
    assert!(result.is_ok());
}

#[test]
fn test_error_type_is_display() {
    let e = OxiWhisperError::ConfigError("test".into());
    let msg = format!("{e}");
    assert!(msg.contains("test"));
}

#[test]
fn test_kv_cache_dtype_variants_accessible() {
    let _f32 = KvCacheDtype::F32;
    let _vhalf = KvCacheDtype::VHalf;
    let _kvhalf = KvCacheDtype::KvHalf;
}

#[test]
fn test_model_info_fields_accessible() {
    let info = ModelInfo {
        n_vocab: 51865,
        n_audio_layers: 4,
        n_text_layers: 4,
        d_model: 384,
        n_mels: 80,
        n_audio_heads: 6,
        n_text_heads: 6,
        n_audio_ctx: 1500,
        n_text_ctx: 448,
    };
    assert_eq!(info.n_vocab, 51865);
    assert_eq!(info.n_mels, 80);
}

#[test]
fn test_model_stats_fields_accessible() {
    let stats = ModelStats {
        total_params: 1000,
        quantized_params: 100,
        float32_params: 900,
        estimated_memory_bytes: 4000,
    };
    assert_eq!(
        stats.total_params,
        stats.quantized_params + stats.float32_params
    );
}

#[test]
fn test_segment_fields_accessible() {
    let seg = Segment {
        text: "hello".into(),
        start: 0.0,
        end: 1.5,
        confidence: -0.3,
        is_hallucination: false,
    };
    assert_eq!(seg.text, "hello");
    assert!(!seg.is_hallucination);
}

#[test]
fn test_transcribe_result_fields_accessible() {
    let result = TranscribeResult {
        text: "hello world".into(),
        segments: Vec::new(),
        language: Some("en".into()),
    };
    assert_eq!(result.text, "hello world");
    assert_eq!(result.language, Some("en".into()));
}

#[test]
fn test_transcribe_timing_fields_accessible() {
    use std::time::Duration;
    let timing = TranscribeTiming {
        mel: Duration::from_millis(10),
        encoder: Duration::from_millis(20),
        decoder: Duration::from_millis(30),
        total: Duration::from_millis(60),
    };
    assert!(timing.total >= timing.mel + timing.encoder + timing.decoder);
}

#[test]
fn test_inference_buffer_type_accessible() {
    // InferenceBuffer is opaque from the outside — just verify the type is importable
    // and that it can be obtained from WhisperModel::create_buffer().
    // We can't construct it directly (mel_buf is pub(crate)).
    let _: fn() -> InferenceBuffer; // type is accessible
}

#[test]
fn test_whisper_model_is_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<WhisperModel>();
    assert_sync::<WhisperModel>();
}

#[test]
fn test_error_all_variants_display() {
    let variants = vec![
        OxiWhisperError::InvalidModel("test".into()),
        OxiWhisperError::InferenceFailed("test".into()),
        OxiWhisperError::ShapeMismatch("test".into()),
        OxiWhisperError::ConfigError("test".into()),
        OxiWhisperError::AudioFormatError("test".into()),
    ];
    for e in &variants {
        let msg = format!("{e}");
        assert!(
            !msg.is_empty(),
            "display impl should produce non-empty string"
        );
    }
}
