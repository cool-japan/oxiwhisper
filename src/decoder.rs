//! Core Whisper text decoder: forward pass, greedy/sample decoding, language detection.
//!
//! This module is split into focused submodules:
//! - `kv_cache` — KV cache storage types (KvStorage, LayerKVCache)
//! - `sdpa` — Scaled dot-product attention kernels (HeadScratch, SdpaScratch, SDPA fns)
//! - `forward` — Decoder forward pass and prompt construction (ForwardCtx, decode, forward)
//! - `sampler` — Token sampling strategies (decode_greedy, decode_sample)

pub(crate) mod forward;
pub(crate) mod kv_cache;
pub(crate) mod sampler;
pub(crate) mod sdpa;

// Public crate API
pub use forward::{DecodeResult, decode};

// Re-exports for beam_search and other internal consumers
pub(crate) use forward::{ForwardCtx, MAX_DECODE_LENGTH, forward};
pub(crate) use kv_cache::LayerKVCache;
