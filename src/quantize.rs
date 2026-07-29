//! GGML quantization support: Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0 block
//! quantization, dequantization, and SIMD-accelerated dot products.
//!
//! The [`QuantType`] discriminants map 1:1 onto the `ggml_type` enum shared by
//! the legacy GGML whisper container and GGUF — see
//! [`QuantType::from_ggml_type`].

mod dequant;
mod dot_dispatch;
mod dot_scalar;
#[cfg(target_arch = "aarch64")]
mod dot_simd_neon;
#[cfg(target_arch = "x86_64")]
mod dot_simd_x86;
mod quant;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public symbols so callers see the same paths.

pub use types::{
    Q4_0_BLOCK_BYTES, Q4_0_BLOCK_SIZE, Q4_1_BLOCK_BYTES, Q4_1_BLOCK_SIZE, Q5_0_BLOCK_BYTES,
    Q5_0_BLOCK_SIZE, Q5_1_BLOCK_BYTES, Q5_1_BLOCK_SIZE, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_SIZE,
    QuantType, QuantizedTensor,
};

pub use dequant::{
    dequantize, dequantize_q4_0, dequantize_q4_0_block, dequantize_q4_1, dequantize_q4_1_block,
    dequantize_q5_0, dequantize_q5_0_block, dequantize_q5_1, dequantize_q5_1_block,
    dequantize_q8_0, dequantize_q8_0_block,
};

pub use quant::{
    QuantizeStats, quantize_block_q4_0, quantize_block_q4_1, quantize_block_q5_0,
    quantize_block_q5_1, quantize_block_q8_0, quantize_tensor, quantize_to_q4_0, quantize_to_q4_1,
    quantize_to_q5_0, quantize_to_q5_1, quantize_to_q8_0,
};

pub use dot_scalar::{dot_q4_0, dot_q4_1, dot_q5_0, dot_q5_1, dot_q8_0};

pub use dot_dispatch::{dot_q4_0_fast, dot_q4_1_fast, dot_q5_0_fast, dot_q5_1_fast, dot_q8_0_fast};
