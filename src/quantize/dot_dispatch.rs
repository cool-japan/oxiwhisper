//! SIMD dispatch shims: select the fastest available dot product at runtime.

use super::dot_scalar::{dot_q4_0, dot_q5_0, dot_q8_0};

#[cfg(target_arch = "x86_64")]
use super::dot_simd_x86::{dot_q4_0_avx2, dot_q5_0_avx2, dot_q8_0_avx2};

#[cfg(target_arch = "aarch64")]
use super::dot_simd_neon::{dot_q4_0_neon, dot_q5_0_neon, dot_q8_0_neon};

/// SIMD-accelerated Q4_0 dot product when available, falling back to scalar.
///
/// On x86_64 with AVX2+FMA, uses 256-bit SIMD to process each 32-element block.
/// On aarch64, uses NEON intrinsics.
/// Otherwise falls back to the scalar `dot_q4_0`.
pub fn dot_q4_0_fast(input: &[f32], quantized: &[u8], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: feature detection passed — AVX2 + FMA are available.
            return unsafe { dot_q4_0_avx2(input, quantized, n) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return dot_q4_0_neon(input, quantized, n);
    }
    #[allow(unreachable_code)]
    dot_q4_0(input, quantized, n)
}

/// SIMD-accelerated Q5_0 dot product when available, falling back to scalar.
///
/// On x86_64 with AVX2+FMA, uses 256-bit SIMD to process each 32-element block.
/// On aarch64, uses NEON intrinsics.
/// Otherwise falls back to the scalar `dot_q5_0`.
pub fn dot_q5_0_fast(input: &[f32], quantized: &[u8], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: feature detection passed — AVX2 + FMA are available.
            return unsafe { dot_q5_0_avx2(input, quantized, n) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return dot_q5_0_neon(input, quantized, n);
    }
    #[allow(unreachable_code)]
    dot_q5_0(input, quantized, n)
}

/// SIMD-accelerated Q8_0 dot product when available, falling back to scalar.
///
/// On x86_64 with AVX2+FMA, uses 256-bit SIMD to process each 32-element block.
/// On aarch64, uses NEON intrinsics.
/// Otherwise falls back to the scalar `dot_q8_0`.
pub fn dot_q8_0_fast(input: &[f32], quantized: &[u8], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: feature detection passed — AVX2 + FMA are available.
            return unsafe { dot_q8_0_avx2(input, quantized, n) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return dot_q8_0_neon(input, quantized, n);
    }
    #[allow(unreachable_code)]
    dot_q8_0(input, quantized, n)
}
