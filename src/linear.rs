use crate::quantize::{self, QuantType, QuantizedTensor};
use crate::tensor::Tensor;

/// Linear layer: y = x @ W + b
///
/// GGML weight layout: \[in_features, out_features\]  (ne\[0\]=in_f is fastest varying)
/// → element W[o, k] = weight.data[o * in_f + k]
///
/// Strategy:
/// - batch >= 4 (encoder, prefill): matrixmultiply sgemm → AVX2/FMA kernel
/// - batch == 1 (decoder step): scalar GEMV → reads weight exactly once (bandwidth optimal)
pub fn linear(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
    let in_f = input.shape[input.shape.len() - 1];
    assert_eq!(weight.shape[0], in_f, "linear: weight in_features mismatch");
    let out_f = weight.shape[1];
    let batch = input.data.len() / in_f;
    let mut out = vec![0.0f32; batch * out_f];

    if batch >= 4 {
        // BLAS-3 path: matrixmultiply provides AVX2/FMA kernels with proper cache blocking.
        // A = input  [batch, in_f] row-major  → rsa=in_f, csa=1
        // B = weight [in_f, out_f] col-major  → rsb=1, csb=in_f
        // C = out    [batch, out_f] row-major  → rsc=out_f, csc=1
        unsafe {
            matrixmultiply::sgemm(
                batch,
                in_f,
                out_f,
                1.0,
                input.data.as_ptr(),
                in_f as isize,
                1,
                weight.data.as_ptr(),
                1,
                in_f as isize,
                0.0,
                out.as_mut_ptr(),
                out_f as isize,
                1,
            );
        }
    } else {
        // GEMV path: reads each weight row exactly once — optimal for memory bandwidth.
        // Multiple accumulators help LLVM generate wider SIMD reductions.
        for b in 0..batch {
            let x = &input.data[b * in_f..(b + 1) * in_f];
            let y = &mut out[b * out_f..(b + 1) * out_f];
            for (o, y_val) in y.iter_mut().enumerate() {
                let w = &weight.data[o * in_f..(o + 1) * in_f];
                *y_val = dot(x, w);
            }
        }
    }

    if let Some(b) = bias {
        for row in out.chunks_mut(out_f) {
            for (v, &bv) in row.iter_mut().zip(&b.data) {
                *v += bv;
            }
        }
    }

    let mut shape = input.shape.clone();
    let last_idx = shape.len() - 1;
    shape[last_idx] = out_f;
    Tensor::from_vec(out, &shape)
}

// ---------------------------------------------------------------------------
// SIMD dot-product kernels
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod simd_x86 {
    /// AVX2 + FMA dot product for f32 slices.
    ///
    /// # Safety
    /// Caller must ensure AVX2 and FMA are available on the current CPU.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::*;

        unsafe {
            let n = a.len().min(b.len());
            let chunks = n / 8;
            let mut acc = _mm256_setzero_ps();

            for i in 0..chunks {
                let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
                acc = _mm256_fmadd_ps(va, vb, acc);
            }

            // Horizontal sum of 8 f32 lanes
            let hi = _mm256_extractf128_ps(acc, 1);
            let lo = _mm256_castps256_ps128(acc);
            let sum128 = _mm_add_ps(lo, hi);
            let shuf = _mm_movehdup_ps(sum128);
            let sums = _mm_add_ps(sum128, shuf);
            let shuf2 = _mm_movehl_ps(sums, sums);
            let result = _mm_add_ss(sums, shuf2);
            let mut total = _mm_cvtss_f32(result);

            // Handle remainder elements
            for i in (chunks * 8)..n {
                total += *a.get_unchecked(i) * *b.get_unchecked(i);
            }

            total
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod simd_neon {
    /// NEON dot product for f32 slices.
    ///
    /// NEON is always available on aarch64 — no runtime detection needed.
    pub fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::aarch64::*;

        let n = a.len().min(b.len());
        let chunks = n / 4;

        unsafe {
            let mut acc = vdupq_n_f32(0.0);

            for i in 0..chunks {
                let va = vld1q_f32(a.as_ptr().add(i * 4));
                let vb = vld1q_f32(b.as_ptr().add(i * 4));
                acc = vfmaq_f32(acc, va, vb);
            }

            let mut total = vaddvq_f32(acc);

            // Handle remainder elements
            for i in (chunks * 4)..n {
                total += *a.get_unchecked(i) * *b.get_unchecked(i);
            }

            total
        }
    }
}

/// Compute dot product using the best available SIMD implementation.
#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: feature detection passed — AVX2 + FMA are available.
            return unsafe { simd_x86::dot_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        return simd_neon::dot_neon(a, b);
    }

    // Scalar fallback for other architectures or missing features
    #[allow(unreachable_code)]
    dot_scalar(a, b)
}

/// Scalar dot product with 8 parallel accumulators.
///
/// Serves as the fallback when no explicit SIMD path is available.
/// The accumulator pattern still helps LLVM auto-vectorise on architectures
/// we don't have explicit intrinsics for.
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let chunks = n / 8;
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut s4, mut s5, mut s6, mut s7) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for c in 0..chunks {
        let base = c * 8;
        s0 += a[base] * b[base];
        s1 += a[base + 1] * b[base + 1];
        s2 += a[base + 2] * b[base + 2];
        s3 += a[base + 3] * b[base + 3];
        s4 += a[base + 4] * b[base + 4];
        s5 += a[base + 5] * b[base + 5];
        s6 += a[base + 6] * b[base + 6];
        s7 += a[base + 7] * b[base + 7];
    }
    let mut sum = (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);
    for i in (chunks * 8)..n {
        sum += a[i] * b[i];
    }
    sum
}

/// Compute the dot product of two f32 slices using the best available SIMD path.
#[inline(always)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    dot_product(a, b)
}

/// Linear layer using quantized weights: y = x @ W_quant + b
///
/// The quantized weight has logical shape `[in_features, out_features]` (matching the
/// f32 `linear()` convention), but the physical memory layout is
/// `out_features` rows of `in_features` elements each — identical to the GEMV path
/// in `linear()`.
///
/// For each output row of input, for each output feature, we compute the dot product
/// between the input row and the corresponding quantized weight row using the
/// quantized dot product functions (`dot_q4_0` / `dot_q8_0`).
pub fn linear_quantized(input: &Tensor, weight: &QuantizedTensor, bias: Option<&Tensor>) -> Tensor {
    let in_f = input.shape[input.shape.len() - 1];
    assert_eq!(
        weight.shape[0], in_f,
        "linear_quantized: weight in_features mismatch (expected {in_f}, got {})",
        weight.shape[0]
    );
    let out_f = weight.shape[1];
    let batch = input.data.len() / in_f;
    let mut out = vec![0.0f32; batch * out_f];

    let dot_fn: fn(&[f32], &[u8], usize) -> f32 = match weight.qtype {
        QuantType::Q4_0 => quantize::dot_q4_0_fast,
        QuantType::Q5_0 => quantize::dot_q5_0_fast,
        QuantType::Q8_0 => quantize::dot_q8_0_fast,
    };

    for b in 0..batch {
        let x = &input.data[b * in_f..(b + 1) * in_f];
        let y = &mut out[b * out_f..(b + 1) * out_f];
        for (o, y_val) in y.iter_mut().enumerate() {
            let w_row = weight.row_bytes(o);
            *y_val = dot_fn(x, w_row, in_f);
        }
    }

    if let Some(b) = bias {
        for row in out.chunks_mut(out_f) {
            for (v, &bv) in row.iter_mut().zip(&b.data) {
                *v += bv;
            }
        }
    }

    let mut shape = input.shape.clone();
    let last_idx = shape.len() - 1;
    shape[last_idx] = out_f;
    Tensor::from_vec(out, &shape)
}

/// Linear layer that automatically dispatches to quantized or f32 path.
///
/// Prefers quantized if available, falls back to f32.
/// Returns an error if neither weight is provided.
pub fn linear_auto(
    input: &Tensor,
    weight_f32: Option<&Tensor>,
    weight_quant: Option<&QuantizedTensor>,
    bias: Option<&Tensor>,
) -> Result<Tensor, String> {
    if let Some(wq) = weight_quant {
        Ok(linear_quantized(input, wq, bias))
    } else if let Some(wf) = weight_f32 {
        Ok(linear(input, wf, bias))
    } else {
        Err("linear_auto: neither f32 nor quantized weight provided".into())
    }
}

/// 1D convolution: [batch, in_ch, len] → [batch, out_ch, out_len]
/// GGML weight format: \[kernel, in_ch, out_ch\]  (ne\[0\]=kernel fastest)
/// → W[oc, ic, k] = weight.data[oc * in_ch * kernel + ic * kernel + k]
pub fn conv1d(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    stride: usize,
    padding: usize,
) -> Tensor {
    assert_eq!(input.ndim(), 3);
    assert_eq!(weight.ndim(), 3);

    let (batch, in_ch, in_len) = (input.shape[0], input.shape[1], input.shape[2]);
    let (kernel, out_ch) = (weight.shape[0], weight.shape[2]);
    assert_eq!(weight.shape[1], in_ch, "conv1d: weight in_ch mismatch");

    let out_len = (in_len + 2 * padding).saturating_sub(kernel) / stride + 1;
    let mut output = vec![0.0f32; batch * out_ch * out_len];

    for b in 0..batch {
        for oc in 0..out_ch {
            let bias_val = bias.data[oc];
            for ol in 0..out_len {
                let start = ol * stride;
                let mut sum = bias_val;
                for ic in 0..in_ch {
                    let w_row = &weight.data[oc * in_ch * kernel + ic * kernel..][..kernel];
                    for (k, &w_val) in w_row.iter().enumerate() {
                        let pos = start + k;
                        if pos >= padding && pos < padding + in_len {
                            let x_idx = b * in_ch * in_len + ic * in_len + (pos - padding);
                            sum += input.data[x_idx] * w_val;
                        }
                    }
                }
                output[b * out_ch * out_len + oc * out_len + ol] = sum;
            }
        }
    }

    Tensor::from_vec(output, &[batch, out_ch, out_len])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_shape() {
        // Input [3, 4], weight [4, 5] -> output should be [3, 5]
        let input = Tensor::from_vec(vec![0.01; 3 * 4], &[3, 4]);
        let weight = Tensor::from_vec(vec![0.01; 4 * 5], &[4, 5]);
        let result = linear(&input, &weight, None);
        assert_eq!(result.shape, vec![3, 5]);
        assert_eq!(result.data.len(), 3 * 5);
    }

    #[test]
    fn test_linear_with_bias() {
        // Input [3, 4], weight [4, 5], bias [5] -> output should be [3, 5]
        let input = Tensor::from_vec(vec![0.01; 3 * 4], &[3, 4]);
        let weight = Tensor::from_vec(vec![0.01; 4 * 5], &[4, 5]);
        let bias = Tensor::from_vec(vec![1.0; 5], &[5]);

        let result_no_bias = linear(&input, &weight, None);
        let result_with_bias = linear(&input, &weight, Some(&bias));

        assert_eq!(result_with_bias.shape, vec![3, 5]);

        // Every element of result_with_bias should be result_no_bias + 1.0
        for (wb, nb) in result_with_bias.data.iter().zip(result_no_bias.data.iter()) {
            assert!((wb - nb - 1.0).abs() < 1e-5, "bias was not added correctly");
        }
    }

    #[test]
    fn test_linear_known_values() {
        // 2x2 weight with known values
        // input = [[1, 2]], weight = [[3, 4], [5, 6]]
        // output = [1*3 + 2*5, 1*4 + 2*6] = [13, 16]
        let input = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]);
        let weight = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], &[2, 2]);
        // Weight layout: [in_f, out_f] -> W[o, k] = weight.data[o * in_f + k]
        // So weight is row-major with rows indexed by output:
        //   row 0 (out=0): [3, 4] -> but wait, shape is [in_f=2, out_f=2]
        // Actually the linear function: weight.shape[0] = in_f, weight.shape[1] = out_f
        // It uses dot(x, w_row) where w_row = weight.data[o * in_f..(o+1)*in_f]
        // In the GEMV path: for output o, w = weight.data[o * in_f..(o+1) * in_f]
        // So weight data is laid out as [out_f rows of in_f elements]
        // With data [3.0, 4.0, 5.0, 6.0] and shape [2, 2]:
        //   out=0: w=[3, 4], dot([1,2], [3,4]) = 11
        //   out=1: w=[5, 6], dot([1,2], [5,6]) = 17
        // But wait - the sgemm path treats it differently from GEMV.
        // batch=1 < 4, so GEMV path is used.
        let result = linear(&input, &weight, None);
        assert_eq!(result.shape, vec![1, 2]);
        assert!(
            (result.data[0] - 11.0).abs() < 1e-5,
            "expected 11.0, got {}",
            result.data[0]
        );
        assert!(
            (result.data[1] - 17.0).abs() < 1e-5,
            "expected 17.0, got {}",
            result.data[1]
        );

        // Also test with bias
        let bias = Tensor::from_vec(vec![0.5, -0.5], &[2]);
        let result_b = linear(&input, &weight, Some(&bias));
        assert!((result_b.data[0] - 11.5).abs() < 1e-5);
        assert!((result_b.data[1] - 16.5).abs() < 1e-5);
    }

    #[test]
    fn test_linear_batch_sgemm_path() {
        // Use batch >= 4 to exercise the sgemm path
        // Input [5, 2], weight [2, 3]
        // Fill input with 1.0, weight with 0.5
        // Each output element = 2 * 0.5 = 1.0 (sum of 2 elements, each 1.0 * 0.5)
        // Wait - sgemm treats weight as col-major: rsb=1, csb=in_f
        // This means sgemm reads weight as if it were [in_f, out_f] column-major,
        // which is actually [out_f, in_f] row-major -- same layout as GEMV path.
        let input = Tensor::from_vec(vec![1.0; 5 * 2], &[5, 2]);
        let weight = Tensor::from_vec(vec![0.5; 2 * 3], &[2, 3]);
        let result = linear(&input, &weight, None);
        assert_eq!(result.shape, vec![5, 3]);
        for val in &result.data {
            assert!((*val - 1.0).abs() < 1e-5, "expected 1.0, got {}", val);
        }
    }

    #[test]
    fn test_conv1d_shape() {
        // Input [1, 2, 8], weight [kernel=4, in_ch=2, out_ch=3], stride=1, padding=1
        // out_len = (8 + 2*1 - 4) / 1 + 1 = 7
        let input = Tensor::from_vec(vec![0.01; 2 * 8], &[1, 2, 8]);
        let weight = Tensor::from_vec(vec![0.01; 4 * 2 * 3], &[4, 2, 3]);
        let bias = Tensor::from_vec(vec![0.0; 3], &[3]);
        let result = conv1d(&input, &weight, &bias, 1, 1);
        assert_eq!(result.shape[0], 1, "batch dim");
        assert_eq!(result.shape[1], 3, "out_ch dim");
        let expected_out_len = (8 + 2 - 4) + 1; // 7
        assert_eq!(result.shape[2], expected_out_len, "out_len dim");
    }

    /// Helper: create a Q8_0 quantized tensor from f32 data.
    /// The f32 data is arranged as [rows, cols] (cols must be multiple of 32).
    /// Each block of 32 f32 values is quantized to Q8_0 format.
    fn quantize_to_q8_0(data: &[f32], shape: &[usize]) -> QuantizedTensor {
        use crate::quantize::{Q8_0_BLOCK_BYTES, Q8_0_BLOCK_SIZE};
        let n_elements = data.len();
        let n_blocks = n_elements / Q8_0_BLOCK_SIZE;
        let mut raw = vec![0u8; n_blocks * Q8_0_BLOCK_BYTES];

        for b in 0..n_blocks {
            let block_data = &data[b * Q8_0_BLOCK_SIZE..(b + 1) * Q8_0_BLOCK_SIZE];
            // Find max absolute value for scale
            let amax = block_data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let scale = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            let scale_f16 = half::f16::from_f32(scale);
            let block_start = b * Q8_0_BLOCK_BYTES;
            let le = scale_f16.to_le_bytes();
            raw[block_start] = le[0];
            raw[block_start + 1] = le[1];
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            for i in 0..32 {
                let quantized = (block_data[i] * inv_scale).round().clamp(-128.0, 127.0) as i8;
                raw[block_start + 2 + i] = quantized as u8;
            }
        }

        QuantizedTensor {
            raw,
            shape: shape.to_vec(),
            qtype: QuantType::Q8_0,
        }
    }

    #[test]
    fn test_linear_quantized_matches_f32() {
        // Create a weight matrix [in_f=64, out_f=32] with known values.
        // The GEMV layout is [out_f][in_f] in memory, shape = [in_f, out_f].
        let in_f = 64;
        let out_f = 32;
        let mut weight_data = vec![0.0f32; out_f * in_f];
        for o in 0..out_f {
            for k in 0..in_f {
                weight_data[o * in_f + k] = ((o * in_f + k) as f32 * 0.01) - 1.0;
            }
        }
        let weight_f32 = Tensor::from_vec(weight_data.clone(), &[in_f, out_f]);

        // Quantize the same data to Q8_0
        let weight_quant = quantize_to_q8_0(&weight_data, &[in_f, out_f]);

        // Input
        let mut input_data = vec![0.0f32; in_f];
        for (i, val) in input_data.iter_mut().enumerate().take(in_f) {
            *val = (i as f32 + 1.0) * 0.1;
        }
        let input = Tensor::from_vec(input_data, &[1, in_f]);

        let result_f32 = linear(&input, &weight_f32, None);
        let result_quant = linear_quantized(&input, &weight_quant, None);

        assert_eq!(result_f32.shape, result_quant.shape);

        // Q8_0 has ~0.5% relative error. Allow a generous tolerance.
        for (i, (f, q)) in result_f32
            .data
            .iter()
            .zip(result_quant.data.iter())
            .enumerate()
        {
            let diff = (f - q).abs();
            let scale = f.abs().max(1.0);
            assert!(
                diff / scale < 0.05,
                "linear_quantized mismatch at output {i}: f32={f}, quant={q}, diff={diff}"
            );
        }
    }

    #[test]
    fn test_linear_quantized_with_bias() {
        let in_f = 32;
        let out_f = 32;
        let mut weight_data = vec![0.0f32; out_f * in_f];
        for (i, val) in weight_data.iter_mut().enumerate() {
            *val = (i as f32 * 0.005) - 0.5;
        }
        let weight_quant = quantize_to_q8_0(&weight_data, &[in_f, out_f]);

        let input = Tensor::from_vec(vec![1.0; in_f], &[1, in_f]);
        let bias = Tensor::from_vec(vec![0.5; out_f], &[out_f]);

        let result_no_bias = linear_quantized(&input, &weight_quant, None);
        let result_with_bias = linear_quantized(&input, &weight_quant, Some(&bias));

        for (nb, wb) in result_no_bias.data.iter().zip(result_with_bias.data.iter()) {
            assert!(
                (wb - nb - 0.5).abs() < 1e-5,
                "bias not added correctly: no_bias={nb}, with_bias={wb}"
            );
        }
    }

    #[test]
    fn test_linear_quantized_shape() {
        let in_f = 64;
        let out_f = 32;
        let weight_data = vec![0.0f32; out_f * in_f];
        let weight_quant = quantize_to_q8_0(&weight_data, &[in_f, out_f]);

        // Batch of 3
        let input = Tensor::from_vec(vec![0.01; 3 * in_f], &[3, in_f]);
        let result = linear_quantized(&input, &weight_quant, None);
        assert_eq!(result.shape, vec![3, out_f]);
        assert_eq!(result.data.len(), 3 * out_f);
    }

    #[test]
    fn test_linear_auto_dispatch() {
        let in_f = 32;
        let out_f = 32;
        let weight_data = vec![0.1f32; out_f * in_f];
        let weight_f32 = Tensor::from_vec(weight_data.clone(), &[in_f, out_f]);
        let weight_quant = quantize_to_q8_0(&weight_data, &[in_f, out_f]);

        let input = Tensor::from_vec(vec![1.0; in_f], &[1, in_f]);

        // Test f32 path
        let result_f32 = linear_auto(&input, Some(&weight_f32), None, None)
            .expect("linear_auto f32 should succeed");
        assert_eq!(result_f32.shape, vec![1, out_f]);

        // Test quantized path
        let result_quant = linear_auto(&input, None, Some(&weight_quant), None)
            .expect("linear_auto quant should succeed");
        assert_eq!(result_quant.shape, vec![1, out_f]);

        // Test preferring quantized when both provided
        let result_both = linear_auto(&input, Some(&weight_f32), Some(&weight_quant), None)
            .expect("linear_auto both should succeed");
        // Should match quantized result
        assert_eq!(result_both.data, result_quant.data);

        // Test error when neither provided
        let err = linear_auto(&input, None, None, None);
        assert!(err.is_err());
    }

    // -----------------------------------------------------------------------
    // dot_product / SIMD tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_dot_product_basic() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        let result = super::dot_product(&a, &b);
        assert!((result - 70.0).abs() < 1e-4, "expected 70.0, got {result}");
    }

    #[test]
    fn test_dot_product_empty() {
        let result = super::dot_product(&[], &[]);
        assert!((result - 0.0).abs() < 1e-10, "expected 0.0, got {result}");
    }

    #[test]
    fn test_dot_product_unaligned() {
        // Length 13 — not a multiple of 4 or 8
        let a: Vec<f32> = (1..=13).map(|x| x as f32).collect();
        let b: Vec<f32> = (1..=13).map(|x| x as f32 * 0.5).collect();
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let result = super::dot_product(&a, &b);
        assert!(
            (result - expected).abs() < 1e-3,
            "expected {expected}, got {result}"
        );
    }

    #[test]
    fn test_dot_product_large() {
        let n = 1000;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..n).map(|i| 1.0 - (i as f32) * 0.001).collect();
        let expected: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| x as f64 * y as f64)
            .sum();
        let result = super::dot_product(&a, &b) as f64;
        let rel_err = (result - expected).abs() / expected.abs().max(1e-10);
        assert!(
            rel_err < 1e-4,
            "expected {expected}, got {result}, rel_err={rel_err}"
        );
    }

    #[test]
    fn test_dot_product_simd_path_active() {
        // Verify we are using the expected SIMD path on the current platform
        #[cfg(target_arch = "x86_64")]
        {
            assert!(
                is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
                "AVX2+FMA should be available on this x86_64 machine"
            );
        }
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is always available on aarch64 — nothing to check,
            // but confirm dot_product works through the NEON path.
            let a = [1.0f32; 16];
            let b = [2.0f32; 16];
            let r = super::dot_product(&a, &b);
            assert!((r - 32.0).abs() < 1e-4, "NEON path produced {r}");
        }
    }

    #[test]
    fn test_conv1d_stride2() {
        // Input [1, 2, 8], weight [kernel=4, in_ch=2, out_ch=3], stride=2, padding=1
        // out_len = (8 + 2*1 - 4) / 2 + 1 = 4
        let input = Tensor::from_vec(vec![0.01; 2 * 8], &[1, 2, 8]);
        let weight = Tensor::from_vec(vec![0.01; 4 * 2 * 3], &[4, 2, 3]);
        let bias = Tensor::from_vec(vec![0.0; 3], &[3]);

        let result_s1 = conv1d(&input, &weight, &bias, 1, 1);
        let result_s2 = conv1d(&input, &weight, &bias, 2, 1);

        let out_len_s1 = result_s1.shape[2]; // 7
        let out_len_s2 = result_s2.shape[2]; // 4
        // stride=2 should roughly halve the output length
        assert_eq!(out_len_s2, (8 + 2 - 4) / 2 + 1);
        assert!(
            out_len_s2 < out_len_s1,
            "stride=2 output should be shorter than stride=1"
        );
        assert_eq!(result_s2.shape[0], 1);
        assert_eq!(result_s2.shape[1], 3);
    }
}
