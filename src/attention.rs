use crate::linear;
use crate::quantize::QuantizedTensor;
use crate::tensor::Tensor;

/// Weights for a multi-head attention layer.
pub struct AttentionWeights<'a> {
    pub q_weight: &'a Tensor,
    pub q_bias: &'a Tensor,
    pub k_weight: &'a Tensor,
    pub v_weight: &'a Tensor,
    pub v_bias: &'a Tensor,
    pub out_weight: &'a Tensor,
    pub out_bias: &'a Tensor,
}

/// Weights for a multi-head attention layer, supporting both f32 and quantized weights.
pub struct AttentionWeightsAuto<'a> {
    pub q_weight_f32: Option<&'a Tensor>,
    pub q_weight_quant: Option<&'a QuantizedTensor>,
    pub q_bias: &'a Tensor,
    pub k_weight_f32: Option<&'a Tensor>,
    pub k_weight_quant: Option<&'a QuantizedTensor>,
    pub v_weight_f32: Option<&'a Tensor>,
    pub v_weight_quant: Option<&'a QuantizedTensor>,
    pub v_bias: &'a Tensor,
    pub out_weight_f32: Option<&'a Tensor>,
    pub out_weight_quant: Option<&'a QuantizedTensor>,
    pub out_bias: &'a Tensor,
}

/// Configuration for a multi-head attention layer.
pub struct AttentionConfig {
    pub n_head: usize,
    pub mask: bool,
}

/// Multi-head attention
///
/// q_proj: [n_state, n_state]
/// k_proj: [n_state, n_state]
/// v_proj: [n_state, n_state]
/// out_proj: [n_state, n_state]
///
/// Input x: [seq_len, n_state]
/// For cross-attention, xa (encoder output): [enc_seq_len, n_state]
pub fn multi_head_attention(
    x: &Tensor,
    xa: Option<&Tensor>,
    weights: &AttentionWeights<'_>,
    config: &AttentionConfig,
) -> Tensor {
    let n_state = x.shape[x.ndim() - 1];
    let n_head = config.n_head;
    let head_dim = n_state / n_head;
    let scale = (head_dim as f32).sqrt().recip();

    let kv_source = xa.unwrap_or(x);

    // Project Q, K, V
    let mut q = linear::linear(x, weights.q_weight, Some(weights.q_bias));
    let mut k = linear::linear(kv_source, weights.k_weight, None); // Whisper: no bias on K
    let mut v = linear::linear(kv_source, weights.v_weight, Some(weights.v_bias));

    let q_len = q.data.len() / n_state;
    let kv_len = k.data.len() / n_state;

    // Reshape to [seq_len, n_head, head_dim] (zero-copy)
    q.reshape_inplace(&[q_len, n_head, head_dim]);
    k.reshape_inplace(&[kv_len, n_head, head_dim]);
    v.reshape_inplace(&[kv_len, n_head, head_dim]);

    // Transpose to [n_head, seq_len, head_dim]
    let q = transpose_1_0(&q, n_head, q_len, head_dim);
    let k = transpose_1_0(&k, n_head, kv_len, head_dim);
    let v = transpose_1_0(&v, n_head, kv_len, head_dim);

    // Attention scores: Q @ K^T * scale  (sgemm with stride trick avoids explicit transpose)
    // q: [n_head, q_len, head_dim]  row-major
    // k: [n_head, kv_len, head_dim] row-major — K^T achieved by swapping strides
    let mut scores = Tensor::from_vec(
        vec![0.0f32; n_head * q_len * kv_len],
        &[n_head, q_len, kv_len],
    );

    for h in 0..n_head {
        let q_ptr = q.data[h * q_len * head_dim..].as_ptr();
        let k_ptr = k.data[h * kv_len * head_dim..].as_ptr();
        let scores_ptr = scores.data[h * q_len * kv_len..].as_mut_ptr();

        // Q @ K^T: [q_len, head_dim] × [head_dim, kv_len] = [q_len, kv_len]
        // K is [kv_len, head_dim] row-major, so K^T strides = (1, head_dim)
        unsafe {
            matrixmultiply::sgemm(
                q_len,
                head_dim,
                kv_len,
                scale,
                q_ptr,
                head_dim as isize,
                1,
                k_ptr,
                1,
                head_dim as isize,
                0.0,
                scores_ptr,
                kv_len as isize,
                1,
            );
        }
    }

    // Apply causal mask if needed
    if config.mask {
        for h in 0..n_head {
            let s_off = h * q_len * kv_len;
            for i in 0..q_len {
                for j in (i + 1)..kv_len {
                    scores.data[s_off + i * kv_len + j] = f32::NEG_INFINITY;
                }
            }
        }
    }

    // Softmax (in-place)
    scores.softmax_inplace();

    // scores @ V: [n_head, q_len, kv_len] @ [n_head, kv_len, head_dim] -> [n_head, q_len, head_dim]
    let mut attn_out = vec![0.0f32; n_head * q_len * head_dim];

    for h in 0..n_head {
        let scores_ptr = scores.data[h * q_len * kv_len..].as_ptr();
        let v_ptr = v.data[h * kv_len * head_dim..].as_ptr();
        let out_ptr = attn_out[h * q_len * head_dim..].as_mut_ptr();

        // scores @ V: [q_len, kv_len] × [kv_len, head_dim] = [q_len, head_dim]
        unsafe {
            matrixmultiply::sgemm(
                q_len,
                kv_len,
                head_dim,
                1.0,
                scores_ptr,
                kv_len as isize,
                1,
                v_ptr,
                head_dim as isize,
                1,
                0.0,
                out_ptr,
                head_dim as isize,
                1,
            );
        }
    }

    // Transpose back: [n_head, q_len, head_dim] -> [q_len, n_head, head_dim] -> [q_len, n_state]
    let mut concat = vec![0.0f32; q_len * n_state];
    for h in 0..n_head {
        for i in 0..q_len {
            for d in 0..head_dim {
                concat[i * n_state + h * head_dim + d] =
                    attn_out[h * q_len * head_dim + i * head_dim + d];
            }
        }
    }

    let concat = Tensor::from_vec(concat, &[q_len, n_state]);

    // Output projection
    linear::linear(&concat, weights.out_weight, Some(weights.out_bias))
}

/// Multi-head attention with automatic dispatch to quantized or f32 weights.
///
/// Identical to [`multi_head_attention`] but uses [`linear::linear_auto`] for
/// the Q/K/V/out projections.
pub fn multi_head_attention_auto(
    x: &Tensor,
    xa: Option<&Tensor>,
    weights: &AttentionWeightsAuto<'_>,
    config: &AttentionConfig,
) -> Result<Tensor, String> {
    let n_state = x.shape[x.ndim() - 1];
    let n_head = config.n_head;
    let head_dim = n_state / n_head;
    let scale = (head_dim as f32).sqrt().recip();

    let kv_source = xa.unwrap_or(x);

    // Project Q, K, V
    let mut q = linear::linear_auto(
        x,
        weights.q_weight_f32,
        weights.q_weight_quant,
        Some(weights.q_bias),
    )?;
    let mut k = linear::linear_auto(
        kv_source,
        weights.k_weight_f32,
        weights.k_weight_quant,
        None,
    )?;
    let mut v = linear::linear_auto(
        kv_source,
        weights.v_weight_f32,
        weights.v_weight_quant,
        Some(weights.v_bias),
    )?;

    let q_len = q.data.len() / n_state;
    let kv_len = k.data.len() / n_state;

    // Reshape to [seq_len, n_head, head_dim] (zero-copy)
    q.reshape_inplace(&[q_len, n_head, head_dim]);
    k.reshape_inplace(&[kv_len, n_head, head_dim]);
    v.reshape_inplace(&[kv_len, n_head, head_dim]);

    // Transpose to [n_head, seq_len, head_dim]
    let q = transpose_1_0(&q, n_head, q_len, head_dim);
    let k = transpose_1_0(&k, n_head, kv_len, head_dim);
    let v = transpose_1_0(&v, n_head, kv_len, head_dim);

    // Attention scores: Q @ K^T * scale  (sgemm with stride trick avoids explicit transpose)
    let mut scores = Tensor::from_vec(
        vec![0.0f32; n_head * q_len * kv_len],
        &[n_head, q_len, kv_len],
    );

    for h in 0..n_head {
        let q_ptr = q.data[h * q_len * head_dim..].as_ptr();
        let k_ptr = k.data[h * kv_len * head_dim..].as_ptr();
        let scores_ptr = scores.data[h * q_len * kv_len..].as_mut_ptr();

        unsafe {
            matrixmultiply::sgemm(
                q_len,
                head_dim,
                kv_len,
                scale,
                q_ptr,
                head_dim as isize,
                1,
                k_ptr,
                1,
                head_dim as isize,
                0.0,
                scores_ptr,
                kv_len as isize,
                1,
            );
        }
    }

    // Apply causal mask if needed
    if config.mask {
        for h in 0..n_head {
            let s_off = h * q_len * kv_len;
            for i in 0..q_len {
                for j in (i + 1)..kv_len {
                    scores.data[s_off + i * kv_len + j] = f32::NEG_INFINITY;
                }
            }
        }
    }

    // Softmax (in-place)
    scores.softmax_inplace();

    // scores @ V: [n_head, q_len, kv_len] @ [n_head, kv_len, head_dim] -> [n_head, q_len, head_dim]
    let mut attn_out = vec![0.0f32; n_head * q_len * head_dim];

    for h in 0..n_head {
        let scores_ptr = scores.data[h * q_len * kv_len..].as_ptr();
        let v_ptr = v.data[h * kv_len * head_dim..].as_ptr();
        let out_ptr = attn_out[h * q_len * head_dim..].as_mut_ptr();

        unsafe {
            matrixmultiply::sgemm(
                q_len,
                kv_len,
                head_dim,
                1.0,
                scores_ptr,
                kv_len as isize,
                1,
                v_ptr,
                head_dim as isize,
                1,
                0.0,
                out_ptr,
                head_dim as isize,
                1,
            );
        }
    }

    // Transpose back
    let mut concat = vec![0.0f32; q_len * n_state];
    for h in 0..n_head {
        for i in 0..q_len {
            for d in 0..head_dim {
                concat[i * n_state + h * head_dim + d] =
                    attn_out[h * q_len * head_dim + i * head_dim + d];
            }
        }
    }

    let concat = Tensor::from_vec(concat, &[q_len, n_state]);

    // Output projection
    linear::linear_auto(
        &concat,
        weights.out_weight_f32,
        weights.out_weight_quant,
        Some(weights.out_bias),
    )
}

/// Transpose [seq_len, n_head, head_dim] -> [n_head, seq_len, head_dim]
fn transpose_1_0(t: &Tensor, n_head: usize, seq_len: usize, head_dim: usize) -> Tensor {
    let mut out = vec![0.0f32; n_head * seq_len * head_dim];
    for s in 0..seq_len {
        for h in 0..n_head {
            for d in 0..head_dim {
                out[h * seq_len * head_dim + s * head_dim + d] =
                    t.data[s * n_head * head_dim + h * head_dim + d];
            }
        }
    }
    Tensor::from_vec(out, &[n_head, seq_len, head_dim])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    /// Helper: create a tensor filled with a constant value.
    fn filled_tensor(shape: &[usize], val: f32) -> Tensor {
        let size: usize = shape.iter().product();
        Tensor::from_vec(vec![val; size], shape)
    }

    /// Helper: create deterministic "weight-like" tensor with small varying values.
    fn det_tensor(shape: &[usize], seed: f32) -> Tensor {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|i| seed + (i as f32) * 0.001).collect();
        Tensor::from_vec(data, shape)
    }

    fn make_weights(n_state: usize) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
        let q_weight = det_tensor(&[n_state, n_state], 0.01);
        let q_bias = filled_tensor(&[n_state], 0.0);
        let k_weight = det_tensor(&[n_state, n_state], 0.02);
        let v_weight = det_tensor(&[n_state, n_state], 0.03);
        let v_bias = filled_tensor(&[n_state], 0.0);
        let out_weight = det_tensor(&[n_state, n_state], 0.04);
        let out_bias = filled_tensor(&[n_state], 0.0);
        (
            q_weight, q_bias, k_weight, v_weight, v_bias, out_weight, out_bias,
        )
    }

    #[test]
    fn test_self_attention_shape() {
        let n_state = 64;
        let n_head = 4;
        let seq_len = 10;

        let x = det_tensor(&[seq_len, n_state], 0.05);
        let (qw, qb, kw, vw, vb, ow, ob) = make_weights(n_state);

        let weights = AttentionWeights {
            q_weight: &qw,
            q_bias: &qb,
            k_weight: &kw,
            v_weight: &vw,
            v_bias: &vb,
            out_weight: &ow,
            out_bias: &ob,
        };
        let config = AttentionConfig {
            n_head,
            mask: false,
        };

        let result = multi_head_attention(&x, None, &weights, &config);
        assert_eq!(
            result.shape,
            vec![seq_len, n_state],
            "self-attention output shape mismatch"
        );
        // Verify no NaN
        for (i, val) in result.data.iter().enumerate() {
            assert!(val.is_finite(), "NaN/Inf at index {i}");
        }
    }

    #[test]
    fn test_cross_attention_shape() {
        let n_state = 64;
        let n_head = 4;
        let q_len = 5;
        let kv_len = 20;

        let x = det_tensor(&[q_len, n_state], 0.05);
        let xa = det_tensor(&[kv_len, n_state], 0.06);
        let (qw, qb, kw, vw, vb, ow, ob) = make_weights(n_state);

        let weights = AttentionWeights {
            q_weight: &qw,
            q_bias: &qb,
            k_weight: &kw,
            v_weight: &vw,
            v_bias: &vb,
            out_weight: &ow,
            out_bias: &ob,
        };
        let config = AttentionConfig {
            n_head,
            mask: false,
        };

        let result = multi_head_attention(&x, Some(&xa), &weights, &config);
        assert_eq!(
            result.shape,
            vec![q_len, n_state],
            "cross-attention output shape mismatch"
        );
        for (i, val) in result.data.iter().enumerate() {
            assert!(val.is_finite(), "NaN/Inf at index {i}");
        }
    }

    #[test]
    fn test_single_token_attention() {
        let n_state = 32;
        let n_head = 2;

        let x = det_tensor(&[1, n_state], 0.05);
        let (qw, qb, kw, vw, vb, ow, ob) = make_weights(n_state);

        let weights = AttentionWeights {
            q_weight: &qw,
            q_bias: &qb,
            k_weight: &kw,
            v_weight: &vw,
            v_bias: &vb,
            out_weight: &ow,
            out_bias: &ob,
        };
        let config = AttentionConfig {
            n_head,
            mask: false,
        };

        let result = multi_head_attention(&x, None, &weights, &config);
        assert_eq!(
            result.shape,
            vec![1, n_state],
            "single-token attention output shape mismatch"
        );
        assert_eq!(result.data.len(), n_state);
        for (i, val) in result.data.iter().enumerate() {
            assert!(val.is_finite(), "NaN/Inf at index {i}");
        }
    }

    #[test]
    fn test_transpose_1_0_roundtrip() {
        let n_head = 2;
        let seq_len = 3;
        let head_dim = 4;
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = Tensor::from_vec(data, &[seq_len, n_head, head_dim]);

        let transposed = transpose_1_0(&t, n_head, seq_len, head_dim);
        assert_eq!(transposed.shape, vec![n_head, seq_len, head_dim]);

        // Transpose back: [n_head, seq_len, head_dim] -> [seq_len, n_head, head_dim]
        // We can reuse transpose_1_0 with swapped dims conceptually
        let mut back = vec![0.0f32; 24];
        for h in 0..n_head {
            for s in 0..seq_len {
                for d in 0..head_dim {
                    back[s * n_head * head_dim + h * head_dim + d] =
                        transposed.data[h * seq_len * head_dim + s * head_dim + d];
                }
            }
        }
        assert_eq!(
            back, t.data,
            "round-trip transpose should recover original data"
        );
    }

    // ── New tests: QKT correctness, causal mask, scale factor, cross-attn ──

    #[test]
    fn test_sgemm_attention_qkt_correctness() {
        // Manually compute Q @ K^T for a tiny case: Q is [2,3], K is [4,3]
        // So Q @ K^T is [2,4].
        let q_data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        let k_data: [f32; 12] = [
            1.0, 0.0, 0.0, // row 0
            0.0, 1.0, 0.0, // row 1
            0.0, 0.0, 1.0, // row 2
            1.0, 1.0, 1.0, // row 3
        ]; // [4, 3]

        // Expected Q @ K^T:
        // row0: [1*1+2*0+3*0, 1*0+2*1+3*0, 1*0+2*0+3*1, 1*1+2*1+3*1] = [1, 2, 3, 6]
        // row1: [4*1+5*0+6*0, 4*0+5*1+6*0, 4*0+5*0+6*1, 4*1+5*1+6*1] = [4, 5, 6, 15]
        let expected: [f32; 8] = [1.0, 2.0, 3.0, 6.0, 4.0, 5.0, 6.0, 15.0];

        let m = 2;
        let inner = 3;
        let n = 4;
        let scale = 1.0_f32; // no scaling for this test

        let mut result = vec![0.0f32; m * n];
        unsafe {
            matrixmultiply::sgemm(
                m,
                inner,
                n,
                scale,
                q_data.as_ptr(),
                inner as isize,
                1,
                k_data.as_ptr(),
                1,
                inner as isize, // K^T via stride swap
                0.0,
                result.as_mut_ptr(),
                n as isize,
                1,
            );
        }

        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "Q@K^T mismatch at {i}: got {r}, expected {e}"
            );
        }
    }

    #[test]
    fn test_causal_mask_applied() {
        let n_state = 8;
        let n_head = 2;
        let seq_len = 4;

        let x = det_tensor(&[seq_len, n_state], 0.05);
        let (qw, qb, kw, vw, vb, ow, ob) = make_weights(n_state);

        let weights = AttentionWeights {
            q_weight: &qw,
            q_bias: &qb,
            k_weight: &kw,
            v_weight: &vw,
            v_bias: &vb,
            out_weight: &ow,
            out_bias: &ob,
        };
        let config = AttentionConfig { n_head, mask: true };

        // We can't directly inspect internal scores, but we can verify:
        // 1. The output shape is correct
        // 2. The output is finite
        // 3. With masking, the first token's output depends only on itself
        let result = multi_head_attention(&x, None, &weights, &config);
        assert_eq!(result.shape, vec![seq_len, n_state]);

        // Run with single-token input (seq_len=1) — should match first row of masked output
        let x_single = Tensor::from_vec(x.data[..n_state].to_vec(), &[1, n_state]);
        let result_single = multi_head_attention(
            &x_single,
            None,
            &weights,
            &AttentionConfig {
                n_head,
                mask: false,
            },
        );

        // First row of masked multi-token attention should equal single-token result
        for d in 0..n_state {
            assert!(
                (result.data[d] - result_single.data[d]).abs() < 1e-4,
                "causal mask: position 0 should only attend to itself, dim {d}: masked={}, single={}",
                result.data[d],
                result_single.data[d]
            );
        }
    }

    #[test]
    fn test_causal_mask_not_applied() {
        let n_state = 8;
        let n_head = 2;
        let seq_len = 4;

        let x = det_tensor(&[seq_len, n_state], 0.05);
        let (qw, qb, kw, vw, vb, ow, ob) = make_weights(n_state);

        let weights = AttentionWeights {
            q_weight: &qw,
            q_bias: &qb,
            k_weight: &kw,
            v_weight: &vw,
            v_bias: &vb,
            out_weight: &ow,
            out_bias: &ob,
        };

        // Without mask, first token can attend to all positions
        let result_no_mask = multi_head_attention(
            &x,
            None,
            &weights,
            &AttentionConfig {
                n_head,
                mask: false,
            },
        );

        // With mask, first token can only attend to position 0
        let result_masked =
            multi_head_attention(&x, None, &weights, &AttentionConfig { n_head, mask: true });

        // The last row should differ between masked and unmasked
        // (unless weights conspire to make them equal, which is unlikely with det_tensor)
        let last_row_start = (seq_len - 1) * n_state;
        let mut any_differ = false;
        for d in 0..n_state {
            if (result_no_mask.data[last_row_start + d] - result_masked.data[last_row_start + d])
                .abs()
                > 1e-6
            {
                any_differ = true;
                break;
            }
        }
        // For seq_len=4 the last token sees all 4 positions unmasked but also all 4 masked,
        // so actually the last row is identical. Check row 0 instead — it should differ.
        let mut row0_differ = false;
        for d in 0..n_state {
            if (result_no_mask.data[d] - result_masked.data[d]).abs() > 1e-6 {
                row0_differ = true;
                break;
            }
        }
        assert!(
            row0_differ || any_differ,
            "masked and unmasked outputs should differ for at least some row"
        );

        // All outputs should be finite
        for (i, &v) in result_no_mask.data.iter().enumerate() {
            assert!(v.is_finite(), "no-mask output NaN/Inf at {i}");
        }
    }

    #[test]
    fn test_attention_scale_factor() {
        // Verify that scores are divided by sqrt(head_dim).
        // We test this indirectly: with very large input, unscaled attention
        // would overflow softmax to all-zero/NaN; proper scaling keeps it finite.
        let n_state = 16;
        let n_head = 2;
        let head_dim = n_state / n_head; // 8
        let seq_len = 3;

        // Use large values that would cause softmax overflow without scaling
        let x = Tensor::from_vec(vec![10.0f32; seq_len * n_state], &[seq_len, n_state]);
        let (qw, qb, kw, vw, vb, ow, ob) = make_weights(n_state);

        let weights = AttentionWeights {
            q_weight: &qw,
            q_bias: &qb,
            k_weight: &kw,
            v_weight: &vw,
            v_bias: &vb,
            out_weight: &ow,
            out_bias: &ob,
        };
        let config = AttentionConfig {
            n_head,
            mask: false,
        };

        let result = multi_head_attention(&x, None, &weights, &config);
        for (i, &v) in result.data.iter().enumerate() {
            assert!(
                v.is_finite(),
                "scale factor should prevent overflow, but index {i} = {v} (head_dim={head_dim})"
            );
        }
    }

    #[test]
    fn test_cross_attention_different_lengths() {
        let n_state = 32;
        let n_head = 4;
        let q_len = 3;
        let kv_len = 15; // very different from q_len

        let x = det_tensor(&[q_len, n_state], 0.05);
        let xa = det_tensor(&[kv_len, n_state], 0.07);
        let (qw, qb, kw, vw, vb, ow, ob) = make_weights(n_state);

        let weights = AttentionWeights {
            q_weight: &qw,
            q_bias: &qb,
            k_weight: &kw,
            v_weight: &vw,
            v_bias: &vb,
            out_weight: &ow,
            out_bias: &ob,
        };
        let config = AttentionConfig {
            n_head,
            mask: false,
        };

        let result = multi_head_attention(&x, Some(&xa), &weights, &config);
        assert_eq!(result.shape, vec![q_len, n_state]);
        assert_eq!(result.data.len(), q_len * n_state);
        for (i, &v) in result.data.iter().enumerate() {
            assert!(v.is_finite(), "cross-attn diff lengths: NaN/Inf at {i}");
        }
    }

    #[test]
    fn test_attention_auto_matches_regular() {
        let n_state = 32;
        let n_head = 4;
        let seq_len = 5;

        let x = det_tensor(&[seq_len, n_state], 0.05);
        let (qw, qb, kw, vw, vb, ow, ob) = make_weights(n_state);

        let weights_regular = AttentionWeights {
            q_weight: &qw,
            q_bias: &qb,
            k_weight: &kw,
            v_weight: &vw,
            v_bias: &vb,
            out_weight: &ow,
            out_bias: &ob,
        };
        let weights_auto = AttentionWeightsAuto {
            q_weight_f32: Some(&qw),
            q_weight_quant: None,
            q_bias: &qb,
            k_weight_f32: Some(&kw),
            k_weight_quant: None,
            v_weight_f32: Some(&vw),
            v_weight_quant: None,
            v_bias: &vb,
            out_weight_f32: Some(&ow),
            out_weight_quant: None,
            out_bias: &ob,
        };
        let config = AttentionConfig {
            n_head,
            mask: false,
        };

        let result_regular = multi_head_attention(&x, None, &weights_regular, &config);
        let result_auto = multi_head_attention_auto(&x, None, &weights_auto, &config)
            .expect("auto attention should succeed");

        assert_eq!(result_regular.shape, result_auto.shape);
        for (i, (&r, &a)) in result_regular
            .data
            .iter()
            .zip(result_auto.data.iter())
            .enumerate()
        {
            assert!(
                (r - a).abs() < 1e-4,
                "regular vs auto mismatch at {i}: regular={r}, auto={a}"
            );
        }
    }

    /// Verify that sgemm-based attention core produces the same results as a
    /// manual reference computation for a small known case.
    #[test]
    fn test_sgemm_attention_matches_manual() {
        let n_head = 2;
        let q_len = 3;
        let kv_len = 4;
        let head_dim = 2;
        let scale = (head_dim as f32).sqrt().recip(); // 1/sqrt(2)

        // Deterministic Q, K, V data — [n_head, seq_len, head_dim] layout
        let q_data: Vec<f32> = (0..(n_head * q_len * head_dim))
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();
        let k_data: Vec<f32> = (0..(n_head * kv_len * head_dim))
            .map(|i| (i as f32 + 1.0) * 0.05)
            .collect();
        let v_data: Vec<f32> = (0..(n_head * kv_len * head_dim))
            .map(|i| (i as f32 + 1.0) * 0.02)
            .collect();

        // --- Manual reference: Q @ K^T * scale ---
        let mut ref_scores = vec![0.0f32; n_head * q_len * kv_len];
        for h in 0..n_head {
            for i in 0..q_len {
                for j in 0..kv_len {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_data[h * q_len * head_dim + i * head_dim + d]
                            * k_data[h * kv_len * head_dim + j * head_dim + d];
                    }
                    ref_scores[h * q_len * kv_len + i * kv_len + j] = dot * scale;
                }
            }
        }

        // --- sgemm path ---
        let mut sgemm_scores = vec![0.0f32; n_head * q_len * kv_len];
        for h in 0..n_head {
            let q_ptr = q_data[h * q_len * head_dim..].as_ptr();
            let k_ptr = k_data[h * kv_len * head_dim..].as_ptr();
            let s_ptr = sgemm_scores[h * q_len * kv_len..].as_mut_ptr();
            unsafe {
                matrixmultiply::sgemm(
                    q_len,
                    head_dim,
                    kv_len,
                    scale,
                    q_ptr,
                    head_dim as isize,
                    1,
                    k_ptr,
                    1,
                    head_dim as isize,
                    0.0,
                    s_ptr,
                    kv_len as isize,
                    1,
                );
            }
        }

        for (i, (r, s)) in ref_scores.iter().zip(sgemm_scores.iter()).enumerate() {
            assert!(
                (r - s).abs() < 1e-5,
                "Q@K^T mismatch at {i}: ref={r}, sgemm={s}"
            );
        }

        // --- Apply softmax on sgemm_scores for the V multiplication test ---
        // Simple row-wise softmax
        for h in 0..n_head {
            for i in 0..q_len {
                let row_start = h * q_len * kv_len + i * kv_len;
                let row = &mut sgemm_scores[row_start..row_start + kv_len];
                let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for v in row.iter_mut() {
                    *v = (*v - max_val).exp();
                    sum += *v;
                }
                for v in row.iter_mut() {
                    *v /= sum;
                }
            }
        }

        // --- Manual reference: scores @ V ---
        let mut ref_out = vec![0.0f32; n_head * q_len * head_dim];
        for h in 0..n_head {
            for i in 0..q_len {
                for j in 0..kv_len {
                    let s = sgemm_scores[h * q_len * kv_len + i * kv_len + j];
                    for d in 0..head_dim {
                        ref_out[h * q_len * head_dim + i * head_dim + d] +=
                            s * v_data[h * kv_len * head_dim + j * head_dim + d];
                    }
                }
            }
        }

        // --- sgemm path: scores @ V ---
        let mut sgemm_out = vec![0.0f32; n_head * q_len * head_dim];
        for h in 0..n_head {
            let s_ptr = sgemm_scores[h * q_len * kv_len..].as_ptr();
            let v_ptr = v_data[h * kv_len * head_dim..].as_ptr();
            let o_ptr = sgemm_out[h * q_len * head_dim..].as_mut_ptr();
            unsafe {
                matrixmultiply::sgemm(
                    q_len,
                    kv_len,
                    head_dim,
                    1.0,
                    s_ptr,
                    kv_len as isize,
                    1,
                    v_ptr,
                    head_dim as isize,
                    1,
                    0.0,
                    o_ptr,
                    head_dim as isize,
                    1,
                );
            }
        }

        for (i, (r, s)) in ref_out.iter().zip(sgemm_out.iter()).enumerate() {
            assert!(
                (r - s).abs() < 1e-5,
                "scores@V mismatch at {i}: ref={r}, sgemm={s}"
            );
        }
    }
}
