//! Advanced shape-checking example for oxiwhisper.
//!
//! Tests attention and linear layer shapes including:
//! - Linear layer shapes
//! - Conv1d shapes with varying stride and padding
//! - Multi-head attention (self-attention and cross-attention)
//! - cat_seq operation
//! - Row extraction

use oxiwhisper::attention::{AttentionConfig, AttentionWeights, multi_head_attention};
use oxiwhisper::linear::{conv1d, linear};
use oxiwhisper::tensor::Tensor;

macro_rules! check_shape {
    ($pass:expr, $fail:expr, $label:expr, $tensor:expr, $expected:expr $(,)?) => {{
        let tensor = &$tensor;
        let expected: &[usize] = $expected;
        if tensor.shape == expected {
            println!(
                "[PASS] {}: shape {:?} == {:?}",
                $label, tensor.shape, expected
            );
            $pass += 1;
        } else {
            println!(
                "[FAIL] {}: shape {:?} != expected {:?}",
                $label, tensor.shape, expected
            );
            $fail += 1;
        }
    }};
}

macro_rules! check_bool {
    ($pass:expr, $fail:expr, $label:expr, $cond:expr, $fail_msg:expr $(,)?) => {{
        if $cond {
            println!("[PASS] {}", $label);
            $pass += 1;
        } else {
            println!("[FAIL] {}: {}", $label, $fail_msg);
            $fail += 1;
        }
    }};
}

fn main() {
    let mut pass_count = 0usize;
    let mut fail_count = 0usize;

    // ===============================================================
    // 1. Linear layer shapes
    // ===============================================================
    println!("\n=== Linear layer shapes ===");

    // 1a. Basic 2D input: [batch=4, in_f=8] x W[8, 16] -> [4, 16]
    {
        let input = Tensor::zeros(&[4, 8]);
        let weight = Tensor::zeros(&[8, 16]);
        let output = linear(&input, &weight, None);
        check_shape!(
            pass_count,
            fail_count,
            "linear 2D no bias",
            &output,
            &[4, 16]
        );
    }

    // 1b. With bias
    {
        let input = Tensor::zeros(&[4, 8]);
        let weight = Tensor::zeros(&[8, 16]);
        let bias = Tensor::zeros(&[16]);
        let output = linear(&input, &weight, Some(&bias));
        check_shape!(
            pass_count,
            fail_count,
            "linear 2D with bias",
            &output,
            &[4, 16]
        );
    }

    // 1c. Single-row input (GEMV path, batch < 4)
    {
        let input = Tensor::zeros(&[1, 32]);
        let weight = Tensor::zeros(&[32, 64]);
        let output = linear(&input, &weight, None);
        check_shape!(
            pass_count,
            fail_count,
            "linear GEMV path (batch=1)",
            &output,
            &[1, 64]
        );
    }

    // 1d. Larger batch (BLAS-3 path, batch >= 4)
    {
        let input = Tensor::zeros(&[16, 64]);
        let weight = Tensor::zeros(&[64, 128]);
        let bias = Tensor::zeros(&[128]);
        let output = linear(&input, &weight, Some(&bias));
        check_shape!(
            pass_count,
            fail_count,
            "linear BLAS-3 path (batch=16)",
            &output,
            &[16, 128]
        );
    }

    // 1e. Square weight matrix (common in attention projections)
    {
        let n_state = 256;
        let input = Tensor::zeros(&[10, n_state]);
        let weight = Tensor::zeros(&[n_state, n_state]);
        let output = linear(&input, &weight, None);
        check_shape!(
            pass_count,
            fail_count,
            "linear square weight (attention-like)",
            &output,
            &[10, 256]
        );
    }

    // ===============================================================
    // 2. Conv1d shapes
    // ===============================================================
    println!("\n=== Conv1d shapes ===");

    // 2a. Basic conv1d: stride=1, padding=0
    //     input [1, 80, 3000], weight [3, 80, 512], bias [512]
    //     out_len = (3000 + 0 - 3) / 1 + 1 = 2998
    {
        let input = Tensor::zeros(&[1, 80, 3000]);
        let weight = Tensor::zeros(&[3, 80, 512]);
        let bias = Tensor::zeros(&[512]);
        let output = conv1d(&input, &weight, &bias, 1, 0);
        check_shape!(
            pass_count,
            fail_count,
            "conv1d stride=1 pad=0",
            &output,
            &[1, 512, 2998]
        );
    }

    // 2b. Conv1d with padding=1, stride=1
    //     input [1, 80, 3000], weight [3, 80, 512], bias [512]
    //     out_len = (3000 + 2*1 - 3) / 1 + 1 = 3000
    {
        let input = Tensor::zeros(&[1, 80, 3000]);
        let weight = Tensor::zeros(&[3, 80, 512]);
        let bias = Tensor::zeros(&[512]);
        let output = conv1d(&input, &weight, &bias, 1, 1);
        check_shape!(
            pass_count,
            fail_count,
            "conv1d stride=1 pad=1 (same)",
            &output,
            &[1, 512, 3000]
        );
    }

    // 2c. Conv1d with stride=2 (Whisper encoder first conv)
    //     input [1, 80, 3000], weight [3, 80, 512], bias [512]
    //     out_len = (3000 + 2*1 - 3) / 2 + 1 = 1500
    {
        let input = Tensor::zeros(&[1, 80, 3000]);
        let weight = Tensor::zeros(&[3, 80, 512]);
        let bias = Tensor::zeros(&[512]);
        let output = conv1d(&input, &weight, &bias, 2, 1);
        check_shape!(
            pass_count,
            fail_count,
            "conv1d stride=2 pad=1 (downsample)",
            &output,
            &[1, 512, 1500]
        );
    }

    // 2d. Conv1d second stage: stride=2 on already-downsampled
    //     input [1, 512, 1500], weight [3, 512, 512], bias [512]
    //     out_len = (1500 + 2*1 - 3) / 2 + 1 = 750
    {
        let input = Tensor::zeros(&[1, 512, 1500]);
        let weight = Tensor::zeros(&[3, 512, 512]);
        let bias = Tensor::zeros(&[512]);
        let output = conv1d(&input, &weight, &bias, 2, 1);
        check_shape!(
            pass_count,
            fail_count,
            "conv1d second stage stride=2",
            &output,
            &[1, 512, 750]
        );
    }

    // 2e. Multi-batch conv1d
    {
        let input = Tensor::zeros(&[2, 16, 100]);
        let weight = Tensor::zeros(&[5, 16, 32]);
        let bias = Tensor::zeros(&[32]);
        // out_len = (100 + 2*2 - 5) / 1 + 1 = 100
        let output = conv1d(&input, &weight, &bias, 1, 2);
        check_shape!(
            pass_count,
            fail_count,
            "conv1d multi-batch pad=2",
            &output,
            &[2, 32, 100]
        );
    }

    // ===============================================================
    // 3. Multi-head attention shapes
    // ===============================================================
    println!("\n=== Multi-head attention shapes ===");

    let n_state = 64;
    let n_head = 4;

    // 3a. Self-attention (no cross-attention source)
    //     x: [seq_len=10, n_state=64] -> output: [10, 64]
    {
        let seq_len = 10;
        let x = Tensor::zeros(&[seq_len, n_state]);
        let q_w = Tensor::zeros(&[n_state, n_state]);
        let q_b = Tensor::zeros(&[n_state]);
        let k_w = Tensor::zeros(&[n_state, n_state]);
        let v_w = Tensor::zeros(&[n_state, n_state]);
        let v_b = Tensor::zeros(&[n_state]);
        let out_w = Tensor::zeros(&[n_state, n_state]);
        let out_b = Tensor::zeros(&[n_state]);

        let weights = AttentionWeights {
            q_weight: &q_w,
            q_bias: &q_b,
            k_weight: &k_w,
            v_weight: &v_w,
            v_bias: &v_b,
            out_weight: &out_w,
            out_bias: &out_b,
        };
        let config = AttentionConfig { n_head, mask: true };
        let output = multi_head_attention(&x, None, &weights, &config);
        check_shape!(
            pass_count,
            fail_count,
            "self-attention (masked)",
            &output,
            &[seq_len, n_state]
        );
    }

    // 3b. Self-attention without mask
    {
        let seq_len = 20;
        let x = Tensor::zeros(&[seq_len, n_state]);
        let q_w = Tensor::zeros(&[n_state, n_state]);
        let q_b = Tensor::zeros(&[n_state]);
        let k_w = Tensor::zeros(&[n_state, n_state]);
        let v_w = Tensor::zeros(&[n_state, n_state]);
        let v_b = Tensor::zeros(&[n_state]);
        let out_w = Tensor::zeros(&[n_state, n_state]);
        let out_b = Tensor::zeros(&[n_state]);

        let weights = AttentionWeights {
            q_weight: &q_w,
            q_bias: &q_b,
            k_weight: &k_w,
            v_weight: &v_w,
            v_bias: &v_b,
            out_weight: &out_w,
            out_bias: &out_b,
        };
        let config = AttentionConfig {
            n_head,
            mask: false,
        };
        let output = multi_head_attention(&x, None, &weights, &config);
        check_shape!(
            pass_count,
            fail_count,
            "self-attention (unmasked)",
            &output,
            &[seq_len, n_state]
        );
    }

    // 3c. Cross-attention: x: [5, 64], xa: [750, 64] -> [5, 64]
    {
        let dec_len = 5;
        let enc_len = 750;
        let x = Tensor::zeros(&[dec_len, n_state]);
        let xa = Tensor::zeros(&[enc_len, n_state]);
        let q_w = Tensor::zeros(&[n_state, n_state]);
        let q_b = Tensor::zeros(&[n_state]);
        let k_w = Tensor::zeros(&[n_state, n_state]);
        let v_w = Tensor::zeros(&[n_state, n_state]);
        let v_b = Tensor::zeros(&[n_state]);
        let out_w = Tensor::zeros(&[n_state, n_state]);
        let out_b = Tensor::zeros(&[n_state]);

        let weights = AttentionWeights {
            q_weight: &q_w,
            q_bias: &q_b,
            k_weight: &k_w,
            v_weight: &v_w,
            v_bias: &v_b,
            out_weight: &out_w,
            out_bias: &out_b,
        };
        let config = AttentionConfig {
            n_head,
            mask: false,
        };
        let output = multi_head_attention(&x, Some(&xa), &weights, &config);
        check_shape!(
            pass_count,
            fail_count,
            "cross-attention (dec=5, enc=750)",
            &output,
            &[dec_len, n_state],
        );
    }

    // 3d. Cross-attention with single decoder token (autoregressive step)
    {
        let dec_len = 1;
        let enc_len = 750;
        let x = Tensor::zeros(&[dec_len, n_state]);
        let xa = Tensor::zeros(&[enc_len, n_state]);
        let q_w = Tensor::zeros(&[n_state, n_state]);
        let q_b = Tensor::zeros(&[n_state]);
        let k_w = Tensor::zeros(&[n_state, n_state]);
        let v_w = Tensor::zeros(&[n_state, n_state]);
        let v_b = Tensor::zeros(&[n_state]);
        let out_w = Tensor::zeros(&[n_state, n_state]);
        let out_b = Tensor::zeros(&[n_state]);

        let weights = AttentionWeights {
            q_weight: &q_w,
            q_bias: &q_b,
            k_weight: &k_w,
            v_weight: &v_w,
            v_bias: &v_b,
            out_weight: &out_w,
            out_bias: &out_b,
        };
        let config = AttentionConfig {
            n_head,
            mask: false,
        };
        let output = multi_head_attention(&x, Some(&xa), &weights, &config);
        check_shape!(
            pass_count,
            fail_count,
            "cross-attention single token (dec=1, enc=750)",
            &output,
            &[dec_len, n_state],
        );
    }

    // 3e. Self-attention with 8 heads
    {
        let n_state_lg = 128;
        let n_head_lg = 8;
        let seq_len = 12;
        let x = Tensor::zeros(&[seq_len, n_state_lg]);
        let q_w = Tensor::zeros(&[n_state_lg, n_state_lg]);
        let q_b = Tensor::zeros(&[n_state_lg]);
        let k_w = Tensor::zeros(&[n_state_lg, n_state_lg]);
        let v_w = Tensor::zeros(&[n_state_lg, n_state_lg]);
        let v_b = Tensor::zeros(&[n_state_lg]);
        let out_w = Tensor::zeros(&[n_state_lg, n_state_lg]);
        let out_b = Tensor::zeros(&[n_state_lg]);

        let weights = AttentionWeights {
            q_weight: &q_w,
            q_bias: &q_b,
            k_weight: &k_w,
            v_weight: &v_w,
            v_bias: &v_b,
            out_weight: &out_w,
            out_bias: &out_b,
        };
        let config = AttentionConfig {
            n_head: n_head_lg,
            mask: true,
        };
        let output = multi_head_attention(&x, None, &weights, &config);
        check_shape!(
            pass_count,
            fail_count,
            "self-attention 8 heads (d=128)",
            &output,
            &[seq_len, n_state_lg],
        );
    }

    // ===============================================================
    // 4. cat_seq operation
    // ===============================================================
    println!("\n=== cat_seq shapes ===");

    // 4a. Basic 2D cat_seq: [3, 8] cat [5, 8] -> [8, 8]
    {
        let a = Tensor::zeros(&[3, 8]);
        let b = Tensor::zeros(&[5, 8]);
        let c = a.cat_seq(&b);
        check_shape!(
            pass_count,
            fail_count,
            "cat_seq 2D [3,8]+[5,8]",
            &c,
            &[8, 8]
        );
    }

    // 4b. Growing decoder sequence: simulate appending one token
    //     [seq=10, d=64] cat [1, 64] -> [11, 64]
    {
        let seq = Tensor::zeros(&[10, 64]);
        let token = Tensor::zeros(&[1, 64]);
        let grown = seq.cat_seq(&token);
        check_shape!(
            pass_count,
            fail_count,
            "cat_seq decoder grow [10,64]+[1,64]",
            &grown,
            &[11, 64]
        );
    }

    // 4c. 3D cat_seq with batch dim: [2, 4, 16] cat [2, 3, 16] -> [2, 7, 16]
    {
        let a = Tensor::zeros(&[2, 4, 16]);
        let b = Tensor::zeros(&[2, 3, 16]);
        let c = a.cat_seq(&b);
        check_shape!(
            pass_count,
            fail_count,
            "cat_seq 3D batched [2,4,16]+[2,3,16]",
            &c,
            &[2, 7, 16]
        );
    }

    // 4d. Sequential appending (simulating multiple decoder steps)
    {
        let mut seq = Tensor::zeros(&[1, 64]);
        for step in 0..5 {
            let new_tok = Tensor::zeros(&[1, 64]);
            seq = seq.cat_seq(&new_tok);
            let expected_len = step + 2;
            check_shape!(
                pass_count,
                fail_count,
                &format!("cat_seq step {step} -> seq_len={expected_len}"),
                &seq,
                &[expected_len, 64],
            );
        }
    }

    // ===============================================================
    // 5. Row extraction
    // ===============================================================
    println!("\n=== Row extraction shapes ===");

    // 5a. Extract row from 2D tensor: [4, 8] row [2] -> [8]
    {
        let t = Tensor::zeros(&[4, 8]);
        let r = t.row(&[2]);
        check_shape!(
            pass_count,
            fail_count,
            "row from 2D [4,8] idx=[2]",
            &r,
            &[8]
        );
    }

    // 5b. Extract row from 3D tensor: [2, 5, 16] row [1, 3] -> [16]
    {
        let t = Tensor::zeros(&[2, 5, 16]);
        let r = t.row(&[1, 3]);
        check_shape!(
            pass_count,
            fail_count,
            "row from 3D [2,5,16] idx=[1,3]",
            &r,
            &[16]
        );
    }

    // 5c. Extract first and last row
    {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = Tensor::from_vec(data, &[4, 6]);
        let first = t.row(&[0]);
        let last = t.row(&[3]);
        check_shape!(pass_count, fail_count, "row first from [4,6]", &first, &[6]);
        check_shape!(pass_count, fail_count, "row last from [4,6]", &last, &[6]);

        // Verify actual data content
        let first_ok = first.data == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let last_ok = last.data == [18.0, 19.0, 20.0, 21.0, 22.0, 23.0];
        check_bool!(
            pass_count,
            fail_count,
            "row data content verified",
            first_ok && last_ok,
            format!("first={:?} last={:?}", first.data, last.data)
        );
    }

    // 5d. Extract from 4D tensor: [2, 3, 4, 8] row [1, 2, 0] -> [8]
    {
        let t = Tensor::zeros(&[2, 3, 4, 8]);
        let r = t.row(&[1, 2, 0]);
        check_shape!(
            pass_count,
            fail_count,
            "row from 4D [2,3,4,8] idx=[1,2,0]",
            &r,
            &[8]
        );
    }

    // ===============================================================
    // Summary
    // ===============================================================
    println!("\n=== Summary ===");
    println!("Passed: {pass_count}");
    println!("Failed: {fail_count}");
    if fail_count == 0 {
        println!("All shape checks passed!");
    } else {
        std::process::exit(1);
    }
}
