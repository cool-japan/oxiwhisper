use crate::attention::{self, AttentionConfig, AttentionWeightsAuto};
use crate::linear;
use crate::model::ModelData;
use crate::tensor::Tensor;

/// Run the Whisper audio encoder.
/// Input: mel spectrogram [n_mels, n_frames]  (n_frames <= 3000, actual audio length)
/// Output: encoded features [seq_len, n_audio_state]  where seq_len = n_frames/2
pub fn encode(mel: &Tensor, model: &ModelData) -> Result<Tensor, String> {
    let hp = &model.hparams;
    let n_mels = hp.n_mels;
    let n_audio_state = hp.n_audio_state;
    let n_audio_ctx = hp.n_audio_ctx;
    let n_audio_layer = hp.n_audio_layer;
    let n_audio_head = hp.n_audio_head;

    let n_frames = mel.data.len() / n_mels;
    let mel_3d = mel.reshape(&[1, n_mels, n_frames]);

    // Conv1: [1, n_mels, n_frames] -> [1, n_audio_state, n_frames]
    let conv1_w = model.get("encoder.conv1.weight")?;
    let conv1_b = model.get("encoder.conv1.bias")?;
    let mut x = linear::conv1d(&mel_3d, conv1_w, conv1_b, 1, 1);
    x.gelu_inplace();

    // Conv2: [1, n_audio_state, n_frames] -> [1, n_audio_state, seq_len]  (stride=2)
    let conv2_w = model.get("encoder.conv2.weight")?;
    let conv2_b = model.get("encoder.conv2.bias")?;
    x = linear::conv1d(&x, conv2_w, conv2_b, 2, 1);
    x.gelu_inplace();

    // seq_len may be less than n_audio_ctx when audio is shorter than 30s
    let seq_len = x.shape[2].min(n_audio_ctx);

    // Transpose [1, n_audio_state, seq_len] -> [seq_len, n_audio_state]
    let mut transposed = vec![0.0f32; seq_len * n_audio_state];
    for t in 0..seq_len {
        for s in 0..n_audio_state {
            transposed[t * n_audio_state + s] = x.data[s * x.shape[2] + t];
        }
    }
    let mut x = Tensor::from_vec(transposed, &[seq_len, n_audio_state]);

    // Add positional embedding (use only the first seq_len positions)
    let pos_emb = model.get("encoder.positional_embedding")?;
    // pos_emb shape: [n_audio_ctx, n_audio_state] -- take first seq_len rows
    let pos_slice = Tensor::from_vec(
        pos_emb.data[..seq_len * n_audio_state].to_vec(),
        &[seq_len, n_audio_state],
    );
    x.add_inplace(&pos_slice);

    // Transformer blocks
    for i in 0..n_audio_layer {
        let prefix = format!("encoder.blocks.{i}");

        let ln1_w = model.get(&format!("{prefix}.attn_ln.weight"))?;
        let ln1_b = model.get(&format!("{prefix}.attn_ln.bias"))?;
        let normed = x.layer_norm(ln1_w, ln1_b, 1e-5);

        let q_name = format!("{prefix}.attn.query.weight");
        let k_name = format!("{prefix}.attn.key.weight");
        let v_name = format!("{prefix}.attn.value.weight");
        let out_name = format!("{prefix}.attn.out.weight");
        let weights = AttentionWeightsAuto {
            q_weight_f32: model.try_get(&q_name),
            q_weight_quant: model.get_quantized(&q_name),
            q_bias: model.get(&format!("{prefix}.attn.query.bias"))?,
            k_weight_f32: model.try_get(&k_name),
            k_weight_quant: model.get_quantized(&k_name),
            v_weight_f32: model.try_get(&v_name),
            v_weight_quant: model.get_quantized(&v_name),
            v_bias: model.get(&format!("{prefix}.attn.value.bias"))?,
            out_weight_f32: model.try_get(&out_name),
            out_weight_quant: model.get_quantized(&out_name),
            out_bias: model.get(&format!("{prefix}.attn.out.bias"))?,
        };
        let config = AttentionConfig {
            n_head: n_audio_head,
            mask: false,
        };
        let attn_out = attention::multi_head_attention_auto(&normed, None, &weights, &config)?;
        x.add_inplace(&attn_out);

        let ln2_w = model.get(&format!("{prefix}.mlp_ln.weight"))?;
        let ln2_b = model.get(&format!("{prefix}.mlp_ln.bias"))?;
        let normed = x.layer_norm(ln2_w, ln2_b, 1e-5);

        let fc1_name = format!("{prefix}.mlp.0.weight");
        let fc1_b = model.get(&format!("{prefix}.mlp.0.bias"))?;
        let mut h = linear::linear_auto(
            &normed,
            model.try_get(&fc1_name),
            model.get_quantized(&fc1_name),
            Some(fc1_b),
        )?;
        h.gelu_inplace();

        let fc2_name = format!("{prefix}.mlp.2.weight");
        let fc2_b = model.get(&format!("{prefix}.mlp.2.bias"))?;
        let ffn_out = linear::linear_auto(
            &h,
            model.try_get(&fc2_name),
            model.get_quantized(&fc2_name),
            Some(fc2_b),
        )?;
        x.add_inplace(&ffn_out);
    }

    let ln_w = model.get("encoder.ln_post.weight")?;
    let ln_b = model.get("encoder.ln_post.bias")?;
    Ok(x.layer_norm(ln_w, ln_b, 1e-5))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelData;
    use crate::tensor::Tensor;
    use crate::test_utils::generate_synthetic_model;

    fn load_test_model() -> ModelData {
        let path = generate_synthetic_model();
        let model = ModelData::load(&path).expect("failed to load synthetic model");
        let _ = std::fs::remove_file(&path);
        model
    }

    #[test]
    fn test_encode_output_shape() {
        let model = load_test_model();
        let n_mels = model.hparams.n_mels;
        let n_audio_state = model.hparams.n_audio_state;
        let n_frames = 100;

        let mel = Tensor::from_vec(vec![0.01f32; n_mels * n_frames], &[n_mels, n_frames]);
        let output = encode(&mel, &model).expect("encode should succeed");

        let expected_seq_len = n_frames / 2;
        assert_eq!(output.shape.len(), 2, "output should be 2D");
        assert_eq!(
            output.shape[0], expected_seq_len,
            "seq_len should be n_frames/2 = {expected_seq_len}"
        );
        assert_eq!(
            output.shape[1], n_audio_state,
            "second dim should be n_audio_state = {n_audio_state}"
        );
    }

    #[test]
    fn test_encode_no_nan() {
        let model = load_test_model();
        let n_mels = model.hparams.n_mels;
        let n_frames = 100;

        let mel = Tensor::from_vec(vec![0.01f32; n_mels * n_frames], &[n_mels, n_frames]);
        let output = encode(&mel, &model).expect("encode should succeed");

        for (i, &val) in output.data.iter().enumerate() {
            assert!(
                val.is_finite(),
                "output[{i}] = {val} is not finite (NaN or Inf)"
            );
        }
    }

    #[test]
    fn test_encode_short_audio() {
        let model = load_test_model();
        let n_mels = model.hparams.n_mels;
        let n_audio_state = model.hparams.n_audio_state;
        let n_frames = 4;

        let mel = Tensor::from_vec(vec![0.01f32; n_mels * n_frames], &[n_mels, n_frames]);
        let output = encode(&mel, &model).expect("encode should succeed for short audio");

        let expected_seq_len = n_frames / 2; // 2
        assert_eq!(output.shape[0], expected_seq_len);
        assert_eq!(output.shape[1], n_audio_state);
    }
}
