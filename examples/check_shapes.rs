/// Shape-checking diagnostic for oxiwhisper.
///
/// Exercises tensor, mel, and FFT modules to verify that all operations
/// produce outputs with the expected shapes.
use oxiwhisper::fft;
use oxiwhisper::mel;
use oxiwhisper::tensor::Tensor;

struct Results {
    passed: usize,
    failed: usize,
}

impl Results {
    fn new() -> Self {
        Self {
            passed: 0,
            failed: 0,
        }
    }

    fn check(&mut self, name: &str, got: &[usize], expected: &[usize]) {
        if got == expected {
            println!("  [PASS] {name}: shape = {got:?}");
            self.passed += 1;
        } else {
            println!("  [FAIL] {name}: expected {expected:?}, got {got:?}");
            self.failed += 1;
        }
    }

    fn check_len(&mut self, name: &str, got: usize, expected: usize) {
        if got == expected {
            println!("  [PASS] {name}: len = {got}");
            self.passed += 1;
        } else {
            println!("  [FAIL] {name}: expected len {expected}, got {got}");
            self.failed += 1;
        }
    }

    fn summary(&self) {
        println!();
        println!(
            "=== Summary: {} passed, {} failed ===",
            self.passed, self.failed
        );
        if self.failed > 0 {
            std::process::exit(1);
        }
    }
}

fn check_tensor_ops(r: &mut Results) {
    println!("--- Tensor operations ---");

    // Tensor::zeros
    let t = Tensor::zeros(&[3, 4, 5]);
    r.check("zeros [3,4,5]", &t.shape, &[3, 4, 5]);
    r.check_len("zeros numel", t.numel(), 60);

    // Tensor::from_vec
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = Tensor::from_vec(data, &[2, 3, 4]);
    r.check("from_vec [2,3,4]", &t.shape, &[2, 3, 4]);

    // reshape
    let t2 = t.reshape(&[6, 4]);
    r.check("reshape [2,3,4] -> [6,4]", &t2.shape, &[6, 4]);
    r.check_len("reshape numel preserved", t2.numel(), 24);

    let t3 = t.reshape(&[24]);
    r.check("reshape [2,3,4] -> [24]", &t3.shape, &[24]);

    let t4 = t.reshape(&[2, 12]);
    r.check("reshape [2,3,4] -> [2,12]", &t4.shape, &[2, 12]);

    // transpose_2d
    let mat = Tensor::zeros(&[3, 7]);
    let mt = mat.transpose_2d();
    r.check("transpose_2d [3,7] -> [7,3]", &mt.shape, &[7, 3]);

    let mat2 = Tensor::zeros(&[1, 5]);
    let mt2 = mat2.transpose_2d();
    r.check("transpose_2d [1,5] -> [5,1]", &mt2.shape, &[5, 1]);

    // matmul
    let a = Tensor::zeros(&[4, 6]);
    let b = Tensor::zeros(&[6, 3]);
    let c = a.matmul(&b);
    r.check("matmul [4,6] x [6,3] -> [4,3]", &c.shape, &[4, 3]);

    let a2 = Tensor::zeros(&[1, 10]);
    let b2 = Tensor::zeros(&[10, 1]);
    let c2 = a2.matmul(&b2);
    r.check("matmul [1,10] x [10,1] -> [1,1]", &c2.shape, &[1, 1]);

    let a3 = Tensor::zeros(&[8, 5]);
    let b3 = Tensor::zeros(&[5, 12]);
    let c3 = a3.matmul(&b3);
    r.check("matmul [8,5] x [5,12] -> [8,12]", &c3.shape, &[8, 12]);

    // batched_matmul -- 3D
    let ba = Tensor::zeros(&[2, 4, 6]);
    let bb = Tensor::zeros(&[2, 6, 3]);
    let bc = ba.batched_matmul(&bb);
    r.check(
        "batched_matmul [2,4,6] x [2,6,3] -> [2,4,3]",
        &bc.shape,
        &[2, 4, 3],
    );

    // batched_matmul -- 4D
    let ba4 = Tensor::zeros(&[2, 3, 4, 5]);
    let bb4 = Tensor::zeros(&[2, 3, 5, 7]);
    let bc4 = ba4.batched_matmul(&bb4);
    r.check(
        "batched_matmul [2,3,4,5] x [2,3,5,7] -> [2,3,4,7]",
        &bc4.shape,
        &[2, 3, 4, 7],
    );

    // add -- same shape
    let x = Tensor::zeros(&[3, 4]);
    let y = Tensor::zeros(&[3, 4]);
    let z = x.add(&y);
    r.check("add [3,4] + [3,4] -> [3,4]", &z.shape, &[3, 4]);

    // softmax preserves shape
    let s = Tensor::zeros(&[2, 5]);
    let ss = s.softmax();
    r.check("softmax [2,5] -> [2,5]", &ss.shape, &[2, 5]);

    let s2 = Tensor::zeros(&[3, 4, 8]);
    let ss2 = s2.softmax();
    r.check("softmax [3,4,8] -> [3,4,8]", &ss2.shape, &[3, 4, 8]);

    // gelu preserves shape
    let g = Tensor::from_vec(vec![0.5, -0.3, 1.0, 0.0, 2.0, -1.0], &[2, 3]);
    let gg = g.gelu();
    r.check("gelu [2,3] -> [2,3]", &gg.shape, &[2, 3]);

    // layer_norm preserves shape
    let ln_input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let weight = Tensor::from_vec(vec![1.0, 1.0, 1.0], &[3]);
    let bias = Tensor::from_vec(vec![0.0, 0.0, 0.0], &[3]);
    let ln_out = ln_input.layer_norm(&weight, &bias, 1e-5);
    r.check("layer_norm [2,3] -> [2,3]", &ln_out.shape, &[2, 3]);

    let ln3 = Tensor::zeros(&[2, 4, 8]);
    let w3 = Tensor::from_vec(vec![1.0; 8], &[8]);
    let b3 = Tensor::from_vec(vec![0.0; 8], &[8]);
    let ln3_out = ln3.layer_norm(&w3, &b3, 1e-5);
    r.check("layer_norm [2,4,8] -> [2,4,8]", &ln3_out.shape, &[2, 4, 8]);

    // scale preserves shape
    let sc = Tensor::zeros(&[5, 3]);
    let sc2 = sc.scale(2.0);
    r.check("scale [5,3] -> [5,3]", &sc2.shape, &[5, 3]);

    // causal_mask preserves shape (must be square in last two dims)
    let cm = Tensor::zeros(&[4, 4]);
    let cm2 = cm.causal_mask();
    r.check("causal_mask [4,4] -> [4,4]", &cm2.shape, &[4, 4]);

    let cm3 = Tensor::zeros(&[2, 6, 6]);
    let cm4 = cm3.causal_mask();
    r.check("causal_mask [2,6,6] -> [2,6,6]", &cm4.shape, &[2, 6, 6]);
}

fn check_mel_shapes(r: &mut Results) {
    println!("--- Mel spectrogram shapes ---");

    let n_fft = mel::WHISPER_N_FFT;
    let hop = mel::WHISPER_HOP_LENGTH;
    let n_mels = mel::WHISPER_N_MELS;
    let sample_rate = mel::WHISPER_SAMPLE_RATE;
    let n_bins = n_fft / 2 + 1; // 201

    // Generate 1 second of synthetic audio (sine wave at 440 Hz)
    let duration_secs = 1.0_f32;
    let n_samples = (sample_rate as f32 * duration_secs) as usize;
    let audio: Vec<f32> = (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
        })
        .collect();

    // Fake mel filter bank: n_mels x n_bins, filled with small positive values
    let mel_filters: Vec<f32> = (0..n_mels * n_bins)
        .map(|i| ((i % 7) as f32 + 1.0) * 0.001)
        .collect();

    let mel_data = mel::log_mel_spectrogram_unpadded(&audio, &mel_filters)
        .expect("mel filter bank must be well formed");

    // Expected number of frames for 1 second of audio
    let expected_n_frames = (n_samples / hop).clamp(1, sample_rate * 30 / hop);
    let expected_len = n_mels * expected_n_frames;
    r.check_len(
        &format!("mel spectrogram total len ({n_mels} x {expected_n_frames})"),
        mel_data.len(),
        expected_len,
    );

    // Verify we can reshape into [n_mels, n_frames]
    let n_frames = mel_data.len() / n_mels;
    r.check_len(
        "mel spectrogram n_frames from 1s audio",
        n_frames,
        expected_n_frames,
    );

    let mel_tensor = Tensor::from_vec(mel_data, &[n_mels, n_frames]);
    r.check(
        &format!("mel tensor shape [{n_mels}, {n_frames}]"),
        &mel_tensor.shape,
        &[n_mels, n_frames],
    );

    // Test with 3 seconds of audio
    let n_samples_3s = sample_rate * 3;
    let audio_3s: Vec<f32> = (0..n_samples_3s)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * 1000.0 * t).sin() * 0.3
        })
        .collect();

    let mel_data_3s = mel::log_mel_spectrogram_unpadded(&audio_3s, &mel_filters)
        .expect("mel filter bank must be well formed");
    let expected_frames_3s = (n_samples_3s / hop).clamp(1, sample_rate * 30 / hop);
    let expected_len_3s = n_mels * expected_frames_3s;
    r.check_len(
        &format!("mel spectrogram 3s total len ({n_mels} x {expected_frames_3s})"),
        mel_data_3s.len(),
        expected_len_3s,
    );

    // Test with very short audio (less than one window)
    let short_audio: Vec<f32> = vec![0.1; 100];
    let mel_short = mel::log_mel_spectrogram_unpadded(&short_audio, &mel_filters)
        .expect("mel filter bank must be well formed");
    let short_frames = mel_short.len() / n_mels;
    // Even very short audio should produce at least 1 frame
    let short_pass = short_frames >= 1 && mel_short.len() == n_mels * short_frames;
    if short_pass {
        println!("  [PASS] mel spectrogram short audio: {n_mels} x {short_frames}");
        r.passed += 1;
    } else {
        println!(
            "  [FAIL] mel spectrogram short audio: unexpected len {} (frames={short_frames})",
            mel_short.len()
        );
        r.failed += 1;
    }
}

fn check_fft_shapes(r: &mut Results) {
    println!("--- FFT power_spectrum shapes ---");

    // Standard Whisper FFT size
    let n_fft = mel::WHISPER_N_FFT; // 400
    let expected_bins = n_fft / 2 + 1; // 201

    // Input exactly n_fft samples
    let input: Vec<f32> = (0..n_fft).map(|i| (i as f32 * 0.1).sin()).collect();
    let ps = fft::power_spectrum(&input, n_fft);
    r.check_len(
        &format!("power_spectrum(n={n_fft}, n_fft={n_fft}) -> {expected_bins} bins"),
        ps.len(),
        expected_bins,
    );

    // Input shorter than n_fft (zero-padded internally)
    let short_input: Vec<f32> = vec![1.0, 0.5, -0.5, -1.0];
    let ps_short = fft::power_spectrum(&short_input, n_fft);
    r.check_len(
        &format!("power_spectrum(n=4, n_fft={n_fft}) -> {expected_bins} bins"),
        ps_short.len(),
        expected_bins,
    );

    // Smaller n_fft
    let n_fft_small = 64;
    let expected_bins_small = n_fft_small / 2 + 1; // 33
    let input_small: Vec<f32> = (0..n_fft_small).map(|i| (i as f32 * 0.2).cos()).collect();
    let ps_small = fft::power_spectrum(&input_small, n_fft_small);
    r.check_len(
        &format!(
            "power_spectrum(n={n_fft_small}, n_fft={n_fft_small}) -> {expected_bins_small} bins"
        ),
        ps_small.len(),
        expected_bins_small,
    );

    // Larger n_fft
    let n_fft_large = 1024;
    let expected_bins_large = n_fft_large / 2 + 1; // 513
    let input_large: Vec<f32> = (0..n_fft_large).map(|i| (i as f32 * 0.05).sin()).collect();
    let ps_large = fft::power_spectrum(&input_large, n_fft_large);
    r.check_len(
        &format!(
            "power_spectrum(n={n_fft_large}, n_fft={n_fft_large}) -> {expected_bins_large} bins"
        ),
        ps_large.len(),
        expected_bins_large,
    );

    // Verify power spectrum values are non-negative
    let all_nonneg = ps.iter().all(|&v| v >= 0.0);
    if all_nonneg {
        println!("  [PASS] power_spectrum values are all non-negative");
        r.passed += 1;
    } else {
        println!("  [FAIL] power_spectrum contains negative values");
        r.failed += 1;
    }
}

fn main() {
    println!("oxiwhisper shape-checking diagnostics");
    println!("=====================================");
    println!();

    let mut r = Results::new();

    check_tensor_ops(&mut r);
    println!();
    check_mel_shapes(&mut r);
    println!();
    check_fft_shapes(&mut r);

    r.summary();
}
