//! FFT module backed by OxiFFT.
//! Provides Complex type and FFT functions used throughout the codebase.

#[derive(Clone, Copy, Debug)]
pub struct Complex {
    pub re: f32,
    pub im: f32,
}

impl Complex {
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    pub fn magnitude_squared(&self) -> f32 {
        self.re * self.re + self.im * self.im
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

/// In-place complex FFT using OxiFFT.
/// Input length must be a power of 2.
pub fn fft(input: &mut [Complex]) {
    let n = input.len();
    assert!(n.is_power_of_two(), "FFT length must be power of 2");

    // Convert our Complex to oxifft::Complex
    let oxi_input: Vec<oxifft::Complex<f32>> = input
        .iter()
        .map(|c| oxifft::Complex::new(c.re, c.im))
        .collect();

    let oxi_output = oxifft::fft(&oxi_input);

    // Copy results back
    for (dst, src) in input.iter_mut().zip(oxi_output.iter()) {
        dst.re = src.re;
        dst.im = src.im;
    }
}

/// Compute power spectrum (magnitude squared) of real input using OxiFFT's rfft.
/// Input is zero-padded to next power of 2 if needed.
/// Returns n_fft/2 + 1 values.
pub fn power_spectrum(input: &[f32], n_fft: usize) -> Vec<f32> {
    let padded_len = n_fft.next_power_of_two();

    // Build zero-padded real buffer
    let mut buf = vec![0.0f32; padded_len];
    let copy_len = input.len().min(padded_len);
    buf[..copy_len].copy_from_slice(&input[..copy_len]);

    // Use rfft for real-to-complex (returns padded_len/2 + 1 values)
    let spectrum = oxifft::rfft::<f32>(&buf);

    // Return n_fft/2 + 1 magnitude-squared values
    let n_bins = n_fft / 2 + 1;
    spectrum[..n_bins]
        .iter()
        .map(|c| c.re * c.re + c.im * c.im)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_dc() {
        let mut input = vec![Complex::new(1.0, 0.0); 4];
        fft(&mut input);
        assert!((input[0].re - 4.0).abs() < 1e-5);
        assert!(input[0].im.abs() < 1e-5);
        for item in input.iter().take(4).skip(1) {
            assert!(item.re.abs() < 1e-5);
            assert!(item.im.abs() < 1e-5);
        }
    }

    #[test]
    fn test_fft_impulse() {
        let mut input = vec![Complex::new(0.0, 0.0); 8];
        input[0] = Complex::new(1.0, 0.0);
        fft(&mut input);
        // All bins should have magnitude 1
        for c in &input {
            assert!((c.magnitude_squared() - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_power_spectrum() {
        let input = vec![1.0, 0.0, -1.0, 0.0];
        let ps = power_spectrum(&input, 4);
        assert_eq!(ps.len(), 3); // n_fft/2 + 1
    }
}
