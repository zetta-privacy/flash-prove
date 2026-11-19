//! Masked Reed–Solomon row encoding (paper Definition 4.2).

use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_goldilocks::Goldilocks as Fp;

/// Parameters of a single encoded row.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct RowParams {
    /// Number of message entries (ℓ).
    pub message_len: usize,
    /// Number of random masking coefficients (t, the opening budget).
    pub num_masks: usize,
    /// log2 of the evaluation-domain size |D|.
    pub log_domain_size: usize,
}

impl RowParams {
    /// |D| = 2^log_domain_size evaluation points.
    pub const fn domain_size(&self) -> usize {
        1usize << self.log_domain_size
    }

    /// Code rate ρ = (ℓ + t) / |D|.
    pub const fn rate(&self) -> f64 {
        (self.message_len + self.num_masks) as f64 / self.domain_size() as f64
    }

    /// The paper's default configuration (Section 3.3).
    pub const PAPER: RowParams = RowParams {
        message_len: 402_653_184, // ℓ
        num_masks: 44_040_191,    // t (opening budget)
        log_domain_size: 32,      // |D| = 2^32
    };
}

/// Encode one row as a Reed–Solomon codeword.
///
/// The polynomial is
///   f(X) = r_0 + r_1·X + … + r_{t−1}·X^{t−1} + m_0·X^t + … + m_{ℓ−1}·X^{t+ℓ−1}
/// with the `t` masks in the lowest coefficients and the message in the next
/// `ℓ`. The codeword is `f` evaluated on the multiplicative subgroup of order
/// `2^log_domain_size`, returned in natural order.
pub fn encode(params: &RowParams, message: &[Fp], masks: &[Fp]) -> Vec<Fp> {
    assert_eq!(message.len(), params.message_len, "message length");
    assert_eq!(masks.len(), params.num_masks, "mask length");

    let n = params.domain_size();
    assert!(
        params.message_len + params.num_masks <= n,
        "degree {} exceeds domain {}",
        params.message_len + params.num_masks,
        n
    );

    // Coefficients: masks in the lowest `t` positions, message next, rest zero.
    let mut coeffs = vec![Fp::new(0); n];
    coeffs[..params.num_masks].copy_from_slice(masks);
    coeffs[params.num_masks..params.num_masks + params.message_len].copy_from_slice(message);

    Radix2Dit::<Fp>::default().dft(coeffs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_idft_roundtrip_recovers_message_and_masks() {
        let params = RowParams {
            message_len: 5,
            num_masks: 3,
            log_domain_size: 3, // |D| = 8
        };

        let message: Vec<Fp> = (0..5).map(|i| Fp::new(i as u64 + 1)).collect();
        let masks: Vec<Fp> = (0..3).map(|i| Fp::new(i as u64 + 10)).collect();

        let codeword = encode(&params, &message, &masks);
        assert_eq!(codeword.len(), 8);

        let coeffs = Radix2Dit::<Fp>::default().idft(codeword);
        assert_eq!(&coeffs[..3], &masks[..]);
        assert_eq!(&coeffs[3..8], &message[..]);
    }
}
