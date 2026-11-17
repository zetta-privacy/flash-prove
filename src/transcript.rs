//! Fiat–Shamir transcript (merlin) and proof-of-work position sampling
//! (paper Construction 4.13).

use sha2::{Digest, Sha256};

/// The underlying Fiat–Shamir transcript (merlin).
pub use merlin::Transcript;

/// Derive an OOD seed from a Merkle root by two queries (Construction D.11):
/// the first query holds (rt, dg_Z, table sizes), the second the first answer's
/// chaining value. `dg_Z` is the digest of the configuration.
pub fn seed_from_root(rt: &[u8; 32], dg_z: &[u8; 32], sizes: &[u8]) -> [u8; 32] {
    let mut t = Transcript::new(b"FlashProve-ood");
    t.append_message(b"root", rt);
    t.append_message(b"dg", dg_z);
    t.append_message(b"sizes", sizes);
    let mut seed = [0u8; 32];
    t.challenge_bytes(b"ood-seed", &mut seed);
    seed
}

/// Hash `a || b` with SHA-256.
fn sha256_pair(a: &[u8], b: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(a);
    h.update(b);
    h.finalize().into()
}

/// Count leading zero bits in a 32-byte hash.
fn leading_zero_bits(h: &[u8; 32]) -> u32 {
    let mut zeros = 0u32;
    for &byte in h.iter() {
        if byte == 0 {
            zeros += 8;
        } else {
            zeros += byte.leading_zeros();
            break;
        }
    }
    zeros
}

/// Map a hash to a position in `[0, domain_size)`.
fn position_from(h: &[u8; 32], domain_size: usize) -> usize {
    let v = u64::from_le_bytes(h[..8].try_into().unwrap());
    (v % domain_size as u64) as usize
}

/// Grind a proof-of-work nonce for a seed, requiring `bits` leading zero bits.
/// Returns the nonce and the sampled position in `[0, domain_size)`.
pub fn grind(seed: &[u8; 32], domain_size: usize, bits: u32) -> (u64, usize) {
    let mut nonce = 0u64;
    loop {
        let h = sha256_pair(seed, &nonce.to_le_bytes());
        if leading_zero_bits(&h) >= bits {
            return (nonce, position_from(&h, domain_size));
        }
        nonce += 1;
    }
}

/// Verify a nonce against a seed. Returns the position if the nonce meets the
/// `bits` work requirement, otherwise `None`.
pub fn verify(seed: &[u8; 32], nonce: u64, domain_size: usize, bits: u32) -> Option<usize> {
    let h = sha256_pair(seed, &nonce.to_le_bytes());
    if leading_zero_bits(&h) < bits {
        return None;
    }
    Some(position_from(&h, domain_size))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grind_and_verify_roundtrip() {
        let seed = [7u8; 32];
        let domain = 1usize << 12; // 4096
        let bits = 8; // small for fast tests

        let (nonce, pos) = grind(&seed, domain, bits);
        assert!(pos < domain);
        assert_eq!(verify(&seed, nonce, domain, bits), Some(pos));
    }

    #[test]
    fn grind_meets_work_requirement() {
        let seed = [7u8; 32];
        let (nonce, _) = grind(&seed, 4096, 8);
        let h = sha256_pair(&seed, &nonce.to_le_bytes());
        assert!(leading_zero_bits(&h) >= 8);
    }

    #[test]
    fn verify_rejects_insufficient_work() {
        let seed = [7u8; 32];
        // No real SHA-256 output has 256 leading zeros, so this always fails.
        assert_eq!(verify(&seed, 0, 4096, 256), None);
    }
}
