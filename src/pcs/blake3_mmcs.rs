//! BLAKE3 Merkle hasher and compressor over Goldilocks, using the SIMD batch path.
//!
//! `p3-blake3` hashes bytes; the WHIR MMCS hashes field elements. This bridges
//! the two: serialise Goldilocks elements to little-endian bytes, hash with BLAKE3
//! (using its batched SIMD `hash_many` for many leaves), and pack the 128-bit
//! digest back into two Goldilocks. The digest stays field-typed, so the stacked
//! WHIR challenger is unchanged.

use crate::field::Fp;
use p3_blake3::Blake3;
use p3_field::PrimeField64;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};

/// Goldilocks modulus `p = 2^64 − 2^32 + 1`.
const P: u64 = 0xFFFF_FFFF_0000_0001;

/// Map the first 16 bytes of a BLAKE3 digest into two Goldilocks elements.
#[inline]
fn digest_to_fp2(d: &[u8; 32]) -> [Fp; 2] {
    let a = u64::from_le_bytes(d[0..8].try_into().unwrap()) % P;
    let b = u64::from_le_bytes(d[8..16].try_into().unwrap()) % P;
    [Fp::new(a), Fp::new(b)]
}

#[inline]
fn push_fp(acc: &mut Vec<u8>, x: Fp) {
    acc.extend_from_slice(&x.as_canonical_u64().to_le_bytes());
}

/// Leaf hasher: `Goldilocks → [Goldilocks; 2]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Blake3Hasher;

impl CryptographicHasher<Fp, [Fp; 2]> for Blake3Hasher {
    const LANES: usize = <Blake3 as CryptographicHasher<u8, [u8; 32]>>::LANES;

    fn hash_iter<I: IntoIterator<Item = Fp>>(&self, input: I) -> [Fp; 2] {
        let mut bytes = Vec::new();
        for x in input {
            push_fp(&mut bytes, x);
        }
        digest_to_fp2(&Blake3.hash_slice(&bytes))
    }

    fn hash_slice(&self, input: &[Fp]) -> [Fp; 2] {
        let mut bytes = Vec::with_capacity(input.len() * 8);
        for &x in input {
            push_fp(&mut bytes, x);
        }
        digest_to_fp2(&Blake3.hash_slice(&bytes))
    }

    fn hash_many(&self, input: &[Fp], out: &mut [[Fp; 2]]) {
        if out.is_empty() {
            return;
        }
        let mut bytes = Vec::with_capacity(input.len() * 8);
        for &x in input {
            push_fp(&mut bytes, x);
        }
        let mut digests = vec![[0u8; 32]; out.len()];
        Blake3.hash_many(&bytes, &mut digests);
        for (o, d) in out.iter_mut().zip(digests.iter()) {
            *o = digest_to_fp2(d);
        }
    }
}

/// Node compressor: two digests → one digest.
#[derive(Clone, Copy, Debug, Default)]
pub struct Blake3Compress;

impl PseudoCompressionFunction<[Fp; 2], 2> for Blake3Compress {
    const LANES: usize = <Blake3 as CryptographicHasher<u8, [u8; 32]>>::LANES;

    fn compress(&self, input: [[Fp; 2]; 2]) -> [Fp; 2] {
        let mut bytes = Vec::with_capacity(32);
        for d in input {
            for x in d {
                push_fp(&mut bytes, x);
            }
        }
        digest_to_fp2(&Blake3.hash_slice(&bytes))
    }

    fn compress_many(&self, inputs: &[[[Fp; 2]; 2]], out: &mut [[Fp; 2]]) {
        assert_eq!(inputs.len(), out.len());
        if out.is_empty() {
            return;
        }
        let mut bytes = Vec::with_capacity(inputs.len() * 32);
        for grp in inputs {
            for d in grp {
                for &x in d {
                    push_fp(&mut bytes, x);
                }
            }
        }
        let mut digests = vec![[0u8; 32]; out.len()];
        Blake3.hash_many(&bytes, &mut digests);
        for (o, d) in out.iter_mut().zip(digests.iter()) {
            *o = digest_to_fp2(d);
        }
    }
}
