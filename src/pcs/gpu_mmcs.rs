//! BLAKE3 Merkle hasher/compressor with a GPU path, for the stacked-WHIR commitment.
//!
//! Same contract as [`crate::pcs::blake3_mmcs`]: field elements are serialised as
//! little-endian bytes, hashed with BLAKE3, and the first 16 bytes of the digest are
//! packed into two Goldilocks elements (`% p`). Our leaves are two field elements
//! (16 B) and nodes two digests (32 B), so every hash is a **single block** — which is
//! what the GPU kernel implements.
//!
//! With `FLASHPROVE_GPU` unset (or the GPU unavailable, or a digest input wider than one
//! block) every method delegates to the CPU implementation, so results and behaviour are
//! unchanged.

use crate::field::Fp;
use crate::pcs::blake3_mmcs::{Blake3Compress, Blake3Hasher};
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};

/// Leaf hasher: `Goldilocks → [Goldilocks; 2]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuBlake3Hasher;

impl CryptographicHasher<Fp, [Fp; 2]> for GpuBlake3Hasher {
    const LANES: usize = <Blake3Hasher as CryptographicHasher<Fp, [Fp; 2]>>::LANES;

    fn hash_iter<I: IntoIterator<Item = Fp>>(&self, input: I) -> [Fp; 2] {
        Blake3Hasher.hash_iter(input)
    }

    fn hash_slice(&self, input: &[Fp]) -> [Fp; 2] {
        Blake3Hasher.hash_slice(input)
    }

    fn hash_many(&self, input: &[Fp], out: &mut [[Fp; 2]]) {
        if out.is_empty() {
            return;
        }
        if input.len() % out.len() == 0 {
            let words = input.len() / out.len();
            if let Some(v) = crate::backend::hash_many(input, words) {
                out.copy_from_slice(&v);
                return;
            }
        }
        Blake3Hasher.hash_many(input, out);
    }
}

/// Node compressor: two digests → one digest.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuBlake3Compress;

impl PseudoCompressionFunction<[Fp; 2], 2> for GpuBlake3Compress {
    const LANES: usize = <Blake3Compress as PseudoCompressionFunction<[Fp; 2], 2>>::LANES;

    fn compress(&self, input: [[Fp; 2]; 2]) -> [Fp; 2] {
        Blake3Compress.compress(input)
    }

    fn compress_many(&self, inputs: &[[[Fp; 2]; 2]], out: &mut [[Fp; 2]]) {
        if out.is_empty() {
            return;
        }
        // Flatten to four field elements per node and hash in one batch.
        let mut flat: Vec<Fp> = Vec::with_capacity(inputs.len() * 4);
        for grp in inputs {
            flat.extend_from_slice(&grp[0]);
            flat.extend_from_slice(&grp[1]);
        }
        if let Some(v) = crate::backend::hash_many(&flat, 4) {
            out.copy_from_slice(&v);
            return;
        }
        Blake3Compress.compress_many(inputs, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The GPU is off in tests, so this pins the fallback: byte-identical to the CPU
    /// hasher for both leaves and nodes.
    #[test]
    fn fallback_matches_the_cpu_hasher() {
        let mut s = 0x1234_5678_9ABC_DEF0u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let values: Vec<Fp> = (0..64).map(|_| Fp::new(rnd())).collect();

        let mut mine = vec![[Fp::default(); 2]; 32];
        GpuBlake3Hasher.hash_many(&values, &mut mine);
        let mut theirs = vec![[Fp::default(); 2]; 32];
        Blake3Hasher.hash_many(&values, &mut theirs);
        assert_eq!(mine, theirs);

        let groups: Vec<[[Fp; 2]; 2]> = mine.chunks(2).map(|c| [c[0], c[1]]).collect();
        let mut m2 = vec![[Fp::default(); 2]; groups.len()];
        GpuBlake3Compress.compress_many(&groups, &mut m2);
        let mut t2 = vec![[Fp::default(); 2]; groups.len()];
        Blake3Compress.compress_many(&groups, &mut t2);
        assert_eq!(m2, t2);
    }
}
