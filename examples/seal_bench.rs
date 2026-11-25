//! Where the seal's time goes: the Reed–Solomon LDE (NTT) versus the leaf hashing.
//!
//! The commit is these two passes over the codeword, so measuring them separately says
//! which one a GPU port would pay for.
//!
//! Run: cargo run --release --example seal_bench

use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_symmetric::CryptographicHasher;
use std::time::Instant;

use flashprove::pcs::blake3_mmcs::Blake3Hasher;

fn main() {
    println!(
        "{:<6} {:>12} {:>12} {:>10} {:>10}",
        "size", "lde (s)", "hash (s)", "lde %", "leaf ns"
    );

    let mut s = 0x243F_6A88_85A3_08D3u64;
    for log_n in [22usize, 24, 26] {
        let n = 1usize << log_n;
        let vals: Vec<F> = (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                F::new(s)
            })
            .collect();

        // Rate 1/2: `n` evaluations become a codeword of `2n`.
        let mat = RowMajorMatrix::new(vals, 1);
        let dft = Radix2DitParallel::<F>::default();

        let t0 = Instant::now();
        let lde = dft.coset_lde_batch(mat, 1, F::ONE);
        let lde_s = t0.elapsed().as_secs_f64();
        let codeword = lde.to_row_major_matrix().values;

        // Leaves are pairs of codeword elements (one BLAKE3 call each).
        let leaves = codeword.len() / 2;
        let mut out = vec![[F::ZERO; 2]; leaves];
        let t0 = Instant::now();
        Blake3Hasher.hash_many(&codeword, &mut out);
        let hash_s = t0.elapsed().as_secs_f64();

        println!(
            "2^{:<4} {:>12.2} {:>12.2} {:>9.0}% {:>10.1}",
            log_n,
            lde_s,
            hash_s,
            100.0 * lde_s / (lde_s + hash_s),
            hash_s * 1e9 / leaves as f64
        );
        assert_ne!(out[0][0], F::ZERO);
    }
}
