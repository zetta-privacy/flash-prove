//! A GPU-backed [`TwoAdicSubgroupDft`], used as the FFT implementation of the
//! stacked-WHIR commitment.
//!
//! Only [`TwoAdicSubgroupDft::dft_batch`] has to be implemented — every other method
//! (including the coset LDE) derives from it. The GPU path runs `log2(h)` radix-2 DIF
//! layers, each a single dispatch with one thread per butterfly and no threadgroup
//! memory; the result is in bit-reversed order, which is exactly the layout
//! `BitReversedMatrixView` wraps, so the committed codeword is identical to the one
//! p3's CPU implementation produces.
//!
//! With `FLASHPROVE_GPU` unset (or the GPU unavailable) this is `Radix2DitParallel`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_matrix::bitrev::{BitReversalPerm, BitReversedMatrixView};
use p3_matrix::Matrix;
use p3_field::{PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;

use crate::field::Fp;

// Twiddle tables (`root^0 … root^(n/2-1)`), cached per transform size. Building one
// for `n = 2^27` is 67 M multiplies, so it must not happen per seal.
thread_local! {
    static TWIDDLES: RefCell<HashMap<usize, Arc<Vec<Fp>>>> = RefCell::new(HashMap::new());
}

fn twiddles(log_h: usize) -> Arc<Vec<Fp>> {
    TWIDDLES.with(|c| {
        let mut c = c.borrow_mut();
        c.entry(log_h)
            .or_insert_with(|| {
                let root = Fp::two_adic_generator(log_h);
                let mut v = Vec::with_capacity(1 << log_h.saturating_sub(1));
                let mut acc = Fp::from_u64(1);
                for _ in 0..(1usize << log_h.saturating_sub(1)) {
                    v.push(acc);
                    acc *= root;
                }
                Arc::new(v)
            })
            .clone()
    })
}

#[derive(Clone, Default, Debug)]
pub struct MetalDft;

impl TwoAdicSubgroupDft<Fp> for MetalDft {
    type Evaluations = BitReversedMatrixView<RowMajorMatrix<Fp>>;

    fn dft_batch(&self, mat: RowMajorMatrix<Fp>) -> Self::Evaluations {
        let h = mat.height();
        let w = mat.width();
        if h.is_power_of_two() && h >= 2 {
            let log_h = h.trailing_zeros() as usize;
            if let Some(v) = crate::backend::dft(&mat.values, h, w, &twiddles(log_h)) {
                return BitReversalPerm::new_view(RowMajorMatrix::new(v, w));
            }
        }
        Radix2DitParallel::<Fp>::default().dft_batch(mat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::PrimeCharacteristicRing;

    /// The GPU path is off by default, so this checks the fallback wiring: our type must
    /// produce exactly what `Radix2DitParallel` does.
    #[test]
    fn dft_matches_the_reference_implementation() {
        for log_h in [1usize, 2, 5, 8] {
            let h = 1usize << log_h;
            let w = 3usize;
            let values: Vec<Fp> = (0..(h * w) as u64).map(|i| Fp::new(i * 3 + 1)).collect();
            let mat = RowMajorMatrix::new(values, w);

            let mine = MetalDft.dft_batch(mat.clone()).to_row_major_matrix().values;
            let reference = Radix2DitParallel::<Fp>::default()
                .dft_batch(mat)
                .to_row_major_matrix()
                .values;
            assert_eq!(mine, reference, "log_h = {log_h}");
        }
    }

    /// The twiddle table must be the powers of the transform's root.
    #[test]
    fn twiddles_are_powers_of_the_root() {
        let log_h = 6usize;
        let tw = twiddles(log_h);
        let root = Fp::two_adic_generator(log_h);
        assert_eq!(tw.len(), 1 << (log_h - 1));
        let mut acc = Fp::ONE;
        for &t in tw.iter() {
            assert_eq!(t, acc);
            acc *= root;
        }
    }
}
