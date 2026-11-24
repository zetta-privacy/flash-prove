//! Compute backend for the prover's elementwise kernels.
//!
//! The prover's hot loops are all "flat field arrays, elementwise arithmetic, then
//! reduce". That shape is what a GPU is good at, but only above a size threshold:
//! one Metal dispatch costs ~180 µs on this machine (measured), so arrays below
//! `GPU_MIN` elements are always cheaper on the CPU.
//!
//! GPU use is **opt-in** (`FLASHPROVE_GPU=1`) so the test suite never needs a GPU.
//! The first GPU call validates itself against the CPU implementation and falls back
//! permanently if the two disagree.

use crate::field::{Ext, Fp};
use p3_field::PrimeField64;

#[cfg(target_os = "macos")]
pub mod metal;

/// Below this many elements the CPU wins (dispatch overhead dominates).
pub const GPU_MIN: usize = 1 << 18;

pub(crate) fn gpu_requested() -> bool {
    matches!(
        std::env::var("FLASHPROVE_GPU").as_deref(),
        Ok("1") | Ok("true")
    )
}

/// `Σ_i eq(point, i)·values[i]` — the MLE of `values` at `point`.
///
/// This is the kernel behind every cell and weight opening; at 2²² entries it costs
/// ~2.8 s on one core.
pub fn mle_eval(values: &[Fp], point: &[Ext]) -> Ext {
    #[cfg(target_os = "macos")]
    if gpu_requested() && values.len() >= GPU_MIN {
        if let Some(v) = metal::mle_eval(values, point) {
            return v;
        }
    }
    mle_eval_cpu(values, point)
}

/// Goldilocks modulus, as a plain integer (for the bisect checks).
pub const ORDER: u64 = 0xFFFF_FFFF_0000_0001;

/// One Goldilocks multiply on canonical `u64`s — the reference for the GPU kernel.
pub fn mul_cpu(a: u64, b: u64) -> u64 {
    (crate::field::Fp::new(a) * crate::field::Fp::new(b)).as_canonical_u64()
}

/// One sumcheck round: fold the low (LSB) variable out of `a` in place.
///
/// `a[j] = a[2j] + r·(a[2j+1] − a[2j])`, then `a` is truncated to half its length.
/// This is the inner loop of every linear claim and every GKR round.
pub fn fold(a: &mut Vec<Ext>, r: Ext) {
    let n = a.len();
    debug_assert!(n.is_power_of_two() && n >= 2);
    #[cfg(target_os = "macos")]
    if gpu_requested() && n >= GPU_MIN {
        if let Some(folded) = metal::fold(a, r) {
            *a = folded;
            return;
        }
    }
    fold_cpu(a, r);
}

/// The reference fold (and the fallback).
pub fn fold_cpu(a: &mut Vec<Ext>, r: Ext) {
    let half = a.len() / 2;
    for j in 0..half {
        a[j] = a[2 * j] + r * (a[2 * j + 1] - a[2 * j]);
    }
    a.truncate(half);
}

/// The reference implementation (and the fallback).
pub fn mle_eval_cpu(values: &[Fp], point: &[Ext]) -> Ext {
    debug_assert_eq!(values.len(), 1 << point.len());
    values
        .iter()
        .enumerate()
        .map(|(j, &v)| crate::field::eq(point, j) * Ext::from(v))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn fold_matches_the_definition() {
        let mut a: Vec<Ext> = (0..8u64).map(|i| Ext::from(Fp::new(i + 1))).collect();
        // a[j] = a[2j] + 3·(a[2j+1] − a[2j]) on [1..8] gives [4, 6, 8, 10].
        let want: Vec<Ext> = [4u64, 6, 8, 10].iter().map(|&i| Ext::from(Fp::new(i))).collect();
        let orig = a.clone();
        let r = Ext::from(Fp::new(3));
        fold_cpu(&mut a, r);
        assert_eq!(a, want);
        // And the definition, written out: a0 + r*(a1 - a0) per pair.
        for j in 0..4 {
            assert_eq!(a[j], orig[2 * j] + r * (orig[2 * j + 1] - orig[2 * j]));
        }
    }

    #[test]
    fn cpu_mle_matches_the_definition() {
        let values: Vec<Fp> = (0..8u64).map(|i| Fp::new(i + 1)).collect();
        let point: Vec<Ext> = (0..3u64).map(|i| Ext::from(Fp::new(5 + i))).collect();
        let want: Ext = values
            .iter()
            .enumerate()
            .map(|(j, &v)| crate::field::eq(&point, j) * Ext::from(v))
            .sum();
        assert_eq!(mle_eval_cpu(&values, &point), want);
        assert_ne!(want, Ext::ZERO);
    }
}
