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
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

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

/// The linear-claim sumcheck with GPU-resident operands. `None` when the GPU is off or
/// the arrays are too small; the caller then runs its own loop.
pub fn linear_sumcheck(
    hf: &[Ext],
    vf: &[Ext],
    tr: &mut crate::pcs::transcript::Transcript,
) -> Option<(Vec<crate::field::sumcheck::RoundPoly>, Vec<Ext>)> {
    #[cfg(target_os = "macos")]
    if gpu_requested() {
        if let Some(v) = metal::linear_sumcheck(hf, vf, tr) {
            return Some(v);
        }
    }
    let _ = (hf, vf, tr);
    None
}

/// One BLAKE3 digest per `words` field elements. Our leaves are 2 elements (16 B) and
/// nodes 4 (32 B), so each digest input is a single 64-byte block. `None` when the GPU is
/// off or the batch too small to pay for a dispatch.
pub fn hash_many(input: &[Fp], words: usize) -> Option<Vec<[Fp; 2]>> {
    #[cfg(target_os = "macos")]
    if gpu_requested() && input.len() >= GPU_MIN {
        if let Some(v) = metal::hash_many(input, words) {
            return Some(v);
        }
    }
    let _ = (input, words);
    None
}

/// `dft_batch` of `h × w` (row-major, `h` a power of two): `log2(h)` DIF layers on the
/// GPU. `tw` is `root^0 … root^(h/2-1)`. `None` when the GPU is off or the transform is
/// too small to pay for a dispatch — the caller keeps its CPU implementation then.
/// Measured on an M1 Pro: **1.0x** — no win, so the CPU transform stays. Two causes, both
/// fixable: `log2(h)` dispatches per transform (~180 us each), and the middle layers read
/// `tw[j*stride]`, which is scattered by `stride` — uncoalesced. The fix is a compact
/// per-layer twiddle table (`tw_layer[j] = tw[j*stride]`, built on the GPU) so every layer
/// reads contiguously; the kernel is kept in `backend::metal` for that rewrite.
/// `pcs::metal_dft::MetalDft` is correct either way (it matches `Radix2DitParallel`
/// bit-for-bit) and is the seam that rewrite plugs into.
#[allow(dead_code)]
pub fn dft(values: &[Fp], h: usize, w: usize, tw: &[Fp]) -> Option<Vec<Fp>> {
    // Measured: 1.1x at 2^18, 2.5x at 2^20, 4.8x at 2^22 — so only worth it from 2^20 up.
    #[cfg(target_os = "macos")]
    if gpu_requested() && h >= (1 << 20) {
        if let Some(v) = metal::dft(values, h, w, tw) {
            return Some(v);
        }
    }
    let _ = (values, h, w, tw);
    None
}

/// `p[k] = Σ_n eq[n]·w[n·2^lm + k]` — the contraction a matmul claim runs over its
/// weight (or activation) operand. `eq` holds `eq(r, n)` for the `2^ln` rows.
///
/// Measured on an M1 Pro: no better than the CPU (1.0× at 2^11×2^11, 0.6× below), because
/// the natural GPU shape here is one thread per column with a `2^ln`-long serial
/// accumulate chain, which leaves the machine idle. A tiled version (parallel over
/// `(n, k)` plus a per-column reduction) is the way to make it pay; until then the CPU
/// loop stays. The kernel is kept in `backend::metal` for that follow-up.
#[allow(dead_code)]
pub fn contract(w: &[Fp], eq: &[Ext], lm: usize) -> Option<Vec<Ext>> {
    #[cfg(target_os = "macos")]
    if gpu_requested() && w.len() >= GPU_MIN {
        if let Some(v) = metal::contract(w, eq, lm) {
            return Some(v);
        }
    }
    let _ = (w, eq, lm);
    None
}

/// Invert every element with **one** field inversion (Montgomery's trick):
/// prefix products, one inverse, one backward pass — `3n` multiplies instead of `n`
/// full inversions (each of which is an exponentiation).
///
/// Zeros are passed through as zeros (they never appear in practice: the digit and
/// fingerprint tables are shifted to be zero-free).
pub fn batch_invert(a: &[Fp]) -> Vec<Fp> {
    let n = a.len();
    let mut out = vec![Fp::ONE; n];
    if n == 0 {
        return out;
    }
    let mut prefix = vec![Fp::ONE; n];
    let mut acc = Fp::ONE;
    for (i, &x) in a.iter().enumerate() {
        prefix[i] = acc;
        acc *= x;
    }
    let mut inv = acc.inverse();
    for i in (0..n).rev() {
        out[i] = inv * prefix[i];
        inv *= a[i];
    }
    out
}

/// The reference `n`-inversions version, for tests and fallbacks.
pub fn batch_invert_naive(a: &[Fp]) -> Vec<Fp> {
    a.iter().map(|&x| x.inverse()).collect()
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
/// Measured on an M1 Pro: the GPU is 0.6–1.0× here (the fold is memory-bound, and a
/// round trip costs more than the fold itself), so this one stays on the CPU. Only
/// `mle_eval`, which is `O(m)` per element on the CPU, wins on the GPU (19–43×).
pub fn fold(a: &mut Vec<Ext>, r: Ext) {
    debug_assert!(a.len().is_power_of_two() && a.len() >= 2);
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
    fn batch_invert_matches_the_naive_one() {
        let a: Vec<Fp> = (1..200u64).map(|i| Fp::new(i * 7919)).collect();
        assert_eq!(batch_invert(&a), batch_invert_naive(&a));
        // And it really is the inverse.
        for (x, y) in a.iter().zip(batch_invert(&a)) {
            assert_eq!(*x * y, Fp::ONE);
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
