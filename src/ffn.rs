//! FFN arithmetization: the algebraic (linear + bilinear) backbone of the
//! feed-forward / MoE subgraph, verified succinctly via the inner-product
//! argument (matmul, weighted-sum) and the MLE linear-claim bridge (pointwise
//! linear ops) and a degree-3 sumcheck (pointwise product).
//!
//! The nonlinear *lookup* ops (SiLU, the matmul `requant` rounding, softmax,
//! top-k) are a separate concern and are *not* handled here — they need a
//! log-derivative / plookup argument (see the "lookups" phase). What this module
//! covers is the exact part the IPA + linear claim + degree-3 sumcheck already
//! check:
//!
//!   - matmul `y = W·x` (the *linear* accumulation, before any rounding),
//!   - pointwise `add` (`out[i] = a[i] + b[i]`), a *linear* op,
//!   - pointwise `mul` (`out[i] = a[i] * b[i]`), a *bilinear* op,
//!   - MoE `weighted_sum` (`out[s,j] = Σ_e choice[s,e] · term_e[s,j]`).
//!
//! The split is principled: linear ops are checkable by a single MLE evaluation
//! (the linear claim), while bilinear/trilinear ops need an inner-product or
//! sumcheck reduction.

use crate::field::{Ext, Fp};
use crate::fri::{Challenger, Pcs};
use crate::ipa::ipa_verify;
use crate::protocol::eq;
use crate::sumcheck::{mle_eval, prove3, verify3};
use crate::zk::{mle_value, verify_mle};
use p3_field::PrimeCharacteristicRing;

/// The next power of two `>= n` (the IPA folds the contraction dimension, which
/// requires power-of-two lengths).
fn next_pow2(n: usize) -> usize {
    n.next_power_of_two()
}

fn to_ext(v: &[Fp]) -> Vec<Ext> {
    v.iter().map(|&x| Ext::from(x)).collect()
}

/// Verify a linear matmul `y = W·x` (no requant): `y_i = Σ_j W[i,j] · x[j]`.
///
/// - `w` is the weight matrix, row-major, `out × in`.
/// - `x` is the input vector (length `in`).
/// - `y` is the output vector (length `out`).
///
/// The `out` inner products are batched with a random linear combination
/// `yc = Σ_i r_i·y_i` and `wc[j] = Σ_i r_i·W[i,j]`, reducing the matmul to a
/// *single* inner product `⟨wc, x⟩ = yc`, which [`ipa_verify`] checks. The
/// contraction dimension is zero-padded to a power of two (padding contributes
/// `0·0` to the sum, so it is exact).
pub fn verify_matmul(
    w: &[Fp],
    x: &[Fp],
    y: &[Fp],
    out: usize,
    mut sample: impl FnMut() -> Fp,
) -> bool {
    let in_ = x.len();
    assert_eq!(w.len(), out * in_, "weight matrix size");
    assert_eq!(y.len(), out, "output size");

    let padded = next_pow2(in_);

    // Random linear combination over the `out` outputs.
    let mut wc = vec![Fp::ZERO; padded];
    let mut yc = Fp::ZERO;
    for i in 0..out {
        let r = sample();
        yc += r * y[i];
        for j in 0..in_ {
            wc[j] += r * w[i * in_ + j];
        }
    }
    let mut xp = vec![Fp::ZERO; padded];
    xp[..in_].copy_from_slice(x);

    ipa_verify(&wc, &xp, yc, sample)
}

/// Verify a pointwise `out[i] = a[i] + b[i]` at a random point `point`, using the
/// MLE linear-claim bridge (each evaluation is checked via [`verify_mle`]).
/// Addition is linear, so `out̃(r) = ã(r) + b̃(r)` holds at every point.
pub fn verify_add(
    pcs: &Pcs,
    challenger: &Challenger,
    a: &[Fp],
    b: &[Fp],
    out: &[Fp],
    point: &[Ext],
) -> bool {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    let ca = mle_value(a, point);
    let cb = mle_value(b, point);
    let co = mle_value(out, point);
    if co != ca + cb {
        return false;
    }
    verify_mle(pcs, challenger, a, point, ca)
        && verify_mle(pcs, challenger, b, point, cb)
        && verify_mle(pcs, challenger, out, point, co)
}

/// Verify a pointwise `out[i] = a[i] * b[i]` via a degree-3 sumcheck.
///
/// The product's multilinear extension is *not* the product of the MLEs, so a
/// single-point identity `out̃(r) = ã(r)·b̃(r)` would be wrong. Instead we check
/// `out̃(r) = Σ_i eq(r,i)·a_i·b_i`, a sum over the cube of the cubic `eq·a·b`,
/// with a degree-3 sumcheck. (Arithmetic-level: the final single-point
/// evaluations of `a`, `b` wire to the linear claim in the succinct layer.)
pub fn verify_mul(
    a: &[Fp],
    b: &[Fp],
    out: &[Fp],
    r: &[Ext],
    sample: impl FnMut(usize) -> Ext,
) -> bool {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    assert!(
        a.len().is_power_of_two(),
        "mul requires power-of-two length"
    );
    let n = a.len().trailing_zeros() as usize;

    let a = to_ext(a);
    let b = to_ext(b);
    let out = to_ext(out);

    // g[i] = eq(r, i) · a[i] · b[i].
    let mut g = vec![Ext::ZERO; 1 << n];
    for i in 0..(1 << n) {
        g[i] = eq(r, i) * a[i] * b[i];
    }
    // claimed = out̃(r) = Σ_i eq(r,i)·out[i].
    let claimed = mle_eval(&out, r);
    let (rounds, _final, r_fold) = prove3(&g, n, sample);
    verify3(&g, n, &rounds, &r_fold, claimed)
}

/// Verify a MoE `weighted_sum`: `out[s,j] = Σ_e choice[s,e] · term_e[s,j]`, where
/// `choice` is the top-k indicator (`[seq × experts]`, `0`/`1`) and `terms` is
/// `[experts × seq × hidden]`.
///
/// For each sequence position `s` this is the matmul `out[s,:] = choice[s,:] ·
/// terms[:,s,:]` (contraction over `experts`), so it reduces to a per-`s`
/// [`verify_matmul`] (an IPA). `choice` being `0`/`1` is separately constrained
/// by the top-k argument.
pub fn verify_weighted_sum(
    choice: &[Fp], // [seq × experts]
    terms: &[Fp],  // [experts × seq × hidden]
    out: &[Fp],    // [seq × hidden]
    seq: usize,
    experts: usize,
    hidden: usize,
    mut sample: impl FnMut() -> Fp,
) -> bool {
    assert_eq!(choice.len(), seq * experts);
    assert_eq!(terms.len(), experts * seq * hidden);
    assert_eq!(out.len(), seq * hidden);
    for s in 0..seq {
        // Build the matmul as `out[s,:] = choice[s,:] · terms[:,s,:]`, i.e.
        // `w` is `hidden × experts` with `w[j,e] = terms[e,s,j]`, `x` is the
        // `choice[s,:]` row (length `experts`), `y` is `out[s,:]` (length `hidden`).
        let mut w = vec![Fp::ZERO; hidden * experts];
        for e in 0..experts {
            for j in 0..hidden {
                w[j * experts + e] = terms[e * seq * hidden + s * hidden + j];
            }
        }
        let choice_s: Vec<Fp> = (0..experts).map(|e| choice[s * experts + e]).collect();
        let out_s: Vec<Fp> = (0..hidden).map(|j| out[s * hidden + j]).collect();
        if !verify_matmul(&w, &choice_s, &out_s, hidden, &mut sample) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::from_u64;

    fn f(x: i64) -> Fp {
        if x >= 0 {
            Fp::new(x as u64)
        } else {
            -Fp::new((-x) as u64)
        }
    }

    fn e(x: i64) -> Ext {
        if x >= 0 {
            Ext::from(from_u64(x as u64))
        } else {
            -Ext::from(from_u64((-x) as u64))
        }
    }

    /// Deterministic Fiat–Shamir sampler for the IPA (base-field challenges).
    fn sampler() -> impl FnMut() -> Fp {
        let mut c = 0u64;
        move || {
            c += 1;
            Fp::new(c)
        }
    }

    /// Deterministic sampler for the sumcheck (extension-field challenges).
    fn sampler_ext() -> impl FnMut(usize) -> Ext {
        move |i| Ext::from(from_u64(i as u64 + 1))
    }

    #[test]
    fn matmul_ipa_honest() {
        // W = [[1,2],[3,4],[5,6],[7,8]] (4×2), x = [9,10].
        // y = W·x = [1·9+2·10, 3·9+4·10, 5·9+6·10, 7·9+8·10] = [29, 67, 105, 143].
        let w = [1, 2, 3, 4, 5, 6, 7, 8].map(f);
        let x = [9, 10].map(f);
        let y = [29, 67, 105, 143].map(f);
        assert!(verify_matmul(&w, &x, &y, 4, sampler()));
    }

    #[test]
    fn matmul_ipa_rejects_wrong_output() {
        let w = [1, 2, 3, 4, 5, 6, 7, 8].map(f);
        let x = [9, 10].map(f);
        let y = [29, 67, 105, 144].map(f); // last entry wrong: 143 → 144
        assert!(!verify_matmul(&w, &x, &y, 4, sampler()));
    }

    #[test]
    fn matmul_ipa_pads_contraction_dim() {
        // in = 3 (not a power of two). W is 2×3, x is 3.
        // y0 = 1·1+2·2+3·3 = 14, y1 = 4·1+5·2+6·3 = 32.
        let w = [1, 2, 3, 4, 5, 6].map(f);
        let x = [1, 2, 3].map(f);
        let y = [14, 32].map(f);
        assert!(verify_matmul(&w, &x, &y, 2, sampler()));
    }

    #[test]
    fn add_mle_honest() {
        let (pcs, challenger) = crate::fri::new_pcs(1, 10);
        let a = [1, 2, 3, 4].map(f);
        let b = [5, 6, 7, 8].map(f);
        let out = [6, 8, 10, 12].map(f);
        let point = vec![e(2), e(3)];
        assert!(verify_add(&pcs, &challenger, &a, &b, &out, &point));
    }

    #[test]
    fn add_mle_rejects_wrong() {
        let (pcs, challenger) = crate::fri::new_pcs(1, 10);
        let a = [1, 2, 3, 4].map(f);
        let b = [5, 6, 7, 8].map(f);
        let out = [6, 8, 10, 13].map(f); // last wrong: 12 → 13
        let point = vec![e(2), e(3)];
        assert!(!verify_add(&pcs, &challenger, &a, &b, &out, &point));
    }

    #[test]
    fn mul_sumcheck_honest() {
        // a = [1,2,3,4], b = [5,6,7,8], out = a⊙b = [5,12,21,32].
        let a = [1, 2, 3, 4].map(f);
        let b = [5, 6, 7, 8].map(f);
        let out = [5, 12, 21, 32].map(f);
        let r = vec![e(2), e(3)];
        assert!(verify_mul(&a, &b, &out, &r, sampler_ext()));
    }

    #[test]
    fn mul_sumcheck_rejects_wrong() {
        let a = [1, 2, 3, 4].map(f);
        let b = [5, 6, 7, 8].map(f);
        let out = [5, 12, 21, 33].map(f); // last wrong: 32 → 33
        let r = vec![e(2), e(3)];
        assert!(!verify_mul(&a, &b, &out, &r, sampler_ext()));
    }

    #[test]
    fn weighted_sum_honest() {
        // seq=2, experts=2, hidden=2.
        // choice[s,e]: s0 picks e0, s1 picks e1.
        let choice = [1, 0, 0, 1].map(f);
        // terms[e][s][j]: e0 = [[1,2],[3,4]], e1 = [[5,6],[7,8]].
        let terms = [1, 2, 3, 4, 5, 6, 7, 8].map(f);
        // out[s][j] = Σ_e choice[s,e]·term[e][s][j].
        // s0: 1·[1,2] + 0·[5,6] = [1,2]; s1: 0·[3,4] + 1·[7,8] = [7,8].
        let out = [1, 2, 7, 8].map(f);
        assert!(verify_weighted_sum(
            &choice,
            &terms,
            &out,
            2,
            2,
            2,
            sampler()
        ));
    }

    #[test]
    fn weighted_sum_rejects_wrong() {
        let choice = [1, 0, 0, 1].map(f);
        let terms = [1, 2, 3, 4, 5, 6, 7, 8].map(f);
        let out = [1, 2, 7, 9].map(f); // last wrong: 8 → 9
        assert!(!verify_weighted_sum(
            &choice,
            &terms,
            &out,
            2,
            2,
            2,
            sampler()
        ));
    }
}
