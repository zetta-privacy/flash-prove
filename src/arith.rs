//! Arithmetization → sumcheck: verify `C = A·B` (matrix multiplication) by
//! sampling one random `(row, col)` and running an inner-product sumcheck over
//! the contraction dimension, instead of recomputing all of `C`.
//!
//! This is the core reduction that turns the transformer's dominant matmuls
//! (attention Q/K/V/O, MoE gate/up/down) into O(log n) verifier work.

use crate::field::Ext;
use crate::sumcheck::{mle_eval, prove, verify};
use p3_field::PrimeCharacteristicRing;

/// Boolean point of `k` as `log` field elements (bit 0 lowest).
pub fn bool_point(k: usize, log: usize) -> Vec<Ext> {
    (0..log)
        .map(|b| {
            if (k >> b) & 1 == 1 {
                Ext::ONE
            } else {
                Ext::ZERO
            }
        })
        .collect()
}

/// Evaluate the MLE of a matrix (flattened row-major) at `point` (row bits ‖ col bits).
pub fn matrix_mle(flat: &[Ext], point: &[Ext]) -> Ext {
    mle_eval(flat, point)
}

/// Verify `C = A·B` at a randomly sampled `(r, s)`, via the inner-product sumcheck
/// over the contraction dimension.
///
/// - `A` is `2^ln × 2^lm`, `B` is `2^lm × 2^lp`, `C = A·B` is `2^ln × 2^lp`.
/// - `r ∈ E^ln`, `s ∈ E^lp` are the verifier's sampled row/col points.
///
/// The claim reduces to `C̃(r,s) = Σ_k Ã(r,k)·B̃(k,s)`, which is a sumcheck over the
/// `lm` contraction variables. Returns `true` iff that sumcheck verifies.
pub fn matmul_check(
    a: &[Ext],
    b: &[Ext],
    c: &[Ext],
    ln: usize,
    lm: usize,
    lp: usize,
    r: &[Ext],
    s: &[Ext],
    sample: impl FnMut(usize) -> Ext,
) -> bool {
    // claimed value D = C̃(r, s).
    let mut rs = r.to_vec();
    rs.extend_from_slice(s);
    let d = matrix_mle(c, &rs);

    // g(k) = Ã(r, k) · B̃(k, s), evaluated over the Boolean cube of k.
    let mut g = vec![Ext::ZERO; 1 << lm];
    for k in 0..(1 << lm) {
        let kb = bool_point(k, lm);
        let mut rk = r.to_vec();
        rk.extend_from_slice(&kb);
        let mut ks = kb.clone();
        ks.extend_from_slice(s);
        g[k] = matrix_mle(a, &rk) * matrix_mle(b, &ks);
    }

    let (rounds, _final_claim, rk) = prove(&g, lm, sample);
    verify(&g, lm, &rounds, &rk, d)
}

/// Verify a pointwise op `out[i] = f(in[i])` at a random point `r` (Schwartz–Zippel).
/// `inp`/`out` are the MLE evaluations of the input/output vectors (length `2^ln`).
/// This covers element-wise ops (SiLU, `mul`, `add`, RMSNorm scale) with no inner sum.
pub fn pointwise_check(inp: &[Ext], out: &[Ext], r: &[Ext], f: impl Fn(Ext) -> Ext) -> bool {
    matrix_mle(out, r) == f(matrix_mle(inp, r))
}

/// Verify `out[i] = a[i] + b[i]` (an `add` op) at a random point `r`.
pub fn add_check(a: &[Ext], b: &[Ext], out: &[Ext], r: &[Ext]) -> bool {
    matrix_mle(out, r) == matrix_mle(a, r) + matrix_mle(b, r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::from_u64;

    fn e(x: u64) -> Ext {
        Ext::from(from_u64(x))
    }

    #[test]
    fn pointwise_add_check_honest() {
        // a = [1, 3], b = [2, 4] (first-index-lowest-bit), out = a + b = [3, 7].
        let a = [e(1), e(3)];
        let b = [e(2), e(4)];
        let out = [e(3), e(7)];
        let r = [e(5)];
        assert!(add_check(&a, &b, &out, &r));
    }

    #[test]
    fn pointwise_add_check_rejects_wrong() {
        let a = [e(1), e(3)];
        let b = [e(2), e(4)];
        let out = [e(3), e(8)]; // out[1] wrong: 7 → 8
        let r = [e(5)];
        assert!(!add_check(&a, &b, &out, &r));
    }

    #[test]
    fn matmul_check_honest() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]], C = A·B = [[19,22],[43,50]].
        // MLE flat is first-index-lowest-bit: [X[0][0], X[1][0], X[0][1], X[1][1]].
        let a = [e(1), e(3), e(2), e(4)];
        let b = [e(5), e(7), e(6), e(8)];
        let c = [e(19), e(43), e(22), e(50)];
        let r = [e(2)];
        let s = [e(3)];
        let sample = |i: usize| e(i as u64 + 4);
        assert!(matmul_check(&a, &b, &c, 1, 1, 1, &r, &s, sample));
    }

    #[test]
    fn matmul_check_rejects_wrong_product() {
        let a = [e(1), e(3), e(2), e(4)];
        let b = [e(5), e(7), e(6), e(8)];
        // C[0][0] is wrong: 19 → 20.
        let c = [e(20), e(43), e(22), e(50)];
        let r = [e(2)];
        let s = [e(3)];
        let sample = |i: usize| e(i as u64 + 4);
        assert!(!matmul_check(&a, &b, &c, 1, 1, 1, &r, &s, sample));
    }
}
