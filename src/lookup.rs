//! LogUp (log-derivative) lookup argument for the nonlinear pointwise ops of the
//! integer model — SiLU, the matmul `requant` rounding, softmax, and top-k.
//!
//! A lookup proves that a committed multiset of tuples `{q[i]}` is a subset of a
//! public table `{t[j]}` (with multiplicities). For `out = silu(in)` the queried
//! tuple is `(in[i], out[i])` and the table is `{(x, silu(x))}`; for `requant` the
//! table is `{(x, requant(x, k))}`.
//!
//! # The argument (Haböck 2022, ePrint 2022/1530)
//!
//! For random challenges `α, β` the verifier fingerprints each tuple into a
//! single field element `fp(x) = α + Σ_k β^k·x_k` and checks the log-derivative
//! identity
//!
//! ```text
//!     Σ_i 1 / fp(q[i])  ==  Σ_j m[j] / fp(t[j])
//! ```
//!
//! where `m[j]` is the prover-committed multiplicity of `t[j]` among the queries.
//! The verifier also requires `Σ_j m[j] = n` (and, in the committed version,
//! `m[j] ≥ 0`). If `{q} ⊆ {t}` the identity holds for every `α, β`; otherwise the
//! cleared-denominator polynomial vanishes for at most a negligible fraction of
//! challenges, so the check is sound.
//!
//! The table `t` is *not* materialized by the succinct verifier: the per-op table
//! (`silu`, `requant`, …) is a public function over a bounded integer domain,
//! represented as a committed/virtual polynomial and summed via sumcheck. This
//! module checks the LogUp identity given the table in the clear (arithmetic
//! level), which is the exact identity the succinct sumcheck reduces to.

use crate::field::{Ext, Fp};
use p3_field::Field;

/// `fp(x) = α + Σ_k β^{k+1} · x_k` — fold a `W`-column tuple into one challenge-field
/// element (Horner over the columns). The leading coefficient is `β` (not `β⁰`),
/// so the tuple's first coordinate is also challenge-weighted and the constant
/// term `α` is reserved for the folding offset; this keeps the fingerprint
/// injective in a random `β` while avoiding the degenerate `α + x₀ = 0` collision.
pub fn fingerprint(tuple: &[Ext], alpha: Ext, beta: Ext) -> Ext {
    let mut acc = alpha;
    let mut b = beta;
    for &x in tuple {
        acc += b * x;
        b *= beta;
    }
    acc
}

/// The honest prover's multiplicity `m[j] = #{i : q[i] == t[j]}` of each table
/// entry among the queries. The verifier does not compute this — it receives `m`
/// from the prover and constrains it via the LogUp identity and `Σ_j m[j] = n`.
pub fn multiplicities(q: &[Vec<Ext>], t: &[Vec<Ext>]) -> Vec<usize> {
    use std::collections::HashMap;
    let mut index: HashMap<&Vec<Ext>, usize> = HashMap::with_capacity(t.len());
    for (j, tj) in t.iter().enumerate() {
        index.entry(tj).or_insert(j);
    }
    let mut m = vec![0usize; t.len()];
    for qi in q {
        if let Some(&j) = index.get(qi) {
            m[j] += 1;
        }
    }
    m
}

/// Verify the LogUp lookup `{q[i]} ⊆ {t[j]}` given the prover's multiplicity
/// vector `m` (length `t.len()`). Checks:
///
/// 1. `Σ_j m[j] = n` (every query is accounted for), and
/// 2. `Σ_i 1/fp(q[i]) == Σ_j m[j]/fp(t[j])` (the log-derivative identity).
///
/// The challenges `α, β` must be such that no fingerprint is zero (which holds
/// with overwhelming probability for random challenges; a fixed test pair must
/// avoid it). `m[j] ≥ 0` is separately range-checked in the committed version.
pub fn verify_lookup(q: &[Vec<Ext>], t: &[Vec<Ext>], m: &[usize], alpha: Ext, beta: Ext) -> bool {
    let n = q.len();
    assert_eq!(m.len(), t.len(), "multiplicity length");
    if m.iter().sum::<usize>() != n {
        return false;
    }
    let fq: Vec<Ext> = q.iter().map(|qi| fingerprint(qi, alpha, beta)).collect();
    let ft: Vec<Ext> = t.iter().map(|tj| fingerprint(tj, alpha, beta)).collect();
    let lhs: Ext = fq.iter().map(|&x| x.inverse()).sum();
    let rhs: Ext = ft
        .iter()
        .zip(m)
        .map(|(&x, &mj)| Ext::from(Fp::new(mj as u64)) * x.inverse())
        .sum();
    lhs == rhs
}

/// Convert a signed integer to an extension-field element.
pub fn i64_to_ext(x: i64) -> Ext {
    if x >= 0 {
        Ext::from(Fp::new(x as u64))
    } else {
        -Ext::from(Fp::new((-x) as u64))
    }
}

/// Generic 2-column LogUp lookup `out[i] = f(in[i])` over a public table
/// `{(x, f(x)) : x ∈ [lo, hi]}`. `f` is any public integer function (SiLU,
/// `requant`, `round(·/scale)`, …); `lo`/`hi` must bound the actual inputs.
pub fn verify_function_lookup(
    inp: &[i64],
    out: &[i64],
    f: impl Fn(i64) -> i64,
    lo: i64,
    hi: i64,
    alpha: Ext,
    beta: Ext,
) -> bool {
    assert_eq!(inp.len(), out.len());
    let t: Vec<Vec<Ext>> = (lo..=hi)
        .map(|x| vec![i64_to_ext(x), i64_to_ext(f(x))])
        .collect();
    let q: Vec<Vec<Ext>> = inp
        .iter()
        .zip(out)
        .map(|(&a, &b)| vec![i64_to_ext(a), i64_to_ext(b)])
        .collect();
    let m = multiplicities(&q, &t);
    verify_lookup(&q, &t, &m, alpha, beta)
}

/// Verify `out[i] = requant(in[i], k)` for all `i`, as a 2-column LogUp lookup into
/// the public requant table over the integer domain `[lo, hi]`. `lo`/`hi` must
/// bound the actual inputs (a fixed domain, not adapted to the query, keeps the
/// lookup sound).
pub fn verify_requant(
    inp: &[i64],
    out: &[i64],
    k: u32,
    lo: i64,
    hi: i64,
    alpha: Ext,
    beta: Ext,
) -> bool {
    assert_eq!(inp.len(), out.len());
    let t: Vec<Vec<Ext>> = (lo..=hi)
        .map(|x| vec![i64_to_ext(x), i64_to_ext(crate::spec::requant(x, k))])
        .collect();
    let q: Vec<Vec<Ext>> = inp
        .iter()
        .zip(out)
        .map(|(&a, &b)| vec![i64_to_ext(a), i64_to_ext(b)])
        .collect();
    let m = multiplicities(&q, &t);
    verify_lookup(&q, &t, &m, alpha, beta)
}

/// Verify `out[i] = silu(in[i])` for all `i`, as a 2-column LogUp lookup into the
/// public SiLU table over the integer domain `[lo, hi]` (a fixed sub-domain of the
/// clamped `[−2^15, 2^15)` range; `lo`/`hi` must bound the actual inputs).
pub fn verify_silu(inp: &[i64], out: &[i64], lo: i64, hi: i64, alpha: Ext, beta: Ext) -> bool {
    assert_eq!(inp.len(), out.len());
    let t: Vec<Vec<Ext>> = (lo..=hi)
        .map(|x| vec![i64_to_ext(x), i64_to_ext(crate::spec::silu(x))])
        .collect();
    let q: Vec<Vec<Ext>> = inp
        .iter()
        .zip(out)
        .map(|(&a, &b)| vec![i64_to_ext(a), i64_to_ext(b)])
        .collect();
    let m = multiplicities(&q, &t);
    verify_lookup(&q, &t, &m, alpha, beta)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(x: i64) -> Ext {
        if x >= 0 {
            Ext::from(Fp::new(x as u64))
        } else {
            -Ext::from(Fp::new((-x) as u64))
        }
    }

    fn alpha() -> Ext {
        e(7)
    }
    fn beta() -> Ext {
        e(3)
    }

    /// f(x) = x², a nonlinear table over x ∈ {0,1,2,3}.
    fn table() -> Vec<Vec<Ext>> {
        (0..4).map(|x| vec![e(x), e(x * x)]).collect()
    }

    #[test]
    fn lookup_honest() {
        let t = table();
        // Queries: (1,1), (3,9), (1,1) — a subset with a repeated entry.
        let q = vec![vec![e(1), e(1)], vec![e(3), e(9)], vec![e(1), e(1)]];
        let m = multiplicities(&q, &t); // [2, 0, 0, 1]
        assert!(verify_lookup(&q, &t, &m, alpha(), beta()));
    }

    #[test]
    fn lookup_rejects_missing_mult() {
        let t = table();
        // (5, 25) is not in the table, so the honest m sums to 1 ≠ 2.
        let q = vec![vec![e(1), e(1)], vec![e(5), e(25)]];
        let m = multiplicities(&q, &t);
        assert!(!verify_lookup(&q, &t, &m, alpha(), beta()));
    }

    #[test]
    fn lookup_rejects_forged_mult() {
        let t = table();
        // A cheating prover claims both queries hit (1,1) by inflating m[0]=2.
        // Σ m = 2 = n, but the identity Σ 1/fp(q) = Σ m/fp(t) no longer balances
        // because 1/fp(5,25) has no counterpart on the table side.
        let q = vec![vec![e(1), e(1)], vec![e(5), e(25)]];
        let m = vec![2, 0, 0, 0];
        assert!(!verify_lookup(&q, &t, &m, alpha(), beta()));
    }

    #[test]
    fn lookup_rejects_wrong_output() {
        let t = table();
        // (3, 10) has the wrong output (f(3) = 9). Honest m is empty → Σ m = 0.
        let q = vec![vec![e(3), e(10)]];
        let m = multiplicities(&q, &t);
        assert!(!verify_lookup(&q, &t, &m, alpha(), beta()));
    }

    #[test]
    fn requant_honest() {
        let inp = [-1024, -1, 0, 1, 1024, 2000];
        let out: Vec<i64> = inp.iter().map(|&x| crate::spec::requant(x, 4)).collect();
        assert!(verify_requant(&inp, &out, 4, -2048, 2048, alpha(), beta()));
    }

    #[test]
    fn requant_rejects_wrong() {
        let inp = [-1024, 0, 1024];
        let mut out: Vec<i64> = inp.iter().map(|&x| crate::spec::requant(x, 4)).collect();
        out[2] += 1;
        assert!(!verify_requant(&inp, &out, 4, -2048, 2048, alpha(), beta()));
    }

    #[test]
    fn function_lookup_honest() {
        let inp = [0, 1, 2, 3, 4];
        let out: Vec<i64> = inp.iter().map(|&x| x * x).collect();
        assert!(verify_function_lookup(
            &inp,
            &out,
            |x| x * x,
            0,
            10,
            alpha(),
            beta()
        ));
    }

    #[test]
    fn function_lookup_rejects_wrong() {
        let inp = [0, 1, 2, 3];
        let mut out: Vec<i64> = inp.iter().map(|&x| x * x).collect();
        out[2] += 1;
        assert!(!verify_function_lookup(
            &inp,
            &out,
            |x| x * x,
            0,
            10,
            alpha(),
            beta()
        ));
    }
}
