//! Linear-code inner product: the bridge from a univariate FRI commitment to a
//! multilinear MLE evaluation.
//!
//! Given data `v` (committed as the evaluations `V(ω^j) = v[j]`) and a public
//! vector `h`, the inner product is the 0-th DFT coefficient of the pointwise
//! product `P[j] = h[j]·v[j]`:
//!
//!   ⟨h, v⟩ = Σ_j h[j]·v[j] = Σ_j P(ω^j) = N · c_0,
//!
//! where `c_0` is the constant coefficient of `P(X)`. So the verifier checks the
//! claim `b` by committing `P` and the quotient `Q(X) = (P(X) − c_0)/X`, then at a
//! random `α` checking `P(α) = c_0 + α·Q(α)` — a constant-coefficient extraction,
//! which (unlike the `X^{N−1}` extraction) is valid because `P − c_0` has no
//! constant term.

use crate::field::{Ext, Fp};
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};

/// The constant coefficient `c_0 = ⟨h, v⟩ / N` of the pointwise-product polynomial.
pub fn constant_coeff(h: &[Fp], v: &[Fp]) -> Fp {
    assert_eq!(h.len(), v.len());
    let p: Vec<Fp> = h.iter().zip(v).map(|(&a, &b)| a * b).collect();
    Radix2Dit::<Fp>::default().idft(p)[0]
}

/// The quotient `Q(X) = (P(X) − c_0)/X`, in coefficient form, padded to length N.
pub fn constant_quotient(h: &[Fp], v: &[Fp], c0: Fp) -> Vec<Fp> {
    let p: Vec<Fp> = h.iter().zip(v).map(|(&a, &b)| a * b).collect();
    let coeffs = Radix2Dit::<Fp>::default().idft(p);
    debug_assert_eq!(coeffs[0], c0, "constant coefficient mismatch");
    let mut q = coeffs[1..].to_vec();
    q.push(Fp::new(0)); // pad back to length N (power of two)
    q
}

/// Evaluate a coefficient-form polynomial at a base-field point (Horner).
fn eval_base(coeffs: &[Fp], x: Fp) -> Fp {
    let mut acc = Fp::new(0);
    for &c in coeffs.iter().rev() {
        acc = acc * x + c;
    }
    acc
}

/// Check the constant-coefficient identity `P(α) = c_0 + α·Q(α)` at a random `α`.
pub fn check_constant_claim(h: &[Fp], v: &[Fp], b: Fp, c0: Fp, q: &[Fp], alpha: Fp) -> bool {
    let n = h.len();
    // c_0 must encode b: b = N · c_0.
    if c0 * Fp::new(n as u64) != b {
        return false;
    }
    // P(α) = c_0 + α · Q(α).
    let p: Vec<Fp> = h.iter().zip(v).map(|(&a, &bb)| a * bb).collect();
    let p_coeffs = Radix2Dit::<Fp>::default().idft(p);
    let lhs = eval_base(&p_coeffs, alpha);
    let rhs = c0 + alpha * eval_base(q, alpha);
    lhs == rhs
}

/// Split `eq(point, ·)` into its three base-field component vectors `(a, b, c)`,
/// so the extension-field MLE evaluation `ṽ(point) = Σ_j eq(point, j)·v[j]`
/// decomposes into three base-field inner products `⟨a,v⟩ + ⟨b,v⟩·ϑ + ⟨c,v⟩·ϑ²`.
pub fn eq_components(point: &[Ext], n: usize) -> [Vec<Fp>; 3] {
    let mut a = Vec::with_capacity(1 << n);
    let mut b = Vec::with_capacity(1 << n);
    let mut c = Vec::with_capacity(1 << n);
    for j in 0..(1 << n) {
        let e = crate::field::eq(point, j);
        let coeffs = <Ext as p3_field::BasedVectorSpace<Fp>>::as_basis_coefficients_slice(&e);
        a.push(coeffs[0]);
        b.push(coeffs[1]);
        c.push(coeffs[2]);
    }
    [a, b, c]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(x: u64) -> Fp {
        Fp::new(x)
    }

    #[test]
    fn constant_coeff_is_ip_over_n() {
        // N = 4. h = [1,2,3,4], v = [5,6,7,8]. ⟨h,v⟩ = 5+12+21+32 = 70.
        let h = [f(1), f(2), f(3), f(4)];
        let v = [f(5), f(6), f(7), f(8)];
        let c0 = constant_coeff(&h, &v);
        assert_eq!(c0 * f(4), f(70)); // 4·c0 = 70
    }

    #[test]
    fn constant_claim_roundtrip() {
        let h = [f(1), f(2), f(3), f(4)];
        let v = [f(5), f(6), f(7), f(8)];
        let b = f(70);
        let c0 = constant_coeff(&h, &v);
        let q = constant_quotient(&h, &v, c0);
        let alpha = f(9);
        assert!(check_constant_claim(&h, &v, b, c0, &q, alpha));
    }

    #[test]
    fn constant_claim_rejects_wrong_b() {
        let h = [f(1), f(2), f(3), f(4)];
        let v = [f(5), f(6), f(7), f(8)];
        let b_wrong = f(71); // ⟨h,v⟩ = 70
        let c0 = constant_coeff(&h, &v);
        let q = constant_quotient(&h, &v, c0);
        let alpha = f(9);
        assert!(!check_constant_claim(&h, &v, b_wrong, c0, &q, alpha));
    }

    #[test]
    fn linear_claim_via_fri() {
        use crate::field::Ext;
        use crate::pcs::fri;
        use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};

        let (pcs, challenger) = fri::new_pcs(1, 10);

        // data v (length 4), committed via FRI (evaluations over the coset).
        let v = [f(5), f(6), f(7), f(8)];
        let (commitment_v, prover_data_v, domain_v) = fri::commit_vec(&pcs, &v);
        let _ = (commitment_v, prover_data_v, domain_v);

        // public h, pointwise product P[j] = h[j]·v[j].
        let h = [f(1), f(2), f(3), f(4)];
        let p: Vec<Fp> = h.iter().zip(&v).map(|(&a, &b)| a * b).collect();
        let (commitment_p, prover_data_p, domain_p) = fri::commit_vec(&pcs, &p);

        // c0 = ⟨h,v⟩/N, and the quotient Q(X) = (P − c0)/X.
        let c0 = constant_coeff(&h, &v);
        let q_coeffs = constant_quotient(&h, &v, c0);
        let q_evals = Radix2Dit::<Fp>::default().dft(q_coeffs);
        let (commitment_q, prover_data_q, domain_q) = fri::commit_vec(&pcs, &q_evals);

        // sample an extension-field point α and open P, Q there.
        let alpha = Ext::from(f(9));
        let p_alpha = fri::open_at(
            &pcs,
            &challenger,
            &commitment_p,
            &prover_data_p,
            domain_p,
            alpha,
        );
        let q_alpha = fri::open_at(
            &pcs,
            &challenger,
            &commitment_q,
            &prover_data_q,
            domain_q,
            alpha,
        );

        // P(α) = c0 + α·Q(α).
        assert_eq!(p_alpha, Ext::from(c0) + alpha * q_alpha);
    }

    #[test]
    fn mle_eval_via_fri() {
        use crate::field::Ext;
        use crate::pcs::fri;
        use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};

        let (pcs, challenger) = fri::new_pcs(1, 10);

        // data v (length 4), committed.
        let v = [f(5), f(6), f(7), f(8)];

        // sample a point (2 field coords for length 4).
        let point = vec![Ext::from(f(2)), Ext::from(f(3))];

        // decompose eq(point, ·) into 3 base-field components and compute the 3
        // base-field inner products (the claimed MLE evaluation's coordinates).
        let [a, b, c] = eq_components(&point, 2);
        let ip = |h: &[Fp]| h.iter().zip(&v).map(|(&x, &y)| x * y).sum::<Fp>();
        let (claimed_a, claimed_b, claimed_c) = (ip(&a), ip(&b), ip(&c));

        // Each component is a base-field inner product, checked via the linear claim.
        for (hcomp, bcomp) in [(&a, claimed_a), (&b, claimed_b), (&c, claimed_c)] {
            let c0 = constant_coeff(hcomp, &v);
            assert_eq!(c0 * f(4), bcomp, "component inner product");
            let p: Vec<Fp> = hcomp.iter().zip(&v).map(|(&x, &y)| x * y).collect();
            let (cp, pdp, dp) = fri::commit_vec(&pcs, &p);
            let q = constant_quotient(hcomp, &v, c0);
            let q_evals = Radix2Dit::<Fp>::default().dft(q);
            let (cq, pdq, dq) = fri::commit_vec(&pcs, &q_evals);
            let alpha = Ext::from(f(11));
            let pa = fri::open_at(&pcs, &challenger, &cp, &pdp, dp, alpha);
            let qa = fri::open_at(&pcs, &challenger, &cq, &pdq, dq, alpha);
            assert_eq!(pa, Ext::from(c0) + alpha * qa);
        }
    }
}
