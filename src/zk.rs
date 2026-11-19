//! Succinct verifier: MLE evaluation and matmul/pointwise constraint checks
//! built on the FRI commitment + the linear-claim bridge, replacing the O(n)
//! `decode_exact` + full model recompute.

use crate::field::{Ext, Fp};
use crate::fri::{self, Challenger, Pcs};
use crate::linear_code::{constant_coeff, constant_quotient, eq_components};
use crate::protocol::eq;
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};

/// ⌈log2 n⌉.
fn log2(n: usize) -> usize {
    (usize::BITS - (n - 1).leading_zeros()) as usize
}

/// The MLE of `v` at a field point (prover-side; the prover holds `v` in the clear).
pub fn mle_value(v: &[Fp], point: &[Ext]) -> Ext {
    v.iter()
        .enumerate()
        .map(|(j, &vj)| eq(point, j) * Ext::from(vj))
        .sum()
}

/// Verify one base-field inner product `⟨h, v⟩ = b` via the linear claim.
fn verify_inner_product(pcs: &Pcs, challenger: &Challenger, v: &[Fp], h: &[Fp], b: Fp) -> bool {
    let c0 = constant_coeff(h, v);
    if c0 * Fp::new(v.len() as u64) != b {
        return false;
    }
    let p: Vec<Fp> = h.iter().zip(v).map(|(&x, &y)| x * y).collect();
    let (cp, pdp, dp) = fri::commit_vec(pcs, &p);
    let q = constant_quotient(h, v, c0);
    let q_evals = Radix2Dit::<Fp>::default().dft(q);
    let (cq, pdq, dq) = fri::commit_vec(pcs, &q_evals);
    let alpha = Ext::from(Fp::new(11));
    let pa = fri::open_at(pcs, challenger, &cp, &pdp, dp, alpha);
    let qa = fri::open_at(pcs, challenger, &cq, &pdq, dq, alpha);
    pa == Ext::from(c0) + alpha * qa
}

/// Verify `ṽ(point) = claimed` by decomposing into 3 base-field inner products.
pub fn verify_mle(
    pcs: &Pcs,
    challenger: &Challenger,
    v: &[Fp],
    point: &[Ext],
    claimed: Ext,
) -> bool {
    let [a, b, c] = eq_components(point, log2(v.len()));
    let coeffs = <Ext as BasedVectorSpace<Fp>>::as_basis_coefficients_slice(&claimed);
    for (h, bcomp) in [(&a, coeffs[0]), (&b, coeffs[1]), (&c, coeffs[2])] {
        if !verify_inner_product(pcs, challenger, v, h, bcomp) {
            return false;
        }
    }
    true
}

/// Prove the committed MLE evaluation `ṽ(point) = claimed`: produce the three
/// base-field linear-claim proofs for the already-committed `v`.
pub fn prove_mle_committed(
    pcs: &Pcs,
    challenger: &Challenger,
    com_v: &fri::Commitment,
    pd_v: &fri::ProverData,
    dom_v: fri::Domain,
    v: &[Fp],
    point: &[Ext],
) -> [InnerProductProof; 3] {
    let [a, b, c] = eq_components(point, log2(v.len()));
    [
        prove_inner_product_committed_given(pcs, challenger, com_v, pd_v, dom_v, v, &a),
        prove_inner_product_committed_given(pcs, challenger, com_v, pd_v, dom_v, v, &b),
        prove_inner_product_committed_given(pcs, challenger, com_v, pd_v, dom_v, v, &c),
    ]
}

/// Verify the committed MLE evaluation `ṽ(point) = claimed` from commitments and
/// the three linear-claim proofs alone (no `v` in the clear — only its length `n`).
pub fn verify_mle_committed(
    pcs: &Pcs,
    challenger: &Challenger,
    com_v: &fri::Commitment,
    dom_v: fri::Domain,
    n: usize,
    point: &[Ext],
    claimed: Ext,
    proofs: &[InnerProductProof; 3],
) -> bool {
    let [a, b, c] = eq_components(point, log2(n));
    let coeffs = <Ext as BasedVectorSpace<Fp>>::as_basis_coefficients_slice(&claimed);
    verify_inner_product_committed(pcs, challenger, com_v, dom_v, &a, coeffs[0], &proofs[0])
        && verify_inner_product_committed(pcs, challenger, com_v, dom_v, &b, coeffs[1], &proofs[1])
        && verify_inner_product_committed(pcs, challenger, com_v, dom_v, &c, coeffs[2], &proofs[2])
}

/// One FRI opening: the claimed value and its opening proof. (The verifier holds
/// this instead of any prover material.)
pub struct Opening {
    pub value: Ext,
    pub proof: fri::Proof,
}

/// The committed linear-claim proof (prover's output): the `P`/`Q` commitments,
/// the claimed constant coefficient `c0`, and the four openings at the derived
/// challenge points. The verifier checks this with commitments alone — no prover
/// material.
///
/// This is the O(log n) form of [`verify_inner_product`]: the verifier checks
/// `⟨h, v⟩ = b` from commitments alone (a handful of FRI openings), never
/// re-reading `v`. The three checks:
///
///  1. `c0 · N == b` (the claim encodes the inner product),
///  2. opening `P` and `v` at a random coset point `ω^k` gives `P(ω^k) = h[k]·v(ω^k)`
///     (so `P = h·v`, hence `c0` is the *right* constant coefficient), and
///  3. opening `P` and `Q` at a random `α` gives `P(α) = c0 + α·Q(α)` (so `c0` is
///     indeed the constant coefficient of `P`).
pub struct InnerProductProof {
    pub com_p: fri::Commitment,
    pub com_q: fri::Commitment,
    pub c0: Fp,
    pub v_k: Opening,
    pub p_k: Opening,
    pub p_alpha: Opening,
    pub q_alpha: Opening,
}

/// Build the [`InnerProductProof`] for an *already-committed* `v`: commit
/// `P = h·v` and `Q = (P − c0)/X`, derive the challenge points from the transcript,
/// and open at them (reusing the caller's `com_v`/`pd_v`). Used by the matmul and
/// IPA, where `v` is committed once and shared across several linear claims.
pub fn prove_inner_product_committed_given(
    pcs: &Pcs,
    challenger: &Challenger,
    com_v: &fri::Commitment,
    pd_v: &fri::ProverData,
    dom: fri::Domain,
    v: &[Fp],
    h: &[Fp],
) -> InnerProductProof {
    let n = v.len();
    let c0 = constant_coeff(h, v);
    let p: Vec<Fp> = h.iter().zip(v).map(|(&x, &y)| x * y).collect();
    let (com_p, pd_p, _) = fri::commit_vec(pcs, &p);
    let q = constant_quotient(h, v, c0);
    let q_evals = Radix2Dit::<Fp>::default().dft(q);
    let (com_q, pd_q, _) = fri::commit_vec(pcs, &q_evals);

    let mut tr = fri::Transcript::new(challenger);
    tr.observe(com_v);
    tr.observe(&com_p);
    tr.observe(&com_q);
    let k = tr.sample_index(n);
    let alpha = tr.sample_ext();
    let point_k = Ext::from(dom.element(k));

    let (v_k_val, v_k_proof) = fri::open_at_prove(pcs, challenger, com_v, pd_v, point_k);
    let (p_k_val, p_k_proof) = fri::open_at_prove(pcs, challenger, &com_p, &pd_p, point_k);
    let (p_alpha_val, p_alpha_proof) = fri::open_at_prove(pcs, challenger, &com_p, &pd_p, alpha);
    let (q_alpha_val, q_alpha_proof) = fri::open_at_prove(pcs, challenger, &com_q, &pd_q, alpha);

    InnerProductProof {
        com_p,
        com_q,
        c0,
        v_k: Opening {
            value: v_k_val,
            proof: v_k_proof,
        },
        p_k: Opening {
            value: p_k_val,
            proof: p_k_proof,
        },
        p_alpha: Opening {
            value: p_alpha_val,
            proof: p_alpha_proof,
        },
        q_alpha: Opening {
            value: q_alpha_val,
            proof: q_alpha_proof,
        },
    }
}

/// Prove `⟨h, v⟩ = b` with `v` committed: commit `v`, then build the
/// [`InnerProductProof`]. Returns the commitment to `v`, its domain, and the proof.
pub fn prove_inner_product_committed(
    pcs: &Pcs,
    challenger: &Challenger,
    v: &[Fp],
    h: &[Fp],
) -> (fri::Commitment, fri::Domain, InnerProductProof) {
    let (com_v, pd_v, dom) = fri::commit_vec(pcs, v);
    let proof = prove_inner_product_committed_given(pcs, challenger, &com_v, &pd_v, dom, v, h);
    (com_v, dom, proof)
}

/// Verify `⟨h, v⟩ = b` from the committed linear-claim proof: replay the transcript
/// challenges, verify the four FRI openings (no prover material), and check the two
/// identities.
pub fn verify_inner_product_committed(
    pcs: &Pcs,
    challenger: &Challenger,
    com_v: &fri::Commitment,
    dom: fri::Domain,
    h: &[Fp],
    b: Fp,
    proof: &InnerProductProof,
) -> bool {
    let n = h.len();
    let mut tr = fri::Transcript::new(challenger);
    tr.observe(com_v);
    tr.observe(&proof.com_p);
    tr.observe(&proof.com_q);
    let k = tr.sample_index(n);
    let alpha = tr.sample_ext();

    if proof.c0 * Fp::new(n as u64) != b {
        return false;
    }
    let point_k = Ext::from(dom.element(k));
    if !fri::open_at_verify(
        pcs,
        challenger,
        com_v,
        dom,
        point_k,
        proof.v_k.value,
        &proof.v_k.proof,
    ) {
        return false;
    }
    if !fri::open_at_verify(
        pcs,
        challenger,
        &proof.com_p,
        dom,
        point_k,
        proof.p_k.value,
        &proof.p_k.proof,
    ) {
        return false;
    }
    if proof.p_k.value != Ext::from(h[k]) * proof.v_k.value {
        return false;
    }
    if !fri::open_at_verify(
        pcs,
        challenger,
        &proof.com_p,
        dom,
        alpha,
        proof.p_alpha.value,
        &proof.p_alpha.proof,
    ) {
        return false;
    }
    if !fri::open_at_verify(
        pcs,
        challenger,
        &proof.com_q,
        dom,
        alpha,
        proof.q_alpha.value,
        &proof.q_alpha.proof,
    ) {
        return false;
    }
    proof.p_alpha.value == Ext::from(proof.c0) + alpha * proof.q_alpha.value
}

/// Random-linear-combination batching of `m` linear claims sharing one committed
/// vector `v`: `⟨h_i, v⟩ = b_i` for all `i` collapses to a single claim
/// `⟨Σ_i r_i·h_i, v⟩ = Σ_i r_i·b_i`, so the verifier pays one linear claim (two
/// FRI openings) instead of `m`. Sound by Schwartz–Zippel in `r`.
pub fn batch_inner_products(h_list: &[&[Fp]], b_list: &[Fp], r: &[Fp]) -> (Vec<Fp>, Fp) {
    assert_eq!(h_list.len(), b_list.len());
    assert_eq!(h_list.len(), r.len());
    let n = h_list[0].len();
    let mut h = vec![Fp::ZERO; n];
    let mut b = Fp::ZERO;
    for (i, (&hi, &bi)) in h_list.iter().zip(b_list).enumerate() {
        assert_eq!(hi.len(), n);
        let ri = r[i];
        b += ri * bi;
        for j in 0..n {
            h[j] += ri * hi[j];
        }
    }
    (h, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::from_u64;

    fn f(x: u64) -> Fp {
        Fp::new(x)
    }

    #[test]
    fn verify_mle_roundtrip() {
        let (pcs, challenger) = fri::new_pcs(1, 10);
        let v = [f(5), f(6), f(7), f(8)];
        let point = vec![Ext::from(f(2)), Ext::from(f(3))];
        let claimed = mle_value(&v, &point);
        assert!(verify_mle(&pcs, &challenger, &v, &point, claimed));
    }

    #[test]
    fn verify_mle_committed_roundtrip() {
        let (pcs, challenger) = fri::new_pcs(1, 10);
        let v = [f(5), f(6), f(7), f(8)];
        let point = vec![Ext::from(f(2)), Ext::from(f(3))];
        let claimed = mle_value(&v, &point);
        let (com_v, pd_v, dom) = fri::commit_vec(&pcs, &v);
        let proofs = prove_mle_committed(&pcs, &challenger, &com_v, &pd_v, dom, &v, &point);
        assert!(verify_mle_committed(
            &pcs,
            &challenger,
            &com_v,
            dom,
            4,
            &point,
            claimed,
            &proofs
        ));
    }

    #[test]
    fn verify_mle_committed_rejects_wrong_claim() {
        let (pcs, challenger) = fri::new_pcs(1, 10);
        let v = [f(5), f(6), f(7), f(8)];
        let point = vec![Ext::from(f(2)), Ext::from(f(3))];
        let claimed = mle_value(&v, &point);
        let (com_v, pd_v, dom) = fri::commit_vec(&pcs, &v);
        let proofs = prove_mle_committed(&pcs, &challenger, &com_v, &pd_v, dom, &v, &point);
        // Wrong claimed value → the basis coefficients shift, breaking the linear claims.
        let wrong = claimed + Ext::ONE;
        assert!(!verify_mle_committed(
            &pcs,
            &challenger,
            &com_v,
            dom,
            4,
            &point,
            wrong,
            &proofs
        ));
    }

    #[test]
    fn committed_inner_product_honest() {
        let (pcs, challenger) = fri::new_pcs(1, 10);
        // v = [5,6,7,8], h = [1,2,3,4], ⟨h,v⟩ = 5+12+21+32 = 70.
        let v = [f(5), f(6), f(7), f(8)];
        let h = [f(1), f(2), f(3), f(4)];
        let b = f(70);
        let (com_v, dom, proof) = prove_inner_product_committed(&pcs, &challenger, &v, &h);
        assert!(verify_inner_product_committed(
            &pcs,
            &challenger,
            &com_v,
            dom,
            &h,
            b,
            &proof
        ));
    }

    #[test]
    fn committed_inner_product_rejects_wrong_b() {
        let (pcs, challenger) = fri::new_pcs(1, 10);
        let v = [f(5), f(6), f(7), f(8)];
        let h = [f(1), f(2), f(3), f(4)];
        let b = f(71); // ⟨h,v⟩ = 70
        let (com_v, dom, proof) = prove_inner_product_committed(&pcs, &challenger, &v, &h);
        assert!(!verify_inner_product_committed(
            &pcs,
            &challenger,
            &com_v,
            dom,
            &h,
            b,
            &proof
        ));
    }

    #[test]
    fn committed_inner_product_rejects_tampered_opening() {
        let (pcs, challenger) = fri::new_pcs(1, 10);
        let v = [f(5), f(6), f(7), f(8)];
        let h = [f(1), f(2), f(3), f(4)];
        let b = f(70);
        let (com_v, dom, mut proof) = prove_inner_product_committed(&pcs, &challenger, &v, &h);
        // Tamper the claimed P(ω^k) value: the P = h·v check must reject.
        proof.p_k.value = proof.p_k.value + Ext::ONE;
        assert!(!verify_inner_product_committed(
            &pcs,
            &challenger,
            &com_v,
            dom,
            &h,
            b,
            &proof
        ));
    }

    #[test]
    fn batch_inner_products_honest() {
        let (pcs, challenger) = fri::new_pcs(1, 10);
        let v = [f(5), f(6), f(7), f(8)];
        let h1 = [f(1), f(0), f(0), f(0)];
        let h2 = [f(0), f(1), f(0), f(0)];
        let h3 = [f(0), f(0), f(1), f(0)];
        let r = [f(1), f(2), f(3)];
        // ⟨h_i, v⟩ = 5, 6, 7; batched ⟨h,v⟩ = 1·5 + 2·6 + 3·7 = 38.
        let (h, b) = batch_inner_products(&[&h1, &h2, &h3], &[f(5), f(6), f(7)], &r);
        let (com_v, dom, proof) = prove_inner_product_committed(&pcs, &challenger, &v, &h);
        assert!(verify_inner_product_committed(
            &pcs,
            &challenger,
            &com_v,
            dom,
            &h,
            b,
            &proof
        ));
    }

    #[test]
    fn batch_inner_products_rejects_wrong() {
        let (pcs, challenger) = fri::new_pcs(1, 10);
        let v = [f(5), f(6), f(7), f(8)];
        let h1 = [f(1), f(0), f(0), f(0)];
        let h2 = [f(0), f(1), f(0), f(0)];
        let h3 = [f(0), f(0), f(1), f(0)];
        let r = [f(1), f(2), f(3)];
        // Wrong b3 = 8 (true ⟨h3, v⟩ = 7) shifts the batched claim by r[2] = 3.
        let (h, b) = batch_inner_products(&[&h1, &h2, &h3], &[f(5), f(6), f(8)], &r);
        let (com_v, dom, proof) = prove_inner_product_committed(&pcs, &challenger, &v, &h);
        assert!(!verify_inner_product_committed(
            &pcs,
            &challenger,
            &com_v,
            dom,
            &h,
            b,
            &proof
        ));
    }
}
