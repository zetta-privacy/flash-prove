//! Succinct verifier: MLE evaluation and matmul/pointwise constraint checks
//! built on the FRI commitment + the linear-claim bridge, replacing the O(n)
//! `decode_exact` + full model recompute.

use crate::field::{Ext, Fp};
use crate::fri::{self, Challenger, Pcs, Prover};
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
    ctx: &mut Prover,
    com_v: &fri::Commitment,
    pd_v: &fri::ProverData,
    dom_v: fri::Domain,
    v: &[Fp],
    point: &[Ext],
) -> [InnerProductProof; 3] {
    let [a, b, c] = eq_components(point, log2(v.len()));
    [
        prove_inner_product_committed_given(ctx, com_v, pd_v, dom_v, v, &a),
        prove_inner_product_committed_given(ctx, com_v, pd_v, dom_v, v, &b),
        prove_inner_product_committed_given(ctx, com_v, pd_v, dom_v, v, &c),
    ]
}

/// Verify the committed MLE evaluation `ṽ(point) = claimed` from commitments and
/// the three linear-claim proofs alone (no `v` in the clear — only its length `n`).
/// The three linear claims' openings are appended to `acc`; the shared FRI proof is
/// checked once, at the top level, via [`fri::open_many_verify`].
pub fn verify_mle_committed(
    challenger: &Challenger,
    com_v: &fri::Commitment,
    dom_v: fri::Domain,
    n: usize,
    point: &[Ext],
    claimed: Ext,
    proofs: &[InnerProductProof; 3],
    values: &[Ext],
    acc: &mut Openings,
) -> bool {
    let [a, b, c] = eq_components(point, log2(n));
    let coeffs = <Ext as BasedVectorSpace<Fp>>::as_basis_coefficients_slice(&claimed);
    verify_inner_product_committed(
        challenger, com_v, dom_v, &a, coeffs[0], &proofs[0], values, acc,
    ) && verify_inner_product_committed(
        challenger, com_v, dom_v, &b, coeffs[1], &proofs[1], values, acc,
    ) && verify_inner_product_committed(
        challenger, com_v, dom_v, &c, coeffs[2], &proofs[2], values, acc,
    )
}

/// One FRI opening: the claimed value and its opening proof. (The verifier holds
/// this instead of any prover material.)
pub struct Opening {
    pub value: Ext,
    pub proof: fri::Proof,
}

/// A batch of FRI openings `(commitment, domain, point, value)`, accumulated across
/// every linear claim and verified with a single shared [`fri::open_many_verify`] at
/// the top level. This is what makes the verifier O(log n): one FRI proof covers all
/// openings instead of one per claim.
pub type Openings = Vec<(fri::Commitment, fri::Domain, Ext, Ext)>;

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
    pub idx: usize,
}

/// Build the [`InnerProductProof`] for an *already-committed* `v`: commit
/// `P = h·v` and `Q = (P − c0)/X`, derive the challenge points from the transcript,
/// and open at them (reusing the caller's `com_v`/`pd_v`). Used by the matmul and
/// IPA, where `v` is committed once and shared across several linear claims.
pub fn prove_inner_product_committed_given(
    ctx: &mut Prover,
    com_v: &fri::Commitment,
    pd_v: &fri::ProverData,
    dom: fri::Domain,
    v: &[Fp],
    h: &[Fp],
) -> InnerProductProof {
    let n = v.len();
    let c0 = constant_coeff(h, v);
    let p: Vec<Fp> = h.iter().zip(v).map(|(&x, &y)| x * y).collect();
    let (com_p, pd_p, _) = fri::commit_vec(&ctx.pcs, &p);
    let q = constant_quotient(h, v, c0);
    let q_evals = Radix2Dit::<Fp>::default().dft(q);
    let (com_q, pd_q, _) = fri::commit_vec(&ctx.pcs, &q_evals);
    let mut tr = fri::Transcript::new(&ctx.challenger);
    tr.observe(com_v);
    tr.observe(&com_p);
    tr.observe(&com_q);
    let k = tr.sample_index(n);
    let alpha = tr.sample_ext();
    let point_k = Ext::from(dom.element(k));
    let idx = ctx.record(com_v.clone(), pd_v.clone(), point_k);
    ctx.record(com_p.clone(), pd_p.clone(), point_k);
    ctx.record(com_p.clone(), pd_p.clone(), alpha);
    ctx.record(com_q.clone(), pd_q.clone(), alpha);
    InnerProductProof {
        com_p,
        com_q,
        c0,
        idx,
    }
}

/// Prove `⟨h, v⟩ = b` with `v` committed: commit `v`, then build the
/// [`InnerProductProof`]. Returns the commitment to `v`, its domain, and the proof.
pub fn prove_inner_product_committed(
    ctx: &mut Prover,
    v: &[Fp],
    h: &[Fp],
) -> (fri::Commitment, fri::Domain, InnerProductProof) {
    let (com_v, pd_v, dom) = fri::commit_vec(&ctx.pcs, v);
    let proof = prove_inner_product_committed_given(ctx, &com_v, &pd_v, dom, v, h);
    (com_v, dom, proof)
}

/// Verify `⟨h, v⟩ = b` from the committed linear-claim proof: replay the transcript
/// challenges, check the two identities, and append the four FRI openings to `acc`.
/// The actual FRI verification is a single shared [`fri::open_many_verify`] at the
/// top level, so this returns `true` iff the scalar checks pass.
pub fn verify_inner_product_committed(
    challenger: &Challenger,
    com_v: &fri::Commitment,
    dom: fri::Domain,
    h: &[Fp],
    b: Fp,
    proof: &InnerProductProof,
    values: &[Ext],
    acc: &mut Openings,
) -> bool {
    match ip_openings(challenger, com_v, dom, h, b, proof, values) {
        Some(openings) => {
            acc.extend(openings);
            true
        }
        None => false,
    }
}

pub fn ip_openings(
    challenger: &Challenger,
    com_v: &fri::Commitment,
    dom: fri::Domain,
    h: &[Fp],
    b: Fp,
    proof: &InnerProductProof,
    values: &[Ext],
) -> Option<Vec<(fri::Commitment, fri::Domain, Ext, Ext)>> {
    let n = h.len();
    let mut tr = fri::Transcript::new(challenger);
    tr.observe(com_v);
    tr.observe(&proof.com_p);
    tr.observe(&proof.com_q);
    let k = tr.sample_index(n);
    let alpha = tr.sample_ext();
    let point_k = Ext::from(dom.element(k));
    let v_k = values[proof.idx];
    let p_k = values[proof.idx + 1];
    let p_alpha = values[proof.idx + 2];
    let q_alpha = values[proof.idx + 3];
    if proof.c0 * Fp::new(n as u64) != b {
        return None;
    }
    if p_k != Ext::from(h[k]) * v_k {
        return None;
    }
    if p_alpha != Ext::from(proof.c0) + alpha * q_alpha {
        return None;
    }
    Some(vec![
        (com_v.clone(), dom, point_k, v_k),
        (proof.com_p.clone(), dom, point_k, p_k),
        (proof.com_p.clone(), dom, alpha, p_alpha),
        (proof.com_q.clone(), dom, alpha, q_alpha),
    ])
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
        let mut ctx = fri::Prover::new(1, 10);
        let v = [f(5), f(6), f(7), f(8)];
        let point = vec![Ext::from(f(2)), Ext::from(f(3))];
        let claimed = mle_value(&v, &point);
        let (com_v, pd_v, dom) = fri::commit_vec(&ctx.pcs, &v);
        let proofs = prove_mle_committed(&mut ctx, &com_v, &pd_v, dom, &v, &point);
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = Openings::new();
        assert!(verify_mle_committed(
            &ctx.challenger,
            &com_v,
            dom,
            4,
            &point,
            claimed,
            &proofs,
            &values,
            &mut acc,
        ));
        assert!(fri::open_many_verify(
            &ctx.pcs,
            &ctx.challenger,
            &acc,
            &fri_proof
        ));
    }

    #[test]
    fn verify_mle_committed_rejects_wrong_claim() {
        let mut ctx = fri::Prover::new(1, 10);
        let v = [f(5), f(6), f(7), f(8)];
        let point = vec![Ext::from(f(2)), Ext::from(f(3))];
        let claimed = mle_value(&v, &point);
        let (com_v, pd_v, dom) = fri::commit_vec(&ctx.pcs, &v);
        let proofs = prove_mle_committed(&mut ctx, &com_v, &pd_v, dom, &v, &point);
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        // Wrong claimed value → the basis coefficients shift, breaking the linear claims.
        let wrong = claimed + Ext::ONE;
        let mut acc = Openings::new();
        assert!(!verify_mle_committed(
            &ctx.challenger,
            &com_v,
            dom,
            4,
            &point,
            wrong,
            &proofs,
            &values,
            &mut acc,
        ));
    }

    #[test]
    fn committed_inner_product_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        // v = [5,6,7,8], h = [1,2,3,4], ⟨h,v⟩ = 5+12+21+32 = 70.
        let v = [f(5), f(6), f(7), f(8)];
        let h = [f(1), f(2), f(3), f(4)];
        let b = f(70);
        let (com_v, dom, proof) = prove_inner_product_committed(&mut ctx, &v, &h);
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = Openings::new();
        assert!(verify_inner_product_committed(
            &ctx.challenger,
            &com_v,
            dom,
            &h,
            b,
            &proof,
            &values,
            &mut acc,
        ));
        assert!(fri::open_many_verify(
            &ctx.pcs,
            &ctx.challenger,
            &acc,
            &fri_proof
        ));
    }

    #[test]
    fn committed_inner_product_rejects_wrong_b() {
        let mut ctx = fri::Prover::new(1, 10);
        let v = [f(5), f(6), f(7), f(8)];
        let h = [f(1), f(2), f(3), f(4)];
        let b = f(71); // ⟨h,v⟩ = 70
        let (com_v, dom, proof) = prove_inner_product_committed(&mut ctx, &v, &h);
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = Openings::new();
        assert!(!verify_inner_product_committed(
            &ctx.challenger,
            &com_v,
            dom,
            &h,
            b,
            &proof,
            &values,
            &mut acc,
        ));
    }

    #[test]
    fn committed_inner_product_rejects_tampered_opening() {
        let mut ctx = fri::Prover::new(1, 10);
        let v = [f(5), f(6), f(7), f(8)];
        let h = [f(1), f(2), f(3), f(4)];
        let b = f(70);
        let (com_v, dom, proof) = prove_inner_product_committed(&mut ctx, &v, &h);
        let fri_proof = ctx.open_all();
        let mut values = ctx.values.clone();
        // Tamper the claimed P(ω^k) value (idx+1): the P = h·v check must reject.
        values[proof.idx + 1] += Ext::ONE;
        let mut acc = Openings::new();
        assert!(!verify_inner_product_committed(
            &ctx.challenger,
            &com_v,
            dom,
            &h,
            b,
            &proof,
            &values,
            &mut acc,
        ));
    }

    #[test]
    fn batch_inner_products_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        let v = [f(5), f(6), f(7), f(8)];
        let h1 = [f(1), f(0), f(0), f(0)];
        let h2 = [f(0), f(1), f(0), f(0)];
        let h3 = [f(0), f(0), f(1), f(0)];
        let r = [f(1), f(2), f(3)];
        // ⟨h_i, v⟩ = 5, 6, 7; batched ⟨h,v⟩ = 1·5 + 2·6 + 3·7 = 38.
        let (h, b) = batch_inner_products(&[&h1, &h2, &h3], &[f(5), f(6), f(7)], &r);
        let (com_v, dom, proof) = prove_inner_product_committed(&mut ctx, &v, &h);
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = Openings::new();
        assert!(verify_inner_product_committed(
            &ctx.challenger,
            &com_v,
            dom,
            &h,
            b,
            &proof,
            &values,
            &mut acc,
        ));
        assert!(fri::open_many_verify(
            &ctx.pcs,
            &ctx.challenger,
            &acc,
            &fri_proof
        ));
    }

    #[test]
    fn batch_inner_products_rejects_wrong() {
        let mut ctx = fri::Prover::new(1, 10);
        let v = [f(5), f(6), f(7), f(8)];
        let h1 = [f(1), f(0), f(0), f(0)];
        let h2 = [f(0), f(1), f(0), f(0)];
        let h3 = [f(0), f(0), f(1), f(0)];
        let r = [f(1), f(2), f(3)];
        // Wrong b3 = 8 (true ⟨h3, v⟩ = 7) shifts the batched claim by r[2] = 3.
        let (h, b) = batch_inner_products(&[&h1, &h2, &h3], &[f(5), f(6), f(8)], &r);
        let (com_v, dom, proof) = prove_inner_product_committed(&mut ctx, &v, &h);
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = Openings::new();
        assert!(!verify_inner_product_committed(
            &ctx.challenger,
            &com_v,
            dom,
            &h,
            b,
            &proof,
            &values,
            &mut acc,
        ));
    }
}
