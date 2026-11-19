//! Inner-product argument (Bulletproofs folding): verify `⟨a, b⟩ = c` in
//! O(log n) verifier rounds, for committed vectors `a`, `b`.
//!
//! This is the missing primitive for the matmul (and attention / MoE) check:
//! `C̃(r,s) = ⟨Ã(r,·), B̃(·,s)⟩` is an inner product of the committed row/column
//! MLEs. Each round the prover sends `L = ⟨a_lo, b_hi⟩`, `R = ⟨a_hi, b_lo⟩`; the
//! verifier samples `x` and folds `a' = a_lo·x + a_hi·x⁻¹`, `b' = b_lo·x⁻¹ +
//! b_hi·x`, with the new claim `c' = c + x²·L + x⁻²·R`.

use crate::field::{Ext, Fp};
use crate::fri::{self, Challenger, Pcs, Prover};
use crate::zk::{
    prove_inner_product_committed_given, verify_inner_product_committed, InnerProductProof,
    Openings,
};
use p3_field::{Field, PrimeCharacteristicRing};

/// Verify `⟨a, b⟩ = c` recursively (self-contained, in the clear). The challenge
/// `sample` is injected so tests are deterministic.
pub fn ipa_verify(a: &[Fp], b: &[Fp], c: Fp, mut sample: impl FnMut() -> Fp) -> bool {
    assert_eq!(a.len(), b.len());
    if a.len() == 1 {
        return a[0] * b[0] == c;
    }
    let half = a.len() / 2;
    let (a_lo, a_hi) = a.split_at(half);
    let (b_lo, b_hi) = b.split_at(half);
    let l = a_lo.iter().zip(b_hi).map(|(&x, &y)| x * y).sum::<Fp>();
    let r = a_hi.iter().zip(b_lo).map(|(&x, &y)| x * y).sum::<Fp>();
    let x = sample();
    let xi = x.inverse();
    let a_fold: Vec<Fp> = a_lo
        .iter()
        .zip(a_hi)
        .map(|(&al, &ar)| al * x + ar * xi)
        .collect();
    let b_fold: Vec<Fp> = b_lo
        .iter()
        .zip(b_hi)
        .map(|(&bl, &br)| bl * xi + br * x)
        .collect();
    let c_fold = c + x * x * l + xi * xi * r;
    ipa_verify(&a_fold, &b_fold, c_fold, sample)
}

/// The linear-combination coefficients for the base-case folded element: after
/// folding `n` times with challenges `x[0..n]`, the single surviving element of
/// the folded `a` vector is `Σ_j a[j]·Π_fold x[fold]^(1−2·bit_fold(j))`.
///
/// The IPA ([`ipa_verify`], [`prove_ipa_committed`]) splits `a.split_at(half)` each
/// round, so round `fold` folds the *high* bit `(j >> (n−1−fold)) & 1` — not the
/// low bit. This matches that convention.
pub fn fold_coeffs(n: usize, x: &[Fp]) -> Vec<Fp> {
    let mut coeffs = vec![Fp::ONE; 1 << n];
    for (j, c) in coeffs.iter_mut().enumerate() {
        for (fold, &xf) in x.iter().enumerate() {
            if (j >> (n - 1 - fold)) & 1 == 0 {
                *c = *c * xf;
            } else {
                *c = *c * xf.inverse();
            }
        }
    }
    coeffs
}

/// The committed inner-product-argument proof for `⟨a, b⟩ = c` (both `a` and `b`
/// FRI-committed). No prover material — the verifier checks this with commitments
/// and FRI opening proofs alone.
pub struct CommittedIpa {
    pub com_a: fri::Commitment,
    pub com_b: fri::Commitment,
    pub dom: fri::Domain,
    /// `(L_j, R_j)` per folding round.
    pub rounds: Vec<(Fp, Fp)>,
    pub a_prime: Fp,
    pub b_prime: Fp,
    /// Base-case linear claims `⟨coeff, a⟩ = a'` and `⟨coeff⁻¹, b⟩ = b'`.
    pub a_claim: InnerProductProof,
    pub b_claim: InnerProductProof,
}

/// Build the base-case [`InnerProductProof`] for an already-committed `v`: commit
/// `P = h·v` and `Q = (P − c0)/X`, derive the challenge points from the transcript,
/// and open at them (reusing the caller's `com_v`/`pd_v`).
#[allow(clippy::too_many_arguments)]
fn prove_base_case(
    ctx: &mut Prover,
    com_v: &fri::Commitment,
    pd_v: &fri::ProverData,
    dom: fri::Domain,
    v: &[Fp],
    h: &[Fp],
) -> InnerProductProof {
    prove_inner_product_committed_given(ctx, com_v, pd_v, dom, v, h)
}

/// Prove `⟨a, b⟩ = c` with both vectors *already committed* (the caller supplies
/// the commitments and prover data). See [`prove_ipa_committed`] for the form that
/// commits them itself. `a` and `b` must share a power-of-two length (hence a
/// domain).
pub fn prove_ipa_committed_given(
    ctx: &mut Prover,
    com_a: &fri::Commitment,
    pd_a: &fri::ProverData,
    dom_a: fri::Domain,
    com_b: &fri::Commitment,
    pd_b: &fri::ProverData,
    dom_b: fri::Domain,
    a: &[Fp],
    b: &[Fp],
    c: Fp,
) -> CommittedIpa {
    let log_n = a.len().trailing_zeros() as usize;
    assert_eq!(b.len(), a.len());
    assert!(a.len().is_power_of_two());

    let mut tr = fri::Transcript::new(&ctx.challenger);
    tr.observe(com_a);
    tr.observe(com_b);

    let mut a_cur = a.to_vec();
    let mut b_cur = b.to_vec();
    let mut c_cur = c;
    let mut rounds = Vec::with_capacity(log_n);
    let mut x = Vec::with_capacity(log_n);
    for _ in 0..log_n {
        let half = a_cur.len() / 2;
        let (a_lo, a_hi) = a_cur.split_at(half);
        let (b_lo, b_hi) = b_cur.split_at(half);
        let l = a_lo.iter().zip(b_hi).map(|(&p, &q)| p * q).sum::<Fp>();
        let r = a_hi.iter().zip(b_lo).map(|(&p, &q)| p * q).sum::<Fp>();
        rounds.push((l, r));
        tr.observe_val(l);
        tr.observe_val(r);
        let xf = tr.sample_val();
        x.push(xf);
        let xi = xf.inverse();
        a_cur = a_lo
            .iter()
            .zip(a_hi)
            .map(|(&al, &ar)| al * xf + ar * xi)
            .collect();
        b_cur = b_lo
            .iter()
            .zip(b_hi)
            .map(|(&bl, &br)| bl * xi + br * xf)
            .collect();
        c_cur = c_cur + xf * xf * l + xi * xi * r;
    }

    let a_prime = a_cur[0];
    let b_prime = b_cur[0];

    let coeff = fold_coeffs(log_n, &x);
    let coeff_inv: Vec<Fp> = coeff.iter().map(|&v| v.inverse()).collect();
    let a_claim = prove_base_case(ctx, com_a, pd_a, dom_a, a, &coeff);
    let b_claim = prove_base_case(ctx, com_b, pd_b, dom_b, b, &coeff_inv);

    CommittedIpa {
        com_a: com_a.clone(),
        com_b: com_b.clone(),
        dom: dom_a,
        rounds,
        a_prime,
        b_prime,
        a_claim,
        b_claim,
    }
}

/// Prove `⟨a, b⟩ = c` with both vectors committed: fold via the IPA (sending
/// `L_j`, `R_j`), sampling each folding challenge `x_j` from the Fiat–Shamir
/// transcript (interleaved with the rounds), and commit the two base-case linear
/// claims.
pub fn prove_ipa_committed(ctx: &mut Prover, a: &[Fp], b: &[Fp], c: Fp) -> CommittedIpa {
    let (com_a, pd_a, dom) = fri::commit_vec(&ctx.pcs, a);
    let (com_b, pd_b, _) = fri::commit_vec(&ctx.pcs, b);
    prove_ipa_committed_given(ctx, &com_a, &pd_a, dom, &com_b, &pd_b, dom, a, b, c)
}

/// Verify `⟨a, b⟩ = c` from the committed IPA proof: replay the Fiat–Shamir
/// challenges from the transcript (interleaved with the rounds), fold the claim,
/// check the base case `a'·b' = c'`, and discharge the two linear claims via the
/// O(log n) committed inner-product argument.
pub fn verify_ipa_committed(
    challenger: &Challenger,
    c: Fp,
    proof: &CommittedIpa,
    values: &[Ext],
    acc: &mut Openings,
) -> bool {
    let log_n = proof.rounds.len();
    let mut tr = fri::Transcript::new(challenger);
    tr.observe(&proof.com_a);
    tr.observe(&proof.com_b);

    let mut c_cur = c;
    let mut x = Vec::with_capacity(log_n);
    for j in 0..log_n {
        let (l, r) = proof.rounds[j];
        tr.observe_val(l);
        tr.observe_val(r);
        let xf = tr.sample_val();
        x.push(xf);
        let xi = xf.inverse();
        c_cur = c_cur + xf * xf * l + xi * xi * r;
    }
    if proof.a_prime * proof.b_prime != c_cur {
        return false;
    }
    let coeff = fold_coeffs(log_n, &x);
    let coeff_inv: Vec<Fp> = coeff.iter().map(|&v| v.inverse()).collect();
    verify_inner_product_committed(
        challenger,
        &proof.com_a,
        proof.dom,
        &coeff,
        proof.a_prime,
        &proof.a_claim,
        values,
        acc,
    ) && verify_inner_product_committed(
        challenger,
        &proof.com_b,
        proof.dom,
        &coeff_inv,
        proof.b_prime,
        &proof.b_claim,
        values,
        acc,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::from_u64;

    fn f(x: u64) -> Fp {
        Fp::new(x)
    }

    #[test]
    fn ipa_honest() {
        // a = [1,2,3,4], b = [5,6,7,8], ⟨a,b⟩ = 5+12+21+32 = 70.
        let a = [f(1), f(2), f(3), f(4)];
        let b = [f(5), f(6), f(7), f(8)];
        let c = a.iter().zip(&b).map(|(&x, &y)| x * y).sum::<Fp>();
        let sample = || f(9);
        assert!(ipa_verify(&a, &b, c, sample));
    }

    #[test]
    fn ipa_rejects_wrong_c() {
        let a = [f(1), f(2), f(3), f(4)];
        let b = [f(5), f(6), f(7), f(8)];
        let c_wrong = f(71); // ⟨a,b⟩ = 70
        let sample = || f(9);
        assert!(!ipa_verify(&a, &b, c_wrong, sample));
    }

    #[test]
    fn committed_inner_product() {
        use crate::fri;
        use crate::linear_code::{constant_coeff, constant_quotient};
        use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};

        let (pcs, challenger) = fri::new_pcs(1, 10);

        // a = [1,2,3,4], b = [5,6,7,8], ⟨a,b⟩ = 70.
        let a = [f(1), f(2), f(3), f(4)];
        let b = [f(5), f(6), f(7), f(8)];
        let c = f(70);

        // n = 2 folding challenges.
        let x = [f(9), f(10)];

        // The folded base-case elements: a' = ⟨coeff, a⟩, b' = ⟨coeff⁻¹, b⟩.
        let coeff = fold_coeffs(2, &x);
        let coeff_inv: Vec<Fp> = coeff.iter().map(|&c| c.inverse()).collect();
        let a_prime = coeff.iter().zip(&a).map(|(&c, &v)| c * v).sum::<Fp>();
        let b_prime = coeff_inv.iter().zip(&b).map(|(&c, &v)| c * v).sum::<Fp>();

        // Verify a' = ⟨coeff, a⟩ and b' = ⟨coeff⁻¹, b⟩ via the linear claim (the
        // folding of the committed vectors is checkable).
        for (v, h, claimed) in [(&a, &coeff, a_prime), (&b, &coeff_inv, b_prime)] {
            let c0 = constant_coeff(h, v);
            assert_eq!(c0 * f(4), claimed);
            let p: Vec<Fp> = h.iter().zip(v).map(|(&x, &y)| x * y).collect();
            let (cp, pdp, dp) = fri::commit_vec(&pcs, &p);
            let q = constant_quotient(h, v, c0);
            let q_evals = Radix2Dit::<Fp>::default().dft(q);
            let (cq, pdq, dq) = fri::commit_vec(&pcs, &q_evals);
            let alpha = crate::field::Ext::from(f(11));
            let pa = fri::open_at(&pcs, &challenger, &cp, &pdp, dp, alpha);
            let qa = fri::open_at(&pcs, &challenger, &cq, &pdq, dq, alpha);
            assert_eq!(pa, crate::field::Ext::from(c0) + alpha * qa);
        }
    }

    #[test]
    fn committed_ipa_honest() {
        let mut ctx = crate::fri::Prover::new(1, 10);
        // a = [1,2,3,4], b = [5,6,7,8], ⟨a,b⟩ = 70.
        let a = [f(1), f(2), f(3), f(4)];
        let b = [f(5), f(6), f(7), f(8)];
        let c = f(70);
        let proof = prove_ipa_committed(&mut ctx, &a, &b, c);
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = Openings::new();
        assert!(verify_ipa_committed(
            &ctx.challenger,
            c,
            &proof,
            &values,
            &mut acc
        ));
        assert!(fri::open_many_verify(
            &ctx.pcs,
            &ctx.challenger,
            &acc,
            &fri_proof
        ));
    }

    #[test]
    fn committed_ipa_rejects_wrong_c() {
        let mut ctx = crate::fri::Prover::new(1, 10);
        let a = [f(1), f(2), f(3), f(4)];
        let b = [f(5), f(6), f(7), f(8)];
        let c = f(70);
        let proof = prove_ipa_committed(&mut ctx, &a, &b, c);
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = Openings::new();
        assert!(!verify_ipa_committed(
            &ctx.challenger,
            f(71),
            &proof,
            &values,
            &mut acc
        ));
    }
}
