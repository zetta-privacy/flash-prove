//! Committed pointwise `mul`: verify `out[i] = a[i]·b[i]` on FRI-committed data,
//! without a degree-3 product sumcheck.
//!
//! Fold a public challenge `r` into one operand (`g = r ⊙ a`, a diagonal linear
//! claim), then the bilinear `a ⊙ b` reduces to one inner product:
//!
//!   ⟨g, b⟩ = ⟨r, out⟩   (with g = r ⊙ a)
//!
//! checked by one IPA (`⟨g, b⟩ = c`) and one linear claim (`⟨r, out⟩ = c`).

use crate::field::Fp;
use crate::fri::{self, Challenger, Pcs};
use crate::ipa::{prove_ipa_committed, verify_ipa_committed, CommittedIpa};
use crate::zk::{
    prove_inner_product_committed_given, verify_inner_product_committed, InnerProductProof,
};
use p3_field::PrimeCharacteristicRing;

fn next_pow2(n: usize) -> usize {
    n.next_power_of_two()
}

fn pad(mut v: Vec<Fp>, n: usize) -> Vec<Fp> {
    v.resize(next_pow2(n), Fp::ZERO);
    v
}

pub struct MulProof {
    pub n: usize,
    pub com_a: fri::Commitment,
    pub dom_a: fri::Domain,
    pub com_b: fri::Commitment,
    pub dom_b: fri::Domain,
    pub com_out: fri::Commitment,
    pub dom_out: fri::Domain,
    pub com_g: fri::Commitment,
    pub dom_g: fri::Domain,
    /// g = r ⊙ a: ⟨s, g⟩ = b_s = ⟨s ⊙ r, a⟩.
    pub diag_g: InnerProductProof,
    pub diag_a: InnerProductProof,
    pub b_s: Fp,
    /// ⟨r, out⟩ = c.
    pub out_claim: InnerProductProof,
    /// ⟨g, b⟩ = c.
    pub ipa: CommittedIpa,
    pub c: Fp,
}

pub fn prove_mul_committed(
    pcs: &Pcs,
    challenger: &Challenger,
    a: &[Fp],
    b: &[Fp],
    out: &[Fp],
) -> MulProof {
    let n = a.len();
    assert_eq!(b.len(), n);
    assert_eq!(out.len(), n);

    let a = pad(a.to_vec(), n);
    let b = pad(b.to_vec(), n);
    let out = pad(out.to_vec(), n);

    let (com_a, pd_a, dom_a) = fri::commit_vec(pcs, &a);
    let (com_b, _pd_b, dom_b) = fri::commit_vec(pcs, &b);
    let (com_out, pd_out, dom_out) = fri::commit_vec(pcs, &out);

    let mut tr = fri::Transcript::new(challenger);
    tr.observe(&com_a);
    tr.observe(&com_b);
    tr.observe(&com_out);
    let mut r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    r.resize(next_pow2(n), Fp::ZERO);
    let mut s: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    s.resize(next_pow2(n), Fp::ZERO);

    let g: Vec<Fp> = (0..next_pow2(n)).map(|i| r[i] * a[i]).collect();
    let (com_g, pd_g, dom_g) = fri::commit_vec(pcs, &g);
    let b_s: Fp = s.iter().zip(&g).map(|(&sj, &gj)| sj * gj).sum();
    let sr: Vec<Fp> = (0..next_pow2(n)).map(|i| s[i] * r[i]).collect();
    let c: Fp = g.iter().zip(&b).map(|(&gk, &bk)| gk * bk).sum();

    let diag_g = prove_inner_product_committed_given(pcs, challenger, &com_g, &pd_g, dom_g, &g, &s);
    let diag_a =
        prove_inner_product_committed_given(pcs, challenger, &com_a, &pd_a, dom_a, &a, &sr);
    let out_claim =
        prove_inner_product_committed_given(pcs, challenger, &com_out, &pd_out, dom_out, &out, &r);
    let ipa = prove_ipa_committed(pcs, challenger, &g, &b, c);

    MulProof {
        n,
        com_a,
        dom_a,
        com_b,
        dom_b,
        com_out,
        dom_out,
        com_g,
        dom_g,
        diag_g,
        diag_a,
        b_s,
        out_claim,
        ipa,
        c,
    }
}

pub fn verify_mul_committed(pcs: &Pcs, challenger: &Challenger, proof: &MulProof) -> bool {
    let n = proof.n;

    let mut tr = fri::Transcript::new(challenger);
    tr.observe(&proof.com_a);
    tr.observe(&proof.com_b);
    tr.observe(&proof.com_out);
    let mut r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    r.resize(next_pow2(n), Fp::ZERO);
    let mut s: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    s.resize(next_pow2(n), Fp::ZERO);
    let sr: Vec<Fp> = (0..next_pow2(n)).map(|i| s[i] * r[i]).collect();

    // g = r ⊙ a (diagonal).
    if !verify_inner_product_committed(
        pcs,
        challenger,
        &proof.com_g,
        proof.dom_g,
        &s,
        proof.b_s,
        &proof.diag_g,
    ) {
        return false;
    }
    if !verify_inner_product_committed(
        pcs,
        challenger,
        &proof.com_a,
        proof.dom_a,
        &sr,
        proof.b_s,
        &proof.diag_a,
    ) {
        return false;
    }
    // ⟨r, out⟩ = c.
    if !verify_inner_product_committed(
        pcs,
        challenger,
        &proof.com_out,
        proof.dom_out,
        &r,
        proof.c,
        &proof.out_claim,
    ) {
        return false;
    }
    // ⟨g, b⟩ = c.
    verify_ipa_committed(pcs, challenger, proof.c, &proof.ipa)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(x: u64) -> Fp {
        Fp::new(x)
    }

    #[test]
    fn mul_committed_honest() {
        let (pcs, challenger) = fri::new_pcs(1, 10);
        // a = [1,2,3,4], b = [5,6,7,8], out = a⊙b = [5,12,21,32].
        let a = [f(1), f(2), f(3), f(4)];
        let b = [f(5), f(6), f(7), f(8)];
        let out = [f(5), f(12), f(21), f(32)];
        let proof = prove_mul_committed(&pcs, &challenger, &a, &b, &out);
        assert!(verify_mul_committed(&pcs, &challenger, &proof));
    }

    #[test]
    fn mul_committed_rejects_wrong() {
        let (pcs, challenger) = fri::new_pcs(1, 10);
        let a = [f(1), f(2), f(3), f(4)];
        let b = [f(5), f(6), f(7), f(8)];
        let out = [f(5), f(12), f(21), f(33)]; // out[3] should be 32
        let proof = prove_mul_committed(&pcs, &challenger, &a, &b, &out);
        assert!(!verify_mul_committed(&pcs, &challenger, &proof));
    }
}
