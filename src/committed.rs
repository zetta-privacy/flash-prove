//! Committed matmul: verify `y = W·x` (`out × in` matrix times vector) on FRI-
//! committed data, via the fold → inner-product reduction.
//!
//! The verifier holds only commitments + opening proofs (no prover material). The
//! reduction:
//!
//!  1. sample `r ∈ F^out`, `s ∈ F^in` from the transcript,
//!  2. fold the output `yc = ⟨r, y⟩` and the weight rows `wc[j] = Σ_i r_i·W[i,j]`,
//!  3. check the folds with three linear claims (`⟨r,y⟩ = yc`, `⟨s,wc⟩ = b_s`,
//!     `⟨H,W⟩ = b_s` where `H[i,j] = r_i·s_j`), and
//!  4. check the single inner product `⟨wc, x⟩ = yc` via the committed IPA.

use crate::field::Fp;
use crate::fri::{self, Challenger, Pcs};
use crate::ipa::{prove_ipa_committed, verify_ipa_committed, CommittedIpa};
use crate::zk::{
    prove_inner_product_committed_given, verify_inner_product_committed, InnerProductProof,
};


/// The committed matmul proof (prover's output): commitments, the folded claims,
/// the three linear-claim proofs, and the IPA proof.
pub struct MatmulProof {
    pub com_w: fri::Commitment,
    pub com_x: fri::Commitment,
    pub com_y: fri::Commitment,
    pub com_wc: fri::Commitment,
    pub dom_w: fri::Domain,
    pub dom_x: fri::Domain,
    pub dom_y: fri::Domain,
    pub yc: Fp,
    pub b_s: Fp,
    pub claim_y: InnerProductProof,
    pub claim_wc: InnerProductProof,
    pub claim_w: InnerProductProof,
    pub ipa: CommittedIpa,
}

/// Prove `y = W·x` on committed data. `w` is `out × in` (row-major), `x` is `in`,
/// `y` is `out`.
pub fn prove_matmul_committed(
    pcs: &Pcs,
    challenger: &Challenger,
    w: &[Fp],
    x: &[Fp],
    y: &[Fp],
    out: usize,
    in_: usize,
) -> MatmulProof {
    assert_eq!(w.len(), out * in_);
    assert_eq!(x.len(), in_);
    assert_eq!(y.len(), out);

    let (com_w, pd_w, dom_w) = fri::commit_vec(pcs, w);
    let (com_x, _pd_x, dom_x) = fri::commit_vec(pcs, x);
    let (com_y, pd_y, dom_y) = fri::commit_vec(pcs, y);

    // Sample the fold challenges from the transcript.
    let mut tr = fri::Transcript::new(challenger);
    tr.observe(&com_w);
    tr.observe(&com_x);
    tr.observe(&com_y);
    let r: Vec<Fp> = (0..out).map(|_| tr.sample_val()).collect();
    let s: Vec<Fp> = (0..in_).map(|_| tr.sample_val()).collect();

    // Fold output and weight rows.
    let yc: Fp = r.iter().zip(y).map(|(&ri, &yi)| ri * yi).sum();
    let wc: Vec<Fp> = (0..in_)
        .map(|j| (0..out).map(|i| r[i] * w[i * in_ + j]).sum())
        .collect();
    let (com_wc, pd_wc, dom_wc) = fri::commit_vec(pcs, &wc);
    let b_s: Fp = s.iter().zip(&wc).map(|(&sj, &wcj)| sj * wcj).sum();
    let h: Vec<Fp> = (0..out * in_).map(|k| r[k / in_] * s[k % in_]).collect();

    // Linear claims and the IPA.
    let claim_y =
        prove_inner_product_committed_given(pcs, challenger, &com_y, &pd_y, dom_y, y, &r);
    let claim_wc =
        prove_inner_product_committed_given(pcs, challenger, &com_wc, &pd_wc, dom_wc, &wc, &s);
    let claim_w =
        prove_inner_product_committed_given(pcs, challenger, &com_w, &pd_w, dom_w, w, &h);
    let ipa = prove_ipa_committed(pcs, challenger, &wc, x, yc);

    MatmulProof {
        com_w,
        com_x,
        com_y,
        com_wc,
        dom_w,
        dom_x,
        dom_y,
        yc,
        b_s,
        claim_y,
        claim_wc,
        claim_w,
        ipa,
    }
}

/// Verify `y = W·x` from the committed matmul proof: replay the fold challenges,
/// check the three linear claims and the IPA — all from commitments alone.
pub fn verify_matmul_committed(
    pcs: &Pcs,
    challenger: &Challenger,
    out: usize,
    in_: usize,
    proof: &MatmulProof,
) -> bool {
    let mut tr = fri::Transcript::new(challenger);
    tr.observe(&proof.com_w);
    tr.observe(&proof.com_x);
    tr.observe(&proof.com_y);
    let r: Vec<Fp> = (0..out).map(|_| tr.sample_val()).collect();
    let s: Vec<Fp> = (0..in_).map(|_| tr.sample_val()).collect();
    let h: Vec<Fp> = (0..out * in_).map(|k| r[k / in_] * s[k % in_]).collect();

    if !verify_inner_product_committed(pcs, challenger, &proof.com_y, proof.dom_y, &r, proof.yc, &proof.claim_y)
    {
        return false;
    }
    if !verify_inner_product_committed(
        pcs,
        challenger,
        &proof.com_wc,
        proof.dom_x,
        &s,
        proof.b_s,
        &proof.claim_wc,
    ) {
        return false;
    }
    if !verify_inner_product_committed(
        pcs,
        challenger,
        &proof.com_w,
        proof.dom_w,
        &h,
        proof.b_s,
        &proof.claim_w,
    ) {
        return false;
    }
    verify_ipa_committed(pcs, challenger, proof.yc, &proof.ipa)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(x: u64) -> Fp {
        Fp::new(x)
    }

    #[test]
    fn matmul_committed_honest() {
        let (pcs, challenger) = fri::new_pcs(1, 10);
        // W = [[1,2],[3,4]], x = [5,6] → y = [1·5+2·6, 3·5+4·6] = [17, 39].
        let w = [f(1), f(2), f(3), f(4)];
        let x = [f(5), f(6)];
        let y = [f(17), f(39)];
        let proof = prove_matmul_committed(&pcs, &challenger, &w, &x, &y, 2, 2);
        assert!(verify_matmul_committed(&pcs, &challenger, 2, 2, &proof));
    }

    #[test]
    fn matmul_committed_rejects_wrong_y() {
        let (pcs, challenger) = fri::new_pcs(1, 10);
        let w = [f(1), f(2), f(3), f(4)];
        let x = [f(5), f(6)];
        let y = [f(17), f(40)]; // y[1] should be 39
        let proof = prove_matmul_committed(&pcs, &challenger, &w, &x, &y, 2, 2);
        assert!(!verify_matmul_committed(&pcs, &challenger, 2, 2, &proof));
    }
}
