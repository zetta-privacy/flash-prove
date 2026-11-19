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

use crate::field::{Ext, Fp};
use crate::fri::{self, Challenger, Prover};
use crate::ipa::{prove_ipa_committed_given, verify_ipa_committed, CommittedIpa};
use crate::zk::{
    prove_inner_product_committed_given, verify_inner_product_committed, InnerProductProof,
    Openings,
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
    ctx: &mut Prover,
    w: &[Fp],
    x: &[Fp],
    y: &[Fp],
    out: usize,
    in_: usize,
) -> MatmulProof {
    assert_eq!(w.len(), out * in_);
    assert_eq!(x.len(), in_);
    assert_eq!(y.len(), out);

    let (com_w, pd_w, dom_w) = fri::commit_vec(&ctx.pcs, w);
    let (com_x, pd_x, dom_x) = fri::commit_vec(&ctx.pcs, x);
    let (com_y, pd_y, dom_y) = fri::commit_vec(&ctx.pcs, y);

    prove_matmul_committed_given(
        ctx, &com_w, &pd_w, dom_w, &com_x, &pd_x, dom_x, &com_y, &pd_y, dom_y, w, x, y, out, in_,
    )
}

/// Prove `y = W·x` on *already-committed* `W`, `x`, `y`. The derived folded-weight
/// vector `wc` and the IPA are committed fresh; `x` is reused via the IPA's
/// shared-commitment form so a single commitment to `x` backs both the matmul
/// transcript and the inner-product argument.
pub fn prove_matmul_committed_given(
    ctx: &mut Prover,
    com_w: &fri::Commitment,
    pd_w: &fri::ProverData,
    dom_w: fri::Domain,
    com_x: &fri::Commitment,
    pd_x: &fri::ProverData,
    dom_x: fri::Domain,
    com_y: &fri::Commitment,
    pd_y: &fri::ProverData,
    dom_y: fri::Domain,
    w: &[Fp],
    x: &[Fp],
    y: &[Fp],
    out: usize,
    in_: usize,
) -> MatmulProof {
    assert_eq!(w.len(), out * in_);
    assert_eq!(x.len(), in_);
    assert_eq!(y.len(), out);

    let mut tr = fri::Transcript::new(&ctx.challenger);
    tr.observe(com_w);
    tr.observe(com_x);
    tr.observe(com_y);
    let r: Vec<Fp> = (0..out).map(|_| tr.sample_val()).collect();
    let s: Vec<Fp> = (0..in_).map(|_| tr.sample_val()).collect();

    let yc: Fp = r.iter().zip(y).map(|(&ri, &yi)| ri * yi).sum();
    let wc: Vec<Fp> = (0..in_)
        .map(|j| (0..out).map(|i| r[i] * w[i * in_ + j]).sum())
        .collect();
    let (com_wc, pd_wc, dom_wc) = fri::commit_vec(&ctx.pcs, &wc);
    let b_s: Fp = s.iter().zip(&wc).map(|(&sj, &wcj)| sj * wcj).sum();
    let h: Vec<Fp> = (0..out * in_).map(|k| r[k / in_] * s[k % in_]).collect();

    let claim_y = prove_inner_product_committed_given(ctx, com_y, pd_y, dom_y, y, &r);
    let claim_wc = prove_inner_product_committed_given(ctx, &com_wc, &pd_wc, dom_wc, &wc, &s);
    let claim_w = prove_inner_product_committed_given(ctx, com_w, pd_w, dom_w, w, &h);
    let ipa =
        prove_ipa_committed_given(ctx, &com_wc, &pd_wc, dom_wc, com_x, pd_x, dom_x, &wc, x, yc);

    MatmulProof {
        com_w: com_w.clone(),
        com_x: com_x.clone(),
        com_y: com_y.clone(),
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
    challenger: &Challenger,
    out: usize,
    in_: usize,
    proof: &MatmulProof,
    values: &[Ext],
    acc: &mut Openings,
) -> bool {
    let mut tr = fri::Transcript::new(challenger);
    tr.observe(&proof.com_w);
    tr.observe(&proof.com_x);
    tr.observe(&proof.com_y);
    let r: Vec<Fp> = (0..out).map(|_| tr.sample_val()).collect();
    let s: Vec<Fp> = (0..in_).map(|_| tr.sample_val()).collect();
    let h: Vec<Fp> = (0..out * in_).map(|k| r[k / in_] * s[k % in_]).collect();

    if !verify_inner_product_committed(
        challenger,
        &proof.com_y,
        proof.dom_y,
        &r,
        proof.yc,
        &proof.claim_y,
        values,
        acc,
    ) {
        return false;
    }
    if !verify_inner_product_committed(
        challenger,
        &proof.com_wc,
        proof.dom_x,
        &s,
        proof.b_s,
        &proof.claim_wc,
        values,
        acc,
    ) {
        return false;
    }
    if !verify_inner_product_committed(
        challenger,
        &proof.com_w,
        proof.dom_w,
        &h,
        proof.b_s,
        &proof.claim_w,
        values,
        acc,
    ) {
        return false;
    }
    verify_ipa_committed(challenger, proof.yc, &proof.ipa, values, acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(x: u64) -> Fp {
        Fp::new(x)
    }

    #[test]
    fn matmul_committed_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        // W = [[1,2],[3,4]], x = [5,6] → y = [1·5+2·6, 3·5+4·6] = [17, 39].
        let w = [f(1), f(2), f(3), f(4)];
        let x = [f(5), f(6)];
        let y = [f(17), f(39)];
        let proof = prove_matmul_committed(&mut ctx, &w, &x, &y, 2, 2);
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = crate::zk::Openings::new();
        assert!(verify_matmul_committed(
            &ctx.challenger,
            2,
            2,
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
    fn matmul_committed_rejects_wrong_y() {
        let mut ctx = fri::Prover::new(1, 10);
        let w = [f(1), f(2), f(3), f(4)];
        let x = [f(5), f(6)];
        let y = [f(17), f(40)]; // y[1] should be 39
        let proof = prove_matmul_committed(&mut ctx, &w, &x, &y, 2, 2);
        ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = crate::zk::Openings::new();
        assert!(!verify_matmul_committed(
            &ctx.challenger,
            2,
            2,
            &proof,
            &values,
            &mut acc
        ));
    }
}
