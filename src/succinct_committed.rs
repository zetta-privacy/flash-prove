//! Committed `verify_succinct`: wire the committed matmul + linear claim into a
//! full model verifier for a linear FFN (matmul → add → matmul), all on
//! FRI-committed data — the verifier holds only commitments + opening proofs.
//!
//! This is the final assembly of the committed primitives for the *linear* ops
//! (the dominant matmuls, plus `add`). The nonlinear lookups (SiLU/requant) wire
//! to the committed LogUp, and `mul` to a committed degree-3 sumcheck, which are
//! separate follow-ons.

use crate::committed::{prove_matmul_committed, verify_matmul_committed, MatmulProof};
use crate::field::Fp;
use crate::fri::{self, Challenger, Pcs};
use crate::mul_committed::{prove_mul_committed, verify_mul_committed, MulProof};
use crate::zk::{
    prove_inner_product_committed_given, verify_inner_product_committed, InnerProductProof,
};


/// A committed linear-FFN proof: two matmul proofs and one `add` (three linear
/// claims at a shared random `r`).
pub struct LinearFfnProof {
    pub mm1: MatmulProof, // h1 = W1·x
    pub mm2: MatmulProof, // logits = W2·residual
    // add: residual = h1 + x, checked via ⟨r,residual⟩ = ⟨r,h1⟩ + ⟨r,x⟩.
    pub com_h1: fri::Commitment,
    pub dom_h1: fri::Domain,
    pub com_x: fri::Commitment,
    pub dom_x: fri::Domain,
    pub com_res: fri::Commitment,
    pub dom_res: fri::Domain,
    pub r: Vec<Fp>,
    pub claim_h1: InnerProductProof,
    pub claim_x: InnerProductProof,
    pub claim_res: InnerProductProof,
    pub b_h1: Fp,
    pub b_x: Fp,
    pub b_res: Fp,
}

pub fn prove_linear_ffn_committed(
    pcs: &Pcs,
    challenger: &Challenger,
    w1: &[Fp],
    x: &[Fp],
    h1: &[Fp],
    w2: &[Fp],
    residual: &[Fp],
    logits: &[Fp],
    out1: usize,
    in_: usize,
    out2: usize,
) -> LinearFfnProof {
    // Commit the shared activations for the add check.
    let (com_h1, pd_h1, dom_h1) = fri::commit_vec(pcs, h1);
    let (com_x, pd_x, dom_x) = fri::commit_vec(pcs, x);
    let (com_res, pd_res, dom_res) = fri::commit_vec(pcs, residual);

    // Sample the shared r for the add.
    let mut tr = fri::Transcript::new(challenger);
    tr.observe(&com_h1);
    tr.observe(&com_x);
    tr.observe(&com_res);
    let r: Vec<Fp> = (0..in_).map(|_| tr.sample_val()).collect();

    let b_h1: Fp = r.iter().zip(h1).map(|(&ri, &vi)| ri * vi).sum();
    let b_x: Fp = r.iter().zip(x).map(|(&ri, &vi)| ri * vi).sum();
    let b_res: Fp = r.iter().zip(residual).map(|(&ri, &vi)| ri * vi).sum();

    let claim_h1 =
        prove_inner_product_committed_given(pcs, challenger, &com_h1, &pd_h1, dom_h1, h1, &r);
    let claim_x = prove_inner_product_committed_given(pcs, challenger, &com_x, &pd_x, dom_x, x, &r);
    let claim_res = prove_inner_product_committed_given(
        pcs, challenger, &com_res, &pd_res, dom_res, residual, &r,
    );

    let mm1 = prove_matmul_committed(pcs, challenger, w1, x, h1, out1, in_);
    let mm2 = prove_matmul_committed(pcs, challenger, w2, residual, logits, out2, out1);

    LinearFfnProof {
        mm1,
        mm2,
        com_h1,
        dom_h1,
        com_x,
        dom_x,
        com_res,
        dom_res,
        r,
        claim_h1,
        claim_x,
        claim_res,
        b_h1,
        b_x,
        b_res,
    }
}

pub fn verify_linear_ffn_committed(
    pcs: &Pcs,
    challenger: &Challenger,
    out1: usize,
    in_: usize,
    out2: usize,
    proof: &LinearFfnProof,
) -> bool {
    // Matmuls.
    if !verify_matmul_committed(pcs, challenger, out1, in_, &proof.mm1) {
        return false;
    }
    if !verify_matmul_committed(pcs, challenger, out2, out1, &proof.mm2) {
        return false;
    }
    // Add: replay r, check the three linear claims, and the scalar relation.
    let mut tr = fri::Transcript::new(challenger);
    tr.observe(&proof.com_h1);
    tr.observe(&proof.com_x);
    tr.observe(&proof.com_res);
    let r: Vec<Fp> = (0..in_).map(|_| tr.sample_val()).collect();
    if !verify_inner_product_committed(
        pcs,
        challenger,
        &proof.com_h1,
        proof.dom_h1,
        &r,
        proof.b_h1,
        &proof.claim_h1,
    ) {
        return false;
    }
    if !verify_inner_product_committed(
        pcs,
        challenger,
        &proof.com_x,
        proof.dom_x,
        &r,
        proof.b_x,
        &proof.claim_x,
    ) {
        return false;
    }
    if !verify_inner_product_committed(
        pcs,
        challenger,
        &proof.com_res,
        proof.dom_res,
        &r,
        proof.b_res,
        &proof.claim_res,
    ) {
        return false;
    }
    proof.b_res == proof.b_h1 + proof.b_x
}

/// A committed FFN proof wiring three distinct committed ops: matmul
/// (`h1 = W·x`), pointwise `mul` (`h2 = h1 ⊙ x`), and `add` (`out = h2 + h1`).
pub struct CommittedFfnProof {
    pub mm: MatmulProof,
    pub mul: MulProof,
    pub com_h1: fri::Commitment,
    pub dom_h1: fri::Domain,
    pub com_h2: fri::Commitment,
    pub dom_h2: fri::Domain,
    pub com_out: fri::Commitment,
    pub dom_out: fri::Domain,
    pub r: Vec<Fp>,
    pub b_h1: Fp,
    pub b_h2: Fp,
    pub b_out: Fp,
    pub claim_h1: InnerProductProof,
    pub claim_h2: InnerProductProof,
    pub claim_out: InnerProductProof,
}

pub fn prove_ffn_committed(
    pcs: &Pcs,
    challenger: &Challenger,
    w: &[Fp],
    x: &[Fp],
    h1: &[Fp],
    h2: &[Fp],
    out: &[Fp],
    out_dim: usize,
    in_dim: usize,
) -> CommittedFfnProof {
    let mm = prove_matmul_committed(pcs, challenger, w, x, h1, out_dim, in_dim);
    let mul = prove_mul_committed(pcs, challenger, h1, x, h2);

    let (com_h1, pd_h1, dom_h1) = fri::commit_vec(pcs, h1);
    let (com_h2, pd_h2, dom_h2) = fri::commit_vec(pcs, h2);
    let (com_out, pd_out, dom_out) = fri::commit_vec(pcs, out);
    let mut tr = fri::Transcript::new(challenger);
    tr.observe(&com_h1);
    tr.observe(&com_h2);
    tr.observe(&com_out);
    let r: Vec<Fp> = (0..out_dim).map(|_| tr.sample_val()).collect();
    let b_h1: Fp = r.iter().zip(h1).map(|(&ri, &vi)| ri * vi).sum();
    let b_h2: Fp = r.iter().zip(h2).map(|(&ri, &vi)| ri * vi).sum();
    let b_out: Fp = r.iter().zip(out).map(|(&ri, &vi)| ri * vi).sum();
    let claim_h1 =
        prove_inner_product_committed_given(pcs, challenger, &com_h1, &pd_h1, dom_h1, h1, &r);
    let claim_h2 =
        prove_inner_product_committed_given(pcs, challenger, &com_h2, &pd_h2, dom_h2, h2, &r);
    let claim_out =
        prove_inner_product_committed_given(pcs, challenger, &com_out, &pd_out, dom_out, out, &r);

    CommittedFfnProof {
        mm,
        mul,
        com_h1,
        dom_h1,
        com_h2,
        dom_h2,
        com_out,
        dom_out,
        r,
        b_h1,
        b_h2,
        b_out,
        claim_h1,
        claim_h2,
        claim_out,
    }
}

pub fn verify_ffn_committed(
    pcs: &Pcs,
    challenger: &Challenger,
    out_dim: usize,
    in_dim: usize,
    proof: &CommittedFfnProof,
) -> bool {
    if !verify_matmul_committed(pcs, challenger, out_dim, in_dim, &proof.mm) {
        return false;
    }
    if !verify_mul_committed(pcs, challenger, &proof.mul) {
        return false;
    }
    let mut tr = fri::Transcript::new(challenger);
    tr.observe(&proof.com_h1);
    tr.observe(&proof.com_h2);
    tr.observe(&proof.com_out);
    let r: Vec<Fp> = (0..out_dim).map(|_| tr.sample_val()).collect();
    if !verify_inner_product_committed(
        pcs,
        challenger,
        &proof.com_h1,
        proof.dom_h1,
        &r,
        proof.b_h1,
        &proof.claim_h1,
    ) {
        return false;
    }
    if !verify_inner_product_committed(
        pcs,
        challenger,
        &proof.com_h2,
        proof.dom_h2,
        &r,
        proof.b_h2,
        &proof.claim_h2,
    ) {
        return false;
    }
    if !verify_inner_product_committed(
        pcs,
        challenger,
        &proof.com_out,
        proof.dom_out,
        &r,
        proof.b_out,
        &proof.claim_out,
    ) {
        return false;
    }
    proof.b_out == proof.b_h1 + proof.b_h2
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(x: u64) -> Fp {
        Fp::new(x)
    }

    #[test]
    fn linear_ffn_committed_honest() {
        let (pcs, challenger) = fri::new_pcs(1, 10);
        // W1 = [[1,2],[3,4]], x = [5,6] → h1 = [17, 39].
        // residual = h1 + x = [22, 45].
        // W2 = [[1,0],[0,1]], logits = W2·residual = [22, 45].
        let w1 = [f(1), f(2), f(3), f(4)];
        let x = [f(5), f(6)];
        let h1 = [f(17), f(39)];
        let w2 = [f(1), f(0), f(0), f(1)];
        let residual = [f(22), f(45)];
        let logits = [f(22), f(45)];
        let proof = prove_linear_ffn_committed(
            &pcs,
            &challenger,
            &w1,
            &x,
            &h1,
            &w2,
            &residual,
            &logits,
            2,
            2,
            2,
        );
        assert!(verify_linear_ffn_committed(
            &pcs,
            &challenger,
            2,
            2,
            2,
            &proof
        ));
    }

    #[test]
    fn linear_ffn_committed_rejects_wrong_add() {
        let (pcs, challenger) = fri::new_pcs(1, 10);
        let w1 = [f(1), f(2), f(3), f(4)];
        let x = [f(5), f(6)];
        let h1 = [f(17), f(39)];
        let w2 = [f(1), f(0), f(0), f(1)];
        let residual = [f(22), f(46)]; // wrong: h1 + x = [22, 45]
        let logits = [f(22), f(46)];
        let proof = prove_linear_ffn_committed(
            &pcs,
            &challenger,
            &w1,
            &x,
            &h1,
            &w2,
            &residual,
            &logits,
            2,
            2,
            2,
        );
        assert!(!verify_linear_ffn_committed(
            &pcs,
            &challenger,
            2,
            2,
            2,
            &proof
        ));
    }

    #[test]
    fn ffn_committed_honest() {
        let (pcs, challenger) = fri::new_pcs(1, 10);
        // x = [5,6]; W = [[1,2],[3,4]] → h1 = [17,39];
        // h2 = h1 ⊙ x = [85,234]; out = h2 + h1 = [102,273].
        let w = [f(1), f(2), f(3), f(4)];
        let x = [f(5), f(6)];
        let h1 = [f(17), f(39)];
        let h2 = [f(85), f(234)];
        let out = [f(102), f(273)];
        let proof = prove_ffn_committed(&pcs, &challenger, &w, &x, &h1, &h2, &out, 2, 2);
        assert!(verify_ffn_committed(&pcs, &challenger, 2, 2, &proof));
    }

    #[test]
    fn ffn_committed_rejects_wrong() {
        let (pcs, challenger) = fri::new_pcs(1, 10);
        let w = [f(1), f(2), f(3), f(4)];
        let x = [f(5), f(6)];
        let h1 = [f(17), f(39)];
        let h2 = [f(85), f(234)];
        let out = [f(102), f(274)]; // wrong: h2 + h1 = [102, 273]
        let proof = prove_ffn_committed(&pcs, &challenger, &w, &x, &h1, &h2, &out, 2, 2);
        assert!(!verify_ffn_committed(&pcs, &challenger, 2, 2, &proof));
    }
}
