//! Fingerprint folding: fold a multi-column query tuple into a single LogUp
//! fingerprint `fp[i] = α + Σ_k β^{k+1}·q_k[i]`, verified on committed data via
//! the committed linear claim.
//!
//! The fingerprint is an extension-field element, so it is split into its three
//! base-field components, each a linear combination of the committed columns with
//! public coefficients `coeff_c(β^{k+1})`. Each component is discharged by the
//! committed linear claim at a shared random `r`.

use crate::field::{Ext, Fp};
use crate::fri::{self, Challenger, Prover};
use crate::zk::{
    prove_inner_product_committed_given, verify_inner_product_committed, InnerProductProof,
    Openings,
};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};

fn next_pow2(n: usize) -> usize {
    n.next_power_of_two()
}

fn pad(mut v: Vec<Fp>, n: usize) -> Vec<Fp> {
    v.resize(next_pow2(n), Fp::ZERO);
    v
}

/// The committed fingerprint proof (prover's output).
pub struct FingerprintProof {
    pub n: usize,
    pub w: usize,
    pub com_cols: Vec<fri::Commitment>,
    pub dom_cols: Vec<fri::Domain>,
    pub com_fp: [fri::Commitment; 3],
    pub dom_fp: [fri::Domain; 3],
    pub r: Vec<Fp>,
    pub b_cols: Vec<Fp>,
    pub b_fp: [Fp; 3],
    pub claim_cols: Vec<InnerProductProof>,
    pub claim_fp: [InnerProductProof; 3],
    pub alpha: Ext,
    pub beta: Ext,
}

/// The `c`-th base-field coefficient of `β^{k+1}` (for the folding scalars).
fn beta_coeff(beta: Ext, k: usize, c: usize) -> Fp {
    let mut b = beta;
    for _ in 0..k {
        b *= beta;
    }
    <Ext as BasedVectorSpace<Fp>>::as_basis_coefficients_slice(&b)[c]
}

/// Prove `fp[i] = α + Σ_k β^{k+1}·q_k[i]` for the committed columns `q_k`.
pub fn prove_fingerprint(
    ctx: &mut Prover,
    cols: &[&[Fp]],
    alpha: Ext,
    beta: Ext,
) -> FingerprintProof {
    let w = cols.len();
    let n = cols[0].len();
    for col in cols {
        assert_eq!(col.len(), n);
    }

    // fp[i] = α + Σ_k β^{k+1}·q_k[i], split into 3 base-field components.
    let mut fp = [Vec::new(), Vec::new(), Vec::new()];
    for i in 0..n {
        let mut acc = alpha;
        let mut b = beta;
        for k in 0..w {
            acc += b * Ext::from(cols[k][i]);
            b *= beta;
        }
        let c = <Ext as BasedVectorSpace<Fp>>::as_basis_coefficients_slice(&acc);
        fp[0].push(c[0]);
        fp[1].push(c[1]);
        fp[2].push(c[2]);
    }
    let fp0 = pad(fp[0].clone(), n);
    let fp1 = pad(fp[1].clone(), n);
    let fp2 = pad(fp[2].clone(), n);

    let mut com_cols = Vec::with_capacity(w);
    let mut dom_cols = Vec::with_capacity(w);
    let mut pd_cols = Vec::with_capacity(w);
    for k in 0..w {
        let padded = pad(cols[k].to_vec(), n);
        let (c, pd, d) = fri::commit_vec(&ctx.pcs, &padded);
        com_cols.push(c);
        pd_cols.push(pd);
        dom_cols.push(d);
    }
    let (com_fp0, pd_fp0, dom_fp0) = fri::commit_vec(&ctx.pcs, &fp0);
    let (com_fp1, pd_fp1, dom_fp1) = fri::commit_vec(&ctx.pcs, &fp1);
    let (com_fp2, pd_fp2, dom_fp2) = fri::commit_vec(&ctx.pcs, &fp2);

    let mut tr = fri::Transcript::new(&ctx.challenger);
    for k in 0..w {
        tr.observe(&com_cols[k]);
    }
    tr.observe(&com_fp0);
    tr.observe(&com_fp1);
    tr.observe(&com_fp2);
    let mut r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    r.resize(next_pow2(n), Fp::ZERO);

    let b_cols: Vec<Fp> = cols
        .iter()
        .map(|col| r.iter().zip(*col).map(|(&ri, &vi)| ri * vi).sum())
        .collect();
    let b_fp = [
        r.iter().zip(&fp0).map(|(&ri, &vi)| ri * vi).sum(),
        r.iter().zip(&fp1).map(|(&ri, &vi)| ri * vi).sum(),
        r.iter().zip(&fp2).map(|(&ri, &vi)| ri * vi).sum(),
    ];

    let mut claim_cols = Vec::with_capacity(w);
    for k in 0..w {
        let padded = pad(cols[k].to_vec(), n);
        claim_cols.push(prove_inner_product_committed_given(
            ctx,
            &com_cols[k],
            &pd_cols[k],
            dom_cols[k],
            &padded,
            &r,
        ));
    }
    let claim_fp = [
        prove_inner_product_committed_given(ctx, &com_fp0, &pd_fp0, dom_fp0, &fp0, &r),
        prove_inner_product_committed_given(ctx, &com_fp1, &pd_fp1, dom_fp1, &fp1, &r),
        prove_inner_product_committed_given(ctx, &com_fp2, &pd_fp2, dom_fp2, &fp2, &r),
    ];

    FingerprintProof {
        n,
        w,
        com_cols,
        dom_cols,
        com_fp: [com_fp0, com_fp1, com_fp2],
        dom_fp: [dom_fp0, dom_fp1, dom_fp2],
        r,
        b_cols,
        b_fp,
        claim_cols,
        claim_fp,
        alpha,
        beta,
    }
}

/// Verify `fp[i] = α + Σ_k β^{k+1}·q_k[i]` from commitments alone.
pub fn verify_fingerprint(
    challenger: &Challenger,
    proof: &FingerprintProof,
    values: &[Ext],
    acc: &mut Openings,
) -> bool {
    let w = proof.w;
    let n = proof.n;

    let mut tr = fri::Transcript::new(challenger);
    for k in 0..w {
        tr.observe(&proof.com_cols[k]);
    }
    tr.observe(&proof.com_fp[0]);
    tr.observe(&proof.com_fp[1]);
    tr.observe(&proof.com_fp[2]);
    let mut r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    r.resize(next_pow2(n), Fp::ZERO);

    for k in 0..w {
        if !verify_inner_product_committed(
            challenger,
            &proof.com_cols[k],
            proof.dom_cols[k],
            &r,
            proof.b_cols[k],
            &proof.claim_cols[k],
            values,
            acc,
        ) {
            return false;
        }
    }
    for c in 0..3 {
        if !verify_inner_product_committed(
            challenger,
            &proof.com_fp[c],
            proof.dom_fp[c],
            &r,
            proof.b_fp[c],
            &proof.claim_fp[c],
            values,
            acc,
        ) {
            return false;
        }
    }
    // Scalar relations: b_fp[c] = coeff_c(α)·Σr + Σ_k coeff_c(β^{k+1})·b_cols[k].
    let sum_r: Fp = r.iter().copied().sum();
    let alpha_c = <Ext as BasedVectorSpace<Fp>>::as_basis_coefficients_slice(&proof.alpha);
    for c in 0..3 {
        let mut rhs = alpha_c[c] * sum_r;
        for k in 0..w {
            rhs += beta_coeff(proof.beta, k, c) * proof.b_cols[k];
        }
        if proof.b_fp[c] != rhs {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(x: u64) -> Fp {
        Fp::new(x)
    }

    #[test]
    fn fingerprint_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        // 2 columns: in = [1,2,3,4], out = [5,6,7,8]. α = 7, β = 3.
        let col0 = [f(1), f(2), f(3), f(4)];
        let col1 = [f(5), f(6), f(7), f(8)];
        let alpha = Ext::from(f(7));
        let beta = Ext::from(f(3));
        let proof = prove_fingerprint(&mut ctx, &[&col0, &col1], alpha, beta);
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = crate::zk::Openings::new();
        assert!(verify_fingerprint(
            &ctx.challenger,
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
    fn fingerprint_rejects_wrong() {
        let mut ctx = fri::Prover::new(1, 10);
        let col0 = [f(1), f(2), f(3), f(4)];
        let col1 = [f(5), f(6), f(7), f(8)];
        let alpha = Ext::from(f(7));
        let beta = Ext::from(f(3));
        let mut proof = prove_fingerprint(&mut ctx, &[&col0, &col1], alpha, beta);
        // Tamper the claimed ⟨r, fp_0⟩: the scalar relation must reject.
        proof.b_fp[0] += f(1);
        ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = crate::zk::Openings::new();
        assert!(!verify_fingerprint(
            &ctx.challenger,
            &proof,
            &values,
            &mut acc,
        ));
    }
}
