//! Committed 2-column lookup via fingerprint folding + **sparse membership**.
//!
//! For a function table `f` (silu/requant/exp — distinct values), the lookup
//! `out[i] = f(in[i])` is exactly the membership check `fp[i] = ft[idx[i]]`, where
//! `fp[i] = α + β·in[i] + β²·out[i]` is the committed fingerprint and `idx[i] =
//! in[i] − lo` is the *revealed* table index. This is O(n) in the number of queries,
//! with **no** O(N) table commitment or dense multiplicity — the table is public
//! and the verifier evaluates `ft[idx[i]]` directly.
//!
//!  1. fold `in`/`out` into `fp` (three linear claims at a shared `r`),
//!  2. membership: `⟨r_mem, fp⟩ = Σ_i r_mem[i]·ft[idx[i]]` (one linear claim).

use crate::field::{Ext, Fp};
use crate::fri::{self, Challenger, Prover};
use crate::zk::{
    prove_inner_product_committed_given, verify_inner_product_committed, InnerProductProof,
    Openings,
};
use p3_field::PrimeCharacteristicRing;

fn next_pow2(n: usize) -> usize {
    n.next_power_of_two()
}

fn pad(mut v: Vec<Fp>, n: usize) -> Vec<Fp> {
    v.resize(next_pow2(n), Fp::ZERO);
    v
}

fn f(x: i64) -> Fp {
    if x < 0 {
        -Fp::new(x.unsigned_abs())
    } else {
        Fp::new(x as u64)
    }
}

pub struct Lookup2Proof {
    pub n: usize,
    pub com_in: fri::Commitment,
    pub dom_in: fri::Domain,
    pub com_out: fri::Commitment,
    pub dom_out: fri::Domain,
    pub com_fp: fri::Commitment,
    pub dom_fp: fri::Domain,
    pub r: Vec<Fp>,
    pub b_in: Fp,
    pub b_out: Fp,
    pub b_fp: Fp,
    pub claim_in: InnerProductProof,
    pub claim_out: InnerProductProof,
    pub claim_fp: InnerProductProof,
    pub alpha: Fp,
    pub beta: Fp,
    // membership: fp[i] = ft[idx[i]].
    pub idx: Vec<usize>,
    pub r_mem: Vec<Fp>,
    pub b_mem: Fp,
    pub claim_mem: InnerProductProof,
}

/// Prove `out[i] = f(in[i])` for all `i` using *already-committed* `in`/`out`.
///
/// This is the shared-commitment form: the caller supplies the commitments (and
/// prover data) for the query `in` and `out`, so a single committed vector can
/// back several claims (e.g. `δ` appears in both the linear relation and the sign
/// lookup) without committing it twice. Only the derived fingerprint `fp` is
/// committed here.
pub fn prove_lookup2_committed_given(
    ctx: &mut Prover,
    com_in: &fri::Commitment,
    pd_in: &fri::ProverData,
    dom_in: fri::Domain,
    com_out: &fri::Commitment,
    pd_out: &fri::ProverData,
    dom_out: fri::Domain,
    inp: &[i64],
    out: &[i64],
    lo: i64,
    alpha: Fp,
    beta: Fp,
) -> Lookup2Proof {
    let n = inp.len();
    assert_eq!(out.len(), n);

    // Query fingerprint fp[i] = α + β·in[i] + β²·out[i].
    let fp: Vec<Fp> = (0..n)
        .map(|i| alpha + beta * f(inp[i]) + beta * beta * f(out[i]))
        .collect();

    let inp_fp: Vec<Fp> = pad(inp.iter().map(|&x| f(x)).collect(), n);
    let out_fp: Vec<Fp> = pad(out.iter().map(|&x| f(x)).collect(), n);
    let fp = pad(fp, n);

    let (com_fp, pd_fp, dom_fp) = fri::commit_vec(&ctx.pcs, &fp);

    let mut tr = fri::Transcript::new(&ctx.challenger);
    tr.observe(com_in);
    tr.observe(com_out);
    tr.observe(&com_fp);
    let mut r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    r.resize(next_pow2(n), Fp::ZERO);

    let b_in: Fp = r.iter().zip(&inp_fp).map(|(&ri, &vi)| ri * vi).sum();
    let b_out: Fp = r.iter().zip(&out_fp).map(|(&ri, &vi)| ri * vi).sum();
    let b_fp: Fp = r.iter().zip(&fp).map(|(&ri, &vi)| ri * vi).sum();

    let claim_in = prove_inner_product_committed_given(ctx, com_in, pd_in, dom_in, &inp_fp, &r);
    let claim_out = prove_inner_product_committed_given(ctx, com_out, pd_out, dom_out, &out_fp, &r);
    let claim_fp = prove_inner_product_committed_given(ctx, &com_fp, &pd_fp, dom_fp, &fp, &r);

    // Membership: idx[i] = in[i] − lo, and fp[i] = ft[idx[i]].
    let idx: Vec<usize> = inp.iter().map(|&x| (x - lo) as usize).collect();
    let mut r_mem: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    r_mem.resize(next_pow2(n), Fp::ZERO);
    let b_mem: Fp = r_mem.iter().zip(&fp).map(|(&ri, &vi)| ri * vi).sum();
    let claim_mem = prove_inner_product_committed_given(ctx, &com_fp, &pd_fp, dom_fp, &fp, &r_mem);

    Lookup2Proof {
        n,
        com_in: com_in.clone(),
        dom_in,
        com_out: com_out.clone(),
        dom_out,
        com_fp,
        dom_fp,
        r,
        b_in,
        b_out,
        b_fp,
        claim_in,
        claim_out,
        claim_fp,
        alpha,
        beta,
        idx,
        r_mem,
        b_mem,
        claim_mem,
    }
}

/// Prove `out[i] = f(in[i])` for all `i`, committing `in`/`out` first. See
/// [`prove_lookup2_committed_given`] for the shared-commitment form.
pub fn prove_lookup2_committed(
    ctx: &mut Prover,
    inp: &[i64],
    out: &[i64],
    lo: i64,
    alpha: Fp,
    beta: Fp,
) -> Lookup2Proof {
    let n = inp.len();
    let inp_fp: Vec<Fp> = pad(inp.iter().map(|&x| f(x)).collect(), n);
    let out_fp: Vec<Fp> = pad(out.iter().map(|&x| f(x)).collect(), n);
    let (com_in, pd_in, dom_in) = fri::commit_vec(&ctx.pcs, &inp_fp);
    let (com_out, pd_out, dom_out) = fri::commit_vec(&ctx.pcs, &out_fp);
    prove_lookup2_committed_given(
        ctx, &com_in, &pd_in, dom_in, &com_out, &pd_out, dom_out, inp, out, lo, alpha, beta,
    )
}

pub fn verify_lookup2_committed(
    challenger: &Challenger,
    ftbl: impl Fn(i64) -> i64,
    lo: i64,
    alpha: Fp,
    beta: Fp,
    proof: &Lookup2Proof,
    values: &[Ext],
    acc: &mut Openings,
) -> bool {
    let n = proof.n;

    let mut tr = fri::Transcript::new(challenger);
    tr.observe(&proof.com_in);
    tr.observe(&proof.com_out);
    tr.observe(&proof.com_fp);
    let mut r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    r.resize(next_pow2(n), Fp::ZERO);

    // Folding checks.
    if !verify_inner_product_committed(
        challenger,
        &proof.com_in,
        proof.dom_in,
        &r,
        proof.b_in,
        &proof.claim_in,
        values,
        acc,
    ) {
        return false;
    }
    if !verify_inner_product_committed(
        challenger,
        &proof.com_out,
        proof.dom_out,
        &r,
        proof.b_out,
        &proof.claim_out,
        values,
        acc,
    ) {
        return false;
    }
    if !verify_inner_product_committed(
        challenger,
        &proof.com_fp,
        proof.dom_fp,
        &r,
        proof.b_fp,
        &proof.claim_fp,
        values,
        acc,
    ) {
        return false;
    }
    let sum_r: Fp = r.iter().copied().sum();
    if proof.b_fp
        != proof.alpha * sum_r + proof.beta * proof.b_in + proof.beta * proof.beta * proof.b_out
    {
        return false;
    }

    // Membership: ⟨r_mem, fp⟩ = Σ_i r_mem[i]·ft[idx[i]].
    let mut r_mem: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    r_mem.resize(next_pow2(n), Fp::ZERO);
    if !verify_inner_product_committed(
        challenger,
        &proof.com_fp,
        proof.dom_fp,
        &r_mem,
        proof.b_mem,
        &proof.claim_mem,
        values,
        acc,
    ) {
        return false;
    }
    let rhs: Fp = r_mem
        .iter()
        .zip(&proof.idx)
        .map(|(&ri, &idx)| {
            let x = lo + idx as i64;
            let fx = ftbl(x);
            ri * (alpha + beta * f(x) + beta * beta * f(fx))
        })
        .sum();
    proof.b_mem == rhs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(x: u64) -> Fp {
        Fp::new(x)
    }

    #[test]
    fn lookup2_committed_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        // f(x) = x² over x ∈ 0..4; queries (1,1),(3,9),(1,1).
        let inp = [1i64, 3, 1];
        let out = [1i64, 9, 1];
        let alpha = f(11);
        let beta = f(5);
        let proof = prove_lookup2_committed(&mut ctx, &inp, &out, 0, alpha, beta);
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = crate::zk::Openings::new();
        assert!(verify_lookup2_committed(
            &ctx.challenger,
            |x| x * x,
            0,
            alpha,
            beta,
            &proof,
            &values,
            &mut acc,
        ));
        assert!(fri::open_many_verify(
            &ctx.pcs,
            &ctx.challenger,
            &acc,
            &fri_proof,
        ));
    }

    #[test]
    fn lookup2_committed_rejects_wrong_out() {
        let mut ctx = fri::Prover::new(1, 10);
        // (3, 10) is wrong: f(3) = 9.
        let inp = [1i64, 3, 1];
        let out = [1i64, 10, 1];
        let alpha = f(11);
        let beta = f(5);
        let proof = prove_lookup2_committed(&mut ctx, &inp, &out, 0, alpha, beta);
        ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = crate::zk::Openings::new();
        assert!(!verify_lookup2_committed(
            &ctx.challenger,
            |x| x * x,
            0,
            alpha,
            beta,
            &proof,
            &values,
            &mut acc,
        ));
    }
}
