//! Succinct 2-column *function* lookup via fingerprint folding + **LogUp**.
//!
//! This is the O(log n) replacement for [`crate::lookup::lookup_committed`]'s sparse
//! membership: it proves `out[i] = f(in[i])` for a public table `f` without
//! revealing the `n` query indices to the verifier. Instead the verifier checks
//!  1. the fingerprint `fp[i] = α + β·in[i] + β²·out[i]` is the fold of the
//!     committed `in`/`out` (three linear claims), and
//!  2. `{fp} ⊆ {ft}` via the committed LogUp ([`crate::lookup::logup`]), where `ft` is the
//!     committed graph table `ft[j] = α + β·(lo+j) + β²·f(lo+j)`.
//!
//! By Schwartz–Zippel over the random `α, β`, `fp[i] ∈ {ft}` forces `out[i] =
//! f(in[i])`. The verifier cost is O(log n + log N) (FRI + the LogUp identity);
//! the O(N) table commit happens once and is shared across every lookup.
//!
//! `α, β` must be *random* (Fiat–Shamir): they make the fingerprint collision-free
//! and, crucially, keep `ft[j]` nonzero so the LogUp inverses are defined.

use crate::field::{Ext, Fp};
use crate::pcs::fri::{self, Challenger, Pcs, Prover};
use crate::lookup::logup::{self, LogUpProof, TableCommitment};
use crate::pcs::zk::{
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

fn f_int(x: i64) -> Fp {
    if x < 0 {
        -Fp::new(x.unsigned_abs())
    } else {
        Fp::new(x as u64)
    }
}

/// A committed function table: the graph fingerprints `ft[j] = α + β·(lo+j) +
/// β²·f(lo+j)` over `[lo, hi]`. Committed once and shared across every lookup of
/// the same function.
pub struct FunctionTable {
    pub lo: i64,
    pub hi: i64,
    pub table: TableCommitment,
}

pub fn commit_function_table(
    pcs: &Pcs,
    f: impl Fn(i64) -> i64,
    lo: i64,
    hi: i64,
    alpha: Fp,
    beta: Fp,
) -> FunctionTable {
    let ft: Vec<Fp> = (lo..=hi)
        .map(|x| alpha + beta * f_int(x) + beta * beta * f_int(f(x)))
        .collect();
    let table = logup::commit_table(pcs, &ft);
    FunctionTable { lo, hi, table }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Lookup2LogUpProof {
    pub n: usize,
    pub com_in: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom_in: fri::Domain,
    pub com_out: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom_out: fri::Domain,
    pub com_fp: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
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
    pub logup: LogUpProof,
}

/// Prove `out[i] = f(in[i])` with `in`/`out` already committed (shared form).
pub fn prove_lookup2_logup_given(
    ctx: &mut Prover,
    com_in: &fri::Commitment,
    pd_in: &fri::ProverData,
    dom_in: fri::Domain,
    com_out: &fri::Commitment,
    pd_out: &fri::ProverData,
    dom_out: fri::Domain,
    inp: &[i64],
    out: &[i64],
    table: &FunctionTable,
    alpha: Fp,
    beta: Fp,
) -> Lookup2LogUpProof {
    let n = inp.len();
    assert_eq!(out.len(), n);

    // Query fingerprint fp[i] = α + β·in[i] + β²·out[i].
    let fp_unpad: Vec<Fp> = (0..n)
        .map(|i| alpha + beta * f_int(inp[i]) + beta * beta * f_int(out[i]))
        .collect();
    let inp_fp: Vec<Fp> = pad(inp.iter().map(|&x| f_int(x)).collect(), n);
    let out_fp: Vec<Fp> = pad(out.iter().map(|&x| f_int(x)).collect(), n);
    let fp = pad(fp_unpad.clone(), n);
    let (com_fp, pd_fp, dom_fp) = fri::commit_vec(&ctx.pcs, &fp);

    // Fingerprint folding (three linear claims at a shared r).
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

    // Multiplicities m[j] = #{i : in[i] = lo + j}.
    let mut m = vec![0usize; (table.hi - table.lo + 1) as usize];
    for &x in inp {
        m[(x - table.lo) as usize] += 1;
    }
    let m_fp: Vec<Fp> = m.iter().map(|&c| Fp::new(c as u64)).collect();

    // LogUp: {fp} ⊆ {ft}.
    let logup = logup::prove_logup_committed(ctx, &fp_unpad, &m_fp, &table.table);

    Lookup2LogUpProof {
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
        logup,
    }
}

pub fn verify_lookup2_logup(
    challenger: &Challenger,
    com_ft: &fri::Commitment,
    dom_ft: fri::Domain,
    proof: &Lookup2LogUpProof,
    values: &[Ext],
    acc: &mut Openings,
) -> bool {
    let n = proof.n;

    // Fingerprint folding.
    let mut tr = fri::Transcript::new(challenger);
    tr.observe(&proof.com_in);
    tr.observe(&proof.com_out);
    tr.observe(&proof.com_fp);
    let mut r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    r.resize(next_pow2(n), Fp::ZERO);
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

    // LogUp membership.
    logup::verify_logup_committed(challenger, com_ft, dom_ft, &proof.logup, values, acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(x: u64) -> Fp {
        Fp::new(x)
    }

    #[test]
    fn lookup2_logup_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        let alpha = f(11);
        let beta = f(5);
        // f(x) = x² over x ∈ 0..=3; queries (1,1),(3,9),(1,1).
        let table = commit_function_table(&ctx.pcs, |x| x * x, 0, 3, alpha, beta);
        let inp = [1i64, 3, 1];
        let out = [1i64, 9, 1];
        let inp_fp: Vec<Fp> = inp.iter().map(|&x| f(x as u64)).collect();
        let out_fp: Vec<Fp> = out.iter().map(|&x| f(x as u64)).collect();
        let (com_in, pd_in, dom_in) = fri::commit_vec(&ctx.pcs, &pad(inp_fp, 3));
        let (com_out, pd_out, dom_out) = fri::commit_vec(&ctx.pcs, &pad(out_fp, 3));
        let proof = prove_lookup2_logup_given(
            &mut ctx, &com_in, &pd_in, dom_in, &com_out, &pd_out, dom_out, &inp, &out, &table,
            alpha, beta,
        );
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = crate::pcs::zk::Openings::new();
        assert!(verify_lookup2_logup(
            &ctx.challenger,
            &table.table.com_ft,
            table.table.dom_ft,
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
    fn lookup2_logup_rejects_wrong_out() {
        let mut ctx = fri::Prover::new(1, 10);
        let alpha = f(11);
        let beta = f(5);
        let table = commit_function_table(&ctx.pcs, |x| x * x, 0, 3, alpha, beta);
        let inp = [1i64, 3, 1];
        let out = [1i64, 10, 1]; // (3,10) wrong: f(3)=9
        let inp_fp: Vec<Fp> = inp.iter().map(|&x| f(x as u64)).collect();
        let out_fp: Vec<Fp> = out.iter().map(|&x| f(x as u64)).collect();
        let (com_in, pd_in, dom_in) = fri::commit_vec(&ctx.pcs, &pad(inp_fp, 3));
        let (com_out, pd_out, dom_out) = fri::commit_vec(&ctx.pcs, &pad(out_fp, 3));
        let proof = prove_lookup2_logup_given(
            &mut ctx, &com_in, &pd_in, dom_in, &com_out, &pd_out, dom_out, &inp, &out, &table,
            alpha, beta,
        );
        ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = crate::pcs::zk::Openings::new();
        assert!(!verify_lookup2_logup(
            &ctx.challenger,
            &table.table.com_ft,
            table.table.dom_ft,
            &proof,
            &values,
            &mut acc
        ));
    }
}
