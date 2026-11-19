//! Committed 2-column lookup: wire the fingerprint folding + the committed LogUp
//! into one lookup check for `(in[i], out[i]) ∈ {(x, f(x))}` (SiLU, requant, …).
//!
//!  1. fold the committed `in`/`out` into a single base-field fingerprint
//!     `fp[i] = α + β·in[i] + β²·out[i]` (three linear claims at a shared `r`),
//!  2. run the committed LogUp `fp ⊆ fp_table` against the public table.

use crate::field::Fp;
use crate::fri::{self, Challenger, Pcs};
use crate::logup::{prove_logup_committed, verify_logup_committed, LogUpProof};
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

fn i64_to_fp(x: i64) -> Fp {
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
    pub logup: LogUpProof,
}

/// Prove `(in[i], out[i]) ∈ {(x, f(x)) : x ∈ [lo, hi]}`. `m` are the multiplicities
/// of the table entries. `f` is the public table function (e.g. `silu`).
pub fn prove_lookup2_committed(
    pcs: &Pcs,
    challenger: &Challenger,
    inp: &[Fp],
    out: &[Fp],
    f: impl Fn(i64) -> i64,
    lo: i64,
    hi: i64,
    m: &[Fp],
    alpha: Fp,
    beta: Fp,
) -> Lookup2Proof {
    let n = inp.len();
    assert_eq!(out.len(), n);

    // Query fingerprint fp[i] = α + β·in[i] + β²·out[i].
    let fp: Vec<Fp> = (0..n)
        .map(|i| alpha + beta * inp[i] + beta * beta * out[i])
        .collect();

    let inp = pad(inp.to_vec(), n);
    let out = pad(out.to_vec(), n);
    let fp = pad(fp, n);

    let (com_in, pd_in, dom_in) = fri::commit_vec(pcs, &inp);
    let (com_out, pd_out, dom_out) = fri::commit_vec(pcs, &out);
    let (com_fp, pd_fp, dom_fp) = fri::commit_vec(pcs, &fp);

    let mut tr = fri::Transcript::new(challenger);
    tr.observe(&com_in);
    tr.observe(&com_out);
    tr.observe(&com_fp);
    let mut r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    r.resize(next_pow2(n), Fp::ZERO);

    let b_in: Fp = r.iter().zip(&inp).map(|(&ri, &vi)| ri * vi).sum();
    let b_out: Fp = r.iter().zip(&out).map(|(&ri, &vi)| ri * vi).sum();
    let b_fp: Fp = r.iter().zip(&fp).map(|(&ri, &vi)| ri * vi).sum();

    let claim_in =
        prove_inner_product_committed_given(pcs, challenger, &com_in, &pd_in, dom_in, &inp, &r);
    let claim_out =
        prove_inner_product_committed_given(pcs, challenger, &com_out, &pd_out, dom_out, &out, &r);
    let claim_fp =
        prove_inner_product_committed_given(pcs, challenger, &com_fp, &pd_fp, dom_fp, &fp, &r);

    // Table fingerprints (public): ft[j] = α + β·x + β²·f(x), with multiplicities m.
    let ft: Vec<Fp> = (lo..=hi)
        .map(|x| {
            let fx = f(x);
            alpha + beta * i64_to_fp(x) + beta * beta * i64_to_fp(fx)
        })
        .collect();

    let logup = prove_logup_committed(pcs, challenger, &fp[..n], &ft, m);

    Lookup2Proof {
        n,
        com_in,
        dom_in,
        com_out,
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

pub fn verify_lookup2_committed(pcs: &Pcs, challenger: &Challenger, proof: &Lookup2Proof) -> bool {
    let n = proof.n;

    let mut tr = fri::Transcript::new(challenger);
    tr.observe(&proof.com_in);
    tr.observe(&proof.com_out);
    tr.observe(&proof.com_fp);
    let mut r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    r.resize(next_pow2(n), Fp::ZERO);

    // Folding checks.
    if !verify_inner_product_committed(
        pcs,
        challenger,
        &proof.com_in,
        proof.dom_in,
        &r,
        proof.b_in,
        &proof.claim_in,
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
    if !verify_inner_product_committed(
        pcs,
        challenger,
        &proof.com_fp,
        proof.dom_fp,
        &r,
        proof.b_fp,
        &proof.claim_fp,
    ) {
        return false;
    }
    // fp = α + β·in + β²·out → ⟨r, fp⟩ = α·Σr + β·⟨r, in⟩ + β²·⟨r, out⟩.
    let sum_r: Fp = r.iter().copied().sum();
    if proof.b_fp
        != proof.alpha * sum_r + proof.beta * proof.b_in + proof.beta * proof.beta * proof.b_out
    {
        return false;
    }
    // LogUp fp ⊆ fp_table.
    verify_logup_committed(pcs, challenger, &proof.logup)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(x: u64) -> Fp {
        Fp::new(x)
    }

    #[test]
    fn lookup2_committed_honest() {
        let (pcs, challenger) = fri::new_pcs(1, 10);
        // f(x) = x² over x ∈ 0..4; queries (1,1),(3,9),(1,1) with m = [0,2,0,1,0].
        let inp = [f(1), f(3), f(1)];
        let out = [f(1), f(9), f(1)];
        let m = [f(0), f(2), f(0), f(1), f(0)]; // table {0,1,2,3,4}
        let alpha = f(11);
        let beta = f(5);
        let proof = prove_lookup2_committed(
            &pcs,
            &challenger,
            &inp,
            &out,
            |x| x * x,
            0,
            4,
            &m,
            alpha,
            beta,
        );
        assert!(verify_lookup2_committed(&pcs, &challenger, &proof));
    }

    #[test]
    fn lookup2_committed_rejects_wrong_out() {
        let (pcs, challenger) = fri::new_pcs(1, 10);
        // (3, 10) is wrong: f(3) = 9.
        let inp = [f(1), f(3), f(1)];
        let out = [f(1), f(10), f(1)];
        let m = [f(0), f(2), f(0), f(1), f(0)];
        let alpha = f(11);
        let beta = f(5);
        let proof = prove_lookup2_committed(
            &pcs,
            &challenger,
            &inp,
            &out,
            |x| x * x,
            0,
            4,
            &m,
            alpha,
            beta,
        );
        assert!(!verify_lookup2_committed(&pcs, &challenger, &proof));
    }
}
