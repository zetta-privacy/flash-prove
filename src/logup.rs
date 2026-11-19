//! Committed LogUp (log-derivative) lookup argument — fully sound.
//!
//! Proves `{fq[0..n]} ⊆ {ft[0..N]}` (with committed multiplicities `m`) on
//! FRI-committed data: the verifier holds only commitments + opening proofs.
//!
//! The identity `Σ_i 1/fq[i] = Σ_j m[j]/ft[j]` is the "fraction" formulation on
//! **separate query/table sides** (so the query multiplicity is the public constant
//! `1`, not a committed value — this closes the sign-gap):
//!
//!   - query:   `fq ⊙ diff_q = 1`   (pointwise, RHS is the public `1`),
//!   - table:   `ft ⊙ diff_t = m`   (pointwise),
//!   - balance: `Σ diff_q = Σ diff_t`, and `Σ m = n`.
//!
//! Each bilinear `fp ⊙ diff` is handled without a degree-3 product sumcheck, by
//! folding a public challenge `r` into the fingerprints (`g = r ⊙ fp`, a diagonal
//! linear claim) and running one IPA `⟨g, diff⟩ = ⟨r, rhs⟩`.

use crate::field::{Ext, Fp};
use crate::fri::{self, Challenger, Pcs, Prover};
use crate::ipa::{prove_ipa_committed, verify_ipa_committed, CommittedIpa};
use crate::zk::{
    prove_inner_product_committed_given, verify_inner_product_committed, InnerProductProof,
    Openings,
};
use p3_field::{Field, PrimeCharacteristicRing};

pub struct LogUpProof {
    pub n: usize,
    pub big_n: usize,
    pub com_fq: fri::Commitment,
    pub dom_fq: fri::Domain,
    pub com_m: fri::Commitment,
    pub dom_m: fri::Domain,
    pub com_diff_q: fri::Commitment,
    pub dom_diff_q: fri::Domain,
    pub com_diff_t: fri::Commitment,
    pub dom_diff_t: fri::Domain,
    pub com_g_q: fri::Commitment,
    pub dom_g_q: fri::Domain,
    pub com_g_t: fri::Commitment,
    pub dom_g_t: fri::Domain,
    // query side: g_q = r_q ⊙ fq, ⟨g_q, diff_q⟩ = Σ r_q
    pub diag_q_g: InnerProductProof,
    pub diag_q_fq: InnerProductProof,
    pub ipa_q: CommittedIpa,
    pub c_q: Fp,
    pub b_q: Fp,
    // table side: g_t = r_t ⊙ ft, ⟨g_t, diff_t⟩ = ⟨r_t, m⟩
    pub diag_t_g: InnerProductProof,
    pub diag_t_ft: InnerProductProof,
    pub ipa_t: CommittedIpa,
    pub m_claim: InnerProductProof,
    pub c_t: Fp,
    pub b_t: Fp,
    // balance Σ diff_q = Σ diff_t, and Σ m = n
    pub bal_q: InnerProductProof,
    pub bal_t: InnerProductProof,
    pub sum_claim: InnerProductProof,
}

fn next_pow2(n: usize) -> usize {
    n.next_power_of_two()
}

fn pad(mut v: Vec<Fp>, n: usize) -> Vec<Fp> {
    v.resize(next_pow2(n), Fp::ZERO);
    v
}

/// A committed public lookup table (the `ft` fingerprints). Committed once and
/// shared across every lookup of that table, so the O(N) FRI commit happens a
/// single time instead of per lookup.
pub struct TableCommitment {
    pub n: usize,
    pub com_ft: fri::Commitment,
    pub dom_ft: fri::Domain,
    pub pd_ft: fri::ProverData,
    pub ft: Vec<Fp>, // padded (prover-side)
}

pub fn commit_table(pcs: &Pcs, ft: &[Fp]) -> TableCommitment {
    let n = ft.len();
    let ft_pad = pad(ft.to_vec(), n);
    let (com_ft, pd_ft, dom_ft) = fri::commit_vec(pcs, &ft_pad);
    TableCommitment {
        n,
        com_ft,
        dom_ft,
        pd_ft,
        ft: ft_pad,
    }
}

pub fn prove_logup_committed(
    ctx: &mut Prover,
    fq: &[Fp],
    m: &[Fp],
    table: &TableCommitment,
) -> LogUpProof {
    let n = fq.len();
    let big_n = table.n;

    let fq = pad(fq.to_vec(), n);
    let m = pad(m.to_vec(), big_n);
    // diff_q[i] = 1/fq[i]; diff_t[j] = m[j]/ft[j].
    let diff_q: Vec<Fp> = (0..n).map(|i| fq[i].inverse()).collect();
    let diff_t: Vec<Fp> = (0..big_n).map(|j| m[j] * table.ft[j].inverse()).collect();
    let diff_q = pad(diff_q, n);
    let diff_t = pad(diff_t, big_n);

    let (com_fq, pd_fq, dom_fq) = fri::commit_vec(&ctx.pcs, &fq);
    let (com_m, pd_m, dom_m) = fri::commit_vec(&ctx.pcs, &m);
    let (com_diff_q, pd_diff_q, dom_diff_q) = fri::commit_vec(&ctx.pcs, &diff_q);
    let (com_diff_t, pd_diff_t, dom_diff_t) = fri::commit_vec(&ctx.pcs, &diff_t);

    let mut tr = fri::Transcript::new(&ctx.challenger);
    tr.observe(&com_fq);
    tr.observe(&table.com_ft);
    tr.observe(&com_m);
    tr.observe(&com_diff_q);
    tr.observe(&com_diff_t);
    let mut r_q: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    r_q.resize(next_pow2(n), Fp::ZERO);
    let mut s_q: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    s_q.resize(next_pow2(n), Fp::ZERO);
    let mut r_t: Vec<Fp> = (0..big_n).map(|_| tr.sample_val()).collect();
    r_t.resize(next_pow2(big_n), Fp::ZERO);
    let mut s_t: Vec<Fp> = (0..big_n).map(|_| tr.sample_val()).collect();
    s_t.resize(next_pow2(big_n), Fp::ZERO);

    // Query side.
    let g_q: Vec<Fp> = (0..next_pow2(n)).map(|i| r_q[i] * fq[i]).collect();
    let (com_g_q, pd_g_q, dom_g_q) = fri::commit_vec(&ctx.pcs, &g_q);
    let b_q: Fp = s_q.iter().zip(&g_q).map(|(&sj, &gj)| sj * gj).sum();
    let s_q_r_q: Vec<Fp> = (0..next_pow2(n)).map(|i| s_q[i] * r_q[i]).collect();
    let c_q: Fp = r_q.iter().copied().sum(); // ⟨r_q, 1⟩
    let ipa_q = prove_ipa_committed(ctx, &g_q, &diff_q, c_q);

    // Table side.
    let g_t: Vec<Fp> = (0..next_pow2(big_n))
        .map(|j| r_t[j] * table.ft[j])
        .collect();
    let (com_g_t, pd_g_t, dom_g_t) = fri::commit_vec(&ctx.pcs, &g_t);
    let b_t: Fp = s_t.iter().zip(&g_t).map(|(&sj, &gj)| sj * gj).sum();
    let s_t_r_t: Vec<Fp> = (0..next_pow2(big_n)).map(|j| s_t[j] * r_t[j]).collect();
    let c_t: Fp = g_t.iter().zip(&diff_t).map(|(&gk, &dk)| gk * dk).sum();
    let ipa_t = prove_ipa_committed(ctx, &g_t, &diff_t, c_t);

    // Linear claims.
    let ones_n: Vec<Fp> = pad(vec![Fp::ONE; n], n);
    let ones_t: Vec<Fp> = pad(vec![Fp::ONE; big_n], big_n);
    let diag_q_g = prove_inner_product_committed_given(ctx, &com_g_q, &pd_g_q, dom_g_q, &g_q, &s_q);
    let diag_q_fq =
        prove_inner_product_committed_given(ctx, &com_fq, &pd_fq, dom_fq, &fq, &s_q_r_q);
    let diag_t_g = prove_inner_product_committed_given(ctx, &com_g_t, &pd_g_t, dom_g_t, &g_t, &s_t);
    let diag_t_ft = prove_inner_product_committed_given(
        ctx,
        &table.com_ft,
        &table.pd_ft,
        table.dom_ft,
        &table.ft,
        &s_t_r_t,
    );
    let m_claim = prove_inner_product_committed_given(ctx, &com_m, &pd_m, dom_m, &m, &r_t);
    // Balance Σ diff_q = Σ diff_t (the verifier derives the shared claim from bal_q).
    let bal_q = prove_inner_product_committed_given(
        ctx,
        &com_diff_q,
        &pd_diff_q,
        dom_diff_q,
        &diff_q,
        &ones_n,
    );
    let bal_t = prove_inner_product_committed_given(
        ctx,
        &com_diff_t,
        &pd_diff_t,
        dom_diff_t,
        &diff_t,
        &ones_t,
    );
    let sum_claim = prove_inner_product_committed_given(ctx, &com_m, &pd_m, dom_m, &m, &ones_t);

    LogUpProof {
        n,
        big_n,
        com_fq,
        dom_fq,
        com_m,
        dom_m,
        com_diff_q,
        dom_diff_q,
        com_diff_t,
        dom_diff_t,
        com_g_q,
        dom_g_q,
        com_g_t,
        dom_g_t,
        diag_q_g,
        diag_q_fq,
        ipa_q,
        c_q,
        b_q,
        diag_t_g,
        diag_t_ft,
        ipa_t,
        m_claim,
        c_t,
        b_t,
        bal_q,
        bal_t,
        sum_claim,
    }
}

pub fn verify_logup_committed(
    challenger: &Challenger,
    com_ft: &fri::Commitment,
    dom_ft: fri::Domain,
    proof: &LogUpProof,
    values: &[Ext],
    acc: &mut Openings,
) -> bool {
    let n = proof.n;
    let big_n = proof.big_n;

    let mut tr = fri::Transcript::new(challenger);
    tr.observe(&proof.com_fq);
    tr.observe(com_ft);
    tr.observe(&proof.com_m);
    tr.observe(&proof.com_diff_q);
    tr.observe(&proof.com_diff_t);
    let mut r_q: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    r_q.resize(next_pow2(n), Fp::ZERO);
    let mut s_q: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    s_q.resize(next_pow2(n), Fp::ZERO);
    let mut r_t: Vec<Fp> = (0..big_n).map(|_| tr.sample_val()).collect();
    r_t.resize(next_pow2(big_n), Fp::ZERO);
    let mut s_t: Vec<Fp> = (0..big_n).map(|_| tr.sample_val()).collect();
    s_t.resize(next_pow2(big_n), Fp::ZERO);
    let s_q_r_q: Vec<Fp> = (0..next_pow2(n)).map(|i| s_q[i] * r_q[i]).collect();
    let s_t_r_t: Vec<Fp> = (0..next_pow2(big_n)).map(|j| s_t[j] * r_t[j]).collect();
    let ones_n: Vec<Fp> = pad(vec![Fp::ONE; n], n);
    let ones_t: Vec<Fp> = pad(vec![Fp::ONE; big_n], big_n);

    // Opening order must match `prove_logup_committed`: the two IPAs are recorded
    // first, then the diagonal claims, then balance/sum.
    let c_q: Fp = r_q.iter().copied().sum();
    if c_q != proof.c_q || !verify_ipa_committed(challenger, proof.c_q, &proof.ipa_q, values, acc) {
        return false;
    }
    if !verify_ipa_committed(challenger, proof.c_t, &proof.ipa_t, values, acc) {
        return false;
    }
    // Query-side diagonal claims.
    if !verify_inner_product_committed(
        challenger,
        &proof.com_g_q,
        proof.dom_g_q,
        &s_q,
        proof.b_q,
        &proof.diag_q_g,
        values,
        acc,
    ) {
        return false;
    }
    if !verify_inner_product_committed(
        challenger,
        &proof.com_fq,
        proof.dom_fq,
        &s_q_r_q,
        proof.b_q,
        &proof.diag_q_fq,
        values,
        acc,
    ) {
        return false;
    }
    // Table-side diagonal claims.
    if !verify_inner_product_committed(
        challenger,
        &proof.com_g_t,
        proof.dom_g_t,
        &s_t,
        proof.b_t,
        &proof.diag_t_g,
        values,
        acc,
    ) {
        return false;
    }
    if !verify_inner_product_committed(
        challenger,
        com_ft,
        dom_ft,
        &s_t_r_t,
        proof.b_t,
        &proof.diag_t_ft,
        values,
        acc,
    ) {
        return false;
    }
    if !verify_inner_product_committed(
        challenger,
        &proof.com_m,
        proof.dom_m,
        &r_t,
        proof.c_t,
        &proof.m_claim,
        values,
        acc,
    ) {
        return false;
    }
    // Balance Σ diff_q = Σ diff_t (both equal the same shared claim).
    let b_bal = proof.bal_q.c0 * Fp::new(next_pow2(n) as u64);
    if !verify_inner_product_committed(
        challenger,
        &proof.com_diff_q,
        proof.dom_diff_q,
        &ones_n,
        b_bal,
        &proof.bal_q,
        values,
        acc,
    ) {
        return false;
    }
    if !verify_inner_product_committed(
        challenger,
        &proof.com_diff_t,
        proof.dom_diff_t,
        &ones_t,
        b_bal,
        &proof.bal_t,
        values,
        acc,
    ) {
        return false;
    }
    // Σ m = n.
    verify_inner_product_committed(
        challenger,
        &proof.com_m,
        proof.dom_m,
        &ones_t,
        Fp::new(n as u64),
        &proof.sum_claim,
        values,
        acc,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(x: i64) -> Fp {
        if x >= 0 {
            Fp::new(x as u64)
        } else {
            -Fp::new((-x) as u64)
        }
    }

    #[test]
    fn logup_committed_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        let ft = [f(1), f(2), f(3), f(4)];
        let fq = [f(2), f(4), f(2)];
        let m = [f(0), f(2), f(0), f(1)];
        let table = commit_table(&ctx.pcs, &ft);
        let proof = prove_logup_committed(&mut ctx, &fq, &m, &table);
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = crate::zk::Openings::new();
        assert!(verify_logup_committed(
            &ctx.challenger,
            &table.com_ft,
            table.dom_ft,
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
    fn logup_committed_rejects_wrong_mult() {
        let mut ctx = fri::Prover::new(1, 10);
        let ft = [f(1), f(2), f(3), f(4)];
        let fq = [f(2), f(4), f(2)];
        let m = [f(0), f(2), f(0), f(2)]; // Σ m = 4 ≠ 3
        let table = commit_table(&ctx.pcs, &ft);
        let proof = prove_logup_committed(&mut ctx, &fq, &m, &table);
        ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = crate::zk::Openings::new();
        assert!(!verify_logup_committed(
            &ctx.challenger,
            &table.com_ft,
            table.dom_ft,
            &proof,
            &values,
            &mut acc
        ));
    }
}
