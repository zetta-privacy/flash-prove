//! Committed `top_k` and `causal_softmax` — the two remaining selection
//! nonlinearities — on FRI-committed data.
//!
//! Both reduce a comparison/selection to a *threshold* (a public scalar revealed
//! by the prover) plus a sign/normalize lookup, so the verifier never re-reads the
//! committed inputs. The threshold is pinned to the committed data by the linear
//! relations below, so revealing it is sound (it leaks no more than the committed
//! values already imply).
//!
//! ## top_k
//!
//! ```text
//!   δ[i] = e·scores[i] − i − τ          (linear; e ties break by lower index)
//!   choice[i] = 1 ⟺ δ[i] ≥ 0           (sign lookup)
//!   Σ choice = k                       (count)
//! ```
//! `δ` uses the adjusted score `e·scores[i] − i`, which is injective, so the
//! tie-break ("lower index wins") is folded into the ordering and dominance is a
//! strict threshold. `choice = sign(δ)` forces `choice ∈ {0,1}` and
//! `choice[i]=1 ⟺ δ[i]≥0`; the count forces exactly `k` of them, which is exactly
//! the top-k of the adjusted scores.
//!
//! ## causal_softmax (one row, position `l`)
//!
//! ```text
//!   m   = max_{t≤l} scores[t],  t* = argmax            (revealed)
//!   d[t] = mask[t]·(scores[t] − m) + (1−mask[t])·exp_lo (linear)
//!   e[t] = exp_q15(d[t])                                (exp lookup)
//!   S   = ⟨mask, e⟩                                     (sum)
//!   nrm[t] = e[t]·2^15 / S                              (normalize lookup)
//!   probs[t] = mask[t]·nrm[t]                           (linear)
//!   d[t*] = 0                                           (max tightness)
//! ```
//! The `exp_lo` sentinel zeroes out the masked (t>l) positions because
//! `exp_q15(exp_lo) = 0` for a sufficiently negative `exp_lo`.

use crate::field::{Ext, Fp};
use crate::pcs::fri::{self, Challenger, Prover};
use crate::lookup::logup::TableCommitment;
use crate::lookup::lookup_logup::{
    prove_lookup2_logup_given, verify_lookup2_logup, FunctionTable, Lookup2LogUpProof,
};
use crate::lookup::requant::{prove_requant_given, verify_requant, RequantProof};
use crate::model::spec::RECIP_Q;
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

fn f(x: i64) -> Fp {
    if x < 0 {
        -Fp::new(x.unsigned_abs())
    } else {
        Fp::new(x as u64)
    }
}

// --- top_k ---------------------------------------------------------------------

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TopKProof {
    pub n: usize, // e = number of experts
    pub k: usize,
    pub tau: i64,
    pub com_scores: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom_scores: fri::Domain,
    pub com_choice: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom_choice: fri::Domain,
    pub com_delta: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom_delta: fri::Domain,
    /// count: ⟨1, choice⟩ = k.
    pub count_claim: InnerProductProof,
    /// δ = e·scores + pub: linear relation at a shared `r`.
    pub r: Vec<Fp>,
    pub b_delta: Fp,
    pub b_scores: Fp,
    pub delta_claim: InnerProductProof,
    pub scores_claim: InnerProductProof,
    /// choice = sign(δ).
    pub sign: Lookup2LogUpProof,
    pub sign_com_ft: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub sign_dom_ft: fri::Domain,
    pub alpha: Fp,
    pub beta: Fp,
}

pub fn prove_topk_committed(
    ctx: &mut Prover,
    scores: &[i64],
    choice: &[i64],
    k: usize,
    tau: i64,
    sign_lo: i64,
    alpha: Fp,
    beta: Fp,
) -> TopKProof {
    let n = scores.len();
    let scores_fp = pad(scores.iter().map(|&x| f(x)).collect(), n);
    let choice_fp = pad(choice.iter().map(|&x| f(x)).collect(), n);
    let (com_scores, pd_scores, dom_scores) = fri::commit_vec(&ctx.pcs, &scores_fp);
    let (com_choice, pd_choice, dom_choice) = fri::commit_vec(&ctx.pcs, &choice_fp);
    let sign_table = crate::lookup::lookup_logup::commit_function_table(
        &ctx.pcs,
        |x| if x >= 0 { 1 } else { 0 },
        sign_lo,
        -sign_lo,
        alpha,
        beta,
    );
    prove_topk_committed_given(
        ctx,
        &com_scores,
        &pd_scores,
        dom_scores,
        &com_choice,
        &pd_choice,
        dom_choice,
        scores,
        choice,
        k,
        tau,
        &sign_table,
        alpha,
        beta,
    )
}

/// Prove top-k on *already-committed* `scores` and `choice`. The derived `δ` is
/// committed fresh; `scores` and `choice` are reused so they are shared with the
/// surrounding op graph.
pub fn prove_topk_committed_given(
    ctx: &mut Prover,
    com_scores: &fri::Commitment,
    pd_scores: &fri::ProverData,
    dom_scores: fri::Domain,
    com_choice: &fri::Commitment,
    pd_choice: &fri::ProverData,
    dom_choice: fri::Domain,
    scores: &[i64],
    choice: &[i64],
    k: usize,
    tau: i64,
    sign_table: &FunctionTable,
    alpha: Fp,
    beta: Fp,
) -> TopKProof {
    let n = scores.len();
    assert_eq!(choice.len(), n);
    let e = n;

    let delta: Vec<i64> = (0..n)
        .map(|i| (scores[i] as i64) * (e as i64) - (i as i64) - tau)
        .collect();

    let scores_fp = pad(scores.iter().map(|&x| f(x)).collect(), n);
    let choice_fp = pad(choice.iter().map(|&x| f(x)).collect(), n);
    let delta_fp = pad(delta.iter().map(|&x| f(x)).collect(), n);

    let (com_delta, pd_delta, dom_delta) = fri::commit_vec(&ctx.pcs, &delta_fp);

    // count: ⟨1, choice⟩ = k.
    let ones = vec![Fp::ONE; next_pow2(n)];
    let count_claim = prove_inner_product_committed_given(
        ctx, com_choice, pd_choice, dom_choice, &choice_fp, &ones,
    );

    // δ = e·scores + pub, checked at a shared r.
    let mut tr = fri::Transcript::new(&ctx.challenger);
    tr.observe(&com_delta);
    tr.observe(com_scores);
    let mut r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    r.resize(next_pow2(n), Fp::ZERO);
    let b_delta: Fp = r.iter().zip(&delta_fp).map(|(&ri, &vi)| ri * vi).sum();
    let b_scores: Fp = r.iter().zip(&scores_fp).map(|(&ri, &vi)| ri * vi).sum();
    let delta_claim =
        prove_inner_product_committed_given(ctx, &com_delta, &pd_delta, dom_delta, &delta_fp, &r);
    let scores_claim =
        prove_inner_product_committed_given(ctx, com_scores, pd_scores, dom_scores, &scores_fp, &r);

    // choice = sign(δ): reuse the already-committed δ and choice.
    let sign = prove_lookup2_logup_given(
        ctx, &com_delta, &pd_delta, dom_delta, com_choice, pd_choice, dom_choice, &delta, choice,
        sign_table, alpha, beta,
    );

    TopKProof {
        n,
        k,
        tau,
        com_scores: com_scores.clone(),
        dom_scores,
        com_choice: com_choice.clone(),
        dom_choice,
        com_delta,
        dom_delta,
        count_claim,
        r,
        b_delta,
        b_scores,
        delta_claim,
        scores_claim,
        sign,
        sign_com_ft: sign_table.table.com_ft.clone(),
        sign_dom_ft: sign_table.table.dom_ft,
        alpha,
        beta,
    }
}

pub fn verify_topk_committed(
    challenger: &Challenger,
    proof: &TopKProof,
    values: &[Ext],
    acc: &mut Openings,
) -> bool {
    let n = proof.n;
    let e = n;
    let p = next_pow2(n);

    // count.
    let ones = vec![Fp::ONE; p];
    if !verify_inner_product_committed(
        challenger,
        &proof.com_choice,
        proof.dom_choice,
        &ones,
        Fp::new(proof.k as u64),
        &proof.count_claim,
        values,
        acc,
    ) {
        return false;
    }

    // δ = e·scores + pub.
    let mut tr = fri::Transcript::new(challenger);
    tr.observe(&proof.com_delta);
    tr.observe(&proof.com_scores);
    let mut r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    r.resize(p, Fp::ZERO);
    if !verify_inner_product_committed(
        challenger,
        &proof.com_delta,
        proof.dom_delta,
        &r,
        proof.b_delta,
        &proof.delta_claim,
        values,
        acc,
    ) {
        return false;
    }
    if !verify_inner_product_committed(
        challenger,
        &proof.com_scores,
        proof.dom_scores,
        &r,
        proof.b_scores,
        &proof.scores_claim,
        values,
        acc,
    ) {
        return false;
    }
    let pub_dot: Fp = (0..n).map(|i| r[i] * f(-(i as i64) - proof.tau)).sum();
    if proof.b_delta != Fp::new(e as u64) * proof.b_scores + pub_dot {
        return false;
    }

    // choice = sign(δ).
    verify_lookup2_logup(
        challenger,
        &proof.sign_com_ft,
        proof.sign_dom_ft,
        &proof.sign,
        values,
        acc,
    )
}

// --- causal softmax ------------------------------------------------------------

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CausalSoftmaxProof {
    pub n: usize, // sequence length (window)
    pub l: usize, // causal position
    pub m: i64,   // running max
    pub tstar: usize,
    pub s: i64, // sum of exponentials
    pub exp_lo: i64,
    pub com_scores: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom_scores: fri::Domain,
    pub com_d: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom_d: fri::Domain,
    pub com_e: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom_e: fri::Domain,
    pub com_nrm: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom_nrm: fri::Domain,
    pub com_probs: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom_probs: fri::Domain,
    /// d = mask⊙scores + pub (linear).
    pub r1: Vec<Fp>,
    pub b_d: Fp,
    pub b_sm: Fp, // ⟨r1⊙mask, scores⟩
    pub claim_d: InnerProductProof,
    pub claim_sm: InnerProductProof,
    /// e = exp_q15(d).
    pub exp_lk: Lookup2LogUpProof,
    pub exp_com_ft: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub exp_dom_ft: fri::Domain,
    /// S = ⟨mask, e⟩.
    pub sum_claim: InnerProductProof,
    /// nrm = requant(e·inv_q, Q), inv_q = round(2^(15+Q)/S).
    pub inv_q: i64,
    pub com_y: fri::Commitment, // y = e·inv_q
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom_y: fri::Domain,
    pub r_y: Vec<Fp>,
    pub b_y: Fp,
    pub b_e: Fp,
    pub claim_y: InnerProductProof,
    pub claim_e: InnerProductProof,
    pub req: RequantProof,
    /// probs = mask⊙nrm (linear).
    pub r2: Vec<Fp>,
    pub b_probs: Fp,
    pub b_nm: Fp, // ⟨r2⊙mask, nrm⟩
    pub claim_probs: InnerProductProof,
    pub claim_nm: InnerProductProof,
    /// d[t*] = 0 (max tightness).
    pub tight_claim: InnerProductProof,
    pub alpha: Fp,
    pub beta: Fp,
}

pub fn prove_causal_softmax_committed(
    ctx: &mut Prover,
    scores: &[i64],
    probs: &[i64],
    l: usize,
    exp_lo: i64,
    alpha: Fp,
    beta: Fp,
) -> CausalSoftmaxProof {
    let n = scores.len();
    let scores_fp = pad(scores.iter().map(|&x| f(x)).collect(), n);
    let probs_fp = pad(probs.iter().map(|&x| f(x)).collect(), n);
    let (com_scores, pd_scores, dom_scores) = fri::commit_vec(&ctx.pcs, &scores_fp);
    let (com_probs, pd_probs, dom_probs) = fri::commit_vec(&ctx.pcs, &probs_fp);
    let exp_table = crate::lookup::lookup_logup::commit_function_table(
        &ctx.pcs,
        crate::model::spec::exp_q15,
        exp_lo,
        0,
        alpha,
        beta,
    );
    let digit_table = crate::lookup::range_check::commit_digit_table(&ctx.pcs);
    let bit_table = crate::lookup::requant::commit_bit_table(&ctx.pcs);
    prove_causal_softmax_committed_given(
        ctx,
        &com_scores,
        &pd_scores,
        dom_scores,
        &com_probs,
        &pd_probs,
        dom_probs,
        scores,
        probs,
        l,
        &exp_table,
        &digit_table,
        &bit_table,
        alpha,
        beta,
    )
}

/// Prove causal softmax on *already-committed* `scores` and `probs`. The derived
/// `d`/`e`/`nrm` are committed fresh; `scores` and `probs` are reused.
#[allow(clippy::too_many_arguments)]
pub fn prove_causal_softmax_committed_given(
    ctx: &mut Prover,
    com_scores: &fri::Commitment,
    pd_scores: &fri::ProverData,
    dom_scores: fri::Domain,
    com_probs: &fri::Commitment,
    pd_probs: &fri::ProverData,
    dom_probs: fri::Domain,
    scores: &[i64],
    probs: &[i64],
    l: usize,
    exp_table: &FunctionTable,
    digit_table: &TableCommitment,
    bit_table: &TableCommitment,
    alpha: Fp,
    beta: Fp,
) -> CausalSoftmaxProof {
    let n = scores.len();
    assert_eq!(probs.len(), n);
    assert!(l < n);

    let m = *scores[..=l].iter().max().unwrap();
    let tstar = (0..=l).find(|&t| scores[t] == m).unwrap();

    let mut d = vec![0i64; n];
    let mut e = vec![0i64; n];
    for i in 0..n {
        d[i] = if i <= l { scores[i] - m } else { exp_table.lo };
        e[i] = crate::model::spec::exp_q15(d[i]);
    }
    let s: i64 = (0..=l).map(|t| e[t]).sum();
    let inv_q = ((1i64 << (RECIP_Q + 15)) as f64 / s as f64).round() as i64;
    let y_vals: Vec<i64> = e.iter().map(|&x| x * inv_q).collect();
    let nrm: Vec<i64> = y_vals
        .iter()
        .map(|&y| crate::model::spec::requant(y, RECIP_Q))
        .collect();

    let scores_fp = pad(scores.iter().map(|&x| f(x)).collect(), n);
    let probs_fp = pad(probs.iter().map(|&x| f(x)).collect(), n);
    let d_fp = pad(d.iter().map(|&x| f(x)).collect(), n);
    let e_fp = pad(e.iter().map(|&x| f(x)).collect(), n);
    let nrm_fp = pad(nrm.iter().map(|&x| f(x)).collect(), n);
    let y_fp = pad(y_vals.iter().map(|&x| f(x)).collect(), n);

    let (com_d, pd_d, dom_d) = fri::commit_vec(&ctx.pcs, &d_fp);
    let (com_e, pd_e, dom_e) = fri::commit_vec(&ctx.pcs, &e_fp);
    let (com_nrm, pd_nrm, dom_nrm) = fri::commit_vec(&ctx.pcs, &nrm_fp);
    let (com_y, pd_y, dom_y) = fri::commit_vec(&ctx.pcs, &y_fp);

    let p = next_pow2(n);
    let mask: Vec<Fp> = (0..p)
        .map(|i| if i <= l { Fp::ONE } else { Fp::ZERO })
        .collect();

    // 1. d = mask⊙scores + pub.
    let mut tr = fri::Transcript::new(&ctx.challenger);
    tr.observe(&com_d);
    tr.observe(com_scores);
    let mut r1: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    r1.resize(p, Fp::ZERO);
    let r1m: Vec<Fp> = (0..p).map(|i| r1[i] * mask[i]).collect();
    let b_d: Fp = r1.iter().zip(&d_fp).map(|(&ri, &vi)| ri * vi).sum();
    let b_sm: Fp = r1m.iter().zip(&scores_fp).map(|(&ri, &vi)| ri * vi).sum();
    let claim_d = prove_inner_product_committed_given(ctx, &com_d, &pd_d, dom_d, &d_fp, &r1);
    let claim_sm = prove_inner_product_committed_given(
        ctx, com_scores, pd_scores, dom_scores, &scores_fp, &r1m,
    );

    // 2. e = exp_q15(d).
    let exp_lk = prove_lookup2_logup_given(
        ctx, &com_d, &pd_d, dom_d, &com_e, &pd_e, dom_e, &d, &e, exp_table, alpha, beta,
    );

    // 3. S = ⟨mask, e⟩.
    let sum_claim = prove_inner_product_committed_given(ctx, &com_e, &pd_e, dom_e, &e_fp, &mask);

    // 4. y = e·inv_q, nrm = requant(y, Q).
    let mut tr_y = fri::Transcript::new(&ctx.challenger);
    tr_y.observe(&com_y);
    tr_y.observe(&com_e);
    let mut r_y: Vec<Fp> = (0..n).map(|_| tr_y.sample_val()).collect();
    r_y.resize(p, Fp::ZERO);
    let b_y: Fp = r_y.iter().zip(&y_fp).map(|(&ri, &vi)| ri * vi).sum();
    let b_e: Fp = r_y.iter().zip(&e_fp).map(|(&ri, &vi)| ri * vi).sum();
    let claim_y = prove_inner_product_committed_given(ctx, &com_y, &pd_y, dom_y, &y_fp, &r_y);
    let claim_e = prove_inner_product_committed_given(ctx, &com_e, &pd_e, dom_e, &e_fp, &r_y);
    let req = prove_requant_given(
        ctx,
        &com_y,
        &pd_y,
        dom_y,
        &y_vals,
        &nrm,
        RECIP_Q,
        6,
        digit_table,
        bit_table,
    );

    // 5. probs = mask⊙nrm.
    let mut tr2 = fri::Transcript::new(&ctx.challenger);
    tr2.observe(com_probs);
    tr2.observe(&com_nrm);
    let mut r2: Vec<Fp> = (0..n).map(|_| tr2.sample_val()).collect();
    r2.resize(p, Fp::ZERO);
    let r2m: Vec<Fp> = (0..p).map(|i| r2[i] * mask[i]).collect();
    let b_probs: Fp = r2.iter().zip(&probs_fp).map(|(&ri, &vi)| ri * vi).sum();
    let b_nm: Fp = r2m.iter().zip(&nrm_fp).map(|(&ri, &vi)| ri * vi).sum();
    let claim_probs =
        prove_inner_product_committed_given(ctx, com_probs, pd_probs, dom_probs, &probs_fp, &r2);
    let claim_nm =
        prove_inner_product_committed_given(ctx, &com_nrm, &pd_nrm, dom_nrm, &nrm_fp, &r2m);

    // 6. d[t*] = 0.
    let mut e_tstar = vec![Fp::ZERO; p];
    e_tstar[tstar] = Fp::ONE;
    let tight_claim =
        prove_inner_product_committed_given(ctx, &com_d, &pd_d, dom_d, &d_fp, &e_tstar);

    CausalSoftmaxProof {
        n,
        l,
        m,
        tstar,
        s,
        exp_lo: exp_table.lo,
        com_scores: com_scores.clone(),
        dom_scores,
        com_d,
        dom_d,
        com_e,
        dom_e,
        com_nrm,
        dom_nrm,
        com_probs: com_probs.clone(),
        dom_probs,
        r1,
        b_d,
        b_sm,
        claim_d,
        claim_sm,
        exp_lk,
        exp_com_ft: exp_table.table.com_ft.clone(),
        exp_dom_ft: exp_table.table.dom_ft,
        sum_claim,
        inv_q,
        com_y,
        dom_y,
        r_y,
        b_y,
        b_e,
        claim_y,
        claim_e,
        req,
        r2,
        b_probs,
        b_nm,
        claim_probs,
        claim_nm,
        tight_claim,
        alpha,
        beta,
    }
}

pub fn verify_causal_softmax_committed(
    challenger: &Challenger,
    digit_table: &TableCommitment,
    bit_table: &TableCommitment,
    proof: &CausalSoftmaxProof,
    values: &[Ext],
    acc: &mut Openings,
) -> bool {
    let n = proof.n;
    let p = next_pow2(n);
    let mask: Vec<Fp> = (0..p)
        .map(|i| if i <= proof.l { Fp::ONE } else { Fp::ZERO })
        .collect();

    // 1. d = mask⊙scores + pub.
    let mut tr = fri::Transcript::new(challenger);
    tr.observe(&proof.com_d);
    tr.observe(&proof.com_scores);
    let mut r1: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    r1.resize(p, Fp::ZERO);
    let r1m: Vec<Fp> = (0..p).map(|i| r1[i] * mask[i]).collect();
    if !verify_inner_product_committed(
        challenger,
        &proof.com_d,
        proof.dom_d,
        &r1,
        proof.b_d,
        &proof.claim_d,
        values,
        acc,
    ) {
        return false;
    }
    if !verify_inner_product_committed(
        challenger,
        &proof.com_scores,
        proof.dom_scores,
        &r1m,
        proof.b_sm,
        &proof.claim_sm,
        values,
        acc,
    ) {
        return false;
    }
    let pub_dot: Fp = (0..n)
        .map(|i| {
            let pub_i = if i <= proof.l { -proof.m } else { proof.exp_lo };
            r1[i] * f(pub_i)
        })
        .sum();
    if proof.b_d != proof.b_sm + pub_dot {
        return false;
    }

    // 2. e = exp_q15(d).
    if !verify_lookup2_logup(
        challenger,
        &proof.exp_com_ft,
        proof.exp_dom_ft,
        &proof.exp_lk,
        values,
        acc,
    ) {
        return false;
    }

    // 3. S = ⟨mask, e⟩.
    if !verify_inner_product_committed(
        challenger,
        &proof.com_e,
        proof.dom_e,
        &mask,
        f(proof.s),
        &proof.sum_claim,
        values,
        acc,
    ) {
        return false;
    }

    // 4. y = e·inv_q, nrm = requant(y, Q).
    let inv_q = ((1i64 << (RECIP_Q + 15)) as f64 / proof.s as f64).round() as i64;
    if inv_q != proof.inv_q {
        return false;
    }
    let mut tr_y = fri::Transcript::new(challenger);
    tr_y.observe(&proof.com_y);
    tr_y.observe(&proof.com_e);
    let mut r_y: Vec<Fp> = (0..n).map(|_| tr_y.sample_val()).collect();
    r_y.resize(p, Fp::ZERO);
    if !verify_inner_product_committed(
        challenger,
        &proof.com_y,
        proof.dom_y,
        &r_y,
        proof.b_y,
        &proof.claim_y,
        values,
        acc,
    ) || !verify_inner_product_committed(
        challenger,
        &proof.com_e,
        proof.dom_e,
        &r_y,
        proof.b_e,
        &proof.claim_e,
        values,
        acc,
    ) || proof.b_y != Fp::new(inv_q as u64) * proof.b_e
    {
        return false;
    }
    if !verify_requant(challenger, digit_table, bit_table, &proof.req, values, acc) {
        return false;
    }

    // 5. probs = mask⊙nrm.
    let mut tr2 = fri::Transcript::new(challenger);
    tr2.observe(&proof.com_probs);
    tr2.observe(&proof.com_nrm);
    let mut r2: Vec<Fp> = (0..n).map(|_| tr2.sample_val()).collect();
    r2.resize(p, Fp::ZERO);
    let r2m: Vec<Fp> = (0..p).map(|i| r2[i] * mask[i]).collect();
    if !verify_inner_product_committed(
        challenger,
        &proof.com_probs,
        proof.dom_probs,
        &r2,
        proof.b_probs,
        &proof.claim_probs,
        values,
        acc,
    ) {
        return false;
    }
    if !verify_inner_product_committed(
        challenger,
        &proof.com_nrm,
        proof.dom_nrm,
        &r2m,
        proof.b_nm,
        &proof.claim_nm,
        values,
        acc,
    ) {
        return false;
    }
    if proof.b_probs != proof.b_nm {
        return false;
    }

    // 6. d[t*] = 0.
    let mut e_tstar = vec![Fp::ZERO; p];
    e_tstar[proof.tstar] = Fp::ONE;
    verify_inner_product_committed(
        challenger,
        &proof.com_d,
        proof.dom_d,
        &e_tstar,
        Fp::ZERO,
        &proof.tight_claim,
        values,
        acc,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(x: u64) -> Fp {
        Fp::new(x)
    }

    #[test]
    fn topk_committed_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        // top-2 of {10,30,20,40} = indices {3 (40), 1 (30)}.
        let scores = [10i64, 30, 20, 40];
        let choice = [0i64, 1, 0, 1];
        // tau is the boundary adjusted score; any value strictly between the
        // k-th and (k+1)-th adjusted scores works. Adjusted: e*s-i = 4*s-i →
        // [40,119,78,160]; sorted desc [160,119,78,40]; k=2 boundary between 119 and 78.
        let tau = 80;
        let proof = prove_topk_committed(&mut ctx, &scores, &choice, 2, tau, -1000, f(11), f(5));
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = crate::pcs::zk::Openings::new();
        assert!(verify_topk_committed(
            &ctx.challenger,
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
    fn topk_committed_rejects_wrong_choice() {
        let mut ctx = fri::Prover::new(1, 10);
        let scores = [10i64, 30, 20, 40];
        let choice = [0i64, 0, 1, 1]; // picks {20,40} instead of {30,40}
        let proof = prove_topk_committed(&mut ctx, &scores, &choice, 2, 80, -1000, f(11), f(5));
        ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = crate::pcs::zk::Openings::new();
        assert!(!verify_topk_committed(
            &ctx.challenger,
            &proof,
            &values,
            &mut acc
        ));
    }

    #[test]
    fn topk_committed_rejects_wrong_count() {
        let mut ctx = fri::Prover::new(1, 10);
        let scores = [10i64, 30, 20, 40];
        let choice = [0i64, 1, 0, 0]; // only one selected
        let proof = prove_topk_committed(&mut ctx, &scores, &choice, 2, 80, -1000, f(11), f(5));
        ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = crate::pcs::zk::Openings::new();
        assert!(!verify_topk_committed(
            &ctx.challenger,
            &proof,
            &values,
            &mut acc
        ));
    }

    #[test]
    fn causal_softmax_committed_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        // scores = [3,1,5,2], l=2 → m=5, d=[-2,-4,0,·], e, S, probs.
        let scores = [3i64, 1, 5, 2];
        let l = 2;
        let m = 5i64;
        let e: Vec<i64> = (0..=l)
            .map(|t| crate::model::spec::exp_q15(scores[t] - m))
            .collect();
        let sum: i64 = e.iter().sum();
        let mut probs = vec![0i64; 4];
        for t in 0..=l {
            probs[t] = crate::model::spec::requant(
                e[t] * ((1i64 << (crate::model::spec::RECIP_Q + 15)) as f64 / sum as f64).round() as i64,
                crate::model::spec::RECIP_Q,
            );
        }
        let proof = prove_causal_softmax_committed(&mut ctx, &scores, &probs, l, -100, f(11), f(5));
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = crate::pcs::zk::Openings::new();
        assert!(verify_causal_softmax_committed(
            &ctx.challenger,
            &crate::lookup::range_check::commit_digit_table(&ctx.pcs),
            &crate::lookup::requant::commit_bit_table(&ctx.pcs),
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
    fn causal_softmax_committed_rejects_wrong() {
        let mut ctx = fri::Prover::new(1, 10);
        let scores = [3i64, 1, 5, 2];
        let l = 2;
        let m = 5i64;
        let e: Vec<i64> = (0..=l)
            .map(|t| crate::model::spec::exp_q15(scores[t] - m))
            .collect();
        let sum: i64 = e.iter().sum();
        let mut probs = vec![0i64; 4];
        for t in 0..=l {
            probs[t] = crate::model::spec::requant(
                e[t] * ((1i64 << (crate::model::spec::RECIP_Q + 15)) as f64 / sum as f64).round() as i64,
                crate::model::spec::RECIP_Q,
            );
        }
        probs[1] += 1; // tamper
        let proof = prove_causal_softmax_committed(&mut ctx, &scores, &probs, l, -100, f(11), f(5));
        ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = crate::pcs::zk::Openings::new();
        assert!(!verify_causal_softmax_committed(
            &ctx.challenger,
            &crate::lookup::range_check::commit_digit_table(&ctx.pcs),
            &crate::lookup::requant::commit_bit_table(&ctx.pcs),
            &proof,
            &values,
            &mut acc,
        ));
    }

    #[test]
    fn causal_softmax_committed_rejects_unmasked() {
        let mut ctx = fri::Prover::new(1, 10);
        let scores = [3i64, 1, 5, 2];
        let l = 1;
        let m = *scores[..=l].iter().max().unwrap();
        let e: Vec<i64> = (0..=l)
            .map(|t| crate::model::spec::exp_q15(scores[t] - m))
            .collect();
        let sum: i64 = e.iter().sum();
        let mut probs = vec![0i64; 4];
        for t in 0..=l {
            probs[t] = crate::model::spec::requant(
                e[t] * ((1i64 << (crate::model::spec::RECIP_Q + 15)) as f64 / sum as f64).round() as i64,
                crate::model::spec::RECIP_Q,
            );
        }
        probs[3] = 1; // t=3 > l should be 0
        let proof = prove_causal_softmax_committed(&mut ctx, &scores, &probs, l, -100, f(11), f(5));
        ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = crate::pcs::zk::Openings::new();
        assert!(!verify_causal_softmax_committed(
            &ctx.challenger,
            &crate::lookup::range_check::commit_digit_table(&ctx.pcs),
            &crate::lookup::requant::commit_bit_table(&ctx.pcs),
            &proof,
            &values,
            &mut acc,
        ));
    }
}
