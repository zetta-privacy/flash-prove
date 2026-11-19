//! Committed `requant`: `out = requant(x, k) = round_half_away(x / 2^k)`.
//!
//! With per-element sign `s[i] = (x[i] < 0) ∈ {0,1}` and magnitude
//! `abs_x[i] = (1 − 2·s[i])·x[i]` (likewise `abs_out`), the identity
//!
//! ```text
//!   out = requant(x, k)  ⟺  rem[i] = abs_x[i] − abs_out[i]·2^k + 2^(k−1)  ∈  [0, 2^k)
//! ```
//!
//! and `sign(out) = sign(x)` (shared `s`). The half-open `[0, 2^k)` range encodes
//! the "round half away from zero" tie-break (the `+2^k` endpoint is excluded).
//!
//!  1. `x` bounded (base-2^8 digit range check),
//!  2. `s ∈ {0,1}` (via `s ⊙ s = s`), `sx = s ⊙ x`, `sout = s ⊙ out` (two muls),
//!  3. `abs_x = x − 2·sx`, `abs_out = out − 2·sout` (linear),
//!  4. `rem = abs_x − abs_out·2^k + 2^(k−1)` (linear),
//!  5. `rem ∈ [0, 2^k)` via a k-bit decomposition (each bit ∈ {0,1}).

use crate::field::{Ext, Fp};
use crate::fri::{self, Challenger, Pcs, Prover};
use crate::logup::{self, LogUpProof, TableCommitment};
use crate::mul_committed::{prove_mul_committed_given, verify_mul_committed, MulProof};
use crate::range_check::{prove_range_check_given, verify_range_check, RangeCheckProof};
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

pub fn requant(x: i64, k: u32) -> i64 {
    crate::spec::requant(x, k)
}

/// The `{0,1}` bit table shifted `+1` → `{1,2}` (avoids the fraction-LogUp zero
/// inverse), committed once.
pub fn commit_bit_table(pcs: &Pcs) -> TableCommitment {
    let ft: Vec<Fp> = (1i64..=2).map(f).collect();
    logup::commit_table(pcs, &ft)
}

/// Bit-decomposition range check: `rem[i] ∈ [0, 2^k)` for all `i`.
pub struct BitDecompProof {
    pub n: usize,
    pub k: u32,
    pub com_rem: fri::Commitment,
    pub dom_rem: fri::Domain,
    pub com_bits: Vec<fri::Commitment>,
    pub dom_bits: Vec<fri::Domain>,
    pub r: Vec<Fp>,
    pub b_rem: Fp,
    pub b_bits: Vec<Fp>,
    pub claim_rem: InnerProductProof,
    pub claim_bits: Vec<InnerProductProof>,
    pub logup: LogUpProof,
}

fn prove_bit_decomp_given(
    ctx: &mut Prover,
    com_rem: &fri::Commitment,
    pd_rem: &fri::ProverData,
    dom_rem: fri::Domain,
    rem: &[i64],
    k: u32,
    bit_table: &TableCommitment,
) -> BitDecompProof {
    let n = rem.len();
    let rem_fp = pad(rem.iter().map(|&v| f(v)).collect(), n);
    let mut com_bits = Vec::with_capacity(k as usize);
    let mut dom_bits = Vec::with_capacity(k as usize);
    let mut pd_bits = Vec::with_capacity(k as usize);
    let mut bit_fps = Vec::with_capacity(k as usize);
    let mut all_bits = Vec::with_capacity(n * k as usize);
    for j in 0..k {
        let col: Vec<i64> = rem.iter().map(|&v| ((v >> j) & 1) as i64).collect();
        all_bits.extend_from_slice(&col);
        let col_fp = pad(col.iter().map(|&v| f(v)).collect(), n);
        let (c, pd, dm) = fri::commit_vec(&ctx.pcs, &col_fp);
        com_bits.push(c);
        pd_bits.push(pd);
        dom_bits.push(dm);
        bit_fps.push(col_fp);
    }

    let mut tr = fri::Transcript::new(&ctx.challenger);
    tr.observe(com_rem);
    for c in &com_bits {
        tr.observe(c);
    }
    let mut r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    r.resize(next_pow2(n), Fp::ZERO);
    let b_rem: Fp = r.iter().zip(&rem_fp).map(|(&ri, &vi)| ri * vi).sum();
    let claim_rem = prove_inner_product_committed_given(ctx, com_rem, pd_rem, dom_rem, &rem_fp, &r);
    let mut b_bits = Vec::with_capacity(k as usize);
    let mut claim_bits = Vec::with_capacity(k as usize);
    for j in 0..k as usize {
        let bb: Fp = r.iter().zip(&bit_fps[j]).map(|(&ri, &vi)| ri * vi).sum();
        b_bits.push(bb);
        let cl = prove_inner_product_committed_given(
            ctx,
            &com_bits[j],
            &pd_bits[j],
            dom_bits[j],
            &bit_fps[j],
            &r,
        );
        claim_bits.push(cl);
    }

    // All bits ∈ {0,1} (shifted +1 → {1,2}).
    let all_fp: Vec<Fp> = all_bits.iter().map(|&v| f(v + 1)).collect();
    let mut m = vec![0usize; 2];
    for &v in &all_bits {
        m[v as usize] += 1;
    }
    let m_fp: Vec<Fp> = m.iter().map(|&c| Fp::new(c as u64)).collect();
    let logup = logup::prove_logup_committed(ctx, &all_fp, &m_fp, bit_table);

    BitDecompProof {
        n,
        k,
        com_rem: com_rem.clone(),
        dom_rem,
        com_bits,
        dom_bits,
        r,
        b_rem,
        b_bits,
        claim_rem,
        claim_bits,
        logup,
    }
}

fn verify_bit_decomp(
    challenger: &Challenger,
    bit_table: &TableCommitment,
    proof: &BitDecompProof,
    values: &[Ext],
    acc: &mut Openings,
) -> bool {
    let n = proof.n;
    let k = proof.k as usize;
    let mut tr = fri::Transcript::new(challenger);
    tr.observe(&proof.com_rem);
    for c in &proof.com_bits {
        tr.observe(c);
    }
    let mut r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    r.resize(next_pow2(n), Fp::ZERO);
    if !verify_inner_product_committed(
        challenger,
        &proof.com_rem,
        proof.dom_rem,
        &r,
        proof.b_rem,
        &proof.claim_rem,
        values,
        acc,
    ) {
        return false;
    }
    let mut sum_bits = Fp::ZERO;
    for j in 0..k {
        if !verify_inner_product_committed(
            challenger,
            &proof.com_bits[j],
            proof.dom_bits[j],
            &r,
            proof.b_bits[j],
            &proof.claim_bits[j],
            values,
            acc,
        ) {
            return false;
        }
        sum_bits += Fp::new(2u64.pow(j as u32)) * proof.b_bits[j];
    }
    if proof.b_rem != sum_bits {
        return false;
    }
    logup::verify_logup_committed(
        challenger,
        &bit_table.com_ft,
        bit_table.dom_ft,
        &proof.logup,
        values,
        acc,
    )
}

pub struct RequantProof {
    pub n: usize,
    pub k: u32,
    pub range_x: RangeCheckProof,
    pub com_s: fri::Commitment,
    pub dom_s: fri::Domain,
    pub com_out: fri::Commitment,
    pub dom_out: fri::Domain,
    pub com_abs_x: fri::Commitment,
    pub dom_abs_x: fri::Domain,
    pub com_sx: fri::Commitment,
    pub dom_sx: fri::Domain,
    pub com_abs_out: fri::Commitment,
    pub dom_abs_out: fri::Domain,
    pub com_sout: fri::Commitment,
    pub dom_sout: fri::Domain,
    pub com_rem: fri::Commitment,
    pub dom_rem: fri::Domain,
    pub mul_ss: MulProof,   // s ⊙ s = s  (s ∈ {0,1})
    pub mul_sx: MulProof,   // sx = s ⊙ x
    pub mul_sout: MulProof, // sout = s ⊙ out
    // abs_x = x − 2·sx, abs_out = out − 2·sout, rem = abs_x − abs_out·2^k + 2^{k−1}.
    pub r: Vec<Fp>,
    pub b_abs_x: Fp,
    pub b_x: Fp,
    pub b_sx: Fp,
    pub b_abs_out: Fp,
    pub b_out: Fp,
    pub b_sout: Fp,
    pub b_rem: Fp,
    pub claim_abs_x: InnerProductProof,
    pub claim_x: InnerProductProof,
    pub claim_sx: InnerProductProof,
    pub claim_abs_out: InnerProductProof,
    pub claim_out: InnerProductProof,
    pub claim_sout: InnerProductProof,
    pub claim_rem: InnerProductProof,
    pub bits: BitDecompProof,
}

/// Prove `out[i] = requant(x[i], k)` with `x` already committed.
pub fn prove_requant_given(
    ctx: &mut Prover,
    com_x: &fri::Commitment,
    pd_x: &fri::ProverData,
    dom_x: fri::Domain,
    x: &[i64],
    out: &[i64],
    k: u32,
    digits: usize,
    digit_table: &TableCommitment,
    bit_table: &TableCommitment,
) -> RequantProof {
    let n = x.len();
    assert_eq!(out.len(), n);

    let range_x = prove_range_check_given(ctx, com_x, pd_x, dom_x, x, digits, digit_table);

    // Per-element sign s[i] = (x[i] < 0); magnitudes.
    let s: Vec<i64> = x.iter().map(|&v| if v < 0 { 1 } else { 0 }).collect();
    let sx: Vec<i64> = (0..n).map(|i| s[i] * x[i]).collect();
    let sout: Vec<i64> = (0..n).map(|i| s[i] * out[i]).collect();
    let abs_x: Vec<i64> = (0..n).map(|i| x[i] - 2 * sx[i]).collect();
    let abs_out: Vec<i64> = (0..n).map(|i| out[i] - 2 * sout[i]).collect();
    let rem: Vec<i64> = (0..n)
        .map(|i| abs_x[i] - abs_out[i] * (1i64 << k) + (1i64 << (k - 1)))
        .collect();

    let x_fp = pad(x.iter().map(|&v| f(v)).collect(), n);
    let out_fp = pad(out.iter().map(|&v| f(v)).collect(), n);
    let s_fp = pad(s.iter().map(|&v| f(v)).collect(), n);
    let sx_fp = pad(sx.iter().map(|&v| f(v)).collect(), n);
    let sout_fp = pad(sout.iter().map(|&v| f(v)).collect(), n);
    let abs_x_fp = pad(abs_x.iter().map(|&v| f(v)).collect(), n);
    let abs_out_fp = pad(abs_out.iter().map(|&v| f(v)).collect(), n);
    let rem_fp = pad(rem.iter().map(|&v| f(v)).collect(), n);
    let (com_s, pd_s, dom_s) = fri::commit_vec(&ctx.pcs, &s_fp);
    let (com_out, pd_out, dom_out) = fri::commit_vec(&ctx.pcs, &out_fp);
    let (com_sx, pd_sx, dom_sx) = fri::commit_vec(&ctx.pcs, &sx_fp);
    let (com_sout, pd_sout, dom_sout) = fri::commit_vec(&ctx.pcs, &sout_fp);
    let (com_abs_x, pd_abs_x, dom_abs_x) = fri::commit_vec(&ctx.pcs, &abs_x_fp);
    let (com_abs_out, pd_abs_out, dom_abs_out) = fri::commit_vec(&ctx.pcs, &abs_out_fp);
    let (com_rem, pd_rem, dom_rem) = fri::commit_vec(&ctx.pcs, &rem_fp);

    // s ∈ {0,1} via s⊙s = s; sx = s⊙x; sout = s⊙out.
    let mul_ss = prove_mul_committed_given(
        ctx, &com_s, &pd_s, dom_s, &com_s, &pd_s, dom_s, &com_s, &pd_s, dom_s, &s_fp, &s_fp, &s_fp,
    );
    let mul_sx = prove_mul_committed_given(
        ctx, &com_s, &pd_s, dom_s, com_x, pd_x, dom_x, &com_sx, &pd_sx, dom_sx, &s_fp, &x_fp,
        &sx_fp,
    );
    let mul_sout = prove_mul_committed_given(
        ctx, &com_s, &pd_s, dom_s, &com_out, &pd_out, dom_out, &com_sout, &pd_sout, dom_sout,
        &s_fp, &out_fp, &sout_fp,
    );

    let mut tr = fri::Transcript::new(&ctx.challenger);
    tr.observe(&com_abs_x);
    tr.observe(com_x);
    tr.observe(&com_sx);
    tr.observe(&com_abs_out);
    tr.observe(&com_out);
    tr.observe(&com_sout);
    tr.observe(&com_rem);
    let mut r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    r.resize(next_pow2(n), Fp::ZERO);

    let b_abs_x: Fp = r.iter().zip(&abs_x_fp).map(|(&ri, &vi)| ri * vi).sum();
    let b_x: Fp = r.iter().zip(&x_fp).map(|(&ri, &vi)| ri * vi).sum();
    let b_sx: Fp = r.iter().zip(&sx_fp).map(|(&ri, &vi)| ri * vi).sum();
    let b_abs_out: Fp = r.iter().zip(&abs_out_fp).map(|(&ri, &vi)| ri * vi).sum();
    let b_out: Fp = r.iter().zip(&out_fp).map(|(&ri, &vi)| ri * vi).sum();
    let b_sout: Fp = r.iter().zip(&sout_fp).map(|(&ri, &vi)| ri * vi).sum();
    let b_rem: Fp = r.iter().zip(&rem_fp).map(|(&ri, &vi)| ri * vi).sum();

    let claim_abs_x =
        prove_inner_product_committed_given(ctx, &com_abs_x, &pd_abs_x, dom_abs_x, &abs_x_fp, &r);
    let claim_x = prove_inner_product_committed_given(ctx, com_x, pd_x, dom_x, &x_fp, &r);
    let claim_sx = prove_inner_product_committed_given(ctx, &com_sx, &pd_sx, dom_sx, &sx_fp, &r);
    let claim_abs_out = prove_inner_product_committed_given(
        ctx,
        &com_abs_out,
        &pd_abs_out,
        dom_abs_out,
        &abs_out_fp,
        &r,
    );
    let claim_out =
        prove_inner_product_committed_given(ctx, &com_out, &pd_out, dom_out, &out_fp, &r);
    let claim_sout =
        prove_inner_product_committed_given(ctx, &com_sout, &pd_sout, dom_sout, &sout_fp, &r);
    let claim_rem =
        prove_inner_product_committed_given(ctx, &com_rem, &pd_rem, dom_rem, &rem_fp, &r);

    let bits = prove_bit_decomp_given(ctx, &com_rem, &pd_rem, dom_rem, &rem, k, bit_table);

    RequantProof {
        n,
        k,
        range_x,
        com_s,
        dom_s,
        com_out,
        dom_out,
        com_abs_x,
        dom_abs_x,
        com_sx,
        dom_sx,
        com_abs_out,
        dom_abs_out,
        com_sout,
        dom_sout,
        com_rem,
        dom_rem,
        mul_ss,
        mul_sx,
        mul_sout,
        r,
        b_abs_x,
        b_x,
        b_sx,
        b_abs_out,
        b_out,
        b_sout,
        b_rem,
        claim_abs_x,
        claim_x,
        claim_sx,
        claim_abs_out,
        claim_out,
        claim_sout,
        claim_rem,
        bits,
    }
}

pub fn verify_requant(
    challenger: &Challenger,
    digit_table: &TableCommitment,
    bit_table: &TableCommitment,
    proof: &RequantProof,
    values: &[Ext],
    acc: &mut Openings,
) -> bool {
    let n = proof.n;
    let k = proof.k;
    let two_k = Fp::new(1u64 << k);
    let half = Fp::new(1u64 << (k - 1));

    if !verify_range_check(challenger, digit_table, &proof.range_x, values, acc) {
        return false;
    }
    if !verify_mul_committed(challenger, &proof.mul_ss, values, acc) {
        return false;
    }
    if !verify_mul_committed(challenger, &proof.mul_sx, values, acc) {
        return false;
    }
    if !verify_mul_committed(challenger, &proof.mul_sout, values, acc) {
        return false;
    }

    let mut tr = fri::Transcript::new(challenger);
    tr.observe(&proof.com_abs_x);
    tr.observe(&proof.range_x.com_x);
    tr.observe(&proof.com_sx);
    tr.observe(&proof.com_abs_out);
    tr.observe(&proof.com_out);
    tr.observe(&proof.com_sout);
    tr.observe(&proof.com_rem);
    let mut r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    r.resize(next_pow2(n), Fp::ZERO);

    if !verify_inner_product_committed(
        challenger,
        &proof.com_abs_x,
        proof.dom_abs_x,
        &r,
        proof.b_abs_x,
        &proof.claim_abs_x,
        values,
        acc,
    ) || !verify_inner_product_committed(
        challenger,
        &proof.range_x.com_x,
        proof.range_x.dom_x,
        &r,
        proof.b_x,
        &proof.claim_x,
        values,
        acc,
    ) || !verify_inner_product_committed(
        challenger,
        &proof.com_sx,
        proof.dom_sx,
        &r,
        proof.b_sx,
        &proof.claim_sx,
        values,
        acc,
    ) || !verify_inner_product_committed(
        challenger,
        &proof.com_abs_out,
        proof.dom_abs_out,
        &r,
        proof.b_abs_out,
        &proof.claim_abs_out,
        values,
        acc,
    ) || !verify_inner_product_committed(
        challenger,
        &proof.com_out,
        proof.dom_out,
        &r,
        proof.b_out,
        &proof.claim_out,
        values,
        acc,
    ) || !verify_inner_product_committed(
        challenger,
        &proof.com_sout,
        proof.dom_sout,
        &r,
        proof.b_sout,
        &proof.claim_sout,
        values,
        acc,
    ) || !verify_inner_product_committed(
        challenger,
        &proof.com_rem,
        proof.dom_rem,
        &r,
        proof.b_rem,
        &proof.claim_rem,
        values,
        acc,
    ) {
        return false;
    }

    // abs_x = x − 2·sx, abs_out = out − 2·sout.
    if proof.b_abs_x != proof.b_x - Fp::new(2) * proof.b_sx {
        return false;
    }
    if proof.b_abs_out != proof.b_out - Fp::new(2) * proof.b_sout {
        return false;
    }
    // rem = abs_x − abs_out·2^k + 2^{k−1}  (the constant is per-element, so it
    // scales by ⟨r, 1⟩ = Σ r).
    let sum_r: Fp = r.iter().copied().sum();
    if proof.b_rem != proof.b_abs_x - proof.b_abs_out * two_k + half * sum_r {
        return false;
    }

    verify_bit_decomp(challenger, bit_table, &proof.bits, values, acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range_check::commit_digit_table;

    #[test]
    fn requant_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        let dt = commit_digit_table(&ctx.pcs);
        let bt = commit_bit_table(&ctx.pcs);
        // x = [5, -5, 100, -100], k=4, digits=2.
        let x = [5i64, -5, 100, -100];
        let k = 4u32;
        let out: Vec<i64> = x.iter().map(|&v| requant(v, k)).collect();
        let x_fp = pad(x.iter().map(|&v| f(v)).collect(), 4);
        let (com_x, pd_x, dom_x) = fri::commit_vec(&ctx.pcs, &x_fp);
        let proof = prove_requant_given(&mut ctx, &com_x, &pd_x, dom_x, &x, &out, k, 2, &dt, &bt);
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = crate::zk::Openings::new();
        assert!(verify_requant(
            &ctx.challenger,
            &dt,
            &bt,
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
    fn requant_rejects_wrong_out() {
        let mut ctx = fri::Prover::new(1, 10);
        let dt = commit_digit_table(&ctx.pcs);
        let bt = commit_bit_table(&ctx.pcs);
        let x = [5i64, -5, 100, -100];
        let k = 4u32;
        let out: Vec<i64> = x.iter().map(|&v| requant(v, k)).collect();
        let mut wrong = out.clone();
        wrong[0] += 1;
        let x_fp = pad(x.iter().map(|&v| f(v)).collect(), 4);
        let (com_x, pd_x, dom_x) = fri::commit_vec(&ctx.pcs, &x_fp);
        let proof = prove_requant_given(&mut ctx, &com_x, &pd_x, dom_x, &x, &wrong, k, 2, &dt, &bt);
        ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = crate::zk::Openings::new();
        assert!(!verify_requant(
            &ctx.challenger,
            &dt,
            &bt,
            &proof,
            &values,
            &mut acc,
        ));
    }
}
