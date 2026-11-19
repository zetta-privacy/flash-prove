//! Committed range check via signed base-2^8 digit decomposition.
//!
//! Proves a committed integer vector `x` is bounded and reconstructable by writing
//! each `x[i] = Σ_d digit_d[i] · 256^d` with `digit_d[i] ∈ [-2^7, 2^7)`. The two
//! parts:
//!
//!  1. reconstruction `⟨r, x⟩ = Σ_d 256^d · ⟨r, digit_d⟩` (one linear claim on `x`
//!     and one per digit column, at a shared `r`), and
//!  2. every digit value is in `[-128, 127]` (one LogUp into the 256-entry digit
//!     table).
//!
//! This is the building block for the arithmetic nonlinearities (`requant`,
//! `round_div`, norm `1/sqrt`, softmax normalize) — their "lookup" is replaced by a
//! digit/bit decomposition plus a range membership, which is O(log n + log 256).

use crate::field::{Ext, Fp};
use crate::fri::{self, Challenger, Pcs, Prover};
use crate::logup::{self, LogUpProof, TableCommitment};
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

/// Split an integer into `n` signed base-2^8 digits (least significant first),
/// each in `[-2^7, 2^7)`, with `value = Σ digit_i · 256^i` (matches
/// [`crate::model::to_digits`]).
fn to_digits(mut value: i64, n: usize) -> Vec<i8> {
    let mut digits = Vec::with_capacity(n);
    for _ in 0..n {
        let mut d = value & 0xff;
        value >>= 8;
        if d >= 128 {
            d -= 256;
            value += 1;
        }
        digits.push(d as i8);
    }
    digits
}

/// The digit table `[1, 256]` (the digit range `[-128, 127]` shifted by `+129` so
/// no entry is zero — the fraction LogUp inverts table values), committed once and
/// shared.
pub fn commit_digit_table(pcs: &Pcs) -> TableCommitment {
    let ft: Vec<Fp> = (1i64..=256).map(f).collect();
    logup::commit_table(pcs, &ft)
}

pub struct RangeCheckProof {
    pub n: usize,
    pub digits: usize,
    pub com_x: fri::Commitment,
    pub dom_x: fri::Domain,
    pub com_digits: Vec<fri::Commitment>,
    pub dom_digits: Vec<fri::Domain>,
    pub r: Vec<Fp>,
    pub b_x: Fp,
    pub b_digits: Vec<Fp>,
    pub claim_x: InnerProductProof,
    pub claim_digits: Vec<InnerProductProof>,
    pub logup: LogUpProof,
}

/// Prove `x` decomposes into `digits` signed base-2^8 digits each in `[-128, 127]`,
/// with `x` already committed.
pub fn prove_range_check_given(
    ctx: &mut Prover,
    com_x: &fri::Commitment,
    pd_x: &fri::ProverData,
    dom_x: fri::Domain,
    x: &[i64],
    digits: usize,
    digit_table: &TableCommitment,
) -> RangeCheckProof {
    let n = x.len();

    // Decompose and commit each digit column.
    let x_fp = pad(x.iter().map(|&v| f(v)).collect(), n);
    let mut com_digits = Vec::with_capacity(digits);
    let mut dom_digits = Vec::with_capacity(digits);
    let mut pd_digits = Vec::with_capacity(digits);
    let mut digit_fps = Vec::with_capacity(digits);
    let mut all_digits = Vec::with_capacity(n * digits);
    for d in 0..digits {
        let col: Vec<i64> = x.iter().map(|&v| to_digits(v, digits)[d] as i64).collect();
        all_digits.extend_from_slice(&col);
        let col_fp = pad(col.iter().map(|&v| f(v)).collect(), n);
        let (c, pd, dm) = fri::commit_vec(&ctx.pcs, &col_fp);
        com_digits.push(c);
        pd_digits.push(pd);
        dom_digits.push(dm);
        digit_fps.push(col_fp);
    }

    // Reconstruction: ⟨r, x⟩ = Σ_d 256^d · ⟨r, digit_d⟩.
    let mut tr = fri::Transcript::new(&ctx.challenger);
    tr.observe(com_x);
    for c in &com_digits {
        tr.observe(c);
    }
    let mut r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    r.resize(next_pow2(n), Fp::ZERO);
    let b_x: Fp = r.iter().zip(&x_fp).map(|(&ri, &vi)| ri * vi).sum();
    let claim_x = prove_inner_product_committed_given(ctx, com_x, pd_x, dom_x, &x_fp, &r);
    let mut b_digits = Vec::with_capacity(digits);
    let mut claim_digits = Vec::with_capacity(digits);
    for d in 0..digits {
        let bd: Fp = r.iter().zip(&digit_fps[d]).map(|(&ri, &vi)| ri * vi).sum();
        b_digits.push(bd);
        let cl = prove_inner_product_committed_given(
            ctx,
            &com_digits[d],
            &pd_digits[d],
            dom_digits[d],
            &digit_fps[d],
            &r,
        );
        claim_digits.push(cl);
    }

    // Range: every digit value ∈ [-128, 127] (shifted +129 → [1,256], no zero).
    let all_fp: Vec<Fp> = all_digits.iter().map(|&v| f(v + 129)).collect();
    let mut m = vec![0usize; 256];
    for &v in &all_digits {
        m[(v + 128) as usize] += 1;
    }
    let m_fp: Vec<Fp> = m.iter().map(|&c| Fp::new(c as u64)).collect();
    let logup = logup::prove_logup_committed(ctx, &all_fp, &m_fp, digit_table);

    RangeCheckProof {
        n,
        digits,
        com_x: com_x.clone(),
        dom_x,
        com_digits,
        dom_digits,
        r,
        b_x,
        b_digits,
        claim_x,
        claim_digits,
        logup,
    }
}

pub fn verify_range_check(
    challenger: &Challenger,
    digit_table: &TableCommitment,
    proof: &RangeCheckProof,
    values: &[Ext],
    acc: &mut Openings,
) -> bool {
    let n = proof.n;
    let digits = proof.digits;

    let mut tr = fri::Transcript::new(challenger);
    tr.observe(&proof.com_x);
    for c in &proof.com_digits {
        tr.observe(c);
    }
    let mut r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    r.resize(next_pow2(n), Fp::ZERO);

    if !verify_inner_product_committed(
        challenger,
        &proof.com_x,
        proof.dom_x,
        &r,
        proof.b_x,
        &proof.claim_x,
        values,
        acc,
    ) {
        return false;
    }
    let mut acc_scalar = Fp::ZERO;
    for d in 0..digits {
        if !verify_inner_product_committed(
            challenger,
            &proof.com_digits[d],
            proof.dom_digits[d],
            &r,
            proof.b_digits[d],
            &proof.claim_digits[d],
            values,
            acc,
        ) {
            return false;
        }
        acc_scalar += Fp::new(256u64.pow(d as u32)) * proof.b_digits[d];
    }
    if proof.b_x != acc_scalar {
        return false;
    }

    logup::verify_logup_committed(
        challenger,
        &digit_table.com_ft,
        digit_table.dom_ft,
        &proof.logup,
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
    fn range_check_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        let table = commit_digit_table(&ctx.pcs);
        // x = [5, 300, -1, 256] needs 2 digits each.
        let x = [5i64, 300, -1, 256];
        let digits = 2;
        let x_fp = pad(x.iter().map(|&v| f(v as u64)).collect(), 4);
        let (com_x, pd_x, dom_x) = fri::commit_vec(&ctx.pcs, &x_fp);
        let proof = prove_range_check_given(&mut ctx, &com_x, &pd_x, dom_x, &x, digits, &table);
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = crate::zk::Openings::new();
        assert!(verify_range_check(
            &ctx.challenger,
            &table,
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
    fn range_check_rejects_out_of_range() {
        let mut ctx = fri::Prover::new(1, 10);
        let table = commit_digit_table(&ctx.pcs);
        // 300 needs 2 digits (300 = 44 + 1·256); with 1 digit it overflows the
        // [-128,127] digit range, so the LogUp must reject.
        let x = [300i64];
        let digits = 1;
        let x_fp = pad(x.iter().map(|&v| f(v as u64)).collect(), 1);
        let (com_x, pd_x, dom_x) = fri::commit_vec(&ctx.pcs, &x_fp);
        let proof = prove_range_check_given(&mut ctx, &com_x, &pd_x, dom_x, &x, digits, &table);
        ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = crate::zk::Openings::new();
        assert!(!verify_range_check(
            &ctx.challenger,
            &table,
            &proof,
            &values,
            &mut acc
        ));
    }
}
