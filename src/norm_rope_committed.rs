//! Committed `rope` and `norm` (the two remaining per-layer nonlinearities that
//! reduce to a public linear map + a lookup).
//!
//!   rope: rot = R·inp (public block-diagonal rotation), out = requant(rot, 15)
//!   norm: sum_sq = ⟨x,x⟩ (IPA), h = x ⊙ gamma (mul), out = round(h · inv)

use crate::field::{Ext, Fp};
use crate::fri::{self, Challenger, Prover};
use crate::ipa::{prove_ipa_committed, verify_ipa_committed, CommittedIpa};
use crate::lookup_committed::{prove_lookup2_committed, verify_lookup2_committed, Lookup2Proof};
use crate::mul_committed::{prove_mul_committed, verify_mul_committed, MulProof};
use crate::zk::{
    prove_inner_product_committed_given, verify_inner_product_committed, InnerProductProof,
    Openings,
};
use p3_field::PrimeCharacteristicRing;

fn f(x: i64) -> Fp {
    if x < 0 {
        -Fp::new(x.unsigned_abs())
    } else {
        Fp::new(x as u64)
    }
}

// --- RoPE ---------------------------------------------------------------------

pub struct RopeProof {
    pub n: usize,
    pub com_rot: fri::Commitment,
    pub dom_rot: fri::Domain,
    pub com_inp: fri::Commitment,
    pub dom_inp: fri::Domain,
    pub r: Vec<Fp>,
    pub rt: Vec<Fp>, // R^T r (public, derived from pos/theta/r)
    pub b_rot: Fp,
    pub b_inp: Fp,
    pub claim_rot: InnerProductProof,
    pub claim_inp: InnerProductProof,
    pub req: Lookup2Proof,
    pub req_lo: i64,
}

pub fn prove_rope_committed(
    ctx: &mut Prover,
    inp: &[i64],
    out: &[i64],
    pos: usize,
    theta: f64,
    req_lo: i64,
    alpha: Fp,
    beta: Fp,
) -> RopeProof {
    let d = inp.len();
    // rot = R·inp (public rotation).
    let mut rot = vec![0i64; d];
    for j in 0..(d / 2) {
        let i = 2 * j;
        let angle = (pos as f64) * theta.powf(-(i as f64) / d as f64);
        let c = (angle.cos() * 32768.0).round() as i64;
        let s = (angle.sin() * 32768.0).round() as i64;
        rot[i] = inp[i] * c - inp[i + 1] * s;
        rot[i + 1] = inp[i] * s + inp[i + 1] * c;
    }

    let inp_fp: Vec<Fp> = inp.iter().map(|&x| f(x)).collect();
    let rot_fp: Vec<Fp> = rot.iter().map(|&x| f(x)).collect();

    let (com_rot, pd_rot, dom_rot) = fri::commit_vec(&ctx.pcs, &rot_fp);
    let (com_inp, pd_inp, dom_inp) = fri::commit_vec(&ctx.pcs, &inp_fp);

    // Rotation check: ⟨r, rot⟩ = ⟨R^T r, inp⟩.
    let mut tr = fri::Transcript::new(&ctx.challenger);
    tr.observe(&com_rot);
    tr.observe(&com_inp);
    let r: Vec<Fp> = (0..d).map(|_| tr.sample_val()).collect();
    // R^T r, per pair.
    let mut rt = vec![Fp::ZERO; d];
    for j in 0..(d / 2) {
        let i = 2 * j;
        let angle = (pos as f64) * theta.powf(-(i as f64) / d as f64);
        let c = f((angle.cos() * 32768.0).round() as i64);
        let s = f((angle.sin() * 32768.0).round() as i64);
        rt[i] = r[i] * c + r[i + 1] * s;
        rt[i + 1] = -r[i] * s + r[i + 1] * c;
    }
    let b_rot: Fp = r.iter().zip(&rot_fp).map(|(&ri, &vi)| ri * vi).sum();
    let b_inp: Fp = rt.iter().zip(&inp_fp).map(|(&ri, &vi)| ri * vi).sum();
    let claim_rot =
        prove_inner_product_committed_given(ctx, &com_rot, &pd_rot, dom_rot, &rot_fp, &r);
    let claim_inp =
        prove_inner_product_committed_given(ctx, &com_inp, &pd_inp, dom_inp, &inp_fp, &rt);

    let req = prove_lookup2_committed(ctx, &rot, out, req_lo, alpha, beta);

    RopeProof {
        n: d,
        com_rot,
        dom_rot,
        com_inp,
        dom_inp,
        r,
        rt,
        b_rot,
        b_inp,
        claim_rot,
        claim_inp,
        req,
        req_lo,
    }
}

pub fn verify_rope_committed(
    challenger: &Challenger,
    proof: &RopeProof,
    values: &[Ext],
    acc: &mut Openings,
) -> bool {
    // Replay the rotation check: ⟨r, rot⟩ = b_rot = b_inp = ⟨R^T r, inp⟩.
    if !verify_inner_product_committed(
        challenger,
        &proof.com_rot,
        proof.dom_rot,
        &proof.r,
        proof.b_rot,
        &proof.claim_rot,
        values,
        acc,
    ) {
        return false;
    }
    if !verify_inner_product_committed(
        challenger,
        &proof.com_inp,
        proof.dom_inp,
        &proof.rt,
        proof.b_inp,
        &proof.claim_inp,
        values,
        acc,
    ) {
        return false;
    }
    proof.b_rot == proof.b_inp
        && verify_lookup2_committed(
            challenger,
            |x| crate::spec::requant(x, 15),
            proof.req_lo,
            proof.req.alpha,
            proof.req.beta,
            &proof.req,
            values,
            acc,
        )
}

// --- RMSNorm ------------------------------------------------------------------

pub struct NormProof {
    pub n: usize,
    pub sum_sq: Fp,
    pub ipa: CommittedIpa,   // ⟨x, x⟩ = sum_sq
    pub mul: MulProof,       // h = x ⊙ gamma
    pub scale: Lookup2Proof, // out = round(h · inv)
    pub scale_lo: i64,
    pub inv: f64,
}

#[allow(clippy::too_many_arguments)]
pub fn prove_norm_committed(
    ctx: &mut Prover,
    x: &[i64],
    gamma: &[i8],
    out: &[i64],
    eps: f64,
    scale_lo: i64,
    alpha: Fp,
    beta: Fp,
) -> NormProof {
    let d = x.len();
    let sum_sq: i64 = x.iter().map(|&v| v * v).sum();
    let inv = 1.0 / ((sum_sq as f64 / d as f64) + eps).sqrt();
    // h[i] = x[i] · gamma[i].
    let h: Vec<i64> = x
        .iter()
        .zip(gamma)
        .map(|(&xi, &gi)| xi * gi as i64)
        .collect();
    // out[i] = round(h[i] · inv).
    let out_expected: Vec<i64> = h
        .iter()
        .map(|&hi| (hi as f64 * inv).round() as i64)
        .collect();
    debug_assert_eq!(out_expected, out);

    let x_fp: Vec<Fp> = x.iter().map(|&v| f(v)).collect();
    let g_fp: Vec<Fp> = gamma.iter().map(|&g| f(g as i64)).collect();
    let h_fp: Vec<Fp> = h.iter().map(|&v| f(v)).collect();

    let ipa = prove_ipa_committed(ctx, &x_fp, &x_fp, f(sum_sq));
    let mul = prove_mul_committed(ctx, &x_fp, &g_fp, &h_fp);
    let scale = prove_lookup2_committed(ctx, &h, out, scale_lo, alpha, beta);

    NormProof {
        n: d,
        sum_sq: f(sum_sq),
        ipa,
        mul,
        scale,
        scale_lo,
        inv,
    }
}

pub fn verify_norm_committed(
    challenger: &Challenger,
    proof: &NormProof,
    values: &[Ext],
    acc: &mut Openings,
) -> bool {
    // sum_sq = ⟨x,x⟩, h = x⊙gamma, out = round(h·inv).
    verify_ipa_committed(challenger, proof.sum_sq, &proof.ipa, values, acc)
        && verify_mul_committed(challenger, &proof.mul, values, acc)
        && verify_lookup2_committed(
            challenger,
            |hi| (hi as f64 * proof.inv).round() as i64,
            proof.scale_lo,
            proof.scale.alpha,
            proof.scale.beta,
            &proof.scale,
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
    fn rope_committed_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        let inp = [1i64, 2];
        let mut out = inp.to_vec();
        crate::spec::rope_apply(&mut out, 3, 10000.0);
        let proof = prove_rope_committed(&mut ctx, &inp, &out, 3, 10000.0, -65_000, f(11), f(5));
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = crate::zk::Openings::new();
        assert!(verify_rope_committed(
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
    fn norm_committed_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        let x = [1i64, 2, 3, 4];
        let gamma = [1i8, 1, 1, 1];
        let out = crate::spec::rmsnorm_vec(&x, &gamma, 1e-6);
        let proof = prove_norm_committed(&mut ctx, &x, &gamma, &out, 1e-6, -1000, f(11), f(5));
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = crate::zk::Openings::new();
        assert!(verify_norm_committed(
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
}
