//! Committed attention ops: `scores` and `weighted_v`, the attention block's two
//! matmuls plus a nonlinear rounding. Each `(position, head)` is one committed
//! matmul + one committed lookup (the same primitives as the FFN).
//!
//!   scores:     dot = K_h · q[l,h,:],   scores = round(dot / scale)
//!   weighted_v: acc = V_h^T · w[l,h,:], out    = requant(acc, 15)

use crate::committed::{prove_matmul_committed, verify_matmul_committed, MatmulProof};
use crate::field::{Ext, Fp};
use crate::fri::{self, Challenger, Prover};
use crate::lookup_committed::{prove_lookup2_committed, verify_lookup2_committed, Lookup2Proof};
use crate::zk::Openings;

fn f(x: i64) -> Fp {
    if x < 0 {
        -Fp::new(x.unsigned_abs())
    } else {
        Fp::new(x as u64)
    }
}

/// `scores[t] = round(Σ_j k[t,j]·q[j] / scale)` for one `(position, head)`.
pub struct ScoresProof {
    pub l: usize,
    pub d: usize,
    pub mm: MatmulProof,
    pub round: Lookup2Proof,
    pub round_lo: i64,
    pub scale: f64,
}

pub fn prove_scores_committed(
    ctx: &mut Prover,
    k: &[i64],
    q: &[i64],
    scores: &[i64],
    d: usize,
    scale: f64,
    round_lo: i64,
    alpha: Fp,
    beta: Fp,
) -> ScoresProof {
    let l = scores.len();
    let k_fp: Vec<Fp> = k.iter().map(|&x| f(x)).collect();
    let q_fp: Vec<Fp> = q.iter().map(|&x| f(x)).collect();
    let dot: Vec<i64> = (0..l)
        .map(|t| (0..d).map(|j| k[t * d + j] * q[j]).sum())
        .collect();
    let dot_fp: Vec<Fp> = dot.iter().map(|&x| f(x)).collect();

    let mm = prove_matmul_committed(ctx, &k_fp, &q_fp, &dot_fp, l, d);
    let round = prove_lookup2_committed(ctx, &dot, scores, round_lo, alpha, beta);
    ScoresProof {
        l,
        d,
        mm,
        round,
        round_lo,
        scale,
    }
}

pub fn verify_scores_committed(
    challenger: &Challenger,
    proof: &ScoresProof,
    values: &[Ext],
    acc: &mut Openings,
) -> bool {
    verify_matmul_committed(challenger, proof.l, proof.d, &proof.mm, values, acc)
        && verify_lookup2_committed(
            challenger,
            |x| crate::spec::round_div(x, proof.scale),
            proof.round_lo,
            proof.round.alpha,
            proof.round.beta,
            &proof.round,
            values,
            acc,
        )
}

/// `out[j] = requant(Σ_t v[t,j]·w[t], 15)` for one `(position, head)`.
pub struct WeightedVProof {
    pub d: usize,
    pub l: usize,
    pub mm: MatmulProof,
    pub req: Lookup2Proof,
    pub req_lo: i64,
}

pub fn prove_weighted_v_committed(
    ctx: &mut Prover,
    v: &[i64],
    w: &[i64],
    out: &[i64],
    req_lo: i64,
    alpha: Fp,
    beta: Fp,
) -> WeightedVProof {
    let l = w.len();
    let d = out.len();
    // V^T[j][t] = v[t][j], so vh[j*l + t] = v[t*d + j].
    let mut vh = vec![0i64; d * l];
    for j in 0..d {
        for t in 0..l {
            vh[j * l + t] = v[t * d + j];
        }
    }
    let vh_fp: Vec<Fp> = vh.iter().map(|&x| f(x)).collect();
    let w_fp: Vec<Fp> = w.iter().map(|&x| f(x)).collect();
    let acc: Vec<i64> = (0..d)
        .map(|j| (0..l).map(|t| vh[j * l + t] * w[t]).sum())
        .collect();
    let acc_fp: Vec<Fp> = acc.iter().map(|&x| f(x)).collect();

    let mm = prove_matmul_committed(ctx, &vh_fp, &w_fp, &acc_fp, d, l);
    let req = prove_lookup2_committed(ctx, &acc, out, req_lo, alpha, beta);
    WeightedVProof {
        d,
        l,
        mm,
        req,
        req_lo,
    }
}

pub fn verify_weighted_v_committed(
    challenger: &Challenger,
    proof: &WeightedVProof,
    values: &[Ext],
    acc: &mut Openings,
) -> bool {
    verify_matmul_committed(challenger, proof.d, proof.l, &proof.mm, values, acc)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn f(x: u64) -> Fp {
        Fp::new(x)
    }

    #[test]
    fn scores_committed_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        // k = [[1,0],[0,1]], q = [3,4], d=2 → dot = [3,4], scale=1 → scores=[3,4].
        let k = [1i64, 0, 0, 1];
        let q = [3i64, 4];
        let scores = [3i64, 4];
        let proof = prove_scores_committed(&mut ctx, &k, &q, &scores, 2, 1.0, -64, f(11), f(5));
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = crate::zk::Openings::new();
        assert!(verify_scores_committed(
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
    fn weighted_v_committed_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        // v = [[1,2],[3,4]] (L=2, d=2), w = [5,6].
        // out[0] = requant(1·5+3·6, 15) = requant(23,15) = 0; out[1] = requant(34,15) = 0.
        let v = [1i64, 2, 3, 4];
        let w = [5i64, 6];
        let out = [0i64, 0];
        let proof = prove_weighted_v_committed(&mut ctx, &v, &w, &out, -128, f(11), f(5));
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = crate::zk::Openings::new();
        assert!(verify_weighted_v_committed(
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
