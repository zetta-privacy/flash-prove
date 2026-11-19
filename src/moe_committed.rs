//! Committed MoE `weighted_sum`: `out[j] = Σ_e choice[e]·term_e[j]`, a matmul over
//! the expert dimension (the same committed matmul primitive as the FFN).

use crate::committed::{prove_matmul_committed, verify_matmul_committed, MatmulProof};
use crate::field::{Ext, Fp};
use crate::fri::{Challenger, Prover};
use crate::zk::Openings;

fn f(x: i64) -> Fp {
    if x < 0 {
        -Fp::new(x.unsigned_abs())
    } else {
        Fp::new(x as u64)
    }
}

/// Committed `out[j] = Σ_e choice[e]·term_e[j]`. `terms` is `E×hidden` row-major,
/// `choice` is `E` (0/1 indicator), `out` is `hidden`.
pub struct WeightedSumProof {
    pub experts: usize,
    pub hidden: usize,
    pub mm: MatmulProof,
}

pub fn prove_weighted_sum_committed(
    ctx: &mut Prover,
    terms: &[i64],
    choice: &[i64],
    out: &[i64],
    experts: usize,
    hidden: usize,
) -> WeightedSumProof {
    // The matmul is `out[j] = Σ_e choice[e]·terms[e,j]`, so the weight is the
    // transpose `W[j][e] = terms[e·hidden + j]` (a `hidden × experts` matrix).
    let mut w = vec![0i64; hidden * experts];
    for j in 0..hidden {
        for e in 0..experts {
            w[j * experts + e] = terms[e * hidden + j];
        }
    }
    let w_fp: Vec<Fp> = w.iter().map(|&x| f(x)).collect();
    let choice_fp: Vec<Fp> = choice.iter().map(|&x| f(x)).collect();
    let out_fp: Vec<Fp> = out.iter().map(|&x| f(x)).collect();
    let mm = prove_matmul_committed(ctx, &w_fp, &choice_fp, &out_fp, hidden, experts);
    WeightedSumProof {
        experts,
        hidden,
        mm,
    }
}

pub fn verify_weighted_sum_committed(
    challenger: &Challenger,
    proof: &WeightedSumProof,
    values: &[Ext],
    acc: &mut Openings,
) -> bool {
    verify_matmul_committed(
        challenger,
        proof.hidden,
        proof.experts,
        &proof.mm,
        values,
        acc,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fri;

    fn f(x: u64) -> Fp {
        Fp::new(x)
    }

    #[test]
    fn weighted_sum_committed_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        // E=2, hidden=2. terms = [[1,2],[3,4]], choice = [1,1].
        // out[0] = 1·1 + 1·3 = 4; out[1] = 1·2 + 1·4 = 6.
        let terms = [1i64, 2, 3, 4];
        let choice = [1i64, 1];
        let out = [4i64, 6];
        let proof = prove_weighted_sum_committed(&mut ctx, &terms, &choice, &out, 2, 2);
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = crate::zk::Openings::new();
        assert!(verify_weighted_sum_committed(
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
