//! Head argmax on the stacked-WHIR layer.
//!
//! The head token must index a maximum of the last logits row:
//!
//! ```text
//!   ones[j] = sign(head_max − scores[j])   claim_lookup2 against the sign table
//!   Σ_j ones[j] = cols                     one linear claim (no score escaped)
//!   scores[head_token] = head_max          one point evaluation at a Boolean point
//!   scores = logits' last row              two views at one point
//! ```
//!
//! `head_max` comes from the prover and is fixed by the point evaluation, so the
//! sign table it induces is public. Together the three say: every logit is at most
//! `head_max`, and `head_token` *is* `head_max` — i.e. the token is **a** maximum.
//! (Not necessarily the first one; the arithmetization never claimed that.)

use crate::field::arith::bool_point;
use crate::field::{Ext, Fp};
use crate::gkr::logup::{claim_lookup2, verify_lookup2, Lookup2Cells, Lookup2SegmentProof};
use crate::gkr::segment::{
    claim_value_at_id, linear_finish, linear_verify, CellClaim, CellSegment, CellSegmentCommitment,
    View,
};
use crate::pcs::transcript::Transcript;
use crate::pcs::linear_claim::{LinearClaimProof, LinearH};
use p3_field::PrimeCharacteristicRing;

/// One random-point linear claim value.
pub type Lin = (Ext, LinearClaimProof);

/// A head-argmax claim.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HeadSegmentProof {
    /// The shared logits cell and the row being checked.
    pub logits: usize,
    pub row: usize,
    pub n: usize,
    pub head_token: usize,
    pub head_max: i64,
    /// The lookup: `input = scores` (the row), `output = ones`.
    pub cells: Lookup2Cells,
    /// The per-row score cell.
    pub scores: usize,
    pub ones: usize,
    pub r: Vec<Fp>,
    pub b_scores: Lin,
    pub b_logits: Lin,
    pub b_count: Lin,
    /// `scores(head_token)`.
    pub tok_val: Ext,
    pub lookup: Lookup2SegmentProof,
}

/// The public sign table `α + β·x + β²·sign(max − x)` over `[lo, hi]`, padded
/// with `1`.
pub fn head_table(lo: i64, hi: i64, head_max: i64, alpha: Fp, beta: Fp) -> Vec<Fp> {
    let n = (hi - lo + 1) as usize;
    let mut v: Vec<Fp> = (lo..=hi)
        .map(|x| {
            let xs = if x < 0 {
                -Fp::new(x.unsigned_abs())
            } else {
                Fp::new(x as u64)
            };
            let s = Fp::new(if head_max - x >= 0 { 1 } else { 0 });
            alpha + beta * xs + beta * beta * s
        })
        .collect();
    v.resize(n.next_power_of_two(), Fp::ONE);
    v
}

/// Claim that `head_token` is a maximum of the logits row.
#[allow(clippy::too_many_arguments)]
pub fn claim_head(
    seg: &mut CellSegment,
    cells: Lookup2Cells,
    ones: usize,
    (logits, row, n, head_token, head_max): (usize, usize, usize, usize, i64),
    (alpha, beta): (Fp, Fp),
    (table, big_n): (&[Fp], usize),
    tr: &mut Transcript,
) -> HeadSegmentProof {
    for id in [logits, cells.input, cells.output, ones] {
        tr.observe_val(Fp::new(id as u64));
    }

    // The row tie: two views at one point.
    let r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    let h = LinearH::Dense(r.clone());
    let b_scores = seg.claim_linear_id(cells.input, 0, &h, tr);
    let m_logits = seg.cells_meta(logits).m();
    let b_logits = seg.claim_linear_view(
        logits,
        &View::block(m_logits, row * n, n.trailing_zeros() as usize),
        &h,
        tr,
    );

    // `Σ_j ones[j] = n`.
    let b_count = seg.claim_linear_id(ones, 0, &LinearH::Ones { len: n }, tr);

    // `scores[head_token]` is a plain point evaluation at a Boolean point.
    let tok_point = bool_point(head_token, n.trailing_zeros() as usize)
        .into_iter()
        .map(|x| Ext::from(x))
        .collect::<Vec<_>>();
    let tok_val = seg.claim_id(cells.input, &[0], &tok_point)[0];
    assert_eq!(
        tok_val,
        Ext::from(if head_max < 0 {
            -Fp::new(head_max.unsigned_abs())
        } else {
            Fp::new(head_max as u64)
        }),
        "head_max must be the score at the bound token"
    );

    let lookup = claim_lookup2(seg, cells.clone(), (alpha, beta), table, (n, big_n), tr);

    HeadSegmentProof {
        logits,
        row,
        n,
        head_token,
        head_max,
        cells,
        scores: cells.input,
        ones,
        r,
        b_scores,
        b_logits,
        b_count,
        tok_val,
        lookup,
    }
}

/// Verify a [`HeadSegmentProof`].
#[allow(clippy::too_many_arguments)]
pub fn verify_head(
    commitment: &CellSegmentCommitment,
    claims: &[CellClaim],
    proof: &HeadSegmentProof,
    (logits, row, n, head_token): (usize, usize, usize, usize),
    (table, big_n): (&[Fp], usize),
    tr: &mut Transcript,
) -> bool {
    if proof.logits != logits || proof.row != row || proof.n != n || proof.head_token != head_token
    {
        return false;
    }
    let cells = proof.cells.clone();
    let Some(m_logits) = commitment.cells.get(logits).map(|c| c.m()) else {
        return false;
    };
    for id in [logits, cells.input, cells.output, proof.ones] {
        tr.observe_val(Fp::new(id as u64));
    }

    let r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    if r != proof.r {
        return false;
    }
    let h = LinearH::Dense(r.clone());
    let bits = n.trailing_zeros() as usize;

    let Some((z, cf)) = linear_verify(&proof.b_scores.1, proof.b_scores.0, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, cells.input, &z) else {
        return false;
    };
    if !linear_finish(&h, &z, cf, v) {
        return false;
    }
    let blk = View::block(m_logits, row * n, bits);
    let Some((z, cf)) = linear_verify(&proof.b_logits.1, proof.b_logits.0, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, logits, &blk.point(&z, m_logits)) else {
        return false;
    };
    if !linear_finish(&h, &z, cf, v) {
        return false;
    }
    if proof.b_scores.0 != proof.b_logits.0 {
        return false;
    }

    let Some((z, cf)) = linear_verify(&proof.b_count.1, proof.b_count.0, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, proof.ones, &z) else {
        return false;
    };
    if !linear_finish(&LinearH::Ones { len: n }, &z, cf, v) {
        return false;
    }
    if proof.b_count.0 != Ext::from(Fp::new(n as u64)) {
        return false;
    }

    // `scores(head_token) = head_max`, which fixes the table the verifier rebuilt.
    let tok_point = bool_point(head_token, bits)
        .into_iter()
        .map(|x| Ext::from(x))
        .collect::<Vec<_>>();
    let expected = Ext::from(if proof.head_max < 0 {
        -Fp::new(proof.head_max.unsigned_abs())
    } else {
        Fp::new(proof.head_max as u64)
    });
    if claim_value_at_id(commitment, claims, cells.input, &tok_point) != Some(expected) {
        return false;
    }

    verify_lookup2(commitment, claims, &proof.lookup, table, (n, big_n), tr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gkr::segment::{verify, SegmentCells};
    use crate::pcs::transcript::new_challenger;
    use crate::pcs::whir_stacked::Params;
    use p3_field::Field;

    fn fp(x: i64) -> Fp {
        if x < 0 {
            -Fp::new(x.unsigned_abs())
        } else {
            Fp::new(x as u64)
        }
    }

    #[test]
    fn head_token_is_a_maximum() {
        let (rows, n, row, token) = (2usize, 4usize, 1usize, 1usize);
        let alpha = Fp::new(11);
        let beta = Fp::new(5);
        let mut logits_vals = vec![0i64; rows * n];
        let scores = vec![5i64, 9, 3, 7];
        logits_vals[row * n..(row + 1) * n].copy_from_slice(&scores);
        let head_max = scores[token];
        let ones = vec![1i64; n];

        let (lo, hi) = (*scores.iter().min().unwrap(), *scores.iter().max().unwrap());
        let big_n = (hi - lo + 1) as usize;
        let table = head_table(lo, hi, head_max, alpha, beta);
        let ft_d = big_n.next_power_of_two();
        assert_eq!(table.len(), ft_d);

        let fp_of = |x: i64| {
            alpha + beta * fp(x) + beta * beta * fp(if head_max - x >= 0 { 1 } else { 0 })
        };
        let fpv: Vec<Fp> = scores.iter().map(|&x| fp_of(x)).collect();
        let mut counts = vec![0usize; big_n];
        for &x in &scores {
            counts[(x - lo) as usize] += 1;
        }
        let mut m: Vec<Fp> = counts.iter().map(|&c| Fp::new(c as u64)).collect();
        m.resize(ft_d, Fp::ONE);
        let mut diff_t: Vec<Fp> = counts
            .iter()
            .zip(table.iter().take(big_n))
            .map(|(&c, &t)| Fp::new(c as u64) * t.inverse())
            .collect();
        diff_t.resize(ft_d, Fp::ONE);
        let diff_q: Vec<Fp> = fpv.iter().map(|&v| v.inverse()).collect();

        let mut cells = SegmentCells::new();
        cells.cell_fp("logits", rows * n, 1, logits_vals.iter().map(|&z| fp(z)).collect());
        cells.cell_fp("scores", n, 1, scores.iter().map(|&z| fp(z)).collect());
        cells.cell_fp("ones", n, 1, ones.iter().map(|&z| fp(z)).collect());
        cells.cell_fp("fp", n, 1, fpv.clone());
        cells.cell_fp("diff_q", n, 1, diff_q);
        cells.cell_fp("ft", ft_d, 1, table.clone());
        cells.cell_fp("diff_t", ft_d, 1, diff_t);
        cells.cell_fp("m", ft_d, 1, m);

        let mut seg = cells.seal(Params::default());
        let com = seg.commitment();
        let lc = Lookup2Cells {
            input: seg.id("scores"),
            output: seg.id("ones"),
            fp: seg.id("fp"),
            diff_q: seg.id("diff_q"),
            ft: seg.id("ft"),
            diff_t: seg.id("diff_t"),
            m: seg.id("m"),
        };
        let (lg, on) = (seg.id("logits"), seg.id("ones"));
        let mut tr = Transcript::new(&new_challenger());
        let proof = claim_head(
            &mut seg,
            lc,
            on,
            (lg, row, n, token, head_max),
            (alpha, beta),
            (&table, big_n),
            &mut tr,
        );
        let claims = seg.claims().to_vec();
        let wproof = seg.prove(&mut new_challenger());
        assert!(verify(&com, &claims, &wproof, &mut new_challenger()));
        let mut vtr = Transcript::new(&new_challenger());
        assert!(verify_head(
            &com,
            &claims,
            &proof,
            (lg, row, n, token),
            (&table, big_n),
            &mut vtr
        ));

        // A token whose score is not the maximum is rejected: the point value is
        // not `head_max`.
        let mut vtr = Transcript::new(&new_challenger());
        let mut bad = HeadSegmentProof {
            head_token: 0,
            ..proof.clone()
        };
        bad.head_token = 0;
        assert!(!verify_head(
            &com,
            &claims,
            &bad,
            (lg, row, n, 0),
            (&table, big_n),
            &mut vtr
        ));
    }
}
