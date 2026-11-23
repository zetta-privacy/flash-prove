//! `top_k` expert selection on the stacked-WHIR layer.
//!
//! Per row, with `e` experts:
//!
//! ```text
//!   δ[i]    = e·score[i] − i − τ                one global linear claim each
//!   choice  = sign(δ)                           claim_lookup2 against the sign table
//!   Σ_i choice[i] = k                           one linear claim per row
//! ```
//!
//! `δ` and the count are the only things that need `δ` to be a committed vector;
//! `choice` is already a trace cell, so the per-row counts are *view* claims on it
//! and the sign lookup runs once over the whole cell instead of once per row.
//!
//! # Known gap (inherited, not introduced)
//!
//! `τ` comes from the prover. "Exactly `k` entries exceed `τ`" does not force the
//! selected set to be the `k` largest — only a valid `k`-subset. The FRI path has
//! the same gap ([`crate::lookup::selection_committed::verify_topk_committed`]
//! takes `proof.tau` unchecked). Closing it needs an extra obligation (every
//! unselected score ≤ every selected one), which is a change to the
//! arithmetization, not to the commitment layer.

use crate::field::{Ext, Fp};
use crate::gkr::logup::{claim_lookup2, verify_lookup2, Lookup2Cells, Lookup2SegmentProof};
use crate::gkr::segment::{
    claim_value_at_id, linear_finish, linear_verify, CellClaim, CellSegment, CellSegmentCommitment,
    View,
};
use crate::pcs::fri::Transcript;
use crate::pcs::linear_claim::{LinearClaimProof, LinearH};
use p3_field::PrimeCharacteristicRing;

/// One random-point linear claim value.
pub type Lin = (Ext, LinearClaimProof);

/// A `top_k` selection claim.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TopKSegmentProof {
    pub rows: usize,
    pub e: usize,
    pub k: usize,
    pub tau: i64,
    /// The sign lookup, with `input = δ` and `output = choice`.
    pub cells: Lookup2Cells,
    /// The score cell the `δ` relation reads.
    pub scores: usize,
    pub alpha: Fp,
    pub beta: Fp,
    pub r: Vec<Fp>,
    pub b_delta: Lin,
    pub b_scores: Lin,
    /// `⟨1, choice_row⟩` per row.
    pub counts: Vec<Lin>,
    pub lookup: Lookup2SegmentProof,
}

/// Claim the selection for every row.
#[allow(clippy::too_many_arguments)]
pub fn claim_top_k(
    seg: &mut CellSegment,
    cells: Lookup2Cells,
    scores: usize,
    (rows, e, k, tau): (usize, usize, usize, i64),
    (alpha, beta): (Fp, Fp),
    (sign_table, big_n): (&[Fp], usize),
    tr: &mut Transcript,
) -> TopKSegmentProof {
    let n = rows * e;
    for id in [scores, cells.input, cells.output, cells.fp] {
        tr.observe_val(Fp::new(id as u64));
    }

    // `δ = e·scores − idx − τ` at one global point.
    let r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    let h = LinearH::Dense(r.clone());
    let b_delta = seg.claim_linear_id(cells.input, 0, &h, tr);
    let b_scores = seg.claim_linear_id(scores, 0, &h, tr);

    // `Σ_i choice[i] = k` for every row, as a view claim on the shared cell.
    let m_choice = seg.cells_meta(cells.output).m();
    let ones = LinearH::Dense(vec![Fp::ONE; e]);
    let mut counts = Vec::with_capacity(rows);
    for b in 0..rows {
        let view = View::block(m_choice, b * e, e.trailing_zeros() as usize);
        counts.push(seg.claim_linear_view(cells.output, &view, &ones, tr));
    }

    // `choice = sign(δ)` over the whole cell.
    let lookup = claim_lookup2(
        seg,
        cells.clone(),
        (alpha, beta),
        sign_table,
        (n, big_n),
        tr,
    );

    TopKSegmentProof {
        rows,
        e,
        k,
        tau,
        cells,
        scores,
        alpha,
        beta,
        r,
        b_delta,
        b_scores,
        counts,
        lookup,
    }
}

/// Verify a [`TopKSegmentProof`].
#[allow(clippy::too_many_arguments)]
pub fn verify_top_k(
    commitment: &CellSegmentCommitment,
    claims: &[CellClaim],
    proof: &TopKSegmentProof,
    (rows, e, k): (usize, usize, usize),
    (sign_table, big_n): (&[Fp], usize),
    tr: &mut Transcript,
) -> bool {
    let cells = proof.cells.clone();
    if proof.rows != rows || proof.e != e || proof.k != k || proof.counts.len() != rows {
        return false;
    }
    let n = rows * e;
    let Some(m_choice) = commitment.cells.get(cells.output).map(|c| c.m()) else {
        return false;
    };
    for id in [proof.scores, cells.input, cells.output, cells.fp] {
        tr.observe_val(Fp::new(id as u64));
    }

    let r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    if r != proof.r {
        return false;
    }
    let h = LinearH::Dense(r.clone());

    // `δ = e·scores − idx − τ`.
    let Some((z, cf)) = linear_verify(&proof.b_delta.1, proof.b_delta.0, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, cells.input, &z) else {
        return false;
    };
    if !linear_finish(&h, &z, cf, v) {
        return false;
    }
    let Some((z, cf)) = linear_verify(&proof.b_scores.1, proof.b_scores.0, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, proof.scores, &z) else {
        return false;
    };
    if !linear_finish(&h, &z, cf, v) {
        return false;
    }

    // ⟨r, idx⟩ with `idx[i] = i mod e`, and `Σ r` for the constant τ.
    let mut ip_idx = Ext::ZERO;
    let mut sum_r = Ext::ZERO;
    for (i, &ri) in r.iter().enumerate() {
        let re = Ext::from(ri);
        ip_idx += re * Ext::from(Fp::new((i % e) as u64));
        sum_r += re;
    }
    let tau = Ext::from(if proof.tau < 0 {
        -Fp::new(proof.tau.unsigned_abs())
    } else {
        Fp::new(proof.tau as u64)
    });
    if proof.b_delta.0 != Ext::from(Fp::new(e as u64)) * proof.b_scores.0 - ip_idx - tau * sum_r {
        return false;
    }

    // Every row selects exactly `k`.
    let ones = LinearH::Dense(vec![Fp::ONE; e]);
    for (b, (bval, ref cl)) in proof.counts.iter().enumerate() {
        let view = View::block(m_choice, b * e, e.trailing_zeros() as usize);
        let Some((z, cf)) = linear_verify(cl, *bval, tr) else {
            return false;
        };
        let Some(v) =
            claim_value_at_id(commitment, claims, cells.output, &view.point(&z, m_choice))
        else {
            return false;
        };
        if !linear_finish(&ones, &z, cf, v) {
            return false;
        }
        if *bval != Ext::from(Fp::new(k as u64)) {
            return false;
        }
    }

    verify_lookup2(
        commitment,
        claims,
        &proof.lookup,
        sign_table,
        (n, big_n),
        tr,
    )
}

/// The public sign table fingerprints `α + β·x + β²·sign(x)` over `[lo, hi]`.
pub fn sign_table(lo: i64, hi: i64, alpha: Fp, beta: Fp) -> Vec<Fp> {
    let n = (hi - lo + 1) as usize;
    let mut v: Vec<Fp> = (lo..=hi)
        .map(|x| {
            let xs = if x < 0 {
                -Fp::new(x.unsigned_abs())
            } else {
                Fp::new(x as u64)
            };
            let s = Fp::new(if x >= 0 { 1 } else { 0 });
            alpha + beta * xs + beta * beta * s
        })
        .collect();
    v.resize(n.next_power_of_two(), Fp::ONE);
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gkr::segment::{verify, SegmentCells};
    use crate::pcs::fri::new_challenger;
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
    fn top_k_per_row() {
        let (rows, e, k, tau) = (2usize, 2usize, 1usize, 2i64);
        let (lo, hi) = (-2i64, 8i64);
        let big_n = (hi - lo + 1) as usize;
        let alpha = Fp::new(11);
        let beta = Fp::new(5);
        let table = sign_table(lo, hi, alpha, beta);
        assert_eq!(table.len(), big_n.next_power_of_two());

        let scores = vec![5i64, 1, 0, 3];
        let delta: Vec<i64> = (0..rows * e)
            .map(|i| e as i64 * scores[i] - (i % e) as i64 - tau)
            .collect();
        let choice: Vec<i64> = delta.iter().map(|&d| if d >= 0 { 1 } else { 0 }).collect();

        let n = rows * e;
        let d = n.next_power_of_two();
        let fpv: Vec<Fp> = (0..n)
            .map(|i| alpha + beta * fp(delta[i]) + beta * beta * fp(choice[i]))
            .collect();

        // Multiplicities over the table range, padding `1`.
        let mut counts = vec![0usize; big_n];
        for &v in &fpv {
            let idx = (0..big_n)
                .find(|&j| {
                    let x = lo + j as i64;
                    let s = if x >= 0 { 1 } else { 0 };
                    alpha + beta * fp(x) + beta * beta * fp(s) == v
                })
                .expect("fingerprint in table");
            counts[idx] += 1;
        }
        let ft_d = big_n.next_power_of_two();
        let mut m: Vec<Fp> = counts.iter().map(|&c| Fp::new(c as u64)).collect();
        m.resize(ft_d, Fp::ONE);
        let mut diff_t: Vec<Fp> = counts
            .iter()
            .zip(table.iter())
            .map(|(&c, &t)| Fp::new(c as u64) * t.inverse())
            .collect();
        diff_t.resize(ft_d, Fp::ONE);
        let diff_q: Vec<Fp> = fpv.iter().map(|&x| x.inverse()).collect();

        let mut cells = SegmentCells::new();
        let col = |c: &mut SegmentCells, name: &str, v: &[Fp], rows: usize| {
            let mut x = v.to_vec();
            x.resize(rows.next_power_of_two(), Fp::ZERO);
            c.cell_fp(name, rows, 1, x);
        };
        col(
            &mut cells,
            "scores",
            &scores.iter().map(|&x| fp(x)).collect::<Vec<_>>(),
            n,
        );
        col(
            &mut cells,
            "choice",
            &choice.iter().map(|&x| fp(x)).collect::<Vec<_>>(),
            n,
        );
        col(
            &mut cells,
            "delta",
            &delta.iter().map(|&x| fp(x)).collect::<Vec<_>>(),
            n,
        );
        col(&mut cells, "fp", &fpv, n);
        col(&mut cells, "diff_q", &diff_q, n);
        cells.cell_fp("ft", ft_d, 1, table.clone());
        cells.cell_fp("diff_t", ft_d, 1, diff_t);
        cells.cell_fp("m", ft_d, 1, m);
        let _ = d;

        let mut seg = cells.seal(Params::default());
        let com = seg.commitment();
        let cells2 = Lookup2Cells {
            input: seg.id("delta"),
            output: seg.id("choice"),
            fp: seg.id("fp"),
            diff_q: seg.id("diff_q"),
            ft: seg.id("ft"),
            diff_t: seg.id("diff_t"),
            m: seg.id("m"),
        };
        let scores_id = seg.id("scores");
        let mut tr = Transcript::new(&new_challenger());
        let proof = claim_top_k(
            &mut seg,
            cells2,
            scores_id,
            (rows, e, k, tau),
            (alpha, beta),
            (&table, big_n),
            &mut tr,
        );
        let claims = seg.claims().to_vec();
        let wproof = seg.prove(&mut new_challenger());
        assert!(verify(&com, &claims, &wproof, &mut new_challenger()));
        let mut vtr = Transcript::new(&new_challenger());
        assert!(verify_top_k(
            &com,
            &claims,
            &proof,
            (rows, e, k),
            (&table, big_n),
            &mut vtr
        ));

        // A wrong threshold breaks `δ = e·scores − idx − τ`.
        let mut bad = TopKSegmentProof {
            tau: tau + 1,
            ..proof.clone()
        };
        let mut vtr = Transcript::new(&new_challenger());
        assert!(!verify_top_k(
            &com,
            &claims,
            &bad,
            (rows, e, k),
            (&table, big_n),
            &mut vtr
        ));
        bad.tau = tau;
        // A wrong delta cell breaks the lookup's fingerprint binding.
        let mut bad_claims = claims.clone();
        let di = bad_claims
            .iter()
            .position(|c| c.cell == cells2.input)
            .unwrap();
        bad_claims[di].values[0] += Ext::ONE;
        let mut vtr = Transcript::new(&new_challenger());
        assert!(!verify_top_k(
            &com,
            &bad_claims,
            &bad,
            (rows, e, k),
            (&table, big_n),
            &mut vtr
        ));
    }
}
