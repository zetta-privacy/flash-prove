//! `top_k` expert selection on the stacked-WHIR layer, **per row**.
//!
//! With `adj[i] = e·score[i] − i`, one row proves
//!
//! ```text
//!   δ[i]    = adj[i] − τ               row claims on δ, adj and scores
//!   choice  = sign(δ)                 one global claim_lookup2 over the whole cell
//!   Σ_i choice[i] = k                row count
//!   adj[t*] = τ  ∧  choice[t*] = 1    two point claims
//! ```
//!
//! The last pair is what makes `τ` the **k-th largest**: exactly `k` entries are ≥ `τ`
//! *and* one of them equals `τ`. Without it a prover moves `τ` to select any `k`-subset.
//! `τ` is per row — a single threshold for a layer cannot express a routing decision.

use crate::field::{Ext, Fp};
use crate::gkr::logup::Lookup2Cells;
use crate::gkr::segment::{
    claim_value_at_id, linear_finish, linear_verify, CellSegment, CellSegmentCommitment,
    View,
};
use crate::pcs::transcript::Transcript;
use crate::pcs::linear_claim::{LinearClaimProof, LinearH};
use p3_field::PrimeCharacteristicRing;

/// One random-point linear claim value.
pub type Lin = (Ext, LinearClaimProof);

/// The cells a `top_k` row reads. `adj`, `delta` and the lookup span the whole cell;
/// the row is addressed by [`View::block`].
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TopKCells {
    pub scores: usize,
    pub choice: usize,
    pub adj: usize,
    pub delta: usize,
    /// The global `(δ, choice)` sign lookup, claimed once after all rows.
    pub lookup: Lookup2Cells,
}

/// One row's selection claim.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TopKRowProof {
    pub row: usize,
    pub e: usize,
    pub k: usize,
    pub tau: i64,
    /// An index achieving the threshold, inside the row.
    pub tstar: usize,
    pub r: Vec<Fp>,
    pub b_scores: Lin,
    pub b_adj: Lin,
    pub b_delta: Lin,
    pub count: Lin,
    /// `adj[row·e + t*]`, which must equal `τ`.
    pub adj_at_tstar: Ext,
    /// `choice[row·e + t*]`, which must be `1`.
    pub choice_at_tstar: Ext,
}

/// Claim one row's selection. The global lookup is claimed separately, after the rows.
#[allow(clippy::too_many_arguments)]
pub fn claim_top_k_row(
    seg: &mut CellSegment,
    cells: &TopKCells,
    (row, e, k, tau, tstar): (usize, usize, usize, i64, usize),
    tr: &mut Transcript,
) -> TopKRowProof {
    assert!(tstar < e, "the witness index must lie inside the row");
    let log_e = e.trailing_zeros() as usize;
    for id in [cells.scores, cells.adj, cells.delta, cells.choice] {
        tr.observe_val(Fp::new(id as u64));
    }

    let (m_scores, m_adj, m_delta, m_choice) = (
        seg.cells_meta(cells.scores).m(),
        seg.cells_meta(cells.adj).m(),
        seg.cells_meta(cells.delta).m(),
        seg.cells_meta(cells.choice).m(),
    );
    let (r, coins) = crate::pcs::linear_claim::sample_coins(e, tr);
    let h = LinearH::Dense(r.clone());
    let tau_ext = Ext::from(fp_i64(tau));
    let mut sum_r = Ext::from(Fp::new(0));
    let mut ip_idx = Ext::from(Fp::new(0));
    for (i, &ri) in r.iter().enumerate() {
        let re = Ext::from(ri);
        sum_r += re;
        ip_idx += re * Ext::from(Fp::new(i as u64));
    }

    let b_scores = seg.claim_linear_view(
        cells.scores,
        &View::block(m_scores, row * e, log_e),
        &h,
        tr,
    );
    let b_adj = seg.claim_linear_view(cells.adj, &View::block(m_adj, row * e, log_e), &h, tr);
    assert_eq!(
        b_adj.0,
        Ext::from(Fp::new(e as u64)) * b_scores.0 - ip_idx,
        "adj = e·scores − idx"
    );
    let b_delta = seg.claim_linear_view(cells.delta, &View::block(m_delta, row * e, log_e), &h, tr);
    assert_eq!(b_delta.0, b_adj.0 - tau_ext * sum_r, "delta = adj − tau");

    let count = seg.claim_linear_view(
        cells.choice,
        &View::block(m_choice, row * e, log_e),
        &LinearH::Ones { len: e },
        tr,
    );

    let inner = crate::field::arith::bool_point(tstar, log_e);
    let adj_at_tstar = seg.claim_id(
        cells.adj,
        &[0],
        &View::block(m_adj, row * e, log_e).point(&inner, m_adj),
    )[0];
    let choice_at_tstar = seg.claim_id(
        cells.choice,
        &[0],
        &View::block(m_choice, row * e, log_e).point(&inner, m_choice),
    )[0];

    TopKRowProof {
        row,
        e,
        k,
        tau,
        tstar,
        r: coins,
        b_scores,
        b_adj,
        b_delta,
        count,
        adj_at_tstar,
        choice_at_tstar,
    }
}

/// Verify a [`TopKRowProof`].
pub fn verify_top_k_row(
    commitment: &CellSegmentCommitment,
    claims: &crate::gkr::segment::Claims<'_>,
    cells: &TopKCells,
    proof: &TopKRowProof,
    (row, e, k): (usize, usize, usize),
    tr: &mut Transcript,
) -> bool {
    if proof.row != row || proof.e != e || proof.k != k || proof.tstar >= e {
        return false;
    }
    let (Some(m_scores), Some(m_adj), Some(m_delta), Some(m_choice)) = (
        commitment.cells.get(cells.scores).map(|c| c.m()),
        commitment.cells.get(cells.adj).map(|c| c.m()),
        commitment.cells.get(cells.delta).map(|c| c.m()),
        commitment.cells.get(cells.choice).map(|c| c.m()),
    ) else {
        return false;
    };
    let log_e = e.trailing_zeros() as usize;
    for id in [cells.scores, cells.adj, cells.delta, cells.choice] {
        tr.observe_val(Fp::new(id as u64));
    }

    let (r, coins) = crate::pcs::linear_claim::sample_coins(e, tr);
    if coins != proof.r {
        return false;
    }
    let h = LinearH::Dense(r.clone());
    let tau_ext = Ext::from(fp_i64(proof.tau));
    let mut sum_r = Ext::from(Fp::new(0));
    let mut ip_idx = Ext::from(Fp::new(0));
    for (i, &ri) in r.iter().enumerate() {
        let re = Ext::from(ri);
        sum_r += re;
        ip_idx += re * Ext::from(Fp::new(i as u64));
    }

    // Replay `scores`, `adj`, `delta`, `count` in the prover's order.
    for (cell, m, lin, hh) in [
        (cells.scores, m_scores, &proof.b_scores, h.clone()),
        (cells.adj, m_adj, &proof.b_adj, h.clone()),
        (cells.delta, m_delta, &proof.b_delta, h.clone()),
        (
            cells.choice,
            m_choice,
            &proof.count,
            LinearH::Ones { len: e },
        ),
    ] {
        let Some((z, cf)) = linear_verify(&lin.1, lin.0, tr) else {
            return false;
        };
        let pt = View::block(m, row * e, log_e).point(&z, m);
        let Some(v) = claim_value_at_id(commitment, claims, cell, &pt) else {
            return false;
        };
        if !linear_finish(&hh, &z, cf, v) {
            return false;
        }
    }

    if proof.b_adj.0 != Ext::from(Fp::new(e as u64)) * proof.b_scores.0 - ip_idx {
        return false;
    }
    if proof.b_delta.0 != proof.b_adj.0 - tau_ext * sum_r {
        return false;
    }
    if proof.count.0 != Ext::from(Fp::new(k as u64)) {
        return false;
    }
    if proof.adj_at_tstar != tau_ext
        || proof.choice_at_tstar != Ext::from(Fp::new(1))
    {
        return false;
    }

    // The two point claims.
    let inner = crate::field::arith::bool_point(proof.tstar, log_e);
    let adj_pt = View::block(m_adj, row * e, log_e).point(&inner, m_adj);
    let choice_pt = View::block(m_choice, row * e, log_e).point(&inner, m_choice);
    let adj_found = claim_value_at_id(commitment, claims, cells.adj, &adj_pt);
    let choice_found = claim_value_at_id(commitment, claims, cells.choice, &choice_pt);
    adj_found == Some(proof.adj_at_tstar) && choice_found == Some(proof.choice_at_tstar)
}

/// The public sign table `α + β·x + β²·1[x ≥ 0]` over `[lo, hi]`, padded with `1`.
pub fn sign_table(lo: i64, hi: i64, alpha: Fp, beta: Fp) -> Vec<Fp> {
    let n = (hi - lo + 1) as usize;
    let mut v: Vec<Fp> = (lo..=hi)
        .map(|x| {
            let s = Fp::new(if x >= 0 { 1 } else { 0 });
            alpha + beta * fp_i64(x) + beta * beta * s
        })
        .collect();
    v.resize(n.next_power_of_two(), Fp::ONE);
    v
}

/// A signed integer as a base-field element.
fn fp_i64(x: i64) -> Fp {
    if x < 0 {
        -Fp::new(x.unsigned_abs())
    } else {
        Fp::new(x as u64)
    }
}

#[cfg(test)]
mod tests {
    use crate::gkr::segment::Claims;
    use super::*;
    use crate::gkr::logup::{claim_lookup2, verify_lookup2};
    use crate::gkr::segment::{verify, SegmentCells};
    use crate::pcs::transcript::new_challenger;
    use crate::pcs::whir_stacked::Params;
    use p3_field::Field;

    #[test]
    fn top_k_per_row_has_its_own_threshold() {
        let (rows, e, k) = (2usize, 2usize, 1usize);
        let alpha = Fp::new(11);
        let beta = Fp::new(5);

        // scores, then adj = e·score − i.
        let scores = vec![5i64, 1, 0, 3];
        let adj: Vec<i64> = (0..rows * e)
            .map(|i| e as i64 * scores[i] - (i % e) as i64)
            .collect();
        assert_eq!(adj, vec![10, 1, 0, 5]);

        // Row 0: top-1 is 10 at t*=0. Row 1: top-1 is 5 at t*=1.
        let tau = vec![10i64, 5];
        let tstar = vec![0usize, 1usize];
        let delta: Vec<i64> = (0..rows * e)
            .map(|i| adj[i] - tau[i / e])
            .collect();
        let choice: Vec<i64> = delta.iter().map(|&x| if x >= 0 { 1 } else { 0 }).collect();
        assert_eq!(choice, vec![1, 0, 0, 1]);

        let fp_of = |x: i64, s: i64| alpha + beta * fp_i64(x) + beta * beta * fp_i64(s);
        let (lo, hi) = (*delta.iter().min().unwrap(), *delta.iter().max().unwrap());
        let big_n = (hi - lo + 1) as usize;
        let ft_d = big_n.next_power_of_two();
        let mut table: Vec<Fp> = (lo..=hi)
            .map(|x| fp_of(x, if x >= 0 { 1 } else { 0 }))
            .collect();
        table.resize(ft_d, Fp::ONE);
        assert_eq!(table, {
            let mut t = sign_table(lo, hi, alpha, beta);
            t.resize(ft_d, Fp::ONE);
            t
        });
        let fpv: Vec<Fp> = delta
            .iter()
            .zip(&choice)
            .map(|(&x, &s)| fp_of(x, s))
            .collect();
        let mut counts = vec![0usize; big_n];
        for &x in &delta {
            counts[(x - lo) as usize] += 1;
        }
        let mut mm: Vec<Fp> = counts.iter().map(|&c| Fp::new(c as u64)).collect();
        mm.resize(ft_d, Fp::ONE);
        let mut diff_t: Vec<Fp> = counts
            .iter()
            .zip(table.iter().take(big_n))
            .map(|(&c, &t)| Fp::new(c as u64) * t.inverse())
            .collect();
        diff_t.resize(ft_d, Fp::ONE);
        let diff_q: Vec<Fp> = fpv.iter().map(|&v| v.inverse()).collect();

        let mut cells = SegmentCells::new();
        let col = |c: &mut SegmentCells, name: &str, v: &[i64]| {
            c.cell_fp(name, v.len(), 1, v.iter().map(|&x| fp_i64(x)).collect());
        };
        col(&mut cells, "scores", &scores);
        col(&mut cells, "choice", &choice);
        col(&mut cells, "adj", &adj);
        col(&mut cells, "delta", &delta);
        cells.cell_fp("fp", rows * e, 1, fpv.clone());
        cells.cell_fp("dq", rows * e, 1, diff_q);
        cells.cell_fp("ft", ft_d, 1, table.clone());
        cells.cell_fp("dt", ft_d, 1, diff_t);
        cells.cell_fp("m", ft_d, 1, mm);

        let mut seg = cells.seal(Params::default());
        let com = seg.commitment();
        let tc = TopKCells {
            scores: seg.id("scores"),
            choice: seg.id("choice"),
            adj: seg.id("adj"),
            delta: seg.id("delta"),
            lookup: Lookup2Cells {
                input: seg.id("delta"),
                output: seg.id("choice"),
                fp: seg.id("fp"),
                diff_q: seg.id("dq"),
                ft: seg.id("ft"),
                diff_t: seg.id("dt"),
                m: seg.id("m"),
            },
        };

        let mut tr = Transcript::new(&new_challenger());
        let mut proofs = Vec::new();
        for row in 0..rows {
            proofs.push(claim_top_k_row(
                &mut seg,
                &tc,
                (row, e, k, tau[row], tstar[row]),
                &mut tr,
            ));
        }
        let lk = claim_lookup2(
            &mut seg,
            tc.lookup.clone(),
            (alpha, beta),
            &table,
            (rows * e, big_n),
            &mut tr,
        );

        let claims = seg.claims().to_vec();
        let wproof = seg.prove(&mut new_challenger());
        assert!(verify(&com, &Claims::new(&claims), &wproof, &mut new_challenger()));

        let mut vtr = Transcript::new(&new_challenger());
        for row in 0..rows {
            assert!(verify_top_k_row(&com, &Claims::new(&claims), &tc, &proofs[row], (row, e, k), &mut vtr));
        }
        assert!(verify_lookup2(
            &com,
            &Claims::new(&claims),
            &lk,
            &table,
            (rows * e, big_n),
            &mut vtr
        ));

        // A threshold that no selected element achieves is rejected.
        let mut bad = proofs[0].clone();
        bad.adj_at_tstar += Ext::ONE;
        let mut vtr = Transcript::new(&new_challenger());
        assert!(!verify_top_k_row(&com, &Claims::new(&claims), &tc, &bad, (0, e, k), &mut vtr));
    }
}
