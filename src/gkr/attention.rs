//! Attention head ops on the stacked-WHIR layer: `scores` (and, by the same
//! shape, `weighted_v`).
//!
//! One head of `scores` is
//!
//! ```text
//!   dot[t] = Σ_j k[t, hi, j] · q[li, hi, j]        claim_matmul over views
//!   y[t]   = dot[t] · inv_q                        point pair
//!   s[t]   = requant(y[t], RECIP_Q)                claim_requant (per-head cells)
//!   out[li, hi, t] = s[t]                          two views at one point
//! ```
//!
//! The trace cells (`q`, `k`, `out`) are shared by every layer and head, so each
//! operand is a [`View`]: `q`'s head block, `k`'s per-position head slice (a
//! gather over the `[pos, head, dim]` bit layout), and `out`'s head block. All
//! three are point evaluations of the shared cells — no row ties, no per-head
//! copies of the trace cells.
//!
//! `dot`, `y` and `s` are per-head ephemeral cells (they are recomputed per head,
//! not stored in the trace), which keeps `claim_requant` operating on whole cells.

use crate::field::{Ext, Fp};
use crate::gkr::requant::{claim_requant, verify_requant, RequantCells, RequantSegmentProof};
use crate::gkr::segment::{
    claim_matmul, claim_value_at_id, linear_finish, linear_verify, verify_matmul, CellClaim,
    CellSegment, CellSegmentCommitment, MatmulSegmentProof, View,
};
use crate::pcs::transcript::Transcript;
use crate::pcs::linear_claim::{LinearClaimProof, LinearH};

/// The cells one attention head touches.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct HeadCells {
    /// Shared trace cells.
    pub q: usize,
    pub k: usize,
    pub out: usize,
    /// Per-head ephemeral cells.
    pub dot: usize,
    pub y: usize,
    pub s: usize,
    /// The requant cells for `s = requant(y, RECIP_Q)` (`x = y`, `out = s`).
    pub requant: RequantCells,
}

/// One random-point linear claim value.
pub type Lin = (Ext, LinearClaimProof);

/// A single-head `scores` claim.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScoresSegmentProof {
    pub cells: HeadCells,
    /// `(log positions, log head dim, log heads)`.
    pub dims: (usize, usize, usize),
    pub head: usize,
    pub inv_q: Fp,
    pub mm: MatmulSegmentProof,
    pub r: Vec<Fp>,
    pub dot_v: Lin,
    pub y_v: Lin,
    pub s_v: Lin,
    pub out_v: Lin,
    pub requant: RequantSegmentProof,
}

/// The `k` view of one head: logical bits `[dim, pos]`, head bits fixed.
pub fn head_k_view(log_d: usize, log_h: usize, log_l: usize, head: usize) -> View {
    let mut map: Vec<usize> = (0..log_d).collect();
    map.extend((0..log_l).map(|t| log_d + log_h + t));
    let fixed: Vec<(usize, bool)> = (0..log_h)
        .map(|i| (log_d + i, (head >> i) & 1 == 1))
        .collect();
    View::gather(map, fixed)
}

/// The `[·][row]` view of a `[rows][heads][·]` cell, head bits fixed.
///
/// `lo` is the width of the per-position block: `log_d` for `q`/`k`, `log_l` for the
/// `scores` output. Logical bits are `[lo..., row...]`.
pub fn head_block_view(lo: usize, log_h: usize, log_l: usize, head: usize) -> View {
    let mut map: Vec<usize> = (0..lo).collect();
    map.extend((0..log_l).map(|r| lo + log_h + r));
    let fixed: Vec<(usize, bool)> = (0..log_h)
        .map(|i| (lo + i, (head >> i) & 1 == 1))
        .collect();
    View::gather(map, fixed)
}

/// The transposed `k` head view: logical `[row][dim]` becomes `[dim][row]`.
pub fn head_k_t_view(log_d: usize, log_h: usize, log_l: usize, head: usize) -> View {
    let mut map: Vec<usize> = (0..log_l).map(|t| log_d + log_h + t).collect();
    map.extend(0..log_d);
    let fixed: Vec<(usize, bool)> = (0..log_h)
        .map(|i| (log_d + i, (head >> i) & 1 == 1))
        .collect();
    View::gather(map, fixed)
}

/// Claim one head of `scores`.
/// Claim one **whole head** of `scores`: every query position at once.
///
/// ```text
///   dot[li][t] = Σ_j q[li, hi, j]·k[t, hi, j]        (ln = log_l, lm = log_d, lp = log_l)
///   y          = inv_q · dot
///   s          = requant(y, RECIP_Q)
///   scores[·, hi, ·] = s
/// ```
///
/// Batching the positions into one claim is what keeps the per-op cost proportional to
/// the data rather than to the number of claims: a `64 × 16` head grid used to be 1024
/// separate claims, each with its own requant and its own set of cells.
#[allow(clippy::too_many_arguments)]
pub fn claim_scores(
    seg: &mut CellSegment,
    cells: HeadCells,
    (log_l, log_d, log_h): (usize, usize, usize),
    head: usize,
    inv_q: Fp,
    (n, k, digits): (usize, usize, usize),
    tr: &mut Transcript,
) -> ScoresSegmentProof {
    for id in [cells.q, cells.k, cells.out, cells.dot, cells.y, cells.s] {
        tr.observe_val(Fp::new(id as u64));
    }

    // `dot = q_h · k_hᵀ`, over all query rows at once.
    let vq = head_block_view(log_d, log_h, log_l, head);
    let vkt = head_k_t_view(log_d, log_h, log_l, head);
    let mm = claim_matmul(
        seg,
        (cells.q, cells.k, cells.dot),
        (&vq, &vkt, &View::Whole),
        tr,
        (log_l, log_d, log_l),
    );

    // `y = inv_q · dot` over the whole head block.
    tr.observe_val(Fp::new(cells.dot as u64));
    tr.observe_val(Fp::new(cells.y as u64));
    let r: Vec<Fp> = (0..(1usize << (log_l + log_l)))
        .map(|_| tr.sample_val())
        .collect();
    let h = LinearH::Dense(r.clone());
    let dot_v = seg.claim_linear_view(cells.dot, &View::Whole, &h, tr);
    let y_v = seg.claim_linear_view(cells.y, &View::Whole, &h, tr);

    // `s = requant(y, RECIP_Q)`, then the output's head block must equal `s`.
    let requant = claim_requant(seg, cells.requant.clone(), (n, k, digits), tr);
    let s_v = seg.claim_linear_view(cells.s, &View::Whole, &h, tr);
    let vout = head_block_view(log_l, log_h, log_l, head);
    let out_v = seg.claim_linear_view(cells.out, &vout, &h, tr);

    ScoresSegmentProof {
        cells,
        dims: (log_l, log_d, log_h),
        head,
        inv_q,
        mm,
        r,
        dot_v,
        y_v,
        s_v,
        out_v,
        requant,
    }
}

/// Verify a [`ScoresSegmentProof`].
#[allow(clippy::too_many_arguments)]
pub fn verify_scores(
    commitment: &CellSegmentCommitment,
    claims: &[CellClaim],
    proof: &ScoresSegmentProof,
    (log_l, log_d, log_h): (usize, usize, usize),
    head: usize,
    inv_q: Fp,
    (n, k, digits): (usize, usize, usize),
    tr: &mut Transcript,
) -> bool {
    let cells = proof.cells.clone();
    if proof.dims != (log_l, log_d, log_h) || proof.head != head || proof.inv_q != inv_q {
        return false;
    }
    let (Some(m_q), Some(m_out)) = (
        commitment.cells.get(cells.q).map(|c| c.m()),
        commitment.cells.get(cells.out).map(|c| c.m()),
    ) else {
        return false;
    };
    for id in [cells.q, cells.k, cells.out, cells.dot, cells.y, cells.s] {
        tr.observe_val(Fp::new(id as u64));
    }

    let vq = head_block_view(log_d, log_h, log_l, head);
    let vkt = head_k_t_view(log_d, log_h, log_l, head);
    if !verify_matmul(
        commitment,
        claims,
        (cells.q, cells.k, cells.dot),
        (&vq, &vkt, &View::Whole),
        &proof.mm,
        tr,
    ) {
        return false;
    }

    tr.observe_val(Fp::new(cells.dot as u64));
    tr.observe_val(Fp::new(cells.y as u64));
    let r: Vec<Fp> = (0..(1usize << (log_l + log_l)))
        .map(|_| tr.sample_val())
        .collect();
    if r != proof.r {
        return false;
    }
    let h = LinearH::Dense(r.clone());

    // `y = inv_q · dot`.
    let Some((z, cf)) = linear_verify(&proof.dot_v.1, proof.dot_v.0, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, cells.dot, &z) else {
        return false;
    };
    if !linear_finish(&h, &z, cf, v) {
        return false;
    }
    let Some((z, cf)) = linear_verify(&proof.y_v.1, proof.y_v.0, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, cells.y, &z) else {
        return false;
    };
    if !linear_finish(&h, &z, cf, v) {
        return false;
    }
    if proof.y_v.0 != Ext::from(inv_q) * proof.dot_v.0 {
        return false;
    }

    if !verify_requant(commitment, claims, &proof.requant, (n, k, digits), tr) {
        return false;
    }

    // `scores[·, hi, ·] == s`.
    let Some((z, cf)) = linear_verify(&proof.s_v.1, proof.s_v.0, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, cells.s, &z) else {
        return false;
    };
    if !linear_finish(&h, &z, cf, v) {
        return false;
    }
    let vout = head_block_view(log_l, log_h, log_l, head);
    let Some((z, cf)) = linear_verify(&proof.out_v.1, proof.out_v.0, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, cells.out, &vout.point(&z, m_out)) else {
        return false;
    };
    let _ = m_q;
    linear_finish(&h, &z, cf, v) && proof.s_v.0 == proof.out_v.0
}

/// The cells one `weighted_v` head touches.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WeightedVCells {
    /// Shared trace cells.
    pub w: usize,
    pub v: usize,
    pub out: usize,
    /// Per-head ephemeral cells: the weighted sum and its requant.
    pub acc: usize,
    pub ob: usize,
    pub requant: RequantCells,
}

/// A single-head `weighted_v` claim.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WeightedVSegmentProof {
    pub cells: WeightedVCells,
    pub dims: (usize, usize, usize),
    pub head: usize,
    pub mm: MatmulSegmentProof,
    pub r: Vec<Fp>,
    pub ob_v: Lin,
    pub out_v: Lin,
    pub requant: RequantSegmentProof,
}

/// The `v` view for one head, transposed to `d × lq` (row = dim, col = position).
///
/// Logical index is `i·lq + t`, so its low `log_l` bits address the position and
/// its high `log_d` bits the dimension: those map to the cell's position bit (above
/// the head bits) and its dimension bit respectively.
pub fn head_v_view(log_d: usize, log_h: usize, log_l: usize, head: usize) -> View {
    let mut map: Vec<usize> = Vec::with_capacity(log_l + log_d);
    map.extend((0..log_l).map(|k| log_d + log_h + k));
    map.extend(0..log_d);
    let fixed: Vec<(usize, bool)> = (0..log_h)
        .map(|i| (log_d + i, (head >> i) & 1 == 1))
        .collect();
    View::gather(map, fixed)
}

/// Claim one **whole head** of `weighted_v`: every query position at once.
///
/// ```text
///   acc[li][i] = Σ_t w[li, hi, t]·v[t, hi, i]     (ln = log_l, lm = log_l, lp = log_d)
///   ob         = requant(acc, 15)
///   out[·, hi, ·] = ob
/// ```
#[allow(clippy::too_many_arguments)]
pub fn claim_weighted_v(
    seg: &mut CellSegment,
    cells: WeightedVCells,
    (log_l, log_d, log_h): (usize, usize, usize),
    head: usize,
    (n, k, digits): (usize, usize, usize),
    tr: &mut Transcript,
) -> WeightedVSegmentProof {
    for id in [cells.w, cells.v, cells.out, cells.acc, cells.ob] {
        tr.observe_val(Fp::new(id as u64));
    }

    // A = the head's `[li][t]` block of the attention weights,
    // B = the head's `[t][i]` block of `v`, C = `acc[li][i]`.
    let vw = head_block_view(log_l, log_h, log_l, head);
    let vv = head_k_view(log_d, log_h, log_l, head);
    let mm = claim_matmul(
        seg,
        (cells.w, cells.v, cells.acc),
        (&vw, &vv, &View::Whole),
        tr,
        (log_l, log_l, log_d),
    );

    let requant = claim_requant(seg, cells.requant.clone(), (n, k, digits), tr);

    let r: Vec<Fp> = (0..(1usize << (log_l + log_d)))
        .map(|_| tr.sample_val())
        .collect();
    let h = LinearH::Dense(r.clone());
    let ob_v = seg.claim_linear_view(cells.ob, &View::Whole, &h, tr);
    let vout = head_block_view(log_d, log_h, log_l, head);
    let out_v = seg.claim_linear_view(cells.out, &vout, &h, tr);

    WeightedVSegmentProof {
        cells,
        dims: (log_l, log_d, log_h),
        head,
        mm,
        r,
        ob_v,
        out_v,
        requant,
    }
}

/// Verify a [`WeightedVSegmentProof`].
#[allow(clippy::too_many_arguments)]
pub fn verify_weighted_v(
    commitment: &CellSegmentCommitment,
    claims: &[CellClaim],
    proof: &WeightedVSegmentProof,
    (log_l, log_d, log_h): (usize, usize, usize),
    head: usize,
    (n, k, digits): (usize, usize, usize),
    tr: &mut Transcript,
) -> bool {
    let cells = proof.cells.clone();
    if proof.dims != (log_l, log_d, log_h) || proof.head != head {
        return false;
    }
    let (Some(m_w), Some(m_out)) = (
        commitment.cells.get(cells.w).map(|c| c.m()),
        commitment.cells.get(cells.out).map(|c| c.m()),
    ) else {
        return false;
    };
    for id in [cells.w, cells.v, cells.out, cells.acc, cells.ob] {
        tr.observe_val(Fp::new(id as u64));
    }

    let vw = head_block_view(log_l, log_h, log_l, head);
    let vv = head_k_view(log_d, log_h, log_l, head);
    if !verify_matmul(
        commitment,
        claims,
        (cells.w, cells.v, cells.acc),
        (&vw, &vv, &View::Whole),
        &proof.mm,
        tr,
    ) {
        return false;
    }

    if !verify_requant(commitment, claims, &proof.requant, (n, k, digits), tr) {
        return false;
    }

    let r: Vec<Fp> = (0..(1usize << (log_l + log_d)))
        .map(|_| tr.sample_val())
        .collect();
    if r != proof.r {
        return false;
    }
    let h = LinearH::Dense(r.clone());

    let Some((z, cf)) = linear_verify(&proof.ob_v.1, proof.ob_v.0, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, cells.ob, &z) else {
        return false;
    };
    if !linear_finish(&h, &z, cf, v) {
        return false;
    }

    let _ = m_w;
    let vout = head_block_view(log_d, log_h, log_l, head);
    let Some((z, cf)) = linear_verify(&proof.out_v.1, proof.out_v.0, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, cells.out, &vout.point(&z, m_out)) else {
        return false;
    };
    linear_finish(&h, &z, cf, v) && proof.ob_v.0 == proof.out_v.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gkr::range::Decomp;
    use crate::gkr::segment::{verify, SegmentCells};
    use crate::model::spec::requant as requant_i64;
    use crate::pcs::transcript::new_challenger;
    use crate::pcs::whir_stacked::Params;
    use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

    fn fp(x: i64) -> Fp {
        if x < 0 {
            -Fp::new(x.unsigned_abs())
        } else {
            Fp::new(x as u64)
        }
    }

    fn add_decomp(cells: &mut SegmentCells, prefix: &str, vals: &[i64], decomp: Decomp, n: usize) {
        let d = n.next_power_of_two();
        let ints: Vec<i64> = vals.to_vec();
        let cols = decomp.columns(&ints, d);
        let shift = decomp.shift();
        let all_len = n * decomp.digits;
        let all_d = all_len.next_power_of_two();
        let mut all: Vec<Fp> = Vec::with_capacity(all_d);
        for c in &cols {
            for &v in c.iter().take(n) {
                all.push(v + Fp::new(shift as u64));
            }
        }
        all.resize(all_d, Fp::ONE);
        let table = decomp.table();
        let ft_d = table.len().next_power_of_two();
        let mut counts = vec![0usize; table.len()];
        for &v in all.iter().take(all_len) {
            counts[v.as_canonical_u64() as usize - 1] += 1;
        }
        let mut m: Vec<Fp> = counts.iter().map(|&c| Fp::new(c as u64)).collect();
        m.resize(ft_d, Fp::ONE);
        let diff_q: Vec<Fp> = all.iter().map(|&v| v.inverse()).collect();
        let mut ft = table.clone();
        ft.resize(ft_d, Fp::ONE);
        let mut diff_t: Vec<Fp> = m
            .iter()
            .take(table.len())
            .zip(&table)
            .map(|(&a, &b)| a * b.inverse())
            .collect();
        diff_t.resize(ft_d, Fp::ONE);
        cells.cell_fp(&format!("{prefix}all"), all_d, 1, all);
        cells.cell_fp(&format!("{prefix}dq"), all_d, 1, diff_q);
        cells.cell_fp(&format!("{prefix}ft"), ft_d, 1, ft);
        cells.cell_fp(&format!("{prefix}dt"), ft_d, 1, diff_t);
        cells.cell_fp(&format!("{prefix}m"), ft_d, 1, m);
    }

    fn decomp_cells(
        seg: &CellSegment,
        prefix: &str,
        value: &str,
        _digits: usize,
    ) -> crate::gkr::range::DecompCells {
        crate::gkr::range::DecompCells {
            value: seg.id(value),
            all_digits: seg.id(&format!("{prefix}all")),
            diff_q: seg.id(&format!("{prefix}dq")),
            ft: seg.id(&format!("{prefix}ft")),
            diff_t: seg.id(&format!("{prefix}dt")),
            m: seg.id(&format!("{prefix}m")),
        }
    }

    #[test]
    fn one_head_of_weighted_v() {
        let (lq, h, d) = (2usize, 2usize, 2usize);
        let hi = 1usize;
        let recip_q = 1usize;

        let v_vals: Vec<i64> = (0..(lq * h * d) as i64).map(|i| i % 5 - 2).collect();
        let w_vals: Vec<i64> = (0..(lq * h * lq) as i64).map(|i| i % 3 - 1).collect();

        // Head `hi`, every query position: `acc[li][i] = Σ_t w[li][hi][t]·v[t][hi][i]`.
        let mut acc: Vec<i64> = Vec::with_capacity(lq * d);
        for li in 0..lq {
            for i in 0..d {
                let mut sum = 0i64;
                for t in 0..lq {
                    sum += w_vals[((li * h) + hi) * lq + t] * v_vals[(t * h + hi) * d + i];
                }
                acc.push(sum);
            }
        }
        let ob: Vec<i64> = acc
            .iter()
            .map(|&x| requant_i64(x, recip_q as u32))
            .collect();

        let mut out_vals = vec![0i64; lq * h * d];
        for li in 0..lq {
            for i in 0..d {
                out_vals[((li * h) + hi) * d + i] = ob[li * d + i];
            }
        }

        let n = lq * d;
        let (sg, sx, sout, abs_x, abs_out, rem) = requant_cells(&acc, &ob, recip_q);

        let mut cells = SegmentCells::new();
        for (name, vals) in [("v", &v_vals), ("w", &w_vals), ("out", &out_vals)] {
            let m: Vec<Fp> = vals.iter().map(|&x| fp(x)).collect();
            cells.cell_fp(name, vals.len(), 1, m);
        }
        for (name, vals) in [("acc", &acc), ("ob", &ob)] {
            let m: Vec<Fp> = vals.iter().map(|&x| fp(x)).collect();
            cells.cell_fp(name, vals.len(), 1, m);
        }
        for (name, vals) in [
            ("rs", &sg),
            ("sx", &sx),
            ("sout", &sout),
            ("abs_x", &abs_x),
            ("abs_out", &abs_out),
            ("rem", &rem),
        ] {
            let m: Vec<Fp> = vals.iter().map(|&x| fp(x)).collect();
            cells.cell_fp(name, vals.len(), 1, m);
        }
        add_decomp(&mut cells, "bit", &rem, Decomp::bits(recip_q), n);
        add_decomp(&mut cells, "dig", &acc, Decomp::signed_bytes(1), n);

        let mut seg = cells.seal(Params::default());
        let com = seg.commitment();
        let wc = WeightedVCells {
            w: seg.id("w"),
            v: seg.id("v"),
            out: seg.id("out"),
            acc: seg.id("acc"),
            ob: seg.id("ob"),
            requant: RequantCells {
                x: seg.id("acc"),
                out: seg.id("ob"),
                s: seg.id("rs"),
                sx: seg.id("sx"),
                sout: seg.id("sout"),
                abs_x: seg.id("abs_x"),
                abs_out: seg.id("abs_out"),
                rem: seg.id("rem"),
                bits: decomp_cells(&seg, "bit", "rem", recip_q),
                digits: decomp_cells(&seg, "dig", "acc", 1),
            },
        };
        let dims = (
            lq.trailing_zeros() as usize,
            d.trailing_zeros() as usize,
            h.trailing_zeros() as usize,
        );
        let mut tr = Transcript::new(&new_challenger());
        let proof = claim_weighted_v(
            &mut seg,
            wc.clone(),
            dims,
            hi,
            (n, recip_q, 1),
            &mut tr,
        );
        let claims = seg.claims().to_vec();
        let wproof = seg.prove(&mut new_challenger());
        assert!(verify(&com, &claims, &wproof, &mut new_challenger()));
        let mut vtr = Transcript::new(&new_challenger());
        assert!(verify_weighted_v(
            &com,
            &claims,
            &proof,
            dims,
            hi,
            (n, recip_q, 1),
            &mut vtr
        ));

        // A wrong weighted sum is rejected: `acc` no longer matches the matmul.
        let mut bad_acc = acc.clone();
        bad_acc[0] += 2;
        let bad_ob: Vec<i64> = bad_acc
            .iter()
            .map(|&x| requant_i64(x, recip_q as u32))
            .collect();
        let (bsg, bsx, bsout, babs_x, babs_out, brem) = requant_cells(&bad_acc, &bad_ob, recip_q);
        let mut bad_cells = SegmentCells::new();
        for (name, vals) in [("v", &v_vals), ("w", &w_vals), ("out", &out_vals)] {
            let m: Vec<Fp> = vals.iter().map(|&x| fp(x)).collect();
            bad_cells.cell_fp(name, vals.len(), 1, m);
        }
        for (name, vals) in [("acc", &bad_acc), ("ob", &bad_ob)] {
            let m: Vec<Fp> = vals.iter().map(|&x| fp(x)).collect();
            bad_cells.cell_fp(name, vals.len(), 1, m);
        }
        for (name, vals) in [
            ("rs", &bsg),
            ("sx", &bsx),
            ("sout", &bsout),
            ("abs_x", &babs_x),
            ("abs_out", &babs_out),
            ("rem", &brem),
        ] {
            let m: Vec<Fp> = vals.iter().map(|&x| fp(x)).collect();
            bad_cells.cell_fp(name, vals.len(), 1, m);
        }
        add_decomp(&mut bad_cells, "bit", &brem, Decomp::bits(recip_q), n);
        add_decomp(&mut bad_cells, "dig", &bad_acc, Decomp::signed_bytes(1), n);

        let mut bad = bad_cells.seal(Params::default());
        let bad_com = bad.commitment();
        let bad_wc = WeightedVCells {
            w: bad.id("w"),
            v: bad.id("v"),
            out: bad.id("out"),
            acc: bad.id("acc"),
            ob: bad.id("ob"),
            requant: RequantCells {
                x: bad.id("acc"),
                out: bad.id("ob"),
                s: bad.id("rs"),
                sx: bad.id("sx"),
                sout: bad.id("sout"),
                abs_x: bad.id("abs_x"),
                abs_out: bad.id("abs_out"),
                rem: bad.id("rem"),
                bits: decomp_cells(&bad, "bit", "rem", recip_q),
                digits: decomp_cells(&bad, "dig", "acc", 1),
            },
        };
        let bad_proof = claim_weighted_v(
            &mut bad,
            bad_wc,
            dims,
            hi,
            (n, recip_q, 1),
            &mut Transcript::new(&new_challenger()),
        );
        let bad_claims = bad.claims().to_vec();
        let bad_w = bad.prove(&mut new_challenger());
        assert!(verify(&bad_com, &bad_claims, &bad_w, &mut new_challenger()));
        let mut vtr = Transcript::new(&new_challenger());
        assert!(!verify_weighted_v(
            &bad_com,
            &bad_claims,
            &bad_proof,
            dims,
            hi,
            (n, recip_q, 1),
            &mut vtr
        ));
    }
    /// The six helper vectors of `s = requant(y, recip_q)`.
    fn requant_cells(
        y: &[i64],
        s: &[i64],
        recip_q: usize,
    ) -> (Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>) {
        let sg: Vec<i64> = y.iter().map(|&v| if v < 0 { 1 } else { 0 }).collect();
        let sx: Vec<i64> = (0..y.len()).map(|i| sg[i] * y[i]).collect();
        let sout: Vec<i64> = (0..y.len()).map(|i| sg[i] * s[i]).collect();
        let abs_x: Vec<i64> = (0..y.len()).map(|i| y[i] - 2 * sx[i]).collect();
        let abs_out: Vec<i64> = (0..y.len()).map(|i| s[i] - 2 * sout[i]).collect();
        let half = 1i64 << (recip_q - 1);
        let rem: Vec<i64> = (0..y.len())
            .map(|i| abs_x[i] - abs_out[i] * (1i64 << recip_q) + half)
            .collect();
        (sg, sx, sout, abs_x, abs_out, rem)
    }

    #[test]
    fn one_head_of_scores() {
        let (lq, h, d) = (2usize, 2usize, 2usize);
        let hi = 1usize;
        let recip_q = 1usize;
        let inv_q = 2i64;

        let k_vals: Vec<i64> = (0..(lq * h * d) as i64).map(|i| i % 5 - 2).collect();
        let q_vals: Vec<i64> = (0..(lq * h * d) as i64).map(|i| i % 3 - 1).collect();

        // Head `hi` of both operands: `[row][d]`, every row.
        let kh: Vec<i64> = (0..lq)
            .flat_map(|t| {
                let s = (t * h + hi) * d;
                k_vals[s..s + d].to_vec()
            })
            .collect();
        let qh: Vec<i64> = (0..lq)
            .flat_map(|li| {
                let s = (li * h + hi) * d;
                q_vals[s..s + d].to_vec()
            })
            .collect();

        // The whole `lq x lq` score matrix of this head.
        let mut dot: Vec<i64> = Vec::with_capacity(lq * lq);
        for li in 0..lq {
            for t in 0..lq {
                dot.push((0..d).map(|j| qh[li * d + j] * kh[t * d + j]).sum());
            }
        }
        let y: Vec<i64> = dot.iter().map(|&v| v * inv_q).collect();
        let s: Vec<i64> = y.iter().map(|&v| requant_i64(v, recip_q as u32)).collect();
        assert_eq!(s, dot, "inv_q = 2^RECIP_Q rounds back to the dot product");

        let mut out_vals = vec![0i64; lq * h * lq];
        for li in 0..lq {
            for t in 0..lq {
                out_vals[((li * h) + hi) * lq + t] = s[li * lq + t];
            }
        }

        let n = lq * lq;
        let (sg, sx, sout, abs_x, abs_out, rem) = requant_cells(&y, &s, recip_q);

        let mut cells = SegmentCells::new();
        for (name, vals) in [("q", &q_vals), ("k", &k_vals), ("out", &out_vals)] {
            let m: Vec<Fp> = vals.iter().map(|&x| fp(x)).collect();
            cells.cell_fp(name, vals.len(), 1, m);
        }
        for (name, vals) in [("dot", &dot), ("y", &y), ("s", &s)] {
            let m: Vec<Fp> = vals.iter().map(|&x| fp(x)).collect();
            cells.cell_fp(name, vals.len(), 1, m);
        }
        for (name, vals) in [
            ("rs", &sg),
            ("sx", &sx),
            ("sout", &sout),
            ("abs_x", &abs_x),
            ("abs_out", &abs_out),
            ("rem", &rem),
        ] {
            let m: Vec<Fp> = vals.iter().map(|&x| fp(x)).collect();
            cells.cell_fp(name, vals.len(), 1, m);
        }
        add_decomp(&mut cells, "bit", &rem, Decomp::bits(recip_q), n);
        add_decomp(&mut cells, "dig", &y, Decomp::signed_bytes(1), n);

        let mut seg = cells.seal(Params::default());
        let com = seg.commitment();
        let hc = HeadCells {
            q: seg.id("q"),
            k: seg.id("k"),
            out: seg.id("out"),
            dot: seg.id("dot"),
            y: seg.id("y"),
            s: seg.id("s"),
            requant: RequantCells {
                x: seg.id("y"),
                out: seg.id("s"),
                s: seg.id("rs"),
                sx: seg.id("sx"),
                sout: seg.id("sout"),
                abs_x: seg.id("abs_x"),
                abs_out: seg.id("abs_out"),
                rem: seg.id("rem"),
                bits: decomp_cells(&seg, "bit", "rem", recip_q),
                digits: decomp_cells(&seg, "dig", "y", 1),
            },
        };
        let dims = (
            lq.trailing_zeros() as usize,
            d.trailing_zeros() as usize,
            h.trailing_zeros() as usize,
        );
        let mut tr = Transcript::new(&new_challenger());
        let proof = claim_scores(
            &mut seg,
            hc.clone(),
            dims,
            hi,
            Fp::new(inv_q as u64),
            (n, recip_q, 1),
            &mut tr,
        );
        let claims = seg.claims().to_vec();
        let wproof = seg.prove(&mut new_challenger());
        assert!(verify(&com, &claims, &wproof, &mut new_challenger()));
        let mut vtr = Transcript::new(&new_challenger());
        assert!(verify_scores(
            &com,
            &claims,
            &proof,
            dims,
            hi,
            Fp::new(inv_q as u64),
            (n, recip_q, 1),
            &mut vtr
        ));

        // A wrong output entry is rejected at the `s <-> out` check.
        let mut bad_out = out_vals.clone();
        // Perturb *this* head's block (index 0 belongs to head 0).
        bad_out[hi * lq] += 1;
        let mut bad_cells = SegmentCells::new();
        for (name, vals) in [("q", &q_vals), ("k", &k_vals), ("out", &bad_out)] {
            let m: Vec<Fp> = vals.iter().map(|&x| fp(x)).collect();
            bad_cells.cell_fp(name, vals.len(), 1, m);
        }
        for (name, vals) in [("dot", &dot), ("y", &y), ("s", &s)] {
            let m: Vec<Fp> = vals.iter().map(|&x| fp(x)).collect();
            bad_cells.cell_fp(name, vals.len(), 1, m);
        }
        for (name, vals) in [
            ("rs", &sg),
            ("sx", &sx),
            ("sout", &sout),
            ("abs_x", &abs_x),
            ("abs_out", &abs_out),
            ("rem", &rem),
        ] {
            let m: Vec<Fp> = vals.iter().map(|&x| fp(x)).collect();
            bad_cells.cell_fp(name, vals.len(), 1, m);
        }
        add_decomp(&mut bad_cells, "bit", &rem, Decomp::bits(recip_q), n);
        add_decomp(&mut bad_cells, "dig", &y, Decomp::signed_bytes(1), n);

        let mut bad = bad_cells.seal(Params::default());
        let bad_com = bad.commitment();
        let bad_hc = HeadCells {
            q: bad.id("q"),
            k: bad.id("k"),
            out: bad.id("out"),
            dot: bad.id("dot"),
            y: bad.id("y"),
            s: bad.id("s"),
            requant: RequantCells {
                x: bad.id("y"),
                out: bad.id("s"),
                s: bad.id("rs"),
                sx: bad.id("sx"),
                sout: bad.id("sout"),
                abs_x: bad.id("abs_x"),
                abs_out: bad.id("abs_out"),
                rem: bad.id("rem"),
                bits: decomp_cells(&bad, "bit", "rem", recip_q),
                digits: decomp_cells(&bad, "dig", "y", 1),
            },
        };
        let bad_proof = claim_scores(
            &mut bad,
            bad_hc,
            dims,
            hi,
            Fp::new(inv_q as u64),
            (n, recip_q, 1),
            &mut Transcript::new(&new_challenger()),
        );
        let bad_claims = bad.claims().to_vec();
        let bad_w = bad.prove(&mut new_challenger());
        assert!(verify(&bad_com, &bad_claims, &bad_w, &mut new_challenger()));
        let mut vtr = Transcript::new(&new_challenger());
        assert!(!verify_scores(
            &bad_com,
            &bad_claims,
            &bad_proof,
            dims,
            hi,
            Fp::new(inv_q as u64),
            (n, recip_q, 1),
            &mut vtr
        ));
    }

}
