//! RMSNorm on the stacked-WHIR layer, one row at a time.
//!
//! Per row of `hidden` values:
//!
//! ```text
//!   sum_sq = Σ x²                     claim_sum_squares
//!   h      = x ⊙ γ                    claim_mul
//!   y      = h · inv_q                point pair
//!   ob     = requant(y, RECIP_Q)      claim_requant
//!   x_row  = x's row block            one view opening, compared
//!   ob     = out's row block          one view opening, compared
//! ```
//!
//! The row is a *block* of the shared `x` cell, so the ties are point claims at a
//! Boolean-extended point ([`View`]) — the FRI path needed a `__norm_x` commitment
//! plus a `tie_row` for the same thing.

use crate::field::{Ext, Fp};
use crate::gkr::logup::{claim_eq_public, verify_eq_public, EqPublicProof};
use crate::gkr::requant::{claim_requant, verify_requant, RequantCells, RequantSegmentProof};
use crate::gkr::segment::{
    claim_mul, claim_sum_squares, claim_value_at_id, linear_finish, linear_verify, verify_mul,
    verify_sum_squares, CellClaim, CellSegment, CellSegmentCommitment, MulSegmentProof,
    SumSquaresProof, View,
};
use crate::pcs::linear_claim::{LinearClaimProof, LinearH};
use crate::pcs::transcript::Transcript;
use p3_field::PrimeCharacteristicRing;

/// One random-point linear claim value.
pub type Lin = (Ext, LinearClaimProof);

/// The cells one norm row touches.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NormRowCells {
    /// Shared trace cells (`rows × hidden`).
    pub x: usize,
    pub out: usize,
    /// The (public) scale weight, `hidden` entries.
    pub gamma: usize,
    /// Per-row derived cells.
    pub xr: usize,
    pub h: usize,
    pub y: usize,
    pub ob: usize,
    pub requant: RequantCells,
}

/// A single-row RMSNorm claim.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NormRowProof {
    pub row: usize,
    pub hidden: usize,
    pub cells: NormRowCells,
    pub ss: SumSquaresProof,
    pub mul: MulSegmentProof,
    /// Binds the segment's copy of `γ` to the public weight.
    pub gamma_public: EqPublicProof,
    pub r: Vec<Fp>,
    pub xr_v: Lin,
    pub x_v: Lin,
    pub h_v: Lin,
    pub y_v: Lin,
    pub ob_v: Lin,
    pub out_v: Lin,
    pub requant: RequantSegmentProof,
}

/// The reference reciprocal `inv_q = round(2^Q / sqrt(mean_sq + eps))`.
///
/// It is a *pure function of the proven sum of squares*, so the verifier derives it
/// instead of accepting it from the prover — otherwise `y = inv_q·h` proves nothing
/// about the actual normalisation factor.
pub fn norm_inv_q(sum_sq: i64, n: usize, rec_q: u32, eps: f64) -> i64 {
    let mean = sum_sq as f64 / n as f64;
    let denom = (mean + eps).sqrt();
    if denom <= 0.0 {
        return 0;
    }
    ((1i64 << rec_q) as f64 / denom).round() as i64
}

/// The sum of squares as a small non-negative integer, or `None`.
fn small_int(e: Ext) -> Option<i64> {
    use p3_field::BasedVectorSpace;
    let c = <Ext as BasedVectorSpace<Fp>>::as_basis_coefficients_slice(&e);
    if c[1..].iter().any(|&x| x != Fp::new(0)) {
        return None;
    }
    let v = p3_field::PrimeField64::as_canonical_u64(&c[0]);
    if v >= (1u64 << 62) {
        return None;
    }
    Some(v as i64)
}

/// Claim one RMSNorm row.
#[allow(clippy::too_many_arguments)]
pub fn claim_norm_row(
    seg: &mut CellSegment,
    cells: NormRowCells,
    (row, hidden): (usize, usize),
    (n, k, digits): (usize, usize, usize),
    (rec_q, eps): (u32, f64),
    gamma_public: &[Fp],
    tr: &mut Transcript,
) -> NormRowProof {
    let c = cells.clone();
    for id in [c.x, c.out, c.gamma, c.xr, c.h, c.y, c.ob] {
        tr.observe_val(Fp::new(id as u64));
    }

    let ss = claim_sum_squares(seg, c.xr, tr);
    // `inv_q` is not needed here: the caller builds `y` with it and the *verifier*
    // derives it from `ss.sum`, so the prover never gets to choose it.
    let _ = (rec_q, eps);
    let mul = claim_mul(seg, (c.xr, c.gamma, c.h), tr);
    // γ is copied into the segment (it is small); this binds the copy to the
    // public weight, so the copy is not prover-chosen.
    let gamma_public = claim_eq_public(seg, c.gamma, gamma_public, tr);

    // `y = inv_q · h` at one point.
    let r: Vec<Fp> = (0..hidden).map(|_| tr.sample_val()).collect();
    let h = LinearH::Dense(r.clone());
    let h_v = seg.claim_linear_view(c.h, &View::Whole, &h, tr);
    let y_v = seg.claim_linear_view(c.y, &View::Whole, &h, tr);

    let requant = claim_requant(seg, c.requant.clone(), (n, k, digits), tr);

    // The row ties.
    let m_x = seg.cells_meta(c.x).m();
    let bx = View::block(m_x, row * hidden, hidden.trailing_zeros() as usize);
    let x_v = seg.claim_linear_view(c.x, &bx, &h, tr);
    let xr_v = seg.claim_linear_view(c.xr, &View::Whole, &h, tr);
    let ob_v = seg.claim_linear_view(c.ob, &View::Whole, &h, tr);
    let m_out = seg.cells_meta(c.out).m();
    let bo = View::block(m_out, row * hidden, hidden.trailing_zeros() as usize);
    let out_v = seg.claim_linear_view(c.out, &bo, &h, tr);

    NormRowProof {
        row,
        hidden,
        cells,
        ss,
        mul,
        gamma_public,
        r,
        xr_v,
        x_v,
        h_v,
        y_v,
        ob_v,
        out_v,
        requant,
    }
}

/// Verify a [`NormRowProof`].
#[allow(clippy::too_many_arguments)]
pub fn verify_norm_row(
    commitment: &CellSegmentCommitment,
    claims: &[CellClaim],
    proof: &NormRowProof,
    (row, hidden): (usize, usize),
    (n, k, digits): (usize, usize, usize),
    (rec_q, eps): (u32, f64),
    gamma_public: &[Fp],
    tr: &mut Transcript,
) -> bool {
    let c = proof.cells.clone();
    if proof.row != row || proof.hidden != hidden {
        return false;
    }
    let (Some(m_x), Some(m_out)) = (
        commitment.cells.get(c.x).map(|c| c.m()),
        commitment.cells.get(c.out).map(|c| c.m()),
    ) else {
        return false;
    };
    for id in [c.x, c.out, c.gamma, c.xr, c.h, c.y, c.ob] {
        tr.observe_val(Fp::new(id as u64));
    }

    if !verify_sum_squares(commitment, claims, &proof.ss, tr) {
        return false;
    }
    // `inv_q` is derived from the proven sum, never taken from the proof.
    let Some(sum_sq) = small_int(proof.ss.sum) else {
        return false;
    };
    let inv_q = Fp::new(norm_inv_q(sum_sq, hidden, rec_q, eps) as u64);
    if !verify_mul(commitment, claims, &proof.mul, tr) {
        return false;
    }
    if !verify_eq_public(commitment, claims, &proof.gamma_public, gamma_public, tr) {
        return false;
    }

    let r: Vec<Fp> = (0..hidden).map(|_| tr.sample_val()).collect();
    if r != proof.r {
        return false;
    }
    let h = LinearH::Dense(r.clone());
    let bits = hidden.trailing_zeros() as usize;

    // Replay each claim and collect its value, in the prover's order.
    let mut vals: Vec<Ext> = Vec::with_capacity(6);
    let first = [
        (c.h, View::Whole, &proof.h_v),
        (c.y, View::Whole, &proof.y_v),
    ];
    for (cell, view, lin) in first {
        let Some(v) = replay(commitment, claims, cell, &view, m_x, m_out, lin, &h, tr) else {
            return false;
        };
        vals.push(v);
    }

    // The requant sits between the scale pair and the ties in the prover.
    if !verify_requant(commitment, claims, &proof.requant, (n, k, digits), tr) {
        return false;
    }

    let rest = [
        (c.x, View::block(m_x, row * hidden, bits), &proof.x_v),
        (c.xr, View::Whole, &proof.xr_v),
        (c.ob, View::Whole, &proof.ob_v),
        (c.out, View::block(m_out, row * hidden, bits), &proof.out_v),
    ];
    for (cell, view, lin) in rest {
        let Some(v) = replay(commitment, claims, cell, &view, m_x, m_out, lin, &h, tr) else {
            return false;
        };
        vals.push(v);
    }

    let (b_h, b_y, b_x, b_xr, b_ob, b_out) = (vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]);

    b_y == Ext::from(inv_q) * b_h && b_xr == b_x && b_ob == b_out
}

/// Replay one linear claim on `cell`/`view` and return its value.
#[allow(clippy::too_many_arguments)]
fn replay(
    commitment: &CellSegmentCommitment,
    claims: &[CellClaim],
    cell: usize,
    view: &View,
    m_x: usize,
    m_out: usize,
    lin: &Lin,
    h: &LinearH,
    tr: &mut Transcript,
) -> Option<Ext> {
    let m = match view {
        View::Whole => commitment.cells.get(cell)?.m(),
        _ => {
            if cell == m_x {
                m_x
            } else {
                m_out
            }
        }
    };
    let (z, cf) = linear_verify(&lin.1, lin.0, tr)?;
    let v = claim_value_at_id(commitment, claims, cell, &view.point(&z, m))?;
    if !linear_finish(h, &z, cf, v) {
        return None;
    }
    Some(lin.0)
}

/// `R·inp` for every row; the row's angle comes from `pos = row` and `theta`.
pub fn rotate(inp: &[i64], rows: usize, heads: usize, d: usize, theta: f64) -> Vec<i64> {
    let mut rot = vec![0i64; rows * d];
    for r in 0..rows {
        // The rotation angle is the sequence position, i.e. `row / heads` in a
        // `[pos, head, dim]` cell.
        let pos = if heads == 0 { r } else { r / heads };
        for j in 0..(d / 2) {
            let i = r * d + 2 * j;
            let angle = (pos as f64) * theta.powf(-((2 * j) as f64) / d as f64);
            let c = (angle.cos() * 32768.0).round() as i64;
            let s = (angle.sin() * 32768.0).round() as i64;
            rot[i] = inp[i] * c - inp[i + 1] * s;
            rot[i + 1] = inp[i] * s + inp[i + 1] * c;
        }
    }
    rot
}

/// `Rᵀ·r`, block-diagonal per row (the verifier rebuilds this from `r` alone).
fn rotate_t(r: &[Fp], rows: usize, heads: usize, d: usize, theta: f64) -> Vec<Fp> {
    let mut rt = vec![Fp::new(0); rows * d];
    for row in 0..rows {
        let pos = if heads == 0 { row } else { row / heads };
        for j in 0..(d / 2) {
            let i = row * d + 2 * j;
            let angle = (pos as f64) * theta.powf(-((2 * j) as f64) / d as f64);
            let c = fp_i64((angle.cos() * 32768.0).round() as i64);
            let s = fp_i64((angle.sin() * 32768.0).round() as i64);
            rt[i] = r[i] * c + r[i + 1] * s;
            rt[i + 1] = -r[i] * s + r[i + 1] * c;
        }
    }
    rt
}

/// A signed integer as a base-field element.
fn fp_i64(x: i64) -> Fp {
    if x < 0 {
        -Fp::new(x.unsigned_abs())
    } else {
        Fp::new(x as u64)
    }
}

/// A `rot = R·inp` claim over the whole cell, plus the rounding that produces `out`.
///
/// The rounding is `claim_requant`, the same relation `matmul`, `norm`, `scores` and
/// `softmax` use — **not** a range lookup. A lookup table would be as wide as the
/// rotated values (the rotation scale is 2^15), i.e. hundreds of thousands of
/// entries and a product sumcheck to match; `claim_requant` is O(n) and needs no
/// published domain.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RopeSegmentProof {
    pub rows: usize,
    pub heads: usize,
    pub d: usize,
    pub inp: usize,
    pub rot: usize,
    pub r: Vec<Fp>,
    pub b_rot: Lin,
    pub b_inp: Lin,
    pub requant: RequantSegmentProof,
}

/// Claim `out = requant(R·inp)` for every row.
#[allow(clippy::too_many_arguments)]
pub fn claim_rope(
    seg: &mut CellSegment,
    rot: usize,
    requant_cells: RequantCells,
    inp: usize,
    (rows, heads, d): (usize, usize, usize),
    theta: f64,
    (k, digits): (usize, usize),
    tr: &mut Transcript,
) -> RopeSegmentProof {
    let n = rows * d;
    tr.observe_val(Fp::new(inp as u64));
    tr.observe_val(Fp::new(rot as u64));

    let r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    let rt = rotate_t(&r, rows, heads, d, theta);
    let b_rot = seg.claim_linear_id(rot, 0, &LinearH::Dense(r.clone()), tr);
    let b_inp = seg.claim_linear_id(inp, 0, &LinearH::Dense(rt), tr);

    let requant = claim_requant(seg, requant_cells, (n, k, digits), tr);

    RopeSegmentProof {
        rows,
        heads,
        d,
        inp,
        rot,
        r,
        b_rot,
        b_inp,
        requant,
    }
}

/// Verify a [`RopeSegmentProof`].
#[allow(clippy::too_many_arguments)]
pub fn verify_rope(
    commitment: &CellSegmentCommitment,
    claims: &[CellClaim],
    proof: &RopeSegmentProof,
    (rows, heads, d): (usize, usize, usize),
    theta: f64,
    (k, digits): (usize, usize),
    tr: &mut Transcript,
) -> bool {
    let n = rows * d;
    if proof.rows != rows || proof.heads != heads || proof.d != d || proof.r.len() != n {
        return false;
    }
    tr.observe_val(Fp::new(proof.inp as u64));
    tr.observe_val(Fp::new(proof.rot as u64));

    let r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    if r != proof.r {
        return false;
    }
    let rt = rotate_t(&r, rows, heads, d, theta);

    let Some((z, cf)) = linear_verify(&proof.b_rot.1, proof.b_rot.0, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, proof.rot, &z) else {
        return false;
    };
    if !linear_finish(&LinearH::Dense(r.clone()), &z, cf, v) {
        return false;
    }

    let Some((z, cf)) = linear_verify(&proof.b_inp.1, proof.b_inp.0, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, proof.inp, &z) else {
        return false;
    };
    if !linear_finish(&LinearH::Dense(rt), &z, cf, v) {
        return false;
    }

    // ⟨r, rot⟩ = ⟨Rᵀr, inp⟩ is the rotation check.
    if proof.b_rot.0 != proof.b_inp.0 {
        return false;
    }

    verify_requant(commitment, claims, &proof.requant, (n, k, digits), tr)
}

/// The reference reciprocal `inv_q = round(2^Q / s)` for a softmax total `s`.
pub fn softmax_inv_q(s: i64, rec_q: u32) -> i64 {
    if s <= 0 {
        return 0;
    }
    ((1i64 << (rec_q + 15)) as f64 / s as f64).round() as i64
}

/// The row-max sign table over the softmax `d` range `[exp_lo, 0]`.
///
/// `d = mask ⊙ (scores − m)` is always ≤ 0, so every entry's indicator is `0` and
/// the lookup degenerates to the range check `d ∈ [exp_lo, 0]` — which is exactly
/// what "every prefix score is at most `m`" needs, given `scores[tstar] = m`.
pub fn max_table(exp_lo: i64, alpha: Fp, beta: Fp) -> Vec<Fp> {
    let n = max_big_n(exp_lo);
    let mut v: Vec<Fp> = (exp_lo..=0).map(|x| alpha + beta * fp_i64(x)).collect();
    v.resize(n.next_power_of_two(), Fp::ONE);
    v
}

pub fn max_big_n(exp_lo: i64) -> usize {
    (0 - exp_lo + 1) as usize
}

/// The cells one causal-softmax row touches.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SoftmaxRowCells {
    pub scores: usize,
    pub probs: usize,
    pub d: usize,
    pub e: usize,
    pub y: usize,
    pub nrm: usize,
    /// `1[d > 0]`, for the row-max obligation.
    pub over: usize,
    pub requant: RequantCells,
}

/// A single-row causal-softmax claim.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SoftmaxRowProof {
    pub row: usize,
    pub n: usize,
    pub l: usize,
    /// The exp domain's lower bound. The reference max lives in `row_max.m`.
    pub exp_lo: i64,
    /// `m = max(scores[0..=l])`, proven rather than taken on faith.
    pub row_max: crate::gkr::range::RowMaxProof,
    pub cells: SoftmaxRowCells,
    pub r: Vec<Fp>,
    pub r2: Vec<Fp>,
    pub b_d: Lin,
    pub b_scores: Lin,
    pub b_e_mask: Lin,
    /// `s = ⟨mask, e⟩`.
    pub s: Ext,
    pub b_e: Lin,
    pub b_inv: Lin,
    pub b_nrm: Lin,
    pub b_probs: Lin,
    pub requant: RequantSegmentProof,
}

/// Claim one causal-softmax row.
#[allow(clippy::too_many_arguments)]
pub fn claim_softmax_row(
    seg: &mut CellSegment,
    cells: SoftmaxRowCells,
    (row, n, l): (usize, usize, usize),
    (m, exp_lo, tstar): (i64, i64, usize),
    (_rec_q, k, digits): (u32, usize, usize),
    tr: &mut Transcript,
) -> SoftmaxRowProof {
    let c = cells.clone();
    for id in [c.scores, c.probs, c.d, c.e, c.y, c.nrm] {
        tr.observe_val(Fp::new(id as u64));
    }
    let log_n = n.trailing_zeros() as usize;

    let r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    let hm: Vec<Fp> = (0..n)
        .map(|i| if i <= l { r[i] } else { Fp::new(0) })
        .collect();
    let b_d = seg.claim_linear_id(c.d, 0, &LinearH::Dense(r.clone()), tr);
    let m_scores = seg.cells_meta(c.scores).m();
    let b_scores = seg.claim_linear_view(
        c.scores,
        &View::block(m_scores, row * n, log_n),
        &LinearH::Dense(hm),
        tr,
    );

    // `m` is the maximum of the causal prefix, not a number the prover chose.
    let row_max = crate::gkr::range::claim_row_max(
        seg,
        crate::gkr::range::RowMaxCells {
            scores: c.scores,
            d: c.d,
            over: c.over,
        },
        (row, n),
        l,
        m,
        tstar,
        tr,
    );

    let mask = LinearH::Dense(
        (0..n)
            .map(|i| if i <= l { Fp::ONE } else { Fp::ZERO })
            .collect(),
    );
    let b_e_mask = seg.claim_linear_id(c.e, 0, &mask, tr);
    let s = b_e_mask.0;

    let r2: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    let h2 = LinearH::Dense(r2.clone());
    let b_e = seg.claim_linear_id(c.e, 0, &h2, tr);
    let b_inv = seg.claim_linear_id(c.y, 0, &h2, tr);

    let requant = claim_requant(seg, c.requant.clone(), (n, k, digits), tr);
    let b_nrm = seg.claim_linear_id(c.nrm, 0, &h2, tr);
    let m_probs = seg.cells_meta(c.probs).m();
    let b_probs = seg.claim_linear_view(c.probs, &View::block(m_probs, row * n, log_n), &h2, tr);

    SoftmaxRowProof {
        row,
        n,
        l,
        exp_lo,
        row_max,
        cells,
        r,
        r2,
        b_d,
        b_scores,
        b_e_mask,
        s,
        b_e,
        b_inv,
        b_nrm,
        b_probs,
        requant,
    }
}

/// Verify a [`SoftmaxRowProof`].
#[allow(clippy::too_many_arguments)]
pub fn verify_softmax_row(
    commitment: &CellSegmentCommitment,
    claims: &[CellClaim],
    proof: &SoftmaxRowProof,
    (row, n, l): (usize, usize, usize),
    (exp_lo, tstar): (i64, usize),
    (rec_q, k, digits): (u32, usize, usize),
    tr: &mut Transcript,
) -> bool {
    if proof.row != row
        || proof.n != n
        || proof.l != l
        || proof.exp_lo != exp_lo
        || proof.row_max.tstar != tstar
    {
        return false;
    }
    let c = proof.cells.clone();
    let (Some(m_scores), Some(m_probs)) = (
        commitment.cells.get(c.scores).map(|x| x.m()),
        commitment.cells.get(c.probs).map(|x| x.m()),
    ) else {
        return false;
    };
    for id in [c.scores, c.probs, c.d, c.e, c.y, c.nrm] {
        tr.observe_val(Fp::new(id as u64));
    }
    let log_n = n.trailing_zeros() as usize;

    let r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    if r != proof.r {
        return false;
    }
    let hm: Vec<Fp> = (0..n)
        .map(|i| if i <= l { r[i] } else { Fp::new(0) })
        .collect();
    let Some((z, cf)) = linear_verify(&proof.b_d.1, proof.b_d.0, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, c.d, &z) else {
        return false;
    };
    if !linear_finish(&LinearH::Dense(r.clone()), &z, cf, v) {
        return false;
    }
    let Some((z, cf)) = linear_verify(&proof.b_scores.1, proof.b_scores.0, tr) else {
        return false;
    };
    let bxs = View::block(m_scores, row * n, log_n);
    let Some(v) = claim_value_at_id(commitment, claims, c.scores, &bxs.point(&z, m_scores)) else {
        return false;
    };
    if !linear_finish(&LinearH::Dense(hm.clone()), &z, cf, v) {
        return false;
    }
    let mut sum_hm = Ext::from(Fp::new(0));
    let mut sum_r = Ext::from(Fp::new(0));
    for i in 0..n {
        sum_hm += Ext::from(hm[i]);
        sum_r += Ext::from(r[i]);
    }
    let m_ext = Ext::from(fp_i64(proof.row_max.m));
    let exp_lo_ext = Ext::from(fp_i64(exp_lo));
    if proof.b_d.0 != proof.b_scores.0 - m_ext * sum_hm + exp_lo_ext * (sum_r - sum_hm) {
        return false;
    }

    if !crate::gkr::range::verify_row_max(
        commitment,
        claims,
        &proof.row_max,
        (row, n),
        l,
        tr,
    ) {
        return false;
    }

    let mask = LinearH::Dense(
        (0..n)
            .map(|i| if i <= l { Fp::ONE } else { Fp::ZERO })
            .collect(),
    );
    let Some((z, cf)) = linear_verify(&proof.b_e_mask.1, proof.b_e_mask.0, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, c.e, &z) else {
        return false;
    };
    if !linear_finish(&mask, &z, cf, v) {
        return false;
    }
    if proof.b_e_mask.0 != proof.s {
        return false;
    }
    let Some(s_int) = small_int(proof.s) else {
        return false;
    };
    if s_int <= 0 {
        return false;
    }
    let inv_q = fp_i64(softmax_inv_q(s_int, rec_q));

    let r2: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    if r2 != proof.r2 {
        return false;
    }
    let h2 = LinearH::Dense(r2.clone());
    let Some((z, cf)) = linear_verify(&proof.b_e.1, proof.b_e.0, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, c.e, &z) else {
        return false;
    };
    if !linear_finish(&h2, &z, cf, v) {
        return false;
    }
    let Some((z, cf)) = linear_verify(&proof.b_inv.1, proof.b_inv.0, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, c.y, &z) else {
        return false;
    };
    if !linear_finish(&h2, &z, cf, v) {
        return false;
    }
    if proof.b_inv.0 != Ext::from(inv_q) * proof.b_e.0 {
        return false;
    }

    if !verify_requant(commitment, claims, &proof.requant, (n, k, digits), tr) {
        return false;
    }

    let Some((z, cf)) = linear_verify(&proof.b_nrm.1, proof.b_nrm.0, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, c.nrm, &z) else {
        return false;
    };
    if !linear_finish(&h2, &z, cf, v) {
        return false;
    }
    let bp = View::block(m_probs, row * n, log_n);
    let Some((z, cf)) = linear_verify(&proof.b_probs.1, proof.b_probs.0, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, c.probs, &bp.point(&z, m_probs)) else {
        return false;
    };
    linear_finish(&h2, &z, cf, v) && proof.b_nrm.0 == proof.b_probs.0
}

/// One window row of an `embed` claim: `out[t, ·] == W[row, ·]`, checked at one
/// random point per side.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EmbedRow {
    /// The weight row this activation row must equal.
    pub row: usize,
    /// The point on the activation cell (the row block at `r`).
    pub out_point: Vec<Ext>,
    /// The point on the weight (the same `r` over the row's hidden block).
    pub weight_point: Vec<Ext>,
    pub out_value: Ext,
    pub weight_value: Ext,
}

/// An `embed` claim: `out[b, ·] = emb[window[b], ·]`.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EmbedSegmentProof {
    pub out: usize,
    pub weight_cell: usize,
    pub weight_name: String,
    /// The random point `r` over a hidden block, sampled after both commitments.
    pub r: Vec<Ext>,
    pub rows: Vec<EmbedRow>,
}

/// The two points one embed row is opened at: the activation's row block and the
/// weight's row, both at the same random `r` over `hidden`.
///
/// `hidden` must be a power of two (the block and the row are aligned views of the
/// two commitments), and `token < vocab`.
fn embed_points(
    t: usize,
    token: usize,
    r: &[Ext],
    (hidden_bits, m_out, m_w): (usize, usize, usize),
) -> (Vec<Ext>, Vec<Ext>) {
    let out_view = View::block(m_out, t << hidden_bits, hidden_bits);
    let row_bits: Vec<(usize, bool)> = (0..m_w - hidden_bits)
        .map(|i| (hidden_bits + i, (token >> i) & 1 == 1))
        .collect();
    let weight_view = View::gather((0..hidden_bits).collect(), row_bits);
    (out_view.point(r, m_out), weight_view.point(r, m_w))
}

/// Claim `out = emb[window]` for a whole activation row block.
///
/// Cost is `O(l · hidden)`: each row is one point opening on the activation and one
/// on the weight's own root, instead of a dense scan of the whole vocabulary.
#[allow(clippy::too_many_arguments)]
pub fn claim_embed(
    seg: &mut CellSegment,
    out: usize,
    window: &[usize],
    (l, hidden, vocab): (usize, usize, usize),
    weight: &mut crate::pcs::whir_stacked::SegmentProver,
    weight_cell: usize,
    weight_name: &str,
    tr: &mut Transcript,
) -> EmbedSegmentProof {
    assert_eq!(window.len(), l, "one window position per row");
    let m_out = seg.cells_meta(out).m();
    let m_w = weight.table_m(weight_cell);
    let hidden_bits = hidden.trailing_zeros() as usize;
    assert!(
        hidden.is_power_of_two(),
        "embed hidden must be a power of two"
    );
    assert_eq!(1usize << hidden_bits, hidden);
    assert!(
        (l << hidden_bits) <= (1usize << m_out),
        "activation rows fit the cell"
    );
    assert!(
        (1usize << (m_w - hidden_bits)) >= vocab,
        "vocabulary fits the weight table"
    );

    tr.observe_val(Fp::new(out as u64));
    tr.observe_val(Fp::new(weight_cell as u64));
    let r: Vec<Ext> = (0..hidden_bits).map(|_| tr.sample_ext()).collect();

    // One read of the tensor; the window's rows are sliced out of it. (`claim` would
    // evaluate the whole `2^m_w` MLE once per row, which is `l · vocab · hidden`.)
    let col = weight.column_logical(weight_cell, 0);
    let block = 1usize << hidden_bits;

    let mut rows = Vec::with_capacity(l);
    for (t, &token) in window.iter().enumerate() {
        let (out_point, weight_point) = embed_points(t, token, &r, (hidden_bits, m_out, m_w));
        let out_value = seg.claim_id(out, &[0], &out_point)[0];
        let start = token << hidden_bits;
        // The MLE of the row block at `r`: `Σ_j eq(r, j)·value_j` over the `hidden` entries.
        let weight_value: Ext = (0..block)
            .map(|j| crate::field::eq(&r, j) * Ext::from(col[start + j]))
            .sum();
        weight.claim_value(weight_cell, &[0], &weight_point, vec![weight_value]);
        debug_assert_eq!(out_value, weight_value, "embed row matches the weight row");
        rows.push(EmbedRow {
            row: token,
            out_point,
            weight_point,
            out_value,
            weight_value,
        });
    }

    EmbedSegmentProof {
        out,
        weight_cell,
        weight_name: weight_name.to_string(),
        r,
        rows,
    }
}

/// Verify an [`EmbedSegmentProof`]; `w_com`/`w_proof` are the tensor's.
#[allow(clippy::too_many_arguments)]
pub fn verify_embed(
    commitment: &CellSegmentCommitment,
    claims: &[CellClaim],
    w_com: &crate::pcs::whir_stacked::SegmentCommitment,
    w_proof: &crate::pcs::whir_stacked::Proof,
    proof: &EmbedSegmentProof,
    window: &[usize],
    (l, hidden, vocab): (usize, usize, usize),
    tr: &mut Transcript,
) -> bool {
    if window.len() != l {
        return false;
    }
    let Some(meta) = commitment.cells.get(proof.out) else {
        return false;
    };
    let (Some(wm), Some(wshape)) = (
        w_com.tables.get(proof.weight_cell).map(|t| t.m),
        w_com.tables.get(proof.weight_cell),
    ) else {
        return false;
    };
    let _ = wshape;
    if proof.rows.len() != l {
        return false;
    }
    let hidden_bits = hidden.trailing_zeros() as usize;
    if !hidden.is_power_of_two() || proof.r.len() != hidden_bits {
        return false;
    }
    tr.observe_val(Fp::new(proof.out as u64));
    tr.observe_val(Fp::new(proof.weight_cell as u64));
    let r: Vec<Ext> = (0..hidden_bits).map(|_| tr.sample_ext()).collect();
    if r != proof.r {
        return false;
    }

    // Every row must be opened at the two derived points, and the two values agree.
    let m_out = meta.m();
    let mut w_claims = Vec::with_capacity(l);
    for (t, row) in proof.rows.iter().enumerate() {
        if row.row != window[t] || row.row >= vocab {
            return false;
        }
        let (out_point, weight_point) = embed_points(t, row.row, &r, (hidden_bits, m_out, wm));
        if row.out_point != out_point || row.weight_point != weight_point {
            return false;
        }
        if claim_value_at_id(commitment, claims, proof.out, &out_point) != Some(row.out_value)
            || row.out_value != row.weight_value
        {
            return false;
        }
        w_claims.push(crate::pcs::whir_stacked::Claim {
            table: proof.weight_cell,
            cols: vec![0],
            point: weight_point,
            values: vec![row.weight_value],
        });
    }

    crate::pcs::whir_stacked::verify(
        w_com,
        &w_claims,
        w_proof,
        &mut crate::pcs::transcript::new_challenger(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gkr::range::Decomp;
    use crate::gkr::segment::{verify, SegmentCells};
    use crate::model::spec::requant as requant_i64;
    use crate::pcs::transcript::new_challenger;
    use crate::pcs::whir_stacked::Params;
    use p3_field::{Field, PrimeCharacteristicRing};

    fn fp(x: i64) -> Fp {
        if x < 0 {
            -Fp::new(x.unsigned_abs())
        } else {
            Fp::new(x as u64)
        }
    }

    fn decomp(cells: &mut SegmentCells, prefix: &str, vals: &[i64], dec: Decomp, n: usize) {
        let d = n.next_power_of_two();
        let cols = dec.columns(vals, d);
        let shift = dec.shift();
        let all_len = n * dec.digits;
        let all_d = all_len.next_power_of_two();
        let mut all: Vec<Fp> = Vec::new();
        for c in &cols {
            for &v in c.iter().take(n) {
                all.push(v + Fp::new(shift as u64));
            }
        }
        all.resize(all_d, Fp::ONE);
        let table = dec.table();
        let ft_d = table.len().next_power_of_two();
        let mut counts = vec![0usize; table.len()];
        for &v in all.iter().take(all_len) {
            let u = p3_field::PrimeField64::as_canonical_u64(&v) as usize;
            counts[u - 1] += 1;
        }
        let mut m: Vec<Fp> = counts.iter().map(|&c| Fp::new(c as u64)).collect();
        m.resize(ft_d, Fp::ONE);
        let diff_q: Vec<Fp> = all.iter().map(|&v| v.inverse()).collect();
        let mut ft = table.clone();
        ft.resize(ft_d, Fp::ONE);
        let mut diff_t: Vec<Fp> = counts
            .iter()
            .zip(&table)
            .map(|(&c, &t)| Fp::new(c as u64) * t.inverse())
            .collect();
        diff_t.resize(ft_d, Fp::ONE);
        cells.cell_fp(&format!("{prefix}all"), all_d, 1, all);
        cells.cell_fp(&format!("{prefix}dq"), all_d, 1, diff_q);
        cells.cell_fp(&format!("{prefix}ft"), ft_d, 1, ft);
        cells.cell_fp(&format!("{prefix}dt"), ft_d, 1, diff_t);
        cells.cell_fp(&format!("{prefix}m"), ft_d, 1, m);
    }

    fn dec_cells(
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
    fn one_norm_row() {
        let (rows, hidden) = (2usize, 4usize);
        let (row, k) = (1usize, 1usize);
        let rec_q = 8u32;
        let gamma = vec![1i64, 2, 1, 2];
        let gamma_pub: Vec<Fp> = gamma.iter().map(|&z| fp(z)).collect();
        let x: Vec<i64> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let xr: Vec<i64> = x[row * hidden..(row + 1) * hidden].to_vec();
        let h: Vec<i64> = (0..hidden).map(|j| xr[j] * gamma[j]).collect();
        let sum_sq: i64 = xr.iter().map(|&v| v * v).sum();
        let inv_q = norm_inv_q(sum_sq, hidden, rec_q, 0.0);
        let y: Vec<i64> = h.iter().map(|&v| v * inv_q).collect();
        let ob: Vec<i64> = y.iter().map(|&v| requant_i64(v, k as u32)).collect();
        let mut out_vals = vec![0i64; rows * hidden];
        out_vals[row * hidden..(row + 1) * hidden].copy_from_slice(&ob);

        let sg: Vec<i64> = y.iter().map(|&v| if v < 0 { 1 } else { 0 }).collect();
        let sx: Vec<i64> = (0..hidden).map(|i| sg[i] * y[i]).collect();
        let sout: Vec<i64> = (0..hidden).map(|i| sg[i] * ob[i]).collect();
        let abs_x: Vec<i64> = (0..hidden).map(|i| y[i] - 2 * sx[i]).collect();
        let abs_out: Vec<i64> = (0..hidden).map(|i| ob[i] - 2 * sout[i]).collect();
        let half = 1i64 << (k - 1);
        let rem: Vec<i64> = (0..hidden)
            .map(|i| abs_x[i] - abs_out[i] * (1i64 << k) + half)
            .collect();

        let mut cells = SegmentCells::new();
        let mut col = |name: &str, v: &[i64]| {
            cells.cell_fp(name, v.len(), 1, v.iter().map(|&z| fp(z)).collect());
        };
        col("x", &x);
        col("out", &out_vals);
        col("gamma", &gamma);
        col("xr", &xr);
        col("h", &h);
        col("y", &y);
        col("ob", &ob);
        col("rs", &sg);
        col("sx", &sx);
        col("sout", &sout);
        col("abs_x", &abs_x);
        col("abs_out", &abs_out);
        col("rem", &rem);
        decomp(&mut cells, "bit", &rem, Decomp::bits(k), hidden);
        decomp(&mut cells, "dig", &y, Decomp::signed_bytes(2), hidden);

        let mut seg = cells.seal(Params::default());
        let com = seg.commitment();
        let nc = NormRowCells {
            x: seg.id("x"),
            out: seg.id("out"),
            gamma: seg.id("gamma"),
            xr: seg.id("xr"),
            h: seg.id("h"),
            y: seg.id("y"),
            ob: seg.id("ob"),
            requant: RequantCells {
                x: seg.id("y"),
                out: seg.id("ob"),
                s: seg.id("rs"),
                sx: seg.id("sx"),
                sout: seg.id("sout"),
                abs_x: seg.id("abs_x"),
                abs_out: seg.id("abs_out"),
                rem: seg.id("rem"),
                bits: dec_cells(&seg, "bit", "rem", k),
                digits: dec_cells(&seg, "dig", "y", 2),
            },
        };
        let mut tr = Transcript::new(&new_challenger());
        let proof = claim_norm_row(
            &mut seg,
            nc,
            (row, hidden),
            (hidden, k, 2),
            (8, 0.0),
            &gamma_pub,
            &mut tr,
        );
        let claims = seg.claims().to_vec();
        let wproof = seg.prove(&mut new_challenger());
        assert!(verify(&com, &claims, &wproof, &mut new_challenger()));
        let mut vtr = Transcript::new(&new_challenger());
        assert!(verify_norm_row(
            &com,
            &claims,
            &proof,
            (row, hidden),
            (hidden, k, 2),
            (8, 0.0),
            &gamma_pub,
            &mut vtr
        ));
    }

    #[test]
    fn one_softmax_row() {
        use crate::model::spec::exp_q15;
        let (n, l, rec_q, k, digits) = (2usize, 1usize, 1u32, 1usize, 3usize);
        let alpha = Fp::new(11);
        let beta = Fp::new(5);
        let scores = vec![3i64, 1];
        let m = 3i64;
        let exp_lo = -2i64;
        let d: Vec<i64> = (0..n)
            .map(|i| if i <= l { scores[i] - m } else { exp_lo })
            .collect();
        let e: Vec<i64> = d.iter().map(|&x| exp_q15(x)).collect();
        let s_int: i64 = e.iter().take(l + 1).sum();
        let inv_q = super::softmax_inv_q(s_int, rec_q);
        let y: Vec<i64> = e.iter().map(|&x| x * inv_q).collect();
        let nrm: Vec<i64> = y.iter().map(|&v| requant_i64(v, k as u32)).collect();
        let probs = nrm.clone();

        let lo = exp_lo;
        let hi = 0i64;
        let big_n = (hi - lo + 1) as usize;
        let ft_d = big_n.next_power_of_two();
        let fp_of = |x: i64| alpha + beta * fp(x) + beta * beta * fp(exp_q15(x));
        let mut table: Vec<Fp> = (lo..=hi).map(fp_of).collect();
        table.resize(ft_d, Fp::ONE);
        let fpv: Vec<Fp> = d.iter().map(|&x| fp_of(x)).collect();
        let mut counts = vec![0usize; big_n];
        for &x in &d {
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

        let sg: Vec<i64> = y.iter().map(|&v| if v < 0 { 1 } else { 0 }).collect();
        let sx: Vec<i64> = (0..n).map(|i| sg[i] * y[i]).collect();
        let sout: Vec<i64> = (0..n).map(|i| sg[i] * nrm[i]).collect();
        let abs_x: Vec<i64> = (0..n).map(|i| y[i] - 2 * sx[i]).collect();
        let abs_out: Vec<i64> = (0..n).map(|i| nrm[i] - 2 * sout[i]).collect();
        let half = 1i64 << (k - 1);
        let rem: Vec<i64> = (0..n)
            .map(|i| abs_x[i] - abs_out[i] * (1i64 << k) + half)
            .collect();

        let mut cells = SegmentCells::new();
        cells.cell_fp("fp", n, 1, fpv.clone());
        cells.cell_fp("diff_q", n, 1, diff_q.clone());
        cells.cell_fp("ft", ft_d, 1, table.clone());
        cells.cell_fp("diff_t", ft_d, 1, diff_t.clone());
        cells.cell_fp("m", ft_d, 1, mm.clone());
        let mut col = |name: &str, v: &[i64]| {
            cells.cell_fp(name, v.len(), 1, v.iter().map(|&z| fp(z)).collect());
        };
        col("scores", &scores);
        col("probs", &probs);
        col("d", &d);
        col("e", &e);
        col("y", &y);
        col("nrm", &nrm);
        col("rs", &sg);
        col("sx", &sx);
        col("sout", &sout);
        col("abs_x", &abs_x);
        col("abs_out", &abs_out);
        col("rem", &rem);
        // `over` (the max indicator) is identically zero: d ≤ 0 by construction.
        let over: Vec<i64> = d.iter().map(|&x| if x > 0 { 1 } else { 0 }).collect();
        col("over", &over);
        let max_tbl = super::max_table(exp_lo, alpha, beta);
        let max_big = super::max_big_n(exp_lo);
        let mfp: Vec<Fp> = d
            .iter()
            .map(|&x| alpha + beta * fp(x) + beta * beta * fp(if x > 0 { 1 } else { 0 }))
            .collect();
        let mut mcounts = vec![0usize; max_big];
        for &x in &d {
            mcounts[(x - exp_lo) as usize] += 1;
        }
        let mut mm: Vec<Fp> = mcounts.iter().map(|&c| Fp::new(c as u64)).collect();
        mm.resize(max_tbl.len(), Fp::ONE);
        let mut mdt: Vec<Fp> = mcounts
            .iter()
            .zip(max_tbl.iter().take(max_big))
            .map(|(&c, &t)| Fp::new(c as u64) * t.inverse())
            .collect();
        mdt.resize(max_tbl.len(), Fp::ONE);
        let mdq: Vec<Fp> = mfp.iter().map(|&v| v.inverse()).collect();
        cells.cell_fp("mfp", n, 1, mfp);
        cells.cell_fp("mdq", n, 1, mdq);
        cells.cell_fp("mft", max_tbl.len(), 1, max_tbl.clone());
        cells.cell_fp("mdt", max_tbl.len(), 1, mdt);
        cells.cell_fp("mm", max_tbl.len(), 1, mm);
        decomp(&mut cells, "bit", &rem, Decomp::bits(k), n);
        decomp(&mut cells, "dig", &y, Decomp::signed_bytes(digits), n);

        let mut seg = cells.seal(Params::default());
        let com = seg.commitment();
        let sc = SoftmaxRowCells {
            scores: seg.id("scores"),
            probs: seg.id("probs"),
            d: seg.id("d"),
            e: seg.id("e"),
            y: seg.id("y"),
            nrm: seg.id("nrm"),
            over: seg.id("over"),
            requant: RequantCells {
                x: seg.id("y"),
                out: seg.id("nrm"),
                s: seg.id("rs"),
                sx: seg.id("sx"),
                sout: seg.id("sout"),
                abs_x: seg.id("abs_x"),
                abs_out: seg.id("abs_out"),
                rem: seg.id("rem"),
                bits: dec_cells(&seg, "bit", "rem", k),
                digits: dec_cells(&seg, "dig", "y", digits),
            },
        };
        let mut tr = Transcript::new(&new_challenger());
        let proof = claim_softmax_row(
            &mut seg,
            sc,
            (0, n, l),
            (m, exp_lo, 0),
            (rec_q, k, digits),
            &mut tr,
        );
        // The two lookups are op-wide now: prove them over the same query cells.
        let _ = (alpha, beta, table, big_n);
        let claims = seg.claims().to_vec();
        let wproof = seg.prove(&mut new_challenger());
        assert!(verify(&com, &claims, &wproof, &mut new_challenger()));
        let mut vtr = Transcript::new(&new_challenger());
        assert!(verify_softmax_row(
            &com,
            &claims,
            &proof,
            (0, n, l),
            (exp_lo, 0),
            (rec_q, k, digits),
            &mut vtr
        ));
    }
}
