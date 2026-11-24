//! Digit / bit decomposition and range checks on the stacked-WHIR layer.
//!
//! Proves that a committed vector decomposes into `digits` digit columns over a
//! fixed radix, with every digit inside the table's range. Two instantiations the
//! op graph needs:
//!
//! | use | radix | digits | digit range | table (shifted, zero-free) |
//! |---|---|---|---|---|
//! | `rem ∈ [0, 2^k)` (requant, top-k, head) | 2 | `k` | `{0,1}` | `{1,2}` |
//! | `x` bounded by a signed decomposition | 256 | `d` | `[-128,127]` | `[1,256]` |
//!
//! Three obligations, all on planned cells:
//!
//! 1. the digit columns are committed as cells;
//! 2. every digit is in range — one LogUp against the shifted table (shifted so no
//!    entry is zero, which the fraction LogUp needs);
//! 3. the digits reconstruct the value: at one random `r`,
//!    `r̃·value = Σ_d radix^d · (r̃·digit_d)`.
//!
//! Digit decomposition used to need extra commitments — the value, every digit
//! column and the flattened digit vector as separate roots. Here they are cells
//! of one segment, so the whole check costs one WHIR proof.

use crate::field::{Ext, Fp};
use crate::gkr::logup::{claim_logup, verify_logup, LogUpSegmentProof};
use crate::gkr::segment::{
    claim_value_at_id, linear_finish, linear_verify, CellClaim, CellSegment, CellSegmentCommitment,
    View,
};
use crate::pcs::linear_claim::{LinearClaimProof, LinearH};
use crate::pcs::transcript::Transcript;

/// A signed integer as a base-field element.
fn fp_i64(x: i64) -> Fp {
    if x < 0 {
        -Fp::new(x.unsigned_abs())
    } else {
        Fp::new(x as u64)
    }
}
use p3_field::PrimeCharacteristicRing;

/// A signed integer as a base-field element.
fn fp(x: i64) -> Fp {
    if x < 0 {
        -Fp::new(x.unsigned_abs())
    } else {
        Fp::new(x as u64)
    }
}

/// A decomposition shape: radix, number of digits, and whether digits are signed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Decomp {
    pub radix: u64,
    pub digits: usize,
    /// Signed digits use the borrow convention and range `[-radix/2, radix/2)`;
    /// unsigned digits are plain base-`radix` digits in `[0, radix)`.
    pub signed: bool,
}

impl Decomp {
    /// Unsigned binary decomposition: `value ∈ [0, 2^k)`.
    pub fn bits(k: usize) -> Self {
        Self {
            radix: 2,
            digits: k,
            signed: false,
        }
    }

    /// Signed byte decomposition: `digits` base-256 digits in `[-128, 127]`.
    pub fn signed_bytes(digits: usize) -> Self {
        Self {
            radix: 256,
            digits,
            signed: true,
        }
    }

    /// The additive shift applied to a digit before the LogUp, so no entry is zero.
    pub fn shift(&self) -> i64 {
        if self.signed {
            self.radix as i64 / 2 + 1
        } else {
            1
        }
    }

    /// The public, shifted table the digits are looked up in.
    pub fn table(&self) -> Vec<Fp> {
        if self.signed {
            (1..=self.radix).map(Fp::new).collect()
        } else {
            vec![Fp::ONE, Fp::new(2)]
        }
    }

    /// `radix^j`, the weight of digit position `j`.
    pub fn weight(&self, j: usize) -> Fp {
        Fp::new(self.radix.pow(j as u32))
    }

    /// The digit columns of `values`, each zero-padded to `d` entries.
    pub fn columns(&self, values: &[i64], d: usize) -> Vec<Vec<Fp>> {
        let mut cols = vec![vec![Fp::ZERO; d]; self.digits];
        for (i, &v) in values.iter().enumerate() {
            let ds = self.split(v);
            for (j, col) in cols.iter_mut().enumerate() {
                col[i] = fp(ds[j]);
            }
        }
        cols
    }

    /// The (signed) digit vector of one value, least significant first.
    pub fn split(&self, value: i64) -> Vec<i64> {
        if self.signed {
            signed_digits(value, self.digits, self.radix as i64)
        } else {
            (0..self.digits)
                .map(|j| (value >> j) & (self.radix as i64 - 1))
                .collect()
        }
    }
}

/// Split `value` into `n` signed base-`radix` digits (least significant first),
/// each in `[-radix/2, radix/2)`, with `value = Σ digit_j · radix^j`.
fn signed_digits(value: i64, n: usize, radix: i64) -> Vec<i64> {
    let half = radix / 2;
    let mut v = value;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        // Floor division, not truncation: only then is the remainder in `[0, radix)`.
        let mut d = v.rem_euclid(radix);
        v = (v - d) / radix;
        if d >= half {
            d -= radix;
            v += 1;
        }
        out.push(d);
    }
    out
}

/// The cells a decomposition occupies inside the segment.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DecompCells {
    /// The value being decomposed and range-checked.
    pub value: usize,
    /// One cell per digit position, least significant first.
    pub digits: Vec<usize>,
    /// The flattened, shifted digits (`digit + shift`).
    pub all_digits: usize,
    pub diff_q: usize,
    pub ft: usize,
    pub diff_t: usize,
    pub m: usize,
}

/// A decomposition / range-check proof.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DecompSegmentProof {
    pub n: usize,
    pub decomp: Decomp,
    pub cells: DecompCells,
    /// The RLC vector, `n` entries zero-extended to the value cell's arity.
    pub r: Vec<Fp>,
    pub b_value: (Ext, LinearClaimProof),
    /// `⟨r, digit_j⟩`, least significant first.
    pub b_digits: Vec<(Ext, LinearClaimProof)>,
    pub logup: LogUpSegmentProof,
}

/// Claim that `value` decomposes into in-range digits of `decomp`.
pub fn claim_decomp(
    seg: &mut CellSegment,
    cells: DecompCells,
    decomp: Decomp,
    (n, _digits): (usize, usize),
    tr: &mut Transcript,
) -> DecompSegmentProof {
    assert_eq!(cells.digits.len(), decomp.digits, "one cell per digit");
    let m = seg.cells_meta(cells.value).m();
    let d = 1usize << m;
    assert!(n <= d, "value cell too small");

    tr.observe_val(Fp::new(cells.value as u64));
    for &c in &cells.digits {
        tr.observe_val(Fp::new(c as u64));
    }
    tr.observe_val(Fp::new(cells.all_digits as u64));

    let mut r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    r.resize(d, Fp::ZERO);
    let h = LinearH::Dense(r.clone());

    let b_value = seg.claim_linear_id(cells.value, 0, &h, tr);
    let mut b_digits = Vec::with_capacity(decomp.digits);
    for &c in &cells.digits {
        b_digits.push(seg.claim_linear_id(c, 0, &h, tr));
    }

    let table = decomp.table();
    let logup = claim_logup(
        seg,
        (cells.all_digits, cells.diff_q),
        (cells.ft, cells.diff_t, cells.m),
        &table,
        (n * decomp.digits, table.len()),
        tr,
    );

    DecompSegmentProof {
        n,
        decomp,
        cells,
        r,
        b_value,
        b_digits,
        logup,
    }
}

/// Verify a [`DecompSegmentProof`].
pub fn verify_decomp(
    commitment: &CellSegmentCommitment,
    claims: &[CellClaim],
    proof: &DecompSegmentProof,
    (n, _digits): (usize, usize),
    tr: &mut Transcript,
) -> bool {
    let decomp = proof.decomp;
    let cells = proof.cells.clone();
    if proof.n != n || proof.b_digits.len() != decomp.digits || cells.digits.len() != decomp.digits
    {
        return false;
    }
    let Some(meta) = commitment.cells.get(cells.value) else {
        return false;
    };
    let d = 1usize << meta.m();
    if proof.r.len() != d || n > d {
        return false;
    }

    tr.observe_val(Fp::new(cells.value as u64));
    for &c in &cells.digits {
        tr.observe_val(Fp::new(c as u64));
    }
    tr.observe_val(Fp::new(cells.all_digits as u64));
    let mut r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    r.resize(d, Fp::ZERO);
    if r != proof.r {
        return false;
    }
    let h = LinearH::Dense(r.clone());

    let Some((z, cf)) = linear_verify(&proof.b_value.1, proof.b_value.0, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, cells.value, &z) else {
        return false;
    };
    if !linear_finish(&h, &z, cf, v) {
        return false;
    }

    let mut sum = Ext::ZERO;
    for (j, &c) in cells.digits.iter().enumerate() {
        let (bval, ref cl) = proof.b_digits[j];
        let Some((z, cf)) = linear_verify(cl, bval, tr) else {
            return false;
        };
        let Some(v) = claim_value_at_id(commitment, claims, c, &z) else {
            return false;
        };
        if !linear_finish(&h, &z, cf, v) {
            return false;
        }
        sum += Ext::from(decomp.weight(j)) * bval;
    }
    if sum != proof.b_value.0 {
        return false;
    }

    let table = decomp.table();
    verify_logup(
        commitment,
        claims,
        &proof.logup,
        &table,
        (n * decomp.digits, table.len()),
        tr,
    )
}

/// Cells for a "this is the maximum of a row" claim.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RowMaxCells {
    /// The row's cell (flat, `n` entries).
    pub scores: usize,
    /// `scores[i] − m`, whose sign must be non-positive.
    pub d: usize,
    /// `1` where `scores[i] > m`, else `0`.
    pub over: usize,
    /// The lookup over `(d, over)`.
    pub lookup: crate::gkr::logup::Lookup2Cells,
}

/// A "`m` is the maximum of this row" proof.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RowMaxProof {
    pub n: usize,
    pub row: usize,
    pub m: i64,
    /// An index achieving the maximum.
    pub tstar: usize,
    pub cells: RowMaxCells,
    pub r: Vec<Fp>,
    pub d_v: (Ext, LinearClaimProof),
    pub scores_v: (Ext, LinearClaimProof),
    /// `Σ over` — must be zero.
    pub count: (Ext, LinearClaimProof),
    /// `scores(tstar)`, which must equal `m`.
    pub tok_val: Ext,
    pub lookup: crate::gkr::logup::Lookup2SegmentProof,
}

/// The shifted sign table `{1, 2}` for the `over` indicator.
pub fn over_table() -> Vec<Fp> {
    vec![Fp::ONE, Fp::new(2)]
}

/// Claim `m = max(scores[row·n .. row·n + prefix + 1])`.
///
/// The prefix mask `1[i ≤ prefix]` is **derived from `prefix`**, which is public, so
/// no mask cell is needed and the two sides cannot disagree about it. `prefix = n−1`
/// constrains the whole row.
///
/// Two obligations, both on committed data:
///
/// ```text
///   every scores[i] − m ≤ 0        Σ_i 1[scores[i] > m] = 0     (sign lookup)
///   scores[tstar] = m             one point evaluation at a Boolean point
/// ```
///
/// Together: `m` is an upper bound **and** it is achieved, so it is the maximum.
/// This is what `causal_softmax`'s `m`, the head's `head_max` and (with one more
/// count) `top_k`'s `τ` need in order to stop being prover-chosen.
#[allow(clippy::too_many_arguments)]
pub fn claim_row_max(
    seg: &mut CellSegment,
    cells: RowMaxCells,
    (row, n): (usize, usize),
    prefix: usize,
    m: i64,
    tstar: usize,
    (alpha, beta): (Fp, Fp),
    (table, big_n): (&[Fp], usize),
    tr: &mut Transcript,
) -> RowMaxProof {
    assert!(
        prefix < n && tstar <= prefix,
        "the maximum must lie in the prefix"
    );
    let log_n = n.trailing_zeros() as usize;
    let mask: Vec<Fp> = (0..n)
        .map(|i| if i <= prefix { Fp::ONE } else { Fp::ZERO })
        .collect();
    tr.observe_val(Fp::new(cells.scores as u64));
    tr.observe_val(Fp::new(cells.d as u64));
    tr.observe_val(Fp::new(cells.over as u64));

    let r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    // `mask ⊙ r`: only the prefix carries the equation.
    let hm: Vec<Fp> = r.iter().zip(&mask).map(|(&ri, &mi)| ri * mi).collect();
    let m_ext = Ext::from(fp_i64(m));
    let mut sum_mr = Ext::from(Fp::new(0));
    for &x in &hm {
        sum_mr += Ext::from(x);
    }

    // Both sides carry the *masked* functional, so the equation constrains only the
    // prefix and masked-out entries cannot leak in.
    let d_v = seg.claim_linear_id(cells.d, 0, &LinearH::Dense(hm.clone()), tr);
    let m_scores = seg.cells_meta(cells.scores).m();
    let scores_v = seg.claim_linear_view(
        cells.scores,
        &View::block(m_scores, row * n, log_n),
        &LinearH::Dense(hm.clone()),
        tr,
    );
    assert_eq!(
        d_v.0,
        scores_v.0 - m_ext * sum_mr,
        "d = mask ⊙ (scores − m)"
    );

    let count = seg.claim_linear_id(cells.over, 0, &LinearH::Dense(vec![Fp::ONE; n]), tr);
    assert_eq!(count.0, Ext::from(Fp::new(0)), "every score is ≤ m");

    // The Boolean point is inside the row, so it goes through the row's view.
    let row_blk = View::block(m_scores, row * n, log_n);
    let tok_point = row_blk.point(&crate::field::arith::bool_point(tstar, log_n), m_scores);
    let tok_val = seg.claim_id(cells.scores, &[0], &tok_point)[0];
    assert_eq!(tok_val, m_ext, "tstar achieves the maximum");

    let lookup = crate::gkr::logup::claim_lookup2(
        seg,
        cells.lookup.clone(),
        (alpha, beta),
        table,
        (n, big_n),
        tr,
    );

    RowMaxProof {
        n,
        row,
        m,
        tstar,
        cells,
        r,
        d_v,
        scores_v,
        count,
        tok_val,
        lookup,
    }
}

/// Verify a [`RowMaxProof`].
#[allow(clippy::too_many_arguments)]
pub fn verify_row_max(
    commitment: &CellSegmentCommitment,
    claims: &[CellClaim],
    proof: &RowMaxProof,
    (row, n): (usize, usize),
    prefix: usize,
    (table, big_n): (&[Fp], usize),
    tr: &mut Transcript,
) -> bool {
    if proof.n != n || proof.row != row || prefix >= n || proof.tstar > prefix {
        return false;
    }
    let c = proof.cells.clone();
    let Some(m_scores) = commitment.cells.get(c.scores).map(|x| x.m()) else {
        return false;
    };
    let log_n = n.trailing_zeros() as usize;
    for id in [c.scores, c.d, c.over] {
        tr.observe_val(Fp::new(id as u64));
    }

    let r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
    if r != proof.r {
        return false;
    }
    let mask: Vec<Fp> = (0..n)
        .map(|i| if i <= prefix { Fp::ONE } else { Fp::ZERO })
        .collect();
    let hm: Vec<Fp> = r.iter().zip(&mask).map(|(&ri, &mi)| ri * mi).collect();
    let m_ext = Ext::from(fp_i64(proof.m));
    let mut sum_mr = Ext::from(Fp::new(0));
    for &x in &hm {
        sum_mr += Ext::from(x);
    }

    let Some((z, cf)) = linear_verify(&proof.d_v.1, proof.d_v.0, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, c.d, &z) else {
        return false;
    };
    if !linear_finish(&LinearH::Dense(hm.clone()), &z, cf, v) {
        return false;
    }
    let blk = View::block(m_scores, row * n, log_n);
    let Some((z, cf)) = linear_verify(&proof.scores_v.1, proof.scores_v.0, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, c.scores, &blk.point(&z, m_scores)) else {
        return false;
    };
    if !linear_finish(&LinearH::Dense(hm.clone()), &z, cf, v) {
        return false;
    }
    if proof.d_v.0 != proof.scores_v.0 - m_ext * sum_mr {
        return false;
    }

    let Some((z, cf)) = linear_verify(&proof.count.1, proof.count.0, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, c.over, &z) else {
        return false;
    };
    if !linear_finish(&LinearH::Dense(vec![Fp::ONE; n]), &z, cf, v) {
        return false;
    }
    if proof.count.0 != Ext::from(Fp::new(0)) {
        return false;
    }

    let row_blk = View::block(m_scores, row * n, log_n);
    let tok_point = row_blk.point(
        &crate::field::arith::bool_point(proof.tstar, log_n),
        m_scores,
    );
    if claim_value_at_id(commitment, claims, c.scores, &tok_point) != Some(m_ext) {
        return false;
    }

    crate::gkr::logup::verify_lookup2(commitment, claims, &proof.lookup, table, (n, big_n), tr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gkr::segment::{verify, SegmentCells};
    use crate::pcs::transcript::new_challenger;
    use crate::pcs::whir_stacked::Params;
    use p3_field::{Field, PrimeField64};

    /// Plan the decomposition cells for `values`.
    fn plan(values: &[i64], decomp: Decomp) -> SegmentCells {
        let n = values.len();
        let d = n.next_power_of_two();
        let all_len = n * decomp.digits;
        let all_d = all_len.next_power_of_two();

        let cols = decomp.columns(values, d);
        let shift = decomp.shift();
        let mut all: Vec<Fp> = Vec::with_capacity(all_d);
        for col in &cols {
            for &x in col.iter().take(n) {
                all.push(x + Fp::new(shift as u64));
            }
        }
        all.resize(all_d, Fp::ONE);

        let table = decomp.table();
        let mut m = vec![Fp::ONE; table.len()];
        let mut counts = vec![0usize; table.len()];
        for &v in all.iter().take(all_len) {
            counts[v.as_canonical_u64() as usize - 1] += 1;
        }
        for (j, c) in counts.iter().enumerate() {
            m[j] = Fp::new(*c as u64);
        }
        let diff_q: Vec<Fp> = all.iter().map(|&x| x.inverse()).collect();
        let diff_t: Vec<Fp> = m
            .iter()
            .zip(&table)
            .map(|(&mm, &tt)| mm * tt.inverse())
            .collect();

        let mut cells = SegmentCells::new();
        cells.cell_fp("value", d, 1, {
            let mut v: Vec<Fp> = values.iter().map(|&x| fp(x)).collect();
            v.resize(d, Fp::ZERO);
            v
        });
        for (j, col) in cols.into_iter().enumerate() {
            cells.cell_fp(&format!("d{j}"), d, 1, col);
        }
        let ft_d = table.len().next_power_of_two();
        let mut ft = table.clone();
        ft.resize(ft_d, Fp::ONE);
        let mut diff_t = diff_t;
        diff_t.resize(ft_d, Fp::ONE);
        let mut m = m;
        m.resize(ft_d, Fp::ONE);
        cells.cell_fp("all", all_d, 1, all);
        cells.cell_fp("diff_q", all_d, 1, diff_q);
        cells.cell_fp("ft", ft_d, 1, ft);
        cells.cell_fp("diff_t", ft_d, 1, diff_t);
        cells.cell_fp("m", ft_d, 1, m);
        cells
    }

    fn run(values: &[i64], decomp: Decomp) -> bool {
        let n = values.len();
        let mut seg = plan(values, decomp).seal(Params::default());
        let com = seg.commitment();
        let cells = DecompCells {
            value: seg.id("value"),
            digits: (0..decomp.digits)
                .map(|j| seg.id(&format!("d{j}")))
                .collect(),
            all_digits: seg.id("all"),
            diff_q: seg.id("diff_q"),
            ft: seg.id("ft"),
            diff_t: seg.id("diff_t"),
            m: seg.id("m"),
        };
        let mut tr = Transcript::new(&new_challenger());
        let proof = claim_decomp(&mut seg, cells, decomp, (n, decomp.digits), &mut tr);
        let claims = seg.claims().to_vec();
        let wproof = seg.prove(&mut new_challenger());
        if !verify(&com, &claims, &wproof, &mut new_challenger()) {
            return false;
        }
        let mut vtr = Transcript::new(&new_challenger());
        verify_decomp(&com, &claims, &proof, (n, decomp.digits), &mut vtr)
    }

    #[test]
    fn bits_cover_the_whole_range() {
        assert!(run(&[0, 1, 2, 3], Decomp::bits(2)));
    }

    #[test]
    fn out_of_range_value_is_rejected() {
        // 4 needs three bits: with k = 2 the reconstruction identity fails.
        assert!(!run(&[0, 1, 2, 4], Decomp::bits(2)));
    }

    #[test]
    fn three_bits_accepts_seven() {
        assert!(run(&[0, 3, 5, 7], Decomp::bits(3)));
    }

    #[test]
    fn signed_bytes_reconstruct_and_reject() {
        // Values spanning several byte digits, including negatives.
        assert!(run(&[0, 1, -1, 255, -255], Decomp::signed_bytes(2)));
        assert!(run(
            &[1_000_000, -1_000_000, 123_456],
            Decomp::signed_bytes(3)
        ));

        // A corrupted digit column must break reconstruction.
        let decomp = Decomp::signed_bytes(2);
        let vals = [300i64, -700];
        let mut cells = plan(&vals, decomp);
        // Rebuild with a tampered first digit column.
        let cols = decomp.columns(&vals, vals.len().next_power_of_two());
        let d = vals.len().next_power_of_two();
        let mut all: Vec<Fp> = Vec::new();
        for col in &cols {
            for &x in col.iter().take(vals.len()) {
                all.push(x + Fp::new(decomp.shift() as u64));
            }
        }
        let all_d = (vals.len() * decomp.digits).next_power_of_two();
        all.resize(all_d, Fp::ONE);
        let table = decomp.table();
        let mut counts = vec![0usize; table.len()];
        for &v in all.iter().take(vals.len() * decomp.digits) {
            counts[v.as_canonical_u64() as usize - 1] += 1;
        }
        let mut m: Vec<Fp> = counts.iter().map(|&c| Fp::new(c as u64)).collect();
        m.resize(table.len(), Fp::ONE);
        let diff_q: Vec<Fp> = all.iter().map(|&x| x.inverse()).collect();
        let diff_t: Vec<Fp> = m
            .iter()
            .zip(&table)
            .map(|(&a, &b)| a * b.inverse())
            .collect();
        let ft_d = table.len().next_power_of_two();
        let mut ft = table.clone();
        ft.resize(ft_d, Fp::ONE);
        let mut diff_t = diff_t;
        diff_t.resize(ft_d, Fp::ONE);
        cells.cell_fp("all", all_d, 1, all);
        cells.cell_fp("diff_q", all_d, 1, diff_q);
        cells.cell_fp("ft", ft_d, 1, ft);
        cells.cell_fp("diff_t", ft_d, 1, diff_t);
        cells.cell_fp("m", ft_d, 1, m);
        let _ = d;

        let mut seg = cells.seal(Params::default());
        let com = seg.commitment();
        let c = DecompCells {
            value: seg.id("value"),
            // Swap two digit columns: reconstruction no longer matches.
            digits: vec![seg.id("d1"), seg.id("d0")],
            all_digits: seg.id("all"),
            diff_q: seg.id("diff_q"),
            ft: seg.id("ft"),
            diff_t: seg.id("diff_t"),
            m: seg.id("m"),
        };
        let proof = claim_decomp(
            &mut seg,
            c,
            decomp,
            (vals.len(), decomp.digits),
            &mut Transcript::new(&new_challenger()),
        );
        let claims = seg.claims().to_vec();
        let wproof = seg.prove(&mut new_challenger());
        assert!(verify(&com, &claims, &wproof, &mut new_challenger()));
        let mut vtr = Transcript::new(&new_challenger());
        assert!(!verify_decomp(
            &com,
            &claims,
            &proof,
            (vals.len(), decomp.digits),
            &mut vtr
        ));
    }

    #[test]
    fn row_max_is_an_upper_bound_and_is_achieved() {
        use crate::gkr::logup::Lookup2Cells;
        use crate::gkr::segment::{verify, SegmentCells};
        use crate::pcs::transcript::{new_challenger, Transcript};
        use crate::pcs::whir_stacked::Params;
        use p3_field::Field;

        let (rows, n) = (2usize, 4usize);
        let scores = vec![3i64, 7, 5, 1, 2, 2, 9, 4];
        let row = 1usize;
        // Only the prefix [0..=1] is constrained: the row is [2, 2, 9, 4] and
        // positions 2..3 must not count, otherwise `m = 2` would be refuted.
        let prefix = 1usize;
        let m = 2i64;
        let tstar = 0usize;
        assert_eq!(scores[row * n + tstar], m);
        let alpha = Fp::new(11);
        let beta = Fp::new(5);

        let d: Vec<i64> = (0..n)
            .map(|i| {
                if i <= prefix {
                    scores[row * n + i] - m
                } else {
                    0
                }
            })
            .collect();
        let over: Vec<i64> = d.iter().map(|&x| if x > 0 { 1 } else { 0 }).collect();
        let fp_of = |x: i64, s: i64| alpha + beta * fp_i64(x) + beta * beta * fp_i64(s);
        // The table is over the *d range*, so the lookup enforces `over = 1[d > 0]`
        // rather than merely "in the set of observed pairs".
        let d_lo = *d.iter().min().unwrap();
        let d_hi = std::cmp::max(*d.iter().max().unwrap(), 0);
        let big_n = (d_hi - d_lo + 1) as usize;
        let ft_d = big_n.next_power_of_two();
        let mut table: Vec<Fp> = (d_lo..=d_hi)
            .map(|x| fp_of(x, if x > 0 { 1 } else { 0 }))
            .collect();
        table.resize(ft_d, Fp::ONE);
        let fpv: Vec<Fp> = d.iter().zip(&over).map(|(&x, &s)| fp_of(x, s)).collect();
        let mut counts = vec![0usize; big_n];
        for &x in &d {
            counts[(x - d_lo) as usize] += 1;
        }
        let mut mcol: Vec<Fp> = counts.iter().map(|&c| Fp::new(c as u64)).collect();
        mcol.resize(ft_d, Fp::ONE);
        let mut diff_t: Vec<Fp> = counts
            .iter()
            .zip(table.iter().take(big_n))
            .map(|(&c, &t)| Fp::new(c as u64) * t.inverse())
            .collect();
        diff_t.resize(ft_d, Fp::ONE);
        let diff_q: Vec<Fp> = fpv.iter().map(|&v| v.inverse()).collect();

        let mut cells = SegmentCells::new();
        cells.cell_fp(
            "scores",
            rows * n,
            1,
            scores.iter().map(|&x| fp_i64(x)).collect(),
        );
        cells.cell_fp("d", n, 1, d.iter().map(|&x| fp_i64(x)).collect());
        cells.cell_fp("over", n, 1, over.iter().map(|&x| fp_i64(x)).collect());
        cells.cell_fp("fp", n, 1, fpv);
        cells.cell_fp("dq", n, 1, diff_q);
        cells.cell_fp("ft", ft_d, 1, table.clone());
        cells.cell_fp("dt", ft_d, 1, diff_t);
        cells.cell_fp("m", ft_d, 1, mcol);

        let mut seg = cells.seal(Params::default());
        let com = seg.commitment();
        let lk = Lookup2Cells {
            input: seg.id("d"),
            output: seg.id("over"),
            fp: seg.id("fp"),
            diff_q: seg.id("dq"),
            ft: seg.id("ft"),
            diff_t: seg.id("dt"),
            m: seg.id("m"),
        };
        let cells2 = super::RowMaxCells {
            scores: seg.id("scores"),
            d: seg.id("d"),
            over: seg.id("over"),
            lookup: lk,
        };
        let mut tr = Transcript::new(&new_challenger());
        let proof = claim_row_max(
            &mut seg,
            cells2,
            (row, n),
            prefix,
            m,
            tstar,
            (alpha, beta),
            (&table, big_n),
            &mut tr,
        );
        let claims = seg.claims().to_vec();
        let wproof = seg.prove(&mut new_challenger());
        assert!(verify(&com, &claims, &wproof, &mut new_challenger()));
        let mut vtr = Transcript::new(&new_challenger());
        assert!(verify_row_max(
            &com,
            &claims,
            &proof,
            (row, n),
            prefix,
            (&table, big_n),
            &mut vtr
        ));

        // An `m` that is too small is rejected: some score exceeds it.
        let mut bad = proof.clone();
        bad.m -= 1;
        let mut vtr = Transcript::new(&new_challenger());
        assert!(!verify_row_max(
            &com,
            &claims,
            &bad,
            (row, n),
            prefix,
            (&table, big_n),
            &mut vtr
        ));
    }
}
