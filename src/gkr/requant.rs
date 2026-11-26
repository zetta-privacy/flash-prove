//! `requant` (integer rounding) on the stacked-WHIR layer.
//!
//! `out[i] = requant(x[i], k)` — round half away from zero — is proven by sign /
//! magnitude arithmetic plus two range checks:
//!
//! ```text
//!   s      = (x < 0)                     s ⊙ s = s        (boolean)
//!   sx     = s ⊙ x                       claim_mul
//!   sout   = s ⊙ out                     claim_mul
//!   abs_x  = x − 2·sx                    linear
//!   abs_out= out − 2·sout                linear
//!   rem    = abs_x − abs_out·2^k + 2^{k−1}   linear, and rem ∈ [0, 2^k)
//!   x      ∈ the signed digit range      claim_decomp(signed_bytes)
//! ```
//!
//! `rem ∈ [0, 2^k)` is exactly `abs_out = ⌊(abs_x + 2^{k−1}) / 2^k⌋`, i.e. the
//! rounding rule. Every operand is a planned cell, so the whole op is one WHIR
//! proof; the FRI path committed seven vectors plus two decompositions with their
//! own trees.

use crate::field::{Ext, Fp};
use crate::gkr::range::{claim_decomp, verify_decomp, Decomp, DecompCells, DecompSegmentProof};
use crate::gkr::segment::{
    claim_mul, claim_value_at_id, linear_finish, linear_verify, verify_mul, CellSegment,
    CellSegmentCommitment, MulSegmentProof,
};
use crate::pcs::linear_claim::{LinearClaimProof, LinearH};
use p3_field::PrimeCharacteristicRing;
use crate::pcs::transcript::Transcript;

/// The cells a requant occupies inside the segment.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RequantCells {
    pub x: usize,
    pub out: usize,
    /// The sign bit.
    pub s: usize,
    pub sx: usize,
    pub sout: usize,
    pub abs_x: usize,
    pub abs_out: usize,
    /// `abs_x − abs_out·2^k + 2^{k−1}`, the rounding remainder.
    pub rem: usize,
    /// The bit decomposition of `rem`.
    pub bits: DecompCells,
    /// The signed digit decomposition of `x`.
    pub digits: DecompCells,
}

/// One random-point linear claim value.
pub type Lin = (Ext, LinearClaimProof);

/// A `out = requant(x, k)` proof.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RequantSegmentProof {
    pub n: usize,
    pub k: usize,
    pub digits: usize,
    pub cells: RequantCells,
    /// `s ⊙ s = s`.
    pub mul_ss: MulSegmentProof,
    /// `s ⊙ x = sx`.
    pub mul_sx: MulSegmentProof,
    /// `s ⊙ out = sout`.
    pub mul_sout: MulSegmentProof,
    /// The RLC vector, `n` entries zero-extended.
    pub r: Vec<Fp>,
    pub c_abs_x: Lin,
    pub c_x: Lin,
    pub c_sx: Lin,
    pub c_abs_out: Lin,
    pub c_out: Lin,
    pub c_sout: Lin,
    pub c_rem: Lin,
    pub bits: DecompSegmentProof,
    pub digits_proof: DecompSegmentProof,
}

/// Claim `out = requant(x, k)`.
#[allow(clippy::too_many_arguments)]
pub fn claim_requant(
    seg: &mut CellSegment,
    cells: RequantCells,
    (n, k, digits): (usize, usize, usize),
    tr: &mut Transcript,
) -> RequantSegmentProof {
    let c = &cells;
    for id in [c.x, c.out, c.s, c.sx, c.sout, c.abs_x, c.abs_out, c.rem] {
        tr.observe_val(Fp::new(id as u64));
    }

    let mul_ss = claim_mul(seg, (c.s, c.s, c.s), tr);
    let mul_sx = claim_mul(seg, (c.s, c.x, c.sx), tr);
    let mul_sout = claim_mul(seg, (c.s, c.out, c.sout), tr);

    let d = 1usize << seg.cells_meta(c.x).m();
    let (mut r, coins) = crate::pcs::linear_claim::sample_coins(n, tr);
    r.resize(d, Fp::ZERO);
    let h = LinearH::Dense(r.clone());

    let c_abs_x = seg.claim_linear_id(c.abs_x, 0, &h, tr);
    let c_x = seg.claim_linear_id(c.x, 0, &h, tr);
    let c_sx = seg.claim_linear_id(c.sx, 0, &h, tr);
    let c_abs_out = seg.claim_linear_id(c.abs_out, 0, &h, tr);
    let c_out = seg.claim_linear_id(c.out, 0, &h, tr);
    let c_sout = seg.claim_linear_id(c.sout, 0, &h, tr);
    let c_rem = seg.claim_linear_id(c.rem, 0, &h, tr);

    let bits = claim_decomp(seg, c.bits.clone(), Decomp::bits(k), (n, k), tr);
    let digits_proof = claim_decomp(
        seg,
        c.digits.clone(),
        Decomp::signed_bytes(digits),
        (n, digits),
        tr,
    );

    RequantSegmentProof {
        n,
        k,
        digits,
        cells,
        mul_ss,
        mul_sx,
        mul_sout,
        r: coins,
        c_abs_x,
        c_x,
        c_sx,
        c_abs_out,
        c_out,
        c_sout,
        c_rem,
        bits,
        digits_proof,
    }
}

/// Verify a [`RequantSegmentProof`].
pub fn verify_requant(
    commitment: &CellSegmentCommitment,
    claims: &crate::gkr::segment::Claims<'_>,
    proof: &RequantSegmentProof,
    (n, k, digits): (usize, usize, usize),
    tr: &mut Transcript,
) -> bool {
    // `digits` is the public *cap*: a narrower decomposition is a tighter range
    // check, so any `proof.digits <= digits` is sound and a wider one is not.
    if proof.n != n || proof.k != k || proof.digits > digits || proof.digits == 0 {
        return false;
    }
    let c = proof.cells.clone();
    let Some(meta) = commitment.cells.get(c.x) else {
        return false;
    };
    let d = 1usize << meta.m();
    if n > d {
        return false;
    }

    for id in [c.x, c.out, c.s, c.sx, c.sout, c.abs_x, c.abs_out, c.rem] {
        tr.observe_val(Fp::new(id as u64));
    }

    if !verify_mul(commitment, claims, &proof.mul_ss, tr)
        || !verify_mul(commitment, claims, &proof.mul_sx, tr)
        || !verify_mul(commitment, claims, &proof.mul_sout, tr)
    {
        return false;
    }

    let (mut r, coins) = crate::pcs::linear_claim::sample_coins(n, tr);
    r.resize(d, Fp::ZERO);
    if coins != proof.r {
        return false;
    }
    let h = LinearH::Dense(r.clone());

    // Replay the seven claims, collecting their values.
    let mut values: Vec<Ext> = Vec::with_capacity(7);
    for (cell, lin) in [
        (c.abs_x, &proof.c_abs_x),
        (c.x, &proof.c_x),
        (c.sx, &proof.c_sx),
        (c.abs_out, &proof.c_abs_out),
        (c.out, &proof.c_out),
        (c.sout, &proof.c_sout),
        (c.rem, &proof.c_rem),
    ] {
        let Some((z, cf)) = linear_verify(&lin.1, lin.0, tr) else {
            return false;
        };
        let Some(v) = claim_value_at_id(commitment, claims, cell, &z) else {
            return false;
        };
        if !linear_finish(&h, &z, cf, v) {
            return false;
        }
        values.push(lin.0);
    }
    let two = Ext::from(Fp::new(2));
    let (b_abs_x, b_x, b_sx) = (values[0], values[1], values[2]);
    let (b_abs_out, b_out, b_sout) = (values[3], values[4], values[5]);
    let b_rem = values[6];

    // The three linear identities.
    if b_abs_x != b_x - two * b_sx
        || b_abs_out != b_out - two * b_sout
        || b_rem
            != b_abs_x - Ext::from(Fp::new(1u64 << k)) * b_abs_out
                + Ext::from(Fp::new(1u64 << (k - 1))) * sum_rlc(&r)
    {
        return false;
    }

    // `rem ∈ [0, 2^k)` and `x` inside the signed digit range.
    verify_decomp(commitment, claims, &proof.bits, (n, k), tr)
        && verify_decomp(commitment, claims, &proof.digits_proof, (n, digits), tr)
}

/// `Σ r`, the RLC weight of a constant column.
fn sum_rlc(r: &[Fp]) -> Ext {
    r.iter().map(|&x| Ext::from(x)).sum()
}

#[cfg(test)]
mod tests {
    use crate::gkr::segment::Claims;
    use super::*;
    use p3_field::PrimeCharacteristicRing;
    use crate::gkr::segment::{verify, SegmentCells};
    use crate::model::spec::requant as requant_i64;
    use crate::pcs::transcript::new_challenger;
    use crate::pcs::whir_stacked::Params;
    use p3_field::{Field, PrimeField64};

    fn fp(x: i64) -> Fp {
        if x < 0 {
            -Fp::new(x.unsigned_abs())
        } else {
            Fp::new(x as u64)
        }
    }

    /// Plan every requant cell (including both decompositions' helper cells).
    fn plan(x: &[i64], out: &[i64], k: usize, digits: usize) -> SegmentCells {
        let n = x.len();
        let d = n.next_power_of_two();
        let s: Vec<i64> = x.iter().map(|&v| if v < 0 { 1 } else { 0 }).collect();
        let sx: Vec<i64> = (0..n).map(|i| s[i] * x[i]).collect();
        let sout: Vec<i64> = (0..n).map(|i| s[i] * out[i]).collect();
        let abs_x: Vec<i64> = (0..n).map(|i| x[i] - 2 * sx[i]).collect();
        let abs_out: Vec<i64> = (0..n).map(|i| out[i] - 2 * sout[i]).collect();
        let rem: Vec<i64> = (0..n)
            .map(|i| abs_x[i] - abs_out[i] * (1i64 << k) + (1i64 << (k - 1)))
            .collect();

        let mut cells = SegmentCells::new();
        let col = |cells: &mut SegmentCells, name: &str, v: &[i64]| {
            let mut f: Vec<Fp> = v.iter().map(|&z| fp(z)).collect();
            f.resize(d, Fp::ZERO);
            cells.cell_fp(name, d, 1, f);
        };
        col(&mut cells, "x", x);
        col(&mut cells, "out", out);
        col(&mut cells, "s", &s);
        col(&mut cells, "sx", &sx);
        col(&mut cells, "sout", &sout);
        col(&mut cells, "abs_x", &abs_x);
        col(&mut cells, "abs_out", &abs_out);
        col(&mut cells, "rem", &rem);

        add_decomp(&mut cells, "bit", &rem, Decomp::bits(k), n);
        add_decomp(&mut cells, "dig", x, Decomp::signed_bytes(digits), n);
        cells
    }

    /// Add one decomposition's cells under `prefix`.
    fn add_decomp(
        cells: &mut SegmentCells,
        prefix: &str,
        values: &[i64],
        decomp: Decomp,
        n: usize,
    ) {
        let d = n.next_power_of_two();
        let cols = decomp.columns(values, d);
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

    fn decomp_cells(seg: &CellSegment, prefix: &str, _digits: usize) -> DecompCells {
        DecompCells {
            value: seg.id(&format!("{}", if prefix == "bit" { "rem" } else { "x" })),
            all_digits: seg.id(&format!("{prefix}all")),
            diff_q: seg.id(&format!("{prefix}dq")),
            ft: seg.id(&format!("{prefix}ft")),
            diff_t: seg.id(&format!("{prefix}dt")),
            m: seg.id(&format!("{prefix}m")),
        }
    }

    fn run(x: &[i64], out: &[i64], k: usize, digits: usize) -> bool {
        let n = x.len();
        let mut seg = plan(x, out, k, digits).seal(Params::default());
        let com = seg.commitment();
        let cells = RequantCells {
            x: seg.id("x"),
            out: seg.id("out"),
            s: seg.id("s"),
            sx: seg.id("sx"),
            sout: seg.id("sout"),
            abs_x: seg.id("abs_x"),
            abs_out: seg.id("abs_out"),
            rem: seg.id("rem"),
            bits: decomp_cells(&seg, "bit", k),
            digits: decomp_cells(&seg, "dig", digits),
        };
        let mut tr = Transcript::new(&new_challenger());
        let proof = claim_requant(&mut seg, cells, (n, k, digits), &mut tr);
        let claims = seg.claims().to_vec();
        let wproof = seg.prove(&mut new_challenger());
        if !verify(&com, &Claims::new(&claims), &wproof, &mut new_challenger()) {
            return false;
        }
        let mut vtr = Transcript::new(&new_challenger());
        verify_requant(&com, &Claims::new(&claims), &proof, (n, k, digits), &mut vtr)
    }

    #[test]
    fn requant_roundtrip() {
        let x = [5i64, -5, 3, -3];
        let out: Vec<i64> = x.iter().map(|&v| requant_i64(v, 2)).collect();
        assert_eq!(out, vec![1, -1, 1, -1]);
        assert!(run(&x, &out, 2, 1));
    }

    #[test]
    fn wrong_rounding_is_rejected() {
        let x = [5i64, -5, 3, -3];
        let out = [1i64, -1, 1, 0];
        assert!(!run(&x, &out, 2, 1));
    }
}
