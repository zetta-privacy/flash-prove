//! The committed LogUp lookup, on the stacked-WHIR layer.
//!
//! Proves `{query} ⊆ {table}` (with committed multiplicities) from cell openings
//! only. The fraction identity
//!
//! ```text
//!   Σ_i 1/(α − q_i) = Σ_j m_j/(α − t_j)
//! ```
//!
//! is decomposed into three pieces that need no auxiliary commitment:
//!
//! | step | claim |
//! |---|---|
//! | the derived table really is the public one | [`claim_eq_public`] |
//! | the query inverses are true inverses | [`claim_mul_ones`] |
//! | the table inverses are `m/ft` | [`claim_mul`] |
//! | balance `Σ diff_q = Σ diff_t`, and `Σ m = n` | two/one `LinearH::Dense` claims |
//!
//! The pre-WHIR version additionally committed `g_q = r⊙fq` and `g_t = r⊙ft`
//! and ran two committed inner-product arguments because it could not express a
//! pointwise product; [`claim_mul`] removes all of that.
//!
//! # Ordering
//!
//! `α` must be fixed **before** the segment holding the fingerprints is sealed,
//! otherwise the prover could choose the query values knowing `α`. This module
//! binds `α` into the transcript itself, so it is sound when `α` is public or
//! derived from an earlier root; deriving `α` from *this* root would require a
//! second segment, which the two-pass [`crate::gkr::segment::SegmentCells`] flow
//! supports.

use crate::field::{eq, Ext, Fp};
use crate::gkr::segment::{
    claim_value_at_id, linear_finish, linear_verify, CellSegment, CellSegmentCommitment,
    MulSegmentProof,
};
use crate::pcs::linear_claim::{LinearClaimProof, LinearH};
use crate::pcs::transcript::Transcript;
use p3_field::PrimeCharacteristicRing;

/// One cell proven equal to a public vector at a random point.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EqPublicProof {
    pub cell: usize,
    pub point: Vec<Ext>,
}

/// Claim that the committed cell `cell` equals the public vector `public`.
///
/// A single random point: the verifier evaluates the public polynomial itself and
/// compares. Sound by Schwartz–Zippel; cost `O(2^m · m)` on the verifier.
pub fn claim_eq_public(
    seg: &mut CellSegment,
    cell: usize,
    public: &[Fp],
    tr: &mut Transcript,
) -> EqPublicProof {
    let d = seg.cells_meta(cell).m();
    assert_eq!(public.len(), 1usize << d, "public vector arity");
    tr.observe_val(Fp::new(cell as u64));
    let point: Vec<Ext> = (0..d).map(|_| tr.sample_ext()).collect();
    seg.claim_id(cell, &[0], &point);
    EqPublicProof { cell, point }
}

/// Verify [`EqPublicProof`] against the verifier's own copy of the public vector.
pub fn verify_eq_public(
    commitment: &CellSegmentCommitment,
    claims: &crate::gkr::segment::Claims<'_>,
    proof: &EqPublicProof,
    public: &[Fp],
    tr: &mut Transcript,
) -> bool {
    let Some(meta) = commitment.cells.get(proof.cell) else {
        return false;
    };
    let d = meta.m();
    if public.len() != 1usize << d || proof.point.len() != d {
        return false;
    }
    tr.observe_val(Fp::new(proof.cell as u64));
    let point: Vec<Ext> = (0..d).map(|_| tr.sample_ext()).collect();
    if point != proof.point {
        return false;
    }
    let expected: Ext = public
        .iter()
        .enumerate()
        .map(|(j, &v)| eq(&point, j) * Ext::from(v))
        .sum();
    claim_value_at_id(commitment, claims, proof.cell, &point) == Some(expected)
}

/// The cells a two-column (graph) lookup touches inside the segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Lookup2Cells {
    pub input: usize,
    pub output: usize,
    /// `α + β·in + β²·out`, the query fingerprint.
    pub fp: usize,
    pub diff_q: usize,
    pub ft: usize,
    pub diff_t: usize,
    pub m: usize,
}

/// A graph lookup `out[i] = f(in[i])`, all sides as segment cells.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Lookup2SegmentProof {
    pub n: usize,
    pub big_n: usize,
    pub cells: Lookup2Cells,
    pub alpha: Fp,
    pub beta: Fp,
    pub r: Vec<Fp>,
    pub b_in: (Ext, LinearClaimProof),
    pub b_out: (Ext, LinearClaimProof),
    pub b_fp: (Ext, LinearClaimProof),
    pub logup: LogUpSegmentProof,
}

/// Claim `out[i] = f(in[i])` where `f` is the public table `public_ft`.
///
/// `fp = α + β·in + β²·out` is bound to `in`/`out` by three linear claims at one
/// random `r`; the verifier's identity is
/// `b_fp = α·Σr + β·b_in + β²·b_out`. The LogUp then proves `{fp} ⊆ {ft}`.
#[allow(clippy::too_many_arguments)]
pub fn claim_lookup2(
    seg: &mut CellSegment,
    cells: Lookup2Cells,
    (alpha, beta): (Fp, Fp),
    public_ft: &[Fp],
    (n, big_n): (usize, usize),
    tr: &mut Transcript,
) -> Lookup2SegmentProof {
    for id in [cells.input, cells.output, cells.fp] {
        tr.observe_val(Fp::new(id as u64));
    }
    let d = seg.cells_meta(cells.input).m();
    let dq = 1usize << d;
    // Only the `n` real query positions carry the RLC; the padding is zero so the
    // `fp` padding (which LogUp fills with `1`) cannot leak into `b_fp`.
    let (mut r, coins) = crate::pcs::linear_claim::sample_coins(n, tr);
    r.resize(dq, Fp::ZERO);
    let h = LinearH::Dense(r.clone());

    let b_in = seg.claim_linear_id(cells.input, 0, &h, tr);
    let b_out = seg.claim_linear_id(cells.output, 0, &h, tr);
    let b_fp = seg.claim_linear_id(cells.fp, 0, &h, tr);

    let logup = claim_logup(
        seg,
        (cells.fp, cells.diff_q),
        (cells.ft, cells.diff_t, cells.m),
        public_ft,
        (n, big_n),
        tr,
    );

    Lookup2SegmentProof {
        n,
        big_n,
        cells,
        alpha,
        beta,
        r: coins,
        b_in,
        b_out,
        b_fp,
        logup,
    }
}

/// Verify a [`Lookup2SegmentProof`].
pub fn verify_lookup2(
    commitment: &CellSegmentCommitment,
    claims: &crate::gkr::segment::Claims<'_>,
    proof: &Lookup2SegmentProof,
    public_ft: &[Fp],
    (n, big_n): (usize, usize),
    tr: &mut Transcript,
) -> bool {
    let cells = proof.cells;
    let Some(meta) = commitment.cells.get(cells.input) else {
        return false;
    };
    let d = meta.m();
    let dq = 1usize << d;
    if proof.n != n
        || proof.big_n != big_n
        || n > dq
        || big_n > 1usize << commitment.cells.get(cells.m).map_or(0, |c| c.m())
    {
        return false;
    }
    for id in [cells.input, cells.output, cells.fp] {
        tr.observe_val(Fp::new(id as u64));
    }
    let (mut r, coins) = crate::pcs::linear_claim::sample_coins(n, tr);
    r.resize(dq, Fp::ZERO);
    if coins != proof.r {
        return false;
    }
    let h = LinearH::Dense(r.clone());

    for (cell, (b, c)) in [
        (cells.input, &proof.b_in),
        (cells.output, &proof.b_out),
        (cells.fp, &proof.b_fp),
    ] {
        let Some((z, cf)) = linear_verify(c, *b, tr) else {
            return false;
        };
        let Some(v) = claim_value_at_id(commitment, claims, cell, &z) else {
            return false;
        };
        if !linear_finish(&h, &z, cf, v) {
            return false;
        }
    }

    if !verify_logup(commitment, claims, &proof.logup, public_ft, (n, big_n), tr) {
        return false;
    }

    // `fp` really is `α + β·in + β²·out`, checked through the shared `r`.
    let sum_r: Ext = r.iter().map(|&x| Ext::from(x)).sum();
    let (a, b) = (Ext::from(proof.alpha), Ext::from(proof.beta));
    proof.b_fp.0 == a * sum_r + b * proof.b_in.0 + b * b * proof.b_out.0
}

/// A full LogUp lookup, all sides as segment cells.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LogUpSegmentProof {
    pub n: usize,
    pub big_n: usize,
    pub fq: usize,
    pub diff_q: usize,
    pub ft: usize,
    pub diff_t: usize,
    pub m: usize,
    pub ft_public: EqPublicProof,
    /// `fq ⊙ diff_q = 1`.
    pub mul_q: MulSegmentProof,
    /// `ft ⊙ diff_t = m`.
    pub mul_t: MulSegmentProof,
    /// `Σ diff_q`.
    pub bal_q: (Ext, LinearClaimProof),
    /// `Σ diff_t`.
    pub bal_t: (Ext, LinearClaimProof),
    /// `Σ m`.
    pub sum_m: (Ext, LinearClaimProof),
}

/// The padded all-ones functional over `2^d` entries.
fn ones(d: usize) -> LinearH {
    LinearH::Ones { len: 1usize << d }
}

/// Claim `fq ⊆ ft` with multiplicities `m`.
///
/// `public_ft` is the verifier-reproducible table fingerprint vector
/// (`α − table_j`, zero-padded); `diff_q[i] = 1/fq[i]`, `diff_t[j] = m[j]/ft[j]`.
#[allow(clippy::too_many_arguments)]
pub fn claim_logup(
    seg: &mut CellSegment,
    (fq, diff_q): (usize, usize),
    (ft, diff_t, m): (usize, usize, usize),
    public_ft: &[Fp],
    (n, big_n): (usize, usize),
    tr: &mut Transcript,
) -> LogUpSegmentProof {
    // 1. The derived table fingerprint is the public one.
    let ft_public = claim_eq_public(seg, ft, public_ft, tr);

    // 2. Query inverses: fq ⊙ diff_q = 1.
    let mul_q = crate::gkr::segment::claim_mul_ones(seg, (fq, diff_q), tr);

    // 3. Table inverses: ft ⊙ diff_t = m.
    let mul_t = crate::gkr::segment::claim_mul(seg, (ft, diff_t, m), tr);

    // 4. The fraction balance and the multiplicity total.
    let dq = seg.cells_meta(diff_q).m();
    let dt = seg.cells_meta(diff_t).m();
    let dm = seg.cells_meta(m).m();
    let bal_q = seg.claim_linear_id(diff_q, 0, &ones(dq), tr);
    let bal_t = seg.claim_linear_id(diff_t, 0, &ones(dt), tr);
    let sum_m = seg.claim_linear_id(m, 0, &ones(dm), tr);

    LogUpSegmentProof {
        n,
        big_n,
        fq,
        diff_q,
        ft,
        diff_t,
        m,
        ft_public,
        mul_q,
        mul_t,
        bal_q,
        bal_t,
        sum_m,
    }
}

/// Verify a [`LogUpSegmentProof`].
pub fn verify_logup(
    commitment: &CellSegmentCommitment,
    claims: &crate::gkr::segment::Claims<'_>,
    proof: &LogUpSegmentProof,
    public_ft: &[Fp],
    (n, big_n): (usize, usize),
    tr: &mut Transcript,
) -> bool {
    use crate::gkr::segment::{verify_mul, verify_mul_ones};

    // `n`/`big_n` are public op parameters; the proof must agree with them.
    if proof.n != n || proof.big_n != big_n {
        return false;
    }

    if !verify_eq_public(commitment, claims, &proof.ft_public, public_ft, tr) {
        return false;
    }
    if !verify_mul_ones(commitment, claims, &proof.mul_q, tr) {
        return false;
    }
    if !verify_mul(commitment, claims, &proof.mul_t, tr) {
        return false;
    }

    let (Some(dq_m), Some(dt_m), Some(dm_m)) = (
        commitment.cells.get(proof.diff_q).map(|c| c.m()),
        commitment.cells.get(proof.diff_t).map(|c| c.m()),
        commitment.cells.get(proof.m).map(|c| c.m()),
    ) else {
        return false;
    };
    let (dq, dt) = (1usize << dq_m, 1usize << dt_m);

    let Some((z, cf)) = linear_verify(&proof.bal_q.1, proof.bal_q.0, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, proof.diff_q, &z) else {
        return false;
    };
    if !linear_finish(&ones(dq_m), &z, cf, v) {
        return false;
    }

    let Some((z, cf)) = linear_verify(&proof.bal_t.1, proof.bal_t.0, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, proof.diff_t, &z) else {
        return false;
    };
    if !linear_finish(&ones(dt_m), &z, cf, v) {
        return false;
    }

    let Some((z, cf)) = linear_verify(&proof.sum_m.1, proof.sum_m.0, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, proof.m, &z) else {
        return false;
    };
    if !linear_finish(&ones(dm_m), &z, cf, v) {
        return false;
    }

    // Balance of the fractions, and the multiplicity total.
    //
    // Padding is filled with `fq = diff_q = 1` and `ft = m = diff_t = 1`, so the
    // pointwise identities hold on the padded cube too and both sums carry a
    // known public offset.
    let pad_q = dq as i64 - proof.n as i64;
    let pad_t = dt as i64 - proof.big_n as i64;
    proof.bal_q.0 - proof.bal_t.0 == fp_i64(pad_q - pad_t)
        && proof.sum_m.0 == fp_i64(proof.n as i64 + pad_t)
}

/// A signed integer as a base-field element.
fn fp_i64(x: i64) -> Ext {
    Ext::from(if x < 0 {
        -Fp::new(x.unsigned_abs())
    } else {
        Fp::new(x as u64)
    })
}

/// The LogUp fingerprint challenge pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Fingerprint {
    pub alpha: Fp,
    pub beta: Fp,
}

#[cfg(test)]
mod tests {
    use crate::gkr::segment::Claims;
    use super::*;
    use crate::gkr::segment::{verify, SegmentCells};
    use crate::pcs::transcript::new_challenger;
    use crate::pcs::whir_stacked::Params;
    use p3_field::Field;

    const ALPHA: i64 = 11;

    /// The LogUp cells for `query` against the value set `table`.
    fn plan(query: &[i64], table: &[i64], m_override: Option<Vec<i64>>) -> SegmentCells {
        let n = query.len();
        let big_n = table.len();
        let dq = n.next_power_of_two();
        let dt = big_n.next_power_of_two();

        let mut m_int = vec![0i64; big_n];
        for q in query {
            let j = table
                .iter()
                .position(|t| t == q)
                .expect("query outside the table");
            m_int[j] += 1;
        }
        if let Some(over) = m_override {
            m_int = over;
        }

        // Fingerprints α − v in the field; padding is `1` so the pointwise
        // identities `fq·diff_q = 1` and `ft·diff_t = m` also hold off-support.
        let fp = |vals: &[i64]| -> Vec<Fp> {
            let mut v: Vec<Fp> = vals.iter().map(|&x| Fp::new((ALPHA - x) as u64)).collect();
            v.resize(vals.len().next_power_of_two(), Fp::ONE);
            v
        };
        let fq = fp(query);
        let ft = fp(table);
        let mut m = vec![Fp::ONE; dt];
        for (j, &x) in m_int.iter().enumerate() {
            m[j] = Fp::new(x as u64);
        }

        let inv = |v: &[Fp]| -> Vec<Fp> {
            v.iter()
                .map(|&x| if x == Fp::ZERO { Fp::ZERO } else { x.inverse() })
                .collect()
        };
        let diff_q = inv(&fq);
        let diff_t: Vec<Fp> = m
            .iter()
            .zip(&ft)
            .map(|(&mm, &tt)| {
                if tt == Fp::ZERO {
                    Fp::ZERO
                } else {
                    mm * tt.inverse()
                }
            })
            .collect();

        let mut cells = SegmentCells::new();
        cells.cell_fp("fq", dq, 1, fq);
        cells.cell_fp("diff_q", dq, 1, diff_q);
        cells.cell_fp("ft", dt, 1, ft);
        cells.cell_fp("diff_t", dt, 1, diff_t);
        cells.cell_fp("m", dt, 1, m);
        cells
    }

    /// The public table fingerprint the verifier reproduces itself.
    fn public_ft(table: &[i64]) -> Vec<Fp> {
        let mut v: Vec<Fp> = table.iter().map(|&x| Fp::new((ALPHA - x) as u64)).collect();
        v.resize(table.len().next_power_of_two(), Fp::ZERO);
        v
    }

    fn run(query: &[i64], table: &[i64], m_override: Option<Vec<i64>>) -> bool {
        let n = query.len();
        let big_n = table.len();
        let mut seg = plan(query, table, m_override).seal(Params::default());
        let com = seg.commitment();
        let ids: Vec<usize> = ["fq", "diff_q", "ft", "diff_t", "m"]
            .iter()
            .map(|name| seg.id(name))
            .collect();
        let pub_ft = public_ft(table);

        let mut tr = Transcript::new(&new_challenger());
        let proof = claim_logup(
            &mut seg,
            (ids[0], ids[1]),
            (ids[2], ids[3], ids[4]),
            &pub_ft,
            (n, big_n),
            &mut tr,
        );

        let claims = seg.claims().to_vec();
        let wproof = seg.prove(&mut new_challenger());
        if !verify(&com, &Claims::new(&claims), &wproof, &mut new_challenger()) {
            return false;
        }
        let mut vtr = Transcript::new(&new_challenger());
        verify_logup(&com, &Claims::new(&claims), &proof, &pub_ft, (n, big_n), &mut vtr)
    }

    #[test]
    fn logup_accepts_a_real_subset() {
        // {0,2,2} ⊆ {0,1,2,3}
        assert!(run(&[0, 2, 2], &[0, 1, 2, 3], None));
    }

    #[test]
    fn logup_rejects_wrong_multiplicities() {
        // Claiming m = [2,0,1,0] for {0,2,2} breaks the fraction balance.
        assert!(!run(&[0, 2, 2], &[0, 1, 2, 3], Some(vec![2, 0, 1, 0])));
    }

    #[test]
    fn logup_rejects_out_of_table_value() {
        // Query 5 is not in the table, yet the fingerprint cells are built as if it
        // were: the derived-table check is what catches it, since the public table
        // cannot contain 5.
        let mut cells = SegmentCells::new();
        let dq = 4usize;
        let dt = 4usize;
        let fq = vec![Fp::new(6), Fp::new(9), Fp::new(9), Fp::ZERO]; // 5, 2, 2
        let ft = public_ft(&[0, 1, 2, 3]);
        let m = vec![Fp::new(1), Fp::ZERO, Fp::new(2), Fp::ZERO];
        let inv = |v: &[Fp]| -> Vec<Fp> {
            v.iter()
                .map(|&x| if x == Fp::ZERO { Fp::ZERO } else { x.inverse() })
                .collect()
        };
        let diff_q = inv(&fq);
        let diff_t: Vec<Fp> = m
            .iter()
            .zip(&ft)
            .map(|(&mm, &tt)| {
                if tt == Fp::ZERO {
                    Fp::ZERO
                } else {
                    mm * tt.inverse()
                }
            })
            .collect();
        cells.cell_fp("fq", dq, 1, fq);
        cells.cell_fp("diff_q", dq, 1, diff_q);
        cells.cell_fp("ft", dt, 1, ft.clone());
        cells.cell_fp("diff_t", dt, 1, diff_t);
        cells.cell_fp("m", dt, 1, m);

        let mut seg = cells.seal(Params::default());
        let com = seg.commitment();
        let ids: Vec<usize> = ["fq", "diff_q", "ft", "diff_t", "m"]
            .iter()
            .map(|name| seg.id(name))
            .collect();
        let proof = claim_logup(
            &mut seg,
            (ids[0], ids[1]),
            (ids[2], ids[3], ids[4]),
            &ft,
            (3, 4),
            &mut Transcript::new(&new_challenger()),
        );
        let claims = seg.claims().to_vec();
        let wproof = seg.prove(&mut new_challenger());
        assert!(verify(&com, &Claims::new(&claims), &wproof, &mut new_challenger()));
        let mut vtr = Transcript::new(&new_challenger());
        assert!(!verify_logup(&com, &Claims::new(&claims), &proof, &ft, (3, 4), &mut vtr));
    }

    #[test]
    fn graph_lookup_roundtrip_and_soundness() {
        // f(x) = x*x over [0,3]; query [0,2,2].
        let (lo, hi) = (0i64, 3i64);
        let f = |x: i64| x * x;
        let alpha = Fp::new(11);
        let beta = Fp::new(5);
        let inp = [0i64, 2, 2];
        let out: Vec<i64> = inp.iter().map(|&x| f(x)).collect();
        let n = inp.len();
        let big_n = (hi - lo + 1) as usize;
        let dq = n.next_power_of_two();
        let dt = big_n.next_power_of_two();

        // Public table fingerprints, padding `1` (LogUp convention).
        let public_ft: Vec<Fp> = {
            let mut v: Vec<Fp> = (lo..=hi)
                .map(|x| {
                    let fx = Fp::new(f(x) as u64);
                    alpha + beta * Fp::new(x as u64) + beta * beta * fx
                })
                .collect();
            v.resize(dt, Fp::ONE);
            v
        };

        let build = |out_vals: &[i64], m_override: Option<[i64; 4]>| -> SegmentCells {
            let mut cells = SegmentCells::new();
            let mut in_fp = vec![Fp::ZERO; dq];
            let mut out_fp = vec![Fp::ZERO; dq];
            let mut fp = vec![Fp::ONE; dq];
            for i in 0..n {
                in_fp[i] = Fp::new(inp[i] as u64);
                out_fp[i] = Fp::new(out_vals[i] as u64);
                fp[i] = alpha + beta * in_fp[i] + beta * beta * out_fp[i];
            }
            let mut m_src = vec![0i64; big_n];
            match m_override {
                Some(over) => m_src.copy_from_slice(&over),
                None => {
                    for &x in &inp {
                        m_src[(x - lo) as usize] += 1;
                    }
                }
            }
            let mut m = vec![Fp::ONE; dt];
            for (j, &x) in m_src.iter().enumerate() {
                m[j] = Fp::new(x as u64);
            }
            let diff_q: Vec<Fp> = fp.iter().map(|&x| x.inverse()).collect();
            let diff_t: Vec<Fp> = m
                .iter()
                .zip(&public_ft)
                .map(|(&mm, &tt)| mm * tt.inverse())
                .collect();

            cells.cell_fp("in", dq, 1, in_fp);
            cells.cell_fp("out", dq, 1, out_fp);
            cells.cell_fp("fp", dq, 1, fp);
            cells.cell_fp("diff_q", dq, 1, diff_q);
            cells.cell_fp("ft", dt, 1, public_ft.clone());
            cells.cell_fp("diff_t", dt, 1, diff_t);
            cells.cell_fp("m", dt, 1, m);
            cells
        };

        let run = |cells: SegmentCells| -> bool {
            let mut seg = cells.seal(Params::default());
            let com = seg.commitment();
            let lc = Lookup2Cells {
                input: seg.id("in"),
                output: seg.id("out"),
                fp: seg.id("fp"),
                diff_q: seg.id("diff_q"),
                ft: seg.id("ft"),
                diff_t: seg.id("diff_t"),
                m: seg.id("m"),
            };
            let mut tr = Transcript::new(&new_challenger());
            let proof = claim_lookup2(&mut seg, lc, (alpha, beta), &public_ft, (n, big_n), &mut tr);
            let claims = seg.claims().to_vec();
            let wproof = seg.prove(&mut new_challenger());
            if !verify(&com, &Claims::new(&claims), &wproof, &mut new_challenger()) {
                return false;
            }
            let mut vtr = Transcript::new(&new_challenger());
            verify_lookup2(&com, &Claims::new(&claims), &proof, &public_ft, (n, big_n), &mut vtr)
        };

        assert!(run(build(&out, None)), "honest graph lookup verifies");

        // A pair that is not on the graph of `f` must be rejected.
        let mut bad_out = out.clone();
        bad_out[1] += 1;
        assert!(!run(build(&bad_out, None)), "a non-graph pair is rejected");

        // Wrong multiplicities break the LogUp balance.
        assert!(!run(build(&out, Some([2, 0, 1, 0]))));
    }

}
