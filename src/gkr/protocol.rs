//! The sumcheck helpers the op-graph claims share.
//!
//! Everything here is commitment-agnostic: point evaluation of a matrix, the
//! degree-2 and degree-3 round polynomials the GKR reductions use, and the equality
//! partial evaluation. The FRI commitment plumbing that used to live beside them is
//! gone; the claims themselves are in `gkr::segment`, `gkr::logup`, `gkr::range`,
//! `gkr::requant` and `gkr::attention`.

use crate::field::sumcheck::{mle_eval, RoundPoly, RoundPoly3};
use crate::field::{Ext, Fp};
use p3_field::{Field, PrimeCharacteristicRing};

pub(crate) fn concat(a: &[Ext], b: &[Ext]) -> Vec<Ext> {
    let mut v = a.to_vec();
    v.extend_from_slice(b);
    v
}

/// MLE of a row-major matrix at `(row_point, col_point)`.
pub(crate) fn matrix_mle(flat: &[Ext], row_point: &[Ext], col_point: &[Ext]) -> Ext {
    mle_eval(flat, &concat(col_point, row_point))
}

fn interpolate_deg2(y0: Ext, y1: Ext, y2: Ext) -> RoundPoly {
    let two = Ext::from(Fp::new(2));
    let bpc = y1 - y0;
    let two_b_four_c = y2 - y0;
    let two_c = two_b_four_c - two * bpc;
    let c = two_c * two.inverse();
    let b = bpc - c;
    RoundPoly([y0, b, c])
}

/// The GKR round polynomial for `Σ_k p(k)·q(k)`: at variable `i` it is
/// `g_i(t) = Σ_{k_{>i}} p̃(r_<i, t, k_>i) · q̃(r_<i, t, k_>i)`.
pub(crate) fn round_poly_product(p: &[Ext], q: &[Ext], lm: usize, i: usize, r_prefix: &[Ext]) -> RoundPoly {
    let suffix = lm - i - 1;
    let mut y = [Ext::ZERO; 3];
    let samples = [Ext::ZERO, Ext::ONE, Ext::from(Fp::new(2))];
    for (idx, t) in samples.iter().enumerate() {
        let mut acc = Ext::ZERO;
        for sb in 0..(1 << suffix) {
            let mut point = vec![Ext::ZERO; lm];
            point[..i].copy_from_slice(r_prefix);
            point[i] = *t;
            for b in 0..suffix {
                point[i + 1 + b] = if (sb >> b) & 1 == 1 {
                    Ext::ONE
                } else {
                    Ext::ZERO
                };
            }
            acc += mle_eval(p, &point) * mle_eval(q, &point);
        }
        y[idx] = acc;
    }
    interpolate_deg2(y[0], y[1], y[2])
}

fn interpolate_deg3(y0: Ext, y1: Ext, y2: Ext, y3: Ext) -> RoundPoly3 {
    let two = Ext::from(Fp::new(2));
    let three = Ext::from(Fp::new(3));
    let four = Ext::from(Fp::new(4));
    let six = Ext::from(Fp::new(6));
    let a = y0;
    let d1 = y1 - y0;
    let d2 = y2 - y0;
    let d3 = y3 - y0;
    let a_const = (d2 - two * d1) * two.inverse();
    let b_const = (d3 - three * d1) * six.inverse();
    let d = b_const - a_const;
    let c = four * a_const - three * b_const;
    let b = d1 - c - d;
    RoundPoly3([a, b, c, d])
}

/// `eq(r, point)` for an *arbitrary* point (not just Boolean).
pub(crate) fn eq_partial(r: &[Ext], point: &[Ext]) -> Ext {
    let mut acc = Ext::ONE;
    for k in 0..r.len() {
        let p = point[k];
        acc = acc * (r[k] * p + (Ext::ONE - r[k]) * (Ext::ONE - p));
    }
    acc
}

/// Degree-3 GKR round polynomial for a pointwise product `Σ_i eq(r,i)·a[i]·b[i]`:
/// `g_j(t) = Σ_{i_>j} eq(r, r_<j, t, i_>)·ã(r_<j, t, i_>)·b̃(r_<j, t, i_>)`.
pub(crate) fn round_poly3_mul(
    a: &[Ext],
    b: &[Ext],
    r: &[Ext],
    n: usize,
    j: usize,
    r_prefix: &[Ext],
) -> RoundPoly3 {
    let suffix = n - j - 1;
    let samples = [
        Ext::ZERO,
        Ext::ONE,
        Ext::from(Fp::new(2)),
        Ext::from(Fp::new(3)),
    ];
    let mut y = [Ext::ZERO; 4];
    for (idx, t) in samples.iter().enumerate() {
        let mut acc = Ext::ZERO;
        for sb in 0..(1 << suffix) {
            let mut point = vec![Ext::ZERO; n];
            point[..j].copy_from_slice(r_prefix);
            point[j] = *t;
            for b in 0..suffix {
                point[j + 1 + b] = if (sb >> b) & 1 == 1 {
                    Ext::ONE
                } else {
                    Ext::ZERO
                };
            }
            acc += eq_partial(r, &point) * mle_eval(a, &point) * mle_eval(b, &point);
        }
        y[idx] = acc;
    }
    interpolate_deg3(y[0], y[1], y[2], y[3])
}
