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
pub(crate) fn round_poly_product(
    p: &[Ext],
    q: &[Ext],
    lm: usize,
    i: usize,
    r_prefix: &[Ext],
) -> RoundPoly {
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
    let head = j + 1;
    debug_assert_eq!(a.len(), 1 << n);
    debug_assert_eq!(b.len(), 1 << n);
    debug_assert_eq!(r.len(), n);
    debug_assert_eq!(r_prefix.len(), j);

    // Variables already fixed: their `eq(r_k, ·)` factors are constants.
    let mut c_pref = Ext::ONE;
    for (k, &p) in r_prefix.iter().enumerate() {
        c_pref *= r[k] * p + (Ext::ONE - r[k]) * (Ext::ONE - p);
    }
    // `eq(r_j, t)` is linear: its two ends are `1 − r_j` and `r_j`.
    let (e0, e1) = (Ext::ONE - r[j], r[j]);

    // The free (suffix) variables contribute `Π_{k>j} eq(r_k, bit_k)` per Boolean point.
    let d = 1usize << suffix;
    let mut pe = vec![Ext::ONE; d];
    for (k, &rk) in r.iter().enumerate().skip(j + 1) {
        let bit = k - (j + 1);
        for (sb, p) in pe.iter_mut().enumerate() {
            *p *= if (sb >> bit) & 1 == 1 {
                rk
            } else {
                Ext::ONE - rk
            };
        }
    }

    // Partial MLEs of `a` and `b` over the suffix, with the first `head` coordinates
    // fixed at `t = 0` and `t = 1`. Both are linear in `t`, so two ends suffice.
    let mut pa0 = vec![Ext::ZERO; d];
    let mut pa1 = vec![Ext::ZERO; d];
    let mut pb0 = vec![Ext::ZERO; d];
    let mut pb1 = vec![Ext::ZERO; d];
    for h in 0..(1usize << head) {
        let mut w = Ext::ONE;
        for k in 0..j {
            w *= if (h >> k) & 1 == 1 {
                r_prefix[k]
            } else {
                Ext::ONE - r_prefix[k]
            };
        }
        let (ta, tb) = if (h >> j) & 1 == 1 {
            (&mut pa1, &mut pb1)
        } else {
            (&mut pa0, &mut pb0)
        };
        for sb in 0..d {
            // LSB-first: the fixed variables are the low `head` bits, the suffix the high.
            let idx = h + (sb << head);
            ta[sb] += w * a[idx];
            tb[sb] += w * b[idx];
        }
    }

    // `g(t) = c_pref · eq(r_j, t) · Σ_sb pe[sb] · ã(t, sb) · b̃(t, sb)`.
    let ev = |t: Ext| -> Ext {
        let mut acc = Ext::ZERO;
        for sb in 0..d {
            let av = pa0[sb] + t * (pa1[sb] - pa0[sb]);
            let bv = pb0[sb] + t * (pb1[sb] - pb0[sb]);
            acc += pe[sb] * av * bv;
        }
        c_pref * (e0 + t * (e1 - e0)) * acc
    };
    interpolate_deg3(
        ev(Ext::ZERO),
        ev(Ext::ONE),
        ev(Ext::from(Fp::new(2))),
        ev(Ext::from(Fp::new(3))),
    )
}

#[cfg(test)]
mod round_poly3_tests {
    use super::*;
    use p3_field::PrimeCharacteristicRing;

    /// The slow definition, kept to check the folding implementation.
    fn slow(a: &[Ext], b: &[Ext], r: &[Ext], n: usize, j: usize, prefix: &[Ext]) -> RoundPoly3 {
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
                point[..j].copy_from_slice(prefix);
                point[j] = *t;
                for bb in 0..suffix {
                    point[j + 1 + bb] = if (sb >> bb) & 1 == 1 { Ext::ONE } else { Ext::ZERO };
                }
                acc += eq_partial(r, &point) * mle_eval(a, &point) * mle_eval(b, &point);
            }
            y[idx] = acc;
        }
        interpolate_deg3(y[0], y[1], y[2], y[3])
    }

    #[test]
    fn folding_matches_the_definition() {
        let n = 3usize;
        let a: Vec<Ext> = (0..8u64).map(|i| Ext::from(Fp::new(i + 1))).collect();
        let b: Vec<Ext> = (0..8u64).map(|i| Ext::from(Fp::new(3 * i + 2))).collect();
        let r: Vec<Ext> = (0..n).map(|i| Ext::from(Fp::new(7 + i as u64))).collect();
        for j in 0..n {
            let pre: Vec<Ext> = (0..j).map(|i| Ext::from(Fp::new(11 + i as u64))).collect();
            let pre = &pre[..];
            assert_eq!(
                round_poly3_mul(&a, &b, &r, n, j, pre),
                slow(&a, &b, &r, n, j, pre),
                "round {j}"
            );
        }
    }
}
