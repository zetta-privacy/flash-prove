//! Sumcheck: the reduction at the heart of the succinct verifier.
//!
//! It turns a claim "Σ_{x∈{0,1}^n} f(x) = S" (a sum over the Boolean cube) into a
//! claim "f(r) = v" at a single random point `r`, via `n` rounds in which the
//! prover sends a degree-≤2 univariate round polynomial and the verifier samples
//! one field element per variable. This is what lets the arithmetization
//! (matmuls, norms, softmax, …) be checked at O(log n) points instead of being
//! recomputed.

use crate::field::Ext;
use crate::protocol::eq;
use p3_field::{Field, PrimeCharacteristicRing};

/// Evaluate the multilinear extension of `f` (given by its `2^n` evaluations,
/// indexed by the Boolean point of `j`) at an arbitrary point `point ∈ E^n`.
pub fn mle_eval(evals: &[Ext], point: &[Ext]) -> Ext {
    debug_assert_eq!(evals.len(), 1 << point.len());
    let mut acc = Ext::ZERO;
    for (j, &e) in evals.iter().enumerate() {
        acc = acc + e * eq(point, j);
    }
    acc
}

/// A degree-≤2 univariate round polynomial `a + b·t + c·t²`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoundPoly(pub [Ext; 3]);

impl RoundPoly {
    pub fn eval(&self, t: Ext) -> Ext {
        self.0[0] + self.0[1] * t + self.0[2] * t * t
    }
}

/// The prover's round polynomial for variable `i`:
///
/// `g_i(t) = Σ_{x_{i+1..n}} f(r_0, …, r_{i-1}, t, x_{i+1}, …, x_{n-1})`.
///
/// Computed by evaluating the partially-specified points directly (O(2^n · 2^{n−i}),
/// fine for small `n`; the real prover computes it in O(2^{n−i}) via the bookkeeping).
pub fn round_poly(evals: &[Ext], n: usize, i: usize, r: &[Ext]) -> RoundPoly {
    let suffix = n - i - 1;
    let mut g = [Ext::ZERO; 3];
    // Evaluate at t = 0, 1, 2 and interpolate a degree-≤2 polynomial.
    for (k, t) in [Ext::ZERO, Ext::ONE, Ext::from(crate::field::from_u64(2))]
        .iter()
        .enumerate()
    {
        let mut acc = Ext::ZERO;
        for sb in 0..(1 << suffix) {
            let mut point = vec![Ext::ZERO; n];
            point[..i].copy_from_slice(&r[..i]);
            point[i] = *t;
            for (b, slot) in point.iter_mut().skip(i + 1).enumerate() {
                *slot = if (sb >> b) & 1 == 1 {
                    Ext::ONE
                } else {
                    Ext::ZERO
                };
            }
            acc = acc + mle_eval(evals, &point);
        }
        g[k] = acc;
    }
    interpolate_deg2(g[0], g[1], g[2])
}

/// A degree-≤3 univariate round polynomial `a + b·t + c·t² + d·t³`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoundPoly3(pub [Ext; 4]);

impl RoundPoly3 {
    pub fn eval(&self, t: Ext) -> Ext {
        let [a, b, c, d] = self.0;
        a + b * t + c * t * t + d * t * t * t
    }
}

/// Interpolate a degree-≤3 polynomial through `(0..=3, y0..=y3)`, returning
/// `[a, b, c, d]` for `a + b·t + c·t² + d·t³`.
fn interpolate_deg3(y0: Ext, y1: Ext, y2: Ext, y3: Ext) -> RoundPoly3 {
    use crate::field::from_u64;
    let two = Ext::from(from_u64(2));
    let three = Ext::from(from_u64(3));
    let four = Ext::from(from_u64(4));
    let six = Ext::from(from_u64(6));
    let a = y0;
    let d1 = y1 - y0;
    let d2 = y2 - y0;
    let d3 = y3 - y0;
    // c + 3d = (d2 − 2·d1)/2 =: A
    let a_const = (d2 - two * d1) * two.inverse();
    // c + 4d = (d3 − 3·d1)/6 =: B
    let b_const = (d3 - three * d1) * six.inverse();
    // d = B − A, c = A − 3d = 4A − 3B, b = d1 − c − d.
    let d = b_const - a_const;
    let c = four * a_const - three * b_const;
    let b = d1 - c - d;
    RoundPoly3([a, b, c, d])
}

/// The prover's degree-≤3 round polynomial for variable `i` (for product checks
/// like `eq·a·b`, which are cubic per variable).
pub fn round_poly3(evals: &[Ext], n: usize, i: usize, r: &[Ext]) -> RoundPoly3 {
    let suffix = n - i - 1;
    let mut g = [Ext::ZERO; 4];
    let samples = [
        Ext::ZERO,
        Ext::ONE,
        Ext::from(crate::field::from_u64(2)),
        Ext::from(crate::field::from_u64(3)),
    ];
    for (k, t) in samples.iter().enumerate() {
        let mut acc = Ext::ZERO;
        for sb in 0..(1 << suffix) {
            let mut point = vec![Ext::ZERO; n];
            point[..i].copy_from_slice(&r[..i]);
            point[i] = *t;
            for (b, slot) in point.iter_mut().skip(i + 1).enumerate() {
                *slot = if (sb >> b) & 1 == 1 {
                    Ext::ONE
                } else {
                    Ext::ZERO
                };
            }
            acc = acc + mle_eval(evals, &point);
        }
        g[k] = acc;
    }
    interpolate_deg3(g[0], g[1], g[2], g[3])
}

/// Degree-≤3 prover: produce the `n` round polynomials.
pub fn prove3(
    evals: &[Ext],
    n: usize,
    mut sample: impl FnMut(usize) -> Ext,
) -> (Vec<RoundPoly3>, Ext, Vec<Ext>) {
    let mut claim = evals.iter().fold(Ext::ZERO, |a, &b| a + b);
    let mut r = Vec::with_capacity(n);
    let mut rounds = Vec::with_capacity(n);
    for i in 0..n {
        let g = round_poly3(evals, n, i, &r);
        rounds.push(g);
        let ri = sample(i);
        r.push(ri);
        claim = g.eval(ri);
    }
    (rounds, claim, r)
}

/// Degree-≤3 verifier.
pub fn verify3(
    evals: &[Ext],
    n: usize,
    rounds: &[RoundPoly3],
    r: &[Ext],
    claimed_sum: Ext,
) -> bool {
    let mut claim = claimed_sum;
    for i in 0..n {
        let g = rounds[i];
        if g.eval(Ext::ZERO) + g.eval(Ext::ONE) != claim {
            return false;
        }
        claim = g.eval(r[i]);
    }
    claim == mle_eval(evals, r)
}

/// Interpolate a degree-≤2 polynomial through `(0, y0)`, `(1, y1)`, `(2, y2)`,
/// returning `[a, b, c]` for `a + b·t + c·t²`.
fn interpolate_deg2(y0: Ext, y1: Ext, y2: Ext) -> RoundPoly {
    // a = y0;  b + c = y1 − y0;  2b + 4c = y2 − y0.
    // 2·(y1−y0) − (y2−y0) = −2c  →  c = −(2(y1−y0) − (y2−y0)) / 2.
    let two = Ext::from(crate::field::from_u64(2));
    let bpc = y1 - y0;
    let two_b_four_c = y2 - y0;
    // two_b_four_c − 2·bpc = 2c
    let two_c = two_b_four_c - two * bpc;
    let c = two_c * two.inverse();
    let b = bpc - c;
    let a = y0;
    RoundPoly([a, b, c])
}

/// Run the prover: given the `2^n` evaluations and a claimed sum, produce the
/// `n` round polynomials. `rng_sample` supplies the verifier's challenge per round
/// (injecting it keeps this deterministic for tests).
pub fn prove(
    evals: &[Ext],
    n: usize,
    mut sample: impl FnMut(usize) -> Ext,
) -> (Vec<RoundPoly>, Ext, Vec<Ext>) {
    let mut claim = evals.iter().fold(Ext::ZERO, |a, &b| a + b);
    let mut r = Vec::with_capacity(n);
    let mut rounds = Vec::with_capacity(n);
    for i in 0..n {
        let g = round_poly(evals, n, i, &r);
        rounds.push(g);
        let ri = sample(i);
        r.push(ri);
        claim = g.eval(ri);
    }
    (rounds, claim, r)
}

/// Run the verifier: check `g_i(0) + g_i(1) == claim` each round and the final
/// MLE evaluation matches. Returns `true` iff the claim is honest.
pub fn verify(evals: &[Ext], n: usize, rounds: &[RoundPoly], r: &[Ext], claimed_sum: Ext) -> bool {
    let mut claim = claimed_sum;
    for i in 0..n {
        let g = rounds[i];
        if g.eval(Ext::ZERO) + g.eval(Ext::ONE) != claim {
            return false;
        }
        claim = g.eval(r[i]);
    }
    claim == mle_eval(evals, r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::from_u64;

    #[test]
    fn mle_eval_matches_boolean_points() {
        let evals = vec![
            Ext::from(from_u64(1)),
            Ext::from(from_u64(2)),
            Ext::from(from_u64(3)),
            Ext::from(from_u64(4)),
        ];
        assert_eq!(mle_eval(&evals, &[Ext::ZERO, Ext::ZERO]), evals[0]);
        assert_eq!(mle_eval(&evals, &[Ext::ONE, Ext::ZERO]), evals[1]);
        assert_eq!(mle_eval(&evals, &[Ext::ZERO, Ext::ONE]), evals[2]);
        assert_eq!(mle_eval(&evals, &[Ext::ONE, Ext::ONE]), evals[3]);
    }

    #[test]
    fn sumcheck_roundtrip_honest() {
        // 3 variables → 8 evaluations of a random multilinear polynomial.
        let evals: Vec<Ext> = (0..8).map(|i| Ext::from(from_u64(i as u64 + 1))).collect();
        let n = 3;
        // deterministic challenges: r_i = i + 3.
        let sample = |i: usize| Ext::from(from_u64(i as u64 + 3));
        let (rounds, _final_claim, r) = prove(&evals, n, sample);
        let claimed_sum = evals.iter().fold(Ext::ZERO, |a, &b| a + b);
        assert!(verify(&evals, n, &rounds, &r, claimed_sum));
    }

    #[test]
    fn sumcheck_rejects_tampered_round() {
        let evals: Vec<Ext> = (0..8).map(|i| Ext::from(from_u64(i as u64 + 1))).collect();
        let n = 3;
        let sample = |i: usize| Ext::from(from_u64(i as u64 + 3));
        let (mut rounds, _final_claim, r) = prove(&evals, n, sample);
        // Tamper with the last round polynomial.
        rounds[2].0[0] = Ext::ZERO;
        let claimed_sum = evals.iter().fold(Ext::ZERO, |a, &b| a + b);
        assert!(!verify(&evals, n, &rounds, &r, claimed_sum));
    }

    #[test]
    fn sumcheck3_roundtrip_honest() {
        // A genuine cubic per-variable function: f(x) = x_0·x_1·x_2 over the cube
        // (the product of three multilinear factors, the shape that `mul` needs).
        let n = 3;
        let evals: Vec<Ext> = (0..8)
            .map(|j| {
                let b0 = Ext::from(from_u64((j & 1) as u64));
                let b1 = Ext::from(from_u64(((j >> 1) & 1) as u64));
                let b2 = Ext::from(from_u64(((j >> 2) & 1) as u64));
                b0 * b1 * b2
            })
            .collect();
        let sample = |i: usize| Ext::from(from_u64(i as u64 + 5));
        let (rounds, _final_claim, r) = prove3(&evals, n, sample);
        let claimed_sum = evals.iter().fold(Ext::ZERO, |a, &b| a + b);
        assert!(verify3(&evals, n, &rounds, &r, claimed_sum));
    }

    #[test]
    fn sumcheck3_rejects_tampered_round() {
        let n = 3;
        let evals: Vec<Ext> = (0..8)
            .map(|j| {
                let b0 = Ext::from(from_u64((j & 1) as u64));
                let b1 = Ext::from(from_u64(((j >> 1) & 1) as u64));
                let b2 = Ext::from(from_u64(((j >> 2) & 1) as u64));
                b0 * b1 * b2
            })
            .collect();
        let sample = |i: usize| Ext::from(from_u64(i as u64 + 5));
        let (mut rounds, _final_claim, r) = prove3(&evals, n, sample);
        // The last round of `x0·x1·x2` is `r0·r1·t`, so its linear coefficient is
        // nonzero; zero it to force a mismatch.
        rounds[2].0[1] = Ext::ZERO;
        let claimed_sum = evals.iter().fold(Ext::ZERO, |a, &b| a + b);
        assert!(!verify3(&evals, n, &rounds, &r, claimed_sum));
    }
}
