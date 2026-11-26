//! Multilinear extension evaluation and the round-polynomial types used by GKR.

use crate::field::{Ext, Fp};
use p3_field::PrimeCharacteristicRing;

/// `Σ_j eq(point, j)·acc[j]`, folding one variable at a time: two extension-field
/// multiplications per entry instead of `log n` of them.
///
/// The naive form evaluates `eq(point, j)` from scratch for every `j`, which is `O(n·m)`
/// — the single largest arithmetic cost in the prover's openings, which run per claim.
fn fold(acc: &mut Vec<Ext>, point: &[Ext]) -> Ext {
    for &z in point.iter().rev() {
        let half = acc.len() / 2;
        let not_z = Ext::ONE - z;
        for i in 0..half {
            acc[i] = acc[i] * not_z + acc[i + half] * z;
        }
        acc.truncate(half);
    }
    acc.first().copied().unwrap_or(Ext::ZERO)
}

/// The MLE of a base-field vector at a field point (prover-side; the prover holds
/// `v` in the clear).
pub fn mle_value(v: &[Fp], point: &[Ext]) -> Ext {
    debug_assert_eq!(v.len(), 1 << point.len());
    let mut acc: Vec<Ext> = v.iter().map(|&x| Ext::from(x)).collect();
    fold(&mut acc, point)
}

/// Evaluate the multilinear extension of `f` (given by its `2^n` evaluations,
/// indexed by the Boolean point of `j`) at an arbitrary point `point ∈ E^n`.
pub fn mle_eval(evals: &[Ext], point: &[Ext]) -> Ext {
    debug_assert_eq!(evals.len(), 1 << point.len());
    let mut acc = evals.to_vec();
    fold(&mut acc, point)
}

/// [`mle_value`] for a point of any arity: zeros beyond `v`'s own cube.
pub fn mle_value_padded(v: &[Fp], point: &[Ext]) -> Ext {
    let mut acc: Vec<Ext> = v.iter().map(|&x| Ext::from(x)).collect();
    if acc.is_empty() {
        return Ext::ZERO;
    }
    // `point` may be longer than `v`'s cube: the extra coordinates multiply by (1 − z)
    // because the missing entries are zero.
    let b = acc.len().trailing_zeros() as usize;
    let mut acc_out = Ext::ZERO;
    if point.len() > b {
        let mut scaled = Ext::ONE;
        for &z in &point[b..] {
            scaled *= Ext::ONE - z;
        }
        acc_out = scaled;
    }
    let v = fold(&mut acc, &point[..b.min(point.len())]);
    if point.len() > b {
        v * acc_out
    } else {
        v
    }
}

/// A degree-≤2 univariate round polynomial `a + b·t + c·t²`.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RoundPoly(pub [Ext; 3]);

impl RoundPoly {
    pub fn eval(&self, t: Ext) -> Ext {
        self.0[0] + self.0[1] * t + self.0[2] * t * t
    }
}

/// A degree-≤3 univariate round polynomial `a + b·t + c·t² + d·t³`.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RoundPoly3(pub [Ext; 4]);

impl RoundPoly3 {
    pub fn eval(&self, t: Ext) -> Ext {
        let [a, b, c, d] = self.0;
        a + b * t + c * t * t + d * t * t * t
    }
}
