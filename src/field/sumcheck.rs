//! Multilinear extension evaluation and the round-polynomial types used by GKR.

use crate::field::{eq, Ext, Fp};
use p3_field::PrimeCharacteristicRing;

/// The MLE of a base-field vector at a field point (prover-side; the prover holds
/// `v` in the clear).
pub fn mle_value(v: &[Fp], point: &[Ext]) -> Ext {
    debug_assert_eq!(v.len(), 1 << point.len());
    v.iter()
        .enumerate()
        .map(|(j, &vj)| eq(point, j) * Ext::from(vj))
        .sum()
}

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
