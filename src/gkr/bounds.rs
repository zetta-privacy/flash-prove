//! Public lookup/range-check bounds.
//!
//! The fixed function tables (`requant`, `silu`, `exp`, `round`, `sign`) are
//! committed once over `[lo, hi]` and checked against the trace via the LogUp
//! argument. These bounds are public: they travel in
//! [`crate::gkr::driver::LookupBounds`], so the verifier never re-derives them
//! from the trace.

use std::collections::HashMap;

use crate::model::spec::{ModelSpec, Op};
use crate::model::Trace;

/// The lookup bounds. Fixed function tables (requant/silu/exp/round/sign) are
/// committed once over `[lo, hi]` and checked via the succinct LogUp.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CommittedConfig {
    pub requant_digits: usize,
    pub silu_lo: i64,
    pub silu_hi: i64,
    pub exp_lo: i64,
    pub sign_lo: i64,
    pub sign_hi: i64,
    /// Bounds of the logits (the score domain of the argmax range check).
    pub score_lo: i64,
    pub score_hi: i64,
}

impl CommittedConfig {
    /// Derive the lookup/range-check bounds from the actual trace, so the fixed
    /// function tables cover exactly the values this model produces.
    ///
    /// * `silu_lo/hi` — the raw gate-projection inputs to `silu` (unclamped).
    /// * `exp_lo` — the most negative `score − max` seen by causal softmax.
    /// * `sign_lo/hi` — the MoE routing delta `score·experts − i − tau`.
    /// * `score_lo/hi` — the head logits range.
    ///
    /// Both prover and verifier build the same tables from these (public) bounds, so
    /// a bound that fails to cover the trace only ever breaks the prover's own proof.
    pub fn derive(spec: &ModelSpec, trace: &Trace) -> Self {
        let cells: HashMap<&str, &crate::model::Cell> = trace
            .segments
            .iter()
            .flat_map(|s| s.cells.iter())
            .map(|c| (c.name.as_str(), c))
            .collect();
        Self::derive_with(spec, |name| cells.get(name).map(|c| (*c).clone()))
    }

    /// [`Self::derive`] over a cell *provider*, so the prover can read only the cells the
    /// bounds actually look at instead of holding the whole trace.
    pub fn derive_with(spec: &ModelSpec, get: impl Fn(&str) -> Option<crate::model::Cell>) -> Self {

        let mut silu_lo = i64::MAX;
        let mut silu_hi = i64::MIN;
        let mut exp_lo = 0i64;
        let mut sign_lo = i64::MAX;
        let mut sign_hi = i64::MIN;
        let mut score_lo = i64::MAX;
        let mut score_hi = i64::MIN;

        if let Some(c) = get("logits") {
            for &v in &c.values {
                score_lo = score_lo.min(v);
                score_hi = score_hi.max(v);
            }
        }

        for op in &spec.ops {
            match op {
                Op::Silu { input, .. } => {
                    if let Some(c) = get(input.as_str()) {
                        for &v in &c.values {
                            silu_lo = silu_lo.min(v);
                            silu_hi = silu_hi.max(v);
                        }
                    }
                }
                Op::CausalSoftmax { input, .. } => {
                    if let Some(c) = get(input.as_str()) {
                        let (mut mn, mut mx) = (i64::MAX, i64::MIN);
                        for &v in &c.values {
                            mn = mn.min(v);
                            mx = mx.max(v);
                        }
                        exp_lo = exp_lo.min(mn - mx);
                    }
                }
                Op::TopK { input, .. } => {
                    if let Some(c) = get(input.as_str()) {
                        let (mut mn, mut mx) = (i64::MAX, i64::MIN);
                        for &v in &c.values {
                            mn = mn.min(v);
                            mx = mx.max(v);
                        }
                        let e = c.cols as i64;
                        sign_lo = sign_lo.min(-(mx - mn) * e - e);
                        sign_hi = sign_hi.max((mx - mn) * e + e);
                    }
                }
                _ => {}
            }
        }

        // Fallbacks for models that omit a nonlinearity (empty table domain).
        if silu_hi < silu_lo {
            silu_lo = -1;
            silu_hi = 1;
        }
        if sign_hi < sign_lo {
            sign_lo = -1;
            sign_hi = 1;
        }
        if score_hi < score_lo {
            score_lo = -1;
            score_hi = 1;
        }

        Self {
            requant_digits: 4,
            silu_lo,
            silu_hi,
            exp_lo,
            sign_lo,
            sign_hi,
            score_lo,
            score_hi,
        }
    }
}
