//! Lookup / table / nonlinearity primitives (LogUp, requant, range checks, and
//! the per-op committed proofs).

pub mod logup;
pub mod lookup_committed;
pub mod lookup_logup;
pub mod mul_committed;
pub mod norm_rope_committed;
pub mod range_check;
pub mod requant;
pub mod selection_committed;
