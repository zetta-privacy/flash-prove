//! The GKR layer: the op-graph segment driver, the per-op claims, and the
//! commitment plumbing they share.

pub mod attention;
pub mod commit;
pub mod driver;
pub mod head;
pub mod logup;
pub mod nonlinear;
pub mod plan;
pub mod protocol;
pub mod range;
pub mod requant;
pub mod segment;
pub mod selection;
