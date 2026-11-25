//! Polynomial commitments: the stacked WHIR commitment layer, the FS transcript
//! it is driven by, and the linear-claim bridge between the two.

pub mod blake3_mmcs;
pub mod gpu_mmcs;
pub mod linear_claim;
pub mod metal_dft;
pub mod transcript;
pub mod whir_stacked;
