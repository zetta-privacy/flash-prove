//! Fiat–Shamir transcript for the succinct verifier: Goldilocks as the base
//! field, its degree-3 extension as the challenge field, the Poseidon2 duplex
//! challenger, and the `Transcript` wrapper the prover and verifier drive in
//! lockstep.
//!
//! Prover and verifier each clone the same base [`Challenger`] and absorb inputs
//! in the same order, so their sampled challenges agree. This is what binds the
//! committed data to the challenge points/indices that the linear claim, the
//! LogUp multiplicity, and the WHIR openings use.

use p3_challenger::{CanObserve, DuplexChallenger, FieldChallenger};
use p3_goldilocks::Poseidon2Goldilocks;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Challenge field (degree-3 extension of Goldilocks).
pub use crate::field::Ext as Challenge;
/// Base field (Goldilocks).
pub use crate::field::Fp as Val;

type Perm = Poseidon2Goldilocks<16>;

/// The challenger that drives Fiat–Shamir.
pub type Challenger = DuplexChallenger<Val, Perm, 16, 8>;

/// A Fiat–Shamir transcript over the succinct verifier's inputs.
#[derive(Clone)]
pub struct Transcript {
    ch: Challenger,
}

impl Transcript {
    pub fn new(challenger: &Challenger) -> Self {
        Transcript {
            ch: challenger.clone(),
        }
    }

    /// Absorb an input (e.g. a commitment) into the transcript.
    pub fn observe<C: Clone>(&mut self, c: &C)
    where
        Challenger: CanObserve<C>,
    {
        self.ch.observe(c.clone());
    }

    /// Absorb a base-field element (e.g. an IPA round value `L`/`R`).
    pub fn observe_val(&mut self, v: Val) {
        self.ch.observe(v);
    }

    /// Absorb a challenge-field (extension) element, e.g. a sumcheck round
    /// polynomial coefficient (decomposed into its base-field coordinates).
    pub fn observe_ext(&mut self, e: Challenge) {
        let coeffs =
            <Challenge as p3_field::BasedVectorSpace<Val>>::as_basis_coefficients_slice(&e);
        for &c in coeffs {
            self.ch.observe(c);
        }
    }

    /// Sample a challenge-field element.
    pub fn sample_ext(&mut self) -> Challenge {
        self.ch.sample_algebra_element()
    }

    /// Sample a base-field element (an IPA folding challenge `x_j`).
    pub fn sample_val(&mut self) -> Val {
        self.ch.sample_algebra_element()
    }

    }

/// Build a fresh challenger for the verifier, deterministically identical to the
/// one the prover builds (same seed). The verifier needs this so it can absorb
/// its own Fiat–Shamir inputs (e.g. the head token) without inheriting the
/// prover's mutated challenger state.
pub fn new_challenger() -> Challenger {
    let mut rng = StdRng::seed_from_u64(0);
    let perm = Perm::new_from_rng_128(&mut rng);
    Challenger::new(perm)
}
