//! Plonky3 FRI polynomial commitment (TwoAdicFriPcs) wrapper for the succinct
//! verifier. This replaces the hand-rolled masked Reed–Solomon + salted-Merkle
//! commitment (`encode_table` + `decode_exact`) with a FRI-based PCS, so the
//! proximity test is O(log n) in the verifier instead of a full IDFT.
//!
//! Challenge field: `CubicTrinomialExtensionField<Goldilocks>` = F[x]/(x³−x−1),
//! Plonky3's degree-3 extension of Goldilocks. (The paper's E = F[ϑ]/(ϑ³−2) is
//! an isomorphic-but-different degree-3 extension; the OOD/`eq` code still uses
//! `crate::field::Ext` and will be reconciled at the boundary.)

use p3_challenger::{CanObserve, CanSampleBits, DuplexChallenger, FieldChallenger};
use p3_commit::{ExtensionMmcs, Pcs as PcsTrait};
use p3_dft::Radix2DitParallel;
use p3_field::extension::CubicTrinomialExtensionField;
use p3_field::PrimeField64;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Base field (Goldilocks).
pub type Val = Goldilocks;
/// Challenge field (degree-3 extension).
pub type Challenge = CubicTrinomialExtensionField<Goldilocks>;

type Perm = Poseidon2Goldilocks<16>;
type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
type Compress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs = MerkleTreeMmcs<Val, Val, Hash, Compress, 2, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Dft = Radix2DitParallel<Val>;

/// The challenger drives Fiat–Shamir (opening points, folding challenges).
pub type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
/// The FRI PCS itself.
pub type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;

/// A commitment plus the prover's retained data for opening.
pub type Commitment = <Pcs as PcsTrait<Challenge, Challenger>>::Commitment;
/// Prover data for opening a commitment.
pub type ProverData = <Pcs as PcsTrait<Challenge, Challenger>>::ProverData;
/// A FRI opening proof.
pub type Proof = <Pcs as PcsTrait<Challenge, Challenger>>::Proof;
/// An evaluation domain.
pub type Domain = <Pcs as PcsTrait<Challenge, Challenger>>::Domain;

/// A Fiat–Shamir transcript over the succinct verifier's commitments. The prover
/// and verifier each clone the same base [`Challenger`] and absorb in the same
/// order, so their sampled challenges agree. This is what binds the committed
/// data to the challenge points/indices the linear claim and IPA use.
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

    /// Absorb a commitment into the transcript.
    pub fn observe(&mut self, c: &Commitment) {
        self.ch.observe(c.clone());
    }

    /// Absorb a base-field element (e.g. an IPA round value `L`/`R`).
    pub fn observe_val(&mut self, v: Val) {
        self.ch.observe(v);
    }

    /// Sample a challenge-field element (the linear-claim point `α`).
    pub fn sample_ext(&mut self) -> Challenge {
        self.ch.sample_algebra_element()
    }

    /// Sample a base-field element (an IPA folding challenge `x_j`).
    pub fn sample_val(&mut self) -> Val {
        self.ch.sample_algebra_element()
    }

    /// Sample a uniform coset index in `[0, n)` (`n` must be a power of two).
    pub fn sample_index(&mut self, n: usize) -> usize {
        self.ch.sample_bits(n.trailing_zeros() as usize)
    }
}

/// Build a deterministic PCS + challenger for `log_blowup` and `num_queries`.
///
/// These parameters are far below the paper's (blowup 1, 10 queries) — chosen
/// for a fast first integration; they must be raised to the security target
/// before this is used for real soundness.
pub fn new_pcs(log_blowup: usize, num_queries: usize) -> (Pcs, Challenger) {
    let mut rng = StdRng::seed_from_u64(0);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = Hash::new(perm.clone());
    let compress = Compress::new(perm.clone());

    let val_mmcs = ValMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    let fri_params = FriParameters {
        log_blowup,
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries,
        batch_proof_of_work_bits: 0,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };

    let pcs = Pcs::new(Dft::default(), val_mmcs, fri_params);
    (pcs, Challenger::new(perm))
}

/// Commit a single flat vector of base-field elements as one polynomial.
///
/// `vals` has length `n` (a power of two); the PCS commits the polynomial whose
/// evaluations over a size-`n` coset are `vals`.
pub fn commit_vec(pcs: &Pcs, vals: &[Val]) -> (Commitment, ProverData, Domain) {
    let n = vals.len();
    let domain = <Pcs as PcsTrait<Challenge, Challenger>>::natural_domain_for_degree(pcs, n);
    let matrix = RowMajorMatrix::new(vals.to_vec(), 1);
    let (commitment, prover_data) =
        <Pcs as PcsTrait<Challenge, Challenger>>::commit(pcs, vec![(domain, matrix)])
            .expect("commit");
    (commitment, prover_data, domain)
}

/// Convert signed base-2^8 digits (`i8`) to base-field elements.
pub fn digits_to_field(digits: &[i8]) -> Vec<Val> {
    digits
        .iter()
        .map(|&d| {
            if d >= 0 {
                Val::new(d as u64)
            } else {
                Val::new(Val::ORDER_U64 - (d.unsigned_abs() as u64))
            }
        })
        .collect()
}

/// Pad a digit vector to a power of two and commit it via FRI.
///
/// `TwoAdicFriPcs` commits polynomials over power-of-two cosets, so an
/// arbitrary-length digit matrix is zero-padded to `next_power_of_two` first.
/// Returns the commitment, prover data, domain, and the padded length.
pub fn commit_digits(pcs: &Pcs, digits: &[i8]) -> (Commitment, ProverData, Domain, usize) {
    let n = digits.len();
    let padded = n.next_power_of_two();
    let mut vals = digits_to_field(digits);
    vals.resize(padded, Val::new(0));
    let (commitment, prover_data, domain) = commit_vec(pcs, &vals);
    (commitment, prover_data, domain, padded)
}

/// Open one polynomial at one point and verify the FRI proof.
#[allow(clippy::type_complexity)]
pub fn open_and_verify(
    pcs: &Pcs,
    challenger: &Challenger,
    commitment: &Commitment,
    prover_data: &ProverData,
    domain: Domain,
) -> Challenge {
    let mut prover_ch = challenger.clone();
    prover_ch.observe(commitment.clone());

    let zeta: Challenge = prover_ch.sample_algebra_element();
    let (opened, proof) = <Pcs as PcsTrait<Challenge, Challenger>>::open(
        pcs,
        vec![p3_commit::OpeningRequest {
            prover_data,
            points: vec![vec![zeta]],
        }],
        &mut prover_ch,
    )
    .expect("open");

    let mut verifier_ch = challenger.clone();
    verifier_ch.observe(commitment.clone());
    let verifier_zeta: Challenge = verifier_ch.sample_algebra_element();
    assert_eq!(verifier_zeta, zeta, "transcript divergence");

    let claim = p3_commit::CommitmentOpening {
        commitment: commitment.clone(),
        matrices: vec![p3_commit::MatrixOpening {
            domain,
            points: vec![p3_commit::PointOpening {
                point: zeta,
                values: opened[0][0][0].clone(),
            }],
        }],
    };
    <Pcs as PcsTrait<Challenge, Challenger>>::verify(pcs, vec![claim], &proof, &mut verifier_ch)
        .expect("verify");

    zeta
}

/// Open one committed polynomial at a *specific* point and verify the FRI
/// proof, returning the opened value. (Unlike [`open_and_verify`], the point is
/// caller-supplied — used by the sumcheck's base case, which opens at Boolean
/// points/indices rather than a challenger-sampled field point.)
pub fn open_at(
    pcs: &Pcs,
    challenger: &Challenger,
    commitment: &Commitment,
    prover_data: &ProverData,
    domain: Domain,
    point: Challenge,
) -> Challenge {
    let mut prover_ch = challenger.clone();
    prover_ch.observe(commitment.clone());
    let (opened, proof) = <Pcs as PcsTrait<Challenge, Challenger>>::open(
        pcs,
        vec![p3_commit::OpeningRequest {
            prover_data,
            points: vec![vec![point]],
        }],
        &mut prover_ch,
    )
    .expect("open");

    let mut verifier_ch = challenger.clone();
    verifier_ch.observe(commitment.clone());
    let claim = p3_commit::CommitmentOpening {
        commitment: commitment.clone(),
        matrices: vec![p3_commit::MatrixOpening {
            domain,
            points: vec![p3_commit::PointOpening {
                point,
                values: opened[0][0][0].clone(),
            }],
        }],
    };
    <Pcs as PcsTrait<Challenge, Challenger>>::verify(pcs, vec![claim], &proof, &mut verifier_ch)
        .expect("verify");

    opened[0][0][0][0]
}

/// Prover-side opening: open one committed polynomial at `point`, returning the
/// opened value and the FRI opening proof. The verifier later checks it with
/// [`open_at_verify`] without any prover material.
pub fn open_at_prove(
    pcs: &Pcs,
    challenger: &Challenger,
    commitment: &Commitment,
    prover_data: &ProverData,
    point: Challenge,
) -> (Challenge, Proof) {
    let mut ch = challenger.clone();
    ch.observe(commitment.clone());
    let (opened, proof) = <Pcs as PcsTrait<Challenge, Challenger>>::open(
        pcs,
        vec![p3_commit::OpeningRequest {
            prover_data,
            points: vec![vec![point]],
        }],
        &mut ch,
    )
    .expect("open");
    (opened[0][0][0][0], proof)
}

/// Verifier-side opening: check the opening proof against the claim (commitment,
/// point, value). Holds no prover material — only the commitment, domain, point,
/// claimed value, and proof.
pub fn open_at_verify(
    pcs: &Pcs,
    challenger: &Challenger,
    commitment: &Commitment,
    domain: Domain,
    point: Challenge,
    value: Challenge,
    proof: &Proof,
) -> bool {
    let mut ch = challenger.clone();
    ch.observe(commitment.clone());
    let claim = p3_commit::CommitmentOpening {
        commitment: commitment.clone(),
        matrices: vec![p3_commit::MatrixOpening {
            domain,
            points: vec![p3_commit::PointOpening {
                point,
                values: vec![value],
            }],
        }],
    };
    <Pcs as PcsTrait<Challenge, Challenger>>::verify(pcs, vec![claim], proof, &mut ch).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_open_verify_roundtrip() {
        let (pcs, challenger) = new_pcs(1, 10);
        let vals = digits_to_field(&[1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, 0, 0, 0, 0]);
        let (commitment, prover_data, domain) = commit_vec(&pcs, &vals);
        let _ = open_and_verify(&pcs, &challenger, &commitment, &prover_data, domain);
    }

    #[test]
    fn commit_digits_pads_to_power_of_two() {
        let (pcs, challenger) = new_pcs(1, 10);
        // length 10, not a power of two → padded to 16.
        let digits = vec![1i8, 2, 3, 4, 5, -1, -2, -3, -4, 0];
        let (commitment, prover_data, domain, padded) = commit_digits(&pcs, &digits);
        assert_eq!(padded, 16);
        let _ = open_and_verify(&pcs, &challenger, &commitment, &prover_data, domain);
    }

    #[test]
    fn open_at_specific_point() {
        let (pcs, challenger) = new_pcs(1, 10);
        let vals = digits_to_field(&[1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, 0, 0, 0, 0]);
        let (commitment, prover_data, domain) = commit_vec(&pcs, &vals);
        // Open at a fixed field point and sanity-check it returns *a* value
        // (the exact value is the committed polynomial's evaluation there).
        let point = Challenge::from(Val::new(3));
        let _ = open_at(&pcs, &challenger, &commitment, &prover_data, domain, point);
    }
}
