//! Minimal Plonky3 FRI (TwoAdicFriPcs) round-trip over Goldilocks: commit a
//! polynomial, open it at a point, verify the FRI proof. This de-risks the
//! succinct-verifier integration before wiring it into `protocol.rs`.

use p3_challenger::{CanObserve, DuplexChallenger, FieldChallenger};
use p3_commit::{ExtensionMmcs, Pcs, PolynomialSpace};
use p3_dft::Radix2DitParallel;
use p3_field::extension::CubicTrinomialExtensionField;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::rngs::StdRng;
use rand::SeedableRng;

type Val = Goldilocks;
type Challenge = CubicTrinomialExtensionField<Goldilocks>;
type Perm = Poseidon2Goldilocks<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs = MerkleTreeMmcs<Val, Val, MyHash, MyCompress, 2, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Dft = Radix2DitParallel<Val>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type MyPcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;

fn get_pcs(log_blowup: usize) -> (MyPcs, Challenger) {
    let mut rng = StdRng::seed_from_u64(0);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());

    let val_mmcs = ValMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    let fri_params = FriParameters {
        log_blowup,
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 10,
        batch_proof_of_work_bits: 0,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };

    let pcs = MyPcs::new(Dft::default(), val_mmcs, fri_params);
    (pcs, Challenger::new(perm))
}

/// Generic like the p3-fri test helper: `P: Pcs<Challenge, C>` ties the PCS's
/// challenger type to `C`, so method resolution is unambiguous.
fn run_demo<P, C>(pcs: &P, challenger: &C)
where
    P: Pcs<Challenge, C>,
    P::Domain: PolynomialSpace<Val = Val>,
    C: Clone + CanObserve<P::Commitment> + FieldChallenger<Val>,
{
    // One polynomial evaluated over a domain of size 32 (degree < 32).
    let degree = 32usize;
    let domain = pcs.natural_domain_for_degree(degree);
    let n = domain.size();
    let mut rng = StdRng::seed_from_u64(1);
    let matrix = RowMajorMatrix::<Val>::rand(&mut rng, n, 1);

    let mut prover_ch = challenger.clone();
    let (commit, data) = pcs.commit(vec![(domain, matrix.clone())]).unwrap();
    prover_ch.observe(commit.clone());

    let zeta: Challenge = prover_ch.sample_algebra_element();
    let (opened, proof) = pcs
        .open(
            vec![p3_commit::OpeningRequest {
                prover_data: &data,
                points: vec![vec![zeta]],
            }],
            &mut prover_ch,
        )
        .unwrap();

    let mut verifier_ch = challenger.clone();
    verifier_ch.observe(commit.clone());
    let verifier_zeta: Challenge = verifier_ch.sample_algebra_element();
    assert_eq!(verifier_zeta, zeta);

    let claim = p3_commit::CommitmentOpening {
        commitment: commit,
        matrices: vec![p3_commit::MatrixOpening {
            domain,
            points: vec![p3_commit::PointOpening {
                point: zeta,
                values: opened[0][0][0].clone(),
            }],
        }],
    };

    pcs.verify(vec![claim], &proof, &mut verifier_ch).unwrap();
}

fn main() {
    let (pcs, challenger) = get_pcs(1);
    run_demo(&pcs, &challenger);
    println!("FRI commit/open/verify round-trip OK (domain 32, blowup 1)");
}
