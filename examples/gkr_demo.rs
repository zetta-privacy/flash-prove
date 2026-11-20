//! Minimal Plonky3 GKR round-trip over Goldilocks: prove + verify a batched
//! product tree. De-risks the GKR integration before wiring it (with the FRI
//! for leaf authentication) into the succinct verifier.

use p3_challenger::DuplexChallenger;
use p3_field::extension::CubicTrinomialExtensionField;
use p3_gkr::{ProductGkrProof, ProductGkrRootShape, ProductGkrShape};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use rand::rngs::StdRng;
use rand::SeedableRng;

type F = Goldilocks;
type EF = CubicTrinomialExtensionField<Goldilocks>;
type Perm = Poseidon2Goldilocks<16>;
type Challenger = DuplexChallenger<F, Perm, 16, 8>;

fn main() {
    let mut rng = StdRng::seed_from_u64(0);
    let perm = Perm::new_from_rng_128(&mut rng);
    let mut prover_ch = Challenger::new(perm.clone());
    let mut verifier_ch = Challenger::new(perm);

    let log_height = 3usize; // 8 leaves per tree
    let num_trees = 3usize;
    let shape = ProductGkrShape::new(log_height, num_trees, ProductGkrRootShape::Distinct).unwrap();

    // 3 trees × 8 leaves, values 1..=8 embedded into the extension field.
    let inputs: Vec<Vec<EF>> = (0..num_trees)
        .map(|_| (1..=8u64).map(|x| EF::from(Goldilocks::new(x))).collect())
        .collect();
    let borrowed: Vec<&[EF]> = inputs.iter().map(|v| v.as_slice()).collect();

    let (proof, prover_output) = ProductGkrProof::prove::<F, _>(&borrowed, shape, &mut prover_ch);
    let verifier_output = proof.verify::<F, _>(shape, &mut verifier_ch).unwrap();
    assert_eq!(prover_output, verifier_output);
    println!(
        "GKR product-tree round-trip OK ({} trees, height {log_height})",
        num_trees
    );
}
