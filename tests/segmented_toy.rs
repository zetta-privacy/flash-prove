//! End-to-end 2-segment toy model on the stacked-WHIR commitment layer.
//!
//! This is the architectural analogue of the paper's segmented proof, minus the
//! op-graph: each *segment* is one stacked commitment, one root, one WHIR proof,
//! and everything of that segment is freed before the next segment starts.
//!
//! ```text
//!   segment 0:  out₀ = in₀ + w₀ + b₀
//!   segment 1:  out₁ = in₁ + w₁ + b₁        with  in₁ = out₀   (segment boundary)
//! ```
//!
//! The boundary is bound the way the paper's link proof does it: both roots are
//! absorbed into one transcript *before* the evaluation point `z` is drawn, so a
//! prover cannot pick `z` to make two unequal boundary vectors collide. The
//! verifier then checks
//!
//! * each segment's linear cell identity at `z`, from that segment's openings,
//! * `in₁(z) == out₀(z)` across the boundary,
//!
//! which soundly forces the two committed boundary vectors to agree.

use flashprove::field::{Ext, Fp};
use flashprove::pcs::transcript::{new_challenger, Challenger};
use flashprove::pcs::whir_stacked::{
    observe_root, sample_point, verify, Claim, Committer, Params, Proof, SegmentCommitment,
};
use p3_field::PrimeCharacteristicRing;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

type F = Fp;
type E = Ext;

const M: usize = 4;
const N: usize = 1 << M;

/// A proved segment as the verifier sees it: root + claims + one WHIR proof.
struct SegmentProof {
    commitment: SegmentCommitment,
    claims: Vec<Claim>,
    proof: Proof,
}

fn rand_vec(rng: &mut StdRng) -> Vec<F> {
    (0..N)
        .map(|_| F::new(rng.random::<u64>() % 0xFFFF_FFFF_0000_0001))
        .collect()
}

/// Build and prove one segment: `out = in + w + b`, with every cell claimed at `z`.
fn prove_segment(
    i: &[F],
    w: &[F],
    b: &[F],
    out: &[F],
    z: &[E],
    mut ch: Challenger,
) -> SegmentProof {
    let mut c = Committer::new();
    let ti = c.add_vec(i);
    let tw = c.add_vec(w);
    let tb = c.add_vec(b);
    let to = c.add_vec(out);

    let mut s = c.seal(Params::default());
    // Claim order is fixed and shared with the verifier: in, w, b, out.
    for (t, col) in [ti, tw, tb, to] {
        s.claim(t, &[col], z);
    }

    let commitment = s.commitment().clone();
    let claims = s.claims().to_vec();
    let proof = s.prove(&mut ch);
    SegmentProof {
        commitment,
        claims,
        proof,
    }
}

#[test]
fn two_segment_toy_model() {
    let mut rng = StdRng::seed_from_u64(20_240_101);

    let in0 = rand_vec(&mut rng);
    let w0 = rand_vec(&mut rng);
    let b0 = rand_vec(&mut rng);
    let out0: Vec<F> = (0..N).map(|k| in0[k] + w0[k] + b0[k]).collect();

    // Segment boundary: segment 1 reads segment 0's output.
    let in1 = out0.clone();
    let w1 = rand_vec(&mut rng);
    let b1 = rand_vec(&mut rng);
    let out1: Vec<F> = (0..N).map(|k| in1[k] + w1[k] + b1[k]).collect();

    // --- prover -----------------------------------------------------------
    // The shared point depends on both roots, so seal both segments first and
    // derive `z` from the roots. A root is a function of the committed data
    // only, so the later re-seal inside `prove_segment` reproduces it exactly.
    let roots: Vec<_> = [(&in0, &w0, &b0, &out0), (&in1, &w1, &b1, &out1)]
        .iter()
        .map(|(i, w, b, o)| {
            let mut c = Committer::new();
            c.add_vec(i);
            c.add_vec(w);
            c.add_vec(b);
            c.add_vec(o);
            c.seal(Params::default()).commitment().root.clone()
        })
        .collect();

    let mut gch = new_challenger();
    for r in &roots {
        observe_root(r, &mut gch);
    }
    let z = sample_point(&mut gch, M);

    let s0 = prove_segment(&in0, &w0, &b0, &out0, &z, gch.clone());
    let s1 = prove_segment(&in1, &w1, &b1, &out1, &z, gch.clone());

    // The prover's roots must equal the roots `z` was derived from.
    assert_eq!(s0.commitment.root, roots[0]);
    assert_eq!(s1.commitment.root, roots[1]);

    // --- verifier ---------------------------------------------------------
    let mut vch = new_challenger();
    observe_root(&s0.commitment.root, &mut vch);
    observe_root(&s1.commitment.root, &mut vch);
    let vz = sample_point(&mut vch, M);
    assert_eq!(vz, z, "verifier derives the same boundary point");

    // Each segment's proof is checked against its own root and claims.
    let mut c0 = vch.clone();
    assert!(verify(&s0.commitment, &s0.claims, &s0.proof, &mut c0));
    let mut c1 = vch.clone();
    assert!(verify(&s1.commitment, &s1.claims, &s1.proof, &mut c1));

    // Claim slots: 0 = in, 1 = w, 2 = b, 3 = out.
    let one = |s: &SegmentProof, k: usize| s.claims[k].values[0];
    for s in [&s0, &s1] {
        assert_eq!(one(s, 3), one(s, 0) + one(s, 1) + one(s, 2));
    }
    // Segment boundary.
    assert_eq!(one(&s1, 0), one(&s0, 3));

    // --- soundness ---------------------------------------------------------
    // A boundary value that does not match segment 0 fails the cross-segment
    // check, and the tampered claim also fails its own segment's opening.
    let mut liar = SegmentProof {
        commitment: s1.commitment.clone(),
        claims: s1.claims.clone(),
        proof: s1.proof.clone(),
    };
    liar.claims[0].values[0] += E::ONE;
    assert_ne!(one(&liar, 0), one(&s0, 3));
    let mut c1 = vch.clone();
    assert!(!verify(
        &liar.commitment,
        &liar.claims,
        &liar.proof,
        &mut c1
    ));

    // `z` must not be derivable before both roots are absorbed.
    let mut early = new_challenger();
    observe_root(&s0.commitment.root, &mut early);
    let z_early = sample_point(&mut early, M);
    observe_root(&s1.commitment.root, &mut early);
    assert_ne!(z_early, z);
}
