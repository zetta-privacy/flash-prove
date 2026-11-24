//! The `embed` op, proven end-to-end on the stacked-WHIR layer.
//!
//! This is the first real op ported off FRI. It exercises everything the port
//! needs that the toy tests did not:
//!
//! * **two roots** — the embedding weights are an offline, pre-shared commitment
//!   (the `cp_commit` phase), the activation cell belongs to the prove-time
//!   segment. The op's claims span both, and the transcript binds both roots
//!   before the random linear combination is drawn.
//! * **a sparse linear functional** — `out[b][j] = emb[window[b]][j]` turns the
//!   RLC vector `r` into a vector with `l·hidden` nonzero entries scattered over
//!   `vocab·hidden` positions ([`LinearH::Sparse`]).
//! * **the claim identity** `⟨r, out⟩ = ⟨e, emb⟩`, checked through two
//!   independent multilinear sumchecks that each end in one WHIR opening.

use flashprove::field::{Ext, Fp};
use flashprove::gkr::segment::{claim_value_at, linear_finish, linear_verify, verify, CellSegment};
use flashprove::model::{Cell, WeightMatrix};
use flashprove::pcs::transcript::{new_challenger, Transcript};
use flashprove::pcs::linear_claim::LinearH;
use flashprove::pcs::whir_stacked::{observe_root, Params};
use p3_field::PrimeCharacteristicRing;

const VOCAB: usize = 8;
const HIDDEN: usize = 4;
const SEQ: usize = 2;
const WINDOW: [usize; SEQ] = [1, 5];
const N_EMB: usize = VOCAB * HIDDEN;
const N_OUT: usize = SEQ * HIDDEN;

fn fp(x: i64) -> Fp {
    if x < 0 {
        -Fp::new(x.unsigned_abs())
    } else {
        Fp::new(x as u64)
    }
}

/// The embedding tensor, flattened the way the op reads it (row-major
/// `vocab × hidden`), as a pre-shared weight commitment.
fn embedding_weights() -> WeightMatrix {
    let vals: Vec<i8> = (0..N_EMB).map(|i| ((i as i64 % 7) - 3) as i8).collect();
    WeightMatrix::new("embed", N_EMB, 1, vals)
}

/// `out[b][j] = emb[window[b]][j]`, flattened to `l·hidden`.
fn embed_output(w: &WeightMatrix) -> Vec<i64> {
    let mut out = vec![0i64; N_OUT];
    for b in 0..SEQ {
        for j in 0..HIDDEN {
            out[b * HIDDEN + j] = w.values[WINDOW[b] * HIDDEN + j] as i64;
        }
    }
    out
}

/// The public sparse functional `e`: `r[b][j]` scattered to `window[b]·hidden + j`.
fn scatter(r: &[Fp], window: &[usize]) -> LinearH {
    let mut entries = Vec::with_capacity(r.len());
    for b in 0..SEQ {
        for j in 0..HIDDEN {
            entries.push((window[b] * HIDDEN + j, r[b * HIDDEN + j]));
        }
    }
    LinearH::Sparse {
        entries,
        len: N_EMB,
    }
}

#[test]
fn embed_op_on_stacked_whir() {
    let w = embedding_weights();
    let out_vals = embed_output(&w);

    // --- prover -----------------------------------------------------------
    // Two segments: the pre-shared weights and the prove-time activation.
    let mut seg_w = CellSegment::seal_weight(&w, Params::default());
    let mut seg_c = CellSegment::seal(
        &[Cell::new("out", N_OUT, 1, out_vals.clone())],
        Params::default(),
    );
    let com_w = seg_w.commitment();
    let com_c = seg_c.commitment();
    assert_ne!(com_w.pcs.root, com_c.pcs.root);

    // The transcript binds both roots *before* the RLC is sampled.
    let mut ch = new_challenger();
    observe_root(&com_w.pcs.root, &mut ch);
    observe_root(&com_c.pcs.root, &mut ch);
    let mut tr = Transcript::new(&ch);
    let r: Vec<Fp> = (0..N_OUT).map(|_| tr.sample_val()).collect();

    let h_out = LinearH::Dense(r.clone());
    let h_emb = scatter(&r, &WINDOW);

    let (b_out, proof_out) = seg_c.claim_linear("out", 0, &h_out, &mut tr);
    let (b_emb, proof_emb) = seg_w.claim_linear("embed", 0, &h_emb, &mut tr);
    assert_eq!(b_out, b_emb, "⟨r, out⟩ = ⟨e, emb⟩");

    let claims_c = seg_c.claims().to_vec();
    let proof_c = seg_c.prove(&mut new_challenger());
    let claims_w = seg_w.claims().to_vec();
    let proof_w = seg_w.prove(&mut new_challenger());

    // --- verifier ---------------------------------------------------------
    let mut vch = new_challenger();
    observe_root(&com_w.pcs.root, &mut vch);
    observe_root(&com_c.pcs.root, &mut vch);
    // The segment proofs run on their own fresh transcripts (the prover used one
    // per segment); what binds them to the op is that `tr` observed both roots.
    assert!(verify(&com_c, &claims_c, &proof_c, &mut new_challenger()));
    assert!(verify(&com_w, &claims_w, &proof_w, &mut new_challenger()));

    let mut trv = Transcript::new(&vch);
    // Consume the same RLC draws the prover made before the claims.
    let r_v: Vec<Fp> = (0..N_OUT).map(|_| trv.sample_val()).collect();
    assert_eq!(r_v, r, "same RLC vector on both sides");
    let (z_out, c_out) = linear_verify(&proof_out, b_out, &mut trv).expect("out sumcheck");
    let v_out = claim_value_at(&com_c, &claims_c, "out", &z_out).expect("out opening");
    assert!(linear_finish(&h_out, &z_out, c_out, v_out));

    let (z_emb, c_emb) = linear_verify(&proof_emb, b_emb, &mut trv).expect("emb sumcheck");
    let v_emb = claim_value_at(&com_w, &claims_w, "embed", &z_emb).expect("emb opening");
    assert!(linear_finish(&h_emb, &z_emb, c_emb, v_emb));

    // --- soundness ---------------------------------------------------------
    // A different window scatters the same `r` elsewhere, so the identity the op
    // needs no longer holds: a prover cannot substitute a different lookup.
    let mut wrong_window = WINDOW;
    wrong_window[0] = 2;
    let h_wrong = scatter(&r, &wrong_window);
    let b_wrong: Fp = h_wrong
        .dense()
        .iter()
        .zip(w.values.iter().map(|&v| fp(v as i64)))
        .map(|(&a, b)| a * b)
        .sum();
    assert_ne!(Ext::from(b_wrong), b_out);

    // A tampered activation cell no longer verifies.
    let mut bad = claims_c.clone();
    bad[0].values[0] += Ext::ONE;
    assert!(!verify(&com_c, &bad, &proof_c, &mut new_challenger()));

    // A tampered embedding opening breaks the terminal identity.
    assert!(!linear_finish(&h_emb, &z_emb, c_emb, v_emb + Ext::ONE));
}
