//! BaseFold-style multilinear PCS over Goldilocks, via plonky3 **WHIR**
//! (`p3-whir`). WHIR is the BaseFold/WHIR-family scheme: commit the multilinear
//! polynomial once, prove evaluations by sumcheck folding + FRI-style proximity,
//! with an `O(log²N)` verifier and `O(N)`-ish prover — and **no per-claim `P`/`Q`
//! commitments**.
//!
//! One vector `v` (padded to `2^m` evaluations) is committed as a single
//! one-column table. Each MLE evaluation `ṽ(z)` is opened at the prescribed point
//! `z` with [`PrescribedPointPcs::open_at`] / verified with `verify_at`.

use crate::field::{Ext, Fp};
use crate::pcs::blake3_mmcs::{Blake3Compress, Blake3Hasher};
use crate::pcs::fri::{new_challenger, Challenger};
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DitParallel;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_sumcheck::layout::{SuffixProver, Table, Witness};
use p3_sumcheck::table::{OpeningBatch, OpeningProtocol, TableShape, TableSpec};
use p3_sumcheck::PrescribedPointPcs;
use p3_whir::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig, WhirProver};

type F = Fp;
type EF = Ext;
type Dft = Radix2DitParallel<F>;
type MT = MerkleTreeMmcs<F, F, Blake3Hasher, Blake3Compress, 2, 2>;
type Layout = SuffixProver<F, EF>;
type Prover = WhirProver<EF, F, Dft, MT, Challenger, Layout>;

pub type Commitment = <Prover as MultilinearPcs<EF, Challenger>>::Commitment;
pub type ProverData = <Prover as MultilinearPcs<EF, Challenger>>::ProverData;
pub type Proof = <Prover as MultilinearPcs<EF, Challenger>>::Proof;

fn build_prover(m: usize) -> Prover {
    let params = ProtocolParameters {
        security_level: 100,
        pow_bits: 16,
        round_log_inv_rates: Vec::new(),
        folding_factor: FoldingFactor::Constant(4),
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 1,
    };
    let config =
        WhirConfig::<EF, F, Challenger>::new(m, params).expect("WHIR config over Goldilocks");
    WhirProver::new(
        config,
        Dft::default(),
        MT::new(Blake3Hasher, Blake3Compress, 0),
    )
}

thread_local! {
    /// One WHIR prover per number of variables, reused across calls (config
    /// derivation is expensive; the prover itself is stateless across openings).
    static PROVERS: std::cell::RefCell<std::collections::HashMap<usize, Prover>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

fn with_prover<R>(m: usize, f: impl FnOnce(&Prover) -> R) -> R {
    PROVERS.with(|cell| {
        if !cell.borrow().contains_key(&m) {
            cell.borrow_mut().insert(m, build_prover(m));
        }
        let borrow = cell.borrow();
        f(&borrow[&m])
    })
}

fn protocol(m: usize) -> OpeningProtocol {
    OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(m, 1),
        vec![OpeningBatch::new(vec![0], Vec::new())],
    )])
}

fn witness(v: &[F], folding: usize) -> Witness<F> {
    let n = v.len().next_power_of_two();
    let mut evals = v.to_vec();
    evals.resize(n, F::ZERO);
    Witness::new(vec![Table::new(RowMajorMatrix::new(evals, n))], folding)
}

/// Commit `v` as a multilinear polynomial. Returns `(commitment, prover_data, m)`
/// where `m = log2(next_pow2(v.len()))` (the polynomial's number of variables).
pub fn commit(v: &[F]) -> (Commitment, ProverData, usize) {
    let m = v.len().next_power_of_two().trailing_zeros() as usize;
    with_prover(m, |prover| {
        let folding = prover.round_folding_factor(0);
        let mut ch = new_challenger();
        let (com, pd) = prover
            .commit(witness(v, folding), &mut ch)
            .expect("WHIR commit");
        (com, pd, m)
    })
}

/// Prove `ṽ(z) = ·` for the committed polynomial at `point` (`z`, length `m`).
pub fn open(com: &Commitment, pd: ProverData, m: usize, point: &[EF]) -> Proof {
    with_prover(m, |prover| {
        let mut ch = new_challenger();
        prover.observe_commitment(com, &mut ch);
        let protocol = protocol(m);
        let pt = Point::new(point.to_vec());
        <Prover as PrescribedPointPcs<EF, Challenger>>::open_at(
            prover,
            pd,
            &protocol,
            &[pt],
            &mut ch,
        )
        .expect("WHIR open_at")
    })
}

/// Verify that the committed polynomial evaluates to `claimed` at `point`.
pub fn verify(com: &Commitment, proof: &Proof, m: usize, point: &[EF], claimed: EF) -> bool {
    with_prover(m, |prover| {
        let mut ch = new_challenger();
        prover.observe_commitment(com, &mut ch);
        let protocol = protocol(m);
        let pt = Point::new(point.to_vec());
        match <Prover as PrescribedPointPcs<EF, Challenger>>::verify_at(
            prover,
            com,
            proof,
            &protocol,
            &[pt],
            &mut ch,
        ) {
            Ok(evals) => {
                evals
                    .first()
                    .and_then(|batch| batch.current().first())
                    .copied()
                    == Some(claimed)
            }
            Err(_) => false,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Ext;
    use p3_field::PrimeCharacteristicRing;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    fn rand_vec(rng: &mut StdRng, n: usize) -> Vec<F> {
        (0..n)
            .map(|_| F::new(rng.random::<u64>() % 0xFFFF_FFFF_0000_0001))
            .collect()
    }

    fn mle_eval(v: &[F], point: &[Ext]) -> Ext {
        // multilinear evaluation: fold with each point coordinate
        let mut vals: Vec<Ext> = v.iter().map(|&x| Ext::from(x)).collect();
        for &z in point {
            let half = vals.len() / 2;
            let mut next = Vec::with_capacity(half);
            for i in 0..half {
                next.push(vals[i] + z * (vals[i + half] - vals[i]));
            }
            vals = next;
        }
        vals[0]
    }

    #[test]
    fn eval_roundtrip() {
        let mut rng = StdRng::seed_from_u64(1);
        let n = 16;
        let v = rand_vec(&mut rng, n);
        let m = n.trailing_zeros() as usize;
        let (com, pd, mm) = commit(&v);
        assert_eq!(mm, m);

        let point: Vec<Ext> = (0..m)
            .map(|_| Ext::from(F::new(rng.random::<u64>() % 0xFFFF_FFFF_0000_0001)))
            .collect();
        let claimed = mle_eval(&v, &point);

        let proof = open(&com, pd, m, &point);
        assert!(
            verify(&com, &proof, m, &point, claimed),
            "honest eval verifies"
        );
    }

    #[test]
    fn wrong_value_rejected() {
        let mut rng = StdRng::seed_from_u64(2);
        let n = 16;
        let v = rand_vec(&mut rng, n);
        let m = n.trailing_zeros() as usize;
        let (com, pd, _) = commit(&v);

        let point: Vec<Ext> = (0..m)
            .map(|_| Ext::from(F::new(rng.random::<u64>() % 0xFFFF_FFFF_0000_0001)))
            .collect();
        let claimed = mle_eval(&v, &point) + Ext::ONE;

        let proof = open(&com, pd, m, &point);
        assert!(
            !verify(&com, &proof, m, &point, claimed),
            "wrong value rejects"
        );
    }
}
