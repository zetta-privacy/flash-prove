//! Linear claims `⟨h, v⟩ = b` on committed data, via multilinear sumcheck.
//!
//! The op graph is full of *linear* claims that are not point evaluations: most
//! importantly the row-slice ties of [`crate::gkr::commit::tie_row`], where an
//! ephemeral slice `row` must be shown to be a contiguous block of a committed
//! cell `full`.
//!
//! The FRI layer answered these with the univariate constant-coefficient trick
//! (`p = h⊙v`, `q = (P−c₀)/X`, check `P(α) = c₀ + α·Q(α)`). That needs committed
//! *products*, which a WHIR/FRI point-opening scheme cannot express without
//! extra roots. The multilinear answer needs none:
//!
//! ```text
//!   ⟨h, v⟩ = Σ_{x∈{0,1}^m} h̃(x)·ṽ(x)
//! ```
//!
//! so a standard quadratic sumcheck over the product `h̃·ṽ` reduces the claim to
//! a single point evaluation `ṽ(z)` — exactly what the stacked WHIR commitment
//! already opens. No derived tables, no extra commitments, no univariate code.
//!
//! Cost: prover `O(n)`, verifier `O(m)` rounds plus `O(|h|)` to evaluate the
//! *public* `h̃(z)` at the end (both `h` shapes here have `O(d)` support).

use crate::field::sumcheck::RoundPoly;
use crate::field::{eq, Ext, Fp};
use crate::pcs::fri::Transcript;
use p3_field::PrimeCharacteristicRing;

/// A quadratic sumcheck for one linear claim, in round order.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LinearClaimProof {
    pub rounds: Vec<RoundPoly>,
}

/// The public linear functional `h` of a claim `⟨h, v⟩ = b`.
///
/// Both shapes the op graph needs are represented: a fully public vector (the
/// random RLC vector of a tie) and a contiguous block (the same RLC placed into
/// the full cell's coordinates).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LinearH {
    /// All `2^m` entries are public.
    Dense(Vec<Fp>),
    /// `len` entries, all zero except `values` at `start..start + values.len()`.
    Block {
        start: usize,
        values: Vec<Fp>,
        len: usize,
    },
    /// `len` entries, zero except at the listed positions (embeddings scatter the
    /// RLC vector into the vocabulary rows the window actually reads).
    Sparse {
        entries: Vec<(usize, Fp)>,
        len: usize,
    },
}

impl LinearH {
    pub fn len(&self) -> usize {
        match self {
            Self::Dense(h) => h.len(),
            Self::Block { len, .. } | Self::Sparse { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The full public vector, zero-padded to `len` (prover side).
    pub fn dense(&self) -> Vec<Fp> {
        match self {
            Self::Dense(h) => h.clone(),
            Self::Block { start, values, len } => {
                debug_assert!(
                    start + values.len() <= *len,
                    "Block out of range: start={start}, values={}, len={len}",
                    values.len()
                );
                let mut out = Fp::zero_vec(*len);
                out[*start..*start + values.len()].copy_from_slice(values);
                out
            }
            Self::Sparse { entries, len } => {
                debug_assert!(
                    entries.iter().all(|&(i, _)| i < *len),
                    "Sparse entry out of range (len={len})"
                );
                let mut out = Fp::zero_vec(*len);
                for &(i, v) in entries {
                    out[i] += v;
                }
                out
            }
        }
    }

    /// `h̃(z)`, evaluated using only the support (verifier side).
    pub fn eval_at(&self, z: &[Ext]) -> Ext {
        match self {
            Self::Dense(h) => h
                .iter()
                .enumerate()
                .map(|(j, &x)| eq(z, j) * Ext::from(x))
                .sum(),
            Self::Block { start, values, .. } => values
                .iter()
                .enumerate()
                .map(|(t, &x)| eq(z, start + t) * Ext::from(x))
                .sum(),
            Self::Sparse { entries, .. } => {
                entries.iter().map(|&(i, x)| eq(z, i) * Ext::from(x)).sum()
            }
        }
    }
}

/// Prove `⟨h, v⟩`, returning the sumcheck proof, the final point `z` (whose
/// `ṽ(z)` the caller must open) and the claim `b`.
pub fn prove(v: &[Fp], h: &LinearH, tr: &mut Transcript) -> (LinearClaimProof, Vec<Ext>, Ext) {
    assert_eq!(v.len(), h.len(), "linear claim length mismatch");
    assert!(v.len().is_power_of_two(), "linear claim needs 2^m entries");
    let m = v.len().trailing_zeros() as usize;

    let mut hf: Vec<Ext> = h.dense().iter().map(|&x| Ext::from(x)).collect();
    let mut vf: Vec<Ext> = v.iter().map(|&x| Ext::from(x)).collect();
    let b: Ext = hf.iter().zip(&vf).map(|(&a, &c)| a * c).sum();

    let mut rounds = Vec::with_capacity(m);
    let mut z = Vec::with_capacity(m);
    for _ in 0..m {
        // g(t) = Σ_{j'} (h₀ + t·Δh)(v₀ + t·Δv) = A + B·t + C·t².
        let (mut a, mut bcoef, mut c) = (Ext::ZERO, Ext::ZERO, Ext::ZERO);
        for j in 0..vf.len() / 2 {
            let h0 = hf[2 * j];
            let dh = hf[2 * j + 1] - h0;
            let v0 = vf[2 * j];
            let dv = vf[2 * j + 1] - v0;
            a += h0 * v0;
            bcoef += h0 * dv + dh * v0;
            c += dh * dv;
        }
        let g = RoundPoly([a, bcoef, c]);
        for &coef in &g.0 {
            tr.observe_ext(coef);
        }
        let r = tr.sample_ext();
        rounds.push(g);
        z.push(r);

        // Fold variable 0 (the LSB: `eq` pairs `z[0]` with bit 0).
        let half = vf.len() / 2;
        for j in 0..half {
            hf[j] = hf[2 * j] + r * (hf[2 * j + 1] - hf[2 * j]);
            vf[j] = vf[2 * j] + r * (vf[2 * j + 1] - vf[2 * j]);
        }
        hf.truncate(half);
        vf.truncate(half);
    }

    (LinearClaimProof { rounds }, z, b)
}

/// Replay the sumcheck against a claimed `b`.
///
/// Returns `(z, claim_final)`, or `None` when a round violates the sumcheck.
/// The caller then checks `h̃(z)·ṽ(z) == claim_final` with the opened value.
pub fn verify(proof: &LinearClaimProof, b: Ext, tr: &mut Transcript) -> Option<(Vec<Ext>, Ext)> {
    let mut claim = b;
    let mut z = Vec::with_capacity(proof.rounds.len());
    for g in &proof.rounds {
        if g.eval(Ext::ZERO) + g.eval(Ext::ONE) != claim {
            return None;
        }
        for &coef in &g.0 {
            tr.observe_ext(coef);
        }
        let r = tr.sample_ext();
        claim = g.eval(r);
        z.push(r);
    }
    Some((z, claim))
}

/// The terminal identity `h̃(z)·ṽ(z) = claim`.
pub fn finish(h: &LinearH, z: &[Ext], claim: Ext, v_at_z: Ext) -> bool {
    h.eval_at(z) * v_at_z == claim
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::fri::new_challenger;
    use crate::pcs::zk::mle_value;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    fn rand_fp(rng: &mut StdRng, n: usize) -> Vec<Fp> {
        (0..n)
            .map(|_| Fp::new(rng.random::<u64>() % 0xFFFF_FFFF_0000_0001))
            .collect()
    }

    fn tr() -> Transcript {
        Transcript::new(&new_challenger())
    }

    fn ip(h: &[Fp], v: &[Fp]) -> Fp {
        h.iter().zip(v).map(|(&a, &b)| a * b).sum()
    }

    #[test]
    fn dense_roundtrip() {
        let mut rng = StdRng::seed_from_u64(1);
        let v = rand_fp(&mut rng, 16);
        let h = LinearH::Dense(rand_fp(&mut rng, 16));

        let (proof, z, b) = prove(&v, &h, &mut tr());
        assert_eq!(b, Ext::from(ip(&h.dense(), &v)));
        assert_eq!(proof.rounds.len(), 4);

        let mut vt = tr();
        let (z2, claim) = verify(&proof, b, &mut vt).expect("honest rounds pass");
        assert_eq!(z2, z, "prover and verifier derive the same point");
        assert!(finish(&h, &z2, claim, mle_value(&v, &z2)));
    }

    #[test]
    fn wrong_b_rejected() {
        let mut rng = StdRng::seed_from_u64(2);
        let v = rand_fp(&mut rng, 16);
        let h = LinearH::Dense(rand_fp(&mut rng, 16));
        let (proof, _, b) = prove(&v, &h, &mut tr());

        let mut vt = tr();
        assert!(verify(&proof, b + Ext::ONE, &mut vt).is_none());
    }

    #[test]
    fn wrong_opened_value_rejected() {
        let mut rng = StdRng::seed_from_u64(3);
        let v = rand_fp(&mut rng, 16);
        let h = LinearH::Dense(rand_fp(&mut rng, 16));
        let (proof, _, b) = prove(&v, &h, &mut tr());

        let mut vt = tr();
        let (z, claim) = verify(&proof, b, &mut vt).unwrap();
        assert!(finish(&h, &z, claim, mle_value(&v, &z)));
        assert!(!finish(&h, &z, claim, mle_value(&v, &z) + Ext::ONE));
    }

    #[test]
    fn block_matches_dense_and_folds_into_full_length() {
        let mut rng = StdRng::seed_from_u64(4);
        let v = rand_fp(&mut rng, 16);
        let values = rand_fp(&mut rng, 3);
        let block = LinearH::Block {
            start: 5,
            values: values.clone(),
            len: 16,
        };
        let mut expect = Fp::zero_vec(16);
        expect[5..8].copy_from_slice(&values);
        assert_eq!(block.dense(), expect);

        let (proof, z, b) = prove(&v, &block, &mut tr());
        let mut vt = tr();
        let (z2, claim) = verify(&proof, b, &mut vt).expect("honest");
        assert_eq!(z, z2);
        assert!(finish(&block, &z2, claim, mle_value(&v, &z2)));
        // The support-only evaluation agrees with the dense one.
        assert_eq!(block.eval_at(&z2), LinearH::Dense(expect).eval_at(&z2));
    }

    #[test]
    fn tie_pattern_row_and_block_share_a_value() {
        // The `tie_row` shape: `row` is `full[start..start+d]`.
        let mut rng = StdRng::seed_from_u64(5);
        let d = 4;
        let full = rand_fp(&mut rng, 16);
        let start = 8;
        let row = full[start..start + d].to_vec();
        let r = rand_fp(&mut rng, d);

        // ⟨r, row⟩ on the slice …
        let b_row = Ext::from(ip(&r, &row));
        // … equals ⟨e, full⟩ with `e` the same `r` placed at the block.
        let e = LinearH::Block {
            start,
            values: r,
            len: full.len(),
        };
        let (proof, z, b_full) = prove(&full, &e, &mut tr());
        assert_eq!(b_row, b_full);

        let mut vt = tr();
        let (z2, claim) = verify(&proof, b_full, &mut vt).unwrap();
        assert_eq!(z, z2);
        assert!(finish(&e, &z2, claim, mle_value(&full, &z2)));
    }

    #[test]
    fn sparse_matches_dense() {
        let mut rng = StdRng::seed_from_u64(7);
        let v = rand_fp(&mut rng, 32);
        let entries = vec![(3, Fp::new(5)), (17, Fp::new(9)), (31, Fp::new(2))];
        let sparse = LinearH::Sparse {
            entries: entries.clone(),
            len: 32,
        };
        let mut expect = Fp::zero_vec(32);
        for (i, x) in &entries {
            expect[*i] = *x;
        }
        assert_eq!(sparse.dense(), expect);

        let (proof, z, b) = prove(&v, &sparse, &mut tr());
        assert_eq!(b, Ext::from(ip(&expect, &v)));
        let mut vt = tr();
        let (z2, claim) = verify(&proof, b, &mut vt).expect("honest");
        assert_eq!(z, z2);
        assert!(finish(&sparse, &z2, claim, mle_value(&v, &z2)));
        assert_eq!(sparse.eval_at(&z2), LinearH::Dense(expect).eval_at(&z2));
    }

    #[test]
    fn round_polynomial_is_quadratic() {
        let mut rng = StdRng::seed_from_u64(6);
        let v = rand_fp(&mut rng, 8);
        let h = LinearH::Dense(rand_fp(&mut rng, 8));
        let (proof, _, _) = prove(&v, &h, &mut tr());
        for g in &proof.rounds {
            assert_eq!(g.0.len(), 3);
            let two = Ext::from(Fp::new(2));
            let (a, b, c) = (g.0[0], g.0[1], g.0[2]);
            let f = |x: Ext| a + b * x + c * x * x;
            assert_eq!(g.eval(two), f(two));
            assert_eq!(g.eval(Ext::ZERO) + g.eval(Ext::ONE), a + (a + b + c));
        }
    }
}
