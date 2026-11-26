//! Linear claims `⟨h, v⟩ = b` on committed data, via multilinear sumcheck.
//!
//! The op graph is full of *linear* claims that are not point evaluations: most
//! importantly the row-slice ties, where an ephemeral slice `row` must be shown
//! to be a contiguous block of a committed cell `full`.
//!
//! The original layer answered these with the univariate constant-coefficient trick
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
use crate::pcs::transcript::Transcript;
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
    /// All `len` entries are one. `Σ_j eq(z, j) = 1` for any point `z` over the
    /// hypercube, so the terminal evaluation is the constant `1` instead of a
    /// brute-force sum over the whole table (this is the LogUp balance/multiplicity
    /// functional, and the all-ones count in `row_max` and the head claim).
    Ones { len: usize },
    /// The equality tensor built from `m` coins: entry `j` is
    /// `Π_k (j_k ? coins[k] : 1 − coins[k])`, zero-padded to `len`.
    ///
    /// Its terminal evaluation is `Π_k ((1−c_k)(1−z_k) + c_k z_k)` — `O(m)`, with no
    /// dependence on the table size. `len` may exceed `2^m` (a short vector padded up
    /// to the cell's arity); the padding is zero, so the product form is still right
    /// only when `len == 2^m`, and the padded case falls back to the fold.
    Tensor { coins: Vec<Fp>, len: usize },
    /// `blocks` contiguous copies of one tensor, each scaled by a public weight:
    /// entry `j·len + i` is `weights[j] · r[i]`, zero-padded to `total`.
    ///
    /// This is the terminal functional of a digit decomposition — one random vector
    /// over the values, scaled per digit block — and it evaluates in
    /// `O(log len + blocks · log blocks)` instead of `O(len · blocks)`.
    Blocks {
        coins: Vec<Fp>,
        /// Values per block (a power of two).
        len: usize,
        /// The public per-block scale.
        weights: Vec<Fp>,
        /// The cell's arity (`offset + len · blocks`, padded up to a power of two).
        total: usize,
        /// Where the first block starts, in entries. A multiple of `len`, so the
        /// blocks land on whole coordinates (a tensor row inside a whole cell).
        offset: usize,
    },
}

impl LinearH {
    pub fn len(&self) -> usize {
        match self {
            Self::Dense(h) => h.len(),
            Self::Block { len, .. } | Self::Sparse { len, .. } | Self::Ones { len } => *len,
            Self::Tensor { len, .. } => *len,
            Self::Blocks { total, .. } => *total,
        }
    }

    /// The coins a `Tensor` functional is built from; empty for the other shapes.
    ///
    /// The proof carries these instead of the expanded vector: `m` field elements
    /// instead of `2^m`.
    pub fn coins(&self) -> &[Fp] {
        match self {
            Self::Tensor { coins, .. } | Self::Blocks { coins, .. } => coins,
            _ => &[],
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
            Self::Ones { len } => vec![Fp::ONE; *len],
            Self::Tensor { coins, len } => {
                let mut v: Vec<Fp> = Vec::with_capacity(1usize << coins.len());
                v.push(Fp::ONE);
                // Highest coin first: the doubling loop puts the *current* coin on
                // the lowest index bit, so this makes `coins[k]` select bit `k`.
                for &a in coins.iter().rev() {
                    let not_a = Fp::ONE - a;
                    let mut next = Vec::with_capacity(v.len() * 2);
                    for &x in &v {
                        next.push(x * not_a);
                        next.push(x * a);
                    }
                    v = next;
                }
                v.resize(*len, Fp::ZERO);
                v
            }
            Self::Blocks {
                coins,
                len,
                weights,
                total,
                offset,
            } => {
                let base = Self::Tensor {
                    coins: coins.clone(),
                    len: *len,
                }
                .dense();
                let mut h = Fp::zero_vec(*total);
                for (j, &w) in weights.iter().enumerate() {
                    let off = offset + j * len;
                    if off + len > *total {
                        break;
                    }
                    for i in 0..*len {
                        h[off + i] = w * base[i];
                    }
                }
                h
            }
        }
    }

    /// `h̃(z)`, evaluated using only the support (verifier side).
    pub fn eval_at(&self, z: &[Ext]) -> Ext {
        match self {
            Self::Dense(h) => fold_eval(h, z),
            Self::Block { start, values, .. } => values
                .iter()
                .enumerate()
                .map(|(t, &x)| eq(z, start + t) * Ext::from(x))
                .sum(),
            Self::Sparse { entries, .. } => {
                entries.iter().map(|&(i, x)| eq(z, i) * Ext::from(x)).sum()
            }
            Self::Ones { len } => {
                // `Σ_{j<2^m} eq(z, j) = 1` whenever `len = 2^m` and `|z| = m`.
                debug_assert!(len.is_power_of_two() && z.len() == len.trailing_zeros() as usize);
                Ext::ONE
            }
            Self::Tensor { coins, len } => {
                if *len == 1usize << coins.len() && z.len() == coins.len() {
                    coins
                        .iter()
                        .zip(z)
                        .map(|(&c, &zk)| {
                            (Ext::ONE - Ext::from(c)) * (Ext::ONE - zk) + Ext::from(c) * zk
                        })
                        .product()
                } else {
                    // Short vector padded with zeros: the product form does not apply.
                    fold_eval(&self.dense(), z)
                }
            }
            Self::Blocks {
                coins,
                len,
                weights,
                total,
                offset,
            } => {
                if !len.is_power_of_two() {
                    // A ragged block length truncates the tensor, so the product form
                    // does not apply; fold the explicit vector instead.
                    return fold_eval(&self.dense(), z);
                }
                let low = (len.trailing_zeros() as usize).min(z.len());
                let (z_low, z_high) = z.split_at(low);
                // `Σ_j w_j·eq(z_high, offset/len + j)`: only the blocks inside the cell
                // count, and the offset must be whole blocks.
                debug_assert_eq!(offset % len, 0, "block offset must be a multiple of len");
                let first = offset / len;
                let mut scaled = Ext::ZERO;
                for (j, &w) in weights.iter().enumerate() {
                    if (offset + j * len) >= *total {
                        break;
                    }
                    scaled += Ext::from(w) * eq(z_high, first + j);
                }
                let base: Ext = coins
                    .iter()
                    .take(low)
                    .zip(z_low)
                    .map(|(&c, &zk)| {
                        (Ext::ONE - Ext::from(c)) * (Ext::ONE - zk) + Ext::from(c) * zk
                    })
                    .product();
                base * scaled
            }
        }
    }
}

/// `Σ_j h_j·eq(z, j)` by folding one variable at a time: two extension-field
/// multiplications per entry instead of `m`, so `O(2^m)` rather than `O(2^m·m)`.
///
/// `h` may be shorter than the cube (`h.len() = 2^b < 2^m`): the entries above it are
/// zero, so the coordinates above `b` contribute the constant factor `Π(1 − z_k)`.
fn fold_eval(h: &[Fp], z: &[Ext]) -> Ext {
    if !h.len().is_power_of_two() {
        return h
            .iter()
            .enumerate()
            .map(|(j, &x)| eq(z, j) * Ext::from(x))
            .sum();
    }
    let b = h.len().trailing_zeros() as usize;
    let mut acc: Vec<Ext> = h.iter().map(|&x| Ext::from(x)).collect();
    for (k, &zk) in z.iter().enumerate().rev() {
        let not_z = Ext::ONE - zk;
        if k >= b {
            for x in acc.iter_mut() {
                *x *= not_z;
            }
        } else {
            let half = acc.len() / 2;
            for i in 0..half {
                acc[i] = acc[i] * not_z + acc[i + half] * zk;
            }
            acc.truncate(half);
        }
    }
    acc.first().copied().unwrap_or(Ext::ZERO)
}

/// `n` public field elements, derived from `log2(n)` transcript coins.
///
/// The vector is the multilinear equality tensor `r_j = Π_k (j_k ? α_k : 1 − α_k)`,
/// so the linear claim `⟨r, v⟩ = b` becomes a degree-1-per-variable polynomial in the
/// `m` coins: unsound with probability at most `m / |F|`, and it costs `m` transcript
/// squeezes instead of `n` on both sides.
///
/// The expansion is `O(n)` (`n − 1` multiplications, doubling one coin at a time), so
/// call sites that consume the vector entry-wise are no worse than before.
pub fn sample_vec(n: usize, tr: &mut Transcript) -> Vec<Fp> {
    sample(n, tr).dense()
}

/// [`sample`] plus the dense expansion — for sites that consume the vector
/// entry-wise but must only *store* the coins.
pub fn sample_coins(n: usize, tr: &mut Transcript) -> (Vec<Fp>, Vec<Fp>) {
    let h = sample(n, tr);
    (h.dense(), h.coins().to_vec())
}

/// The functional to use for a claim over `n` entries: an equality tensor built
/// from `log2(n)` coins. Sites that consume the vector entry-wise call
/// [`LinearH::dense`]; everything else keeps the `O(m)` structure.
pub fn sample(n: usize, tr: &mut Transcript) -> LinearH {
    let m = n.max(2).next_power_of_two().trailing_zeros() as usize;
    LinearH::Tensor {
        coins: (0..m).map(|_| tr.sample_val()).collect(),
        len: n,
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

    // GPU path: both operands stay resident on the device for the whole claim, so the
    // only traffic is one upload and one download. `None` (GPU off / too small) falls
    // through to the loop below, which produces exactly the same transcript.
    if let Some((rounds, z)) = crate::backend::linear_sumcheck(&hf, &vf, tr) {
        return (LinearClaimProof { rounds }, z, b);
    }

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
    use crate::field::sumcheck::mle_value;
    use crate::pcs::transcript::new_challenger;
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
    fn tensor_eval_matches_dense() {
        for n in [1usize, 2, 4, 8, 16, 64] {
            let mut tr = tr();
            let h = sample(n, &mut tr);
            let d = h.dense();
            let m = n.max(2).next_power_of_two().trailing_zeros() as usize;
            let z: Vec<Ext> = (0..m).map(|i| Ext::from(Fp::new(i as u64 + 3))).collect();
            let want: Ext = d
                .iter()
                .enumerate()
                .map(|(j, &x)| eq(&z, j) * Ext::from(x))
                .sum();
            assert_eq!(h.eval_at(&z), want, "n={n}");
        }
    }

    #[test]
    fn blocks_eval_matches_dense() {
        for (n, digits) in [(2usize, 2usize), (4, 2), (4, 3), (2, 5), (8, 2)] {
            let total = (n * digits).next_power_of_two();
            let mut tr = tr();
            let coins: Vec<Fp> = (0..n.trailing_zeros() as usize)
                .map(|_| tr.sample_val())
                .collect();
            let weights: Vec<Fp> = (0..digits).map(|j| Fp::new(1 << j)).collect();
            let h = LinearH::Blocks {
                coins,
                len: n,
                weights,
                total,
                offset: 0,
            };
            let d = h.dense();
            assert_eq!(d.len(), total);
            let m = total.trailing_zeros() as usize;
            let z: Vec<Ext> = (0..m).map(|i| Ext::from(Fp::new(i as u64 + 3))).collect();
            let want: Ext = d
                .iter()
                .enumerate()
                .map(|(j, &x)| eq(&z, j) * Ext::from(x))
                .sum();
            assert_eq!(h.eval_at(&z), want, "n={n} digits={digits}");
        }
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
