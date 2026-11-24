//! Segment-level **stacked** WHIR commitment.
//!
//! The earlier BaseFold layer commits one vector per root. That is fine for a
//! single claim, but the op-graph walk produces thousands of claims per segment,
//! and one root per claim is exactly the memory and FS problem a per-claim
//! commitment has.
//!
//! This module is the segment-level alternative the paper's architecture calls
//! for: **every table of one segment is stacked into a single WHIR commitment**,
//! sealed once, and every claim inside the segment is opened against that one
//! root in a single WHIR run.
//!
//! ```text
//!   segment s
//!     reserve table 0 .. table k   (field evaluations, no hashing yet)
//!     seal                          -> ONE Merkle root + prover data
//!     record claims                 (table, cols, point) -> claimed values
//!     prove                         -> ONE WHIR proof for all claims
//!     free everything               -> peak memory is O(segment)
//! ```
//!
//! Two conventions matter and are enforced here:
//!
//! * **Point order.** The project's `eq(z, j)` / `mle_value` are *LSB-first*
//!   (`point[0]` pairs with bit 0 of the index). WHIR's `Point` is *MSB-first*
//!   (`point[0]` is the most significant variable). [`whir_point`] reverses the
//!   point and prepends zeros for tables padded up to the first folding round.
//! * **Opening order.** The flat order of claims in the proof is: tables in
//!   protocol order, then each table's claims in record order. The verifier
//!   rebuilds the same order from the same claim list, so the caller must pass
//!   the claims in exactly the order the prover recorded them.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::field::{Ext, Fp};
use crate::pcs::blake3_mmcs::{Blake3Compress, Blake3Hasher};
use crate::pcs::transcript::{new_challenger, Challenger};
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
use serde::{Deserialize, Serialize};

type F = Fp;
type EF = Ext;
type Dft = Radix2DitParallel<F>;
type MT = MerkleTreeMmcs<F, F, Blake3Hasher, Blake3Compress, 2, 2>;
type Layout = SuffixProver<F, EF>;
type Prover = WhirProver<EF, F, Dft, MT, Challenger, Layout>;

/// One root per segment.
pub type Root = <Prover as MultilinearPcs<EF, Challenger>>::Commitment;
/// The retained stacked witness (one per segment).
pub type ProverData = <Prover as MultilinearPcs<EF, Challenger>>::ProverData;
/// One WHIR proof covering every claim of a segment.
pub type Proof = <Prover as MultilinearPcs<EF, Challenger>>::Proof;

/// Variables folded per WHIR round. Must match the protocol's `folding_factor`
/// and the padding `Witness::new` applies to short tables.
pub const FOLDING: usize = 4;

/// Soundness parameters. Stored in the [`SegmentCommitment`] so the verifier
/// rebuilds the exact same WHIR configuration the prover used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Params {
    pub security_level: usize,
    pub pow_bits: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            security_level: 100,
            pow_bits: 16,
        }
    }
}

fn build_prover(m: usize, params: Params) -> Prover {
    let protocol_params = ProtocolParameters {
        security_level: params.security_level,
        pow_bits: params.pow_bits,
        round_log_inv_rates: Vec::new(),
        folding_factor: FoldingFactor::Constant(FOLDING),
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 1,
    };
    let config = WhirConfig::<EF, F, Challenger>::new(m, protocol_params)
        .expect("WHIR config over Goldilocks");
    WhirProver::new(
        config,
        Dft::default(),
        MT::new(Blake3Hasher, Blake3Compress, 0),
    )
}

thread_local! {
    /// One WHIR prover per `(stacked arity, security, pow)`; rebuilt only when a
    /// segment of previously unseen size appears.
    static PROVERS: RefCell<HashMap<(usize, usize, usize), Prover>> =
        RefCell::new(HashMap::new());
}

fn with_prover<R>(m: usize, params: Params, f: impl FnOnce(&Prover) -> R) -> R {
    let key = (m, params.security_level, params.pow_bits);
    PROVERS.with(|cell| {
        if !cell.borrow().contains_key(&key) {
            cell.borrow_mut().insert(key, build_prover(m, params));
        }
        let borrow = cell.borrow();
        f(&borrow[&key])
    })
}

/// Public shape of one committed table: `2^m` rows, `width` columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TableShapePub {
    pub m: usize,
    pub width: usize,
}

/// The public half of a sealed segment: one root, the stacked arity, and the
/// table shapes needed to rebuild the opening protocol.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SegmentCommitment {
    pub root: Root,
    /// Variables of the stacked polynomial (the WHIR config's `num_variables`).
    pub m: usize,
    pub security_level: usize,
    pub pow_bits: usize,
    pub tables: Vec<TableShapePub>,
}

/// One prescribed opening: the listed columns of `table` at `point`, with the
/// values the prover claims.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Claim {
    pub table: usize,
    pub cols: Vec<usize>,
    /// LSB-first, length equal to the table's `m`.
    pub point: Vec<EF>,
    pub values: Vec<EF>,
}

struct TableData {
    m: usize,
    cols: Vec<Vec<F>>,
}

/// Reserves the tables of one segment, then seals them into a stacked commitment.
///
/// Claims are recorded **after** sealing (see [`SegmentProver`]): the segment's
/// root, and every challenge derived from it, is fixed before the op-graph walk
/// asks for openings — the same order the paper uses.
pub struct Committer {
    tables: Vec<TableData>,
}

impl Default for Committer {
    fn default() -> Self {
        Self::new()
    }
}

impl Committer {
    pub const fn new() -> Self {
        Self { tables: Vec::new() }
    }

    /// Reserve one table: `cols` column vectors of exactly `1 << m` evaluations.
    pub fn add_table(&mut self, m: usize, cols: Vec<Vec<F>>) -> usize {
        assert!(!cols.is_empty(), "a table needs at least one column");
        let n = 1usize << m;
        assert!(
            cols.iter().all(|c| c.len() == n),
            "every column must hold exactly 2^m evaluations"
        );
        self.tables.push(TableData { m, cols });
        self.tables.len() - 1
    }

    /// Reserve a one-column table from a flat vector, zero-padded to a power of two.
    /// Returns `(table, column)`.
    pub fn add_vec(&mut self, v: &[F]) -> (usize, usize) {
        assert!(!v.is_empty(), "cannot commit an empty vector");
        let m = v.len().next_power_of_two().trailing_zeros() as usize;
        let mut evals = v.to_vec();
        evals.resize(1 << m, F::ZERO);
        (self.add_table(m, vec![evals]), 0)
    }

    /// Stack every reserved table and commit the whole segment to one root.
    pub fn seal(self, params: Params) -> SegmentProver {
        let mut tables = Vec::with_capacity(self.tables.len());
        let mut shapes = Vec::with_capacity(self.tables.len());
        for t in &self.tables {
            let n = 1usize << t.m;
            let mut vals = F::zero_vec(n * t.cols.len());
            for (row, col) in t.cols.iter().enumerate() {
                // One row per polynomial: RowMajor with width `n` = one column.
                vals[row * n..(row + 1) * n].copy_from_slice(col);
            }
            tables.push(Table::new(RowMajorMatrix::new(vals, n)));
            shapes.push(TableShapePub {
                m: t.m,
                width: t.cols.len(),
            });
        }

        let witness = Witness::new(tables, FOLDING);
        let m = witness.num_variables();
        eprintln!(
            "  stack: {} tables, max 2^{}, cells 2^{:.1}, stacked 2^{}",
            shapes.len(),
            shapes.iter().map(|s| s.m).max().unwrap_or(0),
            (shapes
                .iter()
                .map(|s| (s.width as f64) * (1u64 << s.m) as f64)
                .sum::<f64>())
            .log2(),
            m
        );
        let t0 = std::time::Instant::now();
        let (root, pd) = with_prover(m, params, |p| {
            let mut ch = new_challenger();
            p.commit(witness, &mut ch).expect("WHIR stacked commit")
        });
        eprintln!("  committed in {:.1}s", t0.elapsed().as_secs_f64());

        SegmentProver {
            commitment: SegmentCommitment {
                root,
                m,
                security_level: params.security_level,
                pow_bits: params.pow_bits,
                tables: shapes,
            },
            pd,
            claims: Vec::new(),
        }
    }
}

/// A sealed segment: root, retained stacked witness, and the claims recorded
/// against it so far.
pub struct SegmentProver {
    commitment: SegmentCommitment,
    pd: ProverData,
    claims: Vec<Claim>,
}

impl SegmentProver {
    pub const fn commitment(&self) -> &SegmentCommitment {
        &self.commitment
    }

    pub fn claims(&self) -> &[Claim] {
        &self.claims
    }

    /// The logical column `col` of table `id` as base-field values (padded to `2^m`).
    ///
    /// Reads straight out of the retained stacked witness.
    pub fn column(&self, table: usize, col: usize) -> Vec<F> {
        self.pd.table(table).column(col).values().collect()
    }

    /// Record the linear claim `⟨h, col⟩` on this segment's table `id`.
    ///
    /// This is the weight-side counterpart of a cell segment's `claim_linear`: the
    /// quadratic sumcheck reduces the claim to one point evaluation, which is
    /// recorded here and paid for by this tensor's proof. Used by `embed`, whose
    /// operand is a sparse gather of the embedding rows.
    pub fn claim_linear(
        &mut self,
        table: usize,
        col: usize,
        h: &crate::pcs::linear_claim::LinearH,
        tr: &mut crate::pcs::transcript::Transcript,
    ) -> (
        Ext,
        crate::pcs::linear_claim::LinearClaimProof,
        Vec<Ext>,
        Ext,
    ) {
        let n = 1usize << self.table_m(table);
        assert_eq!(h.len(), n, "`h` must be padded to the column length");
        let v = self.column_logical(table, col);
        assert_eq!(v.len(), n);
        let (proof, z, b) = crate::pcs::linear_claim::prove(&v, h, tr);
        let opened = self.claim(table, &[col], &z);
        assert_eq!(opened.len(), 1);
        (b, proof, z, opened[0])
    }

    /// The arity `m` of table `id` (log `2` of its logical length).
    pub fn table_m(&self, table: usize) -> usize {
        self.commitment.tables[table].m
    }

    /// The column truncated to the table's **logical** length `2^m`.
    ///
    /// This is the weight-side read path: a weight tensor kept in its own segment is
    /// read here so a claim can be run on it without copying it into another stack.
    pub fn column_logical(&self, table: usize, col: usize) -> Vec<F> {
        let m = self.commitment.tables[table].m;
        let mut c: Vec<F> = self.pd.table(table).column(col).values().collect();
        c.truncate(1usize << m);
        c
    }

    pub fn num_claims(&self) -> usize {
        self.claims.len()
    }

    /// Record an opening of `cols` at `point`, returning the claimed values.
    ///
    /// `point` is LSB-first and its length must equal the table's `m`. The values
    /// are evaluated straight out of the retained stacked witness, so recording a
    /// claim copies no segment data.
    pub fn claim(&mut self, table: usize, cols: &[usize], point: &[EF]) -> Vec<EF> {
        let shape = self.commitment.tables[table];
        assert_eq!(point.len(), shape.m, "point arity must equal the table's m");
        let padded_m = shape.m.max(FOLDING);
        // Padding appends zero rows, so the extra coordinates lie in the high bits.
        let mut padded_point = point.to_vec();
        padded_point.resize(padded_m, EF::ZERO);

        let t = self.pd.table(table);
        let values: Vec<EF> = cols
            .iter()
            .map(|&c| {
                t.column(c)
                    .values()
                    .enumerate()
                    .map(|(row, v)| crate::field::eq(&padded_point, row) * EF::from(v))
                    .sum()
            })
            .collect();

        self.claims.push(Claim {
            table,
            cols: cols.to_vec(),
            point: point.to_vec(),
            values: values.clone(),
        });
        values
    }

    /// Record an opening whose value the caller already computed.
    ///
    /// [`Self::claim`] evaluates the full `2^m`-entry MLE; when the value is available
    /// cheaply (a single row of a huge table, say), this is the way to record it.
    pub fn claim_value(&mut self, table: usize, cols: &[usize], point: &[EF], values: Vec<EF>) {
        let shape = self.commitment.tables[table];
        assert_eq!(point.len(), shape.m, "point arity must equal the table's m");
        assert_eq!(values.len(), cols.len(), "one value per column");
        self.claims.push(Claim {
            table,
            cols: cols.to_vec(),
            point: point.to_vec(),
            values,
        });
    }

    /// The logical values `[start, start + len)` of a column, without copying the rest.
    pub fn column_range(&self, table: usize, col: usize, start: usize, len: usize) -> Vec<F> {
        self.pd
            .table(table)
            .column(col)
            .values()
            .skip(start)
            .take(len)
            .collect()
    }

    /// Produce the single stacked WHIR proof covering every recorded claim.
    ///
    /// `ch` must be the segment's transcript: this absorbs the root and then
    /// runs the WHIR opening on the same challenger state.
    pub fn prove(self, ch: &mut Challenger) -> Proof {
        let Self {
            commitment,
            pd,
            claims,
        } = self;
        assert!(!claims.is_empty(), "no claims recorded for this segment");
        let params = Params {
            security_level: commitment.security_level,
            pow_bits: commitment.pow_bits,
        };
        with_prover(commitment.m, params, |p| {
            p.observe_commitment(&commitment.root, ch);
            let (protocol, points, _) = build_protocol(&commitment.tables, &claims, FOLDING)
                .expect("claim shape matches the committed tables");
            <Prover as PrescribedPointPcs<EF, Challenger>>::open_at(p, pd, &protocol, &points, ch)
                .expect("WHIR stacked open_at")
        })
    }
}

/// Absorb a segment root into a transcript.
///
/// This is the same binding the WHIR prover applies, exposed so a caller can
/// derive cross-segment challenges (segment boundaries) after every root is fixed.
pub fn observe_root(root: &Root, ch: &mut Challenger) {
    p3_challenger::CanObserve::observe(ch, root.clone());
}

/// Sample an `m`-coordinate LSB-first evaluation point from the transcript.
pub fn sample_point(ch: &mut Challenger, m: usize) -> Vec<EF> {
    use p3_challenger::FieldChallenger;
    (0..m)
        .map(|_| FieldChallenger::sample_algebra_element(ch))
        .collect()
}

/// Verify a segment proof against `claims` (given in the prover's record order).
///
/// Returns `true` only when the WHIR run accepts **and** every opened value
/// matches the claim. A malformed claim list rejects rather than panics.
pub fn verify(
    commitment: &SegmentCommitment,
    claims: &[Claim],
    proof: &Proof,
    ch: &mut Challenger,
) -> bool {
    if claims.is_empty() {
        return false;
    }
    let params = Params {
        security_level: commitment.security_level,
        pow_bits: commitment.pow_bits,
    };
    with_prover(commitment.m, params, |p| {
        let Some((protocol, points, order)) = build_protocol(&commitment.tables, claims, FOLDING)
        else {
            return false;
        };
        p.observe_commitment(&commitment.root, ch);
        match <Prover as PrescribedPointPcs<EF, Challenger>>::verify_at(
            p,
            &commitment.root,
            proof,
            &protocol,
            &points,
            ch,
        ) {
            Ok(evals) => {
                if evals.len() != order.len() {
                    return false;
                }
                order
                    .iter()
                    .enumerate()
                    .all(|(k, &ci)| evals[k].current() == claims[ci].values.as_slice())
            }
            Err(_) => false,
        }
    })
}

/// Build the opening protocol and its point list from the recorded claims.
///
/// Flat order is: tables in protocol order, then each table's claims in record
/// order. The returned `Vec<usize>` maps each flat slot back to its claim index.
/// Returns `None` on any shape disagreement so the verifier fails closed.
fn build_protocol(
    tables: &[TableShapePub],
    claims: &[Claim],
    folding: usize,
) -> Option<(OpeningProtocol, Vec<Point<EF>>, Vec<usize>)> {
    let mut per_table: Vec<Vec<usize>> = vec![Vec::new(); tables.len()];
    for (i, c) in claims.iter().enumerate() {
        let t = tables.get(c.table)?;
        if c.point.len() != t.m || c.cols.is_empty() || c.cols.len() != c.values.len() {
            return None;
        }
        if c.cols.iter().any(|&k| k >= t.width) {
            return None;
        }
        // `TableSpec::new` rejects repeats inside one side; fail closed instead.
        let mut sorted = c.cols.clone();
        sorted.sort_unstable();
        if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
            return None;
        }
        per_table[c.table].push(i);
    }

    let mut specs = Vec::with_capacity(tables.len());
    let mut points = Vec::with_capacity(claims.len());
    let mut order = Vec::with_capacity(claims.len());

    for (t, shape) in tables.iter().enumerate() {
        // Short tables are zero-padded to the first folding round by the witness.
        let padded_m = shape.m.max(folding);
        let batches: Vec<OpeningBatch<usize>> = per_table[t]
            .iter()
            .map(|&i| OpeningBatch::new(claims[i].cols.clone(), Vec::new()))
            .collect();
        specs.push(TableSpec::new(
            TableShape::new(padded_m, shape.width),
            batches,
        ));
        for &i in &per_table[t] {
            points.push(whir_point(&claims[i].point, padded_m));
            order.push(i);
        }
    }

    Some((OpeningProtocol::new(specs), points, order))
}

/// Convert an LSB-first point to WHIR's MSB-first `Point`, prepending zeros for
/// the variables a table gained when padded up to the first folding round.
///
/// Padding appends zero rows, i.e. it sets the *most significant* index bits, so
/// the padding coordinates are the leading MSB-first coordinates and must be 0.
fn whir_point(point_lsb: &[EF], padded_m: usize) -> Point<EF> {
    assert!(
        padded_m >= point_lsb.len(),
        "padded arity must be at least the point's arity"
    );
    let mut out = EF::zero_vec(padded_m - point_lsb.len());
    out.extend(point_lsb.iter().rev().copied());
    Point::new(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    fn rand_vec(rng: &mut StdRng, n: usize) -> Vec<F> {
        (0..n)
            .map(|_| F::new(rng.random::<u64>() % 0xFFFF_FFFF_0000_0001))
            .collect()
    }

    fn rand_point(rng: &mut StdRng, m: usize) -> Vec<EF> {
        (0..m)
            .map(|_| EF::from(F::new(rng.random::<u64>() % 0xFFFF_FFFF_0000_0001)))
            .collect()
    }

    #[test]
    fn whir_point_reverses_and_pads() {
        let z = vec![EF::ONE, EF::from(F::new(2)), EF::from(F::new(3))];
        let p = whir_point(&z, 5);
        let coords: Vec<EF> = p.as_slice().to_vec();
        assert_eq!(
            coords,
            vec![
                EF::ZERO,
                EF::ZERO,
                EF::from(F::new(3)),
                EF::from(F::new(2)),
                EF::ONE
            ]
        );
    }

    #[test]
    fn stacked_segment_roundtrip() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut c = Committer::new();
        let t0 = c.add_table(4, vec![rand_vec(&mut rng, 16)]);
        let t1 = c.add_table(5, vec![rand_vec(&mut rng, 32), rand_vec(&mut rng, 32)]);
        let mut s = c.seal(Params::default());

        // Two claims on table 0, three on table 1 (two of them sharing a point).
        let p0 = rand_point(&mut rng, 4);
        let p1 = rand_point(&mut rng, 4);
        let q0 = rand_point(&mut rng, 5);
        let q1 = rand_point(&mut rng, 5);

        let v0 = s.claim(t0, &[0], &p0);
        let v1 = s.claim(t1, &[0, 1], &q0);
        let v2 = s.claim(t0, &[0], &p1);
        let v3 = s.claim(t1, &[1], &q1);
        let v4 = s.claim(t1, &[0], &q0);
        assert_eq!(v0.len(), 1);
        assert_eq!(v1.len(), 2);
        assert_eq!(v4, vec![v1[0]]);
        let _ = (v2, v3);

        let commitment = s.commitment().clone();
        let claims = s.claims().to_vec();
        assert_eq!(claims.len(), 5);

        let proof = s.prove(&mut new_challenger());

        let mut vch = new_challenger();
        assert!(verify(&commitment, &claims, &proof, &mut vch));

        // A single flipped value in one claim is rejected.
        let mut bad = claims.clone();
        bad[3].values[0] += EF::ONE;
        let mut vch = new_challenger();
        assert!(!verify(&commitment, &bad, &proof, &mut vch));
    }

    #[test]
    fn wrong_segment_rejects() {
        let mut rng = StdRng::seed_from_u64(11);
        let mut a = Committer::new();
        let ta = a.add_vec(&rand_vec(&mut rng, 16));
        let mut sa = a.seal(Params::default());
        let pa = rand_point(&mut rng, 4);
        sa.claim(ta.0, &[ta.1], &pa);

        let mut b = Committer::new();
        let tb = b.add_vec(&rand_vec(&mut rng, 16));
        let mut sb = b.seal(Params::default());
        let pb = rand_point(&mut rng, 4);
        sb.claim(tb.0, &[tb.1], &pb);

        assert_ne!(sa.commitment().root, sb.commitment().root);

        let claims_a = sa.claims().to_vec();
        let proof_a = sa.prove(&mut new_challenger());

        // A's proof must not verify against B's root.
        let mut vch = new_challenger();
        assert!(!verify(sb.commitment(), &claims_a, &proof_a, &mut vch));
    }

    #[test]
    fn padded_small_table_verifies() {
        // m = 2 is below the folding factor, so the witness pads it to 4.
        let mut rng = StdRng::seed_from_u64(13);
        let mut c = Committer::new();
        let (t, col) = c.add_vec(&rand_vec(&mut rng, 4));
        let mut s = c.seal(Params::default());
        let z = rand_point(&mut rng, 2);
        s.claim(t, &[col], &z);

        let commitment = s.commitment().clone();
        let claims = s.claims().to_vec();
        let proof = s.prove(&mut new_challenger());
        let mut vch = new_challenger();
        assert!(verify(&commitment, &claims, &proof, &mut vch));
    }

    #[test]
    fn malformed_claim_rejects() {
        let mut rng = StdRng::seed_from_u64(17);
        let mut c = Committer::new();
        let (t, col) = c.add_vec(&rand_vec(&mut rng, 16));
        let mut s = c.seal(Params::default());
        let z = rand_point(&mut rng, 4);
        s.claim(t, &[col], &z);

        let commitment = s.commitment().clone();
        let mut claims = s.claims().to_vec();
        let proof = s.prove(&mut new_challenger());

        // Wrong point arity fails closed.
        let mut bad = claims.clone();
        bad[0].point.pop();
        let mut vch = new_challenger();
        assert!(!verify(&commitment, &bad, &proof, &mut vch));

        // Duplicate column inside one batch fails closed.
        claims[0].cols = vec![0, 0];
        claims[0].values = vec![EF::ZERO, EF::ZERO];
        let mut vch = new_challenger();
        assert!(!verify(&commitment, &claims, &proof, &mut vch));
    }

    /// Peak-memory probe for one realistic-size segment: 8 columns of 2^18 cells
    /// (16 MB of trace). Run with
    /// `cargo test --release --lib whir_stacked::tests::segment_peak_memory_probe -- --ignored --nocapture`
    /// under `/usr/bin/time -l` to read the resident-set maximum.
    #[test]
    #[ignore = "memory probe: run explicitly with --ignored"]
    fn segment_peak_memory_probe() {
        let m = 18;
        let width = 8;
        let n = 1usize << m;
        let rng = &mut StdRng::seed_from_u64(99);
        let cols: Vec<Vec<F>> = (0..width).map(|_| rand_vec(rng, n)).collect();
        let cells = cols.iter().map(|c| c.len()).sum::<usize>();
        println!("trace cells: {cells} ({:.1} MB)", cells as f64 * 8.0 / 1e6);

        let mut c = Committer::new();
        let t = c.add_table(m, cols);
        let mut s = c.seal(Params::default());
        println!("stacked variables: {}", s.commitment().m);

        let z = rand_point(rng, m);
        for col in 0..width {
            s.claim(t, &[col], &z);
        }

        let commitment = s.commitment().clone();
        let claims = s.claims().to_vec();
        let proof = s.prove(&mut new_challenger());
        println!(
            "proof bytes (bincode): {}",
            bincode::serialize(&proof).map(|v| v.len()).unwrap_or(0)
        );

        let mut vch = new_challenger();
        assert!(verify(&commitment, &claims, &proof, &mut vch));
    }
}
