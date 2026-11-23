//! Shared FRI-commitment plumbing for the GKR succinct verifier: the committed
//! cell/weight types, the row-slice tie, the lookup bounds, and the public weight
//! commitment used by the commit/prove/verify split.

use std::collections::HashMap;

use crate::field::{Ext, Fp};
use crate::model::spec::{ModelSpec, Op};
use crate::model::{Trace, WeightMatrix, Weights};
use crate::pcs::fri::{self, Challenger, Pcs, Prover};
use crate::pcs::zk::{
    prove_inner_product_committed_given, verify_inner_product_committed, InnerProductProof,
    Openings,
};
use p3_field::PrimeCharacteristicRing;

fn next_pow2(n: usize) -> usize {
    n.next_power_of_two()
}

fn pad(mut v: Vec<Fp>, n: usize) -> Vec<Fp> {
    v.resize(next_pow2(n), Fp::ZERO);
    v
}

fn f(x: i64) -> Fp {
    if x < 0 {
        -Fp::new(x.unsigned_abs())
    } else {
        Fp::new(x as u64)
    }
}

/// Maximum rows per committed weight chunk. Bounds the FRI commit (and its
/// Merkle tree) to a small constant, so committing a 100M-element embedding no
/// longer needs a multi-GB tree.
pub const CHUNK_ROWS: usize = 2048;

/// Sequence positions batched per matmul. The block-diagonal `W ⊗ I_batch` grows
/// as `batch²`, so a small fixed batch keeps it (and the `u` challenge vector)
/// bounded; the sequence is processed in `SUB_BATCH`-wide slices.
pub const SUB_BATCH: usize = 16;

/// Number of row-chunks a weight of `rows` rows splits into.
pub fn num_chunks(rows: usize) -> usize {
    (rows + CHUNK_ROWS - 1) / CHUNK_ROWS
}

/// The row range `[start, end)` of a weight's `i`-th chunk.
pub fn chunk_row_range(rows: usize, i: usize) -> (usize, usize) {
    let start = i * CHUNK_ROWS;
    let end = (start + CHUNK_ROWS).min(rows);
    (start, end)
}

/// The chunk name for a weight's `i`-th chunk. Single-chunk weights keep their
/// name (so small weights need no op-graph changes); multi-chunk weights use
/// `{name}.{i}`.
pub fn chunk_name(name: &str, rows: usize, i: usize) -> String {
    if num_chunks(rows) == 1 {
        name.to_string()
    } else {
        format!("{name}.{i}")
    }
}

/// The lookup bounds. Fixed function tables (requant/silu/exp/round/sign) are
/// committed once over `[lo, hi]` and checked via the succinct LogUp.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CommittedConfig {
    pub requant_digits: usize,
    pub silu_lo: i64,
    pub silu_hi: i64,
    pub exp_lo: i64,
    pub sign_lo: i64,
    pub sign_hi: i64,
    /// Bounds of the logits (the score domain of the argmax range check).
    pub score_lo: i64,
    pub score_hi: i64,
}

impl CommittedConfig {
    /// Derive the lookup/range-check bounds from the actual trace, so the fixed
    /// function tables cover exactly the values this model produces.
    ///
    /// * `silu_lo/hi` — the raw gate-projection inputs to `silu` (unclamped).
    /// * `exp_lo` — the most negative `score − max` seen by causal softmax.
    /// * `sign_lo/hi` — the MoE routing delta `score·experts − i − tau`.
    /// * `score_lo/hi` — the head logits range.
    ///
    /// Both prover and verifier build the same tables from these (public) bounds, so
    /// a bound that fails to cover the trace only ever breaks the prover's own proof.
    pub fn derive(spec: &ModelSpec, trace: &Trace) -> Self {
        let cells: HashMap<&str, &crate::model::Cell> = trace
            .segments
            .iter()
            .flat_map(|s| s.cells.iter())
            .map(|c| (c.name.as_str(), c))
            .collect();

        let mut silu_lo = i64::MAX;
        let mut silu_hi = i64::MIN;
        let mut exp_lo = 0i64;
        let mut sign_lo = i64::MAX;
        let mut sign_hi = i64::MIN;
        let mut score_lo = i64::MAX;
        let mut score_hi = i64::MIN;

        if let Some(c) = cells.get("logits") {
            for &v in &c.values {
                score_lo = score_lo.min(v);
                score_hi = score_hi.max(v);
            }
        }

        for op in &spec.ops {
            match op {
                Op::Silu { input, .. } => {
                    if let Some(c) = cells.get(input.as_str()) {
                        for &v in &c.values {
                            silu_lo = silu_lo.min(v);
                            silu_hi = silu_hi.max(v);
                        }
                    }
                }
                Op::CausalSoftmax { input, .. } => {
                    if let Some(c) = cells.get(input.as_str()) {
                        let (mut mn, mut mx) = (i64::MAX, i64::MIN);
                        for &v in &c.values {
                            mn = mn.min(v);
                            mx = mx.max(v);
                        }
                        exp_lo = exp_lo.min(mn - mx);
                    }
                }
                Op::TopK { input, .. } => {
                    if let Some(c) = cells.get(input.as_str()) {
                        let (mut mn, mut mx) = (i64::MAX, i64::MIN);
                        for &v in &c.values {
                            mn = mn.min(v);
                            mx = mx.max(v);
                        }
                        let e = c.cols as i64;
                        sign_lo = sign_lo.min(-(mx - mn) * e - e);
                        sign_hi = sign_hi.max((mx - mn) * e + e);
                    }
                }
                _ => {}
            }
        }

        // Fallbacks for models that omit a nonlinearity (empty table domain).
        if silu_hi < silu_lo {
            silu_lo = -1;
            silu_hi = 1;
        }
        if sign_hi < sign_lo {
            sign_lo = -1;
            sign_hi = 1;
        }
        if score_hi < score_lo {
            score_lo = -1;
            score_hi = 1;
        }

        Self {
            requant_digits: 4,
            silu_lo,
            silu_hi,
            exp_lo,
            sign_lo,
            sign_hi,
            score_lo,
            score_hi,
        }
    }
}

/// A committed cell or weight: clear values (prover side) plus the padded field
/// values and their FRI commitment.
pub struct CellCommit {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<i64>,
    pub fp: Vec<Fp>,
    pub com: fri::Commitment,
    pub pd: fri::ProverData,
    pub dom: fri::Domain,
}

fn commit_one(pcs: &Pcs, name: &str, rows: usize, cols: usize, values: Vec<i64>) -> CellCommit {
    let fp = pad(values.iter().map(|&x| f(x)).collect(), values.len());
    let (com, pd, dom) = fri::commit_vec(pcs, &fp);
    CellCommit {
        name: name.to_string(),
        rows,
        cols,
        values,
        fp,
        com,
        pd,
        dom,
    }
}

/// Commit a weight from its compact `i8` values: the `fp`/commitment/tree come from
/// the `i8` data, but the redundant `i64` `values` copy is omitted (the matmul/norm
/// read `i8` from the original `weights` and widen transiently).
fn commit_weight(pcs: &Pcs, name: &str, rows: usize, cols: usize, vals8: &[i8]) -> CellCommit {
    let fp = pad(vals8.iter().map(|&x| f(x as i64)).collect(), vals8.len());
    let (com, pd, dom) = fri::commit_vec(pcs, &fp);
    CellCommit {
        name: name.to_string(),
        rows,
        cols,
        values: Vec::new(), // no i64 copy; i8 read from `weights`
        fp,
        com,
        pd,
        dom,
    }
}

/// Commit all chunks of one weight, returning `(chunk_name, CellCommit)` pairs in
/// chunk order. Deterministic, so it matches the offline `commit_weights_public`.
pub fn commit_weight_chunks(pcs: &Pcs, w: &WeightMatrix) -> Vec<(String, CellCommit)> {
    let n = num_chunks(w.rows);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let (start, end) = chunk_row_range(w.rows, i);
        let vals8 = &w.values[start * w.cols..end * w.cols];
        let name = chunk_name(&w.name, w.rows, i);
        out.push((
            name.clone(),
            commit_weight(pcs, &name, end - start, w.cols, vals8),
        ));
    }
    out
}

/// The public commitment of one weight (name + dimensions + commitment + domain),
/// shared between the commit and verify phases.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct WeightCommit {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub com: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom: fri::Domain,
}

/// Commit the model's weights (offline / commit phase), returning only the public
/// commitments. The commitment is deterministic, so the prove phase re-derives the
/// same commitment; the verifier binds the proof to these pre-shared roots.
pub fn commit_weights_public(pcs: &Pcs, weights: &Weights) -> Vec<WeightCommit> {
    let mut out = Vec::new();
    for w in weights {
        let n = num_chunks(w.rows);
        for i in 0..n {
            let (start, end) = chunk_row_range(w.rows, i);
            let vals: Vec<i64> = w.values[start * w.cols..end * w.cols]
                .iter()
                .map(|&v| v as i64)
                .collect();
            let fp = pad(vals.iter().map(|&x| f(x)).collect(), vals.len());
            let (com, _pd, dom) = fri::commit_vec(pcs, &fp);
            out.push(WeightCommit {
                name: chunk_name(&w.name, w.rows, i),
                rows: end - start,
                cols: w.cols,
                com,
                dom,
            });
        }
    }
    out
}

/// Commit every trace cell once, keyed by name (weights are committed lazily
/// by the prover on first use).
pub fn commit_cells(pcs: &Pcs, trace: &Trace) -> HashMap<String, CellCommit> {
    let mut cells = HashMap::new();
    for seg in &trace.segments {
        for c in &seg.cells {
            cells.insert(
                c.name.clone(),
                commit_one(pcs, &c.name, c.rows, c.cols, c.values.clone()),
            );
        }
    }
    cells
}

/// Ties a committed row slice to the committed full cell it was sliced from:
/// `row[t] == full[start + t]` for all `t < d`, checked by one random linear
/// combination (Schwartz–Zippel).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RowTie {
    pub r: Vec<Fp>,
    pub b_row: Fp,
    pub b_full: Fp,
    pub claim_row: InnerProductProof,
    pub claim_full: InnerProductProof,
    pub full_len: usize,
    pub row_com: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub row_dom: fri::Domain,
    pub full_com: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub full_dom: fri::Domain,
}

/// One input of an `add` op: the commitment/domain/linear-claim of a single
/// addend, tied to the sum by a shared challenge.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TiedClaim {
    pub com: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom: fri::Domain,
    pub b: Fp,
    pub claim: InnerProductProof,
}

pub fn tie_row(ctx: &mut Prover, row: &CellCommit, full: &CellCommit, idx: &[usize]) -> RowTie {
    let d = idx.len();
    let mut tr = fri::Transcript::new(&ctx.challenger);
    tr.observe(&row.com);
    tr.observe(&full.com);
    let mut r: Vec<Fp> = (0..d).map(|_| tr.sample_val()).collect();
    r.resize(next_pow2(d), Fp::ZERO);
    let b_row: Fp = r.iter().zip(&row.fp).map(|(&ri, &vi)| ri * vi).sum();
    let full_len = next_pow2(full.values.len());
    let mut r_placed = vec![Fp::ZERO; full_len];
    for (t, &i) in idx.iter().enumerate() {
        r_placed[i] = r[t];
    }
    let b_full: Fp = r_placed
        .iter()
        .zip(&full.fp)
        .map(|(&ri, &vi)| ri * vi)
        .sum();
    let claim_row =
        prove_inner_product_committed_given(ctx, &row.com, &row.pd, row.dom, &row.fp, &r);
    let claim_full = prove_inner_product_committed_given(
        ctx, &full.com, &full.pd, full.dom, &full.fp, &r_placed,
    );
    RowTie {
        r,
        b_row,
        b_full,
        claim_row,
        claim_full,
        full_len,
        row_com: row.com.clone(),
        row_dom: row.dom,
        full_com: full.com.clone(),
        full_dom: full.dom,
    }
}

pub fn verify_tie(
    challenger: &Challenger,
    idx: &[usize],
    tie: &RowTie,
    values: &[Ext],
    acc: &mut Openings,
) -> bool {
    let mut r_placed = vec![Fp::ZERO; tie.full_len];
    for (t, &i) in idx.iter().enumerate() {
        r_placed[i] = tie.r[t];
    }
    verify_inner_product_committed(
        challenger,
        &tie.row_com,
        tie.row_dom,
        &tie.r,
        tie.b_row,
        &tie.claim_row,
        values,
        acc,
    ) && verify_inner_product_committed(
        challenger,
        &tie.full_com,
        tie.full_dom,
        &r_placed,
        tie.b_full,
        &tie.claim_full,
        values,
        acc,
    ) && tie.b_row == tie.b_full
}

/// Commit every weight tensor to its own WHIR root (the **pre-shared map**).
///
/// This is the commit phase's output, and it is what the verifier passes as
/// `published` to [`crate::gkr::driver::verify_segment_with_weights`]. The roots are
/// computed with the one-flat-column convention of
/// [`crate::gkr::segment::CellSegment::weight_segment`], so the prover's lazily
/// committed tensors reproduce them exactly.
///
/// Each tensor's witness is dropped as soon as its root is computed, so the commit
/// phase's peak memory is one tensor.
pub fn commit_weights_whir(
    weights: &[WeightMatrix],
    params: crate::pcs::whir_stacked::Params,
) -> Vec<(String, crate::pcs::whir_stacked::SegmentCommitment)> {
    weights
        .iter()
        .map(|w| {
            let seg = crate::gkr::segment::CellSegment::weight_segment(w, params);
            (w.name.clone(), seg.commitment().clone())
        })
        .collect()
}

#[cfg(test)]
mod whir_tests {
    use super::*;
    use crate::pcs::whir_stacked::Params;

    #[test]
    fn commit_phase_roots_match_the_drivers() {
        let weights = vec![
            WeightMatrix::new("a", 4, 1, vec![1, -2, 3, -4]),
            WeightMatrix::new("b", 2, 1, vec![7, -8]),
        ];
        let params = Params::default();
        let published = commit_weights_whir(&weights, params);
        assert_eq!(published.len(), 2);
        assert_ne!(published[0].1.root, published[1].1.root);

        // The prover's lazy commit must reproduce each root exactly.
        for (name, com) in &published {
            let w = weights.iter().find(|w| &w.name == name).unwrap();
            let seg = crate::gkr::segment::CellSegment::weight_segment(w, params);
            assert_eq!(&seg.commitment().root, &com.root, "root for {name}");
            assert_eq!(&seg.commitment().tables, &com.tables, "shape for {name}");
        }
    }
}
