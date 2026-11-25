//! Trace-**cell** segmentation on the stacked-WHIR commitment layer.
//!
//! [`crate::pcs::whir_stacked`] stacks *tables*; the op graph talks in *cells*
//! (`rows × cols` integer matrices from [`crate::model::Cell`]). This module is
//! the translation between the two, and it is the segmentation pipeline the
//! op-graph walk consumes:
//!
//! ```text
//!   trace.segments[s].cells
//!     -> one table per cell:  2^ceil(log2 rows) x cols, row-padded with zeros
//!     -> ONE stacked WHIR root for the whole segment
//!     -> a cell claim opens ALL its columns at one point (one opening batch)
//!     -> ONE WHIR proof per segment
//! ```
//!
//! Why one table per cell with `cols` columns instead of one flat vector:
//! a matrix cell is naturally a multilinear map over *rows*, so `x̃(z)` is the
//! `cols`-tuple of its column MLEs at `z`. Opening a cell is therefore one
//! batch over `0..cols`, which is what an op's sumcheck input claim needs.
//!
//! Row padding appends zeros, so the extra coordinates sit in the *high* index
//! bits. The point handed to the PCS stays in the project's LSB-first basis and
//! [`crate::pcs::whir_stacked`] prepends the padding coordinates.

use std::collections::HashMap;

use crate::field::arith::bool_point;
use crate::field::sumcheck::{mle_eval, RoundPoly, RoundPoly3};
use crate::field::{Ext, Fp};
use crate::gkr::protocol::{concat, eq_partial, matrix_mle, round_poly3_mul, round_poly_product};
use crate::model::{Cell, WeightMatrix};
use crate::pcs::linear_claim::{self, LinearClaimProof, LinearH};
use crate::pcs::transcript::{Challenger, Transcript};
use crate::pcs::whir_stacked::{self, Committer, Params, SegmentProver};
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};

/// Signed integer to field element (the project's canonical conversion).
pub(crate) fn signed(x: i64) -> Fp {
    if x < 0 {
        -Fp::new(x.unsigned_abs())
    } else {
        Fp::new(x as u64)
    }
}

/// Variables of the padded row count: `ceil(log2 rows)`.
fn padded_m(rows: usize) -> usize {
    rows.next_power_of_two().trailing_zeros() as usize
}

/// Commit every weight tensor to its own WHIR root (the **pre-shared map**).
///
/// This is the commit phase's output, and it is what the verifier passes as
/// `published` to [`crate::gkr::driver::verify_segment_with_weights`]. The roots are
/// computed with the one-flat-column convention of
/// [`CellSegment::weight_segment`], so the prover's lazily committed tensors
/// reproduce them exactly.
///
/// Each tensor's witness is dropped as soon as its root is computed, so the commit
/// phase's peak memory is one tensor.
pub fn commit_weights_whir(
    weights: &[WeightMatrix],
    params: Params,
) -> Vec<(String, whir_stacked::SegmentCommitment)> {
    weights
        .iter()
        .map(|w| {
            let seg = CellSegment::weight_segment(w, params);
            (w.name.clone(), seg.commitment().clone())
        })
        .collect()
}

/// Public description of one committed cell.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CellMeta {
    pub name: String,
    /// Flat table height (before zero-padding to a power of two).
    pub rows: usize,
    /// Table width — `1` for a flattened cell.
    pub cols: usize,
    /// The cell's extent as the model states it (`seq × hidden`, `vocab`, …).
    /// The verifier needs these to resolve rows and heads; a flattened table's
    /// height alone cannot say.
    pub logical_rows: usize,
    pub logical_cols: usize,
}

impl CellMeta {
    /// Variables of the padded table (`ceil(log2 rows)`).
    pub fn m(&self) -> usize {
        padded_m(self.rows)
    }
}

/// The public half of a sealed cell segment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CellSegmentCommitment {
    pub pcs: whir_stacked::SegmentCommitment,
    /// Cell table order; index is the cell id used by [`CellClaim::cell`].
    pub cells: Vec<CellMeta>,
}

impl CellSegmentCommitment {
    pub fn cell(&self, name: &str) -> Option<usize> {
        self.cells.iter().position(|c| c.name == name)
    }
}

/// A claim on one cell: a subset of its columns at one LSB-first point.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CellClaim {
    pub cell: usize,
    /// Column indices opened, in transcript order.
    pub cols: Vec<usize>,
    /// LSB-first, length = `CellMeta::m()`.
    pub point: Vec<Ext>,
    /// One value per entry of `cols`.
    pub values: Vec<Ext>,
}

/// Prover-side segment: sealed stack plus the cell name → table map.
pub struct CellSegment {
    prover: SegmentProver,
    cells: Vec<CellMeta>,
    /// Name → cell id, so the op graph does not scan the table list per lookup.
    ids: HashMap<String, usize>,
    /// Cell id → the PCS table it is packed into, and its first column inside it.
    ///
    /// Cells of equal arity share one table (one column each), so the verifier pays
    /// its per-table opening work a handful of times per segment instead of once per
    /// derived cell. Both sides derive the same map from the cell arities alone.
    table_of: Vec<usize>,
    col_of: Vec<usize>,
    claims: Vec<CellClaim>,
}

/// Cells of equal arity in the same table, `PACK_WIDTH` columns per table.
///
/// Deterministic in the cell arities and ids, so the verifier rebuilds it from
/// `CellSegmentCommitment::cells` without any extra wire data.
pub fn packing(metas: &[CellMeta]) -> (Vec<usize>, Vec<usize>) {
    const PACK_WIDTH: usize = 64;
    let mut order: Vec<usize> = (0..metas.len()).collect();
    order.sort_by_key(|&i| (metas[i].m(), i));
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    for &i in &order {
        if cur.len() == PACK_WIDTH || cur.first().is_some_and(|&j| metas[j].m() != metas[i].m()) {
            groups.push(std::mem::take(&mut cur));
        }
        cur.push(i);
    }
    if !cur.is_empty() {
        groups.push(cur);
    }
    let mut table = vec![0usize; metas.len()];
    let mut col = vec![0usize; metas.len()];
    for (t, g) in groups.iter().enumerate() {
        for (c, &i) in g.iter().enumerate() {
            table[i] = t;
            col[i] = c;
        }
    }
    (table, col)
}

impl CellSegment {
    /// Seal from per-column field values (the path for cells that are not small
    /// integers, e.g. LogUp inverses and fingerprints).
    pub(crate) fn from_columns(
        metas: Vec<CellMeta>,
        columns: Vec<Vec<Vec<Fp>>>,
        params: Params,
    ) -> Self {
        assert_eq!(metas.len(), columns.len());
        let (table_of, col_of) = packing(&metas);
        let ntables = table_of.iter().copied().max().map_or(0, |t| t + 1);
        let mut per_table: Vec<(usize, Vec<Vec<Fp>>)> =
            (0..ntables).map(|_| (0, Vec::new())).collect();
        for (i, cols) in columns.into_iter().enumerate() {
            let t = table_of[i];
            per_table[t].0 = metas[i].m();
            per_table[t].1.extend(cols);
        }
        let mut c = Committer::new();
        for (m, cols) in per_table {
            c.add_table(m, cols);
        }
        Self {
            prover: c.seal(params),
            ids: metas
                .iter()
                .enumerate()
                .map(|(i, m)| (m.name.clone(), i))
                .collect(),
            cells: metas,
            table_of,
            col_of,
            claims: Vec::new(),
        }
    }

    /// Reserve every cell as one padded, `cols`-wide table and seal the segment.
    pub fn seal(cells: &[Cell], params: Params) -> Self {
        let mut metas = Vec::with_capacity(cells.len());
        let mut tables = Vec::with_capacity(cells.len());
        for cell in cells {
            let col_values: Vec<Vec<Fp>> = (0..cell.cols)
                .map(|j| {
                    let n = 1usize << padded_m(cell.rows);
                    let mut col = Fp::zero_vec(n);
                    for i in 0..cell.rows {
                        col[i] = signed(cell.values[i * cell.cols + j]);
                    }
                    col
                })
                .collect();
            metas.push(CellMeta {
                name: cell.name.clone(),
                rows: cell.rows,
                cols: cell.cols,
                logical_rows: cell.rows,
                logical_cols: cell.cols,
            });
            tables.push(col_values);
        }
        Self::from_columns(metas, tables, params)
    }

    /// Metadata of a cell (panics on an out-of-range id).
    pub fn cells_meta(&self, cell: usize) -> &CellMeta {
        &self.cells[cell]
    }

    /// The prover for one weight tensor as a **one-flat-column** table.
    ///
    /// This is the pre-shared convention: the commit phase and the prover must agree
    /// on it, or the roots they compute differ and `check_preshared` rejects every
    /// weight claim. Values are signed integers widened to the base field, padded to
    /// a power of two.
    pub fn weight_segment(w: &WeightMatrix, params: Params) -> SegmentProver {
        let n = w.values.len().next_power_of_two();
        let m = n.trailing_zeros() as usize;
        let mut col = Fp::zero_vec(n);
        for (i, &v) in w.values.iter().enumerate() {
            col[i] = signed(v as i64);
        }
        let mut c = Committer::new();
        c.add_table(m, vec![col]);
        c.seal(params)
    }

    /// Seal **one weight tensor** as its own stacked root, without materialising
    /// an `i64` copy of it. The commit phase pre-shares the resulting root; the
    /// prove phase re-derives it (the root is a pure function of the values).
    pub fn seal_weight(w: &WeightMatrix, params: Params) -> Self {
        Self::seal_weights(std::slice::from_ref(&w), params)
    }

    /// Seal several weight tensors as one stack (use when the tensors a segment
    /// consumes are known before the walk starts).
    pub fn seal_weights(ws: &[&WeightMatrix], params: Params) -> Self {
        let mut c = Committer::new();
        let mut metas = Vec::with_capacity(ws.len());
        for w in ws {
            let m = padded_m(w.rows);
            let n = 1usize << m;
            let cols: Vec<Vec<Fp>> = (0..w.cols)
                .map(|j| {
                    let mut col = Fp::zero_vec(n);
                    for i in 0..w.rows {
                        col[i] = signed(w.values[i * w.cols + j] as i64);
                    }
                    col
                })
                .collect();
            c.add_table(m, cols);
            metas.push(CellMeta {
                name: w.name.clone(),
                rows: w.rows,
                cols: w.cols,
                logical_rows: w.rows,
                logical_cols: w.cols,
            });
        }
        Self {
            prover: c.seal(params),
            ids: metas
                .iter()
                .enumerate()
                .map(|(i, m)| (m.name.clone(), i))
                .collect(),
            // Weight tensors stay one table each: `column_logical` and the weight
            // proofs address them by table id.
            table_of: (0..metas.len()).collect(),
            col_of: vec![0; metas.len()],
            cells: metas,
            claims: Vec::new(),
        }
    }

    pub fn commitment(&self) -> CellSegmentCommitment {
        CellSegmentCommitment {
            pcs: self.prover.commitment().clone(),
            cells: self.cells.clone(),
        }
    }

    pub fn claims(&self) -> &[CellClaim] {
        &self.claims
    }

    /// The id of `name`, panicking when the segment does not hold it.
    pub fn id(&self, name: &str) -> usize {
        *self
            .ids
            .get(name)
            .unwrap_or_else(|| panic!("unknown cell `{name}`"))
    }

    /// Logical column values (unpadded length `1 << m`).
    pub fn values(&self, cell: usize, col: usize) -> Vec<Fp> {
        let mut c = self
            .prover
            .column(self.table_of[cell], self.col_of[cell] + col);
        c.truncate(1usize << self.cells[cell].m());
        c
    }

    /// Record a claim of columns `cols` of `cell` at `point`, returning the values.
    pub fn claim_id(&mut self, cell: usize, cols: &[usize], point: &[Ext]) -> Vec<Ext> {
        let pcs_cols: Vec<usize> = cols.iter().map(|&c| self.col_of[cell] + c).collect();
        let values = self.prover.claim(self.table_of[cell], &pcs_cols, point);
        self.claims.push(CellClaim {
            cell,
            cols: cols.to_vec(),
            point: point.to_vec(),
            values: values.clone(),
        });
        values
    }

    /// Record a claim of every column of `name` at `point`, returning the values.
    pub fn claim(&mut self, name: &str, point: &[Ext]) -> Vec<Ext> {
        let id = self.id(name);
        let cols: Vec<usize> = (0..self.cells[id].cols).collect();
        self.claim_id(id, &cols, point)
    }

    /// Record a claim of a *subset* of `name`'s columns (e.g. one head's slice).
    pub fn claim_cols(&mut self, name: &str, cols: &[usize], point: &[Ext]) -> Vec<Ext> {
        let id = self.id(name);
        self.claim_id(id, cols, point)
    }

    /// One WHIR proof for every claim of this segment.
    pub fn prove(self, ch: &mut Challenger) -> whir_stacked::Proof {
        self.prover.prove(ch)
    }

    /// Record the linear claim `⟨h, col⟩ = b` for one column of `name`.
    ///
    /// This is the replacement for the FRI constant-coefficient bridge: the
    /// claim is reduced by a quadratic sumcheck to one MLE opening `ṽ(z)`, which
    /// is recorded on the same segment and paid for by the same WHIR proof.
    ///
    /// `h` must already be padded to the table's arity (use [`LinearH::Dense`] of
    /// length `2^m`); returns the claim value `b` and the sumcheck proof.
    pub fn claim_linear(
        &mut self,
        name: &str,
        col: usize,
        h: &LinearH,
        tr: &mut Transcript,
    ) -> (Ext, LinearClaimProof) {
        let id = self.id(name);
        self.claim_linear_id(id, col, h, tr)
    }

    /// Same as [`Self::claim_linear_id`], over a **view** of the cell.
    ///
    /// The sumcheck runs on the view's logical vector; the resulting opening is
    /// recorded at `view.point(z)`, so the whole cell is opened once at a point
    /// whose extra coordinates are Boolean.
    pub fn claim_linear_view(
        &mut self,
        cell: usize,
        view: &View,
        h: &LinearH,
        tr: &mut Transcript,
    ) -> (Ext, LinearClaimProof) {
        let m = self.cells[cell].m();
        let bits = view.bits(m);
        assert_eq!(h.len(), 1usize << bits, "h must match the view's arity");

        let mut col = self.prover.column(self.table_of[cell], self.col_of[cell]);
        col.truncate(1usize << m);
        let v = view.values(&col, m);
        assert_eq!(v.len(), 1usize << bits);

        let (proof, z, b) = linear_claim::prove(&v, h, tr);
        let point = view.point(&z, m);
        let opened = self
            .prover
            .claim(self.table_of[cell], &[self.col_of[cell]], &point);
        self.claims.push(CellClaim {
            cell,
            cols: vec![0],
            point,
            values: opened,
        });
        (b, proof)
    }

    /// Record an opening of a **view** of `cell` at the logical point `z`.
    ///
    /// [`Self::claim_id`] evaluates the cell's whole `2^m`-entry MLE; when the point
    /// pins the coordinates outside the view to Boolean values, the MLE collapses to
    /// the view's `2^bits` slice, which is what this evaluates.
    pub fn claim_id_view(&mut self, cell: usize, view: &View, z: &[Ext]) -> Vec<Ext> {
        let m = self.cells[cell].m();
        assert_eq!(view.bits(m), z.len(), "view arity");
        let mut col = self.prover.column(self.table_of[cell], self.col_of[cell]);
        col.truncate(1usize << m);
        let v = view.values(&col, m);
        let value: Ext = v
            .iter()
            .enumerate()
            .map(|(j, &x)| crate::field::eq(z, j) * Ext::from(x))
            .sum();
        let point = view.point(z, m);
        self.prover.claim_value(
            self.table_of[cell],
            &[self.col_of[cell]],
            &point,
            vec![value.clone()],
        );
        self.claims.push(CellClaim {
            cell,
            cols: vec![0],
            point,
            values: vec![value.clone()],
        });
        vec![value]
    }

    /// Same as [`Self::claim_linear`], addressed by cell id.
    pub fn claim_linear_id(
        &mut self,
        id: usize,
        col: usize,
        h: &LinearH,
        tr: &mut Transcript,
    ) -> (Ext, LinearClaimProof) {
        let n = 1usize << self.cells[id].m();
        assert_eq!(h.len(), n, "`h` must be padded to the cell's column length");

        let v: Vec<Fp> = {
            let mut c = self.prover.column(self.table_of[id], self.col_of[id] + col);
            // The stored column is padded up to the first folding round; the claim
            // runs over the cell's *logical* cube and the PCS pads the point back.
            c.truncate(n);
            c
        };
        assert_eq!(v.len(), n);
        let (proof, z, b) = linear_claim::prove(&v, h, tr);
        // The sumcheck leaves one point evaluation to open on this same segment.
        let opened = self
            .prover
            .claim(self.table_of[id], &[self.col_of[id] + col], &z);
        self.claims.push(CellClaim {
            cell: id,
            cols: vec![col],
            point: z,
            values: opened,
        });
        (b, proof)
    }
}

/// Replay a linear claim's sumcheck, returning `(z, claim_final)`.
pub fn linear_verify(
    proof: &LinearClaimProof,
    b: Ext,
    tr: &mut Transcript,
) -> Option<(Vec<Ext>, Ext)> {
    linear_claim::verify(proof, b, tr)
}

/// The terminal identity of a linear claim, given the PCS opening `ṽ(z)`.
pub fn linear_finish(h: &LinearH, z: &[Ext], claim: Ext, v_at_z: Ext) -> bool {
    linear_claim::finish(h, z, claim, v_at_z)
}

/// The value a cell claim opened at exactly `point` (verifier side).
///
/// Linear claims and point claims share the `CellClaim` list, so a verifier can
/// find the opening the prover recorded for the reduced sumcheck point.
pub fn claim_value_at(
    commitment: &CellSegmentCommitment,
    claims: &[CellClaim],
    name: &str,
    point: &[Ext],
) -> Option<Ext> {
    let cell = commitment.cell(name)?;
    claim_value_at_id(commitment, claims, cell, point)
}

/// Same, addressed by cell id.
pub fn claim_value_at_id(
    commitment: &CellSegmentCommitment,
    claims: &[CellClaim],
    cell: usize,
    point: &[Ext],
) -> Option<Ext> {
    claim_value_at_col(commitment, claims, cell, 0, point)
}

/// The value a claim recorded for `cell`'s column `col` at `point`.
pub fn claim_value_at_col(
    commitment: &CellSegmentCommitment,
    claims: &[CellClaim],
    cell: usize,
    col: usize,
    point: &[Ext],
) -> Option<Ext> {
    if commitment.cells.get(cell).is_none() {
        return None;
    }
    if !profile::enabled() {
        return claims
            .iter()
            .find(|c| c.cell == cell && c.point == point && c.cols.as_slice() == [col])
            .map(|c| c.values[0]);
    }
    let mut scanned = 0u64;
    let found = claims
        .iter()
        .find(|c| {
            scanned += 1;
            c.cell == cell && c.point == point && c.cols.as_slice() == [col]
        })
        .map(|c| c.values[0]);
    profile::lookup(scanned);
    found
}

/// Cheap counters for the verifier's inner loop (`FLASHPROVE_PROFILE=1` prints them).
/// The flag is read once, so the default path carries a single atomic load.
pub mod profile {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;

    pub static LOOKUPS: AtomicU64 = AtomicU64::new(0);
    pub static SCANNED: AtomicU64 = AtomicU64::new(0);

    pub fn enabled() -> bool {
        static FLAG: OnceLock<bool> = OnceLock::new();
        *FLAG.get_or_init(|| std::env::var("FLASHPROVE_PROFILE").is_ok_and(|v| v == "1"))
    }

    /// Record one claim-list scan that examined `n` entries.
    pub fn lookup(n: u64) {
        LOOKUPS.fetch_add(1, Ordering::Relaxed);
        SCANNED.fetch_add(n, Ordering::Relaxed);
    }

    /// Print and reset the counters.
    pub fn report(what: &str) {
        if !enabled() {
            return;
        }
        let l = LOOKUPS.swap(0, Ordering::Relaxed);
        let s = SCANNED.swap(0, Ordering::Relaxed);
        if l > 0 {
            eprintln!("  profile {what}: {l} claim lookups, {s} entries scanned");
        }
    }
}

/// Plans a segment's tables by name, then seals them.
///
/// The op walk needs this because a stacked commitment is sealed **before** any
/// claim: tables that an op derives (`h = x⊙γ`, `y = h·inv_q`, the softmax
/// denominator, …) must exist when the root is built. Two passes use it:
///
/// ```text
///   pass 1  plan:  cells.cell("h", rows, cols, h_values)   // reserve + write
///   seal:          let seg = cells.seal(params);
///   pass 2  claim: seg.claim("h", &point)
/// ```
///
/// Insertion is **idempotent by name** and first-writer-wins, so two passes that
/// derive the same cell independently agree, and a cell read by several ops is
/// written once.
#[derive(Default)]
pub struct SegmentCells {
    metas: Vec<CellMeta>,
    tables: Vec<Vec<Vec<Fp>>>,
    ids: HashMap<String, usize>,
}

impl SegmentCells {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reserve (and write) an integer-valued table. Returns its cell id; an
    /// existing name keeps its first value and id.
    pub fn cell(&mut self, name: &str, rows: usize, cols: usize, values: Vec<i64>) -> usize {
        assert_eq!(values.len(), rows * cols, "cell length");
        let n = 1usize << padded_m(rows);
        let columns: Vec<Vec<Fp>> = (0..cols)
            .map(|j| {
                let mut col = Fp::zero_vec(n);
                for i in 0..rows {
                    col[i] = signed(values[i * cols + j]);
                }
                col
            })
            .collect();
        self.push(name, rows, cols, columns)
    }

    /// Reserve (and write) a field-valued table (fingerprints, inverses, …).
    pub fn cell_fp(&mut self, name: &str, rows: usize, cols: usize, values: Vec<Fp>) -> usize {
        assert_eq!(values.len(), rows * cols, "cell length");
        let n = 1usize << padded_m(rows);
        let columns: Vec<Vec<Fp>> = (0..cols)
            .map(|j| {
                let mut col = Fp::zero_vec(n);
                for i in 0..rows {
                    col[i] = values[i * cols + j];
                }
                col
            })
            .collect();
        self.push(name, rows, cols, columns)
    }

    /// Reserve a cell **flattened to one column** of `rows·cols` entries, keeping the
    /// logical extents in the metadata.
    pub fn cell_logical(
        &mut self,
        name: &str,
        rows: usize,
        cols: usize,
        values: Vec<i64>,
    ) -> usize {
        assert_eq!(values.len(), rows * cols, "cell length");
        let n = rows * cols;
        let d = n.next_power_of_two();
        let mut col = Fp::zero_vec(d);
        for (i, &v) in values.iter().enumerate() {
            col[i] = signed(v);
        }
        let id = self.push(name, n, 1, vec![col]);
        if let Some(m) = self.metas.get_mut(id) {
            m.logical_rows = rows;
            m.logical_cols = cols;
        }
        id
    }

    fn push(&mut self, name: &str, rows: usize, cols: usize, columns: Vec<Vec<Fp>>) -> usize {
        if let Some(&id) = self.ids.get(name) {
            return id;
        }
        let id = self.metas.len();
        self.metas.push(CellMeta {
            name: name.to_string(),
            rows,
            cols,
            logical_rows: rows,
            logical_cols: cols,
        });
        self.tables.push(columns);
        self.ids.insert(name.to_string(), id);
        id
    }

    /// The id of `name`, if planned.
    pub fn id(&self, name: &str) -> Option<usize> {
        self.ids.get(name).copied()
    }

    pub fn len(&self) -> usize {
        self.metas.len()
    }

    pub fn is_empty(&self) -> bool {
        self.metas.is_empty()
    }

    /// Seal every planned cell into one stacked segment.
    pub fn seal(self, params: Params) -> CellSegment {
        CellSegment::from_columns(self.metas, self.tables, params)
    }
}

/// A pointwise-sum claim: `out = Σ inputs`, cell by cell.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AddSegmentProof {
    pub out: usize,
    pub b_out: Ext,
    pub c_out: LinearClaimProof,
    pub inputs: Vec<(usize, Ext, LinearClaimProof)>,
}

/// Claim `out = Σ inputs` pointwise.
///
/// One random linear combination `r` turns the `n` pointwise identities into the
/// single base-field identity `⟨r, out⟩ = Σ_i ⟨r, in_i⟩`, and each side is then a
/// [`LinearH::Dense`] linear claim (one sumcheck + one opening per operand).
pub fn claim_add(
    seg: &mut CellSegment,
    out: usize,
    inputs: &[usize],
    tr: &mut Transcript,
) -> AddSegmentProof {
    let n = 1usize << seg.cells[out].m();
    tr.observe_val(Fp::new(out as u64));
    for &i in inputs {
        tr.observe_val(Fp::new(i as u64));
    }
    let h = crate::pcs::linear_claim::sample(n, tr);

    let (b_out, c_out) = seg.claim_linear_id(out, 0, &h, tr);
    let mut terms = Vec::with_capacity(inputs.len());
    for &i in inputs {
        let (b, c) = seg.claim_linear_id(i, 0, &h, tr);
        terms.push((i, b, c));
    }
    // No prover-side assertion on `b_out == Σ b`: a wrong witness must produce a
    // proof the verifier rejects, not a panic.

    AddSegmentProof {
        out,
        b_out,
        c_out,
        inputs: terms,
    }
}

/// Verify a [`AddSegmentProof`].
pub fn verify_add(
    commitment: &CellSegmentCommitment,
    claims: &[CellClaim],
    proof: &AddSegmentProof,
    tr: &mut Transcript,
) -> bool {
    let Some(meta) = commitment.cells.get(proof.out) else {
        return false;
    };
    let n = 1usize << meta.m();
    if proof
        .inputs
        .iter()
        .any(|(i, _, _)| *i >= commitment.cells.len())
    {
        return false;
    }

    tr.observe_val(Fp::new(proof.out as u64));
    for (i, _, _) in &proof.inputs {
        tr.observe_val(Fp::new(*i as u64));
    }
    let h = crate::pcs::linear_claim::sample(n, tr);

    let Some((z, cf)) = linear_verify(&proof.c_out, proof.b_out, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, proof.out, &z) else {
        return false;
    };
    if !linear_finish(&h, &z, cf, v) {
        return false;
    }

    let mut sum = Ext::ZERO;
    for (i, b, c) in &proof.inputs {
        let Some((z, cf)) = linear_verify(c, *b, tr) else {
            return false;
        };
        let Some(v) = claim_value_at_id(commitment, claims, *i, &z) else {
            return false;
        };
        if !linear_finish(&h, &z, cf, v) {
            return false;
        }
        sum += *b;
    }
    proof.b_out == sum
}

/// A sum-of-squares claim `Σ_i x[i]² = s`.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SumSquaresProof {
    pub x: usize,
    pub d: usize,
    pub sum: Ext,
    pub rounds: Vec<RoundPoly>,
}

/// Claim `Σ_i x[i]²`, the RMSNorm denominator.
///
/// This replaces the `x2 = x⊙x` table plus an all-halves point evaluation: a
/// single degree-2 sumcheck over `x(t)²` ends at `x(z)` and needs no auxiliary
/// commitment.
pub fn claim_sum_squares(seg: &mut CellSegment, x: usize, tr: &mut Transcript) -> SumSquaresProof {
    let d = seg.cells[x].m();
    tr.observe_val(Fp::new(x as u64));

    let mut xf: Vec<Ext> = seg.values(x, 0).into_iter().map(Ext::from).collect();
    let sum: Ext = xf.iter().map(|&v| v * v).sum();

    let mut rounds = Vec::with_capacity(d);
    let mut z = Vec::with_capacity(d);
    for _ in 0..d {
        // g(t) = Σ_{j'} (x₀ + t·Δ)².
        let (mut p0, mut p1, mut p2) = (Ext::ZERO, Ext::ZERO, Ext::ZERO);
        for j in 0..xf.len() / 2 {
            let x0 = xf[2 * j];
            let dx = xf[2 * j + 1] - x0;
            p0 += x0 * x0;
            p1 += Ext::from(Fp::new(2)) * x0 * dx;
            p2 += dx * dx;
        }
        let g = RoundPoly([p0, p1, p2]);
        for &coef in &g.0 {
            tr.observe_ext(coef);
        }
        let r = tr.sample_ext();
        rounds.push(g);
        z.push(r);

        let half = xf.len() / 2;
        for j in 0..half {
            xf[j] = xf[2 * j] + r * (xf[2 * j + 1] - xf[2 * j]);
        }
        xf.truncate(half);
    }

    seg.claim_id(x, &[0], &z);

    SumSquaresProof { x, d, sum, rounds }
}

/// Verify a [`SumSquaresProof`].
pub fn verify_sum_squares(
    commitment: &CellSegmentCommitment,
    claims: &[CellClaim],
    proof: &SumSquaresProof,
    tr: &mut Transcript,
) -> bool {
    if commitment.cells.get(proof.x).is_none() || proof.rounds.len() != proof.d {
        return false;
    }
    tr.observe_val(Fp::new(proof.x as u64));

    let mut claim = proof.sum;
    let mut z = Vec::with_capacity(proof.d);
    for g in &proof.rounds {
        if g.eval(Ext::ZERO) + g.eval(Ext::ONE) != claim {
            return false;
        }
        for &coef in &g.0 {
            tr.observe_ext(coef);
        }
        let r = tr.sample_ext();
        claim = g.eval(r);
        z.push(r);
    }

    let Some(v) = claim_value_at_id(commitment, claims, proof.x, &z) else {
        return false;
    };
    v * v == claim
}

/// An opening of a **pre-shared weight tensor** at a point.
///
/// This is the (c) mechanism: a weight operand is opened against its own root, so
/// no bytes of it are duplicated into any activation segment. The tensor's proof is
/// kept per *tensor* (one WHIR proof covering every claim made on it), not per op,
/// so a layer pays one weight proof per matrix rather than one per claim.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WeightOpen {
    pub name: String,
    /// The table id inside the weight's own segment.
    pub cell: usize,
    pub point: Vec<Ext>,
    pub value: Ext,
}

/// `c = a ⊙ w`, with `w` in its own pre-shared segment.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MulWeightProof {
    /// The activation operand.
    pub a: usize,
    /// The product (activation) cell.
    pub c: usize,
    pub d: usize,
    pub r: Vec<Ext>,
    pub istar: Vec<Ext>,
    pub c_val: Ext,
    pub a_val: Ext,
    pub w: WeightOpen,
    pub rounds: Vec<RoundPoly3>,
}

/// Claim `c[i] = a[i]·w[i]` with `w` living in its own segment.
///
/// Same sumcheck as [`claim_mul`]; the difference is that the weight's terminal
/// evaluation is recorded on the weight segment, so it is paid for by that
/// tensor's proof instead of by the activation segment's.
#[allow(clippy::too_many_arguments)]
pub fn claim_mul_weight(
    seg: &mut CellSegment,
    a: usize,
    weight: &mut SegmentProver,
    weight_cell: usize,
    c: usize,
    weight_name: &str,
    tr: &mut Transcript,
) -> MulWeightProof {
    let d = seg.cells_meta(a).m();
    assert_eq!(seg.cells_meta(c).m(), d, "mul operands must share an arity");
    for id in [a, c] {
        tr.observe_val(Fp::new(id as u64));
    }

    let to_ext = |v: Vec<Fp>| -> Vec<Ext> { v.into_iter().map(Ext::from).collect() };
    let af = to_ext(seg.values(a, 0));
    let wf = to_ext(weight.column_logical(weight_cell, 0));
    let cf = to_ext(seg.values(c, 0));
    assert_eq!(wf.len(), af.len(), "weight arity");

    let r: Vec<Ext> = (0..d).map(|_| tr.sample_ext()).collect();
    let c_val = mle_eval(&cf, &r);

    let mut rounds = Vec::with_capacity(d);
    let mut istar = Vec::with_capacity(d);
    for j in 0..d {
        let round = round_poly3_mul(&af, &wf, &r, d, j, &istar);
        for coeff in &round.0 {
            tr.observe_ext(*coeff);
        }
        let rj = tr.sample_ext();
        rounds.push(round);
        istar.push(rj);
    }

    let a_val = mle_eval(&af, &istar);
    let w_val = mle_eval(&wf, &istar);
    seg.claim_id(a, &[0], &istar);
    seg.claim_id(c, &[0], &r);
    // The weight operand is opened on its own segment.
    let opened = weight.claim(weight_cell, &[0], &istar);
    assert_eq!(opened[0], w_val, "weight opening matches the sumcheck");

    MulWeightProof {
        a,
        c,
        d,
        r,
        w: WeightOpen {
            name: weight_name.to_string(),
            cell: weight_cell,
            point: istar.clone(),
            value: w_val,
        },
        istar,
        c_val,
        a_val,
        rounds,
    }
}

/// Verify a [`MulWeightProof`].
///
/// `w_com`/`w_proof` are the tensor's pre-shared commitment and its per-tensor
/// proof (one per `(name, tensor)` pair, covering every claim on it).
#[allow(clippy::too_many_arguments)]
pub fn verify_mul_weight(
    commitment: &CellSegmentCommitment,
    claims: &[CellClaim],
    w_com: &whir_stacked::SegmentCommitment,
    w_proof: &whir_stacked::Proof,
    proof: &MulWeightProof,
    tr: &mut Transcript,
) -> bool {
    let (a, c, d) = (proof.a, proof.c, proof.d);
    let Some(ma) = commitment.cells.get(a) else {
        return false;
    };
    if commitment.cells.get(c).is_none()
        || ma.m() != d
        || proof.rounds.len() != d
        || proof.r.len() != d
        || proof.istar.len() != d
    {
        return false;
    }
    for id in [a, c] {
        tr.observe_val(Fp::new(id as u64));
    }

    let r: Vec<Ext> = (0..d).map(|_| tr.sample_ext()).collect();
    if r != proof.r {
        return false;
    }

    let mut claim = proof.c_val;
    let mut istar = Vec::with_capacity(d);
    for round in &proof.rounds {
        if round.eval(Ext::ZERO) + round.eval(Ext::ONE) != claim {
            return false;
        }
        for coeff in &round.0 {
            tr.observe_ext(*coeff);
        }
        let rj = tr.sample_ext();
        istar.push(rj);
        claim = round.eval(rj);
    }
    if istar != proof.istar {
        return false;
    }
    if claim != eq_partial(&r, &istar) * proof.a_val * proof.w.value {
        return false;
    }

    if claim_value_at_id(commitment, claims, a, &istar) != Some(proof.a_val)
        || claim_value_at_id(commitment, claims, c, &r) != Some(proof.c_val)
    {
        return false;
    }

    // The weight's opening, against its own root.
    let w_claim = whir_stacked::Claim {
        table: proof.w.cell,
        cols: vec![0],
        point: proof.w.point.clone(),
        values: vec![proof.w.value],
    };
    whir_stacked::verify(
        w_com,
        &[w_claim],
        w_proof,
        &mut crate::pcs::transcript::new_challenger(),
    )
}

/// `C = W·B` with `W` a pre-shared weight tensor (step 5 of the port).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MatmulWeightProof {
    pub ln: usize,
    pub lm: usize,
    pub lp: usize,
    /// The activation operands.
    pub b: usize,
    pub c: usize,
    pub rounds: Vec<RoundPoly>,
    pub a_val: Ext,
    pub b_val: Ext,
    pub c_val: Ext,
    /// `W`'s opening, on its own root.
    pub w: WeightOpen,
}

/// Claim `C = W·B` with the weight read from its own segment.
///
/// Same sumcheck as [`claim_matmul`]; the only difference is where the three
/// surviving evaluations are opened: `W`'s on the weight root, `B`/`C`'s on the
/// activation segment. No weight bytes enter the activation stack.
///
/// The three operands are laid out logically as `A = W[n][k]` (weight row-major),
/// `B = X[t][k]` read **transposed** as `[k][t]`, and `C = Y[t][n]` read transposed
/// as `[n][t]`, where `t` is the batch (sequence) index. `ln`, `lm`, `lp` are the
/// arities of `n`, `k`, `t`, so a single batched matmul is one claim:
///
/// ```text
///   p[k] = Σ_n r̃(n)·W[n][k]        O(N·K)
///   q[k] = Σ_t s̃(t)·X[t][k]        O(L·K)
///   c    = Σ_{t,n} r̃(n)s̃(t)·Y[t][n]  O(L·N)
/// ```
///
/// i.e. one pass over the data — never the `2^lm`-fold `matrix_mle` sweep.
#[allow(clippy::too_many_arguments)]
pub fn claim_matmul_weight(
    seg: &mut CellSegment,
    weight: &mut SegmentProver,
    weight_cell: usize,
    w_view: &View,
    (b, c): (usize, usize),
    (b_view, c_view): (&View, &View),
    weight_name: &str,
    tr: &mut Transcript,
    (ln, lm, lp): (usize, usize, usize),
) -> MatmulWeightProof {
    for id in [b, c] {
        tr.observe_val(Fp::new(id as u64));
    }
    tr.observe_val(Fp::new(weight_cell as u64));

    let r: Vec<Ext> = (0..ln).map(|_| tr.sample_ext()).collect();
    let s: Vec<Ext> = (0..lp).map(|_| tr.sample_ext()).collect();

    let to_ext = |v: Vec<Fp>| -> Vec<Ext> { v.into_iter().map(Ext::from).collect() };
    let (m_w, m_b, m_c) = (
        weight.table_m(weight_cell),
        seg.cells_meta(b).m(),
        seg.cells_meta(c).m(),
    );
    assert_eq!(w_view.bits(m_w), ln + lm, "W view dimensions");
    assert_eq!(b_view.bits(m_b), lm + lp, "B view dimensions");
    assert_eq!(c_view.bits(m_c), ln + lp, "C view dimensions");

    let bv = to_ext(b_view.values(&seg.values(b, 0), m_b));
    let cv = to_ext(c_view.values(&seg.values(c, 0), m_c));
    assert_eq!(bv.len(), 1 << (lm + lp), "B dimensions");
    assert_eq!(cv.len(), 1 << (ln + lp), "C dimensions");

    let c_val = matrix_mle(&cv, &r, &s);
    drop(cv);

    // p[k] = Ã(r,k): contract the weight's rows against `r`. One pass over the
    // weight, in base-field values (no `Ext` copy of a whole tensor).
    let wv = {
        let column = weight.column_logical(weight_cell, 0);
        match w_view {
            View::Whole => column,
            _ => w_view.values(&column, m_w),
        }
    };
    assert_eq!(wv.len(), 1 << (ln + lm), "W dimensions");
    // CPU: measured ~4 ns per multiply-add here, and the GPU formulation (one thread
    // per column, a 2^ln-long serial chain) is no faster — see `backend::contract`.
    let mut p = vec![Ext::ZERO; 1 << lm];
    for (n, rn) in (0..(1usize << ln))
        .map(|n| (n, crate::field::eq(&r, n)))
        .filter(|&(_, rn)| rn != Ext::ZERO)
    {
        let base = n << lm;
        for k in 0..(1 << lm) {
            p[k] += rn * Ext::from(wv[base + k]);
        }
    }
    drop(wv);

    // q[k] = B̃(k,s): contract the batch axis of `B` against `s`.
    let eq_s: Vec<Ext> = (0..(1usize << lp))
        .map(|t| crate::field::eq(&s, t))
        .collect();
    let mut q = vec![Ext::ZERO; 1 << lm];
    for (k, qk) in q.iter_mut().enumerate() {
        let base = k << lp;
        for (t, &et) in eq_s.iter().enumerate() {
            *qk += et * bv[base + t];
        }
    }
    drop(bv);

    let mut kstar = Vec::with_capacity(lm);
    let mut rounds = Vec::with_capacity(lm);
    for i in 0..lm {
        let round = round_poly_product(&p, &q, lm, i, &kstar);
        for coeff in &round.0 {
            tr.observe_ext(*coeff);
        }
        let ri = tr.sample_ext();
        rounds.push(round);
        kstar.push(ri);
    }

    let a_val = mle_eval(&p, &kstar);
    let b_val = mle_eval(&q, &kstar);

    let a_point = w_view.point(&concat(&kstar, &r), m_w);
    let opened_a = weight.claim(weight_cell, &[0], &a_point);
    let opened_b = seg.claim_id_view(b, b_view, &concat(&s, &kstar))[0];
    let opened_c = seg.claim_id_view(c, c_view, &concat(&s, &r))[0];
    assert_eq!(opened_a[0], a_val, "W opening matches the sumcheck");
    assert_eq!(opened_b, b_val, "B opening matches the sumcheck");
    assert_eq!(opened_c, c_val, "C opening matches the sumcheck");

    MatmulWeightProof {
        ln,
        lm,
        lp,
        b,
        c,
        rounds,
        a_val,
        b_val,
        c_val,
        w: WeightOpen {
            name: weight_name.to_string(),
            cell: weight_cell,
            point: a_point,
            value: a_val,
        },
    }
}

/// Verify a [`MatmulWeightProof`].
#[allow(clippy::too_many_arguments)]
pub fn verify_matmul_weight(
    commitment: &CellSegmentCommitment,
    claims: &[CellClaim],
    w_com: &whir_stacked::SegmentCommitment,
    w_proof: &whir_stacked::Proof,
    w_view: &View,
    (b_view, c_view): (&View, &View),
    proof: &MatmulWeightProof,
    tr: &mut Transcript,
) -> bool {
    let (b, c) = (proof.b, proof.c);
    let Some(m_w) = w_com.tables.get(proof.w.cell).map(|t| t.m) else {
        return false;
    };
    let (Some(m_b), Some(m_c)) = (
        commitment.cells.get(b).map(|c| c.m()),
        commitment.cells.get(c).map(|c| c.m()),
    ) else {
        return false;
    };
    let (ln, lm, lp) = (proof.ln, proof.lm, proof.lp);
    if w_view.bits(m_w) != ln + lm || b_view.bits(m_b) != lm + lp || c_view.bits(m_c) != ln + lp {
        return false;
    }
    for id in [b, c] {
        tr.observe_val(Fp::new(id as u64));
    }
    tr.observe_val(Fp::new(proof.w.cell as u64));

    let r: Vec<Ext> = (0..ln).map(|_| tr.sample_ext()).collect();
    let s: Vec<Ext> = (0..lp).map(|_| tr.sample_ext()).collect();
    if proof.rounds.len() != lm {
        return false;
    }

    let mut claim = proof.c_val;
    let mut kstar = Vec::with_capacity(lm);
    for round in &proof.rounds {
        if round.eval(Ext::ZERO) + round.eval(Ext::ONE) != claim {
            return false;
        }
        for coeff in &round.0 {
            tr.observe_ext(*coeff);
        }
        let ri = tr.sample_ext();
        kstar.push(ri);
        claim = round.eval(ri);
    }
    if claim != proof.a_val * proof.b_val {
        return false;
    }

    if claim_value_at_id(
        commitment,
        claims,
        b,
        &b_view.point(&concat(&s, &kstar), m_b),
    ) != Some(proof.b_val)
        || claim_value_at_id(commitment, claims, c, &c_view.point(&concat(&s, &r), m_c))
            != Some(proof.c_val)
    {
        return false;
    }

    // `W`'s opening against its own root.
    let w_claim = whir_stacked::Claim {
        table: proof.w.cell,
        cols: vec![0],
        point: proof.w.point.clone(),
        values: vec![proof.w.value],
    };
    whir_stacked::verify(
        w_com,
        &[w_claim],
        w_proof,
        &mut crate::pcs::transcript::new_challenger(),
    )
}

/// One chunk of a row-chunked weight, with the chunk index that fixes its `eq`
/// weight in the composition.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ChunkOpen {
    pub chunk: usize,
    pub open: WeightOpen,
}

/// `C = W·B` where `W` is row-chunked across several pre-shared roots.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MatmulChunkedProof {
    pub ln: usize,
    pub lm: usize,
    pub lp: usize,
    pub b: usize,
    pub c: usize,
    pub rounds: Vec<RoundPoly>,
    pub a_val: Ext,
    pub b_val: Ext,
    pub c_val: Ext,
    /// One opening per chunk; `Σ_i eq(r_high, i)·value_i == a_val`.
    pub w: Vec<ChunkOpen>,
}

/// Claim `C = W·B` for a weight split into `2^g` row chunks.
///
/// The row bits of a row-major matrix are its **high** bits, so chunk `i` owns the
/// low `ln−g` row coordinates and its index lives in the high `g`:
///
/// ```text
///   Ã(r, k) = Σ_i eq(r_high, i) · Ã_i(r_low, k)
/// ```
///
/// The sumcheck runs on that weighted sum, and each chunk opens its terminal
/// evaluation at `concat(kstar, r_low)` against its own root. Composing with a
/// plain sum instead is wrong for every chunk count above one.
pub fn claim_matmul_weight_chunked(
    seg: &mut CellSegment,
    chunks: &mut [(&str, &mut SegmentProver, usize)],
    (b, c): (usize, usize),
    tr: &mut Transcript,
    (ln, lm, lp): (usize, usize, usize),
) -> MatmulChunkedProof {
    assert!(
        !chunks.is_empty() && chunks.len().is_power_of_two(),
        "a row-chunked weight needs 2^g chunks"
    );
    let g = chunks.len().trailing_zeros() as usize;
    assert!(g <= ln, "more chunks than rows");
    let ln_low = ln - g;

    for id in [b, c] {
        tr.observe_val(Fp::new(id as u64));
    }
    for (_, _, cell) in chunks.iter() {
        tr.observe_val(Fp::new(*cell as u64));
    }

    let r: Vec<Ext> = (0..ln).map(|_| tr.sample_ext()).collect();
    let s: Vec<Ext> = (0..lp).map(|_| tr.sample_ext()).collect();
    let r_low = r[..ln_low].to_vec();
    let r_high = r[ln_low..].to_vec();

    let to_ext = |v: Vec<Fp>| -> Vec<Ext> { v.into_iter().map(Ext::from).collect() };
    let parts: Vec<Vec<Ext>> = chunks
        .iter()
        .map(|(_, seg_c, cell)| to_ext(seg_c.column_logical(*cell, 0)))
        .collect();
    let bv = to_ext(seg.values(b, 0));
    let cv = to_ext(seg.values(c, 0));
    assert_eq!(bv.len(), 1 << (lm + lp), "B dimensions");
    assert_eq!(cv.len(), 1 << (ln + lp), "C dimensions");
    for a in &parts {
        assert_eq!(a.len(), 1 << (ln_low + lm), "chunk dimensions");
    }

    let c_val = matrix_mle(&cv, &r, &s);
    let eqs: Vec<Ext> = (0..chunks.len())
        .map(|i| crate::field::eq(&r_high, i))
        .collect();

    let mut p = vec![Ext::ZERO; 1 << lm];
    let mut q = vec![Ext::ZERO; 1 << lm];
    for k in 0..(1 << lm) {
        let kb = bool_point(k, lm);
        for (a, &w) in parts.iter().zip(&eqs) {
            p[k] += w * matrix_mle(a, &r_low, &kb);
        }
        q[k] = matrix_mle(&bv, &kb, &s);
    }

    let mut kstar = Vec::with_capacity(lm);
    let mut rounds = Vec::with_capacity(lm);
    for i in 0..lm {
        let round = round_poly_product(&p, &q, lm, i, &kstar);
        for coeff in &round.0 {
            tr.observe_ext(*coeff);
        }
        let ri = tr.sample_ext();
        rounds.push(round);
        kstar.push(ri);
    }

    let a_val = mle_eval(&p, &kstar);
    let b_val = mle_eval(&q, &kstar);
    let a_point = concat(&kstar, &r_low);
    let opened_b = seg.claim_id(b, &[0], &concat(&s, &kstar))[0];
    let opened_c = seg.claim_id(c, &[0], &concat(&s, &r))[0];
    assert_eq!(opened_b, b_val, "B opening matches the sumcheck");
    assert_eq!(opened_c, c_val, "C opening matches the sumcheck");

    let mut w = Vec::with_capacity(chunks.len());
    let mut sum = Ext::ZERO;
    for (i, (name, seg_c, cell)) in chunks.iter_mut().enumerate() {
        let vals = seg_c.claim(*cell, &[0], &a_point);
        sum += eqs[i] * vals[0];
        w.push(ChunkOpen {
            chunk: i,
            open: WeightOpen {
                name: name.to_string(),
                cell: *cell,
                point: a_point.clone(),
                value: vals[0],
            },
        });
    }
    assert_eq!(sum, a_val, "eq-weighted chunk openings must equal a_val");

    MatmulChunkedProof {
        ln,
        lm,
        lp,
        b,
        c,
        rounds,
        a_val,
        b_val,
        c_val,
        w,
    }
}

/// Verify a [`MatmulChunkedProof`]; `parts[i]` is chunk `i`'s `(commitment, proof)`.
pub fn verify_matmul_weight_chunked(
    commitment: &CellSegmentCommitment,
    claims: &[CellClaim],
    parts: &[(&whir_stacked::SegmentCommitment, &whir_stacked::Proof)],
    proof: &MatmulChunkedProof,
    tr: &mut Transcript,
) -> bool {
    let (b, c) = (proof.b, proof.c);
    if commitment.cells.get(b).is_none()
        || commitment.cells.get(c).is_none()
        || parts.len() != proof.w.len()
        || parts.is_empty()
        || !parts.len().is_power_of_two()
    {
        return false;
    }
    let g = parts.len().trailing_zeros() as usize;
    if g > proof.ln {
        return false;
    }
    let ln_low = proof.ln - g;

    for id in [b, c] {
        tr.observe_val(Fp::new(id as u64));
    }
    for o in &proof.w {
        tr.observe_val(Fp::new(o.open.cell as u64));
    }

    let (ln, lm, lp) = (proof.ln, proof.lm, proof.lp);
    let r: Vec<Ext> = (0..ln).map(|_| tr.sample_ext()).collect();
    let s: Vec<Ext> = (0..lp).map(|_| tr.sample_ext()).collect();
    let r_high = r[ln_low..].to_vec();
    if proof.rounds.len() != lm {
        return false;
    }

    let mut claim = proof.c_val;
    let mut kstar = Vec::with_capacity(lm);
    for round in &proof.rounds {
        if round.eval(Ext::ZERO) + round.eval(Ext::ONE) != claim {
            return false;
        }
        for coeff in &round.0 {
            tr.observe_ext(*coeff);
        }
        let ri = tr.sample_ext();
        kstar.push(ri);
        claim = round.eval(ri);
    }
    if claim != proof.a_val * proof.b_val {
        return false;
    }
    if claim_value_at_id(commitment, claims, b, &concat(&s, &kstar)) != Some(proof.b_val)
        || claim_value_at_id(commitment, claims, c, &concat(&s, &r)) != Some(proof.c_val)
    {
        return false;
    }

    let mut sum = Ext::ZERO;
    for (expected, (o, (com, proof_c))) in proof.w.iter().zip(parts).enumerate() {
        if o.chunk != expected {
            return false;
        }
        let w_claim = whir_stacked::Claim {
            table: o.open.cell,
            cols: vec![0],
            point: o.open.point.clone(),
            values: vec![o.open.value],
        };
        if !whir_stacked::verify(
            com,
            &[w_claim],
            proof_c,
            &mut crate::pcs::transcript::new_challenger(),
        ) {
            return false;
        }
        sum += crate::field::eq(&r_high, o.chunk) * o.open.value;
    }
    sum == proof.a_val
}

/// A pointwise-product claim `c = a ⊙ b` over committed cells.
///
/// `c: None` means the target is the **public all-ones vector** (`a⊙b = 1
///` pointwise), which is what a LogUp inverse check needs; the verifier then
/// needs no opening for the target because it knows every entry is `1`.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MulSegmentProof {
    pub a: usize,
    pub b: usize,
    pub c: Option<usize>,
    pub d: usize,
    /// The evaluation point of the target (re-derived from the transcript).
    pub r: Vec<Ext>,
    /// The sumcheck's terminal point (likewise re-derived).
    pub istar: Vec<Ext>,
    pub c_val: Ext,
    pub a_val: Ext,
    pub b_val: Ext,
    pub rounds: Vec<RoundPoly3>,
}

/// Claim `c[i] = a[i]·b[i]` pointwise.
///
/// **Not** a zero-sum check. `Σ(a·b − c) = 0` is satisfied by any `c` that differs
/// from `a⊙b` in compensating places, so it proves nothing. The sound form picks
/// a random `r` and proves
///
/// ```text
///   c̃(r) = Σ_i eq(r,i)·a[i]·b[i]
/// ```
///
/// by a degree-3 sumcheck (`eq` is multilinear, `a·b` bilinear) ending at
/// `eq(r,z)·a(z)·b(z)`. `c̃(r) = (a⊙b)̃(r)` at a random `r` is a
/// Schwartz–Zippel check of the vector identity `c = a⊙b`.
pub fn claim_mul(
    seg: &mut CellSegment,
    (a, b, c): (usize, usize, usize),
    tr: &mut Transcript,
) -> MulSegmentProof {
    claim_mul_impl(seg, a, b, Some(c), tr)
}

/// Claim `a[i]·b[i] = 1` pointwise against the public all-ones vector.
pub fn claim_mul_ones(
    seg: &mut CellSegment,
    (a, b): (usize, usize),
    tr: &mut Transcript,
) -> MulSegmentProof {
    claim_mul_impl(seg, a, b, None, tr)
}

fn claim_mul_impl(
    seg: &mut CellSegment,
    a: usize,
    b: usize,
    c: Option<usize>,
    tr: &mut Transcript,
) -> MulSegmentProof {
    let d = seg.cells[a].m();
    assert_eq!(seg.cells[b].m(), d, "mul operands must share an arity");
    if let Some(c) = c {
        assert_eq!(seg.cells[c].m(), d, "mul operands must share an arity");
    }
    for id in [a, b] {
        tr.observe_val(Fp::new(id as u64));
    }
    if let Some(c) = c {
        tr.observe_val(Fp::new(c as u64));
    }

    let to_ext = |v: Vec<Fp>| -> Vec<Ext> { v.into_iter().map(Ext::from).collect() };
    let af = to_ext(seg.values(a, 0));
    let bf = to_ext(seg.values(b, 0));

    let r: Vec<Ext> = (0..d).map(|_| tr.sample_ext()).collect();
    let c_val = match c {
        Some(c) => mle_eval(&to_ext(seg.values(c, 0)), &r),
        None => Ext::ONE,
    };

    let mut rounds = Vec::with_capacity(d);
    let mut istar = Vec::with_capacity(d);
    for j in 0..d {
        let round = round_poly3_mul(&af, &bf, &r, d, j, &istar);
        for coeff in &round.0 {
            tr.observe_ext(*coeff);
        }
        let rj = tr.sample_ext();
        rounds.push(round);
        istar.push(rj);
    }

    let a_val = mle_eval(&af, &istar);
    let b_val = mle_eval(&bf, &istar);
    seg.claim_id(a, &[0], &istar);
    seg.claim_id(b, &[0], &istar);
    if let Some(c) = c {
        seg.claim_id(c, &[0], &r);
    }

    MulSegmentProof {
        a,
        b,
        c,
        d,
        r,
        istar,
        c_val,
        a_val,
        b_val,
        rounds,
    }
}

/// Verify a [`MulSegmentProof`].
pub fn verify_mul(
    commitment: &CellSegmentCommitment,
    claims: &[CellClaim],
    proof: &MulSegmentProof,
    tr: &mut Transcript,
) -> bool {
    let (a, b, d) = (proof.a, proof.b, proof.d);
    let Some(ma) = commitment.cells.get(a) else {
        return false;
    };
    if commitment.cells.get(b).is_none()
        || ma.m() != d
        || proof.rounds.len() != d
        || proof.r.len() != d
        || proof.istar.len() != d
    {
        return false;
    }
    if let Some(c) = proof.c {
        if commitment.cells.get(c).is_none() {
            return false;
        }
    }
    for id in [a, b] {
        tr.observe_val(Fp::new(id as u64));
    }
    if let Some(c) = proof.c {
        tr.observe_val(Fp::new(c as u64));
    }

    let r: Vec<Ext> = (0..d).map(|_| tr.sample_ext()).collect();
    if r != proof.r {
        return false;
    }

    let mut claim = proof.c_val;
    let mut istar = Vec::with_capacity(d);
    for round in &proof.rounds {
        if round.eval(Ext::ZERO) + round.eval(Ext::ONE) != claim {
            return false;
        }
        for coeff in &round.0 {
            tr.observe_ext(*coeff);
        }
        let rj = tr.sample_ext();
        istar.push(rj);
        claim = round.eval(rj);
    }
    if istar != proof.istar {
        return false;
    }
    if claim != eq_partial(&r, &istar) * proof.a_val * proof.b_val {
        return false;
    }

    let target_ok = match proof.c {
        // The verifier knows the all-ones target, and the prover must agree.
        None => proof.c_val == Ext::ONE,
        Some(c) => claim_value_at_id(commitment, claims, c, &r) == Some(proof.c_val),
    };
    target_ok
        && claim_value_at_id(commitment, claims, a, &istar) == Some(proof.a_val)
        && claim_value_at_id(commitment, claims, b, &istar) == Some(proof.b_val)
}

/// Verify a [`MulSegmentProof`] whose target is the public all-ones vector.
pub fn verify_mul_ones(
    commitment: &CellSegmentCommitment,
    claims: &[CellClaim],
    proof: &MulSegmentProof,
    tr: &mut Transcript,
) -> bool {
    proof.c.is_none() && verify_mul(commitment, claims, proof, tr)
}

/// A reshape claim: the flat element multiset is unchanged, so the same `r`
/// linear functional must give the same value on both cells.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ReshapeSegmentProof {
    pub input: usize,
    pub output: usize,
    pub b_in: Ext,
    pub c_in: LinearClaimProof,
    pub b_out: Ext,
    pub c_out: LinearClaimProof,
}

/// Claim that `output` is `input` re-interpreted with a different shape.
pub fn claim_reshape(
    seg: &mut CellSegment,
    input: usize,
    output: usize,
    tr: &mut Transcript,
) -> ReshapeSegmentProof {
    let n = 1usize << seg.cells[output].m();
    tr.observe_val(Fp::new(input as u64));
    tr.observe_val(Fp::new(output as u64));
    let h = crate::pcs::linear_claim::sample(n, tr);
    let (b_in, c_in) = seg.claim_linear_id(input, 0, &h, tr);
    let (b_out, c_out) = seg.claim_linear_id(output, 0, &h, tr);
    ReshapeSegmentProof {
        input,
        output,
        b_in,
        c_in,
        b_out,
        c_out,
    }
}

/// Verify a [`ReshapeSegmentProof`].
pub fn verify_reshape(
    commitment: &CellSegmentCommitment,
    claims: &[CellClaim],
    proof: &ReshapeSegmentProof,
    tr: &mut Transcript,
) -> bool {
    let Some(meta) = commitment.cells.get(proof.output) else {
        return false;
    };
    if commitment.cells.get(proof.input).is_none() {
        return false;
    }
    let n = 1usize << meta.m();
    tr.observe_val(Fp::new(proof.input as u64));
    tr.observe_val(Fp::new(proof.output as u64));
    let h = crate::pcs::linear_claim::sample(n, tr);

    let Some((z_in, c_in)) = linear_verify(&proof.c_in, proof.b_in, tr) else {
        return false;
    };
    let Some(v_in) = claim_value_at_id(commitment, claims, proof.input, &z_in) else {
        return false;
    };
    let Some((z_out, c_out)) = linear_verify(&proof.c_out, proof.b_out, tr) else {
        return false;
    };
    let Some(v_out) = claim_value_at_id(commitment, claims, proof.output, &z_out) else {
        return false;
    };
    linear_finish(&h, &z_in, c_in, v_in)
        && linear_finish(&h, &z_out, c_out, v_out)
        && proof.b_in == proof.b_out
}

/// One row of a weighted sum: `out[s] = Σ_t choice[s][t] · term_t[s]`.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WeightedSumRowProof {
    pub out: usize,
    pub choice: usize,
    pub s: usize,
    pub hidden: usize,
    pub experts: usize,
    pub b_out: Ext,
    pub c_out: LinearClaimProof,
    /// `(term cell, ⟨r, term row⟩, sumcheck)`, in term order.
    pub terms: Vec<(usize, Ext, LinearClaimProof)>,
    pub b_choice: Ext,
    pub c_choice: LinearClaimProof,
}

/// Claim one row of `out = choice · terms` (a mixture-of-experts combine).
///
/// `r` is sampled per row, then:
/// * `b_out = ⟨r, out[s]⟩` — one `Block` claim on the `out` cell,
/// * `b_t = ⟨r, term_t[s]⟩` — one `Block` claim per term cell,
/// * `b_choice = Σ_t b_t · choice[s][t]` — one `Block` claim on the choice cell.
///
/// No row is materialised or committed: every row is a slice of a cell that the
/// segment already holds.
pub fn claim_weighted_sum(
    seg: &mut CellSegment,
    out: usize,
    terms: &[usize],
    choice: usize,
    s: usize,
    hidden: usize,
    experts: usize,
    tr: &mut Transcript,
) -> WeightedSumRowProof {
    tr.observe_val(Fp::new(out as u64));
    for &t in terms {
        tr.observe_val(Fp::new(t as u64));
    }
    tr.observe_val(Fp::new(choice as u64));
    tr.observe_val(Fp::new(s as u64));

    let (r, _coins) = crate::pcs::linear_claim::sample_coins(hidden, tr);
    let row = |len: usize| LinearH::Block {
        start: s * hidden,
        values: r.clone(),
        len,
    };

    let out_len = 1usize << seg.cells[out].m();
    let (b_out, c_out) = seg.claim_linear_id(out, 0, &row(out_len), tr);
    let mut tvals = Vec::with_capacity(terms.len());
    let mut proofs = Vec::with_capacity(terms.len());
    for &t in terms {
        let len = 1usize << seg.cells[t].m();
        let (b, c) = seg.claim_linear_id(t, 0, &row(len), tr);
        tvals.push(b);
        proofs.push((t, b, c));
    }

    let h_choice = LinearH::Block {
        start: s * experts,
        values: tvals
            .iter()
            .map(|&b| ext_to_fp(b).expect("claim value is in the base field"))
            .collect(),
        len: 1usize << seg.cells[choice].m(),
    };
    let (b_choice, c_choice) = seg.claim_linear_id(choice, 0, &h_choice, tr);

    WeightedSumRowProof {
        out,
        choice,
        s,
        hidden,
        experts,
        b_out,
        c_out,
        terms: proofs,
        b_choice,
        c_choice,
    }
}

/// Verify one row of a weighted sum.
pub fn verify_weighted_sum(
    commitment: &CellSegmentCommitment,
    claims: &[CellClaim],
    proof: &WeightedSumRowProof,
    tr: &mut Transcript,
) -> bool {
    if commitment.cells.get(proof.out).is_none()
        || commitment.cells.get(proof.choice).is_none()
        || proof.terms.len() != proof.experts
        || proof
            .terms
            .iter()
            .any(|(t, _, _)| commitment.cells.get(*t).is_none())
    {
        return false;
    }

    tr.observe_val(Fp::new(proof.out as u64));
    for (t, _, _) in &proof.terms {
        tr.observe_val(Fp::new(*t as u64));
    }
    tr.observe_val(Fp::new(proof.choice as u64));
    tr.observe_val(Fp::new(proof.s as u64));

    let (r, _coins) = crate::pcs::linear_claim::sample_coins(proof.hidden, tr);
    let row = |cell: usize| LinearH::Block {
        start: proof.s * proof.hidden,
        values: r.clone(),
        len: 1usize << commitment.cells[cell].m(),
    };

    let Some((z, cf)) = linear_verify(&proof.c_out, proof.b_out, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, proof.out, &z) else {
        return false;
    };
    if !linear_finish(&row(proof.out), &z, cf, v) {
        return false;
    }

    let mut tvals = Vec::with_capacity(proof.terms.len());
    for (t, b, c) in &proof.terms {
        let Some((z, cf)) = linear_verify(c, *b, tr) else {
            return false;
        };
        let Some(v) = claim_value_at_id(commitment, claims, *t, &z) else {
            return false;
        };
        if !linear_finish(&row(*t), &z, cf, v) {
            return false;
        }
        // A verified claim is a base-field product, so the high coefficients are 0.
        let Some(x) = ext_to_fp(*b) else {
            return false;
        };
        tvals.push(x);
    }

    let h_choice = LinearH::Block {
        start: proof.s * proof.experts,
        values: tvals,
        len: 1usize << commitment.cells[proof.choice].m(),
    };
    let Some((z, cf)) = linear_verify(&proof.c_choice, proof.b_choice, tr) else {
        return false;
    };
    let Some(v) = claim_value_at_id(commitment, claims, proof.choice, &z) else {
        return false;
    };
    linear_finish(&h_choice, &z, cf, v) && proof.b_out == proof.b_choice
}

/// A field element known to lie in the base field (a sumcheck claim value).
fn ext_to_fp(e: Ext) -> Option<Fp> {
    use p3_field::BasedVectorSpace;
    let c = <Ext as BasedVectorSpace<Fp>>::as_basis_coefficients_slice(&e);
    if c[1..].iter().any(|&x| x != Fp::ZERO) {
        return None;
    }
    Some(c[0])
}

/// Where a logical operand lives inside a committed cell.
///
/// Every op works on a *view* of a cell: the whole cell, a contiguous block, or a
/// gather that fixes some index bits and permutes the rest (the attention head
/// layout). Because all extents are powers of two these are **bit maps** on the
/// cell index, so a view's evaluation is a point evaluation of the whole cell at
/// a point whose extra coordinates are Boolean — no separate commitment and no
/// linear claim.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum View {
    /// The whole cell.
    Whole,
    /// Logical bit `k` reads cell bit `map[k]`; every other cell bit is fixed.
    Mapped {
        map: Vec<usize>,
        fixed: Vec<(usize, bool)>,
    },
}

impl View {
    /// A contiguous block of `bits` low bits starting at `start`.
    ///
    /// # Panics
    ///
    /// `start` must be a multiple of `2^bits`.
    pub fn block(m: usize, start: usize, bits: usize) -> Self {
        assert!(bits <= m, "block wider than the cell");
        assert_eq!(start & ((1 << bits) - 1), 0, "block must be aligned");
        let high = start >> bits;
        let fixed = (0..m - bits)
            .map(|i| (bits + i, (high >> i) & 1 == 1))
            .collect();
        Self::Mapped {
            map: (0..bits).collect(),
            fixed,
        }
    }

    /// A gather: logical bit `k` reads cell bit `map[k]`, the rest are fixed.
    pub fn gather(map: Vec<usize>, fixed: Vec<(usize, bool)>) -> Self {
        Self::Mapped { map, fixed }
    }

    /// Swap the low `lo` bits with the high `hi` bits of an `lo + hi`-bit cell.
    ///
    /// This is the transpose of a row-major matrix whose row and column counts are
    /// both powers of two: logical `[hi][lo]` reads cell `[lo][hi]`.
    ///
    /// # Panics
    ///
    /// `lo + hi` must equal the cell's arity.
    pub fn swap_blocks(lo: usize, hi: usize) -> Self {
        let map = (0..lo + hi)
            .map(|k| if k < lo { hi + k } else { k - lo })
            .collect();
        Self::Mapped {
            map,
            fixed: Vec::new(),
        }
    }

    /// Bits of the logical view.
    pub fn bits(&self, m: usize) -> usize {
        match self {
            Self::Whole => m,
            Self::Mapped { map, .. } => map.len(),
        }
    }

    /// The logical values this view exposes from a cell of arity `m`.
    pub fn values(&self, cell: &[Fp], m: usize) -> Vec<Fp> {
        match self {
            Self::Whole => {
                assert_eq!(cell.len(), 1 << m);
                cell.to_vec()
            }
            Self::Mapped { map, fixed } => {
                self.assert_covers(m);
                let bits = map.len();
                (0..(1usize << bits))
                    .map(|logical| {
                        let mut idx = 0usize;
                        for (k, &b) in map.iter().enumerate() {
                            if (logical >> k) & 1 == 1 {
                                idx |= 1 << b;
                            }
                        }
                        for &(b, v) in fixed {
                            if v {
                                idx |= 1 << b;
                            }
                        }
                        cell[idx]
                    })
                    .collect()
            }
        }
    }

    /// The cell point at which this view equals `logical`.
    pub fn point(&self, logical: &[Ext], m: usize) -> Vec<Ext> {
        match self {
            Self::Whole => {
                assert_eq!(logical.len(), m, "whole view arity");
                logical.to_vec()
            }
            Self::Mapped { map, fixed } => {
                self.assert_covers(m);
                assert_eq!(logical.len(), map.len(), "view arity");
                let mut out = vec![Ext::ZERO; m];
                for (k, &b) in map.iter().enumerate() {
                    out[b] = logical[k];
                }
                for &(b, v) in fixed {
                    out[b] = if v { Ext::ONE } else { Ext::ZERO };
                }
                out
            }
        }
    }

    fn assert_covers(&self, m: usize) {
        if let Self::Mapped { map, fixed } = self {
            let mut seen = vec![false; m];
            for &b in map.iter().chain(fixed.iter().map(|(b, _)| b)) {
                assert!(b < m, "view bit outside the cell");
                assert!(!seen[b], "view bit covered twice");
                seen[b] = true;
            }
            assert!(seen.iter().all(|&s| s), "view must cover every cell bit");
        }
    }
}

/// A `C = A·B` claim over three flat committed cells.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MatmulSegmentProof {
    pub ln: usize,
    pub lm: usize,
    pub lp: usize,
    pub rounds: Vec<RoundPoly>,
    pub a_val: Ext,
    pub b_val: Ext,
    pub c_val: Ext,
}

/// Claim `C = A·B` for `A: 2^ln × 2^lm`, `B: 2^lm × 2^lp`, `C: 2^ln × 2^lp`,
/// all row-major flat cells of `seg`.
///
/// This is the segment port of [`crate::gkr::protocol::prove_gkr_matmul`]: the
/// sumcheck is unchanged, but the three surviving evaluations are plain cell
/// openings instead of three `InnerProductProof`s each with their own `p`/`q`
/// commitments.
pub fn claim_matmul(
    seg: &mut CellSegment,
    cells: (usize, usize, usize),
    views: (&View, &View, &View),
    tr: &mut Transcript,
    (ln, lm, lp): (usize, usize, usize),
) -> MatmulSegmentProof {
    let (ca, cb, cc) = cells;
    for id in [ca, cb, cc] {
        tr.observe_val(Fp::new(id as u64));
    }
    let r: Vec<Ext> = (0..ln).map(|_| tr.sample_ext()).collect();
    let s: Vec<Ext> = (0..lp).map(|_| tr.sample_ext()).collect();

    let (m_a, m_b, m_c) = (
        seg.cells_meta(ca).m(),
        seg.cells_meta(cb).m(),
        seg.cells_meta(cc).m(),
    );
    let to_ext = |v: Vec<Fp>| -> Vec<Ext> { v.into_iter().map(Ext::from).collect() };
    let a = to_ext(views.0.values(&seg.values(ca, 0), m_a));
    let b = to_ext(views.1.values(&seg.values(cb, 0), m_b));
    let c = to_ext(views.2.values(&seg.values(cc, 0), m_c));
    assert_eq!(a.len(), 1 << (ln + lm), "A dimensions");
    assert_eq!(b.len(), 1 << (lm + lp), "B dimensions");
    assert_eq!(c.len(), 1 << (ln + lp), "C dimensions");

    let c_val = matrix_mle(&c, &r, &s);
    drop(c);

    // Contraction MLEs, each a single pass over the operand:
    //   p[k] = Ã(r, k) = Σ_n eq(r, n)·a[n·2^lm + k]
    //   q[k] = B̃(k, s) = Σ_t eq(s, t)·b[k·2^lp + t]
    // The `2^lm`-fold `matrix_mle` sweep this replaces is `O(N·K·K)`.
    let mut p = vec![Ext::ZERO; 1 << lm];
    for (n, rn) in (0..(1usize << ln))
        .map(|n| (n, crate::field::eq(&r, n)))
        .filter(|&(_, rn)| rn != Ext::ZERO)
    {
        let base = n << lm;
        for k in 0..(1 << lm) {
            p[k] += rn * a[base + k];
        }
    }
    drop(a);

    let eq_s: Vec<Ext> = (0..(1usize << lp))
        .map(|t| crate::field::eq(&s, t))
        .collect();
    let mut q = vec![Ext::ZERO; 1 << lm];
    for (k, qk) in q.iter_mut().enumerate() {
        let base = k << lp;
        for (t, &et) in eq_s.iter().enumerate() {
            *qk += et * b[base + t];
        }
    }
    drop(b);

    let mut kstar = Vec::with_capacity(lm);
    let mut rounds = Vec::with_capacity(lm);
    for i in 0..lm {
        let round = round_poly_product(&p, &q, lm, i, &kstar);
        for coeff in &round.0 {
            tr.observe_ext(*coeff);
        }
        let ri = tr.sample_ext();
        rounds.push(round);
        kstar.push(ri);
    }

    let a_val = mle_eval(&p, &kstar);
    let b_val = mle_eval(&q, &kstar);

    // The three surviving evaluations are now ordinary cell openings, at the
    // points the views translate to.
    let opened_a = seg.claim_id_view(ca, views.0, &concat(&kstar, &r))[0];
    let opened_b = seg.claim_id_view(cb, views.1, &concat(&s, &kstar))[0];
    let opened_c = seg.claim_id_view(cc, views.2, &concat(&s, &r))[0];
    assert_eq!(opened_a, a_val, "A opening matches the sumcheck");
    assert_eq!(opened_b, b_val, "B opening matches the sumcheck");
    assert_eq!(opened_c, c_val, "C opening matches the sumcheck");

    MatmulSegmentProof {
        ln,
        lm,
        lp,
        rounds,
        a_val,
        b_val,
        c_val,
    }
}

/// Verify a [`MatmulSegmentProof`] against the segment's recorded openings.
pub fn verify_matmul(
    commitment: &CellSegmentCommitment,
    claims: &[CellClaim],
    cells: (usize, usize, usize),
    views: (&View, &View, &View),
    proof: &MatmulSegmentProof,
    tr: &mut Transcript,
) -> bool {
    let (ca, cb, cc) = cells;
    let (Some(ma), Some(mb), Some(mc)) = (
        commitment.cells.get(ca).map(|c| c.m()),
        commitment.cells.get(cb).map(|c| c.m()),
        commitment.cells.get(cc).map(|c| c.m()),
    ) else {
        return false;
    };
    for id in [ca, cb, cc] {
        tr.observe_val(Fp::new(id as u64));
    }
    let (ln, lm, lp) = (proof.ln, proof.lm, proof.lp);
    let r: Vec<Ext> = (0..ln).map(|_| tr.sample_ext()).collect();
    let s: Vec<Ext> = (0..lp).map(|_| tr.sample_ext()).collect();
    if proof.rounds.len() != lm
        || views.0.bits(ma) != ln + lm
        || views.1.bits(mb) != lm + lp
        || views.2.bits(mc) != ln + lp
    {
        return false;
    }

    let mut claim = proof.c_val;
    let mut kstar = Vec::with_capacity(lm);
    for round in &proof.rounds {
        if round.eval(Ext::ZERO) + round.eval(Ext::ONE) != claim {
            return false;
        }
        for coeff in &round.0 {
            tr.observe_ext(*coeff);
        }
        let ri = tr.sample_ext();
        kstar.push(ri);
        claim = round.eval(ri);
    }
    if claim != proof.a_val * proof.b_val {
        return false;
    }

    claim_value_at_id(
        commitment,
        claims,
        ca,
        &views.0.point(&concat(&kstar, &r), ma),
    ) == Some(proof.a_val)
        && claim_value_at_id(
            commitment,
            claims,
            cb,
            &views.1.point(&concat(&s, &kstar), mb),
        ) == Some(proof.b_val)
        && claim_value_at_id(commitment, claims, cc, &views.2.point(&concat(&s, &r), mc))
            == Some(proof.c_val)
}

/// Verify a cell segment against `claims` (in the prover's record order).
pub fn verify(
    commitment: &CellSegmentCommitment,
    claims: &[CellClaim],
    proof: &whir_stacked::Proof,
    ch: &mut Challenger,
) -> bool {
    // Cell id + column count must agree with the commitment before any PCS work.
    if claims.is_empty()
        || claims.iter().any(|c| {
            c.cols.len() != c.values.len()
                || commitment
                    .cells
                    .get(c.cell)
                    .map_or(true, |meta| c.cols.iter().any(|&k| k >= meta.cols))
        })
    {
        return false;
    }
    let (table_of, col_of) = packing(&commitment.cells);
    let flat: Vec<whir_stacked::Claim> = claims
        .iter()
        .map(|c| {
            let base = col_of[c.cell];
            whir_stacked::Claim {
                table: table_of[c.cell],
                cols: c.cols.iter().map(|&k| base + k).collect(),
                point: c.point.clone(),
                values: c.values.clone(),
            }
        })
        .collect();
    whir_stacked::verify(&commitment.pcs, &flat, proof, ch)
}

/// Look up the **first** claim on the cell named `name` (verifier side convenience).
pub fn claim_of<'a>(
    commitment: &CellSegmentCommitment,
    claims: &'a [CellClaim],
    name: &str,
) -> Option<&'a CellClaim> {
    let id = commitment.cell(name)?;
    claims.iter().find(|c| c.cell == id)
}

/// Values of every named cell claim, keyed by name; the **last** claim on a cell wins.
pub fn values_by_name<'a>(
    commitment: &'a CellSegmentCommitment,
    claims: &[CellClaim],
) -> HashMap<&'a str, Vec<Ext>> {
    claims
        .iter()
        .filter_map(|c| {
            commitment
                .cells
                .get(c.cell)
                .map(|m| (m.name.as_str(), c.values.clone()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::transcript::new_challenger;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    fn rand_point(rng: &mut StdRng, m: usize) -> Vec<Ext> {
        (0..m)
            .map(|_| Ext::from(Fp::new(rng.random::<u64>() % 0xFFFF_FFFF_0000_0001)))
            .collect()
    }

    fn cell(name: &str, rows: usize, cols: usize, rng: &mut StdRng) -> Cell {
        let values = (0..rows * cols)
            .map(|_| rng.random::<i64>() % 97 - 48)
            .collect();
        Cell::new(name, rows, cols, values)
    }

    #[test]
    fn non_power_of_two_rows_and_cols() {
        // 6 rows pads to 8; 3 columns open as one batch.
        let mut rng = StdRng::seed_from_u64(1);
        let a = cell("a", 6, 3, &mut rng);
        let b = cell("b", 8, 1, &mut rng);

        let mut seg = CellSegment::seal(&[a.clone(), b.clone()], Params::default());
        let ca = seg.commitment();
        assert_eq!(ca.cells[0].rows, 6);
        assert_eq!(ca.cells[0].m(), 3);

        let z = rand_point(&mut rng, 3);
        let va = seg.claim("a", &z);
        assert_eq!(va.len(), 3);
        let vb = seg.claim("b", &rand_point(&mut rng, 3));
        assert_eq!(vb.len(), 1);

        // The claimed values really are the column MLEs (computed independently).
        for j in 0..3 {
            let col: Vec<Fp> = (0..8)
                .map(|i| {
                    if i < 6 {
                        signed(a.values[i * 3 + j])
                    } else {
                        Fp::ZERO
                    }
                })
                .collect();
            let expect = crate::field::sumcheck::mle_value(&col, &z);
            assert_eq!(va[j], expect);
        }

        let claims = seg.claims().to_vec();
        let proof = seg.prove(&mut new_challenger());
        let mut vch = new_challenger();
        assert!(verify(&ca, &claims, &proof, &mut vch));

        // Flipping one column value is rejected.
        let mut bad = claims.clone();
        bad[0].values[1] += Ext::ONE;
        let mut vch = new_challenger();
        assert!(!verify(&ca, &bad, &proof, &mut vch));

        // A column-count mismatch fails closed (no panic).
        let mut bad = claims.clone();
        bad[1].values.push(Ext::ONE);
        let mut vch = new_challenger();
        assert!(!verify(&ca, &bad, &proof, &mut vch));
    }

    #[test]
    fn subset_of_columns() {
        let mut rng = StdRng::seed_from_u64(2);
        let a = cell("a", 8, 4, &mut rng);
        let mut seg = CellSegment::seal(&[a.clone()], Params::default());
        let ca = seg.commitment();

        let z = rand_point(&mut rng, 3);
        let v = seg.claim_cols("a", &[1, 3], &z);
        assert_eq!(v.len(), 2);

        let full = seg.claim("a", &z);
        assert_eq!(v, vec![full[1], full[3]]);

        let claims = seg.claims().to_vec();
        let proof = seg.prove(&mut new_challenger());
        let mut vch = new_challenger();
        assert!(verify(&ca, &claims, &proof, &mut vch));
        assert_eq!(claim_of(&ca, &claims, "a").unwrap().values.len(), 2);
        assert_eq!(claims[1].values.len(), 4);
        assert_eq!(values_by_name(&ca, &claims)["a"].len(), 4);
    }

    #[test]
    fn weight_tensor_commitment_is_deterministic() {
        let mut rng = StdRng::seed_from_u64(4);
        let vals: Vec<i8> = (0..6 * 3).map(|_| rng.random::<i8>()).collect();
        let w = WeightMatrix::new("w", 6, 3, vals);

        // The commit phase and the prove phase must derive the same root.
        let c0 = CellSegment::seal_weight(&w, Params::default()).commitment();
        let c1 = CellSegment::seal_weight(&w, Params::default()).commitment();
        assert_eq!(c0.pcs.root, c1.pcs.root);
        assert_eq!(c0.cells[0], c1.cells[0]);

        let mut seg = CellSegment::seal_weight(&w, Params::default());
        let z = rand_point(&mut rng, 3);
        let v = seg.claim("w", &z);
        // Independent check: the MLE of the signed, row-padded columns.
        for j in 0..3 {
            let col: Vec<Fp> = (0..8)
                .map(|i| {
                    if i < 6 {
                        signed(w.values[i * 3 + j] as i64)
                    } else {
                        Fp::ZERO
                    }
                })
                .collect();
            assert_eq!(v[j], crate::field::sumcheck::mle_value(&col, &z));
        }

        let claims = seg.claims().to_vec();
        let proof = seg.prove(&mut new_challenger());
        let mut vch = new_challenger();
        assert!(verify(&c0, &claims, &proof, &mut vch));
    }

    #[test]
    fn linear_claim_tie_row_to_full_cell() {
        // The `tie_row` shape: `row` is `full[start..start + d]`.
        use crate::pcs::linear_claim::LinearH;
        use crate::pcs::transcript::Transcript;

        let mut rng = StdRng::seed_from_u64(5);
        let full_vals: Vec<i64> = (0..8).map(|_| rng.random::<i64>() % 50).collect();
        let full = Cell::new("full", 8, 1, full_vals.clone());
        let row = Cell::new("row", 2, 1, full_vals[4..6].to_vec());

        let mut seg = CellSegment::seal(&[full, row], Params::default());
        let com = seg.commitment();

        // The tie's random linear combination has the slice's length.
        let r: Vec<Fp> = (0..2)
            .map(|_| Fp::new(rng.random::<u64>() % 0xFFFF_FFFF_0000_0001))
            .collect();
        let h_row = LinearH::Dense(r.clone());
        let e_full = LinearH::Block {
            start: 4,
            values: r.clone(),
            len: 8,
        };

        let mut tr = Transcript::new(&new_challenger());
        let (b_row, proof_row) = seg.claim_linear("row", 0, &h_row, &mut tr);
        let (b_full, proof_full) = seg.claim_linear("full", 0, &e_full, &mut tr);
        assert_eq!(b_row, b_full, "both sides of the tie agree");

        let claims = seg.claims().to_vec();
        let proof = seg.prove(&mut new_challenger());
        let mut vch = new_challenger();
        assert!(verify(&com, &claims, &proof, &mut vch));

        // Replay both sumchecks against the recorded point openings.
        let mut trv = Transcript::new(&new_challenger());
        let (zr, cr) = linear_verify(&proof_row, b_row, &mut trv).expect("row sumcheck");
        let vr = claim_value_at(&com, &claims, "row", &zr).expect("row opening");
        assert!(linear_finish(&h_row, &zr, cr, vr));

        let (zf, cf) = linear_verify(&proof_full, b_full, &mut trv).expect("full sumcheck");
        let vf = claim_value_at(&com, &claims, "full", &zf).expect("full opening");
        assert!(linear_finish(&e_full, &zf, cf, vf));

        // A wrong opened value breaks the terminal identity.
        assert!(!linear_finish(&h_row, &zr, cr, vr + Ext::ONE));
        // A wrong claim value breaks the first sumcheck round.
        let mut trb = Transcript::new(&new_challenger());
        assert!(linear_verify(&proof_row, b_row + Ext::ONE, &mut trb).is_none());
        // The support-only `e` evaluation matches the dense form.
        let mut dense_e = Fp::zero_vec(8);
        dense_e[4..6].copy_from_slice(&r);
        assert_eq!(e_full.eval_at(&zf), LinearH::Dense(dense_e).eval_at(&zf));
    }

    /// Peak-memory / timing probe for one realistic segment: a 2^18-row cell plus
    /// a 2^18-long linear claim (the `tie_row` primitive). Run with
    /// `cargo test --release --lib gkr::segment::tests::segment_probe -- --ignored --nocapture`
    /// under `/usr/bin/time -l` to read the resident-set maximum.
    #[test]
    #[ignore = "memory probe: run explicitly with --ignored"]
    fn segment_probe() {
        let m = 18;
        let n = 1usize << m;
        let rng = &mut StdRng::seed_from_u64(77);
        let vals: Vec<i64> = (0..n).map(|_| rng.random::<i64>() % 1000).collect();
        let c = Cell::new("c", n, 1, vals);
        println!("cell cells: {n} ({:.1} MB as i64)", n as f64 * 8.0 / 1e6);

        let mut seg = CellSegment::seal(&[c], Params::default());
        println!("stacked variables: {}", seg.commitment().pcs.m);

        let h = crate::pcs::linear_claim::LinearH::Dense(
            (0..n)
                .map(|_| Fp::new(rng.random::<u64>() % 0xFFFF_FFFF_0000_0001))
                .collect(),
        );
        let mut tr = Transcript::new(&new_challenger());
        let started = std::time::Instant::now();
        let (b, _) = seg.claim_linear("c", 0, &h, &mut tr);
        println!(
            "linear claim over 2^{m}: {:?}, b = {b:?}",
            started.elapsed()
        );

        let commitment = seg.commitment();
        let claims = seg.claims().to_vec();
        let started = std::time::Instant::now();
        let proof = seg.prove(&mut new_challenger());
        println!(
            "segment proof: {:?}, {} bytes",
            started.elapsed(),
            bincode::serialize(&proof).map(|v| v.len()).unwrap_or(0)
        );

        let started = std::time::Instant::now();
        let mut vch = new_challenger();
        assert!(verify(&commitment, &claims, &proof, &mut vch));
        println!("verify: {:?}", started.elapsed());
    }

    #[test]
    fn matmul_claim_roundtrip_and_soundness() {
        // A: 4x4, B: 4x2, C = A·B: 4x2, all row-major flat cells.
        let (ln, lm, lp) = (2usize, 2usize, 1usize);
        let mut a_vals = vec![0i64; 16];
        let mut b_vals = vec![0i64; 8];
        for i in 0..16 {
            a_vals[i] = (i as i64 % 5) - 2;
        }
        for i in 0..8 {
            b_vals[i] = (i as i64 % 3) - 1;
        }
        let c_vals = matmul_i64(&a_vals, &b_vals, 4, 4, 2);

        let cells = |c: &[i64]| {
            vec![
                Cell::new("A", 16, 1, a_vals.clone()),
                Cell::new("B", 8, 1, b_vals.clone()),
                Cell::new("C", 8, 1, c.to_vec()),
            ]
        };

        let mut seg = CellSegment::seal(&cells(&c_vals), Params::default());
        let com = seg.commitment();
        let (a, b, c) = (seg.id("A"), seg.id("B"), seg.id("C"));

        let mut tr = Transcript::new(&new_challenger());
        let whole = (View::Whole, View::Whole, View::Whole);
        let proof = claim_matmul(
            &mut seg,
            (a, b, c),
            (&whole.0, &whole.1, &whole.2),
            &mut tr,
            (ln, lm, lp),
        );

        let claims = seg.claims().to_vec();
        let wproof = seg.prove(&mut new_challenger());
        assert!(verify(&com, &claims, &wproof, &mut new_challenger()));

        let mut vtr = Transcript::new(&new_challenger());
        assert!(verify_matmul(
            &com,
            &claims,
            (a, b, c),
            (&whole.0, &whole.1, &whole.2),
            &proof,
            &mut vtr
        ));

        // A wrong product is rejected: the sumcheck cannot start from its C̃(r,s).
        let mut wrong = c_vals.clone();
        wrong[0] += 1;
        let mut bad_seg = CellSegment::seal(&cells(&wrong), Params::default());
        let bad_com = bad_seg.commitment();
        let bad_proof = claim_matmul(
            &mut bad_seg,
            (a, b, c),
            (&whole.0, &whole.1, &whole.2),
            &mut Transcript::new(&new_challenger()),
            (ln, lm, lp),
        );
        let bad_claims = bad_seg.claims().to_vec();
        let bad_w = bad_seg.prove(&mut new_challenger());
        assert!(verify(&bad_com, &bad_claims, &bad_w, &mut new_challenger()));
        let mut vtr = Transcript::new(&new_challenger());
        assert!(!verify_matmul(
            &bad_com,
            &bad_claims,
            (a, b, c),
            (&whole.0, &whole.1, &whole.2),
            &bad_proof,
            &mut vtr
        ));
    }

    #[test]
    fn matmul_over_views_matches_the_attention_layout() {
        // `scores`: dot = kh · qh with `k` laid out [L=2, H=2, D=2] and `q` the
        // same. One head is a gather of `k` (fixing the head bits, dropping the
        // position stride into the low bits) and a block of `q`.
        let (l, h, d) = (2usize, 2usize, 2usize);
        let ln = 1; // positions (lq = L)
        let lp = 0; // one query row
                    // k[q][head][d] and q[q][head][d].
        let k_vals: Vec<i64> = (0..(l * h * d) as i64).map(|i| i % 5 - 2).collect();
        let q_vals: Vec<i64> = (0..(l * h * d) as i64).map(|i| i % 3 - 1).collect();

        // head 1, query position 0.
        let hi = 1usize;
        let li = 0usize;
        // kh[t][j] = k[(t·H + hi)·D + j]  → logical bits [j (log D), t (log L)]
        let kh: Vec<i64> = (0..l)
            .flat_map(|t| {
                let s = (t * h + hi) * d;
                k_vals[s..s + d].to_vec()
            })
            .collect();
        let qh: Vec<i64> = q_vals[((li * h) + hi) * d..((li * h) + hi + 1) * d].to_vec();
        let dot: Vec<i64> = (0..l)
            .map(|t| (0..d).map(|j| kh[t * d + j] * qh[j]).sum())
            .collect();

        let n_k = l * h * d; // 8 → m = 3
        let n_q = l * h * d;
        let n_dot = l; // 2 → m = 1
        let cells = vec![
            Cell::new("k", n_k, 1, k_vals.clone()),
            Cell::new("q", n_q, 1, q_vals.clone()),
            Cell::new("dot", n_dot, 1, dot.clone()),
        ];
        let mut seg = CellSegment::seal(&cells, Params::default());
        let com = seg.commitment();
        let (ck, cq, cd) = (seg.id("k"), seg.id("q"), seg.id("dot"));

        // k's view: logical bit j (log d) → cell bit j; logical bit t (log l) →
        // cell bit log d + log h + t; cell bits [log d, log d+log h) fixed to hi.
        let log_d = d.trailing_zeros() as usize;
        let log_h = h.trailing_zeros() as usize;
        let log_l = l.trailing_zeros() as usize;
        let mut map: Vec<usize> = (0..log_d).collect();
        map.extend((0..log_l).map(|t| log_d + log_h + t));
        let fixed: Vec<(usize, bool)> = (0..log_h)
            .map(|i| (log_d + i, (hi >> i) & 1 == 1))
            .collect();
        let vk = View::gather(map, fixed);
        // q's head block.
        let vq = View::block(n_q.trailing_zeros() as usize, ((li * h) + hi) * d, log_d);
        let vd = View::Whole;

        let mut tr = Transcript::new(&new_challenger());
        let proof = claim_matmul(
            &mut seg,
            (ck, cq, cd),
            (&vk, &vq, &vd),
            &mut tr,
            (ln, log_d, lp),
        );
        let claims = seg.claims().to_vec();
        let wproof = seg.prove(&mut new_challenger());
        assert!(verify(&com, &claims, &wproof, &mut new_challenger()));
        let mut vtr = Transcript::new(&new_challenger());
        assert!(verify_matmul(
            &com,
            &claims,
            (ck, cq, cd),
            (&vk, &vq, &vd),
            &proof,
            &mut vtr
        ));

        // A wrong dot is rejected.
        let mut bad_dot = dot.clone();
        bad_dot[0] += 1;
        let bad_cells = vec![
            Cell::new("k", n_k, 1, k_vals),
            Cell::new("q", n_q, 1, q_vals),
            Cell::new("dot", n_dot, 1, bad_dot),
        ];
        let mut bad = CellSegment::seal(&bad_cells, Params::default());
        let bad_com = bad.commitment();
        let bad_proof = claim_matmul(
            &mut bad,
            (ck, cq, cd),
            (&vk, &vq, &vd),
            &mut Transcript::new(&new_challenger()),
            (ln, log_d, lp),
        );
        let bad_claims = bad.claims().to_vec();
        let bad_w = bad.prove(&mut new_challenger());
        assert!(verify(&bad_com, &bad_claims, &bad_w, &mut new_challenger()));
        let mut vtr = Transcript::new(&new_challenger());
        assert!(!verify_matmul(
            &bad_com,
            &bad_claims,
            (ck, cq, cd),
            (&vk, &vq, &vd),
            &bad_proof,
            &mut vtr
        ));
    }

    fn matmul_i64(a: &[i64], b: &[i64], n: usize, m: usize, p: usize) -> Vec<i64> {
        let mut c = vec![0i64; n * p];
        for i in 0..n {
            for j in 0..p {
                let mut acc = 0i64;
                for k in 0..m {
                    acc += a[i * m + k] * b[k * p + j];
                }
                c[i * p + j] = acc;
            }
        }
        c
    }

    #[test]
    fn add_claim_roundtrip_and_soundness() {
        let mut rng = StdRng::seed_from_u64(9);
        let n = 8;
        let a: Vec<i64> = (0..n).map(|_| rng.random::<i64>() % 40 - 20).collect();
        let b: Vec<i64> = (0..n).map(|_| rng.random::<i64>() % 40 - 20).collect();
        let c: Vec<i64> = (0..n).map(|_| rng.random::<i64>() % 40 - 20).collect();
        let out: Vec<i64> = (0..n).map(|k| a[k] + b[k] + c[k]).collect();

        let mk = |o: &[i64]| {
            vec![
                Cell::new("a", n, 1, a.clone()),
                Cell::new("b", n, 1, b.clone()),
                Cell::new("c", n, 1, c.clone()),
                Cell::new("out", n, 1, o.to_vec()),
            ]
        };

        let mut seg = CellSegment::seal(&mk(&out), Params::default());
        let com = seg.commitment();
        let (ca, cb, cc, co) = (seg.id("a"), seg.id("b"), seg.id("c"), seg.id("out"));

        let mut tr = Transcript::new(&new_challenger());
        let proof = claim_add(&mut seg, co, &[ca, cb, cc], &mut tr);

        let claims = seg.claims().to_vec();
        let wproof = seg.prove(&mut new_challenger());
        assert!(verify(&com, &claims, &wproof, &mut new_challenger()));

        let mut vtr = Transcript::new(&new_challenger());
        assert!(verify_add(&com, &claims, &proof, &mut vtr));

        // A wrong sum is rejected at the sumcheck's first round.
        let mut wrong = out.clone();
        wrong[3] += 1;
        let mut bad = CellSegment::seal(&mk(&wrong), Params::default());
        let bad_com = bad.commitment();
        let bad_proof = claim_add(
            &mut bad,
            co,
            &[ca, cb, cc],
            &mut Transcript::new(&new_challenger()),
        );
        let bad_claims = bad.claims().to_vec();
        let bad_w = bad.prove(&mut new_challenger());
        assert!(verify(&bad_com, &bad_claims, &bad_w, &mut new_challenger()));
        let mut vtr = Transcript::new(&new_challenger());
        assert!(!verify_add(&bad_com, &bad_claims, &bad_proof, &mut vtr));
    }

    #[test]
    fn weighted_sum_row_roundtrip_and_soundness() {
        // seq = 1, experts = 2, hidden = 4.
        let experts = 2;
        let hidden = 4;
        let t0 = vec![1i64, 2, 3, 4];
        let t1 = vec![5i64, 6, 7, 8];
        let choice = vec![2i64, 3];
        let out: Vec<i64> = (0..hidden)
            .map(|j| choice[0] * t0[j] + choice[1] * t1[j])
            .collect();

        let mk = |o: &[i64]| {
            vec![
                Cell::new("t0", hidden, 1, t0.clone()),
                Cell::new("t1", hidden, 1, t1.clone()),
                Cell::new("choice", experts, 1, choice.clone()),
                Cell::new("out", hidden, 1, o.to_vec()),
            ]
        };

        let mut seg = CellSegment::seal(&mk(&out), Params::default());
        let com = seg.commitment();
        let (c0, c1) = (seg.id("t0"), seg.id("t1"));
        let (cc, co) = (seg.id("choice"), seg.id("out"));

        let mut tr = Transcript::new(&new_challenger());
        let proof = claim_weighted_sum(&mut seg, co, &[c0, c1], cc, 0, hidden, experts, &mut tr);

        let claims = seg.claims().to_vec();
        let wproof = seg.prove(&mut new_challenger());
        assert!(verify(&com, &claims, &wproof, &mut new_challenger()));

        let mut vtr = Transcript::new(&new_challenger());
        assert!(verify_weighted_sum(&com, &claims, &proof, &mut vtr));

        // A wrong combine is rejected.
        let mut wrong = out.clone();
        wrong[2] += 1;
        let mut bad = CellSegment::seal(&mk(&wrong), Params::default());
        let bad_com = bad.commitment();
        let bad_proof = claim_weighted_sum(
            &mut bad,
            co,
            &[c0, c1],
            cc,
            0,
            hidden,
            experts,
            &mut Transcript::new(&new_challenger()),
        );
        let bad_claims = bad.claims().to_vec();
        let bad_w = bad.prove(&mut new_challenger());
        assert!(verify(&bad_com, &bad_claims, &bad_w, &mut new_challenger()));
        let mut vtr = Transcript::new(&new_challenger());
        assert!(!verify_weighted_sum(
            &bad_com,
            &bad_claims,
            &bad_proof,
            &mut vtr
        ));
    }

    #[test]
    fn reshape_roundtrip_and_soundness() {
        let vals = vec![3i64, -1, 4, -1, 5, -9, 2, 6];
        let mk = |o: &[i64]| {
            vec![
                Cell::new("in", vals.len(), 1, vals.clone()),
                Cell::new("out", vals.len(), 1, o.to_vec()),
            ]
        };

        let mut seg = CellSegment::seal(&mk(&vals), Params::default());
        let com = seg.commitment();
        let (ci, co) = (seg.id("in"), seg.id("out"));
        let proof = claim_reshape(&mut seg, ci, co, &mut Transcript::new(&new_challenger()));

        let claims = seg.claims().to_vec();
        let wproof = seg.prove(&mut new_challenger());
        assert!(verify(&com, &claims, &wproof, &mut new_challenger()));
        let mut vtr = Transcript::new(&new_challenger());
        assert!(verify_reshape(&com, &claims, &proof, &mut vtr));

        let mut wrong = vals.clone();
        wrong[5] += 1;
        let mut bad = CellSegment::seal(&mk(&wrong), Params::default());
        let bad_com = bad.commitment();
        let bad_proof = claim_reshape(&mut bad, ci, co, &mut Transcript::new(&new_challenger()));
        let bad_claims = bad.claims().to_vec();
        let bad_w = bad.prove(&mut new_challenger());
        let mut vtr = Transcript::new(&new_challenger());
        assert!(!verify_reshape(&bad_com, &bad_claims, &bad_proof, &mut vtr));
        // The tampered segment still proves its own (wrong) data.
        assert!(verify(&bad_com, &bad_claims, &bad_w, &mut new_challenger()));
    }

    #[test]
    fn mul_claim_roundtrip_and_soundness() {
        let n = 8;
        let a: Vec<i64> = (0..n).map(|i| (i as i64 % 5) - 2).collect();
        let b: Vec<i64> = (0..n).map(|i| (i as i64 % 3) - 1).collect();
        let c: Vec<i64> = (0..n).map(|i| a[i] * b[i]).collect();

        let mk = |p: &[i64]| {
            vec![
                Cell::new("a", n, 1, a.clone()),
                Cell::new("b", n, 1, b.clone()),
                Cell::new("c", n, 1, p.to_vec()),
            ]
        };

        let mut seg = CellSegment::seal(&mk(&c), Params::default());
        let com = seg.commitment();
        let (ca, cb, cc) = (seg.id("a"), seg.id("b"), seg.id("c"));
        let proof = claim_mul(
            &mut seg,
            (ca, cb, cc),
            &mut Transcript::new(&new_challenger()),
        );

        let claims = seg.claims().to_vec();
        let wproof = seg.prove(&mut new_challenger());
        assert!(verify(&com, &claims, &wproof, &mut new_challenger()));
        let mut vtr = Transcript::new(&new_challenger());
        assert!(verify_mul(&com, &claims, &proof, &mut vtr));

        // One wrong product entry breaks the claim.
        let mut wrong = c.clone();
        wrong[2] += 1;
        let mut bad = CellSegment::seal(&mk(&wrong), Params::default());
        let bad_com = bad.commitment();
        let bad_proof = claim_mul(
            &mut bad,
            (ca, cb, cc),
            &mut Transcript::new(&new_challenger()),
        );
        let bad_claims = bad.claims().to_vec();
        let bad_w = bad.prove(&mut new_challenger());
        assert!(verify(&bad_com, &bad_claims, &bad_w, &mut new_challenger()));
        let mut vtr = Transcript::new(&new_challenger());
        assert!(!verify_mul(&bad_com, &bad_claims, &bad_proof, &mut vtr));

        // The zero-sum trap: two compensating errors keep `Σ(a·b − c) = 0`, so a
        // zero-sum formulation would accept them. Only `c̃(r) = Σ eq(r,i)a_i b_i`
        // rejects this.
        let mut compensating = c.clone();
        compensating[1] += 1;
        compensating[2] -= 1;
        assert_eq!(compensating.iter().sum::<i64>(), c.iter().sum::<i64>());
        let mut trap = CellSegment::seal(&mk(&compensating), Params::default());
        let trap_com = trap.commitment();
        let trap_proof = claim_mul(
            &mut trap,
            (ca, cb, cc),
            &mut Transcript::new(&new_challenger()),
        );
        let trap_claims = trap.claims().to_vec();
        let trap_w = trap.prove(&mut new_challenger());
        assert!(verify(
            &trap_com,
            &trap_claims,
            &trap_w,
            &mut new_challenger()
        ));
        let mut vtr = Transcript::new(&new_challenger());
        assert!(
            !verify_mul(&trap_com, &trap_claims, &trap_proof, &mut vtr),
            "an equal-sum tamper must be rejected"
        );
    }

    #[test]
    fn sum_squares_roundtrip_and_soundness() {
        let x = vec![1i64, 2, 3, 4, -5, 6, 7, 8];
        let mk = |v: &[i64]| vec![Cell::new("x", v.len(), 1, v.to_vec())];

        let mut seg = CellSegment::seal(&mk(&x), Params::default());
        let com = seg.commitment();
        let cx = seg.id("x");
        let proof = claim_sum_squares(&mut seg, cx, &mut Transcript::new(&new_challenger()));
        assert_eq!(proof.sum, Ext::from(Fp::new(204)));

        let claims = seg.claims().to_vec();
        let wproof = seg.prove(&mut new_challenger());
        assert!(verify(&com, &claims, &wproof, &mut new_challenger()));
        let mut vtr = Transcript::new(&new_challenger());
        assert!(verify_sum_squares(&com, &claims, &proof, &mut vtr));

        // A tampered vector yields a different sum, which the op's reciprocal
        // consistency check would catch; and a proof with a wrong claimed sum is
        // rejected outright.
        let mut wrong = x.clone();
        wrong[0] += 1;
        let mut bad = CellSegment::seal(&mk(&wrong), Params::default());
        let bad_com = bad.commitment();
        let bad_proof = claim_sum_squares(&mut bad, cx, &mut Transcript::new(&new_challenger()));
        assert_ne!(bad_proof.sum, proof.sum);
        let bad_claims = bad.claims().to_vec();
        let mut vtr = Transcript::new(&new_challenger());
        // Self-consistent against its own (tampered) root — the mismatch lives in
        // the sum it reports, which the norm op's reciprocal check consumes.
        assert!(verify_sum_squares(
            &bad_com,
            &bad_claims,
            &bad_proof,
            &mut vtr
        ));

        let mut off_by_one = proof.clone();
        off_by_one.sum += Ext::ONE;
        let mut vtr = Transcript::new(&new_challenger());
        assert!(!verify_sum_squares(&com, &claims, &off_by_one, &mut vtr));
    }

    #[test]
    fn planned_segment_with_derived_cells() {
        // The norm op's shape: x, gamma (weights), and the derived h = x⊙γ, y = h·inv_q.
        let d = 3;
        let n = 1usize << d;
        let x: Vec<i64> = (0..n).map(|i| (i as i64 % 5) - 2).collect();
        let gamma = vec![3i64; n];
        let inv_q = 2i64;
        let h: Vec<i64> = x.iter().map(|&v| v * gamma[0]).collect();
        let y: Vec<i64> = h.iter().map(|&v| v * inv_q).collect();

        // Pass 1: plan. Two independent writers of `h` must agree.
        let mut plan = SegmentCells::new();
        let id_x = plan.cell("x", n, 1, x.clone());
        let id_g = plan.cell("gamma", n, 1, gamma.clone());
        let id_h = plan.cell("h", n, 1, h.clone());
        assert_eq!(plan.cell("h", n, 1, h.clone()), id_h, "idempotent");
        let id_y = plan.cell("y", n, 1, y.clone());
        assert_eq!(plan.len(), 4);

        let mut seg = plan.seal(Params::default());
        assert_eq!(
            (seg.id("x"), seg.id("gamma"), seg.id("h")),
            (id_x, id_g, id_h)
        );
        assert_eq!(seg.id("y"), id_y);
        let com = seg.commitment();

        // Pass 2: claim.
        let mut tr = Transcript::new(&new_challenger());
        let ss = claim_sum_squares(&mut seg, id_x, &mut tr);
        let mul = claim_mul(&mut seg, (id_x, id_g, id_h), &mut tr);

        // y = inv_q · h, checked at one point.
        tr.observe_val(Fp::new(id_h as u64));
        tr.observe_val(Fp::new(id_y as u64));
        let r: Vec<Ext> = (0..d).map(|_| tr.sample_ext()).collect();
        let h_val = seg.claim_id(id_h, &[0], &r)[0];
        let y_val = seg.claim_id(id_y, &[0], &r)[0];
        assert_eq!(y_val, Ext::from(Fp::new(inv_q as u64)) * h_val);

        let claims = seg.claims().to_vec();
        let wproof = seg.prove(&mut new_challenger());
        assert!(verify(&com, &claims, &wproof, &mut new_challenger()));

        // Verifier side.
        let mut vtr = Transcript::new(&new_challenger());
        assert!(verify_sum_squares(&com, &claims, &ss, &mut vtr));
        assert!(verify_mul(&com, &claims, &mul, &mut vtr));
        vtr.observe_val(Fp::new(id_h as u64));
        vtr.observe_val(Fp::new(id_y as u64));
        let vr: Vec<Ext> = (0..d).map(|_| vtr.sample_ext()).collect();
        assert_eq!(vr, r);
        let vh = claim_value_at_id(&com, &claims, id_h, &r).unwrap();
        let vy = claim_value_at_id(&com, &claims, id_y, &r).unwrap();
        assert_eq!(vy, Ext::from(Fp::new(inv_q as u64)) * vh);

        // A segment whose `y` is not `inv_q · h` breaks the op's check.
        let mut wrong_plan = SegmentCells::new();
        wrong_plan.cell("x", n, 1, x.clone());
        wrong_plan.cell("gamma", n, 1, gamma.clone());
        wrong_plan.cell("h", n, 1, h.clone());
        let mut wy = y.clone();
        wy[2] += 1;
        wrong_plan.cell("y", n, 1, wy);
        let mut bad = wrong_plan.seal(Params::default());
        let bad_com = bad.commitment();
        let _ = claim_mul(
            &mut bad,
            (id_x, id_g, id_h),
            &mut Transcript::new(&new_challenger()),
        );
        let _ = claim_sum_squares(&mut bad, id_x, &mut Transcript::new(&new_challenger()));
        let mut btr = Transcript::new(&new_challenger());
        btr.observe_val(Fp::new(id_h as u64));
        btr.observe_val(Fp::new(id_y as u64));
        let br: Vec<Ext> = (0..d).map(|_| btr.sample_ext()).collect();
        bad.claim_id(id_h, &[0], &br);
        bad.claim_id(id_y, &[0], &br);
        let bclaims = bad.claims().to_vec();
        let bh = claim_value_at_id(&bad_com, &bclaims, id_h, &br).unwrap();
        let by = claim_value_at_id(&bad_com, &bclaims, id_y, &br).unwrap();
        assert_ne!(by, Ext::from(Fp::new(inv_q as u64)) * bh);
    }

    #[test]
    fn two_segments_are_independent() {
        let mut rng = StdRng::seed_from_u64(3);
        let a = cell("a", 8, 2, &mut rng);
        let b = cell("b", 8, 2, &mut rng);
        let z = rand_point(&mut rng, 3);

        let mut s0 = CellSegment::seal(std::slice::from_ref(&a), Params::default());
        s0.claim("a", &z);
        let c0 = s0.commitment();
        let claims0 = s0.claims().to_vec();
        let p0 = s0.prove(&mut new_challenger());

        let mut s1 = CellSegment::seal(std::slice::from_ref(&b), Params::default());
        s1.claim("b", &z);
        let c1 = s1.commitment();
        let claims1 = s1.claims().to_vec();
        let p1 = s1.prove(&mut new_challenger());

        assert_ne!(c0.pcs.root, c1.pcs.root);

        let mut vch = new_challenger();
        assert!(verify(&c0, &claims0, &p0, &mut vch));
        let mut vch = new_challenger();
        assert!(verify(&c1, &claims1, &p1, &mut vch));

        // Each proof is bound to its own root.
        let mut vch = new_challenger();
        assert!(!verify(&c1, &claims0, &p0, &mut vch));
        let mut vch = new_challenger();
        assert!(!verify(&c0, &claims1, &p1, &mut vch));

        // An empty claim list fails closed instead of verifying vacuously.
        let mut vch = new_challenger();
        assert!(!verify(&c0, &[], &p0, &mut vch));
    }

    #[test]
    fn mul_against_a_preshared_weight_root() {
        use crate::pcs::whir_stacked::Committer;

        let a_vals = vec![1i64, 2, 3, 4];
        let w_vals = vec![2i64, 3, 4, 5];
        let c_vals: Vec<i64> = (0..4).map(|i| a_vals[i] * w_vals[i]).collect();

        // The weight lives in its own segment (the cp_commit output).
        let mut wc = Committer::new();
        wc.add_vec(
            &w_vals
                .iter()
                .map(|&x| Fp::new(x as u64))
                .collect::<Vec<_>>(),
        );
        let mut w_prov = wc.seal(Params::default());
        let w_com = w_prov.commitment().clone();

        let mut seg = CellSegment::seal(
            &[
                Cell::new("a", 4, 1, a_vals.clone()),
                Cell::new("c", 4, 1, c_vals.clone()),
            ],
            Params::default(),
        );
        let com = seg.commitment();
        let (ca, cc) = (seg.id("a"), seg.id("c"));

        let mut tr = Transcript::new(&new_challenger());
        let proof = claim_mul_weight(&mut seg, ca, &mut w_prov, 0, cc, "w", &mut tr);

        let claims = seg.claims().to_vec();
        let wproof = seg.prove(&mut new_challenger());
        let wt_proof = w_prov.prove(&mut new_challenger());

        assert!(verify(&com, &claims, &wproof, &mut new_challenger()));
        let mut vtr = Transcript::new(&new_challenger());
        assert!(verify_mul_weight(
            &com, &claims, &w_com, &wt_proof, &proof, &mut vtr
        ));

        // A product that is not `a ⊙ w` is rejected by the sumcheck.
        let mut bad_c = c_vals.clone();
        bad_c[1] += 1;
        let mut bad = CellSegment::seal(
            &[Cell::new("a", 4, 1, a_vals), Cell::new("c", 4, 1, bad_c)],
            Params::default(),
        );
        let bad_com = bad.commitment();
        let mut wc2 = Committer::new();
        wc2.add_vec(
            &w_vals
                .iter()
                .map(|&x| Fp::new(x as u64))
                .collect::<Vec<_>>(),
        );
        let mut w_prov2 = wc2.seal(Params::default());
        let bad_proof = claim_mul_weight(
            &mut bad,
            ca,
            &mut w_prov2,
            0,
            cc,
            "w",
            &mut Transcript::new(&new_challenger()),
        );
        let bad_claims = bad.claims().to_vec();
        let mut vtr = Transcript::new(&new_challenger());
        assert!(!verify_mul_weight(
            &bad_com,
            &bad_claims,
            &w_com,
            &wt_proof,
            &bad_proof,
            &mut vtr
        ));
    }

    #[test]
    fn matmul_against_a_preshared_weight_root() {
        use crate::pcs::whir_stacked::Committer;

        // W: 2x2, B: 2x1, C = W·B.
        let w_vals = vec![1i64, 2, 3, 4];
        let b_vals = vec![2i64, 3];
        let c_vals = vec![
            w_vals[0] * b_vals[0] + w_vals[1] * b_vals[1],
            w_vals[2] * b_vals[0] + w_vals[3] * b_vals[1],
        ];

        let mut wc = Committer::new();
        wc.add_vec(
            &w_vals
                .iter()
                .map(|&x| Fp::new(x as u64))
                .collect::<Vec<_>>(),
        );
        let mut w_prov = wc.seal(Params::default());
        let w_com = w_prov.commitment().clone();

        let mut seg = CellSegment::seal(
            &[
                Cell::new("b", 2, 1, b_vals.clone()),
                Cell::new("c", 2, 1, c_vals.clone()),
            ],
            Params::default(),
        );
        let com = seg.commitment();
        let (cb, cc) = (seg.id("b"), seg.id("c"));

        let mut tr = Transcript::new(&new_challenger());
        let proof = claim_matmul_weight(
            &mut seg,
            &mut w_prov,
            0,
            &View::Whole,
            (cb, cc),
            (&View::Whole, &View::Whole),
            "W",
            &mut tr,
            (1, 1, 0),
        );

        let claims = seg.claims().to_vec();
        let wproof = seg.prove(&mut new_challenger());
        let wt_proof = w_prov.prove(&mut new_challenger());

        assert!(verify(&com, &claims, &wproof, &mut new_challenger()));
        let mut vtr = Transcript::new(&new_challenger());
        assert!(verify_matmul_weight(
            &com,
            &claims,
            &w_com,
            &wt_proof,
            &View::Whole,
            (&View::Whole, &View::Whole),
            &proof,
            &mut vtr
        ));

        // A wrong product is rejected.
        let mut bad_c = c_vals.clone();
        bad_c[0] += 1;
        let mut bad = CellSegment::seal(
            &[Cell::new("b", 2, 1, b_vals), Cell::new("c", 2, 1, bad_c)],
            Params::default(),
        );
        let bad_com = bad.commitment();
        let mut wc2 = Committer::new();
        wc2.add_vec(
            &w_vals
                .iter()
                .map(|&x| Fp::new(x as u64))
                .collect::<Vec<_>>(),
        );
        let mut w_prov2 = wc2.seal(Params::default());
        let bad_proof = claim_matmul_weight(
            &mut bad,
            &mut w_prov2,
            0,
            &View::Whole,
            (cb, cc),
            (&View::Whole, &View::Whole),
            "W",
            &mut Transcript::new(&new_challenger()),
            (1, 1, 0),
        );
        let bad_claims = bad.claims().to_vec();
        let mut vtr = Transcript::new(&new_challenger());
        assert!(!verify_matmul_weight(
            &bad_com,
            &bad_claims,
            &w_com,
            &wt_proof,
            &View::Whole,
            (&View::Whole, &View::Whole),
            &bad_proof,
            &mut vtr
        ));
    }

    #[test]
    fn matmul_on_a_block_view_of_a_preshared_weight() {
        use crate::pcs::whir_stacked::Committer;

        // W is 4x2; the op uses rows [2,4) as a 2x2 matrix.
        let w_vals = vec![9i64, 9, 9, 9, 1, 2, 3, 4];
        let b_vals = vec![2i64, 3];
        // C = W_block · B with W_block = [[1,2],[3,4]].
        let c_vals = vec![1 * 2 + 2 * 3, 3 * 2 + 4 * 3];

        let mut wc = Committer::new();
        wc.add_vec(
            &w_vals
                .iter()
                .map(|&x| Fp::new(x as u64))
                .collect::<Vec<_>>(),
        );
        let mut w_prov = wc.seal(Params::default());
        let w_com = w_prov.commitment().clone();
        let m_w = w_prov.table_m(0);
        assert_eq!(m_w, 3, "eight entries padded to 2^3");

        let mut seg = CellSegment::seal(
            &[
                Cell::new("b", 2, 1, b_vals.clone()),
                Cell::new("c", 2, 1, c_vals.clone()),
            ],
            Params::default(),
        );
        let com = seg.commitment();
        let (cb, cc) = (seg.id("b"), seg.id("c"));
        // Rows [2,4) = the high row bit set, both columns.
        let view = View::block(m_w, 4, 2);

        let mut tr = Transcript::new(&new_challenger());
        let proof = claim_matmul_weight(
            &mut seg,
            &mut w_prov,
            0,
            &view,
            (cb, cc),
            (&View::Whole, &View::Whole),
            "W",
            &mut tr,
            (1, 1, 0),
        );
        let claims = seg.claims().to_vec();
        let wproof = seg.prove(&mut new_challenger());
        let wt_proof = w_prov.prove(&mut new_challenger());

        assert!(verify(&com, &claims, &wproof, &mut new_challenger()));
        let mut vtr = Transcript::new(&new_challenger());
        assert!(verify_matmul_weight(
            &com,
            &claims,
            &w_com,
            &wt_proof,
            &view,
            (&View::Whole, &View::Whole),
            &proof,
            &mut vtr
        ));

        // The whole-matrix view is not what the proof used, so it is rejected.
        let mut vtr = Transcript::new(&new_challenger());
        assert!(!verify_matmul_weight(
            &com,
            &claims,
            &w_com,
            &wt_proof,
            &View::Whole,
            (&View::Whole, &View::Whole),
            &proof,
            &mut vtr
        ));
    }

    #[test]
    fn chunked_weight_uses_eq_weights() {
        use crate::pcs::whir_stacked::Committer;

        // W is 4x2, split into two 2x2 chunk roots (embed/lm_head shape).
        let w_rows = [[1i64, 2], [3, 4], [5, 6], [7, 8]];
        let b_vals = vec![2i64, 3];
        let c_vals: Vec<i64> = (0..4)
            .map(|i| w_rows[i][0] * b_vals[0] + w_rows[i][1] * b_vals[1])
            .collect();

        let mut chunks: Vec<(String, SegmentProver, usize)> = Vec::new();
        for (ci, half) in w_rows.chunks(2).enumerate() {
            let flat: Vec<Fp> = half
                .iter()
                .flat_map(|r| r.iter().map(|&x| Fp::new(x as u64)))
                .collect();
            let mut wc = Committer::new();
            wc.add_vec(&flat);
            chunks.push((format!("W{ci}"), wc.seal(Params::default()), 0));
        }
        let coms: Vec<_> = chunks.iter().map(|c| c.1.commitment().clone()).collect();

        let mut seg = CellSegment::seal(
            &[
                Cell::new("b", 2, 1, b_vals.clone()),
                Cell::new("c", 4, 1, c_vals.clone()),
            ],
            Params::default(),
        );
        let com = seg.commitment();
        let (cb, cc) = (seg.id("b"), seg.id("c"));

        // ln = 2 rows, lm = 1 column, lp = 0; two chunks (g = 1).
        let mut handles: Vec<(&str, &mut SegmentProver, usize)> = chunks
            .iter_mut()
            .map(|(n, p, c)| (n.as_str(), p, *c))
            .collect();
        let mut tr = Transcript::new(&new_challenger());
        let proof =
            claim_matmul_weight_chunked(&mut seg, &mut handles, (cb, cc), &mut tr, (2, 1, 0));
        assert_eq!(proof.w.len(), 2);

        let claims = seg.claims().to_vec();
        let wproof = seg.prove(&mut new_challenger());
        let provers: Vec<SegmentProver> = chunks.into_iter().map(|(_, p, _)| p).collect();
        let proofs: Vec<whir_stacked::Proof> = provers
            .into_iter()
            .map(|p| p.prove(&mut new_challenger()))
            .collect();
        let parts: Vec<(&whir_stacked::SegmentCommitment, &whir_stacked::Proof)> = coms
            .iter()
            .zip(proofs.iter())
            .map(|(c, p)| (c, p))
            .collect();

        assert!(verify(&com, &claims, &wproof, &mut new_challenger()));
        let mut vtr = Transcript::new(&new_challenger());
        assert!(verify_matmul_weight_chunked(
            &com, &claims, &parts, &proof, &mut vtr
        ));

        // A proof that composed the chunks with a plain sum instead of eq weights
        // would fail here: the verifier recomputes eq from `r`.
        let mut bad = proof.clone();
        bad.a_val += Ext::ONE;
        let mut vtr = Transcript::new(&new_challenger());
        assert!(!verify_matmul_weight_chunked(
            &com, &claims, &parts, &bad, &mut vtr
        ));

        // Dropping a chunk root is rejected.
        let mut vtr = Transcript::new(&new_challenger());
        assert!(!verify_matmul_weight_chunked(
            &com,
            &claims,
            &parts[..1],
            &proof,
            &mut vtr
        ));
    }
}
