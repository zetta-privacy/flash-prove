//! The segment driver: plan a segment's cells, seal it, walk its ops.
//!
//! This is the shape `prove_gkr_succinct` takes after the port:
//!
//! ```text
//!   plan    SegmentCells::cell(name, rows, cols, values)   // trace + derived cells
//!   seal    SegmentCells::seal(params) -> CellSegment      // ONE root
//!   walk    claim_* per op, in spec order                 // claims bound to that root
//!   prove   seg.prove(challenge) -> ONE WHIR proof
//! ```
//!
//! The verifier replays the same op list against `commitment` + `claims`, so the
//! only shared secret is the op order — which is public (the model spec).
//!
//! # Scope
//!
//! Ops are added here one at a time as their primitives land. Anything not yet
//! listed returns [`DriverError::Unsupported`]; it is never silently skipped.
//! Currently driven: `add`, `mul`, `reshape` — the ops that need no derived
//! cells, so the planner is exactly the trace cells.

use crate::field::{Ext, Fp};
use crate::gkr::requant::RequantSegmentProof;
use crate::gkr::segment::{
    claim_add, claim_mul, claim_reshape, verify_add, verify_mul, verify_reshape, AddSegmentProof,
    CellClaim, CellSegmentCommitment, MatmulWeightProof, MulSegmentProof, ReshapeSegmentProof,
    SegmentCells,
};
use crate::model::spec::Op;
use crate::model::Cell;
use crate::model::WeightMatrix;
use crate::pcs::transcript::{new_challenger, Transcript};
use crate::pcs::whir_stacked::{self, Params, SegmentProver};
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};

/// One op's share of a segment proof.
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SegmentOpProof {
    Add(AddSegmentProof),
    /// A shifted `add`: the raw sum `y = Σ inputs` plus `out = requant(y, shift)`.
    AddRq {
        out: usize,
        y: usize,
        add: AddSegmentProof,
        rq: RequantSegmentProof,
    },
    Mul(MulSegmentProof),
    Reshape(ReshapeSegmentProof),
    /// A shifted `mul`: the raw product `y = a·b` plus `out = requant(y, shift)`.
    MulRq {
        out: usize,
        y: usize,
        mul: crate::gkr::segment::MulSegmentProof,
        rq: RequantSegmentProof,
    },
    /// `y = W·x` plus the requant that produces the trace's output cell.
    Matmul {
        mm: MatmulWeightProof,
        rq: RequantSegmentProof,
    },
    /// A whole-cell graph lookup (`silu`).
    Silu(crate::gkr::logup::Lookup2SegmentProof),
    /// One proof per sequence position.
    WeightedSum(Vec<crate::gkr::segment::WeightedSumRowProof>),
    /// One proof per row of the normalised cell.
    Norm(Vec<crate::gkr::nonlinear::NormRowProof>),
    /// One proof per (token, head) unit of the attention scores.
    Softmax {
        rows: Vec<crate::gkr::nonlinear::SoftmaxRowProof>,
        lookup: crate::gkr::logup::Lookup2SegmentProof,
        over_lookup: crate::gkr::logup::Lookup2SegmentProof,
        /// One requant for the whole op's `y -> nrm`.
        requant: RequantSegmentProof,
    },
    /// `rot = R·inp` plus the requant lookup.
    Rope(crate::gkr::nonlinear::RopeSegmentProof),
    /// `out = emb[window]`, read from the embedding's own root.
    Embed(crate::gkr::nonlinear::EmbedSegmentProof),
    /// Per-row selection plus the global sign lookup.
    TopK {
        /// Resolved ids, carried in the proof so the verifier never rebuilds names.
        cells: crate::gkr::selection::TopKCells,
        rows: Vec<crate::gkr::selection::TopKRowProof>,
        lookup: crate::gkr::logup::Lookup2SegmentProof,
    },
    /// One proof per (position, head) of the weighted sum over `v`.
    WeightedV(Vec<crate::gkr::attention::WeightedVSegmentProof>),
    /// One proof per (position, head) of the attention scores.
    Scores(Vec<crate::gkr::attention::ScoresSegmentProof>),
}

/// Everything a segment's verifier needs.
#[derive(Clone, Serialize, Deserialize)]
pub struct SegmentOutput {
    pub commitment: CellSegmentCommitment,
    pub claims: Vec<CellClaim>,
    pub proof: whir_stacked::Proof,
    pub ops: Vec<SegmentOpProof>,
    /// The head argmax, when this segment carries it.
    pub head: Option<crate::gkr::head::HeadSegmentProof>,
}

/// Why a segment could not be driven.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DriverError {
    /// The op has no driver wiring yet.
    Unsupported(&'static str),
    /// An op names a cell the segment does not hold.
    UnknownCell(String),
    /// An op's inputs do not have the shapes its weight implies.
    ShapeMismatch(String),
}

/// `(rows, head_dim)` of a `rope` input cell.
///
/// A reshape only re-labels the flat data, so the trace records the same `(rows, cols)`
/// for `x` and its reshaped head view. The rotation's shape therefore comes from the
/// model's head width, not from the cell's column count: `rows = flat / head_dim`.
fn rope_dims((lr, lc): (usize, usize), dims: &crate::model::spec::Dims) -> Option<(usize, usize)> {
    let flat = lr * lc;
    let (h, d) = (dims.heads, dims.head_dim);
    if h == 0 || d == 0 || flat % d != 0 {
        return None;
    }
    Some((flat / d, d))
}

/// The op's name, for the [`DriverError::Unsupported`] report.
fn op_name(op: &Op) -> &'static str {
    match op {
        Op::Embed { .. } => "embed",
        Op::Matmul { .. } => "matmul",
        Op::Scores { .. } => "scores",
        Op::CausalSoftmax { .. } => "causal_softmax",
        Op::WeightedV { .. } => "weighted_v",
        Op::Silu { .. } => "silu",
        Op::Mul { .. } => "mul",
        Op::Add { .. } => "add",
        Op::TopK { .. } => "top_k",
        Op::WeightedSum { .. } => "weighted_sum",
        Op::Reshape { .. } => "reshape",
        Op::Norm { .. } => "norm",
        Op::Rope { .. } => "rope",
    }
}

/// Every cell name an op reads or writes.
fn op_cells(op: &Op) -> Vec<&str> {
    match op {
        Op::Add { inputs, output, .. } => inputs
            .iter()
            .map(|s| s.as_str())
            .chain([output.as_str()])
            .collect(),
        Op::Mul { a, b, output, .. } => vec![a, b, output],
        Op::Reshape { input, output, .. } => vec![input, output],
        _ => Vec::new(),
    }
}

/// Prove one segment.
pub fn prove_segment(
    ops: &[Op],
    cells: &[Cell],
    params: Params,
) -> Result<SegmentOutput, DriverError> {
    let mut plan = SegmentCells::new();
    for c in cells {
        plan.cell(&c.name, c.rows, c.cols, c.values.clone());
    }
    // Validate every referenced cell before the plan is consumed by sealing.
    for op in ops {
        for name in op_cells(op) {
            if plan.id(name).is_none() {
                return Err(DriverError::UnknownCell(name.to_string()));
            }
        }
        if !matches!(op, Op::Add { .. } | Op::Mul { .. } | Op::Reshape { .. }) {
            return Err(DriverError::Unsupported(op_name(op)));
        }
    }

    let mut seg = plan.seal(params);
    let mut tr = Transcript::new(&new_challenger());
    let mut proofs = Vec::with_capacity(ops.len());
    for op in ops {
        match op {
            Op::Add { inputs, output, .. } => {
                let ins: Vec<usize> = inputs.iter().map(|n| seg.id(n)).collect();
                let out_id = seg.id(output);
                proofs.push(SegmentOpProof::Add(claim_add(
                    &mut seg, out_id, &ins, &mut tr,
                )));
            }
            Op::Mul { a, b, output, .. } => {
                let ids = (seg.id(a), seg.id(b), seg.id(output));
                proofs.push(SegmentOpProof::Mul(claim_mul(&mut seg, ids, &mut tr)));
            }
            Op::Reshape { input, output, .. } => {
                let (ci, co) = (seg.id(input), seg.id(output));
                proofs.push(SegmentOpProof::Reshape(claim_reshape(
                    &mut seg, ci, co, &mut tr,
                )));
            }
            other => return Err(DriverError::Unsupported(op_name(other))),
        }
    }

    let commitment = seg.commitment();
    let claims = seg.claims().to_vec();
    let proof = seg.prove(&mut new_challenger());
    Ok(SegmentOutput {
        commitment,
        claims,
        proof,
        ops: proofs,
        head: None,
    })
}

/// [`prove_segment_with_head`] without a head tail.
#[allow(clippy::too_many_arguments)]
pub fn prove_segment_with_weights(
    ops: &[Op],
    cells: &[Cell],
    weights: &[WeightMatrix],
    dims: &crate::model::spec::Dims,
    (digits, alpha, beta, rec_q, eps): (usize, Fp, Fp, u32, f64),
    bounds: &LookupBounds,
    params: Params,
    window: &[usize],
) -> Result<SegmentBundle, DriverError> {
    prove_segment_with_head(
        ops,
        cells,
        weights,
        dims,
        (digits, alpha, beta, rec_q, eps),
        bounds,
        params,
        window,
        None,
    )
}

/// [`verify_segment_with_head`] without a head tail.
#[allow(clippy::too_many_arguments)]
pub fn verify_segment_with_weights(
    ops: &[Op],
    bundle: &SegmentBundle,
    published: &std::collections::HashMap<String, whir_stacked::SegmentCommitment>,
    weights: &[WeightMatrix],
    dims: &crate::model::spec::Dims,
    (digits, alpha, beta, rec_q, eps): (usize, Fp, Fp, u32, f64),
    bounds: &LookupBounds,
    window: &[usize],
) -> bool {
    verify_segment_with_head(
        ops,
        bundle,
        published,
        weights,
        dims,
        (digits, alpha, beta, rec_q, eps),
        bounds,
        window,
        None,
    )
}

/// Verify a segment against the op list it was proved for.
pub fn verify_segment(ops: &[Op], out: &SegmentOutput) -> bool {
    let claims = crate::gkr::segment::Claims::new(&out.claims);
    if out.ops.len() != ops.len() {
        return false;
    }
    // The verifier's own name → id map (the plan order is part of the commitment).
    let id = |name: &str| out.commitment.cell(name);
    let mut tr = Transcript::new(&new_challenger());

    for (op, proof) in ops.iter().zip(&out.ops) {
        match (op, proof) {
            (Op::Add { inputs, output, .. }, SegmentOpProof::Add(p)) => {
                let Some(out_id) = id(output) else {
                    return false;
                };
                let mut want: Vec<usize> = Vec::with_capacity(inputs.len());
                for n in inputs {
                    match id(n) {
                        Some(i) => want.push(i),
                        None => return false,
                    }
                }
                if p.out != out_id || p.inputs.iter().map(|t| t.0).ne(want.iter().copied()) {
                    return false;
                }
                if !verify_add(&out.commitment, &claims, p, &mut tr) {
                    return false;
                }
            }
            (Op::Mul { a, b, output, .. }, SegmentOpProof::Mul(p)) => {
                let (Some(ca), Some(cb), Some(cc)) = (id(a), id(b), id(output)) else {
                    return false;
                };
                if (p.a, p.b, p.c) != (ca, cb, Some(cc)) {
                    return false;
                }
                if !verify_mul(&out.commitment, &claims, p, &mut tr) {
                    return false;
                }
            }
            (Op::Reshape { input, output, .. }, SegmentOpProof::Reshape(p)) => {
                let (Some(ci), Some(co)) = (id(input), id(output)) else {
                    return false;
                };
                if (p.input, p.output) != (ci, co) {
                    return false;
                }
                if !verify_reshape(&out.commitment, &claims, p, &mut tr) {
                    return false;
                }
            }
            _ => return false,
        }
    }

    // The op claims are only sound if the segment's own openings check out.
    crate::gkr::segment::verify(&out.commitment, &claims, &out.proof, &mut new_challenger())
}

/// Output shapes, threaded op to op exactly as the arithmetization resolves them.
///
/// The trace does not carry shapes, so the driver derives them from the spec: the
/// root cells are seeded by the caller, and every op's output shape follows from
/// its inputs (plus weight dimensions for `matmul`). Per-row and per-head ops read
/// these to know how many rows to plan and how wide each is.
#[derive(Clone, Debug, Default)]
pub struct Shapes {
    map: std::collections::HashMap<String, Vec<usize>>,
    wdims: std::collections::HashMap<String, (usize, usize)>,
}

impl Shapes {
    /// `wdims` is `(name, rows, cols)` per weight tensor.
    pub fn new(wdims: &[(String, usize, usize)]) -> Self {
        Self {
            map: Default::default(),
            wdims: wdims
                .iter()
                .map(|(n, r, c)| (n.clone(), (*r, *c)))
                .collect(),
        }
    }

    /// Seed a root cell's shape (the trace cells, or `embed`'s output).
    pub fn seed(&mut self, name: &str, shape: Vec<usize>) {
        self.map.insert(name.to_string(), shape);
    }

    pub fn of(&self, name: &str) -> Option<&[usize]> {
        self.map.get(name).map(|v| v.as_slice())
    }

    /// Rows × columns of a 2-D cell, as the trace stores it.
    pub fn rows_cols(&self, name: &str) -> Option<(usize, usize)> {
        let s = self.of(name)?;
        let rows = s[..s.len() - 1].iter().product::<usize>();
        Some((rows, *s.last()?))
    }

    fn copy_from(&mut self, src: &str, dst: &str) -> Result<(), DriverError> {
        let s = self
            .of(src)
            .ok_or_else(|| DriverError::UnknownCell(src.to_string()))?
            .to_vec();
        self.seed(dst, s);
        Ok(())
    }

    /// Resolve one op's output shape(s).
    pub fn apply(&mut self, op: &Op) -> Result<(), DriverError> {
        match op {
            Op::Norm { input, output, .. }
            | Op::Rope { input, output, .. }
            | Op::Silu { input, output, .. }
            | Op::CausalSoftmax { input, output } => self.copy_from(input, output),
            Op::Mul { a, output, .. } => self.copy_from(a, output),
            Op::Add { inputs, output, .. } => {
                let first = inputs
                    .first()
                    .ok_or_else(|| DriverError::Unsupported("add with no inputs"))?;
                self.copy_from(first, output)
            }
            Op::TopK {
                input,
                output,
                choice,
                ..
            } => {
                self.copy_from(input, output)?;
                self.copy_from(input, choice)
            }
            Op::Matmul {
                input,
                weight,
                output,
                ..
            } => {
                let mut s = self
                    .of(input)
                    .ok_or_else(|| DriverError::UnknownCell(input.clone()))?
                    .to_vec();
                let (rows, _cols) = self
                    .wdims
                    .get(weight)
                    .copied()
                    .ok_or_else(|| DriverError::UnknownCell(weight.clone()))?;
                *s.last_mut().expect("non-empty shape") = rows;
                self.seed(output, s);
                Ok(())
            }
            Op::Reshape {
                input,
                output,
                shape,
            } => {
                let ish = self
                    .of(input)
                    .ok_or_else(|| DriverError::UnknownCell(input.clone()))?
                    .to_vec();
                let numel: usize = ish.iter().product();
                let neg = shape.iter().filter(|&&s| s == -1).count();
                let known: usize = shape
                    .iter()
                    .filter(|&&s| s != -1)
                    .map(|&s| s as usize)
                    .product();
                let inferred = if neg == 1 { numel / known.max(1) } else { 1 };
                let resolved: Vec<usize> = shape
                    .iter()
                    .map(|&s| if s == -1 { inferred } else { s as usize })
                    .collect();
                self.seed(output, resolved);
                Ok(())
            }
            other => Err(DriverError::Unsupported(op_name(other))),
        }
    }
}

/// Lazily committed pre-shared weight tensors.
///
/// A weight is committed the first time an op claims against it and freed after its
/// last claim, so peak memory is *activation segment + one tensor*. Each tensor gets
/// its own root (WHIR needs the whole stacked witness to open anything in it, so one
/// global weight root would force every weight resident at once).
pub struct WeightSet {
    tensors: std::collections::HashMap<String, WeightMatrix>,
    provers: std::collections::HashMap<String, SegmentProver>,
    params: Params,
}

impl WeightSet {
    pub fn new(weights: &[WeightMatrix], params: Params) -> Self {
        Self {
            tensors: weights
                .iter()
                .map(|w| (w.name.clone(), w.clone()))
                .collect(),
            provers: Default::default(),
            params,
        }
    }

    /// The tensor's prover, committed on first use. A weight is one flat column, so
    /// the table id an op opens is always `0`.
    pub fn get_or_commit(&mut self, name: &str) -> Result<&mut SegmentProver, DriverError> {
        if !self.provers.contains_key(name) {
            let w = self
                .tensors
                .get(name)
                .ok_or_else(|| DriverError::UnknownCell(name.to_string()))?;
            let t0 = std::time::Instant::now();
            let prov = weight_prover(w, self.params);
            eprintln!(
                "  weight {name} committed in {:.1}s",
                t0.elapsed().as_secs_f64()
            );
            self.provers.insert(name.to_string(), prov);
        }
        Ok(self.provers.get_mut(name).expect("just inserted"))
    }

    /// Drop a tensor's witness once no op reads it any more.
    pub fn release(&mut self, name: &str) {
        self.provers.remove(name);
    }

    pub fn is_open(&self, name: &str) -> bool {
        self.provers.contains_key(name)
    }

    /// Prove every **claimed** tensor once, and hand back its root alongside.
    ///
    /// A tensor with no claims has nothing to prove, so it is skipped rather than
    /// producing an empty proof.
    pub fn prove_all(self) -> Vec<(String, whir_stacked::SegmentCommitment, whir_stacked::Proof)> {
        self.provers
            .into_iter()
            .filter(|(_, p)| p.num_claims() > 0)
            .map(|(name, p)| {
                let com = p.commitment().clone();
                let proof = p.prove(&mut new_challenger());
                (name, com, proof)
            })
            .collect()
    }
}

/// A weight tensor as a one-column table — the same convention the commit phase
/// uses ([`crate::gkr::segment::CellSegment::weight_segment`]).
fn weight_prover(w: &WeightMatrix, params: Params) -> SegmentProver {
    crate::gkr::segment::CellSegment::weight_segment(w, params)
}

/// The lookup table bounds a segment's ops need.
///
/// These are **public** and must be identical on both sides: the verifier cannot
/// recompute them from the trace (it does not have it) and must not take them from
/// the prover, or the table — and with it every lookup — is prover-chosen. They
/// belong in the pre-shared config written by the commit phase, next to the weight
/// roots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LookupBounds {
    /// `silu` gate-projection inputs.
    pub silu_lo: i64,
    pub silu_hi: i64,
    /// The MoE routing delta `e·score − i − τ`.
    pub sign_lo: i64,
    pub sign_hi: i64,
    /// The most negative `score − max` seen by causal softmax; the exp domain is
    /// `[exp_lo, 0]`.
    pub exp_lo: i64,
    /// The head logits' range (the argmax sign table).
    pub score_lo: i64,
    pub score_hi: i64,
    /// The rotated-value domain `[rope_lo, rope_hi]` (public; the caller must widen
    /// it to cover the rotation it intends to prove).
    pub rope_lo: i64,
    pub rope_hi: i64,
}

impl LookupBounds {
    /// The public silu table fingerprints `α + β·x + β²·silu(x)` over the bounds.
    pub fn silu_table(&self, alpha: Fp, beta: Fp) -> Vec<Fp> {
        let n = (self.silu_hi - self.silu_lo + 1) as usize;
        let mut v: Vec<Fp> = (self.silu_lo..=self.silu_hi)
            .map(|x| {
                let fx = Fp::new(crate::model::spec::silu(x) as u64);
                alpha + beta * signed(x) + beta * beta * fx
            })
            .collect();
        v.resize(n.next_power_of_two(), Fp::ONE);
        v
    }

    /// The logits sign table for a given `head_max`.
    pub fn head_table(&self, head_max: i64, alpha: Fp, beta: Fp) -> Vec<Fp> {
        crate::gkr::head::head_table(self.score_lo, self.score_hi, head_max, alpha, beta)
    }

    /// The routing sign table `α + β·δ + β²·sign(δ)`.
    pub fn sign_table(&self, alpha: Fp, beta: Fp) -> Vec<Fp> {
        crate::gkr::selection::sign_table(self.sign_lo, self.sign_hi, alpha, beta)
    }

    /// The exp table `α + β·d + β²·exp_q15(d)` over `[exp_lo, 0]`.
    pub fn exp_table(&self, alpha: Fp, beta: Fp) -> Vec<Fp> {
        let n = (0 - self.exp_lo + 1) as usize;
        let mut v: Vec<Fp> = (self.exp_lo..=0)
            .map(|x| {
                let fx = Fp::new(crate::model::spec::exp_q15(x) as u64);
                alpha + beta * signed(x) + beta * beta * fx
            })
            .collect();
        v.resize(n.next_power_of_two(), Fp::ONE);
        v
    }

    /// The rotated-value table `α + β·x + β²·requant(x, k)` over `[rope_lo, rope_hi]`.
    pub fn rope_table(&self, k: u32, alpha: Fp, beta: Fp) -> Vec<Fp> {
        let n = self.rope_big_n();
        let mut v: Vec<Fp> = (self.rope_lo..=self.rope_hi)
            .map(|x| {
                let out = crate::model::spec::requant(x, k);
                alpha + beta * signed(x) + beta * beta * signed(out)
            })
            .collect();
        v.resize(n.next_power_of_two(), Fp::ONE);
        v
    }

    pub fn rope_big_n(&self) -> usize {
        (self.rope_hi - self.rope_lo + 1).max(1) as usize
    }

    /// The true (unpadded) size of a table, which the LogUp multiplicity offset needs.
    pub fn silu_big_n(&self) -> usize {
        (self.silu_hi - self.silu_lo + 1) as usize
    }
    pub fn sign_big_n(&self) -> usize {
        (self.sign_hi - self.sign_lo + 1) as usize
    }
    pub fn exp_big_n(&self) -> usize {
        (0 - self.exp_lo + 1) as usize
    }
    pub fn score_big_n(&self) -> usize {
        (self.score_hi - self.score_lo + 1) as usize
    }

    /// Derive the bounds from a trace, exactly as [`crate::gkr::bounds::CommittedConfig`] does.
    pub fn derive(spec: &crate::model::spec::ModelSpec, trace: &crate::model::Trace) -> Self {
        let c = crate::gkr::bounds::CommittedConfig::derive(spec, trace);
        Self {
            silu_lo: c.silu_lo,
            silu_hi: c.silu_hi,
            sign_lo: c.sign_lo,
            sign_hi: c.sign_hi,
            exp_lo: c.exp_lo,
            score_lo: c.score_lo,
            score_hi: c.score_hi,
            rope_lo: 0,
            rope_hi: 0,
        }
    }
}

use crate::gkr::segment::signed;

/// The driver's tensor root must be the one `cp_commit` published.
///
/// Without this check a prover supplies its own weight root and every weight claim
/// is vacuous — this function *is* the soundness of the pre-shared path.
pub fn check_preshared(
    name: &str,
    committed: &whir_stacked::SegmentCommitment,
    published: &std::collections::HashMap<String, whir_stacked::SegmentCommitment>,
) -> bool {
    published
        .get(name)
        .is_some_and(|p| p.root == committed.root && p.tables == committed.tables)
}

/// The head argmax, which is a segment *tail* rather than an `Op`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadSpec {
    /// The logits cell; its last row is the one being checked.
    pub logits: String,
    pub token: usize,
    pub max: i64,
}

/// A segment proved together with the weight roots it opened against.
#[derive(Clone, Serialize, Deserialize)]
pub struct SegmentBundle {
    pub segment: SegmentOutput,
    /// `(name, root, proof)` per weight tensor the segment claimed on.
    pub weights: Vec<(String, whir_stacked::SegmentCommitment, whir_stacked::Proof)>,
}

/// The cells an op writes.
fn op_outputs(op: &Op) -> Vec<&str> {
    match op {
        Op::TopK { output, choice, .. } => vec![output.as_str(), choice.as_str()],
        Op::Add { output, .. }
        | Op::Mul { output, .. }
        | Op::Reshape { output, .. }
        | Op::Matmul { output, .. }
        | Op::Scores { output, .. }
        | Op::CausalSoftmax { output, .. }
        | Op::WeightedV { output, .. }
        | Op::Silu { output, .. }
        | Op::WeightedSum { output, .. }
        | Op::Norm { output, .. }
        | Op::Rope { output, .. }
        | Op::Embed { output, .. } => vec![output.as_str()],
    }
}

/// The cells an op reads (weights are committed separately, not as cells).
fn op_inputs(op: &Op) -> Vec<&str> {
    match op {
        Op::Add { inputs, .. } => inputs.iter().map(|s| s.as_str()).collect(),
        Op::Mul { a, b, .. } => vec![a.as_str(), b.as_str()],
        Op::Reshape { input, .. }
        | Op::Matmul { input, .. }
        | Op::CausalSoftmax { input, .. }
        | Op::Silu { input, .. }
        | Op::TopK { input, .. }
        | Op::Norm { input, .. }
        | Op::Rope { input, .. } => vec![input.as_str()],
        Op::Scores { q, k, .. } => vec![q.as_str(), k.as_str()],
        Op::WeightedV { weights, v, .. } => vec![weights.as_str(), v.as_str()],
        Op::WeightedSum { choice, terms, .. } => std::iter::once(choice.as_str())
            .chain(terms.iter().map(|s| s.as_str()))
            .collect(),
        Op::Embed { .. } => Vec::new(),
    }
}

/// Split the model's cells into segments of at most `max_cells` tables each.
///
/// Segments follow the spec's op order, so an op and its output cell always land in
/// the same segment. Both sides run this: the prover from `spec.cells`, the verifier
/// from the same spec. Any input that lives in an earlier segment is **copied** into
/// the later one and tied back by a [`link_point`] opening, so no claim ever needs
/// another segment's root.
pub fn plan_segments(ops: &[Op], cells: &[String], max_cells: usize) -> Vec<Vec<String>> {
    let max_cells = max_cells.max(1);
    let mut out: Vec<Vec<String>> = Vec::new();
    let mut cur: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for op in ops {
        for name in op_outputs(op) {
            if seen.insert(name.to_string()) {
                cur.push(name.to_string());
            }
        }
        if cur.len() >= max_cells {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() || out.is_empty() {
        out.push(cur);
    }
    // Cells no op produces are model inputs: they belong to the first segment, so an
    // op that reads one usually finds it in its own segment.
    let tail: Vec<String> = cells
        .iter()
        .filter(|c| !seen.contains(*c))
        .cloned()
        .collect();
    if !tail.is_empty() {
        out.first_mut().expect("at least one segment").extend(tail);
    }
    out
}

/// One cross-segment tie: `name` is committed in `from` and copied into `to`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinkSpec {
    pub from: usize,
    pub to: usize,
    pub name: String,
}

/// The Fiat–Shamir point at which a link is opened on both sides.
///
/// Derived from public data only (`to`, the cell name, and its arity), so the earlier
/// segment can record its opening before the later one is even planned.
pub fn link_point(to: usize, name: &str, m: usize) -> Vec<Ext> {
    let mut tr = Transcript::new(&crate::pcs::transcript::new_challenger());
    tr.observe_val(Fp::new(to as u64));
    for b in name.as_bytes() {
        tr.observe_val(Fp::new(*b as u64));
    }
    (0..m).map(|_| tr.sample_ext()).collect()
}

/// Every tie the segment split needs, in `(from, to, name)` order.
pub fn link_schedule(segment_cells: &[Vec<String>], per_segment: &[Vec<Op>]) -> Vec<LinkSpec> {
    let mut out = Vec::new();
    for (to, ops) in per_segment.iter().enumerate() {
        let own = &segment_cells[to];
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for op in ops {
            for name in op_inputs(op) {
                if own.iter().any(|c| c == name) || !seen.insert(name) {
                    continue;
                }
                if let Some(from) = segment_cells
                    .iter()
                    .position(|cs| cs.iter().any(|c| c == name))
                {
                    // A copy may look *forward* too (an unproduced model input lives in
                    // segment 0), so both directions are linked.
                    if from != to {
                        out.push(LinkSpec {
                            from,
                            to,
                            name: name.to_string(),
                        });
                    }
                }
            }
        }
    }
    out
}

/// The openings each segment must record for the link schedule.
///
/// Index `s` holds the `(name, point)` pairs for `s`'s outgoing links (the original
/// cell) **and** its incoming ones (the copy), so both sides of every tie end up as
/// ordinary claims in the two segments' proofs.
pub fn link_openings(
    links: &[LinkSpec],
    segments: usize,
    m_of: impl Fn(&str) -> usize,
) -> Vec<Vec<(String, Vec<Ext>)>> {
    let mut out = vec![Vec::new(); segments];
    for l in links {
        let point = link_point(l.to, &l.name, m_of(&l.name));
        out[l.from].push((l.name.clone(), point.clone()));
        out[l.to].push((l.name.clone(), point));
    }
    out
}

/// The cells an op reads that its own segment does not commit, deduplicated.
pub fn external_inputs(own: &[String], ops: &[Op]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for op in ops {
        for name in op_inputs(op) {
            if own.iter().any(|c| c == name) {
                continue;
            }
            if !out.iter().any(|c| c == name) {
                out.push(name.to_string());
            }
        }
    }
    out
}

/// `y = W·x` for a single input row.
fn matvec(w: &WeightMatrix, x: &[i64]) -> Vec<i64> {
    assert_eq!(x.len(), w.cols, "matvec input width");
    (0..w.rows)
        .map(|i| {
            (0..w.cols)
                .map(|j| w.values[i * w.cols + j] as i64 * x[j])
                .sum()
        })
        .collect()
}

/// Prove one segment of `matmul` ops, reading each weight from the pre-shared
/// registry.
///
/// `y = W·x` is a planned cell (its integer values are computed here, since the
/// trace only carries the requantised output), so `claim_matmul_weight` binds `y` to
/// the weight's own root and `claim_requant` binds the trace output to `y`.
pub fn prove_segment_with_head(
    ops: &[Op],
    cells: &[Cell],
    weights: &[WeightMatrix],
    dims: &crate::model::spec::Dims,
    params_digits: (usize, Fp, Fp, u32, f64),
    bounds: &LookupBounds,
    params: Params,
    window: &[usize],
    head: Option<&HeadSpec>,
) -> Result<SegmentBundle, DriverError> {
    prove_segment_full(
        ops,
        cells,
        weights,
        dims,
        params_digits,
        bounds,
        params,
        window,
        head,
        &[],
    )
}

/// [`prove_segment_with_head`] plus extra openings of **this** segment's cells at
/// caller-chosen points — the outgoing side of the cross-segment links.
#[allow(clippy::too_many_arguments)]
pub fn prove_segment_full(
    ops: &[Op],
    cells: &[Cell],
    weights: &[WeightMatrix],
    dims: &crate::model::spec::Dims,
    (digits, alpha, beta, rec_q, eps): (usize, Fp, Fp, u32, f64),
    bounds: &LookupBounds,
    params: Params,
    window: &[usize],
    head: Option<&HeadSpec>,
    extra: &[(String, Vec<Ext>)],
) -> Result<SegmentBundle, DriverError> {
    let _ = (dims, window, head);
    let by_name: std::collections::HashMap<&str, &WeightMatrix> =
        weights.iter().map(|w| (w.name.as_str(), w)).collect();

    // Pass 1: plan every derived cell, keeping the names for the claim pass.
    let mut plan = crate::gkr::plan::Plan::new(cells);
    let mut planned: Vec<Option<PlannedOp>> = Vec::with_capacity(ops.len());
    for (i, op) in ops.iter().enumerate() {
        if i % 25 == 0 {
            eprintln!("  plan {i}/{}: {}", ops.len(), op_name(op));
        }
        match op {
            // Cell-to-cell linear ops: no derived cells, claimed directly.
            Op::Add { shift: None, .. } | Op::Reshape { .. } => {
                planned.push(Some(PlannedOp::Plain))
            }
            // A shifted `add` also rescales: plan the raw sum and its requant.
            Op::Add {
                inputs,
                output,
                shift: Some(k),
            } => {
                if *k == 0 {
                    return Err(DriverError::Unsupported("add shift 0"));
                }
                let (rows, cols) = plan
                    .logical(output)
                    .ok_or_else(|| DriverError::UnknownCell(output.clone()))?;
                let n = rows * cols;
                let mut sum = vec![0i64; n];
                for name in inputs {
                    let v = plan.values(name);
                    assert_eq!(v.len(), n, "add operands must match");
                    for (s, x) in sum.iter_mut().zip(v) {
                        *s += x;
                    }
                }
                let y_name = format!("__add.{i}.y");
                plan.add_logical(&y_name, rows, cols, sum);
                let rq = plan.requant(
                    &format!("__add.{i}.rq"),
                    &y_name,
                    output,
                    *k as usize,
                    digits,
                );
                planned.push(Some(PlannedOp::AddRq(rq)));
            }
            // A shifted `mul` also rescales: plan the raw product and its requant.
            Op::Mul {
                a,
                b,
                output,
                shift: Some(k),
            } => {
                let av = plan.values(a).to_vec();
                let bv = plan.values(b).to_vec();
                assert_eq!(av.len(), bv.len(), "mul operands must match");
                let prod: Vec<i64> = av.iter().zip(&bv).map(|(&x, &y)| x * y).collect();
                let y_name = format!("__mul.{i}.y");
                plan.add(&y_name, prod.len(), 1, prod);
                let rq = plan.requant(
                    &format!("__mul.{i}.rq"),
                    &y_name,
                    output,
                    *k as usize,
                    digits,
                );
                planned.push(Some(PlannedOp::MulRq(rq)));
            }
            Op::Mul { shift: None, .. } => planned.push(Some(PlannedOp::Plain)),
            Op::Silu { input, output } => {
                let table = bounds.silu_table(alpha, beta);
                planned.push(Some(PlannedOp::Silu(plan.lookup2(
                    &format!("__silu.{i}"),
                    input,
                    output,
                    (alpha, beta),
                    &table,
                    bounds.silu_big_n(),
                ))));
            }
            Op::Matmul {
                input,
                weight,
                output,
                shift,
            } => {
                let w = by_name
                    .get(weight.as_str())
                    .ok_or_else(|| DriverError::UnknownCell(weight.clone()))?;
                // The input is `(rows, cols)`: a per-token vector when `cols == 1`,
                // otherwise a `seq × hidden` batch the op applies row by row.
                let Some((in_rows, in_cols)) = plan.logical(input) else {
                    return Err(DriverError::UnknownCell(input.clone()));
                };
                if in_cols != 1 && in_cols != w.cols {
                    return Err(DriverError::ShapeMismatch(format!(
                        "{input}: {} cols, weight has {}",
                        in_cols, w.cols
                    )));
                }
                let xs = plan.values(input).to_vec();
                let k = *shift as usize;
                let y: Vec<i64> = if in_cols == 1 {
                    if xs.len() != w.cols {
                        return Err(DriverError::ShapeMismatch(format!(
                            "{input}: {} values, weight has {} cols",
                            xs.len(),
                            w.cols
                        )));
                    }
                    matvec(w, &xs)
                } else {
                    let mut y = Vec::with_capacity(in_rows * w.rows);
                    for t in 0..in_rows {
                        y.extend(matvec(w, &xs[t * in_cols..(t + 1) * in_cols]));
                    }
                    y
                };
                let (y_rows, y_cols) = if in_cols == 1 {
                    (y.len(), 1)
                } else {
                    (in_rows, w.rows)
                };
                let y_name = format!("__mm.{i}.y");
                plan.add_logical(&y_name, y_rows, y_cols, y);
                let rq = plan.requant(&format!("__mm.{i}.rq"), &y_name, output, k, digits);
                planned.push(Some(PlannedOp::Matmul { rq }));
            }
            Op::WeightedSum { .. } => planned.push(Some(PlannedOp::WeightedSum)),
            Op::Scores {
                q,
                k,
                output,
                scale,
                shift,
            } => {
                let (h, d) = (dims.heads, dims.head_dim);
                let Some((lq, hd)) = plan.logical(q) else {
                    return Err(DriverError::UnknownCell(q.clone()));
                };
                if hd != h * d {
                    return Err(DriverError::UnknownCell(q.clone()));
                }
                let q_shift = shift.unwrap_or(crate::model::spec::RECIP_Q);
                let inv_q = ((1i64 << q_shift) as f64 / *scale).round() as i64;

                let qs = plan.values(q).to_vec();
                let ks = plan.values(k).to_vec();
                // One cell set per head covering **every** query position: the head's
                // whole `lq × lq` score matrix. This is the unit the claim works on.
                let mut per_head = Vec::with_capacity(h);
                for hi in 0..h {
                    let mut dot = Vec::with_capacity(lq * lq);
                    for li in 0..lq {
                        for t in 0..lq {
                            let mut acc = 0i64;
                            for j in 0..d {
                                acc += qs[(li * h + hi) * d + j] * ks[(t * h + hi) * d + j];
                            }
                            dot.push(acc);
                        }
                    }
                    let y: Vec<i64> = dot.iter().map(|&v| v * inv_q).collect();
                    let sr: Vec<i64> = y
                        .iter()
                        .map(|&v| crate::model::spec::requant(v, q_shift))
                        .collect();
                    let pre = format!("__sc.{i}.{hi}");
                    plan.add(&format!("{pre}.dot"), lq * lq, 1, dot);
                    plan.add(&format!("{pre}.y"), lq * lq, 1, y);
                    plan.add(&format!("{pre}.s"), lq * lq, 1, sr);
                    let requant = plan.requant(
                        &format!("{pre}.rq"),
                        &format!("{pre}.y"),
                        &format!("{pre}.s"),
                        q_shift as usize,
                        digits,
                    );
                    per_head.push(ScoresPlan {
                        hi,
                        inv_q,
                        dot: format!("{pre}.dot"),
                        y: format!("{pre}.y"),
                        s: format!("{pre}.s"),
                        requant,
                    });
                }
                let _ = output;
                planned.push(Some(PlannedOp::Scores(per_head)));
            }
            Op::WeightedV { weights, v, .. } => {
                let (h, d) = (dims.heads, dims.head_dim);
                let Some((lq, hd)) = plan.logical(v) else {
                    return Err(DriverError::UnknownCell(v.clone()));
                };
                if hd != h * d {
                    return Err(DriverError::UnknownCell(v.clone()));
                }
                let vs = plan.values(v).to_vec();
                let ws = plan.values(weights).to_vec();
                // One cell set per head, covering every query position.
                let mut per_head = Vec::with_capacity(h);
                for hi in 0..h {
                    let mut acc = Vec::with_capacity(lq * d);
                    for li in 0..lq {
                        for i in 0..d {
                            let mut sum = 0i64;
                            for t in 0..lq {
                                sum += vs[(t * h + hi) * d + i] * ws[((li * h) + hi) * lq + t];
                            }
                            acc.push(sum);
                        }
                    }
                    let ob: Vec<i64> = acc
                        .iter()
                        .map(|&x| crate::model::spec::requant(x, 15))
                        .collect();
                    let pre = format!("__wv.{i}.{hi}");
                    plan.add(&format!("{pre}.acc"), lq * d, 1, acc);
                    plan.add(&format!("{pre}.ob"), lq * d, 1, ob);
                    let requant = plan.requant(
                        &format!("{pre}.rq"),
                        &format!("{pre}.acc"),
                        &format!("{pre}.ob"),
                        15,
                        digits,
                    );
                    per_head.push(WeightedVPlan {
                        hi,
                        acc: format!("{pre}.acc"),
                        ob: format!("{pre}.ob"),
                        requant,
                    });
                }
                planned.push(Some(PlannedOp::WeightedV(per_head)));
            }
            Op::Norm { input, weight, .. } => {
                let w = by_name
                    .get(weight.as_str())
                    .ok_or_else(|| DriverError::UnknownCell(weight.clone()))?;
                let Some((rows, hidden)) = plan.logical(input) else {
                    return Err(DriverError::UnknownCell(input.clone()));
                };
                if hidden != w.values.len() {
                    return Err(DriverError::UnknownCell(weight.clone()));
                }
                let gamma: Vec<i64> = w.values.iter().map(|&v| v as i64).collect();
                plan.add(&format!("__norm.gamma.{i}"), hidden, 1, gamma.clone());
                let xs = plan.values(input).to_vec();
                let k = crate::model::spec::RECIP_Q as usize;
                let mut rqs = Vec::with_capacity(rows);
                for row in 0..rows {
                    let xr: Vec<i64> = xs[row * hidden..(row + 1) * hidden].to_vec();
                    plan.add(&format!("__norm.{i}.{row}.xr"), hidden, 1, xr.clone());
                    let hh: Vec<i64> = (0..hidden).map(|j| xr[j] * gamma[j]).collect();
                    let h_name = format!("__norm.{i}.{row}.h");
                    plan.add(&h_name, hidden, 1, hh.clone());
                    let sum_sq: i64 = xr.iter().map(|&v| v * v).sum();
                    let inv_q = crate::gkr::nonlinear::norm_inv_q(sum_sq, hidden, rec_q, eps);
                    let y: Vec<i64> = hh.iter().map(|&v| v * inv_q).collect();
                    let y_name = format!("__norm.{i}.{row}.y");
                    plan.add(&y_name, hidden, 1, y.clone());
                    let ob: Vec<i64> = y
                        .iter()
                        .map(|&v| crate::model::spec::requant(v, crate::model::spec::RECIP_Q))
                        .collect();
                    let ob_name = format!("__norm.{i}.{row}.ob");
                    plan.add(&ob_name, hidden, 1, ob);
                    rqs.push(plan.requant(
                        &format!("__norm.{i}.{row}.rq"),
                        &y_name,
                        &ob_name,
                        k,
                        digits,
                    ));
                }
                planned.push(Some(PlannedOp::Norm(rqs)));
            }
            Op::CausalSoftmax { input, output } => {
                let Some((rows, cols)) = plan.logical(input) else {
                    return Err(DriverError::UnknownCell(input.clone()));
                };
                // The scores cell is `[L, H, L]` flattened, so one flat row holds one
                // *token's* scores for every head. The softmax runs per (token, head):
                // `units` blocks of length `n = L`, with causal prefix `l = token`.
                let h = dims.heads;
                if h == 0 || cols % h != 0 || !(cols / h).is_power_of_two() {
                    return Err(DriverError::ShapeMismatch("softmax head split".into()));
                }
                let n = cols / h;
                let units = rows * h;
                let total = rows * cols;
                let table = bounds.exp_table(alpha, beta);
                let big_n = bounds.exp_big_n();
                let max_tbl = crate::gkr::nonlinear::max_table(bounds.exp_lo, alpha, beta);
                let max_big = crate::gkr::nonlinear::max_big_n(bounds.exp_lo);
                let xs = plan.values(input).to_vec();
                let mut per_unit = Vec::with_capacity(units);
                // The derived cells are op-wide; a unit claim reads one block of each.
                let mut d_all: Vec<i64> = Vec::with_capacity(total);
                let mut e_all: Vec<i64> = Vec::with_capacity(total);
                let mut over_all: Vec<i64> = Vec::with_capacity(total);
                let mut y_all: Vec<i64> = Vec::with_capacity(total);
                let mut nrm_all: Vec<i64> = Vec::with_capacity(total);
                for unit in 0..units {
                    let l = unit / h;
                    let base = unit * n;
                    let m = (0..=l).map(|i| xs[base + i]).max().unwrap_or(0);
                    let tstar = (0..=l).find(|&i| xs[base + i] == m).unwrap_or(0);
                    let mut d = vec![0i64; n];
                    let mut e = vec![0i64; n];
                    let mut over = vec![0i64; n];
                    for i in 0..n {
                        d[i] = if i <= l {
                            xs[base + i] - m
                        } else {
                            bounds.exp_lo
                        };
                        e[i] = crate::model::spec::exp_q15(d[i]);
                        over[i] = if d[i] > 0 { 1 } else { 0 };
                    }
                    let s_sum: i64 = e.iter().take(l + 1).sum();
                    let inv_q = crate::gkr::nonlinear::softmax_inv_q(s_sum, rec_q);
                    let y: Vec<i64> = e.iter().map(|&v| v * inv_q).collect();
                    let nrm: Vec<i64> = y
                        .iter()
                        .map(|&v| crate::model::spec::requant(v, crate::model::spec::RECIP_Q))
                        .collect();
                    d_all.extend_from_slice(&d);
                    e_all.extend_from_slice(&e);
                    over_all.extend_from_slice(&over);
                    y_all.extend_from_slice(&y);
                    nrm_all.extend_from_slice(&nrm);
                    per_unit.push(SoftmaxPlan { m, tstar });
                }
                let op = format!("__sm.{i}");
                let d_all_name = format!("{op}.d");
                let e_all_name = format!("{op}.e");
                let over_all_name = format!("{op}.over");
                let y_all_name = format!("{op}.y");
                let nrm_all_name = format!("{op}.nrm");
                plan.add(&d_all_name, total, 1, d_all);
                plan.add(&e_all_name, total, 1, e_all);
                plan.add(&over_all_name, total, 1, over_all);
                plan.add(&y_all_name, total, 1, y_all);
                plan.add(&nrm_all_name, total, 1, nrm_all);
                // One requant for the whole op, not one per unit.
                let requant = plan.requant(
                    &format!("{op}.rq"),
                    &y_all_name,
                    &nrm_all_name,
                    crate::model::spec::RECIP_Q as usize,
                    digits,
                );
                let lookup = plan.lookup2(
                    &format!("{op}.lk"),
                    &d_all_name,
                    &e_all_name,
                    (alpha, beta),
                    &table,
                    big_n,
                );
                let over_lookup = plan.lookup2(
                    &format!("{op}.olk"),
                    &d_all_name,
                    &over_all_name,
                    (alpha, beta),
                    &max_tbl,
                    max_big,
                );
                let _ = output;
                planned.push(Some(PlannedOp::Softmax {
                    rows_plan: per_unit,
                    lookup,
                    over_lookup,
                    requant,
                }));
            }
            Op::TopK {
                input, choice, k, ..
            } => {
                let Some((rows, e)) = plan.logical(input) else {
                    return Err(DriverError::UnknownCell(input.clone()));
                };
                let xs = plan.values(input).to_vec();
                let cs = plan.values(choice).to_vec();
                // adj = e·scores − i, per row.
                let adj: Vec<i64> = (0..rows * e)
                    .map(|i| e as i64 * xs[i] - (i % e) as i64)
                    .collect();
                let mut tau = Vec::with_capacity(rows);
                let mut tstar = Vec::with_capacity(rows);
                let mut delta = vec![0i64; rows * e];
                for row in 0..rows {
                    let mut sorted: Vec<i64> = adj[row * e..(row + 1) * e].to_vec();
                    sorted.sort_unstable_by(|a, b| b.cmp(a));
                    // `sorted` is descending: the k-th largest is index `k-1`.
                    let t = sorted[(*k).clamp(1, e) - 1];
                    let idx = (0..e).find(|&t2| adj[row * e + t2] == t).unwrap_or(0);
                    tau.push(t);
                    tstar.push(idx);
                    for i in 0..e {
                        delta[row * e + i] = adj[row * e + i] - t;
                    }
                }
                let pre = format!("__tk.{i}");
                plan.add(&format!("{pre}.adj"), rows * e, 1, adj.clone());
                plan.add(&format!("{pre}.delta"), rows * e, 1, delta.clone());
                let _ = cs;
                let lookup = plan.lookup2(
                    &format!("{pre}.lk"),
                    &format!("{pre}.delta"),
                    choice,
                    (alpha, beta),
                    &bounds.sign_table(alpha, beta),
                    bounds.sign_big_n(),
                );
                planned.push(Some(PlannedOp::TopK(TopKPlan {
                    adj: format!("{pre}.adj"),
                    delta: format!("{pre}.delta"),
                    lookup,
                    tau,
                    tstar,
                })));
            }
            Op::Rope {
                input,
                output,
                theta,
            } => {
                let Some((rows, d)) = plan.logical(input).and_then(|s| rope_dims(s, dims)) else {
                    return Err(DriverError::ShapeMismatch(format!(
                        "{input}: rope input is not seq×heads×head_dim"
                    )));
                };
                let xs = plan.values(input).to_vec();
                let rot = crate::gkr::nonlinear::rotate(&xs, rows, dims.heads, d, *theta);
                let pre = format!("__rope.{i}");
                let rot_name = format!("{pre}.rot");
                plan.add(&rot_name, rows * d, 1, rot);
                // The rotation's cos/sin tables are Q15 (`* 32768`), so the requant
                // shift is 15 — the same on both sides (see `python/run_integer.py`).
                let requant = plan.requant(
                    &format!("{pre}.rq"),
                    &rot_name,
                    output,
                    crate::gkr::nonlinear::ROPE_Q,
                    digits,
                );
                planned.push(Some(PlannedOp::Rope(RopePlan {
                    rot: rot_name,
                    requant,
                })));
            }
            Op::Embed { weight, output } => {
                let Some((l, hidden)) = plan.logical(output) else {
                    return Err(DriverError::UnknownCell(output.clone()));
                };
                let w = by_name
                    .get(weight.as_str())
                    .ok_or_else(|| DriverError::UnknownCell(weight.clone()))?;
                if w.cols != hidden {
                    return Err(DriverError::UnknownCell(weight.clone()));
                }
                if window.len() != l || window.iter().any(|&t| t >= w.rows) {
                    return Err(DriverError::Unsupported(
                        "embed: window outside the vocabulary",
                    ));
                }
                planned.push(Some(PlannedOp::Embed));
            }
        }
    }

    // The head is a segment tail: its cells must be planned before the seal too.
    let mut head_plan = None;
    if let Some(h) = head {
        let Some((rows, n)) = plan.logical(&h.logits) else {
            return Err(DriverError::UnknownCell(h.logits.clone()));
        };
        let row = rows - 1;
        let xs = plan.values(&h.logits).to_vec();
        let scores: Vec<i64> = xs[row * n..(row + 1) * n].to_vec();
        let prefix = "__head";
        plan.add(&format!("{prefix}.scores"), n, 1, scores);
        plan.add(&format!("{prefix}.ones"), n, 1, vec![1i64; n]);
        let table = bounds.head_table(h.max, alpha, beta);
        let lookup = plan.lookup2(
            &format!("{prefix}.lk"),
            &format!("{prefix}.scores"),
            &format!("{prefix}.ones"),
            (alpha, beta),
            &table,
            bounds.score_big_n(),
        );
        head_plan = Some((row, n, lookup));
    }

    eprintln!("  planned; sealing");
    let mut seg = plan.seal(params);
    eprintln!("  sealed");
    let mut set = WeightSet::new(weights, params);
    let mut tr = Transcript::new(&new_challenger());
    let mut proofs = Vec::with_capacity(ops.len());

    // Pass 2: claim, in spec order.
    let t_ops = std::time::Instant::now();
    for (i, op) in ops.iter().enumerate() {
        eprintln!(
            "  op {}/{}: {}  [t={:.1}s]",
            i,
            ops.len(),
            op_name(op),
            t_ops.elapsed().as_secs_f64()
        );
        let names = planned[i].take().expect("planned in pass 1");
        match (op, names) {
            (
                Op::Add {
                    inputs,
                    output,
                    shift: None,
                    ..
                },
                PlannedOp::Plain,
            ) => {
                let ins: Vec<usize> = inputs.iter().map(|n| seg.id(n)).collect();
                let out_id = seg.id(output);
                proofs.push(SegmentOpProof::Add(claim_add(
                    &mut seg, out_id, &ins, &mut tr,
                )));
            }
            (
                Op::Add {
                    inputs,
                    output,
                    shift: Some(k),
                },
                PlannedOp::AddRq(rq),
            ) => {
                let ins: Vec<usize> = inputs.iter().map(|n| seg.id(n)).collect();
                let out_id = seg.id(output);
                let y_id = seg.id(&format!("__add.{i}.y"));
                let add = claim_add(&mut seg, y_id, &ins, &mut tr);
                let rq_cells = rq.cells(&seg);
                let y_meta = seg.cells_meta(y_id);
                let n = y_meta.logical_rows * y_meta.logical_cols;
                let rq = crate::gkr::requant::claim_requant(
                    &mut seg,
                    rq_cells,
                    (n, *k as usize, rq.dbytes),
                    &mut tr,
                );
                proofs.push(SegmentOpProof::AddRq {
                    out: out_id,
                    y: y_id,
                    add,
                    rq,
                });
            }

            (
                Op::Mul {
                    a,
                    b,
                    output,
                    shift: None,
                },
                PlannedOp::Plain,
            ) => {
                let ids = (seg.id(a), seg.id(b), seg.id(output));
                proofs.push(SegmentOpProof::Mul(claim_mul(&mut seg, ids, &mut tr)));
            }
            (
                Op::Mul {
                    a,
                    b,
                    output,
                    shift: Some(k),
                },
                PlannedOp::MulRq(rq),
            ) => {
                let (ia, ib, io) = (seg.id(a), seg.id(b), seg.id(output));
                let y_id = seg.id(&format!("__mul.{i}.y"));
                let mul = claim_mul(&mut seg, (ia, ib, y_id), &mut tr);
                let rq_cells = rq.cells(&seg);
                let y_meta = seg.cells_meta(y_id);
                let n = y_meta.logical_rows * y_meta.logical_cols;
                let rq = crate::gkr::requant::claim_requant(
                    &mut seg,
                    rq_cells,
                    (n, *k as usize, rq.dbytes),
                    &mut tr,
                );
                proofs.push(SegmentOpProof::MulRq {
                    out: io,
                    y: y_id,
                    mul,
                    rq,
                });
            }
            (Op::Reshape { input, output, .. }, PlannedOp::Plain) => {
                let (ci, co) = (seg.id(input), seg.id(output));
                proofs.push(SegmentOpProof::Reshape(claim_reshape(
                    &mut seg, ci, co, &mut tr,
                )));
            }
            (Op::Silu { input, output }, PlannedOp::Silu(n)) => {
                let (ci, co) = (seg.id(input), seg.id(output));
                let n_rows = seg.cells_meta(ci).rows;
                let table = bounds.silu_table(alpha, beta);
                let lc = n.cells(&seg);
                debug_assert_eq!((lc.input, lc.output), (ci, co));
                let proof = crate::gkr::logup::claim_lookup2(
                    &mut seg,
                    lc,
                    (alpha, beta),
                    &table,
                    (n_rows, bounds.silu_big_n()),
                    &mut tr,
                );
                proofs.push(SegmentOpProof::Silu(proof));
            }
            (
                Op::Matmul {
                    input,
                    weight,
                    shift,
                    ..
                },
                PlannedOp::Matmul { rq },
            ) => {
                let w = by_name[weight.as_str()];
                let y_id = seg.id(&format!("__mm.{i}.y"));
                let ci = seg.id(input);
                let in_cols = seg.cells_meta(ci).logical_cols;
                let in_rows = seg.cells_meta(ci).logical_rows;
                // `(ln, lm, lp)` — arities of (output, contraction, batch). A per-token
                // input has no batch axis, so it needs no transposed views.
                let (ln, lm) = (
                    w.rows.trailing_zeros() as usize,
                    w.cols.trailing_zeros() as usize,
                );
                let lp = if in_cols == 1 {
                    0
                } else {
                    in_rows.trailing_zeros() as usize
                };
                let (b_view, c_view) = if lp == 0 {
                    (
                        crate::gkr::segment::View::Whole,
                        crate::gkr::segment::View::Whole,
                    )
                } else {
                    (
                        crate::gkr::segment::View::swap_blocks(lp, lm),
                        crate::gkr::segment::View::swap_blocks(lp, ln),
                    )
                };
                let wp = set.get_or_commit(weight)?;
                let mm = crate::gkr::segment::claim_matmul_weight(
                    &mut seg,
                    wp,
                    0,
                    &crate::gkr::segment::View::Whole,
                    (ci, y_id),
                    (&b_view, &c_view),
                    weight,
                    &mut tr,
                    (ln, lm, lp),
                );
                let rq_cells = rq.cells(&seg);
                let y_meta = seg.cells_meta(y_id);
                let n = y_meta.logical_rows * y_meta.logical_cols;
                let rq = crate::gkr::requant::claim_requant(
                    &mut seg,
                    rq_cells,
                    (n, *shift as usize, rq.dbytes),
                    &mut tr,
                );
                proofs.push(SegmentOpProof::Matmul { mm, rq });
            }
            (
                Op::WeightedSum {
                    choice,
                    terms,
                    output,
                },
                PlannedOp::WeightedSum,
            ) => {
                let co = seg.id(output);
                let cc = seg.id(choice);
                let (seq, hidden) = {
                    let m = seg.cells_meta(co);
                    (m.logical_rows, m.logical_cols)
                };
                let experts = seg.cells_meta(cc).logical_cols;
                let term_ids: Vec<usize> = terms.iter().map(|t| seg.id(t)).collect();
                let mut rows = Vec::with_capacity(seq);
                for row in 0..seq {
                    rows.push(crate::gkr::segment::claim_weighted_sum(
                        &mut seg, co, &term_ids, cc, row, hidden, experts, &mut tr,
                    ));
                }
                proofs.push(SegmentOpProof::WeightedSum(rows));
            }
            (
                Op::Norm {
                    input,
                    weight,
                    output,
                    ..
                },
                PlannedOp::Norm(rqs),
            ) => {
                let co = seg.id(output);
                let (rows, hidden) = {
                    let m = seg.cells_meta(co);
                    (m.logical_rows, m.logical_cols)
                };
                let gamma_id = seg.id(&format!("__norm.gamma.{i}"));
                let gamma_pub: Vec<Fp> = by_name[weight.as_str()]
                    .values
                    .iter()
                    .map(|&v| crate::gkr::segment::signed(v as i64))
                    .collect();
                let mut row_proofs = Vec::with_capacity(rows);
                for row in 0..rows {
                    let cells = crate::gkr::nonlinear::NormRowCells {
                        x: seg.id(input),
                        out: co,
                        gamma: gamma_id,
                        xr: seg.id(&format!("__norm.{i}.{row}.xr")),
                        h: seg.id(&format!("__norm.{i}.{row}.h")),
                        y: seg.id(&format!("__norm.{i}.{row}.y")),
                        ob: seg.id(&format!("__norm.{i}.{row}.ob")),
                        requant: rqs[row].cells(&seg),
                    };
                    row_proofs.push(crate::gkr::nonlinear::claim_norm_row(
                        &mut seg,
                        cells,
                        (row, hidden),
                        (
                            hidden,
                            crate::model::spec::RECIP_Q as usize,
                            rqs[row].dbytes,
                        ),
                        (rec_q, eps),
                        &gamma_pub,
                        &mut tr,
                    ));
                }
                proofs.push(SegmentOpProof::Norm(row_proofs));
            }
            (Op::WeightedV { weights, v, output }, PlannedOp::WeightedV(per_head)) => {
                let (cw, cv, co) = (seg.id(weights), seg.id(v), seg.id(output));
                let (h, d) = (dims.heads, dims.head_dim);
                let lq = seg.cells_meta(cv).logical_rows;
                let (log_l, log_d, log_h) = (
                    lq.trailing_zeros() as usize,
                    d.trailing_zeros() as usize,
                    h.trailing_zeros() as usize,
                );
                let mut out = Vec::with_capacity(per_head.len());
                for hp in per_head.iter() {
                    let cells = crate::gkr::attention::WeightedVCells {
                        w: cw,
                        v: cv,
                        out: co,
                        acc: seg.id(&hp.acc),
                        ob: seg.id(&hp.ob),
                        requant: hp.requant.cells(&seg),
                    };
                    out.push(crate::gkr::attention::claim_weighted_v(
                        &mut seg,
                        cells,
                        (log_l, log_d, log_h),
                        hp.hi,
                        (lq * d, 15, hp.requant.dbytes),
                        &mut tr,
                    ));
                }
                proofs.push(SegmentOpProof::WeightedV(out));
            }
            (
                Op::Scores {
                    q,
                    k,
                    output,
                    shift,
                    ..
                },
                PlannedOp::Scores(per_head),
            ) => {
                let q_shift = shift.unwrap_or(crate::model::spec::RECIP_Q);
                let (cq, ck, co) = (seg.id(q), seg.id(k), seg.id(output));
                let (h, d) = (dims.heads, dims.head_dim);
                let lq = seg.cells_meta(cq).logical_rows;
                let (log_l, log_d, log_h) = (
                    lq.trailing_zeros() as usize,
                    d.trailing_zeros() as usize,
                    h.trailing_zeros() as usize,
                );
                let mut out = Vec::with_capacity(per_head.len());
                for hp in per_head.iter() {
                    let cells = crate::gkr::attention::HeadCells {
                        q: cq,
                        k: ck,
                        out: co,
                        dot: seg.id(&hp.dot),
                        y: seg.id(&hp.y),
                        s: seg.id(&hp.s),
                        requant: hp.requant.cells(&seg),
                    };
                    out.push(crate::gkr::attention::claim_scores(
                        &mut seg,
                        cells,
                        (log_l, log_d, log_h),
                        hp.hi,
                        Fp::new(hp.inv_q as u64),
                        (lq * lq, q_shift as usize, hp.requant.dbytes),
                        &mut tr,
                    ));
                }
                proofs.push(SegmentOpProof::Scores(out));
            }
            (
                Op::CausalSoftmax { input, output },
                PlannedOp::Softmax {
                    rows_plan: per_unit,
                    lookup,
                    over_lookup,
                    requant,
                },
            ) => {
                let (ci, co) = (seg.id(input), seg.id(output));
                let (rows, cols) = {
                    let m = seg.cells_meta(ci);
                    (m.logical_rows, m.logical_cols)
                };
                let h = dims.heads;
                let n = cols / h;
                let total = rows * cols;
                let cells = crate::gkr::nonlinear::SoftmaxRowCells {
                    scores: ci,
                    probs: co,
                    d: seg.id(&format!("__sm.{i}.d")),
                    e: seg.id(&format!("__sm.{i}.e")),
                    y: seg.id(&format!("__sm.{i}.y")),
                    nrm: seg.id(&format!("__sm.{i}.nrm")),
                    over: seg.id(&format!("__sm.{i}.over")),
                };
                // One requant for the whole op, then one claim per (token, head).
                let rq_cells = requant.cells(&seg);
                let rq = crate::gkr::requant::claim_requant(
                    &mut seg,
                    rq_cells,
                    (total, crate::model::spec::RECIP_Q as usize, requant.dbytes),
                    &mut tr,
                );
                let mut row_proofs = Vec::with_capacity(per_unit.len());
                for (unit, pc) in per_unit.iter().enumerate() {
                    row_proofs.push(crate::gkr::nonlinear::claim_softmax_row(
                        &mut seg,
                        cells.clone(),
                        (unit, n, unit / h),
                        (pc.m, bounds.exp_lo, pc.tstar),
                        &mut tr,
                    ));
                }
                // One LogUp per table over every unit's query.
                let lc = lookup.cells(&seg);
                let exp_lookup = crate::gkr::logup::claim_lookup2(
                    &mut seg,
                    lc,
                    (alpha, beta),
                    &bounds.exp_table(alpha, beta),
                    (total, bounds.exp_big_n()),
                    &mut tr,
                );
                let olc = over_lookup.cells(&seg);
                let over_lk = crate::gkr::logup::claim_lookup2(
                    &mut seg,
                    olc,
                    (alpha, beta),
                    &crate::gkr::nonlinear::max_table(bounds.exp_lo, alpha, beta),
                    (total, crate::gkr::nonlinear::max_big_n(bounds.exp_lo)),
                    &mut tr,
                );
                proofs.push(SegmentOpProof::Softmax {
                    rows: row_proofs,
                    lookup: exp_lookup,
                    over_lookup: over_lk,
                    requant: rq,
                });
            }
            (
                Op::TopK {
                    input, choice, k, ..
                },
                PlannedOp::TopK(tp),
            ) => {
                let (cs, cc) = (seg.id(input), seg.id(choice));
                let (rows, e) = {
                    let m = seg.cells_meta(cs);
                    (m.logical_rows, m.logical_cols)
                };
                let cells = crate::gkr::selection::TopKCells {
                    scores: cs,
                    choice: cc,
                    adj: seg.id(&tp.adj),
                    delta: seg.id(&tp.delta),
                    lookup: tp.lookup.cells(&seg),
                };
                let mut row_proofs = Vec::with_capacity(rows);
                for row in 0..rows {
                    row_proofs.push(crate::gkr::selection::claim_top_k_row(
                        &mut seg,
                        &cells,
                        (row, e, *k, tp.tau[row], tp.tstar[row]),
                        &mut tr,
                    ));
                }
                let table = bounds.sign_table(alpha, beta);
                let lookup = crate::gkr::logup::claim_lookup2(
                    &mut seg,
                    cells.lookup.clone(),
                    (alpha, beta),
                    &table,
                    (rows * e, bounds.sign_big_n()),
                    &mut tr,
                );
                proofs.push(SegmentOpProof::TopK {
                    cells,
                    rows: row_proofs,
                    lookup,
                });
            }
            (
                Op::Rope {
                    input,
                    output,
                    theta,
                },
                PlannedOp::Rope(rp),
            ) => {
                let ci = seg.id(input);
                let (rows, d) = {
                    let m = seg.cells_meta(ci);
                    rope_dims((m.logical_rows, m.logical_cols), dims)
                }
                .ok_or_else(|| DriverError::ShapeMismatch("rope input shape".into()))?;
                let _ = output;
                let rot = seg.id(&rp.rot);
                let rq = rp.requant.cells(&seg);
                proofs.push(SegmentOpProof::Rope(crate::gkr::nonlinear::claim_rope(
                    &mut seg,
                    rot,
                    rq,
                    ci,
                    (rows, dims.heads, d),
                    *theta,
                    (crate::gkr::nonlinear::ROPE_Q, rp.requant.dbytes),
                    &mut tr,
                )));
            }
            (Op::Embed { weight, output }, PlannedOp::Embed) => {
                let co = seg.id(output);
                let (l, hidden) = {
                    let m = seg.cells_meta(co);
                    (m.logical_rows, m.logical_cols)
                };
                let w = by_name[weight.as_str()];
                let wp = set.get_or_commit(weight)?;
                let p = crate::gkr::nonlinear::claim_embed(
                    &mut seg,
                    co,
                    window,
                    (l, hidden, w.rows),
                    wp,
                    0,
                    weight,
                    &mut tr,
                );
                proofs.push(SegmentOpProof::Embed(p));
            }
            _ => return Err(DriverError::Unsupported(op_name(op))),
        }
    }

    let head_proof = match (head, head_plan) {
        (Some(h), Some((row, n, lookup))) => {
            let logits = seg.id(&h.logits);
            let lc = lookup.cells(&seg);
            let ones = lc.output;
            Some(crate::gkr::head::claim_head(
                &mut seg,
                lc,
                ones,
                (logits, row, n, h.token, h.max),
                (alpha, beta),
                (&bounds.head_table(h.max, alpha, beta), bounds.score_big_n()),
                &mut tr,
            ))
        }
        _ => None,
    };

    // Outgoing cross-segment links: open this segment's own cells at the points the
    // later segments will read. Those points are public, so the verifier replays them
    // from the commitments without touching the transcript.
    for (name, point) in extra {
        let id = seg.id(name);
        seg.claim_id(id, &[0], point);
    }

    let commitment = seg.commitment();
    let claims = seg.claims().to_vec();
    let proof = seg.prove(&mut new_challenger());
    Ok(SegmentBundle {
        segment: SegmentOutput {
            commitment,
            claims,
            proof,
            ops: proofs,
            head: head_proof,
        },
        weights: set.prove_all(),
    })
}

/// What a `top_k` needed at plan time.
struct TopKPlan {
    adj: String,
    delta: String,
    lookup: crate::gkr::plan::Lookup2Names,
    tau: Vec<i64>,
    tstar: Vec<usize>,
}

/// What a softmax unit needed at plan time.
#[allow(dead_code)]
struct SoftmaxPlan {
    m: i64,
    tstar: usize,
}

/// What pass 1 planned for one op.
#[allow(dead_code)]
enum PlannedOp {
    Silu(crate::gkr::plan::Lookup2Names),
    Matmul {
        rq: crate::gkr::plan::RequantNames,
    },
    WeightedSum,
    AddRq(crate::gkr::plan::RequantNames),
    /// Per-row requant names.
    Norm(Vec<crate::gkr::plan::RequantNames>),
    /// One `(m, tstar)` per (token, head) unit.
    Softmax {
        rows_plan: Vec<SoftmaxPlan>,
        /// The exp table's lookup, over every row's query at once.
        lookup: crate::gkr::plan::Lookup2Names,
        /// The `(d, over)` sign table's lookup, likewise.
        over_lookup: crate::gkr::plan::Lookup2Names,
        /// The op-wide `y -> nrm` requant.
        requant: crate::gkr::plan::RequantNames,
    },
    /// Per-head names.
    WeightedV(Vec<WeightedVPlan>),
    /// Per-head names.
    Scores(Vec<ScoresPlan>),
    TopK(TopKPlan),
    Rope(RopePlan),
    Embed,
    /// A shifted `mul`: the raw product plus its requant.
    MulRq(crate::gkr::plan::RequantNames),
    /// No derived cells: the op is claimed directly from its trace cells.
    Plain,
}

/// What a `rope` needed at plan time.
struct RopePlan {
    rot: String,
    requant: crate::gkr::plan::RequantNames,
}

/// What a `scores` head needed at plan time.
#[allow(dead_code)]
struct ScoresPlan {
    hi: usize,
    inv_q: i64,
    dot: String,
    y: String,
    s: String,
    requant: crate::gkr::plan::RequantNames,
}

/// What a `weighted_v` head needed at plan time.
#[allow(dead_code)]
struct WeightedVPlan {
    hi: usize,
    acc: String,
    ob: String,
    requant: crate::gkr::plan::RequantNames,
}

/// Verify a [`SegmentBundle`] against the published weight roots.
pub fn verify_segment_with_head(
    ops: &[Op],
    bundle: &SegmentBundle,
    published: &std::collections::HashMap<String, whir_stacked::SegmentCommitment>,
    weights: &[WeightMatrix],
    dims: &crate::model::spec::Dims,
    (digits, alpha, beta, rec_q, eps): (usize, Fp, Fp, u32, f64),
    bounds: &LookupBounds,
    window: &[usize],
    head: Option<&HeadSpec>,
) -> bool {
    let _ = (dims, window, head);
    // One `cell -> claims` index for the whole segment: value lookups were scanning
    // every claim in the list.
    let claims = crate::gkr::segment::Claims::new(&bundle.segment.claims);
    // Debug switch: check only the first `FLASHPROVE_VERIFY_OPS` ops. Prover and
    // verifier advance the transcript identically over a prefix, so this isolates
    // which op a rejection comes from.
    let n_ops = std::env::var("FLASHPROVE_VERIFY_OPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(ops.len());
    if bundle.segment.ops.len() != ops.len() || n_ops > ops.len() {
        return false;
    }
    if n_ops != ops.len() {
        eprintln!(
            "WARNING: only the first {n_ops} of {} ops are checked (FLASHPROVE_VERIFY_OPS)",
            ops.len()
        );
    }
    let wmap: std::collections::HashMap<
        &str,
        (&whir_stacked::SegmentCommitment, &whir_stacked::Proof),
    > = bundle
        .weights
        .iter()
        .map(|(n, c, p)| (n.as_str(), (c, p)))
        .collect();
    let mut tr = Transcript::new(&new_challenger());

    for (i, (op, proof)) in ops[..n_ops]
        .iter()
        .zip(&bundle.segment.ops[..n_ops])
        .enumerate()
    {
        match (op, proof) {
            (Op::Silu { input, output }, SegmentOpProof::Silu(p)) => {
                let (Some(ci), Some(co)) = (
                    bundle.segment.commitment.cell(input),
                    bundle.segment.commitment.cell(output),
                ) else {
                    return false;
                };
                if (p.cells.input, p.cells.output) != (ci, co) {
                    return false;
                }
                let rows = bundle.segment.commitment.cells[ci].rows;
                let table = bounds.silu_table(alpha, beta);
                if !crate::gkr::logup::verify_lookup2(
                    &bundle.segment.commitment,
                    &claims,
                    p,
                    &table,
                    (rows, bounds.silu_big_n()),
                    &mut tr,
                ) {
                    return false;
                }
            }
            (
                Op::Matmul {
                    input,
                    weight,
                    output,
                    shift,
                },
                SegmentOpProof::Matmul { mm, rq },
            ) => {
                let Some((w_com, w_proof)) = wmap.get(weight.as_str()) else {
                    return false;
                };
                // The driver's root must be the published one.
                if !check_preshared(weight, w_com, published) {
                    return false;
                }
                let (Some(ci), Some(co)) = (
                    bundle.segment.commitment.cell(input),
                    bundle.segment.commitment.cell(output),
                ) else {
                    return false;
                };
                if mm.b != ci || mm.c != rq.cells.x {
                    return false;
                }
                let (in_rows, in_cols, co_elems) = {
                    let mi = &bundle.segment.commitment.cells[ci];
                    let mo = &bundle.segment.commitment.cells[co];
                    (mi.logical_rows, mi.logical_cols, mo.rows * mo.cols)
                };
                // The batch arity is public: it is the input's row count.
                let lp = if in_cols == 1 {
                    0
                } else {
                    if in_rows != 1 << mm.lp {
                        return false;
                    }
                    mm.lp
                };
                if mm.lp != lp {
                    return false;
                }
                let (b_view, c_view) = if lp == 0 {
                    (
                        crate::gkr::segment::View::Whole,
                        crate::gkr::segment::View::Whole,
                    )
                } else {
                    (
                        crate::gkr::segment::View::swap_blocks(lp, mm.lm),
                        crate::gkr::segment::View::swap_blocks(lp, mm.ln),
                    )
                };
                if !crate::gkr::segment::verify_matmul_weight(
                    &bundle.segment.commitment,
                    &claims,
                    w_com,
                    w_proof,
                    &crate::gkr::segment::View::Whole,
                    (&b_view, &c_view),
                    mm,
                    &mut tr,
                ) {
                    return false;
                }
                if !crate::gkr::requant::verify_requant(
                    &bundle.segment.commitment,
                    &claims,
                    rq,
                    (co_elems, *shift as usize, digits),
                    &mut tr,
                ) {
                    return false;
                }
            }
            (Op::Add { inputs, output, .. }, SegmentOpProof::Add(p)) => {
                let Some(out_id) = bundle.segment.commitment.cell(output) else {
                    return false;
                };
                let mut want: Vec<usize> = Vec::with_capacity(inputs.len());
                for n in inputs {
                    match bundle.segment.commitment.cell(n) {
                        Some(i) => want.push(i),
                        None => return false,
                    }
                }
                if p.out != out_id || p.inputs.iter().map(|t| t.0).ne(want.iter().copied()) {
                    return false;
                }
                if !crate::gkr::segment::verify_add(
                    &bundle.segment.commitment,
                    &claims,
                    p,
                    &mut tr,
                ) {
                    return false;
                }
            }
            (
                Op::Add {
                    inputs,
                    output,
                    shift: Some(k),
                },
                SegmentOpProof::AddRq { out: o, y, add, rq },
            ) => {
                let Some(cc) = bundle.segment.commitment.cell(output) else {
                    return false;
                };
                let mut want: Vec<usize> = Vec::with_capacity(inputs.len());
                for n in inputs {
                    match bundle.segment.commitment.cell(n) {
                        Some(i) => want.push(i),
                        None => return false,
                    }
                }
                if *o != cc
                    || add.out != *y
                    || add.inputs.iter().map(|t| t.0).ne(want.iter().copied())
                {
                    return false;
                }
                if !crate::gkr::segment::verify_add(
                    &bundle.segment.commitment,
                    &claims,
                    add,
                    &mut tr,
                ) {
                    return false;
                }
                let Some(ym) = bundle.segment.commitment.cells.get(*y) else {
                    return false;
                };
                let n = ym.logical_rows * ym.logical_cols;
                if !crate::gkr::requant::verify_requant(
                    &bundle.segment.commitment,
                    &claims,
                    rq,
                    (n, *k as usize, digits),
                    &mut tr,
                ) {
                    return false;
                }
            }
            (
                Op::Mul {
                    a,
                    b,
                    output,
                    shift: None,
                },
                SegmentOpProof::Mul(p),
            ) => {
                let (Some(ca), Some(cb), Some(cc)) = (
                    bundle.segment.commitment.cell(a),
                    bundle.segment.commitment.cell(b),
                    bundle.segment.commitment.cell(output),
                ) else {
                    return false;
                };
                if (p.a, p.b, p.c) != (ca, cb, Some(cc)) {
                    return false;
                }
                if !crate::gkr::segment::verify_mul(
                    &bundle.segment.commitment,
                    &claims,
                    p,
                    &mut tr,
                ) {
                    return false;
                }
            }
            (
                Op::Mul {
                    a,
                    b,
                    output,
                    shift: Some(k),
                },
                SegmentOpProof::MulRq { out: o, y, mul, rq },
            ) => {
                let (Some(ca), Some(cb), Some(cc)) = (
                    bundle.segment.commitment.cell(a),
                    bundle.segment.commitment.cell(b),
                    bundle.segment.commitment.cell(output),
                ) else {
                    return false;
                };
                if *o != cc || (mul.a, mul.b, mul.c) != (ca, cb, Some(*y)) {
                    return false;
                }
                if !crate::gkr::segment::verify_mul(
                    &bundle.segment.commitment,
                    &claims,
                    mul,
                    &mut tr,
                ) {
                    return false;
                }
                let Some(ym) = bundle.segment.commitment.cells.get(*y) else {
                    return false;
                };
                let n = ym.logical_rows * ym.logical_cols;
                if !crate::gkr::requant::verify_requant(
                    &bundle.segment.commitment,
                    &claims,
                    rq,
                    (n, *k as usize, digits),
                    &mut tr,
                ) {
                    return false;
                }
            }
            (Op::Reshape { input, output, .. }, SegmentOpProof::Reshape(p)) => {
                let (Some(ci), Some(co)) = (
                    bundle.segment.commitment.cell(input),
                    bundle.segment.commitment.cell(output),
                ) else {
                    return false;
                };
                if (p.input, p.output) != (ci, co) {
                    return false;
                }
                if !crate::gkr::segment::verify_reshape(
                    &bundle.segment.commitment,
                    &claims,
                    p,
                    &mut tr,
                ) {
                    return false;
                }
            }
            (
                Op::WeightedSum {
                    choice,
                    terms,
                    output,
                },
                SegmentOpProof::WeightedSum(rows),
            ) => {
                let (Some(co), Some(cc)) = (
                    bundle.segment.commitment.cell(output),
                    bundle.segment.commitment.cell(choice),
                ) else {
                    return false;
                };
                let seq = bundle.segment.commitment.cells[co].logical_rows;
                let mut term_ids = Vec::with_capacity(terms.len());
                for t in terms {
                    match bundle.segment.commitment.cell(t) {
                        Some(i) => term_ids.push(i),
                        None => return false,
                    }
                }
                if rows.len() != seq {
                    return false;
                }
                for row in rows.iter() {
                    if row.out != co || row.choice != cc {
                        return false;
                    }
                    if !crate::gkr::segment::verify_weighted_sum(
                        &bundle.segment.commitment,
                        &claims,
                        row,
                        &mut tr,
                    ) {
                        return false;
                    }
                }
            }
            (
                Op::Norm {
                    input,
                    weight,
                    output,
                    ..
                },
                SegmentOpProof::Norm(row_proofs),
            ) => {
                let (Some(co), Some(ci)) = (
                    bundle.segment.commitment.cell(output),
                    bundle.segment.commitment.cell(input),
                ) else {
                    return false;
                };
                let Some(gamma_id) = bundle.segment.commitment.cell(&format!("__norm.gamma.{i}"))
                else {
                    return false;
                };
                let meta = &bundle.segment.commitment.cells[co];
                let (rows, hidden) = (meta.logical_rows, meta.logical_cols);
                // The verifier supplies γ itself; it never reads it from the proof.
                let Some(w) = weights.iter().find(|w| w.name == *weight) else {
                    return false;
                };
                let gamma_pub: Vec<Fp> = w
                    .values
                    .iter()
                    .map(|&v| crate::gkr::segment::signed(v as i64))
                    .collect();
                if row_proofs.len() != rows {
                    return false;
                }
                for (row, rp) in row_proofs.iter().enumerate() {
                    let c = &rp.cells;
                    if c.x != ci || c.out != co || c.gamma != gamma_id {
                        return false;
                    }
                    if !crate::gkr::nonlinear::verify_norm_row(
                        &bundle.segment.commitment,
                        &claims,
                        rp,
                        (row, hidden),
                        (hidden, crate::model::spec::RECIP_Q as usize, digits),
                        (rec_q, eps),
                        &gamma_pub,
                        &mut tr,
                    ) {
                        return false;
                    }
                }
            }
            (Op::WeightedV { output, v, .. }, SegmentOpProof::WeightedV(per_head)) => {
                let (Some(co), Some(cv)) = (
                    bundle.segment.commitment.cell(output),
                    bundle.segment.commitment.cell(v),
                ) else {
                    return false;
                };
                let (h, d) = (dims.heads, dims.head_dim);
                let lq = bundle.segment.commitment.cells[cv].logical_rows;
                let (log_l, log_d, log_h) = (
                    lq.trailing_zeros() as usize,
                    d.trailing_zeros() as usize,
                    h.trailing_zeros() as usize,
                );
                if per_head.len() != h {
                    return false;
                }
                for (hi, hp) in per_head.iter().enumerate() {
                    if hp.cells.out != co || hp.cells.v != cv || hp.head != hi {
                        return false;
                    }
                    if !crate::gkr::attention::verify_weighted_v(
                        &bundle.segment.commitment,
                        &claims,
                        hp,
                        (log_l, log_d, log_h),
                        hi,
                        (lq * d, 15, digits),
                        &mut tr,
                    ) {
                        return false;
                    }
                }
            }
            (
                Op::Scores {
                    q,
                    k,
                    output,
                    scale,
                    shift,
                },
                SegmentOpProof::Scores(per_head),
            ) => {
                let q_shift = shift.unwrap_or(crate::model::spec::RECIP_Q);
                let want_inv_q = Fp::new(((1i64 << q_shift) as f64 / *scale).round() as u64);
                let (Some(cq), Some(ck), Some(co)) = (
                    bundle.segment.commitment.cell(q),
                    bundle.segment.commitment.cell(k),
                    bundle.segment.commitment.cell(output),
                ) else {
                    return false;
                };
                let (h, d) = (dims.heads, dims.head_dim);
                let lq = bundle.segment.commitment.cells[cq].logical_rows;
                let (log_l, log_d, log_h) = (
                    lq.trailing_zeros() as usize,
                    d.trailing_zeros() as usize,
                    h.trailing_zeros() as usize,
                );
                if per_head.len() != h {
                    return false;
                }
                for (hi, hp) in per_head.iter().enumerate() {
                    let c = &hp.cells;
                    if c.q != cq
                        || c.k != ck
                        || c.out != co
                        || hp.head != hi
                        || hp.inv_q != want_inv_q
                    {
                        return false;
                    }
                    if !crate::gkr::attention::verify_scores(
                        &bundle.segment.commitment,
                        &claims,
                        hp,
                        (log_l, log_d, log_h),
                        hi,
                        hp.inv_q,
                        (lq * lq, q_shift as usize, digits),
                        &mut tr,
                    ) {
                        return false;
                    }
                }
            }
            (
                Op::CausalSoftmax { input, output },
                SegmentOpProof::Softmax {
                    rows: row_proofs,
                    lookup,
                    over_lookup,
                    requant,
                },
            ) => {
                let (Some(ci), Some(co)) = (
                    bundle.segment.commitment.cell(input),
                    bundle.segment.commitment.cell(output),
                ) else {
                    return false;
                };
                let (rows, cols) = {
                    let m = &bundle.segment.commitment.cells[co];
                    (m.logical_rows, m.logical_cols)
                };
                let h = dims.heads;
                if h == 0 || cols % h != 0 {
                    return false;
                }
                let n = cols / h;
                let units = rows * h;
                let total = rows * cols;
                if row_proofs.len() != units {
                    return false;
                }
                // The op-wide requant, claimed before the units (same order on both sides).
                if !crate::gkr::requant::verify_requant(
                    &bundle.segment.commitment,
                    &claims,
                    requant,
                    (total, crate::model::spec::RECIP_Q as usize, digits),
                    &mut tr,
                ) {
                    return false;
                }
                for (unit, rp) in row_proofs.iter().enumerate() {
                    let c = &rp.cells;
                    if c.scores != ci || c.probs != co {
                        return false;
                    }
                    if !crate::gkr::nonlinear::verify_softmax_row(
                        &bundle.segment.commitment,
                        &claims,
                        rp,
                        (unit, n, unit / h),
                        (bounds.exp_lo, rp.row_max.tstar),
                        rec_q,
                        &mut tr,
                    ) {
                        return false;
                    }
                }
                if !crate::gkr::logup::verify_lookup2(
                    &bundle.segment.commitment,
                    &claims,
                    lookup,
                    &bounds.exp_table(alpha, beta),
                    (total, bounds.exp_big_n()),
                    &mut tr,
                ) {
                    return false;
                }
                if !crate::gkr::logup::verify_lookup2(
                    &bundle.segment.commitment,
                    &claims,
                    over_lookup,
                    &crate::gkr::nonlinear::max_table(bounds.exp_lo, alpha, beta),
                    (total, crate::gkr::nonlinear::max_big_n(bounds.exp_lo)),
                    &mut tr,
                ) {
                    return false;
                }
            }
            (
                Op::TopK {
                    input, choice, k, ..
                },
                SegmentOpProof::TopK {
                    cells,
                    rows: row_proofs,
                    lookup,
                },
            ) => {
                let (Some(cs), Some(cc)) = (
                    bundle.segment.commitment.cell(input),
                    bundle.segment.commitment.cell(choice),
                ) else {
                    return false;
                };
                // The proof carries resolved ids, so every one of them must be the
                // cell the plan names — otherwise a prover could point an opening at
                // an arbitrary table and the point claim would silently find nothing.
                let name = |suffix: &str| format!("__tk.{i}.{suffix}");
                let (Some(plan_adj), Some(plan_delta)) = (
                    bundle.segment.commitment.cell(&name("adj")),
                    bundle.segment.commitment.cell(&name("delta")),
                ) else {
                    return false;
                };
                let mut plan_lookup = Vec::with_capacity(5);
                for k2 in ["lk.fp", "lk.dq", "lk.ft", "lk.dt", "lk.m"] {
                    match bundle.segment.commitment.cell(&name(k2)) {
                        Some(id) => plan_lookup.push(id),
                        None => return false,
                    }
                }
                if cells.scores != cs
                    || cells.choice != cc
                    || cells.adj != plan_adj
                    || cells.delta != plan_delta
                    || cells.lookup.input != cells.delta
                    || cells.lookup.output != cc
                    || [
                        cells.lookup.fp,
                        cells.lookup.diff_q,
                        cells.lookup.ft,
                        cells.lookup.diff_t,
                        cells.lookup.m,
                    ] != [
                        plan_lookup[0],
                        plan_lookup[1],
                        plan_lookup[2],
                        plan_lookup[3],
                        plan_lookup[4],
                    ]
                {
                    return false;
                }
                let (rows, e) = {
                    let m = &bundle.segment.commitment.cells[cs];
                    (m.logical_rows, m.logical_cols)
                };
                if row_proofs.len() != rows {
                    return false;
                }
                for (row, rp) in row_proofs.iter().enumerate() {
                    if !crate::gkr::selection::verify_top_k_row(
                        &bundle.segment.commitment,
                        &claims,
                        cells,
                        rp,
                        (row, e, *k),
                        &mut tr,
                    ) {
                        return false;
                    }
                }
                let table = bounds.sign_table(alpha, beta);
                if !crate::gkr::logup::verify_lookup2(
                    &bundle.segment.commitment,
                    &claims,
                    lookup,
                    &table,
                    (rows * e, bounds.sign_big_n()),
                    &mut tr,
                ) {
                    return false;
                }
            }
            (Op::Rope { input, theta, .. }, SegmentOpProof::Rope(rp)) => {
                let Some(ci) = bundle.segment.commitment.cell(input) else {
                    return false;
                };
                let (rows, d) = {
                    let m = &bundle.segment.commitment.cells[ci];
                    rope_dims((m.logical_rows, m.logical_cols), dims).unwrap_or((0, 0))
                };
                if rp.inp != ci {
                    return false;
                }
                if !crate::gkr::nonlinear::verify_rope(
                    &bundle.segment.commitment,
                    &claims,
                    rp,
                    (rows, dims.heads, d),
                    *theta,
                    (crate::gkr::nonlinear::ROPE_Q, digits),
                    &mut tr,
                ) {
                    return false;
                }
            }
            (Op::Embed { weight, output }, SegmentOpProof::Embed(p)) => {
                let Some(co) = bundle.segment.commitment.cell(output) else {
                    return false;
                };
                if p.out != co {
                    return false;
                }
                let Some((w_com, w_proof)) = wmap.get(weight.as_str()) else {
                    return false;
                };
                if !check_preshared(weight, w_com, published) {
                    return false;
                }
                let (l, hidden) = {
                    let m = &bundle.segment.commitment.cells[co];
                    (m.logical_rows, m.logical_cols)
                };
                let Some(w) = weights.iter().find(|w| w.name == *weight) else {
                    return false;
                };
                if !crate::gkr::nonlinear::verify_embed(
                    &bundle.segment.commitment,
                    &claims,
                    w_com,
                    w_proof,
                    p,
                    window,
                    (l, hidden, w.rows),
                    &mut tr,
                ) {
                    return false;
                }
            }
            _ => return false,
        }
    }

    // The head tail.
    if let Some(h) = head {
        let Some(hp) = bundle.segment.head.as_ref() else {
            return false;
        };
        let Some(logits) = bundle.segment.commitment.cell(&h.logits) else {
            return false;
        };
        let (rows, n) = {
            let m = &bundle.segment.commitment.cells[logits];
            (m.logical_rows, m.logical_cols)
        };
        let table = bounds.head_table(h.max, alpha, beta);
        if !crate::gkr::head::verify_head(
            &bundle.segment.commitment,
            &claims,
            hp,
            (logits, rows - 1, n, h.token),
            (&table, bounds.score_big_n()),
            &mut tr,
        ) {
            return false;
        }
    }

    // The op claims are only sound if the segment's own openings check out.
    crate::gkr::segment::profile::report("op claims");
    let t_pcs = std::time::Instant::now();
    let ok = crate::gkr::segment::verify(
        &bundle.segment.commitment,
        &claims,
        &bundle.segment.proof,
        &mut new_challenger(),
    );
    if std::env::var("FLASHPROVE_PROFILE").is_ok_and(|v| v == "1") {
        eprintln!("  profile pcs: {:.2}s", t_pcs.elapsed().as_secs_f64());
    }
    ok
}

/// One proved model: a bundle per trace segment, plus the public inputs the
/// verifier needs but cannot derive (window and lookup bounds).
#[derive(Clone, Serialize, Deserialize)]
pub struct ProvedModel {
    pub segments: Vec<SegmentBundle>,
    pub window: Vec<usize>,
    pub bounds: LookupBounds,
    pub head_token: usize,
    pub head_max: i64,
    pub head_segment: usize,
}

/// Assign each op to the segment that owns its **output** cell.
///
/// Both sides can run this: the prover from `trace.segments[s].cells`, the verifier
/// from `bundle.segments[s].commitment.cells`. An op whose output belongs to no
/// segment is an error rather than a silently dropped obligation.
pub fn ops_per_segment(
    ops: &[Op],
    segment_cells: &[Vec<String>],
) -> Result<Vec<Vec<Op>>, DriverError> {
    let mut out = vec![Vec::new(); segment_cells.len()];
    for op in ops {
        let target = match op {
            Op::Add { output, .. }
            | Op::Mul { output, .. }
            | Op::Reshape { output, .. }
            | Op::Matmul { output, .. }
            | Op::Scores { output, .. }
            | Op::CausalSoftmax { output, .. }
            | Op::WeightedV { output, .. }
            | Op::Silu { output, .. }
            | Op::WeightedSum { output, .. }
            | Op::Norm { output, .. }
            | Op::Rope { output, .. }
            | Op::Embed { output, .. } => output,
            // `top_k` writes both the output and the choice; the output is the cell
            // whose segment owns the decision.
            Op::TopK { output, .. } => output,
        };
        let Some(s) = segment_cells
            .iter()
            .position(|cs| cs.iter().any(|c| c == target))
        else {
            return Err(DriverError::UnknownCell(target.clone()));
        };
        out[s].push(op.clone());
    }
    Ok(out)
}

/// Verify every cross-segment link of a proven model.
///
/// A link `(from, to, name)` says the cell `name` committed in segment `to` is a copy
/// of the one committed in segment `from`. Both are opened at the public point
/// [`link_point`], and the two opened values must agree; a prover that fabricates the
/// copy is caught with probability `1 − 1/|F|` per link.
///
/// The schedule is **derived**, not taken from the proof: segments come from
/// [`plan_segments`] over the spec, so a proof cannot drop or add a link.
pub fn verify_links(
    file: &ProvedModel,
    spec: &crate::model::spec::ModelSpec,
    max_cells: usize,
) -> bool {
    let plan = plan_segments(&spec.ops, &spec.cells, max_cells);
    if plan.len() != file.segments.len() {
        return false;
    }
    let Ok(per_segment) = ops_per_segment(&spec.ops, &plan) else {
        return false;
    };
    // Index each segment's claims once; a link opens one cell in each.
    let indexes: Vec<crate::gkr::segment::Claims<'_>> = file
        .segments
        .iter()
        .map(|b| crate::gkr::segment::Claims::new(&b.segment.claims))
        .collect();
    for link in link_schedule(&plan, &per_segment) {
        let (Some(a), Some(b)) = (file.segments.get(link.from), file.segments.get(link.to)) else {
            return false;
        };
        let (Some(ca), Some(cb)) = (
            a.segment.commitment.cell(&link.name),
            b.segment.commitment.cell(&link.name),
        ) else {
            return false;
        };
        let m = b.segment.commitment.cells[cb].m();
        if a.segment.commitment.cells[ca].m() != m {
            return false;
        }
        let point = link_point(link.to, &link.name, m);
        let va = crate::gkr::segment::claim_value_at_id(
            &a.segment.commitment,
            &indexes[link.from],
            ca,
            &point,
        );
        let vb = crate::gkr::segment::claim_value_at_id(
            &b.segment.commitment,
            &indexes[link.to],
            cb,
            &point,
        );
        // `segment_cells` is only used for its length above; the plan is authoritative.
        if va.is_none() || va != vb {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(name: &str, vals: &[i64]) -> Cell {
        Cell::new(name, vals.len(), 1, vals.to_vec())
    }

    fn spec() -> (Vec<Cell>, Vec<Op>) {
        let a = vec![1i64, 2, 3, 4];
        let b = vec![2i64, 3, 4, 5];
        let c = vec![3i64, 4, 5, 6];
        let out: Vec<i64> = (0..4).map(|i| a[i] + b[i] + c[i]).collect();
        let prod: Vec<i64> = (0..4).map(|i| a[i] * b[i]).collect();
        let cells = vec![
            cell("a", &a),
            cell("b", &b),
            cell("c", &c),
            cell("out", &out),
            cell("prod", &prod),
            cell("sh", &prod),
        ];
        let ops = vec![
            Op::Add {
                inputs: vec!["a".into(), "b".into(), "c".into()],
                output: "out".into(),
                shift: None,
            },
            Op::Mul {
                a: "a".into(),
                b: "b".into(),
                output: "prod".into(),
                shift: None,
            },
            Op::Reshape {
                input: "prod".into(),
                output: "sh".into(),
                shape: vec![2, 2],
            },
        ];
        (cells, ops)
    }

    #[test]
    fn one_segment_three_ops() {
        let (cells, ops) = spec();
        let out = prove_segment(&ops, &cells, Params::default()).expect("drivable");
        assert!(verify_segment(&ops, &out));

        // A wrong sum is rejected by the add claim.
        let mut bad_cells = cells.clone();
        bad_cells[3].values[1] += 1;
        let bad = prove_segment(&ops, &bad_cells, Params::default()).unwrap();
        assert!(!verify_segment(&ops, &bad));

        // An op list that does not match the proof is rejected.
        let fewer = vec![ops[0].clone()];
        assert!(!verify_segment(&fewer, &out));

        // Unported ops are reported, not skipped.
        let unsupported = vec![Op::Norm {
            input: "a".into(),
            weight: "w".into(),
            output: "out".into(),
            eps: 1e-6,
        }];
        assert!(matches!(
            prove_segment(&unsupported, &cells, Params::default()),
            Err(DriverError::Unsupported("norm"))
        ));

        // A dangling cell name is reported.
        let dangling = vec![Op::Mul {
            a: "a".into(),
            b: "nope".into(),
            output: "prod".into(),
            shift: None,
        }];
        assert!(matches!(
            prove_segment(&dangling, &cells, Params::default()),
            Err(DriverError::UnknownCell(_))
        ));
    }

    #[test]
    fn shapes_flow_op_to_op() {
        let wd = vec![("w".to_string(), 3usize, 4usize)];
        let mut sh = Shapes::new(&wd);
        // Root cells are seeded by the caller.
        sh.seed("x", vec![2, 4]);
        assert_eq!(sh.rows_cols("x"), Some((2, 4)));

        // A matmul takes its new last dimension from the weight's rows.
        sh.apply(&Op::Matmul {
            input: "x".into(),
            weight: "w".into(),
            output: "y".into(),
            shift: 0,
        })
        .unwrap();
        assert_eq!(sh.of("y"), Some(&[2, 3][..]));

        // Same-shape ops copy their input's shape.
        for op in [
            Op::Norm {
                input: "y".into(),
                weight: "w".into(),
                output: "n".into(),
                eps: 1e-6,
            },
            Op::Rope {
                input: "n".into(),
                output: "r".into(),
                theta: 1.0,
            },
            Op::Silu {
                input: "r".into(),
                output: "s".into(),
            },
        ] {
            sh.apply(&op).unwrap();
        }
        assert_eq!(sh.of("s"), Some(&[2, 3][..]));

        // Reshape resolves a single `-1`.
        sh.apply(&Op::Reshape {
            input: "s".into(),
            output: "t".into(),
            shape: vec![-1, 3],
        })
        .unwrap();
        assert_eq!(sh.of("t"), Some(&[2, 3][..]));

        // Top-k writes both the output and the choice.
        sh.apply(&Op::TopK {
            input: "y".into(),
            output: "y".into(),
            choice: "c".into(),
            k: 1,
        })
        .unwrap();
        assert_eq!(sh.of("c"), Some(&[2, 3][..]));

        // An unseeded input and an unsupported op are both reported.
        assert!(matches!(
            sh.apply(&Op::Mul {
                a: "nope".into(),
                b: "s".into(),
                output: "m".into(),
                shift: None,
            }),
            Err(DriverError::UnknownCell(_))
        ));
        assert!(matches!(
            sh.apply(&Op::Embed {
                weight: "w".into(),
                output: "e".into(),
            }),
            Err(DriverError::Unsupported("embed"))
        ));
    }

    #[test]
    fn weights_commit_lazily_and_match_the_published_root() {
        use crate::pcs::whir_stacked::Committer;

        let w1 = WeightMatrix::new("w1", 4, 1, vec![1, 2, 3, 4]);
        let w2 = WeightMatrix::new("w2", 2, 1, vec![-5, 7]);
        let params = Params::default();

        let mut set = WeightSet::new(&[w1.clone(), w2.clone()], params);
        assert!(!set.is_open("w1"));
        let _ = set.get_or_commit("w1").unwrap();
        assert!(set.is_open("w1") && !set.is_open("w2"));
        let root_first = {
            let p = set.get_or_commit("w1").unwrap();
            p.commitment().root.clone()
        };
        let root_again = {
            let p = set.get_or_commit("w1").unwrap();
            p.commitment().root.clone()
        };
        assert_eq!(root_first, root_again, "committing twice is stable");
        set.release("w1");
        assert!(!set.is_open("w1"));
        // Re-commit for the proof pass (lazy again).
        let _ = set.get_or_commit("w1").unwrap();

        assert!(matches!(
            set.get_or_commit("nope"),
            Err(DriverError::UnknownCell(_))
        ));

        // An independent commit of the same tensor must produce the same root.
        let mut c = Committer::new();
        c.add_vec(
            &w1.values
                .iter()
                .map(|&v| signed(v as i64))
                .collect::<Vec<_>>(),
        );
        let pub_seg = c.seal(params);
        let pub_com = pub_seg.commitment().clone();

        // Nothing was claimed on the tensor, so it has nothing to prove.
        assert!(set.prove_all().is_empty(), "unclaimed tensors get no proof");

        let published: std::collections::HashMap<_, _> =
            [("w1".to_string(), pub_com.clone())].into_iter().collect();
        assert!(check_preshared("w1", &pub_com, &published));
        assert!(!check_preshared("missing", &pub_com, &published));

        // A different tensor's root is rejected.
        let mut other = Committer::new();
        other.add_vec(&[Fp::new(9), Fp::new(9), Fp::new(9), Fp::new(9)]);
        let other_seg = other.seal(params);
        let other_root = other_seg.commitment();
        assert!(!check_preshared("w1", &other_root, &published));
    }

    #[test]
    fn matmul_reads_a_preshared_weight() {
        use crate::model::spec::requant as requant_i64;
        use crate::pcs::whir_stacked::Committer;
        let dims = crate::model::spec::Dims {
            hidden: 2,
            vocab: 2,
            layers: 1,
            heads: 1,
            head_dim: 2,
            experts: 2,
            active: 1,
            intermediate: 2,
        };

        let alpha = Fp::new(11);
        let beta = Fp::new(5);
        let bounds = LookupBounds {
            silu_lo: -1,
            silu_hi: 1,
            sign_lo: -1,
            sign_hi: 1,
            exp_lo: -1,
            score_lo: 0,
            score_hi: 1,
            rope_lo: -1,
            rope_hi: 1,
        };

        let k = 1usize;
        let w = WeightMatrix::new("W", 2, 2, vec![1, 2, 3, 4]);
        let x = vec![2i64, 3];
        let y = vec![1 * 2 + 2 * 3, 3 * 2 + 4 * 3];
        let out: Vec<i64> = y.iter().map(|&v| requant_i64(v, k as u32)).collect();

        let cells = vec![
            Cell::new("x", 2, 1, x.clone()),
            Cell::new("out", 2, 1, out.clone()),
        ];
        let ops = vec![Op::Matmul {
            input: "x".into(),
            weight: "W".into(),
            output: "out".into(),
            shift: k as u32,
        }];

        let bundle = prove_segment_with_weights(
            &ops,
            &cells,
            &[w.clone()],
            &dims,
            (1, alpha, beta, 8, 0.0),
            &bounds,
            Params::default(),
            &[],
        )
        .expect("drivable");

        // The published map is what `cp_commit` would hand over.
        let mut c = Committer::new();
        c.add_vec(
            &w.values
                .iter()
                .map(|&v| signed(v as i64))
                .collect::<Vec<_>>(),
        );
        let pub_seg = c.seal(Params::default());
        let published: std::collections::HashMap<String, whir_stacked::SegmentCommitment> =
            [("W".to_string(), pub_seg.commitment().clone())]
                .into_iter()
                .collect();

        assert!(verify_segment_with_weights(
            &ops,
            &bundle,
            &published,
            &[],
            &dims,
            (1, alpha, beta, 8, 0.0),
            &bounds,
            &[]
        ));

        // A published root that is not the tensor's is rejected.
        let wrong: std::collections::HashMap<String, whir_stacked::SegmentCommitment> =
            [("W".to_string(), bundle.weights[0].1.clone())]
                .into_iter()
                .collect();
        let mut other = Committer::new();
        other.add_vec(&[Fp::new(7), Fp::new(7), Fp::new(7), Fp::new(7)]);
        let other_seg = other.seal(Params::default());
        let mut wrong2 = wrong.clone();
        wrong2.insert("W".to_string(), other_seg.commitment().clone());
        assert!(!verify_segment_with_weights(
            &ops,
            &bundle,
            &wrong2,
            &[],
            &dims,
            (1, alpha, beta, 8, 0.0),
            &bounds,
            &[]
        ));

        // A wrong output breaks the requant link to `y`.
        let bad_cells = vec![
            Cell::new("x", 2, 1, x),
            Cell::new("out", 2, 1, vec![out[0] + 1, out[1]]),
        ];
        let bad = prove_segment_with_weights(
            &ops,
            &bad_cells,
            &[w],
            &dims,
            (1, alpha, beta, 8, 0.0),
            &bounds,
            Params::default(),
            &[],
        )
        .unwrap();
        assert!(!verify_segment_with_weights(
            &ops,
            &bad,
            &published,
            &[],
            &dims,
            (1, alpha, beta, 8, 0.0),
            &bounds,
            &[]
        ));
    }

    #[test]
    fn batched_matmul_proves_the_whole_sequence() {
        use crate::model::spec::requant as requant_i64;
        use crate::pcs::whir_stacked::Committer;

        let dims = crate::model::spec::Dims {
            hidden: 2,
            vocab: 2,
            layers: 1,
            heads: 1,
            head_dim: 2,
            experts: 2,
            active: 1,
            intermediate: 2,
        };
        let (alpha, beta) = (Fp::new(11), Fp::new(5));
        let bounds = LookupBounds {
            silu_lo: -1,
            silu_hi: 1,
            sign_lo: -1,
            sign_hi: 1,
            exp_lo: -1,
            score_lo: 0,
            score_hi: 1,
            rope_lo: -1,
            rope_hi: 1,
        };

        // Two tokens (L = 2), hidden = 2, vocab = 2: every arity is a power of two.
        let k = 1usize;
        let w = WeightMatrix::new("W", 2, 2, vec![1, 2, 3, 4]);
        let x = vec![2i64, 4, 6, 8]; // rows [2,4], [6,8]
        let y: Vec<i64> = (0..2)
            .flat_map(|t| {
                let row = &x[t * 2..(t + 1) * 2];
                (0..2)
                    .map(|i| {
                        (0..2)
                            .map(|j| w.values[i * 2 + j] as i64 * row[j])
                            .sum::<i64>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let out: Vec<i64> = y.iter().map(|&v| requant_i64(v, k as u32)).collect();

        let cells = vec![
            Cell::new("x", 2, 2, x.clone()),
            Cell::new("out", 2, 2, out.clone()),
        ];
        let ops = vec![Op::Matmul {
            input: "x".into(),
            weight: "W".into(),
            output: "out".into(),
            shift: k as u32,
        }];

        let bundle = prove_segment_with_weights(
            &ops,
            &cells,
            &[w.clone()],
            &dims,
            (1, alpha, beta, 8, 0.0),
            &bounds,
            Params::default(),
            &[],
        )
        .expect("drivable");
        // The claim really is batched: 2 output bits, 1 contraction bit, 1 batch bit.
        let SegmentOpProof::Matmul { mm, .. } = &bundle.segment.ops[0] else {
            panic!("matmul proof");
        };
        assert_eq!((mm.ln, mm.lm, mm.lp), (1, 1, 1));

        let mut c = Committer::new();
        c.add_vec(
            &w.values
                .iter()
                .map(|&v| signed(v as i64))
                .collect::<Vec<_>>(),
        );
        let pub_seg = c.seal(Params::default());
        let published: std::collections::HashMap<String, whir_stacked::SegmentCommitment> =
            [("W".to_string(), pub_seg.commitment().clone())]
                .into_iter()
                .collect();
        assert!(verify_segment_with_weights(
            &ops,
            &bundle,
            &published,
            &[],
            &dims,
            (1, alpha, beta, 8, 0.0),
            &bounds,
            &[]
        ));

        // Tampering with one token's output is caught.
        let mut bad = vec![0i64; 4];
        bad.copy_from_slice(&[out[0], out[1], out[2], out[3]]);
        bad[3] += 1;
        let bad_bundle = prove_segment_with_weights(
            &ops,
            &[Cell::new("x", 2, 2, x), Cell::new("out", 2, 2, bad)],
            &[w],
            &dims,
            (1, alpha, beta, 8, 0.0),
            &bounds,
            Params::default(),
            &[],
        )
        .unwrap();
        assert!(!verify_segment_with_weights(
            &ops,
            &bad_bundle,
            &published,
            &[],
            &dims,
            (1, alpha, beta, 8, 0.0),
            &bounds,
            &[]
        ));
    }

    #[test]
    fn segmented_model_links_every_cross_segment_input() {
        use crate::model::spec::{Dims, ModelSpec};

        let dims = Dims {
            hidden: 4,
            vocab: 4,
            layers: 1,
            heads: 1,
            head_dim: 4,
            experts: 4,
            active: 1,
            intermediate: 4,
        };
        let x = vec![1i64, 2, 3, 4];
        // Two chained ops: `p = x·x`, `q = p + p`.
        let p: Vec<i64> = x.iter().map(|&v| v * v).collect();
        let q: Vec<i64> = p.iter().map(|&v| 2 * v).collect();
        let cells = vec![
            Cell::new("x", 4, 1, x),
            Cell::new("p", 4, 1, p),
            Cell::new("q", 4, 1, q),
        ];
        let ops = vec![
            Op::Mul {
                a: "x".into(),
                b: "x".into(),
                output: "p".into(),
                shift: None,
            },
            Op::Add {
                inputs: vec!["p".into(), "p".into()],
                output: "q".into(),
                shift: None,
            },
        ];
        let spec = ModelSpec {
            dims: dims.clone(),
            weights: Vec::new(),
            ops: ops.clone(),
            cells: vec!["x".into(), "p".into(), "q".into()],
        };
        let bounds = LookupBounds {
            silu_lo: -1,
            silu_hi: 1,
            sign_lo: -1,
            sign_hi: 1,
            exp_lo: -1,
            score_lo: 0,
            score_hi: 1,
            rope_lo: -1,
            rope_hi: 1,
        };

        // One cell per segment, so `q`'s read of `p` must cross a boundary.
        let plan = plan_segments(&spec.ops, &spec.cells, 1);
        assert_eq!(plan.len(), 2, "two segments");
        let mut first = plan[0].clone();
        first.sort();
        assert_eq!(first, vec!["p".to_string(), "x".to_string()]);
        assert_eq!(plan[1], vec!["q".to_string()]);
        let per_segment = ops_per_segment(&spec.ops, &plan).unwrap();
        let links = link_schedule(&plan, &per_segment);
        assert_eq!(links.len(), 1, "q reads p across the boundary");
        assert_eq!(
            (links[0].from, links[0].to, links[0].name.as_str()),
            (0, 1, "p")
        );

        // Prove each segment, copying the external input and opening it at the link point.
        let arity = |_name: &str| 2usize; // every cell here is 4 entries -> 2^2
        let openings = link_openings(&links, plan.len(), arity);
        let mut segments = Vec::new();
        for s in 0..plan.len() {
            let mut seg_cells: Vec<Cell> = plan[s]
                .iter()
                .map(|n| cells.iter().find(|c| &c.name == n).unwrap().clone())
                .collect();
            for name in external_inputs(&plan[s], &per_segment[s]) {
                seg_cells.push(cells.iter().find(|c| &c.name == &name).unwrap().clone());
            }
            let extra: Vec<(String, Vec<Ext>)> = openings[s].clone();
            segments.push(
                prove_segment_full(
                    &per_segment[s],
                    &seg_cells,
                    &[],
                    &dims,
                    (1, Fp::new(11), Fp::new(5), 8, 0.0),
                    &bounds,
                    Params::default(),
                    &[],
                    None,
                    &extra,
                )
                .unwrap(),
            );
        }
        let file = ProvedModel {
            segments,
            window: Vec::new(),
            bounds,
            head_token: 0,
            head_max: 0,
            head_segment: 1,
        };
        assert!(verify_links(&file, &spec, 1), "honest links verify");

        // A copy that disagrees with the cell it claims to copy is rejected.
        let mut bad = file.clone();
        let mut forged = bad.segments[1].segment.commitment.clone();
        forged.cells[0].name = "q".into();
        let mut bundle = bad.segments[1].clone();
        bundle.segment.commitment.cells[1].name = "q".into();
        bad.segments[1] = bundle;
        let _ = forged;
        // Simpler forgery: rename the copied cell so the link cannot find it.
        bad.segments[1].segment.commitment.cells[1].name = "p2".into();
        assert!(!verify_links(&bad, &spec, 1));
    }

    #[test]
    fn lookup_bounds_build_their_tables() {
        let b = LookupBounds {
            silu_lo: -2,
            silu_hi: 1,
            sign_lo: -1,
            sign_hi: 2,
            exp_lo: -3,
            score_lo: 0,
            score_hi: 5,
            rope_lo: -1,
            rope_hi: 1,
        };
        let alpha = Fp::new(11);
        let beta = Fp::new(5);

        // Tables are padded to a power of two and zero-free.
        let silu = b.silu_table(alpha, beta);
        assert_eq!(silu.len(), b.silu_big_n().next_power_of_two());
        assert_eq!(b.silu_big_n(), 4);
        let sign = b.sign_table(alpha, beta);
        assert_eq!(sign.len(), b.sign_big_n().next_power_of_two());
        assert_eq!(b.sign_big_n(), 4);
        let exp = b.exp_table(alpha, beta);
        assert_eq!(exp.len(), b.exp_big_n().next_power_of_two());
        assert_eq!(b.exp_big_n(), 4);

        // The head table depends on the maximum and has the logits' extent.
        let head = b.head_table(5, alpha, beta);
        assert_eq!(head.len(), b.score_big_n().next_power_of_two());
        assert_ne!(head, b.head_table(4, alpha, beta));

        // Bound derivation agrees with the bounds struct it wraps.
        let spec = crate::model::spec::ModelSpec {
            dims: crate::model::spec::Dims {
                hidden: 4,
                vocab: 8,
                layers: 1,
                heads: 2,
                head_dim: 2,
                experts: 2,
                active: 1,
                intermediate: 4,
            },
            weights: Vec::new(),
            ops: Vec::new(),
            cells: Vec::new(),
        };
        let trace = crate::model::Trace {
            segments: Vec::new(),
        };
        let d = LookupBounds::derive(&spec, &trace);
        assert!(d.silu_hi >= d.silu_lo, "fallback bounds are non-empty");
    }

    #[test]
    fn silu_driven_against_the_public_table() {
        use crate::model::spec::silu as silu_i64;
        let dims = crate::model::spec::Dims {
            hidden: 2,
            vocab: 2,
            layers: 1,
            heads: 1,
            head_dim: 2,
            experts: 2,
            active: 1,
            intermediate: 2,
        };

        let alpha = Fp::new(11);
        let beta = Fp::new(5);
        let x = vec![-2i64, -1, 0, 3];
        let out: Vec<i64> = x.iter().map(|&v| silu_i64(v)).collect();
        let bounds = LookupBounds {
            silu_lo: -2,
            silu_hi: 3,
            sign_lo: -1,
            sign_hi: 1,
            exp_lo: -1,
            score_lo: 0,
            score_hi: 1,
            rope_lo: -1,
            rope_hi: 1,
        };
        let cells = vec![
            Cell::new("x", 4, 1, x.clone()),
            Cell::new("out", 4, 1, out.clone()),
        ];
        let ops = vec![Op::Silu {
            input: "x".into(),
            output: "out".into(),
        }];

        let bundle = prove_segment_with_weights(
            &ops,
            &cells,
            &[],
            &dims,
            (1, alpha, beta, 8, 0.0),
            &bounds,
            Params::default(),
            &[],
        )
        .expect("drivable");
        let published = std::collections::HashMap::new();
        assert!(verify_segment_with_weights(
            &ops,
            &bundle,
            &published,
            &[],
            &dims,
            (1, alpha, beta, 8, 0.0),
            &bounds,
            &[],
        ));

        // A pair that is not on the silu graph is rejected.
        let bad_cells = vec![
            Cell::new("x", 4, 1, x),
            Cell::new("out", 4, 1, vec![out[0] + 1, out[1], out[2], out[3]]),
        ];
        let bad = prove_segment_with_weights(
            &ops,
            &bad_cells,
            &[],
            &dims,
            (1, alpha, beta, 8, 0.0),
            &bounds,
            Params::default(),
            &[],
        )
        .unwrap();
        assert!(!verify_segment_with_weights(
            &ops,
            &bad,
            &published,
            &[],
            &dims,
            (1, alpha, beta, 8, 0.0),
            &bounds,
            &[],
        ));

        // Bounds that do not cover the table the prover used are rejected.
        let mut narrow = bounds;
        narrow.silu_hi = 0;
        assert!(!verify_segment_with_weights(
            &ops,
            &bundle,
            &published,
            &[],
            &dims,
            (1, alpha, beta, 8, 0.0),
            &narrow,
            &[]
        ));
    }

    #[test]
    fn weighted_sum_driven_over_positions() {
        let dims = crate::model::spec::Dims {
            hidden: 2,
            vocab: 2,
            layers: 1,
            heads: 1,
            head_dim: 2,
            experts: 2,
            active: 1,
            intermediate: 2,
        };
        let alpha = Fp::new(11);
        let beta = Fp::new(5);
        let bounds = LookupBounds {
            silu_lo: -1,
            silu_hi: 1,
            sign_lo: -1,
            sign_hi: 1,
            exp_lo: -1,
            score_lo: 0,
            score_hi: 1,
            rope_lo: -1,
            rope_hi: 1,
        };
        let t0 = vec![1i64, 2, 3, 4];
        let t1 = vec![5i64, 6, 7, 8];
        let choice = vec![2i64, 3, 1, 0];
        let out: Vec<i64> = (0..4)
            .map(|i| {
                let row = i / 2;
                choice[row * 2] * t0[i] + choice[row * 2 + 1] * t1[i]
            })
            .collect();
        let cells = vec![
            Cell::new("t0", 2, 2, t0.clone()),
            Cell::new("t1", 2, 2, t1.clone()),
            Cell::new("choice", 2, 2, choice.clone()),
            Cell::new("out", 2, 2, out.clone()),
        ];
        let ops = vec![Op::WeightedSum {
            choice: "choice".into(),
            terms: vec!["t0".into(), "t1".into()],
            output: "out".into(),
        }];
        let bundle = prove_segment_with_weights(
            &ops,
            &cells,
            &[],
            &dims,
            (1, alpha, beta, 8, 0.0),
            &bounds,
            Params::default(),
            &[],
        )
        .expect("drivable");
        let co = bundle.segment.commitment.cell("out").unwrap();
        assert_eq!(bundle.segment.commitment.cells[co].logical_rows, 2);
        assert_eq!(bundle.segment.commitment.cells[co].logical_cols, 2);
        let published = std::collections::HashMap::new();
        assert!(verify_segment_with_weights(
            &ops,
            &bundle,
            &published,
            &[],
            &dims,
            (1, alpha, beta, 8, 0.0),
            &bounds,
            &[],
        ));
    }

    #[test]
    fn norm_driven_row_by_row() {
        let alpha = Fp::new(11);
        let beta = Fp::new(5);
        let bounds = LookupBounds {
            silu_lo: -1,
            silu_hi: 1,
            sign_lo: -1,
            sign_hi: 1,
            exp_lo: -1,
            score_lo: 0,
            score_hi: 1,
            rope_lo: -1,
            rope_hi: 1,
        };
        let dims = crate::model::spec::Dims {
            hidden: 2,
            vocab: 2,
            layers: 1,
            heads: 1,
            head_dim: 2,
            experts: 2,
            active: 1,
            intermediate: 2,
        };
        // `rec_q` must be the same fixed-point scale the ops round with.
        let (rec_q, eps, digits) = (crate::model::spec::RECIP_Q, 0.0f64, 4usize);
        let gamma = WeightMatrix::new("g", 2, 1, vec![1, 2]);
        let x = vec![1i64, 2, 3, 4];
        let mut out = Vec::new();
        for row in 0..2usize {
            let xr = &x[row * 2..(row + 1) * 2];
            let sum_sq: i64 = xr.iter().map(|&v| v * v).sum();
            let inv_q = crate::gkr::nonlinear::norm_inv_q(sum_sq, 2, rec_q, eps);
            for j in 0..2usize {
                let y = xr[j] * (gamma.values[j] as i64) * inv_q;
                out.push(crate::model::spec::requant(y, crate::model::spec::RECIP_Q));
            }
        }

        assert!(
            out.iter().any(|&v| v != 0),
            "the test needs non-zero norm outputs"
        );
        let cells = vec![
            Cell::new("x", 2, 2, x.clone()),
            Cell::new("out", 2, 2, out.clone()),
        ];
        let ops = vec![Op::Norm {
            input: "x".into(),
            weight: "g".into(),
            output: "out".into(),
            eps: 0.0,
        }];
        let weights = vec![gamma.clone()];
        let bundle = prove_segment_with_weights(
            &ops,
            &cells,
            &weights,
            &dims,
            (digits, alpha, beta, rec_q, eps),
            &bounds,
            Params::default(),
            &[],
        )
        .expect("drivable");
        let published = std::collections::HashMap::new();
        assert!(verify_segment_with_weights(
            &ops,
            &bundle,
            &published,
            &weights,
            &dims,
            (digits, alpha, beta, rec_q, eps),
            &bounds,
            &[],
        ));

        // A wrong output breaks the requant link to `y`.
        let mut bad_out = out.clone();
        bad_out[0] += 1;
        let bad_cells = vec![Cell::new("x", 2, 2, x), Cell::new("out", 2, 2, bad_out)];
        let bad = prove_segment_with_weights(
            &ops,
            &bad_cells,
            &weights,
            &dims,
            (digits, alpha, beta, rec_q, eps),
            &bounds,
            Params::default(),
            &[],
        )
        .unwrap();
        assert!(!verify_segment_with_weights(
            &ops,
            &bad,
            &published,
            &weights,
            &dims,
            (digits, alpha, beta, rec_q, eps),
            &bounds,
            &[],
        ));

        // A γ the verifier does not know is rejected: the copy is bound to it.
        let other = vec![WeightMatrix::new("g", 2, 1, vec![5, 7])];
        assert!(!verify_segment_with_weights(
            &ops,
            &bundle,
            &published,
            &other,
            &dims,
            (digits, alpha, beta, rec_q, eps),
            &bounds,
            &[],
        ));
    }

    fn base_bounds() -> LookupBounds {
        LookupBounds {
            silu_lo: -1,
            silu_hi: 1,
            sign_lo: -1,
            sign_hi: 1,
            exp_lo: -3,
            score_lo: 0,
            score_hi: 1,
            rope_lo: -1,
            rope_hi: 1,
        }
    }

    #[test]
    fn weighted_v_driven_per_head() {
        let alpha = Fp::new(11);
        let beta = Fp::new(5);
        let bounds = base_bounds();
        let (lq, h, d) = (2usize, 2usize, 2usize);
        let dims = crate::model::spec::Dims {
            hidden: 4,
            vocab: 2,
            layers: 1,
            heads: h,
            head_dim: d,
            experts: 2,
            active: 1,
            intermediate: 4,
        };
        let v = vec![32768i64; lq * h * d];
        let w = vec![3i64, -1, 5, -1, 7, -1, 9, -1];
        let mut out = vec![0i64; lq * h * d];
        for li in 0..lq {
            for hi in 0..h {
                for i in 0..d {
                    let acc: i64 = (0..lq)
                        .map(|t| v[(t * h + hi) * d + i] * w[((li * h) + hi) * lq + t])
                        .sum();
                    out[((li * h) + hi) * d + i] = crate::model::spec::requant(acc, 15);
                }
            }
        }
        assert_ne!(out, vec![0; lq * h * d]);

        let cells = vec![
            Cell::new("w", lq, h * lq, w.clone()),
            Cell::new("v", lq, h * d, v.clone()),
            Cell::new("out", lq, h * d, out.clone()),
        ];
        let ops = vec![Op::WeightedV {
            weights: "w".into(),
            v: "v".into(),
            output: "out".into(),
        }];
        let bundle = prove_segment_with_weights(
            &ops,
            &cells,
            &[],
            &dims,
            (3, alpha, beta, 8, 0.0),
            &bounds,
            Params::default(),
            &[],
        )
        .expect("drivable");
        let published = std::collections::HashMap::new();
        assert!(verify_segment_with_weights(
            &ops,
            &bundle,
            &published,
            &[],
            &dims,
            (3, alpha, beta, 8, 0.0),
            &bounds,
            &[],
        ));

        let mut bad = out.clone();
        bad[0] += 1;
        let bad_cells = vec![
            Cell::new("w", lq, h * lq, w),
            Cell::new("v", lq, h * d, v),
            Cell::new("out", lq, h * d, bad),
        ];
        let bad_bundle = prove_segment_with_weights(
            &ops,
            &bad_cells,
            &[],
            &dims,
            (3, alpha, beta, 8, 0.0),
            &bounds,
            Params::default(),
            &[],
        )
        .unwrap();
        assert!(!verify_segment_with_weights(
            &ops,
            &bad_bundle,
            &published,
            &[],
            &dims,
            (3, alpha, beta, 8, 0.0),
            &bounds,
            &[],
        ));
    }

    #[test]
    fn scores_driven_per_head() {
        let alpha = Fp::new(11);
        let beta = Fp::new(5);
        let bounds = base_bounds();
        let (lq, h, d) = (2usize, 2usize, 2usize);
        let dims = crate::model::spec::Dims {
            hidden: 4,
            vocab: 2,
            layers: 1,
            heads: h,
            head_dim: d,
            experts: 2,
            active: 1,
            intermediate: 4,
        };
        let q = vec![1i64; lq * h * d];
        let k = (0..(lq * h * d) as i64).collect::<Vec<_>>();
        let rq = crate::model::spec::RECIP_Q;
        let inv_q = ((1i64 << rq) as f64 / 1.0).round() as i64;
        let mut out = vec![0i64; lq * h * lq];
        for li in 0..lq {
            for hi in 0..h {
                for t in 0..lq {
                    let dot: i64 = (0..d)
                        .map(|j| q[((li * h) + hi) * d + j] * k[(t * h + hi) * d + j])
                        .sum();
                    out[((li * h) + hi) * lq + t] = crate::model::spec::requant(dot * inv_q, rq);
                }
            }
        }
        assert!(out.windows(2).any(|w| w[0] != w[1]));

        let cells = vec![
            Cell::new("q", lq, h * d, q.clone()),
            Cell::new("k", lq, h * d, k.clone()),
            Cell::new("out", lq, h * lq, out.clone()),
        ];
        let ops = vec![Op::Scores {
            q: "q".into(),
            k: "k".into(),
            output: "out".into(),
            scale: 1.0,
            shift: Some(rq),
        }];
        let bundle = prove_segment_with_weights(
            &ops,
            &cells,
            &[],
            &dims,
            (4, alpha, beta, 8, 0.0),
            &bounds,
            Params::default(),
            &[],
        )
        .expect("drivable");
        let published = std::collections::HashMap::new();
        assert!(verify_segment_with_weights(
            &ops,
            &bundle,
            &published,
            &[],
            &dims,
            (4, alpha, beta, 8, 0.0),
            &bounds,
            &[],
        ));

        let mut bad = out.clone();
        bad[0] += 1;
        let bad_cells = vec![
            Cell::new("q", lq, h * d, q),
            Cell::new("k", lq, h * d, k),
            Cell::new("out", lq, h * lq, bad),
        ];
        let bad_bundle = prove_segment_with_weights(
            &ops,
            &bad_cells,
            &[],
            &dims,
            (4, alpha, beta, 8, 0.0),
            &bounds,
            Params::default(),
            &[],
        )
        .unwrap();
        assert!(!verify_segment_with_weights(
            &ops,
            &bad_bundle,
            &published,
            &[],
            &dims,
            (4, alpha, beta, 8, 0.0),
            &bounds,
            &[],
        ));
    }

    #[test]
    fn causal_softmax_driven_row_by_row() {
        let alpha = Fp::new(11);
        let beta = Fp::new(5);
        let bounds = base_bounds();
        let dims = crate::model::spec::Dims {
            hidden: 2,
            vocab: 2,
            layers: 1,
            heads: 1,
            head_dim: 2,
            experts: 2,
            active: 1,
            intermediate: 2,
        };
        let (rec_q, eps, digits) = (crate::model::spec::RECIP_Q, 0.0f64, 6usize);
        let exp_lo = bounds.exp_lo;
        let scores = vec![3i64, 1, 2, 5];
        let (rows, n) = (2usize, 2usize);
        let mut probs = Vec::new();
        for row in 0..rows {
            let l = row.min(n - 1);
            let m = (0..=l).map(|i| scores[row * n + i]).max().unwrap();
            let d: Vec<i64> = (0..n)
                .map(|i| {
                    if i <= l {
                        scores[row * n + i] - m
                    } else {
                        exp_lo
                    }
                })
                .collect();
            let e: Vec<i64> = d.iter().map(|&v| crate::model::spec::exp_q15(v)).collect();
            let s: i64 = e.iter().take(l + 1).sum();
            let inv_q = crate::gkr::nonlinear::softmax_inv_q(s, rec_q);
            for i in 0..n {
                probs.push(crate::model::spec::requant(
                    e[i] * inv_q,
                    crate::model::spec::RECIP_Q,
                ));
            }
        }
        assert!(
            probs.iter().any(|&v| v != 0),
            "the test needs non-zero probabilities"
        );
        let cells = vec![
            Cell::new("scores", rows, n, scores.clone()),
            Cell::new("probs", rows, n, probs.clone()),
        ];
        let ops = vec![Op::CausalSoftmax {
            input: "scores".into(),
            output: "probs".into(),
        }];
        let bundle = prove_segment_with_weights(
            &ops,
            &cells,
            &[],
            &dims,
            (digits, alpha, beta, rec_q, eps),
            &bounds,
            Params::default(),
            &[],
        )
        .expect("drivable");
        let published = std::collections::HashMap::new();
        assert!(verify_segment_with_weights(
            &ops,
            &bundle,
            &published,
            &[],
            &dims,
            (digits, alpha, beta, rec_q, eps),
            &bounds,
            &[],
        ));

        let mut bad = probs.clone();
        bad[0] += 1;
        let bad_cells = vec![
            Cell::new("scores", rows, n, scores),
            Cell::new("probs", rows, n, bad),
        ];
        let bad_bundle = prove_segment_with_weights(
            &ops,
            &bad_cells,
            &[],
            &dims,
            (digits, alpha, beta, rec_q, eps),
            &bounds,
            Params::default(),
            &[],
        )
        .unwrap();
        assert!(!verify_segment_with_weights(
            &ops,
            &bad_bundle,
            &published,
            &[],
            &dims,
            (digits, alpha, beta, rec_q, eps),
            &bounds,
            &[],
        ));
    }

    #[test]
    fn top_k_driven_per_row() {
        let alpha = Fp::new(11);
        let beta = Fp::new(5);
        let mut bounds = base_bounds();
        bounds.sign_lo = -9;
        bounds.sign_hi = 0;
        let dims = crate::model::spec::Dims {
            hidden: 2,
            vocab: 2,
            layers: 1,
            heads: 1,
            head_dim: 2,
            experts: 2,
            active: 1,
            intermediate: 2,
        };
        let (rows, e, k) = (2usize, 2usize, 1usize);
        let scores = vec![5i64, 1, 0, 3];
        let adj: Vec<i64> = (0..rows * e)
            .map(|i| e as i64 * scores[i] - (i % e) as i64)
            .collect();
        let mut choice = vec![0i64; rows * e];
        for row in 0..rows {
            let mut sorted: Vec<i64> = adj[row * e..(row + 1) * e].to_vec();
            sorted.sort_unstable_by(|a, b| b.cmp(a));
            let tau = sorted[k - 1];
            for i in 0..e {
                if adj[row * e + i] >= tau {
                    choice[row * e + i] = 1;
                }
            }
        }
        let cells = vec![
            Cell::new("scores", rows, e, scores.clone()),
            Cell::new("choice", rows, e, choice.clone()),
        ];
        let ops = vec![Op::TopK {
            input: "scores".into(),
            output: "scores".into(),
            choice: "choice".into(),
            k,
        }];
        let bundle = prove_segment_with_weights(
            &ops,
            &cells,
            &[],
            &dims,
            (1, alpha, beta, 8, 0.0),
            &bounds,
            Params::default(),
            &[],
        )
        .expect("drivable");
        let published = std::collections::HashMap::new();
        assert!(verify_segment_with_weights(
            &ops,
            &bundle,
            &published,
            &[],
            &dims,
            (1, alpha, beta, 8, 0.0),
            &bounds,
            &[],
        ));
    }

    #[test]
    fn rope_driven_with_a_requant_lookup() {
        let alpha = Fp::new(11);
        let beta = Fp::new(5);
        let dims = crate::model::spec::Dims {
            hidden: 4,
            vocab: 2,
            layers: 1,
            heads: 2,
            head_dim: 2,
            experts: 2,
            active: 1,
            intermediate: 4,
        };
        // [pos=2, head=2, dim=2] flattened.
        let (rows, h, d) = (4usize, 2usize, 2usize);
        // The rotated-value table is `O(rot range)`, and the rotation scale is
        // 2^15, so a non-zero input would need a ~300k-entry table. This exercises
        // the wiring with the smallest table the format allows; rope on real data
        // needs `claim_requant` instead of a range lookup.
        let inp = vec![1i64, 0, 0, 0, 0, 0, 0, 0];
        let theta = 1.0f64;
        let rot = crate::gkr::nonlinear::rotate(&inp, rows, h, d, theta);
        let out: Vec<i64> = rot
            .iter()
            .map(|&v| crate::model::spec::requant(v, crate::gkr::nonlinear::ROPE_Q as u32))
            .collect();
        let bounds = base_bounds();

        let cells = vec![
            Cell::new("inp", rows, d, inp.clone()),
            Cell::new("out", rows, d, out.clone()),
        ];
        let ops = vec![Op::Rope {
            input: "inp".into(),
            output: "out".into(),
            theta,
        }];
        let bundle = prove_segment_with_weights(
            &ops,
            &cells,
            &[],
            &dims,
            (4, alpha, beta, 8, 0.0),
            &bounds,
            Params::default(),
            &[],
        )
        .expect("drivable");
        let published = std::collections::HashMap::new();
        assert!(verify_segment_with_weights(
            &ops,
            &bundle,
            &published,
            &[],
            &dims,
            (4, alpha, beta, 8, 0.0),
            &bounds,
            &[],
        ));
    }

    #[test]
    fn embed_driven_from_the_embedding_root() {
        let alpha = Fp::new(11);
        let beta = Fp::new(5);
        let bounds = base_bounds();
        let dims = crate::model::spec::Dims {
            hidden: 2,
            vocab: 8,
            layers: 1,
            heads: 1,
            head_dim: 2,
            experts: 2,
            active: 1,
            intermediate: 2,
        };
        let (vocab, hidden) = (4usize, 2usize);
        let emb_vals: Vec<i8> = (0..(vocab * hidden) as i8).map(|i| i - 3).collect();
        let emb = WeightMatrix::new("emb", vocab, hidden, emb_vals.clone());
        let window = vec![2usize, 0, 3];
        let l = window.len();
        let out: Vec<i64> = window
            .iter()
            .flat_map(|&t| {
                (0..hidden)
                    .map(|j| emb_vals[t * hidden + j] as i64)
                    .collect::<Vec<_>>()
            })
            .collect();
        let cells = vec![Cell::new("out", l, hidden, out.clone())];
        let ops = vec![Op::Embed {
            weight: "emb".into(),
            output: "out".into(),
        }];
        let weights = vec![emb.clone()];
        let mut c = crate::pcs::whir_stacked::Committer::new();
        c.add_vec(
            &emb_vals
                .iter()
                .map(|&v| crate::gkr::segment::signed(v as i64))
                .collect::<Vec<_>>(),
        );
        let pub_seg = c.seal(Params::default());
        let published: std::collections::HashMap<String, whir_stacked::SegmentCommitment> =
            [("emb".to_string(), pub_seg.commitment().clone())]
                .into_iter()
                .collect();
        let bundle = prove_segment_with_weights(
            &ops,
            &cells,
            &weights,
            &dims,
            (1, alpha, beta, 8, 0.0),
            &bounds,
            Params::default(),
            &window,
        )
        .expect("drivable");
        assert!(verify_segment_with_weights(
            &ops,
            &bundle,
            &published,
            &weights,
            &dims,
            (1, alpha, beta, 8, 0.0),
            &bounds,
            &window
        ));
    }

    #[test]
    fn head_tail_is_proven_on_the_last_logits_row() {
        let alpha = Fp::new(11);
        let beta = Fp::new(5);
        let mut bounds = base_bounds();
        bounds.score_lo = 0;
        bounds.score_hi = 9;
        let dims = crate::model::spec::Dims {
            hidden: 2,
            vocab: 2,
            layers: 1,
            heads: 1,
            head_dim: 2,
            experts: 2,
            active: 1,
            intermediate: 2,
        };
        // Two rows of four logits; the head is row 1, token 1 (score 9).
        let rows = 2usize;
        let n = 4usize;
        let logits = vec![1i64, 4, 2, 3, 5, 9, 3, 7];
        let token = 1usize;
        let head_max = logits[(rows - 1) * n + token];
        assert_eq!(head_max, 9);

        let cells = vec![Cell::new("logits", rows, n, logits.clone())];
        let ops: Vec<Op> = Vec::new();
        let spec = HeadSpec {
            logits: "logits".into(),
            token,
            max: head_max,
        };
        let bundle = prove_segment_with_head(
            &ops,
            &cells,
            &[],
            &dims,
            (1, alpha, beta, 8, 0.0),
            &bounds,
            Params::default(),
            &[],
            Some(&spec),
        )
        .expect("drivable");
        let published = std::collections::HashMap::new();
        assert!(verify_segment_with_head(
            &ops,
            &bundle,
            &published,
            &[],
            &dims,
            (1, alpha, beta, 8, 0.0),
            &bounds,
            &[],
            Some(&spec)
        ));

        // A token that is not the maximum of that row is rejected.
        let bad = HeadSpec {
            token: 0,
            max: head_max,
            ..spec.clone()
        };
        assert!(!verify_segment_with_head(
            &ops,
            &bundle,
            &published,
            &[],
            &dims,
            (1, alpha, beta, 8, 0.0),
            &bounds,
            &[],
            Some(&bad)
        ));
    }
}
