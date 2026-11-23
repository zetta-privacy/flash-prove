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

use crate::gkr::requant::RequantSegmentProof;
use crate::gkr::segment::{
    claim_add, claim_mul, claim_reshape, verify_add, verify_mul, verify_reshape, AddSegmentProof,
    CellClaim, CellSegmentCommitment, MatmulWeightProof, MulSegmentProof, ReshapeSegmentProof,
    SegmentCells,
};
use crate::field::Fp;
use crate::model::spec::Op;
use crate::model::Cell;
use crate::pcs::fri::{new_challenger, Transcript};
use crate::model::WeightMatrix;
use crate::pcs::whir_stacked::{self, Params, SegmentProver};
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};

/// One op's share of a segment proof.
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SegmentOpProof {
    Add(AddSegmentProof),
    Mul(MulSegmentProof),
    Reshape(ReshapeSegmentProof),
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
    /// One proof per row of the attention scores.
    Softmax(Vec<crate::gkr::nonlinear::SoftmaxRowProof>),
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
}

/// Why a segment could not be driven.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DriverError {
    /// The op has no driver wiring yet.
    Unsupported(&'static str),
    /// An op names a cell the segment does not hold.
    UnknownCell(String),
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
        Op::Add { inputs, output } => inputs
            .iter()
            .map(|s| s.as_str())
            .chain([output.as_str()])
            .collect(),
        Op::Mul { a, b, output } => vec![a, b, output],
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
            Op::Add { inputs, output } => {
                let ins: Vec<usize> = inputs.iter().map(|n| seg.id(n)).collect();
                let out_id = seg.id(output);
                proofs.push(SegmentOpProof::Add(claim_add(
                    &mut seg,
                    out_id,
                    &ins,
                    &mut tr,
                )));
            }
            Op::Mul { a, b, output } => {
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
    })
}

/// Verify a segment against the op list it was proved for.
pub fn verify_segment(ops: &[Op], out: &SegmentOutput) -> bool {
    if out.ops.len() != ops.len() {
        return false;
    }
    // The verifier's own name → id map (the plan order is part of the commitment).
    let id = |name: &str| out.commitment.cell(name);
    let mut tr = Transcript::new(&new_challenger());

    for (op, proof) in ops.iter().zip(&out.ops) {
        match (op, proof) {
            (Op::Add { inputs, output }, SegmentOpProof::Add(p)) => {
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
                if !verify_add(&out.commitment, &out.claims, p, &mut tr) {
                    return false;
                }
            }
            (Op::Mul { a, b, output }, SegmentOpProof::Mul(p)) => {
                let (Some(ca), Some(cb), Some(cc)) = (id(a), id(b), id(output)) else {
                    return false;
                };
                if (p.a, p.b, p.c) != (ca, cb, Some(cc)) {
                    return false;
                }
                if !verify_mul(&out.commitment, &out.claims, p, &mut tr) {
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
                if !verify_reshape(&out.commitment, &out.claims, p, &mut tr) {
                    return false;
                }
            }
            _ => return false,
        }
    }

    // The op claims are only sound if the segment's own openings check out.
    crate::gkr::segment::verify(&out.commitment, &out.claims, &out.proof, &mut new_challenger())
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
            wdims: wdims.iter().map(|(n, r, c)| (n.clone(), (*r, *c))).collect(),
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
            Op::Add { inputs, output } => {
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
            tensors: weights.iter().map(|w| (w.name.clone(), w.clone())).collect(),
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
            let prov = weight_prover(w, self.params);
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

    /// Derive the bounds from a trace, exactly as the FRI config does.
    pub fn derive(spec: &crate::model::spec::ModelSpec, trace: &crate::model::Trace) -> Self {
        let c = crate::gkr::commit::CommittedConfig::derive(spec, trace);
        Self {
            silu_lo: c.silu_lo,
            silu_hi: c.silu_hi,
            sign_lo: c.sign_lo,
            sign_hi: c.sign_hi,
            exp_lo: c.exp_lo,
            score_lo: c.score_lo,
            score_hi: c.score_hi,
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



/// A segment proved together with the weight roots it opened against.
pub struct SegmentBundle {
    pub segment: SegmentOutput,
    /// `(name, root, proof)` per weight tensor the segment claimed on.
    pub weights: Vec<(String, whir_stacked::SegmentCommitment, whir_stacked::Proof)>,
}

/// `y = W·x` for a single input row.
fn matvec(w: &WeightMatrix, x: &[i64]) -> Vec<i64> {
    assert_eq!(x.len(), w.cols, "matvec input width");
    (0..w.rows)
        .map(|i| (0..w.cols).map(|j| w.values[i * w.cols + j] as i64 * x[j]).sum())
        .collect()
}

/// Prove one segment of `matmul` ops, reading each weight from the pre-shared
/// registry.
///
/// `y = W·x` is a planned cell (its integer values are computed here, since the
/// trace only carries the requantised output), so `claim_matmul_weight` binds `y` to
/// the weight's own root and `claim_requant` binds the trace output to `y`.
pub fn prove_segment_with_weights(
    ops: &[Op],
    cells: &[Cell],
    weights: &[WeightMatrix],
    dims: &crate::model::spec::Dims,
    (k, digits, alpha, beta, rec_q, eps): (usize, usize, Fp, Fp, u32, f64),
    bounds: &LookupBounds,
    params: Params,
) -> Result<SegmentBundle, DriverError> {
    let _ = dims;
    let by_name: std::collections::HashMap<&str, &WeightMatrix> =
        weights.iter().map(|w| (w.name.as_str(), w)).collect();

    // Pass 1: plan every derived cell, keeping the names for the claim pass.
    let mut plan = crate::gkr::plan::Plan::new(cells);
    let mut planned: Vec<Option<PlannedOp>> = Vec::with_capacity(ops.len());
    for (i, op) in ops.iter().enumerate() {
        match op {
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
                ..
            } => {
                let w = by_name
                    .get(weight.as_str())
                    .ok_or_else(|| DriverError::UnknownCell(weight.clone()))?;
                let xs = plan.values(input).to_vec();
                if xs.len() != w.cols {
                    return Err(DriverError::UnknownCell(input.clone()));
                }
                let y = matvec(w, &xs);
                let y_name = format!("__mm.{i}.y");
                plan.add(&y_name, y.len(), 1, y);
                let rq =
                    plan.requant(&format!("__mm.{i}.rq"), &y_name, output, k, digits);
                planned.push(Some(PlannedOp::Matmul { rq }));
            }
            Op::WeightedSum { .. } => planned.push(Some(PlannedOp::WeightedSum)),
            Op::Scores {
                q,
                k,
                output,
                scale,
            } => {
                let (h, d) = (dims.heads, dims.head_dim);
                let Some((lq, hd)) = plan.logical(q) else {
                    return Err(DriverError::UnknownCell(q.clone()));
                };
                if hd != h * d {
                    return Err(DriverError::UnknownCell(q.clone()));
                }
                let inv_q =
                    ((1i64 << crate::model::spec::RECIP_Q) as f64 / *scale).round() as i64;
                let qs = plan.values(q).to_vec();
                let ks = plan.values(k).to_vec();
                let mut per_head = Vec::with_capacity(lq * h);
                for li in 0..lq {
                    for hi in 0..h {
                        // dot[t] = q[li,hi,·] · k[t,hi,·]
                        let dot: Vec<i64> = (0..lq)
                            .map(|t| {
                                (0..d)
                                    .map(|j| {
                                        qs[((li * h) + hi) * d + j] * ks[(t * h + hi) * d + j]
                                    })
                                    .sum()
                            })
                            .collect();
                        let y: Vec<i64> = dot.iter().map(|&v| v * inv_q).collect();
                        let sr: Vec<i64> = y
                            .iter()
                            .map(|&v| crate::model::spec::requant(v, crate::model::spec::RECIP_Q))
                            .collect();
                        let pre = format!("__sc.{i}.{li}.{hi}");
                        plan.add(&format!("{pre}.dot"), lq, 1, dot);
                        plan.add(&format!("{pre}.y"), lq, 1, y);
                        plan.add(&format!("{pre}.s"), lq, 1, sr);
                        let requant = plan.requant(
                            &format!("{pre}.rq"),
                            &format!("{pre}.y"),
                            &format!("{pre}.s"),
                            crate::model::spec::RECIP_Q as usize,
                            digits,
                        );
                        per_head.push(ScoresPlan {
                            li,
                            hi,
                            inv_q,
                            dot: format!("{pre}.dot"),
                            y: format!("{pre}.y"),
                            s: format!("{pre}.s"),
                            requant,
                        });
                    }
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
                let mut per_head = Vec::with_capacity(lq * h);
                for li in 0..lq {
                    for hi in 0..h {
                        let acc: Vec<i64> = (0..d)
                            .map(|i| {
                                (0..lq)
                                    .map(|t| vs[(t * h + hi) * d + i] * ws[((li * h) + hi) * lq + t])
                                    .sum()
                            })
                            .collect();
                        let ob: Vec<i64> = acc
                            .iter()
                            .map(|&x| crate::model::spec::requant(x, 15))
                            .collect();
                        let pre = format!("__wv.{i}.{li}.{hi}");
                        plan.add(&format!("{pre}.acc"), d, 1, acc);
                        plan.add(&format!("{pre}.ob"), d, 1, ob);
                        let requant = plan.requant(
                            &format!("{pre}.rq"),
                            &format!("{pre}.acc"),
                            &format!("{pre}.ob"),
                            15,
                            digits,
                        );
                        per_head.push(WeightedVPlan {
                            li,
                            hi,
                            acc: format!("{pre}.acc"),
                            ob: format!("{pre}.ob"),
                            requant,
                        });
                    }
                }
                planned.push(Some(PlannedOp::WeightedV(per_head)));
            }
            Op::Norm {
                input, weight, ..
            } => {
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
                        .map(|&v| crate::model::spec::requant(v, k as u32))
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
                let Some((rows, n)) = plan.logical(input) else {
                    return Err(DriverError::UnknownCell(input.clone()));
                };
                let table = bounds.exp_table(alpha, beta);
                let big_n = bounds.exp_big_n();
                let max_tbl = crate::gkr::nonlinear::max_table(bounds.exp_lo, alpha, beta);
                let max_big = crate::gkr::nonlinear::max_big_n(bounds.exp_lo);
                let xs = plan.values(input).to_vec();
                let mut per_row = Vec::with_capacity(rows);
                for row in 0..rows {
                    let l = row.min(n - 1);
                    let m = (0..=l).map(|i| xs[row * n + i]).max().unwrap_or(0);
                    let tstar = (0..=l).find(|&i| xs[row * n + i] == m).unwrap_or(0);
                    let d: Vec<i64> = (0..n)
                        .map(|i| if i <= l { xs[row * n + i] - m } else { bounds.exp_lo })
                        .collect();
                    let e: Vec<i64> = d.iter().map(|&v| crate::model::spec::exp_q15(v)).collect();
                    let s_sum: i64 = e.iter().take(l + 1).sum();
                    let inv_q = crate::gkr::nonlinear::softmax_inv_q(s_sum, rec_q);
                    let y: Vec<i64> = e.iter().map(|&v| v * inv_q).collect();
                    let nrm: Vec<i64> = y
                        .iter()
                        .map(|&v| crate::model::spec::requant(v, k as u32))
                        .collect();
                    let pre = format!("__sm.{i}.{row}");
                    plan.add(&format!("{pre}.d"), n, 1, d.clone());
                    plan.add(&format!("{pre}.e"), n, 1, e.clone());
                    plan.add(&format!("{pre}.y"), n, 1, y.clone());
                    plan.add(&format!("{pre}.nrm"), n, 1, nrm.clone());
                    let over: Vec<i64> = d.iter().map(|&x| if x > 0 { 1 } else { 0 }).collect();
                    plan.add(&format!("{pre}.over"), n, 1, over);
                    let lookup = plan.lookup2(
                        &format!("{pre}.lk"),
                        &format!("{pre}.d"),
                        &format!("{pre}.e"),
                        (alpha, beta),
                        &table,
                        big_n,
                    );
                    let over_lookup = plan.lookup2(
                        &format!("{pre}.olk"),
                        &format!("{pre}.d"),
                        &format!("{pre}.over"),
                        (alpha, beta),
                        &max_tbl,
                        max_big,
                    );
                    let requant = plan.requant(
                        &format!("{pre}.rq"),
                        &format!("{pre}.y"),
                        &format!("{pre}.nrm"),
                        k,
                        digits,
                    );
                    per_row.push(SoftmaxPlan {
                        d: format!("{pre}.d"),
                        e: format!("{pre}.e"),
                        y: format!("{pre}.y"),
                        nrm: format!("{pre}.nrm"),
                        lookup,
                        requant,
                        over: format!("{pre}.over"),
                        over_lookup,
                        m,
                        tstar,
                    });
                }
                let _ = output;
                planned.push(Some(PlannedOp::Softmax(per_row)));
            }
            Op::TopK { input, choice, k, .. } => {
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
            other => return Err(DriverError::Unsupported(op_name(other))),
        }
    }

    let mut seg = plan.seal(params);
    let mut set = WeightSet::new(weights, params);
    let mut tr = Transcript::new(&new_challenger());
    let mut proofs = Vec::with_capacity(ops.len());

    // Pass 2: claim, in spec order.
    for (i, op) in ops.iter().enumerate() {
        let names = planned[i].take().expect("planned in pass 1");
        match (op, names) {
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
            (Op::Matmul { input, weight, .. }, PlannedOp::Matmul { rq }) => {
                let w = by_name[weight.as_str()];
                let y_id = seg.id(&format!("__mm.{i}.y"));
                let ci = seg.id(input);
                let (ln, lm, lp) = (
                    w.rows.trailing_zeros() as usize,
                    w.cols.trailing_zeros() as usize,
                    0usize,
                );
                let wp = set.get_or_commit(weight)?;
                let mm = crate::gkr::segment::claim_matmul_weight(
                    &mut seg,
                    wp,
                    0,
                    &crate::gkr::segment::View::Whole,
                    (ci, y_id),
                    weight,
                    &mut tr,
                    (ln, lm, lp),
                );
                let rq_cells = rq.cells(&seg);
                let rq = crate::gkr::requant::claim_requant(
                    &mut seg,
                    rq_cells,
                    (w.rows, k, digits),
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
                        (hidden, k, digits),
                        (rec_q, eps),
                        &gamma_pub,
                        &mut tr,
                    ));
                }
                proofs.push(SegmentOpProof::Norm(row_proofs));
            }
            (
                Op::WeightedV { weights, v, output },
                PlannedOp::WeightedV(per_head),
            ) => {
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
                    let block_start = ((hp.li * h) + hp.hi) * d;
                    out.push(crate::gkr::attention::claim_weighted_v(
                        &mut seg,
                        cells,
                        (log_l, log_d, log_h),
                        (hp.hi, block_start),
                        (d, 15, digits),
                        &mut tr,
                    ));
                }
                proofs.push(SegmentOpProof::WeightedV(out));
            }
            (
                Op::Scores { q, k, output, .. },
                PlannedOp::Scores(per_head),
            ) => {
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
                    let q_start = ((hp.li * h) + hp.hi) * d;
                    let out_start = ((hp.li * h) + hp.hi) * lq;
                    out.push(crate::gkr::attention::claim_scores(
                        &mut seg,
                        cells,
                        (log_l, log_d, log_h),
                        hp.hi,
                        (q_start, out_start),
                        Fp::new(hp.inv_q as u64),
                        (lq, crate::model::spec::RECIP_Q as usize, digits),
                        &mut tr,
                    ));
                }
                proofs.push(SegmentOpProof::Scores(out));
            }
            (
                Op::CausalSoftmax { input, output },
                PlannedOp::Softmax(per_row),
            ) => {
                let (ci, co) = (seg.id(input), seg.id(output));
                let (rows, n) = {
                    let m = seg.cells_meta(ci);
                    (m.logical_rows, m.logical_cols)
                };
                let table = bounds.exp_table(alpha, beta);
                let big_n = bounds.exp_big_n();
                let mut row_proofs = Vec::with_capacity(rows);
                for (row, pc) in per_row.iter().enumerate() {
                    let cells = crate::gkr::nonlinear::SoftmaxRowCells {
                        scores: ci,
                        probs: co,
                        d: seg.id(&pc.d),
                        e: seg.id(&pc.e),
                        y: seg.id(&pc.y),
                        nrm: seg.id(&pc.nrm),
                        lookup: pc.lookup.cells(&seg),
                        over: seg.id(&pc.over),
                        over_lookup: pc.over_lookup.cells(&seg),
                        requant: pc.requant.cells(&seg),
                    };
                    let l = row.min(n - 1);
                    row_proofs.push(crate::gkr::nonlinear::claim_softmax_row(
                        &mut seg,
                        cells,
                        (row, n, l),
                        (pc.m, bounds.exp_lo, pc.tstar),
                        (rec_q, k, digits),
                        (alpha, beta),
                        (&table, big_n),
                        &mut tr,
                    ));
                }
                proofs.push(SegmentOpProof::Softmax(row_proofs));
            }
            (
                Op::TopK { input, choice, k, .. },
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
                proofs.push(SegmentOpProof::TopK { cells, rows: row_proofs, lookup });
            }
            _ => return Err(DriverError::Unsupported(op_name(op))),
        }
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

/// What a softmax row needed at plan time.
#[allow(dead_code)]
struct SoftmaxPlan {
    d: String,
    e: String,
    y: String,
    nrm: String,
    lookup: crate::gkr::plan::Lookup2Names,
    requant: crate::gkr::plan::RequantNames,
    over: String,
    over_lookup: crate::gkr::plan::Lookup2Names,
    m: i64,
    tstar: usize,
}

/// What pass 1 planned for one op.
#[allow(dead_code)]
enum PlannedOp {
    Silu(crate::gkr::plan::Lookup2Names),
    Matmul { rq: crate::gkr::plan::RequantNames },
    WeightedSum,
    /// Per-row requant names.
    Norm(Vec<crate::gkr::plan::RequantNames>),
    /// Per-row softmax names.
    Softmax(Vec<SoftmaxPlan>),
    /// Per-head names.
    WeightedV(Vec<WeightedVPlan>),
    /// Per-head names.
    Scores(Vec<ScoresPlan>),
    TopK(TopKPlan),
}

/// What a `scores` head needed at plan time.
#[allow(dead_code)]
struct ScoresPlan {
    li: usize,
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
    li: usize,
    hi: usize,
    acc: String,
    ob: String,
    requant: crate::gkr::plan::RequantNames,
}

/// Verify a [`SegmentBundle`] against the published weight roots.
pub fn verify_segment_with_weights(
    ops: &[Op],
    bundle: &SegmentBundle,
    published: &std::collections::HashMap<String, whir_stacked::SegmentCommitment>,
    weights: &[WeightMatrix],
    dims: &crate::model::spec::Dims,
    (k, digits, alpha, beta, rec_q, eps): (usize, usize, Fp, Fp, u32, f64),
    bounds: &LookupBounds,
) -> bool {
    let _ = dims;
    if bundle.segment.ops.len() != ops.len() {
        return false;
    }
    let wmap: std::collections::HashMap<&str, (&whir_stacked::SegmentCommitment, &whir_stacked::Proof)> =
        bundle
            .weights
            .iter()
            .map(|(n, c, p)| (n.as_str(), (c, p)))
            .collect();
    let mut tr = Transcript::new(&new_challenger());

    for (i, (op, proof)) in ops.iter().zip(&bundle.segment.ops).enumerate() {
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
                    &bundle.segment.claims,
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
                    ..
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
                let (Some(ci), Some(rows)) = (
                    bundle.segment.commitment.cell(input),
                    bundle
                        .segment
                        .commitment
                        .cell(output)
                        .map(|i| bundle.segment.commitment.cells[i].rows),
                ) else {
                    return false;
                };
                if mm.b != ci {
                    return false;
                }
                if !crate::gkr::segment::verify_matmul_weight(
                    &bundle.segment.commitment,
                    &bundle.segment.claims,
                    w_com,
                    w_proof,
                    &crate::gkr::segment::View::Whole,
                    mm,
                    &mut tr,
                ) {
                    return false;
                }
                if !crate::gkr::requant::verify_requant(
                    &bundle.segment.commitment,
                    &bundle.segment.claims,
                    rq,
                    (rows, k, digits),
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
                        &bundle.segment.claims,
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
                let Some(gamma_id) = bundle
                    .segment
                    .commitment
                    .cell(&format!("__norm.gamma.{i}"))
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
                        &bundle.segment.claims,
                        rp,
                        (row, hidden),
                        (hidden, k, digits),
                        (rec_q, eps),
                        &gamma_pub,
                        &mut tr,
                    ) {
                        return false;
                    }
                }
            }
            (
                Op::WeightedV { output, v, .. },
                SegmentOpProof::WeightedV(per_head),
            ) => {
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
                if per_head.len() != lq * h {
                    return false;
                }
                for (n, hp) in per_head.iter().enumerate() {
                    let (li, hi) = (n / h, n % h);
                    if hp.cells.out != co || hp.cells.v != cv {
                        return false;
                    }
                    let block_start = ((li * h) + hi) * d;
                    if !crate::gkr::attention::verify_weighted_v(
                        &bundle.segment.commitment,
                        &bundle.segment.claims,
                        hp,
                        (log_l, log_d, log_h),
                        (hi, block_start),
                        (d, 15, digits),
                        &mut tr,
                    ) {
                        return false;
                    }
                }
            }
            (
                Op::Scores { q, k, output, .. },
                SegmentOpProof::Scores(per_head),
            ) => {
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
                if per_head.len() != lq * h {
                    return false;
                }
                for (n, hp) in per_head.iter().enumerate() {
                    let (li, hi) = (n / h, n % h);
                    let c = &hp.cells;
                    if c.q != cq || c.k != ck || c.out != co || hp.head != hi {
                        return false;
                    }
                    let q_start = ((li * h) + hi) * d;
                    let out_start = ((li * h) + hi) * lq;
                    if !crate::gkr::attention::verify_scores(
                        &bundle.segment.commitment,
                        &bundle.segment.claims,
                        hp,
                        (log_l, log_d, log_h),
                        hi,
                        (q_start, out_start),
                        hp.inv_q,
                        (lq, crate::model::spec::RECIP_Q as usize, digits),
                        &mut tr,
                    ) {
                        return false;
                    }
                }
            }
            (
                Op::CausalSoftmax { input, output },
                SegmentOpProof::Softmax(row_proofs),
            ) => {
                let (Some(ci), Some(co)) = (
                    bundle.segment.commitment.cell(input),
                    bundle.segment.commitment.cell(output),
                ) else {
                    return false;
                };
                let (rows, n) = {
                    let m = &bundle.segment.commitment.cells[co];
                    (m.logical_rows, m.logical_cols)
                };
                let table = bounds.exp_table(alpha, beta);
                let big_n = bounds.exp_big_n();
                if row_proofs.len() != rows {
                    return false;
                }
                for (row, rp) in row_proofs.iter().enumerate() {
                    let c = &rp.cells;
                    if c.scores != ci || c.probs != co {
                        return false;
                    }
                    let l = row.min(n - 1);
                    if !crate::gkr::nonlinear::verify_softmax_row(
                        &bundle.segment.commitment,
                        &bundle.segment.claims,
                        rp,
                        (row, n, l),
                        (rp.row_max.m, bounds.exp_lo, rp.row_max.tstar),
                        (rec_q, k, digits),
                        (&table, big_n),
                        (
                            &crate::gkr::nonlinear::max_table(bounds.exp_lo, alpha, beta),
                            crate::gkr::nonlinear::max_big_n(bounds.exp_lo),
                        ),
                        &mut tr,
                    ) {
                        return false;
                    }
                }
            }
            (
                Op::TopK { input, choice, k, .. },
                SegmentOpProof::TopK { cells, rows: row_proofs, lookup },
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
                    ] != [plan_lookup[0], plan_lookup[1], plan_lookup[2], plan_lookup[3], plan_lookup[4]]
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
                        &bundle.segment.claims,
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
                    &bundle.segment.claims,
                    lookup,
                    &table,
                    (rows * e, bounds.sign_big_n()),
                    &mut tr,
                ) {
                    return false;
                }
            }
            _ => return false,
        }
    }

    // The op claims are only sound if the segment's own openings check out.
    crate::gkr::segment::verify(
        &bundle.segment.commitment,
        &bundle.segment.claims,
        &bundle.segment.proof,
        &mut new_challenger(),
    )
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
            },
            Op::Mul {
                a: "a".into(),
                b: "b".into(),
                output: "prod".into(),
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
        c.add_vec(&w1.values.iter().map(|&v| signed(v as i64)).collect::<Vec<_>>());
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

        let bundle = prove_segment_with_weights(&ops, &cells, &[w.clone()], &dims, (k, 1, alpha, beta, 8, 0.0), &bounds, Params::default())
            .expect("drivable");

        // The published map is what `cp_commit` would hand over.
        let mut c = Committer::new();
        c.add_vec(&w.values.iter().map(|&v| signed(v as i64)).collect::<Vec<_>>());
        let pub_seg = c.seal(Params::default());
        let published: std::collections::HashMap<String, whir_stacked::SegmentCommitment> =
            [("W".to_string(), pub_seg.commitment().clone())]
                .into_iter()
                .collect();

        assert!(verify_segment_with_weights(&ops, &bundle, &published, &[], &dims, (k, 1, alpha, beta, 8, 0.0), &bounds));

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
        assert!(!verify_segment_with_weights(&ops, &bundle, &wrong2, &[], &dims, (k, 1, alpha, beta, 8, 0.0), &bounds));

        // A wrong output breaks the requant link to `y`.
        let bad_cells = vec![
            Cell::new("x", 2, 1, x),
            Cell::new("out", 2, 1, vec![out[0] + 1, out[1]]),
        ];
        let bad = prove_segment_with_weights(&ops, &bad_cells, &[w], &dims, (k, 1, alpha, beta, 8, 0.0), &bounds, Params::default())
            .unwrap();
        assert!(!verify_segment_with_weights(&ops, &bad, &published, &[], &dims, (k, 1, alpha, beta, 8, 0.0), &bounds));
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

        // Bound derivation agrees with the FRI config it wraps.
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
        let trace = crate::model::Trace { segments: Vec::new() };
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
            (1, 1, alpha, beta, 8, 0.0),
            &bounds,
            Params::default(),
        )
        .expect("drivable");
        let published = std::collections::HashMap::new();
        assert!(verify_segment_with_weights(
            &ops,
            &bundle,
            &published,
            &[],
            &dims,
            (1, 1, alpha, beta, 8, 0.0),
            &bounds
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
            (1, 1, alpha, beta, 8, 0.0),
            &bounds,
            Params::default(),
        )
        .unwrap();
        assert!(!verify_segment_with_weights(
            &ops,
            &bad,
            &published,
            &[],
            &dims,
            (1, 1, alpha, beta, 8, 0.0),
            &bounds
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
            (1, 1, alpha, beta, 8, 0.0),
            &narrow
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
            &ops, &cells, &[], &dims, (1, 1, alpha, beta, 8, 0.0), &bounds, Params::default(),
        )
        .expect("drivable");
        let co = bundle.segment.commitment.cell("out").unwrap();
        assert_eq!(bundle.segment.commitment.cells[co].logical_rows, 2);
        assert_eq!(bundle.segment.commitment.cells[co].logical_cols, 2);
        let published = std::collections::HashMap::new();
        assert!(verify_segment_with_weights(
            &ops, &bundle, &published, &[], &dims, (1, 1, alpha, beta, 8, 0.0), &bounds
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
        let (k, rec_q, eps, digits) = (1usize, 8u32, 0.0f64, 2usize);
        let gamma = WeightMatrix::new("g", 2, 1, vec![1, 2]);
        let x = vec![1i64, 2, 3, 4];
        let mut out = Vec::new();
        for row in 0..2usize {
            let xr = &x[row * 2..(row + 1) * 2];
            let sum_sq: i64 = xr.iter().map(|&v| v * v).sum();
            let inv_q = crate::gkr::nonlinear::norm_inv_q(sum_sq, 2, rec_q, eps);
            for j in 0..2usize {
                let y = xr[j] * (gamma.values[j] as i64) * inv_q;
                out.push(crate::model::spec::requant(y, k as u32));
            }
        }

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
            (k, digits, alpha, beta, rec_q, eps),
            &bounds,
            Params::default(),
        )
        .expect("drivable");
        let published = std::collections::HashMap::new();
        assert!(verify_segment_with_weights(
            &ops, &bundle, &published, &weights, &dims, (k, digits, alpha, beta, rec_q, eps), &bounds
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
            (k, digits, alpha, beta, rec_q, eps),
            &bounds,
            Params::default(),
        )
        .unwrap();
        assert!(!verify_segment_with_weights(
            &ops, &bad, &published, &weights, &dims, (k, digits, alpha, beta, rec_q, eps), &bounds
        ));

        // A γ the verifier does not know is rejected: the copy is bound to it.
        let other = vec![WeightMatrix::new("g", 2, 1, vec![5, 7])];
        assert!(!verify_segment_with_weights(
            &ops, &bundle, &published, &other, &dims, (k, digits, alpha, beta, rec_q, eps), &bounds
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
        }
    }

    #[test]
    fn weighted_v_driven_per_head() {
        let alpha = Fp::new(11);
        let beta = Fp::new(5);
        let bounds = base_bounds();
        let (lq, h, d) = (2usize, 2usize, 2usize);
        let dims = crate::model::spec::Dims {
            hidden: 4, vocab: 2, layers: 1, heads: h, head_dim: d,
            experts: 2, active: 1, intermediate: 4,
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
        let ops = vec![Op::WeightedV { weights: "w".into(), v: "v".into(), output: "out".into() }];
        let bundle = prove_segment_with_weights(
            &ops, &cells, &[], &dims, (1, 3, alpha, beta, 8, 0.0), &bounds, Params::default(),
        )
        .expect("drivable");
        let published = std::collections::HashMap::new();
        assert!(verify_segment_with_weights(
            &ops, &bundle, &published, &[], &dims, (1, 3, alpha, beta, 8, 0.0), &bounds
        ));

        let mut bad = out.clone();
        bad[0] += 1;
        let bad_cells = vec![
            Cell::new("w", lq, h * lq, w),
            Cell::new("v", lq, h * d, v),
            Cell::new("out", lq, h * d, bad),
        ];
        let bad_bundle = prove_segment_with_weights(
            &ops, &bad_cells, &[], &dims, (1, 3, alpha, beta, 8, 0.0), &bounds, Params::default(),
        )
        .unwrap();
        assert!(!verify_segment_with_weights(
            &ops, &bad_bundle, &published, &[], &dims, (1, 3, alpha, beta, 8, 0.0), &bounds
        ));
    }

    #[test]
    fn scores_driven_per_head() {
        let alpha = Fp::new(11);
        let beta = Fp::new(5);
        let bounds = base_bounds();
        let (lq, h, d) = (2usize, 2usize, 2usize);
        let dims = crate::model::spec::Dims {
            hidden: 4, vocab: 2, layers: 1, heads: h, head_dim: d,
            experts: 2, active: 1, intermediate: 4,
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
        let ops = vec![Op::Scores { q: "q".into(), k: "k".into(), output: "out".into(), scale: 1.0 }];
        let bundle = prove_segment_with_weights(
            &ops, &cells, &[], &dims, (1, 4, alpha, beta, 8, 0.0), &bounds, Params::default(),
        )
        .expect("drivable");
        let published = std::collections::HashMap::new();
        assert!(verify_segment_with_weights(
            &ops, &bundle, &published, &[], &dims, (1, 4, alpha, beta, 8, 0.0), &bounds
        ));

        let mut bad = out.clone();
        bad[0] += 1;
        let bad_cells = vec![
            Cell::new("q", lq, h * d, q),
            Cell::new("k", lq, h * d, k),
            Cell::new("out", lq, h * lq, bad),
        ];
        let bad_bundle = prove_segment_with_weights(
            &ops, &bad_cells, &[], &dims, (1, 4, alpha, beta, 8, 0.0), &bounds, Params::default(),
        )
        .unwrap();
        assert!(!verify_segment_with_weights(
            &ops, &bad_bundle, &published, &[], &dims, (1, 4, alpha, beta, 8, 0.0), &bounds
        ));
    }

    #[test]
    fn causal_softmax_driven_row_by_row() {
        let alpha = Fp::new(11);
        let beta = Fp::new(5);
        let bounds = base_bounds();
        let dims = crate::model::spec::Dims {
            hidden: 2, vocab: 2, layers: 1, heads: 1, head_dim: 2,
            experts: 2, active: 1, intermediate: 2,
        };
        let (k, rec_q, eps, digits) = (1usize, 1u32, 0.0f64, 3usize);
        let exp_lo = bounds.exp_lo;
        let scores = vec![3i64, 1, 2, 5];
        let (rows, n) = (2usize, 2usize);
        let mut probs = Vec::new();
        for row in 0..rows {
            let l = row.min(n - 1);
            let m = (0..=l).map(|i| scores[row * n + i]).max().unwrap();
            let d: Vec<i64> = (0..n)
                .map(|i| if i <= l { scores[row * n + i] - m } else { exp_lo })
                .collect();
            let e: Vec<i64> = d.iter().map(|&v| crate::model::spec::exp_q15(v)).collect();
            let s: i64 = e.iter().take(l + 1).sum();
            let inv_q = crate::gkr::nonlinear::softmax_inv_q(s, rec_q);
            for i in 0..n {
                probs.push(crate::model::spec::requant(e[i] * inv_q, k as u32));
            }
        }
        let cells = vec![
            Cell::new("scores", rows, n, scores.clone()),
            Cell::new("probs", rows, n, probs.clone()),
        ];
        let ops = vec![Op::CausalSoftmax { input: "scores".into(), output: "probs".into() }];
        let bundle = prove_segment_with_weights(
            &ops, &cells, &[], &dims, (k, digits, alpha, beta, rec_q, eps), &bounds, Params::default(),
        )
        .expect("drivable");
        let published = std::collections::HashMap::new();
        assert!(verify_segment_with_weights(
            &ops, &bundle, &published, &[], &dims, (k, digits, alpha, beta, rec_q, eps), &bounds
        ));

        let mut bad = probs.clone();
        bad[0] += 1;
        let bad_cells = vec![
            Cell::new("scores", rows, n, scores),
            Cell::new("probs", rows, n, bad),
        ];
        let bad_bundle = prove_segment_with_weights(
            &ops, &bad_cells, &[], &dims, (k, digits, alpha, beta, rec_q, eps), &bounds, Params::default(),
        )
        .unwrap();
        assert!(!verify_segment_with_weights(
            &ops, &bad_bundle, &published, &[], &dims, (k, digits, alpha, beta, rec_q, eps), &bounds
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
            hidden: 2, vocab: 2, layers: 1, heads: 1, head_dim: 2,
            experts: 2, active: 1, intermediate: 2,
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
            &ops, &cells, &[], &dims, (1, 1, alpha, beta, 8, 0.0), &bounds, Params::default(),
        )
        .expect("drivable");
        let published = std::collections::HashMap::new();
        assert!(verify_segment_with_weights(
            &ops, &bundle, &published, &[], &dims, (1, 1, alpha, beta, 8, 0.0), &bounds
        ));
    }
}
