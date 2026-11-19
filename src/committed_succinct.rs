//! Full committed `verify_succinct`: walk a model's op graph and check every op
//! against FRI-committed weights + trace, using the per-op committed primitives
//! with a *single* shared commitment per cell (by name).
//!
//! This is the succinct counterpart of [`crate::succinct::verify_succinct`]: the
//! verifier holds only commitments + opening proofs, never the clear activations.
//! The prover commits each cell and weight once ([`commit_cells`]); every primitive
//! reuses the shared commitments via its `*_given` form, so a cell that is the
//! output of one op and the input of the next is literally the same commitment in
//! both (no soundness gap from re-committing).
//!
//! Matmuls are batched across the sequence dimension into a single matmul over a
//! block-diagonal weight `W ⊗ I_L` (committed fresh and checked to be block
//! diagonal), so the sequence cell is consumed at full granularity rather than as
//! row slices.
//!
//! The nonlinear 3-D attention / MoE ops (`norm`, `rope`, `scores`, `causal_softmax`,
//! `weighted_v`, `top_k`, `weighted_sum`) require per-(position, head) row slicing
//! against a whole-tensor commitment and are the next piece; this assembly covers
//! the dense/linear core (embed, matmul, silu, mul, add, reshape) end to end.

use std::collections::HashMap;

use crate::committed::{prove_matmul_committed_given, verify_matmul_committed, MatmulProof};
use crate::field::{Ext, Fp};
use crate::fri::{self, Challenger, Pcs, Prover};
use crate::ipa::{prove_ipa_committed_given, verify_ipa_committed, CommittedIpa};
use crate::lookup_logup::{
    commit_function_table, prove_lookup2_logup_given, verify_lookup2_logup, FunctionTable,
    Lookup2LogUpProof,
};
use crate::model::{Trace, Weights};
use crate::mul_committed::{prove_mul_committed_given, verify_mul_committed, MulProof};
use crate::range_check::commit_digit_table;
use crate::requant::{commit_bit_table, prove_requant_given, verify_requant, RequantProof};
use crate::selection_committed::{
    prove_causal_softmax_committed_given, prove_topk_committed_given,
    verify_causal_softmax_committed, verify_topk_committed, CausalSoftmaxProof, TopKProof,
};
use crate::spec::{exp_q15, requant, silu, ModelSpec, Op, RECIP_Q};
use crate::zk::{
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

/// The lookup bounds. Fixed function tables (requant/silu/exp/round/sign) are
/// committed once over `[lo, hi]` and checked via the succinct LogUp; the
/// parameterized lookups (norm `1/sqrt`, softmax normalize) still use the sparse
/// O(n) form.
pub struct CommittedConfig {
    pub requant_digits: usize,
    pub silu_lo: i64,
    pub silu_hi: i64,
    pub exp_lo: i64,
    pub sign_lo: i64,
    pub sign_hi: i64,
}

/// The committed function tables (fixed-table lookups: silu / exp / sign), built
/// once and shared across every lookup of the same function. The parameterized and
/// arithmetic lookups (requant, round, norm `1/sqrt`, softmax normalize) use the
/// sparse O(n) form.
pub struct Tables {
    pub silu: FunctionTable,
    pub exp: FunctionTable,
    pub sign: FunctionTable,
}

fn build_tables(
    pcs: &Pcs,
    cfg: &CommittedConfig,
    alpha: Fp,
    beta: Fp,
    _spec: &ModelSpec,
) -> Tables {
    let silu = commit_function_table(pcs, silu, cfg.silu_lo, cfg.silu_hi, alpha, beta);
    let exp = commit_function_table(pcs, exp_q15, cfg.exp_lo, 0, alpha, beta);
    let sign = commit_function_table(
        pcs,
        |x| if x >= 0 { 1 } else { 0 },
        cfg.sign_lo,
        cfg.sign_hi,
        alpha,
        beta,
    );
    Tables { silu, exp, sign }
}

/// Derive the lookup challenges α, β by absorbing the binding commitments
/// (Fiat–Shamir). Deterministic, so prover and verifier agree.
fn sample_ab(challenger: &Challenger, binding: &[fri::Commitment]) -> (Fp, Fp) {
    let mut tr = fri::Transcript::new(challenger);
    for c in binding {
        tr.observe(c);
    }
    (tr.sample_val(), tr.sample_val())
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

/// Commit every trace cell and every weight once, keyed by name.
pub fn commit_cells(
    pcs: &Pcs,
    weights: &Weights,
    trace: &Trace,
) -> (HashMap<String, CellCommit>, HashMap<String, CellCommit>) {
    let mut cells = HashMap::new();
    let mut wmap = HashMap::new();
    for seg in &trace.segments {
        for c in &seg.cells {
            cells.insert(
                c.name.clone(),
                commit_one(pcs, &c.name, c.rows, c.cols, c.values.clone()),
            );
        }
    }
    for w in weights {
        let vals: Vec<i64> = w.values.iter().map(|&v| v as i64).collect();
        wmap.insert(
            w.name.clone(),
            commit_one(pcs, &w.name, w.rows, w.cols, vals),
        );
    }
    (cells, wmap)
}

// --- per-op proofs -------------------------------------------------------------

/// embed: `out[b] = emb[tok[b]]`, batched over (b, j).
pub struct EmbedProof {
    pub com_out: fri::Commitment,
    pub dom_out: fri::Domain,
    pub com_emb: fri::Commitment,
    pub dom_emb: fri::Domain,
    pub r: Vec<Fp>,
    pub b_out: Fp,
    pub b_emb: Fp,
    pub claim_out: InnerProductProof,
    pub claim_emb: InnerProductProof,
    pub l: usize,
    pub hidden: usize,
    pub vocab: usize,
}

/// matmul: `out = requant(W·x, shift)` batched over the sequence dim.
pub struct MatmulOpProof {
    pub mm: MatmulProof,        // acc = Wb · x
    pub req: RequantProof,      // out = requant(acc, shift)
    pub com_w: fri::Commitment, // original weight W (for the block-diagonal check)
    pub dom_w: fri::Domain,
    pub bd_u: Vec<Fp>, // block-diagonal check challenges
    pub bd_b_wb: Fp,   // ⟨u, Wb⟩
    pub bd_b_w: Fp,    // ⟨diag(u), W⟩
    pub bd_claim_wb: InnerProductProof,
    pub bd_claim_w: InnerProductProof,
    pub shift: u32,
}

/// silu: elementwise lookup.
pub struct SiluProof {
    pub lk: Lookup2LogUpProof,
}

/// add: `out = Σ inputs`.
pub struct AddProof {
    pub r: Vec<Fp>,
    pub b_out: Fp,
    pub claim_out: InnerProductProof,
    pub com_out: fri::Commitment,
    pub dom_out: fri::Domain,
    pub inputs: Vec<(fri::Commitment, fri::Domain, Fp, InnerProductProof)>,
}

/// mul: elementwise product.
pub struct MulOpProof {
    pub mul: MulProof,
}

/// reshape: `out == in` elementwise.
pub struct ReshapeProof {
    pub r: Vec<Fp>,
    pub b_in: Fp,
    pub b_out: Fp,
    pub claim_in: InnerProductProof,
    pub claim_out: InnerProductProof,
    pub com_in: fri::Commitment,
    pub dom_in: fri::Domain,
    pub com_out: fri::Commitment,
    pub dom_out: fri::Domain,
}

/// norm (per row): `out = round(x⊙gamma / sqrt(mean_sq+eps))`.
pub struct NormOpProof {
    pub sum_sq: i64,            // Σ x² (integer, drives inv_q)
    pub ipa: CommittedIpa,      // ⟨x,x⟩ = sum_sq
    pub mul: MulProof,          // h = x⊙gamma
    pub inv_q: i64,             // round(2^Q / sqrt(sum_sq/d + eps))
    pub com_y: fri::Commitment, // y = h·inv_q
    pub dom_y: fri::Domain,
    pub r: Vec<Fp>,
    pub b_y: Fp,
    pub b_h: Fp,
    pub claim_y: InnerProductProof,
    pub claim_h: InnerProductProof,
    pub req: RequantProof, // out = requant(y, Q)
    pub tie_x: RowTie,
    pub tie_out: RowTie,
}

/// rope (per (position, head)): `out = requant(R·x, 15)`.
pub struct RopeOpProof {
    pub com_rot: fri::Commitment,
    pub dom_rot: fri::Domain,
    pub r: Vec<Fp>,
    pub rt: Vec<Fp>,
    pub b_rot: Fp,
    pub b_inp: Fp,
    pub claim_rot: InnerProductProof,
    pub claim_inp: InnerProductProof,
    pub req: RequantProof,
    pub tie_in: RowTie,
    pub tie_out: RowTie,
}

/// scores (per (position, head)): `scores = round((K_h·q)/scale)`.
pub struct ScoresOpProof {
    pub mm: MatmulProof,
    pub inv_q: i64,             // round(2^Q / scale)
    pub com_y: fri::Commitment, // y = dot·inv_q
    pub dom_y: fri::Domain,
    pub r: Vec<Fp>,
    pub b_y: Fp,
    pub b_dot: Fp,
    pub claim_y: InnerProductProof,
    pub claim_dot: InnerProductProof,
    pub req: RequantProof, // scores = requant(y, Q)
    pub tie_k: RowTie,
    pub tie_q: RowTie,
    pub tie_out: RowTie,
}

/// weighted_v (per (position, head)): `out = requant(V_h^T·w, 15)`.
pub struct WeightedVOpProof {
    pub mm: MatmulProof,
    pub req: RequantProof,
    pub tie_v: RowTie,
    pub tie_w: RowTie,
    pub tie_out: RowTie,
}

/// causal_softmax (per (position, head) row).
pub struct CausalSoftmaxOpProof {
    pub cp: CausalSoftmaxProof,
    pub tie_in: RowTie,
    pub tie_out: RowTie,
}

/// top_k (per row).
pub struct TopKOpProof {
    pub tp: TopKProof,
    pub tie_in: RowTie,
    pub tie_choice: RowTie,
}

/// weighted_sum (per sequence position): `out[s] = Σ_e choice[s,e]·term_e[s]`.
pub struct WeightedSumOpProof {
    pub r: Vec<Fp>,
    pub b_out: Fp,
    pub claim_out: InnerProductProof,
    pub choice_b: Vec<Fp>,
    pub choice_claims: Vec<InnerProductProof>,
    pub term_bs: Vec<Fp>,
    pub term_claims: Vec<InnerProductProof>,
    pub tie_choice: RowTie,
    pub tie_out: RowTie,
    pub tie_terms: Vec<RowTie>,
}

pub enum OpProof {
    Embed(EmbedProof),
    Matmul(MatmulOpProof),
    Silu(SiluProof),
    Add(AddProof),
    Mul(MulOpProof),
    Reshape(ReshapeProof),
    Norm(NormOpProof),
    Rope(RopeOpProof),
    Scores(ScoresOpProof),
    WeightedV(WeightedVOpProof),
    CausalSoftmax(CausalSoftmaxOpProof),
    TopK(TopKOpProof),
    WeightedSum(WeightedSumOpProof),
}

pub struct SuccinctCommittedProof {
    pub ops: Vec<OpProof>,
    /// Commitments (weights then cells, in canonical order) absorbed to derive the
    /// Fiat–Shamir lookup challenges α, β, so the prover cannot adapt committed data
    /// to the challenges.
    pub binding: Vec<fri::Commitment>,
    pub head_logits: Vec<i64>,
    pub head_r: Vec<Fp>,
    pub head_b: Fp,
    pub head_claim: InnerProductProof,
    pub head_com: fri::Commitment,
    pub head_dom: fri::Domain,
    pub values: Vec<Ext>,
    pub fri_proof: fri::Proof,
}

// --- helpers ------------------------------------------------------------------

fn prove_batched_matmul(
    ctx: &mut Prover,
    w: &CellCommit,
    x: &CellCommit,
    acc: &CellCommit,
    batch: usize,
    out_dim: usize,
    in_dim: usize,
) -> (
    MatmulProof,
    Vec<Fp>,
    Fp,
    Fp,
    InnerProductProof,
    InnerProductProof,
) {
    // Wb[(b·out+r)][(b'·in+c)] = W[r][c] if b == b' else 0 (full block-diagonal).
    let out_rows = batch * out_dim;
    let in_cols = batch * in_dim;
    let mut wb = vec![0i64; out_rows * in_cols];
    for b in 0..batch {
        for r in 0..out_dim {
            for c in 0..in_dim {
                wb[(b * out_dim + r) * in_cols + (b * in_dim + c)] = w.values[r * in_dim + c];
            }
        }
    }
    let wb_fp: Vec<Fp> = wb.iter().map(|&v| f(v)).collect();
    let (com_wb, pd_wb, dom_wb) = fri::commit_vec(&ctx.pcs, &wb_fp);

    // Block-diagonal check: ⟨u, Wb⟩ = ⟨diag(u), W⟩.
    let mut tr = fri::Transcript::new(&ctx.challenger);
    tr.observe(&com_wb);
    tr.observe(&w.com);
    let u: Vec<Fp> = (0..out_rows * in_cols).map(|_| tr.sample_val()).collect();
    let bd_b_wb: Fp = u.iter().zip(&wb_fp).map(|(&ui, &vi)| ui * vi).sum();
    let mut u_agg = vec![Fp::ZERO; out_dim * in_dim];
    for b in 0..batch {
        for r in 0..out_dim {
            for c in 0..in_dim {
                u_agg[r * in_dim + c] += u[(b * out_dim + r) * in_cols + (b * in_dim + c)];
            }
        }
    }
    let bd_b_w: Fp = u_agg.iter().zip(&w.fp).map(|(&ui, &vi)| ui * vi).sum();
    let bd_claim_wb = prove_inner_product_committed_given(ctx, &com_wb, &pd_wb, dom_wb, &wb_fp, &u);
    let bd_claim_w = prove_inner_product_committed_given(ctx, &w.com, &w.pd, w.dom, &w.fp, &u_agg);

    // acc = Wb · x.
    let mm = prove_matmul_committed_given(
        ctx,
        &com_wb,
        &pd_wb,
        dom_wb,
        &x.com,
        &x.pd,
        x.dom,
        &acc.com,
        &acc.pd,
        acc.dom,
        &wb_fp,
        &x.fp,
        &acc.fp,
        batch * out_dim,
        batch * in_dim,
    );

    (mm, u, bd_b_wb, bd_b_w, bd_claim_wb, bd_claim_w)
}

fn verify_batched_matmul(
    challenger: &Challenger,
    com_w: &fri::Commitment,
    dom_w: fri::Domain,
    batch: usize,
    out_dim: usize,
    in_dim: usize,
    mm: &MatmulProof,
    u: &[Fp],
    bd_b_wb: Fp,
    bd_b_w: Fp,
    bd_claim_wb: &InnerProductProof,
    bd_claim_w: &InnerProductProof,
    values: &[Ext],
    acc: &mut Openings,
) -> bool {
    let in_cols = batch * in_dim;
    let mut u_agg = vec![Fp::ZERO; out_dim * in_dim];
    for b in 0..batch {
        for r in 0..out_dim {
            for c in 0..in_dim {
                u_agg[r * in_dim + c] += u[(b * out_dim + r) * in_cols + (b * in_dim + c)];
            }
        }
    }
    verify_inner_product_committed(
        challenger,
        &mm.com_w,
        mm.dom_w,
        u,
        bd_b_wb,
        bd_claim_wb,
        values,
        acc,
    ) && verify_inner_product_committed(
        challenger, com_w, dom_w, &u_agg, bd_b_w, bd_claim_w, values, acc,
    ) && verify_matmul_committed(challenger, batch * out_dim, batch * in_dim, mm, values, acc)
}

// --- row-slice tie ------------------------------------------------------------

/// Ties a committed row slice to the committed full cell it was sliced from:
/// `row[t] == full[start + t]` for all `t < d`, checked by one random linear
/// combination (Schwartz–Zippel). This is what makes per-(position, head) row
/// proofs sound against a whole-tensor commitment.
pub struct RowTie {
    pub r: Vec<Fp>,
    pub b_row: Fp,
    pub b_full: Fp,
    pub claim_row: InnerProductProof,
    pub claim_full: InnerProductProof,
    pub full_len: usize,
    pub row_com: fri::Commitment,
    pub row_dom: fri::Domain,
    pub full_com: fri::Commitment,
    pub full_dom: fri::Domain,
}

fn tie_row(ctx: &mut Prover, row: &CellCommit, full: &CellCommit, idx: &[usize]) -> RowTie {
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

fn verify_tie(
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

// --- prover -------------------------------------------------------------------

pub fn prove_succinct_committed(
    ctx: &mut Prover,
    spec: &ModelSpec,
    weights: &Weights,
    trace: &Trace,
    window: &[usize],
    cfg: &CommittedConfig,
) -> SuccinctCommittedProof {
    assert_eq!(trace.segments.len(), 1, "single-segment trace only");
    let (cells, wmap) = commit_cells(&ctx.pcs, weights, trace);
    let l = window.len();

    // Bind α, β to every committed weight and cell (canonical order) so the prover
    // cannot adapt committed data to the lookup challenges.
    let mut binding = Vec::new();
    for w in weights {
        binding.push(wmap[&w.name].com.clone());
    }
    for seg in &trace.segments {
        for c in &seg.cells {
            binding.push(cells[&c.name].com.clone());
        }
    }
    let (alpha, beta) = sample_ab(&ctx.challenger, &binding);
    let tables = build_tables(&ctx.pcs, cfg, alpha, beta, spec);
    let digit_table = commit_digit_table(&ctx.pcs);
    let bit_table = commit_bit_table(&ctx.pcs);

    let mut shapes: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut ops = Vec::new();

    for op in &spec.ops {
        match op {
            Op::Embed { weight, output } => {
                let emb = &wmap[weight.as_str()];
                let out = &cells[output.as_str()];
                let (vocab, hidden) = (emb.rows, emb.cols);
                let n = l * hidden;
                let mut tr = fri::Transcript::new(&ctx.challenger);
                tr.observe(&out.com);
                tr.observe(&emb.com);
                let r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
                let b_out: Fp = r.iter().zip(&out.fp).map(|(&ri, &vi)| ri * vi).sum();
                let mut h_emb = vec![Fp::ZERO; vocab * hidden];
                for b in 0..l {
                    for j in 0..hidden {
                        h_emb[window[b] * hidden + j] += r[b * hidden + j];
                    }
                }
                let b_emb: Fp = h_emb.iter().zip(&emb.fp).map(|(&hi, &vi)| hi * vi).sum();
                let claim_out = prove_inner_product_committed_given(
                    ctx, &out.com, &out.pd, out.dom, &out.fp, &r,
                );
                let claim_emb = prove_inner_product_committed_given(
                    ctx, &emb.com, &emb.pd, emb.dom, &emb.fp, &h_emb,
                );
                ops.push(OpProof::Embed(EmbedProof {
                    com_out: out.com.clone(),
                    dom_out: out.dom,
                    com_emb: emb.com.clone(),
                    dom_emb: emb.dom,
                    r,
                    b_out,
                    b_emb,
                    claim_out,
                    claim_emb,
                    l,
                    hidden,
                    vocab,
                }));
                shapes.insert(output.as_str(), vec![l, hidden]);
            }
            Op::Matmul {
                input,
                weight,
                output,
                shift,
            } => {
                let w = &wmap[weight.as_str()];
                let out_dim = w.rows;
                let in_dim = w.cols;
                let x = &cells[input.as_str()];
                let out = &cells[output.as_str()];
                let batch = x.rows;
                assert_eq!(x.cols, in_dim);
                assert_eq!(out.rows, batch);
                assert_eq!(out.cols, out_dim);

                // acc = W·x (clear), committed fresh.
                let mut acc = vec![0i64; batch * out_dim];
                for b in 0..batch {
                    for r in 0..out_dim {
                        let a: i64 = (0..in_dim)
                            .map(|c| w.values[r * in_dim + c] as i64 * x.values[b * in_dim + c])
                            .sum();
                        acc[b * out_dim + r] = a;
                    }
                }
                let acc_cell = commit_one(&ctx.pcs, "__acc", batch, out_dim, acc.clone());

                let (mm, u, bd_b_wb, bd_b_w, bd_claim_wb, bd_claim_w) =
                    prove_batched_matmul(ctx, w, x, &acc_cell, batch, out_dim, in_dim);

                let out_vals: Vec<i64> = acc.iter().map(|&a| requant(a, *shift)).collect();
                let req = prove_requant_given(
                    ctx,
                    &acc_cell.com,
                    &acc_cell.pd,
                    acc_cell.dom,
                    &acc,
                    &out_vals,
                    *shift,
                    cfg.requant_digits,
                    &digit_table,
                    &bit_table,
                );
                ops.push(OpProof::Matmul(MatmulOpProof {
                    mm,
                    req,
                    com_w: w.com.clone(),
                    dom_w: w.dom,
                    bd_u: u,
                    bd_b_wb,
                    bd_b_w,
                    bd_claim_wb,
                    bd_claim_w,
                    shift: *shift,
                }));

                let mut s = shapes[input.as_str()].clone();
                let n = s.len();
                s[n - 1] = out_dim;
                shapes.insert(output.as_str(), s);
            }
            Op::Silu { input, output } => {
                let x = &cells[input.as_str()];
                let out = &cells[output.as_str()];
                let out_vals: Vec<i64> = x.values.iter().map(|&v| silu(v)).collect();
                let lk = prove_lookup2_logup_given(
                    ctx,
                    &x.com,
                    &x.pd,
                    x.dom,
                    &out.com,
                    &out.pd,
                    out.dom,
                    &x.values,
                    &out_vals,
                    &tables.silu,
                    alpha,
                    beta,
                );
                ops.push(OpProof::Silu(SiluProof { lk }));
                shapes.insert(output.as_str(), shapes[input.as_str()].clone());
            }
            Op::Add { inputs, output } => {
                let out = &cells[output.as_str()];
                let n = out.values.len();
                let mut tr = fri::Transcript::new(&ctx.challenger);
                tr.observe(&out.com);
                for inp in inputs {
                    tr.observe(&cells[inp.as_str()].com);
                }
                let r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
                let b_out: Fp = r.iter().zip(&out.fp).map(|(&ri, &vi)| ri * vi).sum();
                let claim_out = prove_inner_product_committed_given(
                    ctx, &out.com, &out.pd, out.dom, &out.fp, &r,
                );
                let mut ins = Vec::new();
                for inp in inputs {
                    let c = &cells[inp.as_str()];
                    let bi: Fp = r.iter().zip(&c.fp).map(|(&ri, &vi)| ri * vi).sum();
                    let cl =
                        prove_inner_product_committed_given(ctx, &c.com, &c.pd, c.dom, &c.fp, &r);
                    ins.push((c.com.clone(), c.dom, bi, cl));
                }
                ops.push(OpProof::Add(AddProof {
                    r,
                    b_out,
                    claim_out,
                    com_out: out.com.clone(),
                    dom_out: out.dom,
                    inputs: ins,
                }));
                shapes.insert(output.as_str(), shapes[inputs[0].as_str()].clone());
            }
            Op::Mul { a, b, output } => {
                let ca = &cells[a.as_str()];
                let cb = &cells[b.as_str()];
                let co = &cells[output.as_str()];
                let out_fp: Vec<Fp> = ca
                    .values
                    .iter()
                    .zip(&cb.values)
                    .map(|(&x, &y)| f(x * y))
                    .collect();
                let out_fp = pad(out_fp, ca.values.len());
                let mul = prove_mul_committed_given(
                    ctx, &ca.com, &ca.pd, ca.dom, &cb.com, &cb.pd, cb.dom, &co.com, &co.pd, co.dom,
                    &ca.fp, &cb.fp, &out_fp,
                );
                ops.push(OpProof::Mul(MulOpProof { mul }));
                shapes.insert(output.as_str(), shapes[a.as_str()].clone());
            }
            Op::Reshape {
                input,
                output,
                shape,
            } => {
                let cin = &cells[input.as_str()];
                let cout = &cells[output.as_str()];
                let n = cin.values.len();
                let mut tr = fri::Transcript::new(&ctx.challenger);
                tr.observe(&cin.com);
                tr.observe(&cout.com);
                let r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
                let b_in: Fp = r.iter().zip(&cin.fp).map(|(&ri, &vi)| ri * vi).sum();
                let b_out: Fp = r.iter().zip(&cout.fp).map(|(&ri, &vi)| ri * vi).sum();
                let claim_in = prove_inner_product_committed_given(
                    ctx, &cin.com, &cin.pd, cin.dom, &cin.fp, &r,
                );
                let claim_out = prove_inner_product_committed_given(
                    ctx, &cout.com, &cout.pd, cout.dom, &cout.fp, &r,
                );
                ops.push(OpProof::Reshape(ReshapeProof {
                    r,
                    b_in,
                    b_out,
                    claim_in,
                    claim_out,
                    com_in: cin.com.clone(),
                    dom_in: cin.dom,
                    com_out: cout.com.clone(),
                    dom_out: cout.dom,
                }));
                let ish = shapes[input.as_str()].clone();
                let numel: usize = ish.iter().product();
                let neg = shape.iter().filter(|&&s| s == -1).count();
                let known: usize = shape
                    .iter()
                    .filter(|&&s| s != -1)
                    .map(|&s| s as usize)
                    .product();
                let inferred = if neg == 1 { numel / known } else { 1 };
                let resolved: Vec<usize> = shape
                    .iter()
                    .map(|&s| if s == -1 { inferred } else { s as usize })
                    .collect();
                shapes.insert(output.as_str(), resolved);
            }
            Op::Norm {
                input,
                weight,
                output,
                eps,
            } => {
                let g = &wmap[weight.as_str()];
                let x = &cells[input.as_str()];
                let out = &cells[output.as_str()];
                let rows = x.rows;
                let hidden = x.cols;
                let gamma: Vec<i64> = g.values.clone();
                for b in 0..rows {
                    let xb: Vec<i64> = x.values[b * hidden..(b + 1) * hidden].to_vec();
                    let ob: Vec<i64> = out.values[b * hidden..(b + 1) * hidden].to_vec();
                    let xrow = commit_one(&ctx.pcs, "__norm_x", 1, hidden, xb.clone());
                    let orow = commit_one(&ctx.pcs, "__norm_o", 1, hidden, ob.clone());
                    let sum_sq: i64 = xb.iter().map(|&v| v * v).sum();
                    let inv_q = ((1i64 << RECIP_Q) as f64
                        / ((sum_sq as f64 / hidden as f64) + eps).sqrt())
                    .round() as i64;
                    let ipa = prove_ipa_committed_given(
                        ctx,
                        &xrow.com,
                        &xrow.pd,
                        xrow.dom,
                        &xrow.com,
                        &xrow.pd,
                        xrow.dom,
                        &xrow.fp,
                        &xrow.fp,
                        f(sum_sq),
                    );
                    let h_vals: Vec<i64> =
                        xb.iter().zip(&gamma).map(|(&xi, &gi)| xi * gi).collect();
                    let h = commit_one(&ctx.pcs, "__norm_h", 1, hidden, h_vals.clone());
                    let mul = prove_mul_committed_given(
                        ctx, &xrow.com, &xrow.pd, xrow.dom, &g.com, &g.pd, g.dom, &h.com, &h.pd,
                        h.dom, &xrow.fp, &g.fp, &h.fp,
                    );
                    // y = h·inv_q.
                    let y_vals: Vec<i64> = h_vals.iter().map(|&hi| hi * inv_q).collect();
                    let y = commit_one(&ctx.pcs, "__norm_y", 1, hidden, y_vals.clone());
                    let mut tr = fri::Transcript::new(&ctx.challenger);
                    tr.observe(&y.com);
                    tr.observe(&h.com);
                    let mut r: Vec<Fp> = (0..hidden).map(|_| tr.sample_val()).collect();
                    r.resize(next_pow2(hidden), Fp::ZERO);
                    let b_y: Fp = r.iter().zip(&y.fp).map(|(&ri, &vi)| ri * vi).sum();
                    let b_h: Fp = r.iter().zip(&h.fp).map(|(&ri, &vi)| ri * vi).sum();
                    let claim_y =
                        prove_inner_product_committed_given(ctx, &y.com, &y.pd, y.dom, &y.fp, &r);
                    let claim_h =
                        prove_inner_product_committed_given(ctx, &h.com, &h.pd, h.dom, &h.fp, &r);
                    let req = prove_requant_given(
                        ctx,
                        &y.com,
                        &y.pd,
                        y.dom,
                        &y_vals,
                        &ob,
                        RECIP_Q,
                        cfg.requant_digits,
                        &digit_table,
                        &bit_table,
                    );
                    let idx: Vec<usize> = (b * hidden..(b + 1) * hidden).collect();
                    let tie_x = tie_row(ctx, &xrow, x, &idx);
                    let tie_out = tie_row(ctx, &orow, out, &idx);
                    ops.push(OpProof::Norm(NormOpProof {
                        sum_sq,
                        ipa,
                        mul,
                        inv_q,
                        com_y: y.com.clone(),
                        dom_y: y.dom,
                        r,
                        b_y,
                        b_h,
                        claim_y,
                        claim_h,
                        req,
                        tie_x,
                        tie_out,
                    }));
                }
                shapes.insert(output.as_str(), shapes[input.as_str()].clone());
            }
            Op::Rope {
                input,
                output,
                theta,
            } => {
                let sh = shapes[input.as_str()].clone(); // [L,H,d]
                let (lq, h, d) = (sh[0], sh[1], sh[2]);
                let x = &cells[input.as_str()];
                let y = &cells[output.as_str()];
                for li in 0..lq {
                    for hi in 0..h {
                        let base = (li * h + hi) * d;
                        let xb: Vec<i64> = x.values[base..base + d].to_vec();
                        let yb: Vec<i64> = y.values[base..base + d].to_vec();
                        let xrow = commit_one(&ctx.pcs, "__rope_x", 1, d, xb.clone());
                        let yrow = commit_one(&ctx.pcs, "__rope_y", 1, d, yb.clone());
                        let mut rot = vec![0i64; d];
                        for j in 0..(d / 2) {
                            let i = 2 * j;
                            let angle = (li as f64) * theta.powf(-(i as f64) / d as f64);
                            let c = (angle.cos() * 32768.0).round() as i64;
                            let s = (angle.sin() * 32768.0).round() as i64;
                            rot[i] = xb[i] * c - xb[i + 1] * s;
                            rot[i + 1] = xb[i] * s + xb[i + 1] * c;
                        }
                        let rot_c = commit_one(&ctx.pcs, "__rope_rot", 1, d, rot.clone());
                        let mut tr = fri::Transcript::new(&ctx.challenger);
                        tr.observe(&rot_c.com);
                        tr.observe(&xrow.com);
                        let r: Vec<Fp> = (0..d).map(|_| tr.sample_val()).collect();
                        let mut rt = vec![Fp::ZERO; d];
                        for j in 0..(d / 2) {
                            let i = 2 * j;
                            let angle = (li as f64) * theta.powf(-(i as f64) / d as f64);
                            let c = f((angle.cos() * 32768.0).round() as i64);
                            let s = f((angle.sin() * 32768.0).round() as i64);
                            rt[i] = r[i] * c + r[i + 1] * s;
                            rt[i + 1] = -r[i] * s + r[i + 1] * c;
                        }
                        let b_rot: Fp = r.iter().zip(&rot_c.fp).map(|(&ri, &vi)| ri * vi).sum();
                        let b_inp: Fp = rt.iter().zip(&xrow.fp).map(|(&ri, &vi)| ri * vi).sum();
                        let claim_rot = prove_inner_product_committed_given(
                            ctx, &rot_c.com, &rot_c.pd, rot_c.dom, &rot_c.fp, &r,
                        );
                        let claim_inp = prove_inner_product_committed_given(
                            ctx, &xrow.com, &xrow.pd, xrow.dom, &xrow.fp, &rt,
                        );
                        let req = prove_requant_given(
                            ctx,
                            &rot_c.com,
                            &rot_c.pd,
                            rot_c.dom,
                            &rot,
                            &yb,
                            15,
                            cfg.requant_digits,
                            &digit_table,
                            &bit_table,
                        );
                        let idx: Vec<usize> = (base..base + d).collect();
                        let tie_in = tie_row(ctx, &xrow, x, &idx);
                        let tie_out = tie_row(ctx, &yrow, y, &idx);
                        ops.push(OpProof::Rope(RopeOpProof {
                            com_rot: rot_c.com.clone(),
                            dom_rot: rot_c.dom,
                            r,
                            rt,
                            b_rot,
                            b_inp,
                            claim_rot,
                            claim_inp,
                            req,
                            tie_in,
                            tie_out,
                        }));
                    }
                }
                shapes.insert(output.as_str(), sh);
            }
            Op::Scores {
                q,
                k,
                output,
                scale,
            } => {
                let qsh = shapes[q.as_str()].clone();
                let (lq, h, d) = (qsh[0], qsh[1], qsh[2]);
                let qc = &cells[q.as_str()];
                let kc = &cells[k.as_str()];
                let sc = &cells[output.as_str()];
                for li in 0..lq {
                    for hi in 0..h {
                        let mut kh = Vec::with_capacity(lq * d);
                        let mut kh_idx = Vec::with_capacity(lq * d);
                        for t in 0..lq {
                            for j in 0..d {
                                kh.push(kc.values[(t * h + hi) * d + j]);
                                kh_idx.push((t * h + hi) * d + j);
                            }
                        }
                        let qlh: Vec<i64> =
                            (0..d).map(|j| qc.values[(li * h + hi) * d + j]).collect();
                        let dot: Vec<i64> = (0..lq)
                            .map(|t| (0..d).map(|j| kh[t * d + j] * qlh[j]).sum())
                            .collect();
                        let srow: Vec<i64> =
                            (0..lq).map(|t| sc.values[(li * h + hi) * lq + t]).collect();
                        let kh_c = commit_one(&ctx.pcs, "__scores_k", lq, d, kh.clone());
                        let qlh_c = commit_one(&ctx.pcs, "__scores_q", 1, d, qlh.clone());
                        let dot_c = commit_one(&ctx.pcs, "__scores_dot", 1, lq, dot.clone());
                        let srow_c = commit_one(&ctx.pcs, "__scores_s", 1, lq, srow.clone());
                        let mm = prove_matmul_committed_given(
                            ctx, &kh_c.com, &kh_c.pd, kh_c.dom, &qlh_c.com, &qlh_c.pd, qlh_c.dom,
                            &dot_c.com, &dot_c.pd, dot_c.dom, &kh_c.fp, &qlh_c.fp, &dot_c.fp, lq,
                            d,
                        );
                        // y = dot·inv_q, scores = requant(y, Q).
                        let inv_q = ((1i64 << RECIP_Q) as f64 / scale).round() as i64;
                        let y_vals: Vec<i64> = dot.iter().map(|&d| d * inv_q).collect();
                        let y = commit_one(&ctx.pcs, "__scores_y", 1, lq, y_vals.clone());
                        let mut tr = fri::Transcript::new(&ctx.challenger);
                        tr.observe(&y.com);
                        tr.observe(&dot_c.com);
                        let mut r: Vec<Fp> = (0..lq).map(|_| tr.sample_val()).collect();
                        r.resize(next_pow2(lq), Fp::ZERO);
                        let b_y: Fp = r.iter().zip(&y.fp).map(|(&ri, &vi)| ri * vi).sum();
                        let b_dot: Fp = r.iter().zip(&dot_c.fp).map(|(&ri, &vi)| ri * vi).sum();
                        let claim_y = prove_inner_product_committed_given(
                            ctx, &y.com, &y.pd, y.dom, &y.fp, &r,
                        );
                        let claim_dot = prove_inner_product_committed_given(
                            ctx, &dot_c.com, &dot_c.pd, dot_c.dom, &dot_c.fp, &r,
                        );
                        let req = prove_requant_given(
                            ctx,
                            &y.com,
                            &y.pd,
                            y.dom,
                            &y_vals,
                            &srow,
                            RECIP_Q,
                            cfg.requant_digits,
                            &digit_table,
                            &bit_table,
                        );
                        let q_idx: Vec<usize> =
                            ((li * h + hi) * d..(li * h + hi) * d + d).collect();
                        let s_idx: Vec<usize> =
                            ((li * h + hi) * lq..(li * h + hi) * lq + lq).collect();
                        let tie_k = tie_row(ctx, &kh_c, kc, &kh_idx);
                        let tie_q = tie_row(ctx, &qlh_c, qc, &q_idx);
                        let tie_out = tie_row(ctx, &srow_c, sc, &s_idx);
                        ops.push(OpProof::Scores(ScoresOpProof {
                            mm,
                            inv_q,
                            com_y: y.com.clone(),
                            dom_y: y.dom,
                            r,
                            b_y,
                            b_dot,
                            claim_y,
                            claim_dot,
                            req,
                            tie_k,
                            tie_q,
                            tie_out,
                        }));
                    }
                }
                shapes.insert(output.as_str(), vec![lq, h, lq]);
            }
            Op::CausalSoftmax { input, output } => {
                let sh = shapes[input.as_str()].clone(); // [L,H,L]
                let (lq, h, lseq) = (sh[0], sh[1], sh[2]);
                let x = &cells[input.as_str()];
                let y = &cells[output.as_str()];
                for li in 0..lq {
                    for hi in 0..h {
                        let base = (li * h + hi) * lseq;
                        let srow: Vec<i64> = (0..lseq).map(|t| x.values[base + t]).collect();
                        let prow: Vec<i64> = (0..lseq).map(|t| y.values[base + t]).collect();
                        let xrow = commit_one(&ctx.pcs, "__cs_x", 1, lseq, srow.clone());
                        let yrow = commit_one(&ctx.pcs, "__cs_y", 1, lseq, prow.clone());
                        let cp = prove_causal_softmax_committed_given(
                            ctx,
                            &xrow.com,
                            &xrow.pd,
                            xrow.dom,
                            &yrow.com,
                            &yrow.pd,
                            yrow.dom,
                            &srow,
                            &prow,
                            li,
                            &tables.exp,
                            &digit_table,
                            &bit_table,
                            alpha,
                            beta,
                        );
                        let idx: Vec<usize> = (base..base + lseq).collect();
                        let tie_in = tie_row(ctx, &xrow, x, &idx);
                        let tie_out = tie_row(ctx, &yrow, y, &idx);
                        ops.push(OpProof::CausalSoftmax(CausalSoftmaxOpProof {
                            cp,
                            tie_in,
                            tie_out,
                        }));
                    }
                }
                shapes.insert(output.as_str(), sh);
            }
            Op::WeightedV {
                weights: wname,
                v,
                output,
            } => {
                let wsh = shapes[wname.as_str()].clone(); // [L,H,L]
                let vsh = shapes[v.as_str()].clone(); // [L,H,d]
                let (lq, h, d) = (vsh[0], vsh[1], vsh[2]);
                let _ = wsh;
                let wc = &cells[wname.as_str()];
                let vc = &cells[v.as_str()];
                let oc = &cells[output.as_str()];
                for li in 0..lq {
                    for hi in 0..h {
                        // V_h^T[j][t] = v[t][h][j].
                        let mut vh = Vec::with_capacity(d * lq);
                        let mut vh_idx = Vec::with_capacity(d * lq);
                        for j in 0..d {
                            for t in 0..lq {
                                vh.push(vc.values[(t * h + hi) * d + j]);
                                vh_idx.push((t * h + hi) * d + j);
                            }
                        }
                        let wlh: Vec<i64> =
                            (0..lq).map(|t| wc.values[(li * h + hi) * lq + t]).collect();
                        let acc: Vec<i64> = (0..d)
                            .map(|j| (0..lq).map(|t| vh[j * lq + t] * wlh[t]).sum())
                            .collect();
                        let ob: Vec<i64> =
                            (0..d).map(|j| oc.values[(li * h + hi) * d + j]).collect();
                        let vh_c = commit_one(&ctx.pcs, "__wv_v", d, lq, vh.clone());
                        let wlh_c = commit_one(&ctx.pcs, "__wv_w", 1, lq, wlh.clone());
                        let acc_c = commit_one(&ctx.pcs, "__wv_acc", 1, d, acc.clone());
                        let ob_c = commit_one(&ctx.pcs, "__wv_o", 1, d, ob.clone());
                        let mm = prove_matmul_committed_given(
                            ctx, &vh_c.com, &vh_c.pd, vh_c.dom, &wlh_c.com, &wlh_c.pd, wlh_c.dom,
                            &acc_c.com, &acc_c.pd, acc_c.dom, &vh_c.fp, &wlh_c.fp, &acc_c.fp, d,
                            lq,
                        );
                        let req = prove_requant_given(
                            ctx,
                            &acc_c.com,
                            &acc_c.pd,
                            acc_c.dom,
                            &acc,
                            &ob,
                            15,
                            cfg.requant_digits,
                            &digit_table,
                            &bit_table,
                        );
                        let w_idx: Vec<usize> =
                            ((li * h + hi) * lq..(li * h + hi) * lq + lq).collect();
                        let o_idx: Vec<usize> =
                            ((li * h + hi) * d..(li * h + hi) * d + d).collect();
                        let tie_v = tie_row(ctx, &vh_c, vc, &vh_idx);
                        let tie_w = tie_row(ctx, &wlh_c, wc, &w_idx);
                        let tie_out = tie_row(ctx, &ob_c, oc, &o_idx);
                        ops.push(OpProof::WeightedV(WeightedVOpProof {
                            mm,
                            req,
                            tie_v,
                            tie_w,
                            tie_out,
                        }));
                    }
                }
                shapes.insert(output.as_str(), vec![lq, h, d]);
            }
            Op::TopK {
                input,
                output,
                choice,
                k,
            } => {
                let x = &cells[input.as_str()];
                let c = &cells[choice.as_str()];
                let rows = x.rows;
                let e = x.cols;
                for b in 0..rows {
                    let xb: Vec<i64> = x.values[b * e..(b + 1) * e].to_vec();
                    let cb: Vec<i64> = c.values[b * e..(b + 1) * e].to_vec();
                    let mut adj: Vec<i64> = (0..e).map(|i| xb[i] * e as i64 - i as i64).collect();
                    adj.sort_unstable_by(|a, b| b.cmp(a));
                    // Threshold strictly below the (k+1)-th largest adjusted score, so
                    // exactly the top-k have `adjusted ≥ tau` (δ ≥ 0).
                    let tau = if *k < e { adj[*k] + 1 } else { adj[e - 1] - 1 };
                    let xrow = commit_one(&ctx.pcs, "__topk_x", 1, e, xb.clone());
                    let crow = commit_one(&ctx.pcs, "__topk_c", 1, e, cb.clone());
                    let tp = prove_topk_committed_given(
                        ctx,
                        &xrow.com,
                        &xrow.pd,
                        xrow.dom,
                        &crow.com,
                        &crow.pd,
                        crow.dom,
                        &xb,
                        &cb,
                        *k,
                        tau,
                        &tables.sign,
                        alpha,
                        beta,
                    );
                    let idx: Vec<usize> = (b * e..(b + 1) * e).collect();
                    let tie_in = tie_row(ctx, &xrow, x, &idx);
                    let tie_choice = tie_row(ctx, &crow, c, &idx);
                    ops.push(OpProof::TopK(TopKOpProof {
                        tp,
                        tie_in,
                        tie_choice,
                    }));
                }
                let _ = output;
                shapes.insert(output.as_str(), shapes[input.as_str()].clone());
                shapes.insert(choice.as_str(), shapes[input.as_str()].clone());
            }
            Op::WeightedSum {
                choice,
                terms,
                output,
            } => {
                let csh = shapes[choice.as_str()].clone(); // [L,E]
                let seq = csh[0];
                let experts = csh[1];
                let cc = &cells[choice.as_str()];
                let oc = &cells[output.as_str()];
                let hidden = oc.cols;
                for s in 0..seq {
                    let cs: Vec<i64> = cc.values[s * experts..(s + 1) * experts].to_vec();
                    let os: Vec<i64> = oc.values[s * hidden..(s + 1) * hidden].to_vec();
                    let crow = commit_one(&ctx.pcs, "__ws_c", 1, experts, cs.clone());
                    let orow = commit_one(&ctx.pcs, "__ws_o", 1, hidden, os.clone());
                    // sample r over hidden.
                    let mut tr = fri::Transcript::new(&ctx.challenger);
                    tr.observe(&orow.com);
                    for t in terms {
                        tr.observe(&cells[t.as_str()].com);
                    }
                    tr.observe(&crow.com);
                    let r: Vec<Fp> = (0..hidden).map(|_| tr.sample_val()).collect();
                    let b_out: Fp = r.iter().zip(&orow.fp).map(|(&ri, &vi)| ri * vi).sum();
                    let claim_out = prove_inner_product_committed_given(
                        ctx, &orow.com, &orow.pd, orow.dom, &orow.fp, &r,
                    );
                    // for each expert e: t_e = ⟨r, term_e[s]⟩.
                    let mut t_vals = Vec::with_capacity(experts);
                    let mut term_bs = Vec::new();
                    let mut term_claims = Vec::new();
                    let mut tie_terms = Vec::new();
                    for tname in terms {
                        let tcell = &cells[tname.as_str()];
                        let te: Vec<i64> = tcell.values[s * hidden..(s + 1) * hidden].to_vec();
                        let trow = commit_one(&ctx.pcs, "__ws_t", 1, hidden, te.clone());
                        let be: Fp = r.iter().zip(&trow.fp).map(|(&ri, &vi)| ri * vi).sum();
                        let cl = prove_inner_product_committed_given(
                            ctx, &trow.com, &trow.pd, trow.dom, &trow.fp, &r,
                        );
                        t_vals.push(be);
                        term_bs.push(be);
                        term_claims.push(cl);
                        let idx: Vec<usize> = (s * hidden..(s + 1) * hidden).collect();
                        tie_terms.push(tie_row(ctx, &trow, tcell, &idx));
                    }
                    // b_choice = ⟨t, choice_s⟩.
                    let b_choice: Fp = t_vals.iter().zip(&crow.fp).map(|(&ti, &vi)| ti * vi).sum();
                    let choice_claim = prove_inner_product_committed_given(
                        ctx, &crow.com, &crow.pd, crow.dom, &crow.fp, &t_vals,
                    );
                    let c_idx: Vec<usize> = (s * experts..(s + 1) * experts).collect();
                    let o_idx: Vec<usize> = (s * hidden..(s + 1) * hidden).collect();
                    let tie_choice = tie_row(ctx, &crow, cc, &c_idx);
                    let tie_out = tie_row(ctx, &orow, oc, &o_idx);
                    ops.push(OpProof::WeightedSum(WeightedSumOpProof {
                        r,
                        b_out,
                        claim_out,
                        choice_b: vec![b_choice],
                        choice_claims: vec![choice_claim],
                        term_bs,
                        term_claims,
                        tie_choice,
                        tie_out,
                        tie_terms,
                    }));
                }
                shapes.insert(output.as_str(), vec![seq, hidden]);
            }
        }
    }

    // Head check: reveal the last logits row, tied to the committed logits.
    let logits = &cells["logits"];
    let last = logits.rows - 1;
    let cols = logits.cols;
    let head_logits: Vec<i64> = logits.values[last * cols..(last + 1) * cols].to_vec();
    let mut tr = fri::Transcript::new(&ctx.challenger);
    tr.observe(&logits.com);
    let mut head_r: Vec<Fp> = (0..cols).map(|_| tr.sample_val()).collect();
    head_r.resize(next_pow2(cols), Fp::ZERO);
    let mut hrow = vec![Fp::ZERO; next_pow2(logits.values.len())];
    for j in 0..cols {
        hrow[last * cols + j] = head_r[j];
    }
    let head_b: Fp = hrow.iter().zip(&logits.fp).map(|(&hi, &vi)| hi * vi).sum();
    let head_claim = prove_inner_product_committed_given(
        ctx,
        &logits.com,
        &logits.pd,
        logits.dom,
        &logits.fp,
        &hrow,
    );

    let fri_proof = ctx.open_all();
    let values = ctx.values.clone();

    SuccinctCommittedProof {
        ops,
        binding,
        head_logits,
        head_r,
        head_b,
        head_claim,
        head_com: logits.com.clone(),
        head_dom: logits.dom,
        values,
        fri_proof,
    }
}

// --- verifier -----------------------------------------------------------------

pub fn verify_succinct_committed(
    pcs: &Pcs,
    challenger: &Challenger,
    spec: &ModelSpec,
    window: &[usize],
    head_token: Option<usize>,
    cfg: &CommittedConfig,
    proof: &SuccinctCommittedProof,
) -> bool {
    let values = &proof.values;
    let fri_proof = &proof.fri_proof;
    let mut acc = crate::zk::Openings::new();
    let l = window.len();
    let (alpha, beta) = sample_ab(challenger, &proof.binding);
    let tables = build_tables(pcs, cfg, alpha, beta, spec);
    let digit_table = commit_digit_table(pcs);
    let bit_table = commit_bit_table(pcs);

    // weight name -> (rows, cols).
    let wdims: HashMap<&str, (usize, usize)> = spec
        .weights
        .iter()
        .map(|w| (w.name.as_str(), (w.rows, w.cols)))
        .collect();

    let mut shapes: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut idx = 0usize;

    for op in &spec.ops {
        match op {
            Op::Embed { weight, output } => {
                let p = match &proof.ops[idx] {
                    OpProof::Embed(p) => p,
                    _ => return false,
                };
                idx += 1;
                let (vocab, hidden) = wdims[weight.as_str()];
                if !verify_inner_product_committed(
                    challenger,
                    &p.com_out,
                    p.dom_out,
                    &p.r,
                    p.b_out,
                    &p.claim_out,
                    values,
                    &mut acc,
                ) {
                    return false;
                }
                let mut h_emb = vec![Fp::ZERO; vocab * hidden];
                for b in 0..l {
                    for j in 0..hidden {
                        h_emb[window[b] * hidden + j] += p.r[b * hidden + j];
                    }
                }
                if !verify_inner_product_committed(
                    challenger,
                    &p.com_emb,
                    p.dom_emb,
                    &h_emb,
                    p.b_emb,
                    &p.claim_emb,
                    values,
                    &mut acc,
                ) {
                    return false;
                }
                shapes.insert(output.as_str(), vec![l, hidden]);
            }
            Op::Matmul {
                input,
                weight,
                output,
                shift,
            } => {
                let p = match &proof.ops[idx] {
                    OpProof::Matmul(p) => p,
                    _ => return false,
                };
                idx += 1;
                let (out_dim, in_dim) = wdims[weight.as_str()];
                let batch = shapes[input.as_str()][0];
                if !verify_batched_matmul(
                    challenger,
                    &p.com_w,
                    p.dom_w,
                    batch,
                    out_dim,
                    in_dim,
                    &p.mm,
                    &p.bd_u,
                    p.bd_b_wb,
                    p.bd_b_w,
                    &p.bd_claim_wb,
                    &p.bd_claim_w,
                    values,
                    &mut acc,
                ) {
                    return false;
                }
                if !verify_requant(
                    challenger,
                    &digit_table,
                    &bit_table,
                    &p.req,
                    values,
                    &mut acc,
                ) {
                    return false;
                }
                let mut s = shapes[input.as_str()].clone();
                let n = s.len();
                s[n - 1] = out_dim;
                shapes.insert(output.as_str(), s);
                let _ = shift;
            }
            Op::Silu { input, output } => {
                let p = match &proof.ops[idx] {
                    OpProof::Silu(p) => p,
                    _ => return false,
                };
                idx += 1;
                if !verify_lookup2_logup(
                    challenger,
                    &tables.silu.table.com_ft,
                    tables.silu.table.dom_ft,
                    &p.lk,
                    values,
                    &mut acc,
                ) {
                    return false;
                }
                shapes.insert(output.as_str(), shapes[input.as_str()].clone());
            }
            Op::Add { inputs, output } => {
                let p = match &proof.ops[idx] {
                    OpProof::Add(p) => p,
                    _ => return false,
                };
                idx += 1;
                if !verify_inner_product_committed(
                    challenger,
                    &p.com_out,
                    p.dom_out,
                    &p.r,
                    p.b_out,
                    &p.claim_out,
                    values,
                    &mut acc,
                ) {
                    return false;
                }
                let mut sum = Fp::ZERO;
                for (com, dom, b, claim) in &p.inputs {
                    if !verify_inner_product_committed(
                        challenger, com, *dom, &p.r, *b, claim, values, &mut acc,
                    ) {
                        return false;
                    }
                    sum += *b;
                }
                if p.b_out != sum {
                    return false;
                }
                shapes.insert(output.as_str(), shapes[inputs[0].as_str()].clone());
            }
            Op::Mul { a, b, output } => {
                let p = match &proof.ops[idx] {
                    OpProof::Mul(p) => p,
                    _ => return false,
                };
                idx += 1;
                if !verify_mul_committed(challenger, &p.mul, values, &mut acc) {
                    return false;
                }
                shapes.insert(output.as_str(), shapes[a.as_str()].clone());
                let _ = b;
            }
            Op::Reshape {
                input,
                output,
                shape,
            } => {
                let p = match &proof.ops[idx] {
                    OpProof::Reshape(p) => p,
                    _ => return false,
                };
                idx += 1;
                if !verify_inner_product_committed(
                    challenger,
                    &p.com_in,
                    p.dom_in,
                    &p.r,
                    p.b_in,
                    &p.claim_in,
                    values,
                    &mut acc,
                ) || !verify_inner_product_committed(
                    challenger,
                    &p.com_out,
                    p.dom_out,
                    &p.r,
                    p.b_out,
                    &p.claim_out,
                    values,
                    &mut acc,
                ) || p.b_in != p.b_out
                {
                    return false;
                }
                let ish = shapes[input.as_str()].clone();
                let numel: usize = ish.iter().product();
                let neg = shape.iter().filter(|&&s| s == -1).count();
                let known: usize = shape
                    .iter()
                    .filter(|&&s| s != -1)
                    .map(|&s| s as usize)
                    .product();
                let inferred = if neg == 1 { numel / known } else { 1 };
                let resolved: Vec<usize> = shape
                    .iter()
                    .map(|&s| if s == -1 { inferred } else { s as usize })
                    .collect();
                shapes.insert(output.as_str(), resolved);
            }
            Op::Norm {
                input, output, eps, ..
            } => {
                let sh = shapes[input.as_str()].clone();
                let (rows, hidden) = (sh[0], sh[1]);
                for b in 0..rows {
                    let p = match &proof.ops[idx] {
                        OpProof::Norm(p) => p,
                        _ => return false,
                    };
                    idx += 1;
                    if !verify_ipa_committed(challenger, f(p.sum_sq), &p.ipa, values, &mut acc) {
                        return false;
                    }
                    if !verify_mul_committed(challenger, &p.mul, values, &mut acc) {
                        return false;
                    }
                    // Re-derive inv_q and check y = inv_q·h.
                    let inv_q = ((1i64 << RECIP_Q) as f64
                        / ((p.sum_sq as f64 / hidden as f64) + eps).sqrt())
                    .round() as i64;
                    if inv_q != p.inv_q {
                        return false;
                    }
                    let mut tr = fri::Transcript::new(challenger);
                    tr.observe(&p.com_y);
                    tr.observe(&p.mul.com_out);
                    let mut r: Vec<Fp> = (0..hidden).map(|_| tr.sample_val()).collect();
                    r.resize(next_pow2(hidden), Fp::ZERO);
                    if !verify_inner_product_committed(
                        challenger, &p.com_y, p.dom_y, &r, p.b_y, &p.claim_y, values, &mut acc,
                    ) || !verify_inner_product_committed(
                        challenger,
                        &p.mul.com_out,
                        p.mul.dom_out,
                        &r,
                        p.b_h,
                        &p.claim_h,
                        values,
                        &mut acc,
                    ) || p.b_y != Fp::new(inv_q as u64) * p.b_h
                    {
                        return false;
                    }
                    if !verify_requant(
                        challenger,
                        &digit_table,
                        &bit_table,
                        &p.req,
                        values,
                        &mut acc,
                    ) {
                        return false;
                    }
                    let idx: Vec<usize> = (b * hidden..(b + 1) * hidden).collect();
                    if !verify_tie(challenger, &idx, &p.tie_x, values, &mut acc)
                        || !verify_tie(challenger, &idx, &p.tie_out, values, &mut acc)
                    {
                        return false;
                    }
                }
                shapes.insert(output.as_str(), sh);
            }
            Op::Rope {
                input,
                output,
                theta,
            } => {
                let sh = shapes[input.as_str()].clone();
                let (lq, h, d) = (sh[0], sh[1], sh[2]);
                for li in 0..lq {
                    for hi in 0..h {
                        let p = match &proof.ops[idx] {
                            OpProof::Rope(p) => p,
                            _ => return false,
                        };
                        idx += 1;
                        if !verify_inner_product_committed(
                            challenger,
                            &p.com_rot,
                            p.dom_rot,
                            &p.r,
                            p.b_rot,
                            &p.claim_rot,
                            values,
                            &mut acc,
                        ) || !verify_inner_product_committed(
                            challenger,
                            &p.tie_in.row_com,
                            p.tie_in.row_dom,
                            &p.rt,
                            p.b_inp,
                            &p.claim_inp,
                            values,
                            &mut acc,
                        ) || p.b_rot != p.b_inp
                        {
                            return false;
                        }
                        if !verify_requant(
                            challenger,
                            &digit_table,
                            &bit_table,
                            &p.req,
                            values,
                            &mut acc,
                        ) {
                            return false;
                        }
                        let base = (li * h + hi) * d;
                        let idx: Vec<usize> = (base..base + d).collect();
                        if !verify_tie(challenger, &idx, &p.tie_in, values, &mut acc)
                            || !verify_tie(challenger, &idx, &p.tie_out, values, &mut acc)
                        {
                            return false;
                        }
                    }
                }
                let _ = theta;
                shapes.insert(output.as_str(), sh);
            }
            Op::Scores {
                q,
                k,
                output,
                scale,
            } => {
                let qsh = shapes[q.as_str()].clone();
                let (lq, h, d) = (qsh[0], qsh[1], qsh[2]);
                for li in 0..lq {
                    for hi in 0..h {
                        let p = match &proof.ops[idx] {
                            OpProof::Scores(p) => p,
                            _ => return false,
                        };
                        idx += 1;
                        if !verify_matmul_committed(challenger, lq, d, &p.mm, values, &mut acc) {
                            return false;
                        }
                        let inv_q = ((1i64 << RECIP_Q) as f64 / scale).round() as i64;
                        if inv_q != p.inv_q {
                            return false;
                        }
                        let mut tr = fri::Transcript::new(challenger);
                        tr.observe(&p.com_y);
                        tr.observe(&p.mm.com_y);
                        let mut r: Vec<Fp> = (0..lq).map(|_| tr.sample_val()).collect();
                        r.resize(next_pow2(lq), Fp::ZERO);
                        if !verify_inner_product_committed(
                            challenger, &p.com_y, p.dom_y, &r, p.b_y, &p.claim_y, values, &mut acc,
                        ) || !verify_inner_product_committed(
                            challenger,
                            &p.mm.com_y,
                            p.mm.dom_y,
                            &r,
                            p.b_dot,
                            &p.claim_dot,
                            values,
                            &mut acc,
                        ) || p.b_y != Fp::new(inv_q as u64) * p.b_dot
                        {
                            return false;
                        }
                        if !verify_requant(
                            challenger,
                            &digit_table,
                            &bit_table,
                            &p.req,
                            values,
                            &mut acc,
                        ) {
                            return false;
                        }
                        let mut kh_idx = Vec::with_capacity(lq * d);
                        for t in 0..lq {
                            for j in 0..d {
                                kh_idx.push((t * h + hi) * d + j);
                            }
                        }
                        let q_idx: Vec<usize> =
                            ((li * h + hi) * d..(li * h + hi) * d + d).collect();
                        let s_idx: Vec<usize> =
                            ((li * h + hi) * lq..(li * h + hi) * lq + lq).collect();
                        if !verify_tie(challenger, &kh_idx, &p.tie_k, values, &mut acc)
                            || !verify_tie(challenger, &q_idx, &p.tie_q, values, &mut acc)
                            || !verify_tie(challenger, &s_idx, &p.tie_out, values, &mut acc)
                        {
                            return false;
                        }
                    }
                }
                let _ = (k, scale);
                shapes.insert(output.as_str(), vec![lq, h, lq]);
            }
            Op::CausalSoftmax { input, output } => {
                let sh = shapes[input.as_str()].clone();
                let (lq, h, lseq) = (sh[0], sh[1], sh[2]);
                for li in 0..lq {
                    for hi in 0..h {
                        let p = match &proof.ops[idx] {
                            OpProof::CausalSoftmax(p) => p,
                            _ => return false,
                        };
                        idx += 1;
                        if !verify_causal_softmax_committed(
                            challenger,
                            &digit_table,
                            &bit_table,
                            &p.cp,
                            values,
                            &mut acc,
                        ) {
                            return false;
                        }
                        let base = (li * h + hi) * lseq;
                        let idx: Vec<usize> = (base..base + lseq).collect();
                        if !verify_tie(challenger, &idx, &p.tie_in, values, &mut acc)
                            || !verify_tie(challenger, &idx, &p.tie_out, values, &mut acc)
                        {
                            return false;
                        }
                    }
                }
                shapes.insert(output.as_str(), sh);
            }
            Op::WeightedV {
                weights: wname,
                v,
                output,
            } => {
                let vsh = shapes[v.as_str()].clone();
                let (lq, h, d) = (vsh[0], vsh[1], vsh[2]);
                for li in 0..lq {
                    for hi in 0..h {
                        let p = match &proof.ops[idx] {
                            OpProof::WeightedV(p) => p,
                            _ => return false,
                        };
                        idx += 1;
                        if !verify_matmul_committed(challenger, d, lq, &p.mm, values, &mut acc) {
                            return false;
                        }
                        if !verify_requant(
                            challenger,
                            &digit_table,
                            &bit_table,
                            &p.req,
                            values,
                            &mut acc,
                        ) {
                            return false;
                        }
                        let mut vh_idx = Vec::with_capacity(d * lq);
                        for j in 0..d {
                            for t in 0..lq {
                                vh_idx.push((t * h + hi) * d + j);
                            }
                        }
                        let w_idx: Vec<usize> =
                            ((li * h + hi) * lq..(li * h + hi) * lq + lq).collect();
                        let o_idx: Vec<usize> =
                            ((li * h + hi) * d..(li * h + hi) * d + d).collect();
                        if !verify_tie(challenger, &vh_idx, &p.tie_v, values, &mut acc)
                            || !verify_tie(challenger, &w_idx, &p.tie_w, values, &mut acc)
                            || !verify_tie(challenger, &o_idx, &p.tie_out, values, &mut acc)
                        {
                            return false;
                        }
                    }
                }
                let _ = wname;
                shapes.insert(output.as_str(), vec![lq, h, d]);
            }
            Op::TopK {
                input,
                output,
                choice,
                ..
            } => {
                let sh = shapes[input.as_str()].clone();
                let (rows, e) = (sh[0], sh[1]);
                for b in 0..rows {
                    let p = match &proof.ops[idx] {
                        OpProof::TopK(p) => p,
                        _ => return false,
                    };
                    idx += 1;
                    if !verify_topk_committed(challenger, &p.tp, values, &mut acc) {
                        return false;
                    }
                    let idx: Vec<usize> = (b * e..(b + 1) * e).collect();
                    if !verify_tie(challenger, &idx, &p.tie_in, values, &mut acc) {
                        return false;
                    }
                    if !verify_tie(challenger, &idx, &p.tie_choice, values, &mut acc) {
                        return false;
                    }
                }
                shapes.insert(output.as_str(), sh.clone());
                shapes.insert(choice.as_str(), sh);
            }
            Op::WeightedSum {
                choice,
                terms,
                output,
            } => {
                let csh = shapes[choice.as_str()].clone();
                let (seq, experts) = (csh[0], csh[1]);
                let hidden = spec.dims.hidden;
                for s in 0..seq {
                    let p = match &proof.ops[idx] {
                        OpProof::WeightedSum(p) => p,
                        _ => return false,
                    };
                    idx += 1;
                    let r = &p.r;
                    if !verify_inner_product_committed(
                        challenger,
                        &p.tie_out.row_com,
                        p.tie_out.row_dom,
                        r,
                        p.b_out,
                        &p.claim_out,
                        values,
                        &mut acc,
                    ) {
                        return false;
                    }
                    // Per expert: term_claim then tie_term, interleaved to match the
                    // prover's record order.
                    let t_idx: Vec<usize> = (s * hidden..(s + 1) * hidden).collect();
                    for e in 0..experts {
                        if !verify_inner_product_committed(
                            challenger,
                            &p.tie_terms[e].row_com,
                            p.tie_terms[e].row_dom,
                            r,
                            p.term_bs[e],
                            &p.term_claims[e],
                            values,
                            &mut acc,
                        ) {
                            return false;
                        }
                        if !verify_tie(challenger, &t_idx, &p.tie_terms[e], values, &mut acc) {
                            return false;
                        }
                    }
                    let t_vals = &p.term_bs;
                    let b_choice = p.choice_b[0];
                    if !verify_inner_product_committed(
                        challenger,
                        &p.tie_choice.row_com,
                        p.tie_choice.row_dom,
                        t_vals,
                        b_choice,
                        &p.choice_claims[0],
                        values,
                        &mut acc,
                    ) || p.b_out != b_choice
                    {
                        return false;
                    }
                    let c_idx: Vec<usize> = (s * experts..(s + 1) * experts).collect();
                    let o_idx: Vec<usize> = (s * hidden..(s + 1) * hidden).collect();
                    if !verify_tie(challenger, &c_idx, &p.tie_choice, values, &mut acc)
                        || !verify_tie(challenger, &o_idx, &p.tie_out, values, &mut acc)
                    {
                        return false;
                    }
                }
                let _ = terms;
                shapes.insert(output.as_str(), vec![seq, hidden]);
            }
        }
    }

    // Head check: the revealed last-logits row is tied to the committed logits and
    // head_token is its argmax.
    let cols = proof.head_logits.len();
    let logits_len = l * cols;
    let mut hrow = vec![Fp::ZERO; next_pow2(logits_len)];
    for j in 0..cols {
        hrow[(l - 1) * cols + j] = proof.head_r[j];
    }
    let expected: Fp = proof
        .head_r
        .iter()
        .zip(&proof.head_logits)
        .map(|(&ri, &v)| ri * f(v))
        .sum();
    if !verify_inner_product_committed(
        challenger,
        &proof.head_com,
        proof.head_dom,
        &hrow,
        proof.head_b,
        &proof.head_claim,
        values,
        &mut acc,
    ) || proof.head_b != expected
    {
        return false;
    }
    if !fri::open_many_verify(pcs, challenger, &acc, fri_proof) {
        return false;
    }
    match head_token {
        Some(h) => {
            h < proof.head_logits.len()
                && proof.head_logits[h] == *proof.head_logits.iter().max().unwrap()
        }
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Cell, Segment, WeightMatrix};

    fn cfg() -> CommittedConfig {
        CommittedConfig {
            requant_digits: 4,
            silu_lo: -100,
            silu_hi: 100,
            exp_lo: -100,
            sign_lo: -100,
            sign_hi: 100,
        }
    }

    fn rand_i8(seed: &mut u64, n: usize) -> Vec<i8> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((*seed >> 33) % 9) as i8 - 4
            })
            .collect()
    }

    fn demo_weights(seed: &mut u64) -> Weights {
        vec![
            WeightMatrix::new("embed_tokens", 64, 8, rand_i8(seed, 64 * 8)),
            WeightMatrix::new("w1", 16, 8, rand_i8(seed, 16 * 8)),
            WeightMatrix::new("w2", 8, 16, rand_i8(seed, 8 * 16)),
            WeightMatrix::new("lm_head", 64, 8, rand_i8(seed, 64 * 8)),
        ]
    }

    fn forward(weights: &Weights, window: &[usize]) -> (Vec<Cell>, usize) {
        let get = |n: &str| weights.iter().find(|w| w.name == n).unwrap();
        let embed = get("embed_tokens");
        let w1 = get("w1");
        let w2 = get("w2");
        let head = get("lm_head");
        let d = 8usize;
        let l = window.len();
        let mut embedding = vec![0i64; l * d];
        for (b, &tok) in window.iter().enumerate() {
            for j in 0..d {
                embedding[b * d + j] = embed.values[tok * d + j] as i64;
            }
        }
        let mut h1 = vec![0i64; l * 16];
        for b in 0..l {
            for i in 0..16 {
                let acc: i64 = (0..d)
                    .map(|j| w1.values[i * d + j] as i64 * embedding[b * d + j])
                    .sum();
                h1[b * 16 + i] = requant(acc, 4);
            }
        }
        let act: Vec<i64> = h1.iter().map(|&v| silu(v)).collect();
        let mut h2 = vec![0i64; l * d];
        for b in 0..l {
            for i in 0..d {
                let acc: i64 = (0..16)
                    .map(|j| w2.values[i * 16 + j] as i64 * act[b * 16 + j])
                    .sum();
                h2[b * d + i] = requant(acc, 4);
            }
        }
        let residual: Vec<i64> = h2.iter().zip(&embedding).map(|(&a, &b)| a + b).collect();
        let mut logits = vec![0i64; l * 64];
        for b in 0..l {
            for i in 0..64 {
                let acc: i64 = (0..d)
                    .map(|j| head.values[i * d + j] as i64 * residual[b * d + j])
                    .sum();
                logits[b * 64 + i] = requant(acc, 4);
            }
        }
        let head_token = (0..64).max_by_key(|&i| logits[(l - 1) * 64 + i]).unwrap();
        let cells = vec![
            Cell::new("embedding", l, d, embedding),
            Cell::new("h1", l, 16, h1),
            Cell::new("act", l, 16, act),
            Cell::new("h2", l, d, h2),
            Cell::new("residual", l, d, residual),
            Cell::new("logits", l, 64, logits),
        ];
        (cells, head_token)
    }

    #[test]
    fn committed_succinct_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        let spec: ModelSpec = serde_json::from_str(include_str!("../specs/demo.json")).unwrap();
        let mut seed = 12345u64;
        let weights = demo_weights(&mut seed);
        let window = [3, 17, 42, 9];
        let (cells, head_token) = forward(&weights, &window);
        let trace = Trace {
            segments: vec![Segment {
                cells,
                head_token: Some(head_token),
            }],
        };
        let proof = prove_succinct_committed(&mut ctx, &spec, &weights, &trace, &window, &cfg());
        assert!(verify_succinct_committed(
            &ctx.pcs,
            &ctx.challenger,
            &spec,
            &window,
            Some(head_token),
            &cfg(),
            &proof,
        ));
    }

    #[test]
    fn committed_succinct_rejects_wrong_head() {
        let mut ctx = fri::Prover::new(1, 10);
        let spec: ModelSpec = serde_json::from_str(include_str!("../specs/demo.json")).unwrap();
        let mut seed = 12345u64;
        let weights = demo_weights(&mut seed);
        let window = [3, 17, 42, 9];
        let (cells, head_token) = forward(&weights, &window);
        let trace = Trace {
            segments: vec![Segment {
                cells,
                head_token: Some(head_token),
            }],
        };
        let proof = prove_succinct_committed(&mut ctx, &spec, &weights, &trace, &window, &cfg());
        let wrong = (head_token + 1) % 64;
        assert!(!verify_succinct_committed(
            &ctx.pcs,
            &ctx.challenger,
            &spec,
            &window,
            Some(wrong),
            &cfg(),
            &proof,
        ));
    }

    const MINI_ATTN_JSON: &str = r#"{
        "dims": {"hidden": 2, "vocab": 8, "layers": 1, "heads": 1, "head_dim": 2,
                  "experts": 1, "active": 1, "intermediate": 2},
        "weights": [
            {"name": "embed_tokens", "rows": 8, "cols": 2},
            {"name": "q_proj", "rows": 2, "cols": 2},
            {"name": "k_proj", "rows": 2, "cols": 2},
            {"name": "v_proj", "rows": 2, "cols": 2},
            {"name": "o_proj", "rows": 2, "cols": 2},
            {"name": "lm_head", "rows": 8, "cols": 2}
        ],
        "ops": [
            {"op": "embed", "weight": "embed_tokens", "output": "embedding"},
            {"op": "matmul", "input": "embedding", "weight": "q_proj", "output": "q", "shift": 4},
            {"op": "matmul", "input": "embedding", "weight": "k_proj", "output": "k", "shift": 4},
            {"op": "matmul", "input": "embedding", "weight": "v_proj", "output": "v", "shift": 4},
            {"op": "reshape", "input": "q", "output": "q_heads", "shape": [-1, 1, 2]},
            {"op": "reshape", "input": "k", "output": "k_heads", "shape": [-1, 1, 2]},
            {"op": "reshape", "input": "v", "output": "v_heads", "shape": [-1, 1, 2]},
            {"op": "rope", "input": "q_heads", "output": "q_rot", "theta": 10000.0},
            {"op": "rope", "input": "k_heads", "output": "k_rot", "theta": 10000.0},
            {"op": "scores", "q": "q_rot", "k": "k_rot", "output": "scores", "scale": 1.4142135623730951},
            {"op": "causal_softmax", "input": "scores", "output": "attn_weights"},
            {"op": "weighted_v", "weights": "attn_weights", "v": "v_heads", "output": "attn_heads"},
            {"op": "reshape", "input": "attn_heads", "output": "attn", "shape": [-1, 2]},
            {"op": "matmul", "input": "attn", "weight": "o_proj", "output": "attn_out", "shift": 4},
            {"op": "add", "inputs": ["embedding", "attn_out"], "output": "residual"},
            {"op": "matmul", "input": "residual", "weight": "lm_head", "output": "logits", "shift": 4}
        ],
        "cells": ["embedding", "q", "k", "v", "q_heads", "k_heads", "v_heads", "q_rot",
                  "k_rot", "scores", "attn_weights", "attn_heads", "attn", "attn_out",
                  "residual", "logits"]
    }"#;

    #[test]
    fn committed_succinct_attention_honest() {
        use crate::spec::{exp_q15, rope_apply, round_div};
        let mut ctx = fri::Prover::new(1, 10);
        let spec: ModelSpec = serde_json::from_str(MINI_ATTN_JSON).unwrap();
        let mut seed = 99u64;
        let weights = vec![
            WeightMatrix::new("embed_tokens", 8, 2, rand_i8(&mut seed, 16)),
            WeightMatrix::new("q_proj", 2, 2, rand_i8(&mut seed, 4)),
            WeightMatrix::new("k_proj", 2, 2, rand_i8(&mut seed, 4)),
            WeightMatrix::new("v_proj", 2, 2, rand_i8(&mut seed, 4)),
            WeightMatrix::new("o_proj", 2, 2, rand_i8(&mut seed, 4)),
            WeightMatrix::new("lm_head", 8, 2, rand_i8(&mut seed, 16)),
        ];
        let window = [1usize, 3];
        let l = window.len();
        let d = 2usize;

        let embed = &weights[0];
        let mut embedding = vec![0i64; l * d];
        for (b, &tok) in window.iter().enumerate() {
            for j in 0..d {
                embedding[b * d + j] = embed.values[tok * d + j] as i64;
            }
        }
        let matmul_batch = |w: &[i8], x: &[i64], out: usize, in_: usize| -> Vec<i64> {
            let mut r = vec![0i64; l * out];
            for b in 0..l {
                for i in 0..out {
                    let acc: i64 = (0..in_)
                        .map(|j| w[i * in_ + j] as i64 * x[b * in_ + j])
                        .sum();
                    r[b * out + i] = requant(acc, 4);
                }
            }
            r
        };
        let q = matmul_batch(&weights[1].values, &embedding, 2, 2);
        let k = matmul_batch(&weights[2].values, &embedding, 2, 2);
        let v = matmul_batch(&weights[3].values, &embedding, 2, 2);

        let mut q_rot = q.clone();
        let mut k_rot = k.clone();
        for li in 0..l {
            rope_apply(&mut q_rot[li * d..(li + 1) * d], li, 10000.0);
            rope_apply(&mut k_rot[li * d..(li + 1) * d], li, 10000.0);
        }

        let scale = 2.0_f64.sqrt();
        let mut scores = vec![0i64; l * l];
        for li in 0..l {
            for t in 0..l {
                let dot: i64 = (0..d).map(|j| q_rot[li * d + j] * k_rot[t * d + j]).sum();
                scores[li * l + t] = round_div(dot, scale);
            }
        }
        let mut attn_weights = vec![0i64; l * l];
        for li in 0..l {
            let m = *scores[li * l..li * l + li + 1].iter().max().unwrap();
            let mut e = vec![0i64; l];
            for t in 0..=li {
                e[t] = exp_q15(scores[li * l + t] - m);
            }
            let s: i64 = e.iter().sum();
            let inv_q = ((1i64 << (RECIP_Q + 15)) as f64 / s as f64).round() as i64;
            for t in 0..=li {
                attn_weights[li * l + t] = requant(e[t] * inv_q, RECIP_Q);
            }
        }
        let mut attn_heads = vec![0i64; l * d];
        for li in 0..l {
            for j in 0..d {
                let acc: i64 = (0..l)
                    .map(|t| attn_weights[li * l + t] * v[t * d + j])
                    .sum();
                attn_heads[li * d + j] = requant(acc, 15);
            }
        }
        let attn = attn_heads.clone();
        let attn_out = matmul_batch(&weights[4].values, &attn, 2, 2);
        let residual: Vec<i64> = embedding
            .iter()
            .zip(&attn_out)
            .map(|(&a, &b)| a + b)
            .collect();
        let logits = matmul_batch(&weights[5].values, &residual, 8, 2);
        let head_token = (0..8).max_by_key(|&i| logits[(l - 1) * 8 + i]).unwrap();

        let c = |name: &str, rows: usize, cols: usize, v: Vec<i64>| Cell::new(name, rows, cols, v);
        let cells = vec![
            c("embedding", l, d, embedding),
            c("q", l, d, q.clone()),
            c("k", l, d, k.clone()),
            c("v", l, d, v.clone()),
            c("q_heads", l, d, q),
            c("k_heads", l, d, k),
            c("v_heads", l, d, v),
            c("q_rot", l, d, q_rot),
            c("k_rot", l, d, k_rot),
            c("scores", l, l, scores),
            c("attn_weights", l, l, attn_weights),
            c("attn_heads", l, d, attn_heads),
            c("attn", l, d, attn),
            c("attn_out", l, d, attn_out),
            c("residual", l, d, residual),
            c("logits", l, 8, logits),
        ];
        let trace = Trace {
            segments: vec![Segment {
                cells,
                head_token: Some(head_token),
            }],
        };
        let proof = prove_succinct_committed(&mut ctx, &spec, &weights, &trace, &window, &cfg());
        assert!(verify_succinct_committed(
            &ctx.pcs,
            &ctx.challenger,
            &spec,
            &window,
            Some(head_token),
            &cfg(),
            &proof,
        ));
    }

    const MINI_MOE_JSON: &str = r#"{
        "dims": {"hidden": 2, "vocab": 4, "layers": 1, "heads": 1, "head_dim": 2,
                  "experts": 2, "active": 1, "intermediate": 2},
        "weights": [
            {"name": "embed_tokens", "rows": 4, "cols": 2},
            {"name": "norm", "rows": 1, "cols": 2},
            {"name": "router", "rows": 2, "cols": 2},
            {"name": "expert_0", "rows": 2, "cols": 2},
            {"name": "expert_1", "rows": 2, "cols": 2},
            {"name": "lm_head", "rows": 4, "cols": 2}
        ],
        "ops": [
            {"op": "embed", "weight": "embed_tokens", "output": "embedding"},
            {"op": "norm", "input": "embedding", "weight": "norm", "output": "norm_emb", "eps": 1e-6},
            {"op": "matmul", "input": "norm_emb", "weight": "router", "output": "router_logits", "shift": 4},
            {"op": "top_k", "input": "router_logits", "output": "router_logits", "choice": "choice", "k": 1},
            {"op": "matmul", "input": "norm_emb", "weight": "expert_0", "output": "e0", "shift": 4},
            {"op": "matmul", "input": "norm_emb", "weight": "expert_1", "output": "e1", "shift": 4},
            {"op": "weighted_sum", "choice": "choice", "terms": ["e0", "e1"], "output": "ffn_out"},
            {"op": "add", "inputs": ["embedding", "ffn_out"], "output": "residual"},
            {"op": "matmul", "input": "residual", "weight": "lm_head", "output": "logits", "shift": 4}
        ],
        "cells": ["embedding", "norm_emb", "router_logits", "choice", "e0", "e1", "ffn_out",
                  "residual", "logits"]
    }"#;

    #[test]
    fn committed_succinct_moe_honest() {
        use crate::spec::rmsnorm_vec;
        let mut ctx = fri::Prover::new(1, 10);
        let spec: ModelSpec = serde_json::from_str(MINI_MOE_JSON).unwrap();
        let mut seed = 7u64;
        let weights = vec![
            WeightMatrix::new("embed_tokens", 4, 2, rand_i8(&mut seed, 8)),
            WeightMatrix::new("norm", 1, 2, rand_i8(&mut seed, 2)),
            WeightMatrix::new("router", 2, 2, rand_i8(&mut seed, 4)),
            WeightMatrix::new("expert_0", 2, 2, rand_i8(&mut seed, 4)),
            WeightMatrix::new("expert_1", 2, 2, rand_i8(&mut seed, 4)),
            WeightMatrix::new("lm_head", 4, 2, rand_i8(&mut seed, 8)),
        ];
        let window = [1usize, 2];
        let l = window.len();
        let d = 2usize;

        let embed = &weights[0];
        let mut embedding = vec![0i64; l * d];
        for (b, &tok) in window.iter().enumerate() {
            for j in 0..d {
                embedding[b * d + j] = embed.values[tok * d + j] as i64;
            }
        }
        let gamma = &weights[1].values;
        let norm_emb: Vec<i64> = (0..l)
            .flat_map(|b| rmsnorm_vec(&embedding[b * d..(b + 1) * d], gamma, 1e-6))
            .collect();
        let matmul_batch = |w: &[i8], x: &[i64], out: usize, in_: usize| -> Vec<i64> {
            let mut r = vec![0i64; l * out];
            for b in 0..l {
                for i in 0..out {
                    let acc: i64 = (0..in_)
                        .map(|j| w[i * in_ + j] as i64 * x[b * in_ + j])
                        .sum();
                    r[b * out + i] = requant(acc, 4);
                }
            }
            r
        };
        let router_logits = matmul_batch(&weights[2].values, &norm_emb, 2, 2);
        let e0 = matmul_batch(&weights[3].values, &norm_emb, 2, 2);
        let e1 = matmul_batch(&weights[4].values, &norm_emb, 2, 2);
        // top-1 choice (ties broken by lower index, matching the spec).
        let mut choice = vec![0i64; l * 2];
        for b in 0..l {
            let mut argmax = 0usize;
            for e in 1..2 {
                if router_logits[b * 2 + e] > router_logits[b * 2 + argmax] {
                    argmax = e;
                }
            }
            choice[b * 2 + argmax] = 1;
        }
        // weighted_sum.
        let mut ffn_out = vec![0i64; l * d];
        for b in 0..l {
            for j in 0..d {
                ffn_out[b * d + j] = if choice[b * 2] != 0 { e0[b * d + j] } else { 0 }
                    + if choice[b * 2 + 1] != 0 {
                        e1[b * d + j]
                    } else {
                        0
                    };
            }
        }
        let residual: Vec<i64> = embedding
            .iter()
            .zip(&ffn_out)
            .map(|(&a, &b)| a + b)
            .collect();
        let logits = matmul_batch(&weights[5].values, &residual, 4, 2);
        let head_token = (0..4).max_by_key(|&i| logits[(l - 1) * 4 + i]).unwrap();

        let c = |name: &str, rows: usize, cols: usize, v: Vec<i64>| Cell::new(name, rows, cols, v);
        let cells = vec![
            c("embedding", l, d, embedding),
            c("norm_emb", l, d, norm_emb),
            c("router_logits", l, 2, router_logits.clone()),
            c("choice", l, 2, choice),
            c("e0", l, d, e0),
            c("e1", l, d, e1),
            c("ffn_out", l, d, ffn_out),
            c("residual", l, d, residual),
            c("logits", l, 4, logits),
        ];
        let trace = Trace {
            segments: vec![Segment {
                cells,
                head_token: Some(head_token),
            }],
        };
        let proof = prove_succinct_committed(&mut ctx, &spec, &weights, &trace, &window, &cfg());
        assert!(verify_succinct_committed(
            &ctx.pcs,
            &ctx.challenger,
            &spec,
            &window,
            Some(head_token),
            &cfg(),
            &proof,
        ));
    }
}
