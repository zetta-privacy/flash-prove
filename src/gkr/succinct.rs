//! GKR-based op-graph verifier (model-agnostic, driven by `ModelSpec`).
//!
//! GKR counterpart of [`crate::gkr::commit`]: walks a model's op graph and
//! checks every op against FRI-committed weights + trace, but the linear layers use
//! the GKR/sumcheck reduction (O(depth·log n) verifier) instead of IPA.
//!
//! First chunk: the dense/linear core — `embed`, `matmul` (+requant), `silu`, `add`,
//! and the `head` argmax. Attention / MoE / norm / rope are wired next.

use std::collections::HashMap;

use crate::field::sumcheck::mle_eval;
use crate::field::{Ext, Fp};
use crate::gkr::commit::{
    chunk_name, chunk_row_range, commit_cells, num_chunks, tie_row, verify_tie, CellCommit,
    CommittedConfig, RowTie, TiedClaim,
};
use crate::gkr::protocol::{
    prove_gkr_matmul, prove_gkr_norm, verify_gkr_matmul, verify_gkr_norm, GkrMatmulProof,
    GkrNormProof,
};
use crate::lookup::lookup_logup::{
    commit_function_table, prove_lookup2_logup_given, verify_lookup2_logup, FunctionTable,
    Lookup2LogUpProof,
};
use crate::lookup::mul_committed::{prove_mul_committed_given, verify_mul_committed, MulProof};
use crate::lookup::norm_rope_committed::{prove_rope_committed, verify_rope_committed, RopeProof};
use crate::lookup::requant::{commit_bit_table, prove_requant_given, verify_requant, RequantProof};
use crate::lookup::selection_committed::{
    prove_causal_softmax_committed_given, prove_topk_committed_given,
    verify_causal_softmax_committed, verify_topk_committed, CausalSoftmaxProof, TopKProof,
};
use crate::model::spec::{exp_q15, requant, silu, ModelSpec, Op};
use crate::model::{Trace, Weights};
use crate::pcs::fri::{self, Challenger, Pcs, Prover};
use crate::pcs::zk::{
    prove_inner_product_committed_given, prove_mle_committed, verify_inner_product_committed,
    verify_mle_committed, InnerProductProof, Openings,
};
use p3_challenger::CanObserve;
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

fn log2(x: usize) -> usize {
    x.trailing_zeros() as usize
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EmbedProof {
    pub r: Vec<Fp>,
    pub b_out: Fp,
    pub b_emb: Vec<Fp>,
    pub claim_out: InnerProductProof,
    pub claim_emb: Vec<InnerProductProof>,
    pub com_out: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom_out: fri::Domain,
    pub com_emb: Vec<fri::Commitment>,
    #[serde(with = "crate::pcs::fri::serde::domain_vec")]
    pub dom_emb: Vec<fri::Domain>,
    pub l: usize,
    pub hidden: usize,
    pub vocab: usize,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MatmulOpProof {
    pub mm: GkrMatmulProof,
    pub req: RequantProof,
    pub com_w: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom_w: fri::Domain,
    pub com_wb: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom_wb: fri::Domain,
    pub com_x: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom_x: fri::Domain,
    pub com_acc: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom_acc: fri::Domain,
    pub bd_u: Vec<Fp>,
    pub bd_b_wb: Fp,
    pub bd_b_w: Fp,
    pub bd_claim_wb: InnerProductProof,
    pub bd_claim_w: InnerProductProof,
    pub shift: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SiluProof {
    pub lk: Lookup2LogUpProof,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AddProof {
    pub r: Vec<Fp>,
    pub b_out: Fp,
    pub claim_out: InnerProductProof,
    pub com_out: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom_out: fri::Domain,
    pub inputs: Vec<TiedClaim>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct NormOpProof {
    pub np: GkrNormProof,
    pub tie_x: RowTie,
    pub com_gamma: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom_gamma: fri::Domain,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RopeOpProof {
    pub rp: RopeProof,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ScoresOpProof {
    pub mm: GkrMatmulProof,
    pub req: RequantProof,
    pub y_val: Ext,
    pub dot_val: Ext,
    pub mle_y: [InnerProductProof; 3],
    pub mle_dot: [InnerProductProof; 3],
    pub com_y: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom_y: fri::Domain,
    pub com_dot: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom_dot: fri::Domain,
    pub tie_q: RowTie,
    pub tie_k: RowTie,
    pub tie_out: RowTie,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ReshapeProof {
    pub r: Vec<Fp>,
    pub b_in: Fp,
    pub b_out: Fp,
    pub claim_in: InnerProductProof,
    pub claim_out: InnerProductProof,
    pub com_in: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom_in: fri::Domain,
    pub com_out: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom_out: fri::Domain,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CausalSoftmaxOpProof {
    pub cp: CausalSoftmaxProof,
    pub tie_in: RowTie,
    pub tie_out: RowTie,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct WeightedVOpProof {
    pub mm: GkrMatmulProof,
    pub req: RequantProof,
    pub com_acc: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom_acc: fri::Domain,
    pub tie_v: RowTie,
    pub tie_w: RowTie,
    pub tie_out: RowTie,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MulOpProof {
    pub mul: MulProof,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TopKOpProof {
    pub tp: TopKProof,
    pub tie_in: RowTie,
    pub tie_choice: RowTie,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum GkrOpProof {
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

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SuccinctGkrProof {
    pub ops: Vec<GkrOpProof>,
    /// The committed weights (name → commitment), so the verifier can bind the
    /// proof to the pre-shared weight commitments produced by the commit phase.
    pub weight_coms: HashMap<String, fri::Commitment>,
    pub head_range: Lookup2LogUpProof,
    pub head_count: InnerProductProof,
    pub head_tie: RowTie,
    pub head_mle: [InnerProductProof; 3],
    pub head_max: i64,
    pub values: Vec<Ext>,
    pub fri_proof: fri::Proof,
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

struct Tables {
    silu: FunctionTable,
    exp: FunctionTable,
    sign: FunctionTable,
}

fn build_tables(pcs: &Pcs, cfg: &CommittedConfig, alpha: Fp, beta: Fp) -> Tables {
    Tables {
        silu: commit_function_table(pcs, silu, cfg.silu_lo, cfg.silu_hi, alpha, beta),
        exp: commit_function_table(pcs, exp_q15, cfg.exp_lo, 0, alpha, beta),
        sign: commit_function_table(
            pcs,
            |x| if x >= 0 { 1 } else { 0 },
            cfg.sign_lo,
            cfg.sign_hi,
            alpha,
            beta,
        ),
    }
}

/// GKR batched matmul `acc[b] = W·input[b]` via the block-diagonal `W ⊗ I_L`.
fn prove_gkr_batched_matmul(
    ctx: &mut Prover,
    w: &CellCommit,
    x: &CellCommit,
    acc: &CellCommit,
    batch: usize,
    out_dim: usize,
    in_dim: usize,
) -> (
    GkrMatmulProof,
    fri::Commitment,
    fri::Domain,
    Vec<Fp>,
    Fp,
    Fp,
    InnerProductProof,
    InnerProductProof,
) {
    let out_rows = batch * out_dim;
    let in_cols = batch * in_dim;
    let out_rows_p2 = next_pow2(out_rows);
    let in_cols_p2 = next_pow2(in_cols);
    let mut wb = vec![0i64; out_rows_p2 * in_cols_p2];
    for b in 0..batch {
        for r in 0..out_dim {
            for c in 0..in_dim {
                wb[(b * out_dim + r) * in_cols_p2 + (b * in_dim + c)] = w.values[r * in_dim + c];
            }
        }
    }
    let wb_fp: Vec<Fp> = wb.iter().map(|&v| f(v)).collect();
    let (com_wb, pd_wb, dom_wb) = fri::commit_vec(&ctx.pcs, &wb_fp);

    let mut tr = fri::Transcript::new(&ctx.challenger);
    tr.observe(&com_wb);
    tr.observe(&w.com);
    let u: Vec<Fp> = (0..out_rows_p2 * in_cols_p2)
        .map(|_| tr.sample_val())
        .collect();
    let bd_b_wb: Fp = u.iter().zip(&wb_fp).map(|(&ui, &vi)| ui * vi).sum();
    let mut u_agg = vec![Fp::ZERO; next_pow2(out_dim * in_dim)];
    for b in 0..batch {
        for r in 0..out_dim {
            for c in 0..in_dim {
                u_agg[r * in_dim + c] += u[(b * out_dim + r) * in_cols_p2 + (b * in_dim + c)];
            }
        }
    }
    let bd_b_w: Fp = u_agg.iter().zip(&w.fp).map(|(&ui, &vi)| ui * vi).sum();
    let bd_claim_wb = prove_inner_product_committed_given(ctx, &com_wb, &pd_wb, dom_wb, &wb_fp, &u);
    let bd_claim_w = prove_inner_product_committed_given(ctx, &w.com, &w.pd, w.dom, &w.fp, &u_agg);

    let mm = prove_gkr_matmul(
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
        log2(out_rows_p2),
        log2(in_cols_p2),
        0,
    );

    (
        mm,
        com_wb,
        dom_wb,
        u,
        bd_b_wb,
        bd_b_w,
        bd_claim_wb,
        bd_claim_w,
    )
}

fn verify_gkr_batched_matmul(
    challenger: &Challenger,
    com_w: &fri::Commitment,
    dom_w: fri::Domain,
    com_wb: &fri::Commitment,
    dom_wb: fri::Domain,
    com_x: &fri::Commitment,
    dom_x: fri::Domain,
    com_acc: &fri::Commitment,
    dom_acc: fri::Domain,
    batch: usize,
    out_dim: usize,
    in_dim: usize,
    mm: &GkrMatmulProof,
    u: &[Fp],
    bd_b_wb: Fp,
    bd_b_w: Fp,
    bd_claim_wb: &InnerProductProof,
    bd_claim_w: &InnerProductProof,
    values: &[Ext],
    acc: &mut Openings,
) -> bool {
    let in_cols_p2 = next_pow2(batch * in_dim);
    let mut u_agg = vec![Fp::ZERO; next_pow2(out_dim * in_dim)];
    for b in 0..batch {
        for r in 0..out_dim {
            for c in 0..in_dim {
                u_agg[r * in_dim + c] += u[(b * out_dim + r) * in_cols_p2 + (b * in_dim + c)];
            }
        }
    }
    verify_inner_product_committed(
        challenger,
        com_wb,
        dom_wb,
        u,
        bd_b_wb,
        bd_claim_wb,
        values,
        acc,
    ) && verify_inner_product_committed(
        challenger, com_w, dom_w, &u_agg, bd_b_w, bd_claim_w, values, acc,
    ) && verify_gkr_matmul(
        challenger, com_wb, dom_wb, com_x, dom_x, com_acc, dom_acc, mm, values, acc,
    )
}

pub fn prove_gkr_succinct(
    ctx: &mut Prover,
    spec: &ModelSpec,
    weights: &Weights,
    trace: &Trace,
    window: &[usize],
    cfg: &CommittedConfig,
) -> SuccinctGkrProof {
    assert_eq!(trace.segments.len(), 1, "single-segment trace only");
    let (cells, wmap) = commit_cells(&ctx.pcs, weights, trace);
    let weight_coms: HashMap<String, fri::Commitment> = wmap
        .iter()
        .map(|(name, c)| (name.clone(), c.com.clone()))
        .collect();
    let l = window.len();

    // Bind the head token into the transcript (matching the verifier) so a
    // mismatched token shifts every downstream challenge.
    let logits_cell = &cells["logits"];
    let hlast = logits_cell.rows - 1;
    let hcols = logits_cell.cols;
    let head_scores_pre: Vec<i64> = logits_cell.values[hlast * hcols..(hlast + 1) * hcols].to_vec();
    let head_adj: Vec<i64> = (0..hcols)
        .map(|i| head_scores_pre[i] * hcols as i64 - i as i64)
        .collect();
    let head_token = (0..hcols).max_by_key(|&j| head_adj[j]).unwrap();
    ctx.challenger.observe(Fp::new(head_token as u64));

    let alpha = Fp::new(11);
    let beta = Fp::new(5);
    let tables = build_tables(&ctx.pcs, cfg, alpha, beta);
    let digit_table = crate::lookup::range_check::commit_digit_table(&ctx.pcs);
    let bit_table = commit_bit_table(&ctx.pcs);

    let wdims: HashMap<&str, (usize, usize)> = spec
        .weights
        .iter()
        .map(|w| (w.name.as_str(), (w.rows, w.cols)))
        .collect();
    let mut shapes: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut ops = Vec::new();

    for op in &spec.ops {
        match op {
            Op::Embed { weight, output } => {
                let out = &cells[output.as_str()];
                let (vocab, hidden) = wdims[weight.as_str()];
                let n = l * hidden;
                let n_chunks = num_chunks(vocab);
                let mut tr = fri::Transcript::new(&ctx.challenger);
                tr.observe(&out.com);
                for i in 0..n_chunks {
                    tr.observe(&wmap[&chunk_name(weight, vocab, i)].com);
                }
                let mut r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
                r.resize(next_pow2(n), Fp::ZERO);
                let b_out: Fp = r.iter().zip(&out.fp).map(|(&ri, &vi)| ri * vi).sum();
                let mut h_emb = vec![Fp::ZERO; next_pow2(vocab * hidden)];
                for b in 0..l {
                    for j in 0..hidden {
                        h_emb[window[b] * hidden + j] += r[b * hidden + j];
                    }
                }
                let claim_out = prove_inner_product_committed_given(
                    ctx, &out.com, &out.pd, out.dom, &out.fp, &r,
                );
                let mut b_embs = Vec::with_capacity(n_chunks);
                let mut claim_embs = Vec::with_capacity(n_chunks);
                let mut com_embs = Vec::with_capacity(n_chunks);
                let mut dom_embs = Vec::with_capacity(n_chunks);
                for i in 0..n_chunks {
                    let (r0, r1) = chunk_row_range(vocab, i);
                    let emb = &wmap[&chunk_name(weight, vocab, i)];
                    let mut hc: Vec<Fp> = h_emb[r0 * hidden..r1 * hidden].to_vec();
                    hc.resize(next_pow2((r1 - r0) * hidden), Fp::ZERO);
                    let b_emb = hc.iter().zip(&emb.fp).map(|(&hi, &vi)| hi * vi).sum();
                    let claim_emb = prove_inner_product_committed_given(
                        ctx, &emb.com, &emb.pd, emb.dom, &emb.fp, &hc,
                    );
                    b_embs.push(b_emb);
                    claim_embs.push(claim_emb);
                    com_embs.push(emb.com.clone());
                    dom_embs.push(emb.dom);
                }
                ops.push(GkrOpProof::Embed(EmbedProof {
                    r,
                    b_out,
                    b_emb: b_embs,
                    claim_out,
                    claim_emb: claim_embs,
                    com_out: out.com.clone(),
                    dom_out: out.dom,
                    com_emb: com_embs,
                    dom_emb: dom_embs,
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
                let (out_dim, in_dim) = wdims[weight.as_str()];
                let x = &cells[input.as_str()];
                let batch = x.rows;
                assert_eq!(x.cols, in_dim);
                let n_chunks = num_chunks(out_dim);
                for ci in 0..n_chunks {
                    let (r0, r1) = chunk_row_range(out_dim, ci);
                    let chunk_rows = r1 - r0;
                    let w = &wmap[&chunk_name(weight, out_dim, ci)];
                    let mut acc = vec![0i64; batch * chunk_rows];
                    for b in 0..batch {
                        for r in 0..chunk_rows {
                            let a: i64 = (0..in_dim)
                                .map(|c| w.values[r * in_dim + c] as i64 * x.values[b * in_dim + c])
                                .sum();
                            acc[b * chunk_rows + r] = a;
                        }
                    }
                    let acc_cell = commit_one(&ctx.pcs, "__acc", batch, chunk_rows, acc.clone());
                    let (mm, com_wb, dom_wb, u, bd_b_wb, bd_b_w, bd_claim_wb, bd_claim_w) =
                        prove_gkr_batched_matmul(ctx, w, x, &acc_cell, batch, chunk_rows, in_dim);
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
                    ops.push(GkrOpProof::Matmul(MatmulOpProof {
                        mm,
                        req,
                        com_w: w.com.clone(),
                        dom_w: w.dom,
                        com_wb,
                        dom_wb,
                        com_x: x.com.clone(),
                        dom_x: x.dom,
                        com_acc: acc_cell.com.clone(),
                        dom_acc: acc_cell.dom,
                        bd_u: u,
                        bd_b_wb,
                        bd_b_w,
                        bd_claim_wb,
                        bd_claim_w,
                        shift: *shift,
                    }));
                }
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
                ops.push(GkrOpProof::Silu(SiluProof { lk }));
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
                let mut r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
                r.resize(next_pow2(n), Fp::ZERO);
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
                    ins.push(TiedClaim {
                        com: c.com.clone(),
                        dom: c.dom,
                        b: bi,
                        claim: cl,
                    });
                }
                ops.push(GkrOpProof::Add(AddProof {
                    r,
                    b_out,
                    claim_out,
                    com_out: out.com.clone(),
                    dom_out: out.dom,
                    inputs: ins,
                }));
                shapes.insert(output.as_str(), shapes[inputs[0].as_str()].clone());
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
                for b in 0..rows {
                    let xb: Vec<i64> = x.values[b * hidden..(b + 1) * hidden].to_vec();
                    let ob: Vec<i64> = out.values[b * hidden..(b + 1) * hidden].to_vec();
                    let xrow = commit_one(&ctx.pcs, "__norm_x", 1, hidden, xb.clone());
                    let idx: Vec<usize> = (b * hidden..(b + 1) * hidden).collect();
                    let tie_x = tie_row(ctx, &xrow, x, &idx);
                    let np = prove_gkr_norm(
                        ctx,
                        &xrow.com,
                        &xrow.pd,
                        xrow.dom,
                        &g.com,
                        &g.pd,
                        g.dom,
                        &xb,
                        &g.values,
                        &ob,
                        log2(hidden),
                        *eps,
                        &digit_table,
                        &bit_table,
                        cfg.requant_digits,
                    );
                    ops.push(GkrOpProof::Norm(NormOpProof {
                        np,
                        tie_x,
                        com_gamma: g.com.clone(),
                        dom_gamma: g.dom,
                    }));
                }
                shapes.insert(output.as_str(), shapes[input.as_str()].clone());
            }
            Op::Rope {
                input,
                output,
                theta,
            } => {
                let sh = shapes[input.as_str()].clone(); // [L, H, D]
                let (lq, h, d) = (sh[0], sh[1], sh[2]);
                let x = &cells[input.as_str()];
                let out = &cells[output.as_str()];
                for li in 0..lq {
                    for hi in 0..h {
                        let base = (li * h + hi) * d;
                        let xb: Vec<i64> = x.values[base..base + d].to_vec();
                        let ob: Vec<i64> = out.values[base..base + d].to_vec();
                        let rp =
                            prove_rope_committed(ctx, &xb, &ob, li, *theta, -65000, alpha, beta);
                        ops.push(GkrOpProof::Rope(RopeOpProof { rp }));
                    }
                }
                shapes.insert(output.as_str(), sh);
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
                let mut r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
                r.resize(next_pow2(n), Fp::ZERO);
                let b_in: Fp = r.iter().zip(&cin.fp).map(|(&ri, &vi)| ri * vi).sum();
                let b_out: Fp = r.iter().zip(&cout.fp).map(|(&ri, &vi)| ri * vi).sum();
                let claim_in = prove_inner_product_committed_given(
                    ctx, &cin.com, &cin.pd, cin.dom, &cin.fp, &r,
                );
                let claim_out = prove_inner_product_committed_given(
                    ctx, &cout.com, &cout.pd, cout.dom, &cout.fp, &r,
                );
                ops.push(GkrOpProof::Reshape(ReshapeProof {
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
            Op::Scores {
                q,
                k,
                output,
                scale,
            } => {
                let sh = shapes[q.as_str()].clone(); // [L, H, D]
                let (lq, h, d) = (sh[0], sh[1], sh[2]);
                let qc = &cells[q.as_str()];
                let kc = &cells[k.as_str()];
                let sc = &cells[output.as_str()];
                let inv_q = ((1i64 << crate::model::spec::RECIP_Q) as f64 / scale).round() as i64;
                for li in 0..lq {
                    for hi in 0..h {
                        let qlh: Vec<i64> =
                            qc.values[(li * h + hi) * d..(li * h + hi + 1) * d].to_vec();
                        let kh: Vec<i64> = (0..lq)
                            .flat_map(|t| {
                                let s = (t * h + hi) * d;
                                kc.values[s..s + d].to_vec()
                            })
                            .collect();
                        let dot: Vec<i64> = (0..lq)
                            .map(|t| (0..d).map(|j| qlh[j] * kh[t * d + j]).sum())
                            .collect();
                        let y: Vec<i64> = dot.iter().map(|&v| v * inv_q).collect();
                        let srow: Vec<i64> = y
                            .iter()
                            .map(|&v| requant(v, crate::model::spec::RECIP_Q))
                            .collect();
                        let qlh_c = commit_one(&ctx.pcs, "__scores_q", 1, d, qlh.clone());
                        let kh_c = commit_one(&ctx.pcs, "__scores_k", lq, d, kh.clone());
                        let dot_c = commit_one(&ctx.pcs, "__scores_dot", 1, lq, dot.clone());
                        let y_c = commit_one(&ctx.pcs, "__scores_y", 1, lq, y.clone());
                        let srow_c = commit_one(&ctx.pcs, "__scores_s", 1, lq, srow.clone());
                        // dot = kh · qlh (GKR matvec).
                        let mm = prove_gkr_matmul(
                            ctx,
                            &kh_c.com,
                            &kh_c.pd,
                            kh_c.dom,
                            &qlh_c.com,
                            &qlh_c.pd,
                            qlh_c.dom,
                            &dot_c.com,
                            &dot_c.pd,
                            dot_c.dom,
                            &kh_c.fp,
                            &qlh_c.fp,
                            &dot_c.fp,
                            log2(lq),
                            log2(d),
                            0,
                        );
                        // scalar scale y = dot · inv_q.
                        let mut tr = fri::Transcript::new(&ctx.challenger);
                        tr.observe(&dot_c.com);
                        tr.observe(&y_c.com);
                        let r: Vec<Ext> = (0..log2(lq)).map(|_| tr.sample_ext()).collect();
                        let dot_ext: Vec<Ext> = dot.iter().map(|&v| Ext::from(f(v))).collect();
                        let y_ext: Vec<Ext> = y.iter().map(|&v| Ext::from(f(v))).collect();
                        let dot_val = mle_eval(&dot_ext, &r);
                        let y_val = mle_eval(&y_ext, &r);
                        let mle_dot = prove_mle_committed(
                            ctx, &dot_c.com, &dot_c.pd, dot_c.dom, &dot_c.fp, &r,
                        );
                        let mle_y =
                            prove_mle_committed(ctx, &y_c.com, &y_c.pd, y_c.dom, &y_c.fp, &r);
                        // srow = requant(y, RECIP_Q).
                        let req = prove_requant_given(
                            ctx,
                            &y_c.com,
                            &y_c.pd,
                            y_c.dom,
                            &y,
                            &srow,
                            crate::model::spec::RECIP_Q,
                            cfg.requant_digits,
                            &digit_table,
                            &bit_table,
                        );
                        let q_idx: Vec<usize> =
                            ((li * h + hi) * d..(li * h + hi + 1) * d).collect();
                        let s_idx: Vec<usize> =
                            ((li * h + hi) * lq..(li * h + hi + 1) * lq).collect();
                        let tie_q = tie_row(ctx, &qlh_c, qc, &q_idx);
                        let tie_out = tie_row(ctx, &srow_c, sc, &s_idx);
                        ops.push(GkrOpProof::Scores(ScoresOpProof {
                            mm,
                            req,
                            y_val,
                            dot_val,
                            mle_y,
                            mle_dot,
                            com_y: y_c.com,
                            dom_y: y_c.dom,
                            com_dot: dot_c.com,
                            dom_dot: dot_c.dom,
                            tie_q,
                            tie_k: tie_row(
                                ctx,
                                &kh_c,
                                kc,
                                &(0..lq * d)
                                    .map(|i| (i / d * h + hi) * d + i % d)
                                    .collect::<Vec<_>>(),
                            ),
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
                        ops.push(GkrOpProof::CausalSoftmax(CausalSoftmaxOpProof {
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
                let vsh = shapes[v.as_str()].clone(); // [L,H,d]
                let (lq, h, d) = (vsh[0], vsh[1], vsh[2]);
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
                        let mm = prove_gkr_matmul(
                            ctx,
                            &vh_c.com,
                            &vh_c.pd,
                            vh_c.dom,
                            &wlh_c.com,
                            &wlh_c.pd,
                            wlh_c.dom,
                            &acc_c.com,
                            &acc_c.pd,
                            acc_c.dom,
                            &vh_c.fp,
                            &wlh_c.fp,
                            &acc_c.fp,
                            log2(d),
                            log2(lq),
                            0,
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
                        ops.push(GkrOpProof::WeightedV(WeightedVOpProof {
                            mm,
                            req,
                            com_acc: acc_c.com,
                            dom_acc: acc_c.dom,
                            tie_v,
                            tie_w,
                            tie_out,
                        }));
                    }
                }
                shapes.insert(output.as_str(), vec![lq, h, d]);
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
                ops.push(GkrOpProof::Mul(MulOpProof { mul }));
                shapes.insert(output.as_str(), shapes[a.as_str()].clone());
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
                    ops.push(GkrOpProof::TopK(TopKOpProof {
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
                    let b_choice: Fp = t_vals.iter().zip(&crow.fp).map(|(&ti, &vi)| ti * vi).sum();
                    let choice_claim = prove_inner_product_committed_given(
                        ctx, &crow.com, &crow.pd, crow.dom, &crow.fp, &t_vals,
                    );
                    let c_idx: Vec<usize> = (s * experts..(s + 1) * experts).collect();
                    let o_idx: Vec<usize> = (s * hidden..(s + 1) * hidden).collect();
                    let tie_choice = tie_row(ctx, &crow, cc, &c_idx);
                    let tie_out = tie_row(ctx, &orow, oc, &o_idx);
                    ops.push(GkrOpProof::WeightedSum(WeightedSumOpProof {
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

    // Head argmax: prove `head_token` is the argmax of the last logits row via a
    // succinct range check (`scores ≤ max`) + point evaluation. The range table is
    // over the *score* range (O(|score|)), not the vocab-scaled top-k delta.
    let logits = &cells["logits"];
    let last = logits.rows - 1;
    let cols = logits.cols;
    let head_scores: Vec<i64> = logits.values[last * cols..(last + 1) * cols].to_vec();
    let head_max = head_scores[head_token];

    let srow = commit_one(&ctx.pcs, "__head_scores", 1, cols, head_scores.clone());
    let s_idx: Vec<usize> = (last * cols..(last + 1) * cols).collect();
    let head_tie = tie_row(ctx, &srow, logits, &s_idx);

    // Range check: `ones[j] = sign(head_max − scores[j])` for all j, so every score
    // is ≤ head_max.
    let ones_i64 = vec![1i64; cols];
    let orow = commit_one(&ctx.pcs, "__head_ones", 1, cols, ones_i64.clone());
    let range_table = commit_function_table(
        &ctx.pcs,
        |x| if head_max - x >= 0 { 1 } else { 0 },
        cfg.score_lo,
        cfg.score_hi,
        alpha,
        beta,
    );
    let head_range = prove_lookup2_logup_given(
        ctx,
        &srow.com,
        &srow.pd,
        srow.dom,
        &orow.com,
        &orow.pd,
        orow.dom,
        &head_scores,
        &ones_i64,
        &range_table,
        alpha,
        beta,
    );
    // `⟨1, ones⟩ = cols` — every score passed the sign check (all ones).
    let ones_fp = pad(vec![Fp::ONE; cols], cols);
    let orow_fp = pad(ones_i64.iter().map(|&x| f(x)).collect(), cols);
    let head_count =
        prove_inner_product_committed_given(ctx, &orow.com, &orow.pd, orow.dom, &orow_fp, &ones_fp);

    // scores[head_token] == head_max (point evaluation at the bound token).
    let scores_fp = pad(head_scores.iter().map(|&x| f(x)).collect(), cols);
    let log_cols = log2(next_pow2(cols));
    let point: Vec<Ext> = (0..log_cols)
        .map(|k| {
            if (head_token >> k) & 1 == 1 {
                Ext::ONE
            } else {
                Ext::ZERO
            }
        })
        .collect();
    let head_mle = prove_mle_committed(ctx, &srow.com, &srow.pd, srow.dom, &scores_fp, &point);

    let fri_proof = ctx.open_all();
    let values = ctx.values.clone();

    SuccinctGkrProof {
        ops,
        weight_coms,
        head_range,
        head_count,
        head_tie,
        head_mle,
        head_max,
        values,
        fri_proof,
    }
}

pub fn verify_gkr_succinct(
    pcs: &Pcs,
    spec: &ModelSpec,
    window: &[usize],
    head_token: Option<usize>,
    cfg: &CommittedConfig,
    proof: &SuccinctGkrProof,
) -> bool {
    let values = &proof.values;
    let fri_proof = &proof.fri_proof;
    let mut acc = Openings::new();
    let l = window.len();

    let mut ch = fri::new_challenger();
    match head_token {
        Some(h) => ch.observe(Fp::new(h as u64)),
        None => return false,
    }
    let challenger = &ch;

    let alpha = Fp::new(11);
    let beta = Fp::new(5);
    let tables = build_tables(pcs, cfg, alpha, beta);
    let digit_table = crate::lookup::range_check::commit_digit_table(pcs);
    let bit_table = commit_bit_table(pcs);

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
                    GkrOpProof::Embed(p) => p,
                    _ => return false,
                };
                idx += 1;
                let (vocab, hidden) = wdims[weight.as_str()];
                let n_chunks = num_chunks(vocab);
                for i in 0..n_chunks {
                    if proof.weight_coms.get(&chunk_name(weight, vocab, i)) != Some(&p.com_emb[i]) {
                        return false;
                    }
                }
                let mut tr = fri::Transcript::new(challenger);
                tr.observe(&p.com_out);
                for com in &p.com_emb {
                    tr.observe(com);
                }
                let mut r: Vec<Fp> = (0..l * hidden).map(|_| tr.sample_val()).collect();
                r.resize(next_pow2(l * hidden), Fp::ZERO);
                let mut h_emb = vec![Fp::ZERO; next_pow2(vocab * hidden)];
                for b in 0..l {
                    for j in 0..hidden {
                        h_emb[window[b] * hidden + j] += r[b * hidden + j];
                    }
                }
                if !verify_inner_product_committed(
                    challenger,
                    &p.com_out,
                    p.dom_out,
                    &r,
                    p.b_out,
                    &p.claim_out,
                    values,
                    &mut acc,
                ) {
                    return false;
                }
                for i in 0..n_chunks {
                    let (r0, r1) = chunk_row_range(vocab, i);
                    let mut hc: Vec<Fp> = h_emb[r0 * hidden..r1 * hidden].to_vec();
                    hc.resize(next_pow2((r1 - r0) * hidden), Fp::ZERO);
                    if !verify_inner_product_committed(
                        challenger,
                        &p.com_emb[i],
                        p.dom_emb[i],
                        &hc,
                        p.b_emb[i],
                        &p.claim_emb[i],
                        values,
                        &mut acc,
                    ) {
                        return false;
                    }
                }
                shapes.insert(output.as_str(), vec![l, hidden]);
            }
            Op::Matmul {
                input,
                weight,
                output,
                shift,
            } => {
                let (out_dim, in_dim) = wdims[weight.as_str()];
                let batch = shapes[input.as_str()][0];
                let n_chunks = num_chunks(out_dim);
                for ci in 0..n_chunks {
                    let (r0, r1) = chunk_row_range(out_dim, ci);
                    let chunk_rows = r1 - r0;
                    let p = match &proof.ops[idx] {
                        GkrOpProof::Matmul(p) => p,
                        _ => return false,
                    };
                    idx += 1;
                    if proof.weight_coms.get(&chunk_name(weight, out_dim, ci)) != Some(&p.com_w) {
                        return false;
                    }
                    if !verify_gkr_batched_matmul(
                        challenger,
                        &p.com_w,
                        p.dom_w,
                        &p.com_wb,
                        p.dom_wb,
                        &p.com_x,
                        p.dom_x,
                        &p.com_acc,
                        p.dom_acc,
                        batch,
                        chunk_rows,
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
                }
                let mut s = shapes[input.as_str()].clone();
                let n = s.len();
                s[n - 1] = out_dim;
                shapes.insert(output.as_str(), s);
                let _ = shift;
            }
            Op::Silu { input, output } => {
                let p = match &proof.ops[idx] {
                    GkrOpProof::Silu(p) => p,
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
                    GkrOpProof::Add(p) => p,
                    _ => return false,
                };
                idx += 1;
                let n = shapes[inputs[0].as_str()].iter().product::<usize>();
                let mut tr = fri::Transcript::new(challenger);
                tr.observe(&p.com_out);
                for tc in &p.inputs {
                    tr.observe(&tc.com);
                }
                let mut r: Vec<Fp> = (0..n).map(|_| tr.sample_val()).collect();
                r.resize(next_pow2(n), Fp::ZERO);
                if !verify_inner_product_committed(
                    challenger,
                    &p.com_out,
                    p.dom_out,
                    &r,
                    p.b_out,
                    &p.claim_out,
                    values,
                    &mut acc,
                ) {
                    return false;
                }
                let mut b_sum = Fp::ZERO;
                for tc in &p.inputs {
                    if !verify_inner_product_committed(
                        challenger, &tc.com, tc.dom, &r, tc.b, &tc.claim, values, &mut acc,
                    ) {
                        return false;
                    }
                    b_sum += tc.b;
                }
                if p.b_out != b_sum {
                    return false;
                }
                shapes.insert(output.as_str(), shapes[inputs[0].as_str()].clone());
            }
            Op::Norm {
                input,
                weight,
                output,
                ..
            } => {
                let sh = shapes[input.as_str()].clone();
                let (rows, hidden) = (sh[0], sh[sh.len() - 1]);
                for b in 0..rows {
                    let p = match &proof.ops[idx] {
                        GkrOpProof::Norm(p) => p,
                        _ => return false,
                    };
                    idx += 1;
                    if proof.weight_coms.get(weight.as_str()) != Some(&p.com_gamma) {
                        return false;
                    }
                    let x_idx: Vec<usize> = (b * hidden..(b + 1) * hidden).collect();
                    if !verify_tie(challenger, &x_idx, &p.tie_x, values, &mut acc) {
                        return false;
                    }
                    if !verify_gkr_norm(
                        challenger,
                        &p.tie_x.row_com,
                        p.tie_x.row_dom,
                        &p.com_gamma,
                        p.dom_gamma,
                        &digit_table,
                        &bit_table,
                        &p.np,
                        values,
                        &mut acc,
                    ) {
                        return false;
                    }
                }
                shapes.insert(output.as_str(), sh);
                let _ = weight;
            }
            Op::Rope { input, output, .. } => {
                let sh = shapes[input.as_str()].clone();
                let (lq, h, _d) = (sh[0], sh[1], sh[2]);
                for _li in 0..lq {
                    for _hi in 0..h {
                        let p = match &proof.ops[idx] {
                            GkrOpProof::Rope(p) => p,
                            _ => return false,
                        };
                        idx += 1;
                        if !verify_rope_committed(challenger, &p.rp, values, &mut acc) {
                            return false;
                        }
                    }
                }
                shapes.insert(output.as_str(), sh);
            }
            Op::Reshape {
                input,
                output,
                shape,
            } => {
                let p = match &proof.ops[idx] {
                    GkrOpProof::Reshape(p) => p,
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
            Op::Scores {
                q,
                k,
                output,
                scale,
            } => {
                let qsh = shapes[q.as_str()].clone();
                let (lq, h, d) = (qsh[0], qsh[1], qsh[2]);
                let inv_q = ((1i64 << crate::model::spec::RECIP_Q) as f64 / scale).round() as i64;
                for li in 0..lq {
                    for hi in 0..h {
                        let p = match &proof.ops[idx] {
                            GkrOpProof::Scores(p) => p,
                            _ => return false,
                        };
                        idx += 1;
                        // dot = kh · qlh (GKR matvec), against the committed rows.
                        if !verify_gkr_matmul(
                            challenger,
                            &p.tie_k.row_com,
                            p.tie_k.row_dom,
                            &p.tie_q.row_com,
                            p.tie_q.row_dom,
                            &p.com_dot,
                            p.dom_dot,
                            &p.mm,
                            values,
                            &mut acc,
                        ) {
                            return false;
                        }
                        // scalar scale y = dot · inv_q, checked at a sampled MLE point.
                        let mut tr = fri::Transcript::new(challenger);
                        tr.observe(&p.com_dot);
                        tr.observe(&p.com_y);
                        let r: Vec<Ext> = (0..log2(lq)).map(|_| tr.sample_ext()).collect();
                        if !verify_mle_committed(
                            challenger,
                            &p.com_dot,
                            p.dom_dot,
                            next_pow2(lq),
                            &r,
                            p.dot_val,
                            &p.mle_dot,
                            values,
                            &mut acc,
                        ) || !verify_mle_committed(
                            challenger,
                            &p.com_y,
                            p.dom_y,
                            next_pow2(lq),
                            &r,
                            p.y_val,
                            &p.mle_y,
                            values,
                            &mut acc,
                        ) || p.y_val != p.dot_val * Ext::from(Fp::new(inv_q as u64))
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
                            ((li * h + hi) * d..(li * h + hi + 1) * d).collect();
                        let s_idx: Vec<usize> =
                            ((li * h + hi) * lq..(li * h + hi + 1) * lq).collect();
                        if !verify_tie(challenger, &q_idx, &p.tie_q, values, &mut acc)
                            || !verify_tie(challenger, &s_idx, &p.tie_out, values, &mut acc)
                            || !verify_tie(challenger, &kh_idx, &p.tie_k, values, &mut acc)
                        {
                            return false;
                        }
                    }
                }
                let _ = k;
                shapes.insert(output.as_str(), vec![lq, h, lq]);
            }
            Op::CausalSoftmax { input, output } => {
                let sh = shapes[input.as_str()].clone();
                let (lq, h, lseq) = (sh[0], sh[1], sh[2]);
                for li in 0..lq {
                    for hi in 0..h {
                        let p = match &proof.ops[idx] {
                            GkrOpProof::CausalSoftmax(p) => p,
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
                            GkrOpProof::WeightedV(p) => p,
                            _ => return false,
                        };
                        idx += 1;
                        if !verify_gkr_matmul(
                            challenger,
                            &p.tie_v.row_com,
                            p.tie_v.row_dom,
                            &p.tie_w.row_com,
                            p.tie_w.row_dom,
                            &p.com_acc,
                            p.dom_acc,
                            &p.mm,
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
            Op::Mul { a, b, output } => {
                let p = match &proof.ops[idx] {
                    GkrOpProof::Mul(p) => p,
                    _ => return false,
                };
                idx += 1;
                if !verify_mul_committed(challenger, &p.mul, values, &mut acc) {
                    return false;
                }
                shapes.insert(output.as_str(), shapes[a.as_str()].clone());
                let _ = b;
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
                        GkrOpProof::TopK(p) => p,
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
                terms: _terms,
                output,
            } => {
                let csh = shapes[choice.as_str()].clone();
                let (seq, experts) = (csh[0], csh[1]);
                let hidden = spec.dims.hidden;
                for s in 0..seq {
                    let p = match &proof.ops[idx] {
                        GkrOpProof::WeightedSum(p) => p,
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
                    ) {
                        return false;
                    }
                    let c_idx: Vec<usize> = (s * experts..(s + 1) * experts).collect();
                    let o_idx: Vec<usize> = (s * hidden..(s + 1) * hidden).collect();
                    if !verify_tie(challenger, &c_idx, &p.tie_choice, values, &mut acc)
                        || !verify_tie(challenger, &o_idx, &p.tie_out, values, &mut acc)
                        || p.b_out != b_choice
                    {
                        return false;
                    }
                }
                shapes.insert(output.as_str(), vec![seq, hidden]);
            }
        }
    }

    // Head argmax: `head_token` is the argmax of the last logits row, via a succinct
    // range check (`scores ≤ max`) + point evaluation.
    let cols = proof.head_range.n;
    // Opening order must match the prover: tie, then range check, count, MLE.
    let s_idx: Vec<usize> = ((l - 1) * cols..l * cols).collect();
    if !verify_tie(challenger, &s_idx, &proof.head_tie, values, &mut acc) {
        return false;
    }
    let range_table = commit_function_table(
        pcs,
        |x| if proof.head_max - x >= 0 { 1 } else { 0 },
        cfg.score_lo,
        cfg.score_hi,
        alpha,
        beta,
    );
    if !verify_lookup2_logup(
        challenger,
        &range_table.table.com_ft,
        range_table.table.dom_ft,
        &proof.head_range,
        values,
        &mut acc,
    ) {
        return false;
    }
    let ones_fp = pad(vec![Fp::ONE; cols], cols);
    if !verify_inner_product_committed(
        challenger,
        &proof.head_range.com_out,
        proof.head_range.dom_out,
        &ones_fp,
        Fp::new(cols as u64),
        &proof.head_count,
        values,
        &mut acc,
    ) {
        return false;
    }
    // scores[head_token] == head_max (point evaluation at the bound token).
    match head_token {
        Some(h) if h < cols => {
            let log_cols = log2(next_pow2(cols));
            let point: Vec<Ext> = (0..log_cols)
                .map(|k| {
                    if (h >> k) & 1 == 1 {
                        Ext::ONE
                    } else {
                        Ext::ZERO
                    }
                })
                .collect();
            if !verify_mle_committed(
                challenger,
                &proof.head_range.com_in,
                proof.head_range.dom_in,
                next_pow2(cols),
                &point,
                Ext::from(f(proof.head_max)),
                &proof.head_mle,
                values,
                &mut acc,
            ) {
                return false;
            }
        }
        _ => return false,
    }
    fri::open_many_verify(pcs, challenger, &acc, fri_proof)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::spec::{exp_q15, rmsnorm_vec, rope_apply, round_div, RECIP_Q};
    use crate::model::{Cell, Segment, WeightMatrix};

    fn cfg() -> CommittedConfig {
        CommittedConfig {
            requant_digits: 4,
            silu_lo: -100,
            silu_hi: 100,
            exp_lo: -100,
            sign_lo: -10000,
            sign_hi: 10000,
            score_lo: -128,
            score_hi: 127,
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
        let head_token = (0..64)
            .max_by_key(|&i| logits[(l - 1) * 64 + i] * 64 - i as i64)
            .unwrap();
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
    fn gkr_succinct_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        let spec: ModelSpec = serde_json::from_str(include_str!("../../specs/demo.json")).unwrap();
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
        let proof = prove_gkr_succinct(&mut ctx, &spec, &weights, &trace, &window, &cfg());
        assert!(verify_gkr_succinct(
            &ctx.pcs,
            &spec,
            &window,
            Some(head_token),
            &cfg(),
            &proof,
        ));
    }

    #[test]
    fn gkr_succinct_norm_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        let spec: ModelSpec =
            serde_json::from_str(include_str!("../../specs/norm_demo.json")).unwrap();
        let mut seed = 7u64;
        let weights = vec![
            WeightMatrix::new("embed_tokens", 8, 4, rand_i8(&mut seed, 8 * 4)),
            WeightMatrix::new("norm_gamma", 1, 4, vec![1, 1, 1, 1]),
            WeightMatrix::new("lm_head", 8, 4, rand_i8(&mut seed, 8 * 4)),
        ];
        let window = [1usize, 5];
        let l = window.len();
        let get = |n: &str| weights.iter().find(|w| w.name == n).unwrap();
        let embed = get("embed_tokens");
        let gamma = get("norm_gamma");
        let head = get("lm_head");
        let mut embedding = vec![0i64; l * 4];
        for (b, &tok) in window.iter().enumerate() {
            for j in 0..4 {
                embedding[b * 4 + j] = embed.values[tok * 4 + j] as i64;
            }
        }
        let mut normed = vec![0i64; l * 4];
        for b in 0..l {
            let row = crate::model::spec::rmsnorm_vec(
                &embedding[b * 4..(b + 1) * 4],
                &gamma.values,
                1e-6,
            );
            normed[b * 4..(b + 1) * 4].copy_from_slice(&row);
        }
        let mut logits = vec![0i64; l * 8];
        for b in 0..l {
            for i in 0..8 {
                let acc: i64 = (0..4)
                    .map(|j| head.values[i * 4 + j] as i64 * normed[b * 4 + j])
                    .sum();
                logits[b * 8 + i] = requant(acc, 4);
            }
        }
        let head_token = (0..8)
            .max_by_key(|&i| logits[(l - 1) * 8 + i] * 8 - i as i64)
            .unwrap();
        let cells = vec![
            Cell::new("embedding", l, 4, embedding),
            Cell::new("normed", l, 4, normed),
            Cell::new("logits", l, 8, logits),
        ];
        let trace = Trace {
            segments: vec![Segment {
                cells,
                head_token: Some(head_token),
            }],
        };
        let proof = prove_gkr_succinct(&mut ctx, &spec, &weights, &trace, &window, &cfg());
        assert!(verify_gkr_succinct(
            &ctx.pcs,
            &spec,
            &window,
            Some(head_token),
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
    fn gkr_succinct_attention_honest() {
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
        let head_token = (0..8)
            .max_by_key(|&i| logits[(l - 1) * 8 + i] * 8 - i as i64)
            .unwrap();

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
        let proof = prove_gkr_succinct(&mut ctx, &spec, &weights, &trace, &window, &cfg());
        assert!(verify_gkr_succinct(
            &ctx.pcs,
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
    fn gkr_succinct_moe_honest() {
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
        let head_token = (0..4)
            .max_by_key(|&i| logits[(l - 1) * 4 + i] * 4 - i as i64)
            .unwrap();

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
        let proof = prove_gkr_succinct(&mut ctx, &spec, &weights, &trace, &window, &cfg());
        assert!(verify_gkr_succinct(
            &ctx.pcs,
            &spec,
            &window,
            Some(head_token),
            &cfg(),
            &proof,
        ));
    }

    #[test]
    fn gkr_succinct_mul_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        let spec: ModelSpec = serde_json::from_str(
            r#"{
            "dims": {"hidden": 2, "vocab": 4, "layers": 1, "heads": 1, "head_dim": 2,
                      "experts": 1, "active": 1, "intermediate": 2},
            "weights": [
                {"name": "embed_tokens", "rows": 4, "cols": 2},
                {"name": "w1", "rows": 2, "cols": 2},
                {"name": "w2", "rows": 2, "cols": 2},
                {"name": "lm_head", "rows": 4, "cols": 2}
            ],
            "ops": [
                {"op": "embed", "weight": "embed_tokens", "output": "embedding"},
                {"op": "matmul", "input": "embedding", "weight": "w1", "output": "a", "shift": 4},
                {"op": "matmul", "input": "embedding", "weight": "w2", "output": "b", "shift": 4},
                {"op": "mul", "a": "a", "b": "b", "output": "c"},
                {"op": "matmul", "input": "c", "weight": "lm_head", "output": "logits", "shift": 4}
            ],
            "cells": ["embedding", "a", "b", "c", "logits"]
        }"#,
        )
        .unwrap();
        let mut seed = 5u64;
        let weights = vec![
            WeightMatrix::new("embed_tokens", 4, 2, rand_i8(&mut seed, 8)),
            WeightMatrix::new("w1", 2, 2, rand_i8(&mut seed, 4)),
            WeightMatrix::new("w2", 2, 2, rand_i8(&mut seed, 4)),
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
        let matmul_batch = |w: &[i8], x: &[i64], out: usize| -> Vec<i64> {
            let mut r = vec![0i64; l * out];
            for b in 0..l {
                for i in 0..out {
                    let acc: i64 = (0..d).map(|j| w[i * d + j] as i64 * x[b * d + j]).sum();
                    r[b * out + i] = requant(acc, 4);
                }
            }
            r
        };
        let a = matmul_batch(&weights[1].values, &embedding, 2);
        let b = matmul_batch(&weights[2].values, &embedding, 2);
        let c: Vec<i64> = a.iter().zip(&b).map(|(&x, &y)| x * y).collect();
        let logits = matmul_batch(&weights[3].values, &c, 4);
        let head_token = (0..4)
            .max_by_key(|&i| logits[(l - 1) * 4 + i] * 4 - i as i64)
            .unwrap();
        let mk = |n: &str, rows: usize, cols: usize, v: Vec<i64>| Cell::new(n, rows, cols, v);
        let cells = vec![
            mk("embedding", l, d, embedding),
            mk("a", l, 2, a),
            mk("b", l, 2, b),
            mk("c", l, 2, c),
            mk("logits", l, 4, logits),
        ];
        let trace = Trace {
            segments: vec![Segment {
                cells,
                head_token: Some(head_token),
            }],
        };
        let proof = prove_gkr_succinct(&mut ctx, &spec, &weights, &trace, &window, &cfg());
        assert!(verify_gkr_succinct(
            &ctx.pcs,
            &spec,
            &window,
            Some(head_token),
            &cfg(),
            &proof,
        ));
    }

    #[test]
    fn committed_config_derive_covers_trace() {
        let spec: ModelSpec = serde_json::from_str(
            r#"{
            "dims": {"hidden": 4, "vocab": 4, "layers": 1, "heads": 1, "head_dim": 4,
                      "experts": 2, "active": 1, "intermediate": 4},
            "weights": [],
            "ops": [
                {"op": "silu", "input": "gate", "output": "act"},
                {"op": "causal_softmax", "input": "scores", "output": "probs"},
                {"op": "top_k", "input": "router", "output": "router", "choice": "choice", "k": 1}
            ],
            "cells": ["gate", "act", "scores", "probs", "router", "choice", "logits"]
        }"#,
        )
        .unwrap();
        let cells = vec![
            Cell::new("gate", 1, 4, vec![40000, -32000, 0, 100]),
            Cell::new("scores", 1, 4, vec![5000, 3000, -2000, 1000]),
            Cell::new("router", 1, 2, vec![100, -50]),
            Cell::new("logits", 1, 4, vec![10, -5, 3, 7]),
        ];
        let trace = Trace {
            segments: vec![Segment {
                cells,
                head_token: Some(0),
            }],
        };
        let cfg = CommittedConfig::derive(&spec, &trace);
        assert!(cfg.silu_lo <= -32000 && cfg.silu_hi >= 40000);
        assert_eq!(cfg.exp_lo, -7000); // min(scores) - max(scores)
        assert!(cfg.sign_lo <= -302 && cfg.sign_hi >= 302); // (max-min)*experts ± experts
        assert!(cfg.score_lo <= -5 && cfg.score_hi >= 10);
    }

    #[test]
    fn gkr_succinct_non_pow2_vocab() {
        // vocab = 10 (not a power of two) exercises the power-of-two padding in
        // the embed lookup and the lm_head batched matmul.
        let mut ctx = fri::Prover::new(1, 10);
        let spec: ModelSpec = serde_json::from_str(
            r#"{
            "dims": {"hidden": 4, "vocab": 10, "layers": 1, "heads": 1, "head_dim": 4,
                      "experts": 1, "active": 1, "intermediate": 4},
            "weights": [
                {"name": "embed_tokens", "rows": 10, "cols": 4},
                {"name": "lm_head", "rows": 10, "cols": 4}
            ],
            "ops": [
                {"op": "embed", "weight": "embed_tokens", "output": "embedding"},
                {"op": "matmul", "input": "embedding", "weight": "lm_head", "output": "logits", "shift": 4}
            ],
            "cells": ["embedding", "logits"]
        }"#,
        )
        .unwrap();
        let mut seed = 11u64;
        let weights = vec![
            WeightMatrix::new("embed_tokens", 10, 4, rand_i8(&mut seed, 40)),
            WeightMatrix::new("lm_head", 10, 4, rand_i8(&mut seed, 40)),
        ];
        let window = [1usize, 3, 7];
        let l = window.len();
        let d = 4usize;
        let embed = &weights[0];
        let mut embedding = vec![0i64; l * d];
        for (b, &tok) in window.iter().enumerate() {
            for j in 0..d {
                embedding[b * d + j] = embed.values[tok * d + j] as i64;
            }
        }
        let mut logits = vec![0i64; l * 10];
        for b in 0..l {
            for i in 0..10 {
                let acc: i64 = (0..d)
                    .map(|j| weights[1].values[i * d + j] as i64 * embedding[b * d + j])
                    .sum();
                logits[b * 10 + i] = requant(acc, 4);
            }
        }
        let head_token = (0..10)
            .max_by_key(|&i| logits[(l - 1) * 10 + i] * 10 - i as i64)
            .unwrap();
        let mk = |n: &str, rows: usize, cols: usize, v: Vec<i64>| Cell::new(n, rows, cols, v);
        let cells = vec![
            mk("embedding", l, d, embedding),
            mk("logits", l, 10, logits),
        ];
        let trace = Trace {
            segments: vec![Segment {
                cells,
                head_token: Some(head_token),
            }],
        };
        let proof = prove_gkr_succinct(&mut ctx, &spec, &weights, &trace, &window, &cfg());
        assert!(verify_gkr_succinct(
            &ctx.pcs,
            &spec,
            &window,
            Some(head_token),
            &cfg(),
            &proof,
        ));
    }
}
