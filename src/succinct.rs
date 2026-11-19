//! Final assembly: the succinct verifier that walks a model's op graph and checks
//! each op against the committed weights + trace via the per-op primitives,
//! replacing the O(n) forward recompute in [`crate::spec::SpecModel::check`].
//!
//! Every op in [`crate::spec::Op`] is wired:
//!
//!   - embed → window lookup into the embedding rows
//!   - norm → [`crate::ops::verify_norm`]
//!   - matmul → IPA ([`crate::ffn::verify_matmul`]) + requant lookup
//!   - rope → [`crate::ops::verify_rope`] per (position, head)
//!   - reshape → shape tracking (data unchanged)
//!   - scores → IPA + `round(·/scale)` lookup, per (position, head)
//!   - causal_softmax → [`crate::ops::verify_causal_softmax_row`] per (position, head)
//!   - weighted_v → IPA + requant lookup, per (position, head)
//!   - silu → [`crate::lookup::verify_silu`]
//!   - mul → degree-3 sumcheck ([`crate::ffn::verify_mul`]) per row
//!   - add → linear elementwise identity
//!   - top_k → [`crate::ops::verify_topk`] per row
//!   - weighted_sum → [`crate::ffn::verify_weighted_sum`] (IPA over experts)
//!
//! Tensor shapes are reconstructed by walking the `reshape` ops, so the 3-D
//! attention tensors (`scores`, `causal_softmax`, `weighted_v`, `rope`) are handled
//! by the same per-(position, head) arguments.

use std::collections::HashMap;

use crate::ffn::{verify_matmul, verify_mul, verify_weighted_sum};
use crate::field::{Ext, Fp};
use crate::lookup::{verify_function_lookup, verify_requant, verify_silu};
use crate::model::{Trace, WeightMatrix, Weights};
use crate::ops::{verify_causal_softmax_row, verify_norm, verify_rope, verify_topk};
use crate::spec::{round_div, ModelSpec, Op};

/// `i64 -> Fp` (Goldilocks), with negative values mapped to `−Fp(|x|)`.
fn f(x: i64) -> Fp {
    if x >= 0 {
        Fp::new(x as u64)
    } else {
        -Fp::new((-x) as u64)
    }
}

/// The bounded integer domains the lookups must cover. Weights and activations
/// stay within these in the integer model; the domains are fixed (not adapted to
/// the query) so the lookups stay sound.
pub struct SuccinctConfig {
    pub requant_lo: i64,
    pub requant_hi: i64,
    pub attn_requant_lo: i64,
    pub attn_requant_hi: i64,
    pub silu_lo: i64,
    pub silu_hi: i64,
    pub exp_lo: i64,
    pub round_lo: i64,
    pub round_hi: i64,
}

/// Verify a full op graph against the committed weights and trace.
///
/// `window` is the input token sequence (for the `embed` lookup); `cfg` bounds the
/// lookup tables. Returns `true` iff every op's succinct argument verifies and the
/// head token is the argmax of the last position's logits.
pub fn verify_succinct(
    spec: &ModelSpec,
    weights: &Weights,
    trace: &Trace,
    window: &[usize],
    cfg: &SuccinctConfig,
) -> bool {
    if trace.segments.len() != 1 {
        return false;
    }
    let seg = &trace.segments[0];
    let l = window.len();

    // name -> (values, rows, cols).
    let cells: HashMap<&str, (&[i64], usize, usize)> = seg
        .cells
        .iter()
        .map(|c| (c.name.as_str(), (c.values.as_slice(), c.rows, c.cols)))
        .collect();
    let wmap: HashMap<&str, &WeightMatrix> = weights.iter().map(|w| (w.name.as_str(), w)).collect();
    // name -> full tensor shape (reconstructed from the op graph).
    let mut shapes: HashMap<&str, Vec<usize>> = HashMap::new();

    // Fixed challenges (the primitives are proven sound for random ones; a full
    // protocol derives these from a Fiat–Shamir transcript).
    let alpha = Ext::from(Fp::new(11));
    let beta = Ext::from(Fp::new(5));

    let mut ctr = 0u64;
    let mut sample_fp = move || {
        ctr += 1;
        Fp::new(ctr)
    };
    let mut ctr_ext = 0u64;
    let mut sample_ext = move |_i: usize| {
        ctr_ext += 1;
        Ext::from(Fp::new(ctr_ext))
    };

    for op in &spec.ops {
        match op {
            Op::Embed { weight, output } => {
                let t = wmap[weight.as_str()];
                let d = t.cols; // hidden
                let (out, rows, cols) = cells[output.as_str()];
                if rows != l || cols != d {
                    return false;
                }
                for (b, &tok) in window.iter().enumerate() {
                    if tok >= t.rows {
                        return false;
                    }
                    for j in 0..d {
                        if out[b * d + j] != t.values[tok * d + j] as i64 {
                            return false;
                        }
                    }
                }
                shapes.insert(output.as_str(), vec![l, d]);
            }
            Op::Norm {
                input,
                weight,
                output,
                eps,
            } => {
                let g = wmap[weight.as_str()];
                let (x, rows, cols) = cells[input.as_str()];
                let (y, orows, ocols) = cells[output.as_str()];
                if orows != rows || ocols != cols || g.values.len() != cols {
                    return false;
                }
                for b in 0..rows {
                    let xb = &x[b * cols..(b + 1) * cols];
                    let yb = &y[b * cols..(b + 1) * cols];
                    if !verify_norm(xb, &g.values, yb, *eps) {
                        return false;
                    }
                }
                shapes.insert(output.as_str(), shapes[input.as_str()].clone());
            }
            Op::Matmul {
                input,
                weight,
                output,
                shift,
            } => {
                let t = wmap[weight.as_str()];
                let out_dim = t.rows;
                let in_dim = t.cols;
                let (x, rows, xcols) = cells[input.as_str()];
                let (y, orows, ocols) = cells[output.as_str()];
                if xcols != in_dim || orows != rows || ocols != out_dim {
                    return false;
                }
                let wf: Vec<Fp> = t.values.iter().map(|&v| f(v as i64)).collect();
                for b in 0..rows {
                    let xb = &x[b * in_dim..(b + 1) * in_dim];
                    let acc: Vec<i64> = (0..out_dim)
                        .map(|i| {
                            (0..in_dim)
                                .map(|j| t.values[i * in_dim + j] as i64 * xb[j])
                                .sum()
                        })
                        .collect();
                    let xf: Vec<Fp> = xb.iter().map(|&v| f(v)).collect();
                    let af: Vec<Fp> = acc.iter().map(|&v| f(v)).collect();
                    if !verify_matmul(&wf, &xf, &af, out_dim, &mut sample_fp) {
                        return false;
                    }
                    let yb = &y[b * out_dim..(b + 1) * out_dim];
                    if !verify_requant(
                        &acc,
                        yb,
                        *shift,
                        cfg.requant_lo,
                        cfg.requant_hi,
                        alpha,
                        beta,
                    ) {
                        return false;
                    }
                }
                let mut s = shapes[input.as_str()].clone();
                let n = s.len();
                s[n - 1] = out_dim;
                shapes.insert(output.as_str(), s);
            }
            Op::Rope {
                input,
                output,
                theta,
            } => {
                let sh = shapes[input.as_str()].clone(); // [L, H, d]
                if sh.len() != 3 {
                    return false;
                }
                let (h, d) = (sh[1], sh[2]);
                let (x, rows, cols) = cells[input.as_str()];
                let (y, orows, ocols) = cells[output.as_str()];
                if rows != sh[0] || cols != h * d || orows != rows || ocols != cols {
                    return false;
                }
                for li in 0..sh[0] {
                    for hi in 0..h {
                        let base = li * h * d + hi * d;
                        let xb = &x[base..base + d];
                        let yb = &y[base..base + d];
                        if !verify_rope(
                            xb,
                            yb,
                            li,
                            *theta,
                            cfg.attn_requant_lo,
                            cfg.attn_requant_hi,
                            alpha,
                            beta,
                        ) {
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
                let ish = shapes[input.as_str()].clone();
                let numel: usize = ish.iter().product();
                let neg = shape.iter().filter(|&&s| s == -1).count();
                if neg > 1 {
                    return false;
                }
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
                if resolved.iter().product::<usize>() != numel {
                    return false;
                }
                // Data is unchanged (reshape is a pure permutation of the layout).
                let (x, rows, cols) = cells[input.as_str()];
                let (y, orows, ocols) = cells[output.as_str()];
                if orows != rows || ocols != cols || x != y {
                    return false;
                }
                shapes.insert(output.as_str(), resolved);
            }
            Op::Scores {
                q,
                k,
                output,
                scale,
            } => {
                let qsh = shapes[q.as_str()].clone(); // [L, H, d]
                let ksh = shapes[k.as_str()].clone();
                if qsh != ksh || qsh.len() != 3 {
                    return false;
                }
                let (lq, h, d) = (qsh[0], qsh[1], qsh[2]);
                let (qv, qrows, qcols) = cells[q.as_str()];
                let (kv, krows, kcols) = cells[k.as_str()];
                let (sv, srows, scols) = cells[output.as_str()];
                if qrows != lq
                    || qcols != h * d
                    || krows != lq
                    || kcols != h * d
                    || srows != lq
                    || scols != h * lq
                {
                    return false;
                }
                for li in 0..lq {
                    for hi in 0..h {
                        // K_h[t, j] = k[t, h, j]  (a [L, d] matrix).
                        let mut kh = Vec::with_capacity(lq * d);
                        for t in 0..lq {
                            for j in 0..d {
                                kh.push(kv[t * h * d + hi * d + j]);
                            }
                        }
                        let qlh: Vec<i64> = (0..d).map(|j| qv[li * h * d + hi * d + j]).collect();
                        let dot: Vec<i64> = (0..lq)
                            .map(|t| (0..d).map(|j| kh[t * d + j] * qlh[j]).sum())
                            .collect();
                        let khf: Vec<Fp> = kh.iter().map(|&v| f(v)).collect();
                        let qlhf: Vec<Fp> = qlh.iter().map(|&v| f(v)).collect();
                        let dotf: Vec<Fp> = dot.iter().map(|&v| f(v)).collect();
                        if !verify_matmul(&khf, &qlhf, &dotf, lq, &mut sample_fp) {
                            return false;
                        }
                        let srow: Vec<i64> =
                            (0..lq).map(|t| sv[li * h * lq + hi * lq + t]).collect();
                        if !verify_function_lookup(
                            &dot,
                            &srow,
                            |x| round_div(x, *scale),
                            cfg.round_lo,
                            cfg.round_hi,
                            alpha,
                            beta,
                        ) {
                            return false;
                        }
                    }
                }
                shapes.insert(output.as_str(), vec![lq, h, lq]);
            }
            Op::CausalSoftmax { input, output } => {
                let sh = shapes[input.as_str()].clone(); // [L, H, L]
                if sh.len() != 3 {
                    return false;
                }
                let (lq, h, lseq) = (sh[0], sh[1], sh[2]);
                let (x, rows, cols) = cells[input.as_str()];
                let (y, orows, ocols) = cells[output.as_str()];
                if rows != lq || cols != h * lseq || orows != rows || ocols != cols {
                    return false;
                }
                for li in 0..lq {
                    for hi in 0..h {
                        let srow: Vec<i64> = (0..lseq)
                            .map(|t| x[li * h * lseq + hi * lseq + t])
                            .collect();
                        let prow: Vec<i64> = (0..lseq)
                            .map(|t| y[li * h * lseq + hi * lseq + t])
                            .collect();
                        if !verify_causal_softmax_row(&srow, &prow, li, cfg.exp_lo, alpha, beta) {
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
                let wsh = shapes[wname.as_str()].clone(); // [L, H, L]
                let vsh = shapes[v.as_str()].clone(); // [L, H, d]
                if wsh.len() != 3 || vsh.len() != 3 || wsh[0] != vsh[0] || wsh[1] != vsh[1] {
                    return false;
                }
                let (lq, h, d) = (vsh[0], vsh[1], vsh[2]);
                let (wv, wrows, wcols) = cells[wname.as_str()];
                let (vv, vrows, vcols) = cells[v.as_str()];
                let (ov, orows, ocols) = cells[output.as_str()];
                if wrows != lq
                    || wcols != h * lq
                    || vrows != lq
                    || vcols != h * d
                    || orows != lq
                    || ocols != h * d
                {
                    return false;
                }
                for li in 0..lq {
                    for hi in 0..h {
                        // V_h^T[j, t] = v[t, h, j]  (a [d, L] matrix).
                        let mut vh = Vec::with_capacity(d * lq);
                        for j in 0..d {
                            for t in 0..lq {
                                vh.push(vv[t * h * d + hi * d + j]);
                            }
                        }
                        let wlh: Vec<i64> =
                            (0..lq).map(|t| wv[li * h * lq + hi * lq + t]).collect();
                        let acc: Vec<i64> = (0..d)
                            .map(|j| (0..lq).map(|t| vh[j * lq + t] * wlh[t]).sum())
                            .collect();
                        let vhf: Vec<Fp> = vh.iter().map(|&v| f(v)).collect();
                        let wlhf: Vec<Fp> = wlh.iter().map(|&v| f(v)).collect();
                        let accf: Vec<Fp> = acc.iter().map(|&v| f(v)).collect();
                        if !verify_matmul(&vhf, &wlhf, &accf, d, &mut sample_fp) {
                            return false;
                        }
                        let ob: Vec<i64> = (0..d).map(|j| ov[li * h * d + hi * d + j]).collect();
                        if !verify_requant(
                            &acc,
                            &ob,
                            15,
                            cfg.attn_requant_lo,
                            cfg.attn_requant_hi,
                            alpha,
                            beta,
                        ) {
                            return false;
                        }
                    }
                }
                shapes.insert(output.as_str(), vec![lq, h, d]);
            }
            Op::Silu { input, output } => {
                let (x, rows, cols) = cells[input.as_str()];
                let (y, orows, ocols) = cells[output.as_str()];
                if orows != rows || ocols != cols {
                    return false;
                }
                if !verify_silu(x, y, cfg.silu_lo, cfg.silu_hi, alpha, beta) {
                    return false;
                }
                shapes.insert(output.as_str(), shapes[input.as_str()].clone());
            }
            Op::Mul { a, b, output } => {
                let (xa, rows, cols) = cells[a.as_str()];
                let (xb, brows, bcols) = cells[b.as_str()];
                let (y, orows, ocols) = cells[output.as_str()];
                if brows != rows || bcols != cols || orows != rows || ocols != cols {
                    return false;
                }
                // Per-row degree-3 product sumcheck (row length must be a power of two).
                if !cols.is_power_of_two() {
                    return false;
                }
                let r: Vec<Ext> = (0..cols.trailing_zeros())
                    .map(|i| Ext::from(Fp::new(i as u64 + 1)))
                    .collect();
                for b in 0..rows {
                    let ar: Vec<Fp> = xa[b * cols..(b + 1) * cols].iter().map(|&v| f(v)).collect();
                    let br: Vec<Fp> = xb[b * cols..(b + 1) * cols].iter().map(|&v| f(v)).collect();
                    let yr: Vec<Fp> = y[b * cols..(b + 1) * cols].iter().map(|&v| f(v)).collect();
                    if !verify_mul(&ar, &br, &yr, &r, &mut sample_ext) {
                        return false;
                    }
                }
                shapes.insert(output.as_str(), shapes[a.as_str()].clone());
            }
            Op::Add { inputs, output } => {
                let (y, rows, cols) = cells[output.as_str()];
                let n = rows * cols;
                for i in 0..n {
                    let mut s = 0i64;
                    for inp in inputs {
                        let (x, rr, cc) = cells[inp.as_str()];
                        if rr != rows || cc != cols {
                            return false;
                        }
                        s += x[i];
                    }
                    if y[i] != s {
                        return false;
                    }
                }
                shapes.insert(output.as_str(), shapes[inputs[0].as_str()].clone());
            }
            Op::TopK {
                input,
                output,
                choice,
                k,
            } => {
                let (x, rows, cols) = cells[input.as_str()];
                let (y, orows, ocols) = cells[output.as_str()];
                let (c, crows, ccols) = cells[choice.as_str()];
                // output is unchanged; choice is the top-k indicator per row.
                if orows != rows || ocols != cols || crows != rows || ccols != cols || x != y {
                    return false;
                }
                for b in 0..rows {
                    let xb = &x[b * cols..(b + 1) * cols];
                    let cb = &c[b * cols..(b + 1) * cols];
                    if !verify_topk(xb, cb, *k, alpha, beta) {
                        return false;
                    }
                }
                shapes.insert(output.as_str(), shapes[input.as_str()].clone());
                shapes.insert(choice.as_str(), shapes[input.as_str()].clone());
            }
            Op::WeightedSum {
                choice,
                terms,
                output,
            } => {
                let csh = shapes[choice.as_str()].clone(); // [L, E]
                let (cv, crows, ccols) = cells[choice.as_str()];
                let (ov, orows, ocols) = cells[output.as_str()];
                let seq = csh[0];
                let experts = csh[1];
                if crows != seq || ccols != experts || orows != seq {
                    return false;
                }
                // Stack the terms into [E, L, hidden] and the choice into [L, E].
                let mut terms_flat = Vec::with_capacity(experts * seq * ocols);
                for e in 0..experts {
                    let (tv, trows, tcols) = cells[terms[e].as_str()];
                    if trows != seq || tcols != ocols {
                        return false;
                    }
                    terms_flat.extend_from_slice(tv);
                }
                let cf: Vec<Fp> = cv.iter().map(|&v| f(v)).collect();
                let tf: Vec<Fp> = terms_flat.iter().map(|&v| f(v)).collect();
                let of: Vec<Fp> = ov.iter().map(|&v| f(v)).collect();
                if !verify_weighted_sum(&cf, &tf, &of, seq, experts, ocols, &mut sample_fp) {
                    return false;
                }
                shapes.insert(output.as_str(), vec![seq, ocols]);
            }
        }
    }

    // head_token == argmax of the last position's logits.
    let (logits, rows, cols) = cells["logits"];
    let last = rows - 1;
    let out = &logits[last * cols..(last + 1) * cols];
    match seg.head_token {
        Some(h) => h < cols && out[h] == out.iter().copied().max().unwrap_or(0),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Cell, Segment, WeightMatrix};
    use crate::spec::{requant, silu};

    fn cfg() -> SuccinctConfig {
        SuccinctConfig {
            requant_lo: -10_000,
            requant_hi: 10_000,
            attn_requant_lo: -150_000,
            attn_requant_hi: 150_000,
            silu_lo: -100,
            silu_hi: 100,
            exp_lo: -100,
            round_lo: -1_000,
            round_hi: 1_000,
        }
    }

    // Attention intermediates (`attn_weights ∈ [0,2^15]`, rope `rot`) are much
    // larger than the dense-FFN accumulators, so the attention test needs a wider
    // requant domain.
    fn cfg_attn() -> SuccinctConfig {
        SuccinctConfig {
            attn_requant_lo: -200_000,
            attn_requant_hi: 200_000,
            ..cfg()
        }
    }

    /// Deterministic pseudo-random i8 in `[-4, 4]`.
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

    /// Forward pass for the demo FFN (embed → matmul → silu → matmul → add → matmul).
    fn forward(weights: &Weights, window: &[usize]) -> (Vec<Cell>, usize) {
        let get = |name: &str| weights.iter().find(|w| w.name == name).unwrap();
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

    fn demo_spec() -> ModelSpec {
        let s: &str = include_str!("../specs/demo.json");
        serde_json::from_str(s).unwrap()
    }

    #[test]
    fn succinct_accepts_honest_trace() {
        let spec = demo_spec();
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
        assert!(verify_succinct(&spec, &weights, &trace, &window, &cfg()));
    }

    #[test]
    fn succinct_rejects_tampered_cell() {
        let spec = demo_spec();
        let mut seed = 12345u64;
        let weights = demo_weights(&mut seed);
        let window = [3, 17, 42, 9];
        let (mut cells, head_token) = forward(&weights, &window);
        cells[2].values[0] += 1; // "act"
        let trace = Trace {
            segments: vec![Segment {
                cells,
                head_token: Some(head_token),
            }],
        };
        assert!(!verify_succinct(&spec, &weights, &trace, &window, &cfg()));
    }

    #[test]
    fn succinct_rejects_wrong_head_token() {
        let spec = demo_spec();
        let mut seed = 12345u64;
        let weights = demo_weights(&mut seed);
        let window = [3, 17, 42, 9];
        let (cells, head_token) = forward(&weights, &window);
        let trace = Trace {
            segments: vec![Segment {
                cells,
                head_token: Some((head_token + 1) % 64),
            }],
        };
        assert!(!verify_succinct(&spec, &weights, &trace, &window, &cfg()));
    }

    // A minimal multi-head-attention model (L×H×d) exercising reshape/rope/scores/
    // causal_softmax/weighted_v in the assembly.
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

    fn matmul_batch(
        w: &[i8],
        x: &[i64],
        batch: usize,
        out: usize,
        in_: usize,
        shift: u32,
    ) -> Vec<i64> {
        let mut r = vec![0i64; batch * out];
        for b in 0..batch {
            for i in 0..out {
                let acc: i64 = (0..in_)
                    .map(|j| w[i * in_ + j] as i64 * x[b * in_ + j])
                    .sum();
                r[b * out + i] = requant(acc, shift);
            }
        }
        r
    }

    #[test]
    fn succinct_attention_honest() {
        use crate::spec::{exp_q15, rope_apply, round_div};
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
        let q = matmul_batch(&weights[1].values, &embedding, l, 2, 2, 4);
        let k = matmul_batch(&weights[2].values, &embedding, l, 2, 2, 4);
        let v = matmul_batch(&weights[3].values, &embedding, l, 2, 2, 4);

        // q/k/v_heads are identical to q/k/v (reshape preserves data; H·d == hidden).
        // rope per (l, h).
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

        // causal softmax.
        let mut attn_weights = vec![0i64; l * l];
        for li in 0..l {
            let m = *scores[li * l..li * l + li + 1].iter().max().unwrap();
            let mut e = vec![0i64; l];
            for t in 0..=li {
                e[t] = exp_q15(scores[li * l + t] - m);
            }
            let s: i64 = e.iter().sum();
            for t in 0..=li {
                attn_weights[li * l + t] = e[t] * 32768 / s;
            }
        }

        // weighted_v.
        let mut attn_heads = vec![0i64; l * d];
        for li in 0..l {
            for j in 0..d {
                let acc: i64 = (0..l)
                    .map(|t| attn_weights[li * l + t] * v[t * d + j])
                    .sum();
                attn_heads[li * d + j] = requant(acc, 15);
            }
        }

        let attn = attn_heads.clone(); // reshape [L,H,d] -> [L,hidden], data unchanged
        let attn_out = matmul_batch(&weights[4].values, &attn, l, 2, 2, 4);
        let residual: Vec<i64> = embedding
            .iter()
            .zip(&attn_out)
            .map(|(&a, &b)| a + b)
            .collect();
        let logits = matmul_batch(&weights[5].values, &residual, l, 8, 2, 4);
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
        assert!(verify_succinct(
            &spec,
            &weights,
            &trace,
            &window,
            &cfg_attn()
        ));
    }
}
