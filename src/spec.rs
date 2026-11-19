//! Data-driven model spec + generic interpreter.
//!
//! Values are rank-`n` `Tensor`s (a vector is rank 1, a matrix rank 2, the
//! attention tensors rank 3). The op graph is data, so the protocol stays
//! model-agnostic.

use std::collections::HashMap;

use crate::model::{from_digits, DigitMatrix, EncodedRequest, Model, Trace, Weights};

// --- arithmetic primitives (the reference integer model) --------------------

pub fn requant(x: i64, k: u32) -> i64 {
    let half = 1i64 << (k - 1);
    if x >= 0 {
        (x + half) >> k
    } else {
        -(((-x) + half) >> k)
    }
}

/// Fixed-point bits for the quantized reciprocal (`1/scale`, `1/√(·+eps)`). Chosen
/// so every input `|x| < 2^RECIP_Q` (the scores dot and norm `x·γ` stay well below
/// this), making the integer `requant(x·inv_q, RECIP_Q)` bit-consistent with the
/// f64 `round(x/divisor)`.
pub const RECIP_Q: u32 = 24;

/// `round(x / scale)` — the attention-scores rounding, now a quantized reciprocal
/// so it is a clean shift (arith: `requant(x · round(2^Q/scale), Q)`).
pub fn round_div(x: i64, scale: f64) -> i64 {
    let inv_q = ((1i64 << RECIP_Q) as f64 / scale).round() as i64;
    requant(x * inv_q, RECIP_Q)
}

pub fn silu(x: i64) -> i64 {
    let xc = x.clamp(-(1 << 15), (1 << 15) - 1);
    let xf = xc as f64;
    (xf / (1.0 + (-xf).exp())).round() as i64
}

pub fn exp_q15(x: i64) -> i64 {
    let xc = x.clamp(-(1 << 15), 0);
    ((xc as f64).exp() * 32768.0).round() as i64
}

/// RoPE: rotate one head-dim vector by `pos`.
pub fn rope_apply(q: &mut [i64], pos: usize, theta: f64) {
    let d = q.len();
    for i in (0..d).step_by(2) {
        let angle = (pos as f64) * theta.powf(-(i as f64) / d as f64);
        let c = (angle.cos() * 32768.0).round() as i64;
        let s = (angle.sin() * 32768.0).round() as i64;
        let (a, b) = (q[i], q[i + 1]);
        q[i] = requant(a * c - b * s, 15);
        q[i + 1] = requant(a * s + b * c, 15);
    }
}

/// Integer RMSNorm of one vector, via the quantized reciprocal.
pub fn rmsnorm_vec(x: &[i64], gamma: &[i8], eps: f64) -> Vec<i64> {
    let d = x.len() as f64;
    let mean_sq = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / d;
    let inv_q = ((1i64 << RECIP_Q) as f64 / (mean_sq + eps).sqrt()).round() as i64;
    x.iter()
        .zip(gamma)
        .map(|(&v, &g)| requant(v * (g as i64) * inv_q, RECIP_Q))
        .collect()
}

// --- tensors -----------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<i64>, // row-major, last dim contiguous
}

impl Tensor {
    pub fn vec(data: Vec<i64>) -> Self {
        let n = data.len();
        Self {
            shape: vec![n],
            data,
        }
    }
    pub fn tensor(shape: Vec<usize>, data: Vec<i64>) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            n,
            "tensor shape {shape:?} vs {} elements",
            data.len()
        );
        Self { shape, data }
    }
    pub fn numel(&self) -> usize {
        self.data.len()
    }
    pub fn map(&self, f: impl Fn(i64) -> i64) -> Self {
        Self {
            shape: self.shape.clone(),
            data: self.data.iter().map(|&x| f(x)).collect(),
        }
    }
}

/// out = requant(W · x, shift). W is `out × in`; x's last dim is `in`.
fn matmul(w: &[i8], out: usize, in_: usize, x: &Tensor, shift: u32) -> Tensor {
    assert_eq!(*x.shape.last().unwrap(), in_);
    let batch: usize = x.shape[..x.shape.len() - 1].iter().product();
    let mut out_shape = x.shape[..x.shape.len() - 1].to_vec();
    out_shape.push(out);
    let mut data = Vec::with_capacity(batch * out);
    for b in 0..batch {
        let base = b * in_;
        for i in 0..out {
            let mut acc = 0i64;
            for j in 0..in_ {
                acc += w[i * in_ + j] as i64 * x.data[base + j];
            }
            data.push(requant(acc, shift));
        }
    }
    Tensor::tensor(out_shape, data)
}

/// RMSNorm applied to each last-dim row.
fn norm_rows(x: &Tensor, gamma: &[i8], eps: f64) -> Tensor {
    let d = *x.shape.last().unwrap();
    let batch = x.numel() / d;
    let mut data = Vec::with_capacity(x.numel());
    for b in 0..batch {
        data.extend(rmsnorm_vec(&x.data[b * d..(b + 1) * d], gamma, eps));
    }
    Tensor::tensor(x.shape.clone(), data)
}

/// RoPE applied per (seq, head) row; position = sequence index.
fn rope(x: &Tensor, theta: f64) -> Tensor {
    let d = *x.shape.last().unwrap();
    let seq = x.shape[0];
    let heads = if x.shape.len() >= 3 { x.shape[1] } else { 1 };
    let mut data = Vec::with_capacity(x.numel());
    for li in 0..seq {
        for hi in 0..heads {
            let base = (li * heads + hi) * d;
            let mut v = x.data[base..base + d].to_vec();
            rope_apply(&mut v, li, theta);
            data.extend(v);
        }
    }
    Tensor::tensor(x.shape.clone(), data)
}

/// scores[l][h][t] = q[l][h]·k[t][h] / scale. q,k: [L, H, d] -> [L, H, L].
fn scores(q: &Tensor, k: &Tensor, scale: f64) -> Tensor {
    let (l, h, d) = (q.shape[0], q.shape[1], q.shape[2]);
    let mut data = Vec::with_capacity(l * h * l);
    for li in 0..l {
        for hi in 0..h {
            let qbase = (li * h + hi) * d;
            for t in 0..l {
                let kbase = (t * h + hi) * d;
                let mut dot = 0i64;
                for j in 0..d {
                    dot += q.data[qbase + j] * k.data[kbase + j];
                }
                data.push(round_div(dot, scale));
            }
        }
    }
    Tensor::tensor(vec![l, h, l], data)
}

/// causal softmax over the last dim (t <= l), q15. [L, H, L] -> [L, H, L].
fn causal_softmax(s: &Tensor) -> Tensor {
    let (l, h) = (s.shape[0], s.shape[1]);
    let mut data = Vec::with_capacity(l * h * l);
    for li in 0..l {
        for hi in 0..h {
            let base = (li * h + hi) * l;
            let row = &s.data[base..base + l];
            let max = row[..=li].iter().copied().max().unwrap_or(0);
            let exps: Vec<i64> = (0..=li).map(|t| exp_q15(row[t] - max)).collect();
            let sum: i64 = exps.iter().sum();
            // Quantized reciprocal of S: nrm = requant(e·inv_q, RECIP_Q) ≈ e·2^15/S.
            let inv_q = if sum != 0 {
                ((1i64 << (RECIP_Q + 15)) as f64 / sum as f64).round() as i64
            } else {
                0
            };
            for t in 0..l {
                let w = if t <= li && sum != 0 {
                    requant(exps[t] * inv_q, RECIP_Q)
                } else {
                    0
                };
                data.push(w);
            }
        }
    }
    Tensor::tensor(s.shape.clone(), data)
}

/// out[l][h][j] = Σ_t weights[l][h][t] · v[t][h][j]. weights [L,H,L], v [L,H,d] -> [L,H,d].
fn weighted_v(weights: &Tensor, v: &Tensor) -> Tensor {
    let (l, h, d) = (v.shape[0], v.shape[1], v.shape[2]);
    let mut data = Vec::with_capacity(l * h * d);
    for li in 0..l {
        for hi in 0..h {
            let wbase = (li * h + hi) * l;
            for j in 0..d {
                let mut acc = 0i64;
                for t in 0..l {
                    acc += weights.data[wbase + t] * v.data[(t * h + hi) * d + j];
                }
                data.push(requant(acc, 15));
            }
        }
    }
    Tensor::tensor(vec![l, h, d], data)
}

// --- spec types --------------------------------------------------------------

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Dims {
    pub hidden: usize,
    pub vocab: usize,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub experts: usize,
    pub active: usize,
    pub intermediate: usize,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct WeightSpec {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Op {
    /// window -> embedding rows [L, hidden].
    Embed { weight: String, output: String },
    /// RMSNorm (per last-dim row).
    Norm {
        input: String,
        weight: String,
        output: String,
        eps: f64,
    },
    /// out = requant(W · input, shift), on the last dim.
    Matmul {
        input: String,
        weight: String,
        output: String,
        shift: u32,
    },
    /// RoPE rotation, position = sequence index.
    Rope {
        input: String,
        output: String,
        theta: f64,
    },
    /// reshape (a `-1` dim is inferred).
    Reshape {
        input: String,
        output: String,
        shape: Vec<i64>,
    },
    /// attention scores = q·k^T / scale (3-D).
    Scores {
        q: String,
        k: String,
        output: String,
        scale: f64,
    },
    /// causal softmax over the last dim (3-D).
    CausalSoftmax { input: String, output: String },
    /// output = weights · v (3-D).
    WeightedV {
        weights: String,
        v: String,
        output: String,
    },
    /// SiLU lookup, elementwise.
    Silu { input: String, output: String },
    /// elementwise multiply.
    Mul {
        a: String,
        b: String,
        output: String,
    },
    /// elementwise add.
    Add { inputs: Vec<String>, output: String },
    /// top-k on the last dim, choice is an indicator.
    TopK {
        input: String,
        output: String,
        choice: String,
        k: usize,
    },
    /// MoE weighted sum: out[s] = Σ_e choice[s][e] · term_e[s].
    WeightedSum {
        choice: String,
        terms: Vec<String>,
        output: String,
    },
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ModelSpec {
    pub dims: Dims,
    pub weights: Vec<WeightSpec>,
    pub ops: Vec<Op>,
    pub cells: Vec<String>,
}

// --- generic interpreter -----------------------------------------------------

struct Wt {
    rows: usize,
    cols: usize,
    values: Vec<i8>,
}

pub struct SpecModel {
    pub spec: ModelSpec,
}

impl SpecModel {
    /// Upper bound on the number of weight values per segment. Segments are
    /// contiguous value ranges of the flat digit sequence (tensors may be split
    /// across segments), so the per-segment Merkle tree and encoded table stay
    /// bounded in memory regardless of model size.
    const SEGMENT_VALUES: usize = 24_000_000;

    fn total_values(&self) -> usize {
        self.spec.weights.iter().map(|w| w.rows * w.cols).sum()
    }

    fn weights_from(&self, a: &[DigitMatrix]) -> HashMap<String, Wt> {
        // The segment digit matrices are concatenated (in segment order) to form
        // the single logical digit sequence that `spec.weights` indexes into.
        let mut map = HashMap::new();
        let mut seg = 0usize;
        let mut pos = 0usize;
        for w in &self.spec.weights {
            let n = w.rows * w.cols;
            let mut values = Vec::with_capacity(n);
            for _ in 0..n {
                while pos >= a[seg].entries.len() {
                    seg += 1;
                    pos = 0;
                }
                values.push(from_digits(&a[seg].entries[pos..pos + 1]) as i8);
                pos += 1;
            }
            map.insert(
                w.name.clone(),
                Wt {
                    rows: w.rows,
                    cols: w.cols,
                    values,
                },
            );
        }
        map
    }

    fn eval_op(
        &self,
        op: &Op,
        env: &mut HashMap<String, Tensor>,
        weights: &HashMap<String, Wt>,
        window: &[usize],
    ) -> bool {
        match op {
            Op::Embed { weight, output } => {
                let t = &weights[weight];
                let d = self.spec.dims.hidden;
                let mut data = Vec::with_capacity(window.len() * d);
                for &tok in window {
                    if tok >= t.rows {
                        return false;
                    }
                    data.extend(t.values[tok * d..(tok + 1) * d].iter().map(|&v| v as i64));
                }
                env.insert(output.clone(), Tensor::tensor(vec![window.len(), d], data));
            }
            Op::Norm {
                input,
                weight,
                output,
                eps,
            } => {
                let x = &env[input];
                let g = &weights[weight].values;
                env.insert(output.clone(), norm_rows(x, g, *eps));
            }
            Op::Matmul {
                input,
                weight,
                output,
                shift,
            } => {
                let x = &env[input];
                let t = &weights[weight];
                env.insert(output.clone(), matmul(&t.values, t.rows, t.cols, x, *shift));
            }
            Op::Rope {
                input,
                output,
                theta,
            } => {
                env.insert(output.clone(), rope(&env[input], *theta));
            }
            Op::Reshape {
                input,
                output,
                shape,
            } => {
                let x = env[input].clone();
                let numel = x.numel();
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
                env.insert(output.clone(), Tensor::tensor(resolved, x.data));
            }
            Op::Scores {
                q,
                k,
                output,
                scale,
            } => {
                env.insert(output.clone(), scores(&env[q], &env[k], *scale));
            }
            Op::CausalSoftmax { input, output } => {
                env.insert(output.clone(), causal_softmax(&env[input]));
            }
            Op::WeightedV {
                weights: w,
                v,
                output,
            } => {
                env.insert(output.clone(), weighted_v(&env[w], &env[v]));
            }
            Op::Silu { input, output } => {
                env.insert(output.clone(), env[input].map(silu));
            }
            Op::Mul { a, b, output } => {
                let (x, y) = (&env[a], &env[b]);
                if x.shape != y.shape {
                    return false;
                }
                let data = x.data.iter().zip(&y.data).map(|(&p, &q)| p * q).collect();
                env.insert(output.clone(), Tensor::tensor(x.shape.clone(), data));
            }
            Op::Add { inputs, output } => {
                let first = env[&inputs[0]].clone();
                let mut data = first.data.clone();
                for rest in &inputs[1..] {
                    if env[rest].shape != first.shape {
                        return false;
                    }
                    for (p, &q) in data.iter_mut().zip(&env[rest].data) {
                        *p += q;
                    }
                }
                env.insert(output.clone(), Tensor::tensor(first.shape.clone(), data));
            }
            Op::TopK {
                input,
                output,
                choice,
                k,
            } => {
                let scores_t = env[input].clone();
                let e = *scores_t.shape.last().unwrap();
                let batch = scores_t.numel() / e;
                let mut out = Vec::with_capacity(scores_t.numel());
                let mut ind = Vec::with_capacity(scores_t.numel());
                for b in 0..batch {
                    let row = &scores_t.data[b * e..(b + 1) * e];
                    let mut idx: Vec<usize> = (0..e).collect();
                    idx.sort_by(|&i, &j| row[j].cmp(&row[i]).then(i.cmp(&j)));
                    for c in 0..e {
                        out.push(row[c]);
                        ind.push(if idx[..(*k).min(e)].contains(&c) {
                            1
                        } else {
                            0
                        });
                    }
                }
                env.insert(output.clone(), Tensor::tensor(scores_t.shape.clone(), out));
                env.insert(choice.clone(), Tensor::tensor(scores_t.shape.clone(), ind));
            }
            Op::WeightedSum {
                choice,
                terms,
                output,
            } => {
                let ch = &env[choice];
                let e = *ch.shape.last().unwrap();
                let seq = ch.numel() / e;
                let dim = *env[&terms[0]].shape.last().unwrap();
                let mut data = vec![0i64; seq * dim];
                for s in 0..seq {
                    for (ei, term) in terms.iter().enumerate() {
                        if ch.data[s * e + ei] != 0 {
                            let t = &env[term];
                            for j in 0..dim {
                                data[s * dim + j] += t.data[s * dim + j];
                            }
                        }
                    }
                }
                env.insert(output.clone(), Tensor::tensor(vec![seq, dim], data));
            }
        }
        true
    }
}

impl Model for SpecModel {
    fn num_segments(&self) -> usize {
        (self.total_values() + Self::SEGMENT_VALUES - 1) / Self::SEGMENT_VALUES
    }

    fn to_digit_matrix(&self, weights: &Weights, segment: usize) -> DigitMatrix {
        let by_name: HashMap<&str, &[i8]> = weights
            .iter()
            .map(|w| (w.name.as_str(), w.values.as_slice()))
            .collect();
        let total = self.total_values();
        let start = segment * Self::SEGMENT_VALUES;
        let end = (start + Self::SEGMENT_VALUES).min(total);
        let mut entries = Vec::with_capacity(end - start);
        let mut idx = 0usize;
        for w in &self.spec.weights {
            let vals = by_name.get(w.name.as_str()).expect("missing weight tensor");
            let n = vals.len();
            let lo = start.max(idx);
            let hi = end.min(idx + n);
            if lo < hi {
                // int8 weights are already single signed base-2^8 digits.
                entries.extend_from_slice(&vals[lo - idx..hi - idx]);
            }
            idx += n;
        }
        let d = entries.len();
        DigitMatrix::new(d, 0, entries)
    }

    fn check(&self, a: &[DigitMatrix], xi: &EncodedRequest, trace: &Trace) -> bool {
        if trace.segments.len() != 1 {
            return false;
        }
        let got: Vec<&str> = trace.segments[0]
            .cells
            .iter()
            .map(|c| c.name.as_str())
            .collect();
        let want: Vec<&str> = self.spec.cells.iter().map(|s| s.as_str()).collect();
        if got != want {
            return false;
        }

        let weights = self.weights_from(a);
        let window = &xi.window;
        let mut env: HashMap<String, Tensor> = HashMap::new();

        for op in &self.spec.ops {
            if !self.eval_op(op, &mut env, &weights, window) {
                return false;
            }
        }

        for name in &self.spec.cells {
            let computed = match env.get(name) {
                Some(v) => v,
                None => return false,
            };
            let traced = match trace.segments[0].cells.iter().find(|c| &c.name == name) {
                Some(c) => c,
                None => return false,
            };
            // A cell is a flat row-major matrix; rank-n tensors are flattened to
            // 2-D (rows = first dim, cols = the rest) for the comparison.
            let (rows, cols) = match computed.shape.len() {
                1 => (1, computed.shape[0]),
                2 => (computed.shape[0], computed.shape[1]),
                _ => (computed.shape[0], computed.numel() / computed.shape[0]),
            };
            if traced.rows != rows || traced.cols != cols || traced.values != computed.data {
                return false;
            }
        }

        // head_token == argmax of the last position's logits.
        let logits = match env.get("logits") {
            Some(v) => v,
            None => return false,
        };
        let d = *logits.shape.last().unwrap();
        let pos = xi.i0.min(logits.shape[0] - 1);
        let out_logits = &logits.data[pos * d..(pos + 1) * d];
        match trace.segments[0].head_token {
            Some(head) => out_logits[head] == out_logits.iter().copied().max().unwrap_or(0),
            None => false,
        }
    }
}
