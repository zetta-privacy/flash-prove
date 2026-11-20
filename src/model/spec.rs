//! The data-driven model spec (`ModelSpec`) and the integer arithmetic the model
//! uses. The op graph is data, so the protocol stays model-agnostic.

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
