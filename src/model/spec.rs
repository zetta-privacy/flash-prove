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
/// so every input `|x| < 2^RECIP_Q` (the norm `x·γ` stays well below this), making
/// the integer `requant(x·inv_q, RECIP_Q)` bit-consistent with the f64 `round(x/divisor)`.
pub const RECIP_Q: u32 = 14;

pub fn silu(x: i64) -> i64 {
    let xc = x.clamp(-(1 << 15), (1 << 15) - 1);
    let xf = xc as f64;
    (xf / (1.0 + (-xf).exp())).round() as i64
}

pub fn exp_q15(x: i64) -> i64 {
    let xc = x.clamp(-(1 << 15), 0);
    ((xc as f64).exp() * 32768.0).round() as i64
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
    /// attention scores = requant(q·k^T · round(2^shift / scale), shift) (3-D).
    Scores {
        q: String,
        k: String,
        output: String,
        scale: f64,
        #[serde(default)]
        shift: Option<u32>,
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
    /// elementwise multiply; `shift` (if present) rescales it:
    /// `out = requant(a·b, shift)`.
    Mul {
        a: String,
        b: String,
        output: String,
        #[serde(default)]
        shift: Option<u32>,
    },
    /// elementwise add; `shift` (if present) rescales the sum:
    /// `out = requant(Σ inputs, shift)`.
    Add {
        inputs: Vec<String>,
        output: String,
        #[serde(default)]
        shift: Option<u32>,
    },
    /// top-k on the last dim, choice is an indicator.
    TopK {
        input: String,
        output: String,
        choice: String,
        k: usize,
    },
    /// The rows of `input` where `choice[·][expert]` is one, in increasing order.
    Gather {
        input: String,
        choice: String,
        expert: usize,
        output: String,
    },
    /// The MoE combine: each expert's gathered rows added back at their positions.
    MoeCombine {
        choice: String,
        terms: Vec<String>,
        output: String,
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
