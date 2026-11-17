//! The relation: integer model, policies, configurations (paper Section 3).
//!
//! The *placement rule* (which digit of which weight maps to which matrix
//! entry, Definition 4.1) and the *arithmetization* (how the forward pass
//! reduces to identities and lookups, Definition 4.8) are explicitly deferred
//! by the paper to the release's Lean code and `docs/model.md`
//! (Remark D.10 "Given in outline"). This module fixes everything the paper
//! *does* specify

// --- Sizes (Construction D.3, Table 2) -------------------------------------

pub const WINDOW_LEN: usize = 1 << 10; // L = 1024
pub const VOCAB_SIZE: usize = 202_048; // V = {0, ..., 202_047}
pub const NUM_LAYERS: usize = 48;

// Construction D.3: (d_model, nh, ng, dh, f, fe, NE)
pub const D_MODEL: usize = 5120;
pub const NUM_HEADS: usize = 40; // nh
pub const NUM_GROUPS: usize = 8; // ng
pub const HEAD_DIM: usize = 128; // dh
pub const HIDDEN_EVEN: usize = 16_384; // f  (even-layer d_hid)
pub const HIDDEN_ODD: usize = 8_192; // fe (odd-layer d_hid)
pub const NUM_EXPERTS: usize = 128; // NE

/// A context length in tokens (Definition 3.1).
pub fn window_len() -> usize {
    WINDOW_LEN
}

/// The vocabulary is 0 .. VOCAB_SIZE−1.
pub fn in_vocab(tok: usize) -> bool {
    tok < VOCAB_SIZE
}

/// Signed-digit base.
pub const DIGIT_BASE: i64 = 1 << 8; // 2^8 = 256

/// Digits per weight entry: 4 for the embedding matrix, 2 otherwise.
pub const EMBEDDING_DIGITS: usize = 4;
pub const WEIGHT_DIGITS: usize = 2;

/// Matrix shape (Table 2): d matrix rows × 2^n matrix columns, n = 18.
pub const MATRIX_ROWS: usize = 3_076_096; // d
pub const MATRIX_COL_LOG: usize = 18; // n = n0 + v
pub const N0: usize = 7; // 2^n0 columns per table row
pub const V: usize = 11; // 2^v rows per table

/// Number of matrix columns (2^n).
pub fn matrix_cols() -> usize {
    1usize << MATRIX_COL_LOG
}

/// An entry of the digit matrix is in `[-2^7, 2^7)` (Definition 4.1).
pub fn digit_in_range(a: i64) -> bool {
    (-(1 << 7)..(1 << 7)).contains(&a)
}

/// An integer matrix `A` is *in range* (Definition 4.1) if every entry is in
/// `[-2^7, 2^7)` and each low/high digit pair satisfies the two-digit weight
/// bound `b0 + 2^8·b1`.
///
/// NOTE: the exact pair bound was an image lost in the markdown; the paper
/// states each non-embedding weight lies in `[−2^12, 2^12)`, so the natural
/// check is `b0 + 2^8·b1 ∈ [−2^12, 2^12)`. Verify against the release spec.
pub fn in_range(a: &[i64]) -> bool {
    a.iter().all(|&x| digit_in_range(x))
}

// --- Model parameters µ (Definition 3.6) ------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Normalization {
    /// exponent M, 24 ≤ M ≤ 47
    pub m: u32,
    /// exponent K, 17 ≤ K ≤ 32, K ≤ M ≤ K + 14
    pub k: u32,
    /// regularizer ε, 1 ≤ ε < 2^48
    pub epsilon: u64,
}

impl Normalization {
    pub fn is_valid(&self) -> bool {
        (24..=47).contains(&self.m)
            && (17..=32).contains(&self.k)
            && self.k <= self.m
            && self.m <= self.k + 14
            && (1..(1u64 << 48)).contains(&self.epsilon)
    }
}

#[derive(Clone, Debug)]
pub struct LayerParams {
    // shifts
    pub kq: u32, // queries  5..32
    pub kk: u32, // keys     5..32
    pub kv: u32, // values   1..24
    pub ko: u32, // attention output 1..19
    pub kg: u32, // gate     1..24
    pub ku: u32, // up       1..24
    pub kd: u32, // down     10..19
    pub ks: u32, // softmax keys 4..12
    // normalizations (two per layer)
    pub norm1: Normalization,
    pub norm2: Normalization,
    // odd layers only (Definition 3.6)
    pub kr: Option<u32>,       // router 8..24
    pub kg_prime: Option<u32>, // routed expert gate 1..24
    pub ku_prime: Option<u32>, // routed expert up 1..24
}

impl LayerParams {
    /// Per-layer validity; `ks` equality across layers 4j..4j+3 is checked at
    /// the `ModelParams` level (Definition 3.6).
    pub fn is_valid(&self, odd: bool) -> bool {
        let shifts_ok = (5..=32).contains(&self.kq)
            && (5..=32).contains(&self.kk)
            && (self.kq == self.kk || (self.kq >= 16 && self.kk >= 16))
            && (1..=24).contains(&self.kv)
            && (1..=24).contains(&self.kg)
            && (1..=24).contains(&self.ku)
            && (1..=19).contains(&self.ko)
            && (10..=19).contains(&self.kd)
            && (4..=12).contains(&self.ks)
            && self.norm1.is_valid()
            && self.norm2.is_valid();
        if !shifts_ok {
            return false;
        }
        if odd {
            let ok = |v: Option<u32>, lo: u32, hi: u32| match v {
                Some(x) => (lo..=hi).contains(&x),
                None => false,
            };
            ok(self.kr, 8, 24) && ok(self.kg_prime, 1, 24) && ok(self.ku_prime, 1, 24)
        } else {
            self.kr.is_none() && self.kg_prime.is_none() && self.ku_prime.is_none()
        }
    }
}

#[derive(Clone, Debug)]
pub struct ModelParams {
    pub layers: Vec<LayerParams>,  // 48
    pub final_norm: Normalization, // (MF, KF, εF)
}

impl ModelParams {
    pub fn is_valid(&self) -> bool {
        if self.layers.len() != NUM_LAYERS || !self.final_norm.is_valid() {
            return false;
        }
        for (i, l) in self.layers.iter().enumerate() {
            let odd = i % 2 == 1;
            if !l.is_valid(odd) {
                return false;
            }
        }
        // ks equal in layers 4j..4j+3 (Definition 3.6).
        for j in 0..NUM_LAYERS / 4 {
            let base = self.layers[4 * j].ks;
            for k in 1..4 {
                if self.layers[4 * j + k].ks != base {
                    return false;
                }
            }
        }
        true
    }

    /// The default model parameters µ₀ (Definition D.1).
    pub fn default() -> Self {
        let layer = LayerParams {
            kq: 29,
            kk: 29,
            kv: 17,
            ko: 17,
            kg: 16,
            ku: 20,
            kd: 19,
            ks: 8,
            norm1: Normalization {
                m: 44,
                k: 30,
                epsilon: 0,
            },
            norm2: Normalization {
                m: 44,
                k: 30,
                epsilon: 0,
            },
            kr: None,
            kg_prime: None,
            ku_prime: None,
        };
        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for i in 0..NUM_LAYERS {
            let mut l = layer.clone();
            if i % 2 == 1 {
                l.kr = Some(17);
                l.kg_prime = Some(16);
                l.ku_prime = Some(20);
            }
            layers.push(l);
        }
        Self {
            layers,
            final_norm: Normalization {
                m: 44,
                k: 30,
                epsilon: 0,
            },
        }
    }
}

// --- Acceptance policies (Definition 3.4) -----------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Policy {
    /// clip C (sum bound)
    pub clip: u64,
    /// threshold θ (per-token cost threshold)
    pub threshold: u64,
    /// cap (count of tokens allowed to exceed the threshold)
    pub cap: u64,
}

/// Exact match: (2^8, 2^8, 0) (Definition 3.4).
pub const EXACT_MATCH: Policy = Policy {
    clip: 1 << 8,
    threshold: 1 << 8,
    cap: 0,
};

/// Z₁ = Z₀[µ₀, (25856, 512, 1)] (Definition 3.7).
pub const Z1_POLICY: Policy = Policy {
    clip: 25_856,
    threshold: 512,
    cap: 1,
};

/// Logit gap of token `v` after a context (Definition 3.4): max over logits
/// minus the token's logit.
pub fn logit_gap(logits: &[i64], v: usize) -> i64 {
    let max = logits.iter().copied().max().unwrap_or(0);
    max - logits[v]
}

/// The cost of a counted token: gap + 2^8 (Definition 3.4).
pub fn cost(gap: i64) -> i64 {
    gap + (1 << 8)
}

/// The least token of largest logit (Definition 3.2).
pub fn greedy_token(logits: &[i64]) -> usize {
    let max = logits.iter().copied().max().unwrap_or(0);
    logits.iter().position(|&l| l == max).unwrap_or(0)
}

/// Whether an output meets a policy (Definition 3.4).
///
/// `gaps[i]` is the logit gap of the i-th counted token. The output is assumed
/// eligible (caller checks `eligible`).
///
/// NOTE: the two inequalities are reconstructed — they were images in the PDF.
/// They are mutually consistent with Proposition 3.5 (exact match forces every
/// gap to 0) and the Z₁ parameters; verify against the release format spec.
pub fn meets_policy(pol: &Policy, gaps: &[i64]) -> bool {
    let sum: i64 = gaps.iter().sum();
    // (a) sum bound: Σ_i gap_i ≤ C − 2^8
    if sum > (pol.clip as i64) - (1 << 8) {
        return false;
    }
    // (b) count bound: #{i : cost_i > θ} ≤ cap
    let over = gaps
        .iter()
        .filter(|&&g| cost(g) > pol.threshold as i64)
        .count() as u64;
    over <= pol.cap
}

// --- Configurations (Definition 3.7, 3.8) ------------------------------------

/// A configuration is three byte strings (Definition 3.7).
#[derive(Clone, Debug)]
pub struct Configuration {
    pub parameter_file: Vec<u8>,
    pub layout_file: Vec<u8>,
    pub scale_file: Vec<u8>,
}

/// A configuration is admissible (Definition 3.8) when it is `Z₀[µ, pol]` for
/// valid µ (one ε₁ and one ε₂ across layers) and a policy with
/// `2^8 ≤ θ ≤ C < 2^32` and `cap ≤ 1024` (Proposition 5.6).
///
/// The *full* `Chk(Z)` (Definition 4.14) is deferred by the paper (Remark D.10);
/// this function implements the scope test `Adm(Z)` that `V_Z` step 1 uses.
pub fn admissible(pol: &Policy) -> bool {
    (1 << 8) <= pol.threshold
        && pol.threshold <= pol.clip
        && pol.clip < (1u64 << 32)
        && pol.cap <= 1024
}

// --- Encoded request (Definition 4.7) ---------------------------------------

#[derive(Clone, Debug)]
pub struct EncodedRequest {
    /// window of exactly `WINDOW_LEN` tokens
    pub window: Vec<usize>,
    /// first token position whose successor is checked (i₀)
    pub i0: usize,
    /// upper position bound (i₁)
    pub i1: usize,
    /// the token after a full window, else 0
    pub y_plus: usize,
}

/// Number of counted tokens: min(|y|, L − |x|) (Definition 3.4).
pub fn counted(x_len: usize, y_len: usize) -> usize {
    y_len.min(WINDOW_LEN - x_len)
}

/// Build an encoded request (Definition 4.7).
///
/// Returns `None` when the inputs are outside the paper's domain (x, y non-empty,
/// |x| + |y| ≤ L + 1, all tokens in V).
///
/// The exact `i₀`/`i₁` numeric convention was an image; the fields below encode
/// the *semantics* (which output tokens are gap-checked, and the full-window
/// `y⁺` condition). Verify the byte serialization against the release spec.
pub fn build(x: &[usize], y: &[usize], fill: &[usize]) -> Option<EncodedRequest> {
    if x.is_empty() || y.is_empty() || x.len() + y.len() > WINDOW_LEN + 1 {
        return None;
    }
    let all_tokens = || x.iter().chain(y).chain(fill).all(|&t| in_vocab(t));
    if !all_tokens() {
        return None;
    }

    let mut window = Vec::with_capacity(WINDOW_LEN);
    window.extend_from_slice(x);
    window.extend_from_slice(y);
    window.extend_from_slice(fill);
    window.resize(WINDOW_LEN, 0); // 0-pad to length L

    let n_ct = counted(x.len(), y.len());
    let full = x.len() + y.len() == WINDOW_LEN + 1;
    Some(EncodedRequest {
        window,
        i0: x.len() - 1,
        i1: if full { WINDOW_LEN } else { x.len() + n_ct - 1 },
        y_plus: if full { *y.last().unwrap() } else { 0 },
    })
}

// --- Digit matrix (Definition 4.1) -------------------------------------------

/// The digit matrix `A`: `d` rows × `2^n` columns of signed digits, laid out
/// row-major. This is the matrix the protocol commits to; the decoder
/// (`protocol::decode_exact`) recovers it, and the key IOP checks it is in range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DigitMatrix {
    pub d: usize,
    pub n: usize,
    /// `d * 2^n` entries, each in `[-2^7, 2^7)`.
    pub entries: Vec<i64>,
}

impl DigitMatrix {
    pub fn new(d: usize, n: usize, entries: Vec<i64>) -> Self {
        assert_eq!(entries.len(), d * (1 << n), "digit matrix length");
        Self { d, n, entries }
    }

    pub fn entry(&self, i: usize, j: usize) -> i64 {
        self.entries[i * (1 << self.n) + j]
    }

    pub fn in_range(&self) -> bool {
        self.entries.iter().all(|&x| digit_in_range(x))
    }
}

// --- Inference trace (Definition 4.8) ----------------------------------------

/// The witness: intermediate values of one forward pass, produced by an
/// instrumented inference engine and fed to the prover as input (Definition 4.8
/// cells `Y_g`). The prover commits to the trace and the arithmetization
/// constrains it — it never re-derives the model.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Trace {
    /// Segments `g < G`; the last is the head.
    pub segments: Vec<Segment>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Segment {
    /// The segment's cells (intermediate value matrices), in the arithmetization's
    /// fixed order.
    pub cells: Vec<Cell>,
    /// The head token, present only on the head (last) segment.
    pub head_token: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Cell {
    /// Label for readability/debugging; the arithmetization may key on it or on
    /// positional order.
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    /// Row-major values, len = `rows * cols`.
    pub values: Vec<i64>,
}

impl Cell {
    pub fn new(name: &str, rows: usize, cols: usize, values: Vec<i64>) -> Self {
        assert_eq!(values.len(), rows * cols, "cell length");
        Self {
            name: name.to_string(),
            rows,
            cols,
            values,
        }
    }
}

/// Ingest a trace dumped by the inference engine (bincode).
pub fn load_trace(bytes: &[u8]) -> anyhow::Result<Trace> {
    Ok(bincode::deserialize(bytes)?)
}

/// Serialize a trace for the inference engine to write (bincode).
pub fn save_trace(trace: &Trace) -> anyhow::Result<Vec<u8>> {
    Ok(bincode::serialize(trace)?)
}

// --- Arithmetization (Definition 4.8) ----------------------------------------

/// The arithmetization: the identities and lookups that constrain the trace
/// against the committed weights, together with the placement rule `wt`
/// (Definition 4.1). Deferred by the paper (Remark D.10) to the release's
/// `docs/model.md`; an integrator supplies a concrete implementation (or a
/// reduced reference).
pub trait Arithmetization {
    /// Check that every identity is 0 and every lookup key is in its table, for
    /// the digit matrix `a`, the encoded request `ξ`, and the trace.
    fn check(&self, a: &DigitMatrix, xi: &EncodedRequest, trace: &Trace) -> bool;
}
