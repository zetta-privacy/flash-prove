//! `tinyref`: a minimal example of using the `flashprove` library.
//!
//! It defines a tiny integer transformer (a one-block FFN) and its
//! arithmetization, then walks the core flow: the model produces an inference
//! trace, the trace is checked against the committed weights, and tampering is
//! rejected. In a real deployment the trace comes from an instrumented
//! inference engine via `load_trace`, not the in-process `forward`; this example
//! keeps both so it is self-contained.
//!
//! Run with: `cargo run --example tinyref`

use flashprove::model::{
    build, greedy_token, Arithmetization, Cell, DigitMatrix, EncodedRequest, Segment, Trace,
};

// --- arithmetic of the toy model --------------------------------------------

/// Requantization: rounding right shift by `k` bits (round half away from zero).
fn round_shift(x: i64, k: u32) -> i64 {
    let half = 1i64 << (k - 1);
    if x >= 0 {
        (x + half) >> k
    } else {
        -(((-x) + half) >> k)
    }
}

/// The lookup nonlinearity: ReLU.
fn relu(x: i64) -> i64 {
    x.max(0)
}

/// The toy model's forward pass, written once and shared by the model and the
/// checker so they cannot disagree. Returns `(cells, logits, greedy)`.
fn compute_cells(
    embedding: &[i64],
    w1: &[i64],
    w2: &[i64],
    output: &[i64],
    token: usize,
    dim: usize,
    hidden: usize,
    vocab: usize,
    shift: u32,
) -> (Vec<Cell>, Vec<i64>, usize) {
    let e: Vec<i64> = embedding[token * dim..(token + 1) * dim].to_vec();

    let mut u = vec![0i64; hidden];
    for i in 0..hidden {
        let mut s = 0i64;
        for j in 0..dim {
            s += w1[i * dim + j] * e[j];
        }
        u[i] = s;
    }
    let uq: Vec<i64> = u.iter().map(|&x| round_shift(x, shift)).collect();
    let a: Vec<i64> = uq.iter().map(|&x| relu(x)).collect();

    let mut v = vec![0i64; dim];
    for i in 0..dim {
        let mut s = 0i64;
        for j in 0..hidden {
            s += w2[i * hidden + j] * a[j];
        }
        v[i] = s;
    }
    let vq: Vec<i64> = v.iter().map(|&x| round_shift(x, shift)).collect();

    let z: Vec<i64> = vq.iter().zip(&e).map(|(&p, &q)| p + q).collect();

    let mut logits = vec![0i64; vocab];
    for i in 0..vocab {
        let mut s = 0i64;
        for j in 0..dim {
            s += output[i * dim + j] * z[j];
        }
        logits[i] = s;
    }

    let greedy = greedy_token(&logits);
    let cells = vec![
        Cell::new("embedding", 1, dim, e),
        Cell::new("w1_out", 1, hidden, u),
        Cell::new("w1_requant", 1, hidden, uq),
        Cell::new("relu", 1, hidden, a),
        Cell::new("w2_out", 1, dim, v),
        Cell::new("w2_requant", 1, dim, vq),
        Cell::new("residual", 1, dim, z),
        Cell::new("logits", 1, vocab, logits.clone()),
    ];
    (cells, logits, greedy)
}

// --- the model: an engine simulator for this demo ---------------------------

/// A tiny integer FFN. In production this is replaced by an instrumented
/// inference engine whose trace you load with `flashprove::model::load_trace`.
struct ReferenceModel {
    vocab: usize,
    dim: usize,
    hidden: usize,
    shift: u32,
    embedding: Vec<i64>, // V * d
    w1: Vec<i64>,        // h * d
    w2: Vec<i64>,        // d * h
    output: Vec<i64>,    // V * d
}

impl ReferenceModel {
    fn weight_len(&self) -> usize {
        2 * self.vocab * self.dim + 2 * self.hidden * self.dim
    }

    /// Deterministic weights in `[-2^(r-1), 2^(r-1))`.
    fn deterministic(
        vocab: usize,
        dim: usize,
        hidden: usize,
        shift: u32,
        r: u32,
        seed: u64,
    ) -> Self {
        let n = 2 * vocab * dim + 2 * hidden * dim;
        let mut x = seed;
        let mut weights = Vec::with_capacity(n);
        for _ in 0..n {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            weights.push(((x >> 33) % (1u64 << r)) as i64 - (1i64 << (r - 1)));
        }
        let mut it = weights.into_iter();
        Self {
            vocab,
            dim,
            hidden,
            shift,
            embedding: it.by_ref().take(vocab * dim).collect(),
            w1: it.by_ref().take(hidden * dim).collect(),
            w2: it.by_ref().take(dim * hidden).collect(),
            output: it.by_ref().take(vocab * dim).collect(),
        }
    }

    fn forward(&self, token: usize) -> (Trace, usize, Vec<i64>) {
        let (cells, logits, greedy) = compute_cells(
            &self.embedding,
            &self.w1,
            &self.w2,
            &self.output,
            token,
            self.dim,
            self.hidden,
            self.vocab,
            self.shift,
        );
        let trace = Trace {
            segments: vec![Segment {
                cells,
                head_token: Some(greedy),
            }],
        };
        (trace, greedy, logits)
    }

    /// Reference placement: `A = [E ‖ W1 ‖ W2 ‖ O]`, one digit per weight.
    fn to_digit_matrix(&self) -> DigitMatrix {
        let mut entries = Vec::with_capacity(self.weight_len());
        entries.extend_from_slice(&self.embedding);
        entries.extend_from_slice(&self.w1);
        entries.extend_from_slice(&self.w2);
        entries.extend_from_slice(&self.output);
        DigitMatrix::new(self.weight_len(), 0, entries)
    }
}

// --- the arithmetization: what you implement for your real model ------------

/// Checks the trace against the committed weights for this toy model. For a
/// real model, write the analogous `Arithmetization` impl (the identities and
/// lookups of `docs/model.md`).
struct ReferenceArithmetization {
    vocab: usize,
    dim: usize,
    hidden: usize,
    shift: u32,
}

impl Arithmetization for ReferenceArithmetization {
    fn check(&self, a: &DigitMatrix, xi: &EncodedRequest, trace: &Trace) -> bool {
        let (v, d, h, k) = (self.vocab, self.dim, self.hidden, self.shift);
        if a.entries.len() != 2 * v * d + 2 * h * d {
            return false;
        }
        if !a.in_range() {
            return false;
        }
        if xi.i1 == 0 || xi.i1 > xi.window.len() {
            return false;
        }
        let token = xi.window[xi.i1 - 1];
        if token >= v {
            return false;
        }

        let embedding = &a.entries[0..v * d];
        let w1 = &a.entries[v * d..v * d + h * d];
        let w2 = &a.entries[v * d + h * d..v * d + h * d + d * h];
        let output = &a.entries[v * d + h * d + d * h..];

        let (expected, _logits, greedy) =
            compute_cells(embedding, w1, w2, output, token, d, h, v, k);

        if trace.segments.len() != 1 {
            return false;
        }
        let seg = &trace.segments[0];
        if seg.head_token != Some(greedy) || seg.cells.len() != expected.len() {
            return false;
        }
        for (got, exp) in seg.cells.iter().zip(&expected) {
            if got.name != exp.name || got.values != exp.values {
                return false;
            }
        }
        true
    }
}

// --- demo -------------------------------------------------------------------

fn main() {
    let m = ReferenceModel::deterministic(64, 8, 16, 4, 4, 12345);
    let ar = ReferenceArithmetization {
        vocab: 64,
        dim: 8,
        hidden: 16,
        shift: 4,
    };

    let token = 3usize;
    let (trace, greedy, logits) = m.forward(token);
    let a = m.to_digit_matrix();
    let xi = build(&[token], &[greedy], &[]).unwrap();

    println!(
        "token = {token}, greedy = {greedy}, top-3 logits = {:?}",
        &logits[..3]
    );
    println!("weights packed into a {}x{} digit matrix", a.d, a.n);

    // Honest trace is accepted.
    let ok = ar.check(&a, &xi, &trace);
    println!("honest trace accepted: {ok}");
    assert!(ok);

    // Tampered trace is rejected.
    let mut bad_trace = trace.clone();
    bad_trace.segments[0].cells[7].values[0] += 1;
    let ok = ar.check(&a, &xi, &bad_trace);
    println!("tampered trace accepted: {ok}");
    assert!(!ok);

    // Tampered weights are rejected.
    let mut bad_a = a.clone();
    bad_a.entries[0] += 1;
    let ok = ar.check(&bad_a, &xi, &trace);
    println!("tampered weights accepted: {ok}");
    assert!(!ok);

    // The engine→prover boundary: serialize a trace, load it back.
    let bytes = flashprove::model::save_trace(&trace).unwrap();
    let reloaded = flashprove::model::load_trace(&bytes).unwrap();
    assert_eq!(reloaded, trace);
    println!(
        "trace round-trips through save_trace/load_trace ({} bytes)",
        bytes.len()
    );
}
