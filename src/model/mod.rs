//! The trace and weight types shared by the GKR verifier and the inference engine.

pub mod spec;
pub mod trace_file;
pub mod weights;

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

/// One named integer weight tensor.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WeightMatrix {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    /// Row-major values, len = `rows * cols`. Each entry is an 8-bit weight.
    pub values: Vec<i8>,
}

impl WeightMatrix {
    pub fn new(name: &str, rows: usize, cols: usize, values: Vec<i8>) -> Self {
        assert_eq!(values.len(), rows * cols, "weight matrix length");
        Self {
            name: name.to_string(),
            rows,
            cols,
            values,
        }
    }
}

/// The model's weights as a flat set of named tensors. A Mixture-of-Experts
/// model has an embedding, per-layer attention (Q/K/V/O) and FFN (gate/up/down),
/// and in MoE layers a router plus expert tensors, and an output matrix.
pub type Weights = Vec<WeightMatrix>;
