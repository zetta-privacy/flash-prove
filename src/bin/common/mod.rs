//! Shared helpers for the `cp_*` binaries.
#![allow(dead_code)]

/// Load the integer weights from the compact binary format (see
/// `python/load_weights.py`): u32 count, then per tensor u16 name_len, name,
/// u32 rows, u32 cols, and rows*cols int8 values.
pub fn load_weights_bin(path: &std::path::Path) -> flashprove::model::Weights {
    use flashprove::model::WeightMatrix;
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut r = 0usize;
    let n = read_u32(&bytes, &mut r) as usize;
    let mut weights = Vec::with_capacity(n);
    for _ in 0..n {
        let name_len = read_u16(&bytes, &mut r) as usize;
        let name = String::from_utf8(bytes[r..r + name_len].to_vec()).unwrap();
        r += name_len;
        let rows = read_u32(&bytes, &mut r) as usize;
        let cols = read_u32(&bytes, &mut r) as usize;
        let values = bytes[r..r + rows * cols].iter().map(|&b| b as i8).collect();
        r += rows * cols;
        weights.push(WeightMatrix {
            name,
            rows,
            cols,
            values,
        });
    }
    weights
}

/// Load the weights file, materialising the values of only the tensors `keep` accepts.
///
/// Every tensor's header is kept, so a name lookup still finds its shape — the
/// verifier needs the norm γ vectors and the embedding's row count, not 553 MB of
/// int8 weights it never reads.
pub fn load_weights_bin_subset(
    path: &std::path::Path,
    keep: impl Fn(&str) -> bool,
) -> flashprove::model::Weights {
    use flashprove::model::WeightMatrix;
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut r = 0usize;
    let n = read_u32(&bytes, &mut r) as usize;
    let mut weights = Vec::with_capacity(n);
    for _ in 0..n {
        let name_len = read_u16(&bytes, &mut r) as usize;
        let name = String::from_utf8(bytes[r..r + name_len].to_vec()).unwrap();
        r += name_len;
        let rows = read_u32(&bytes, &mut r) as usize;
        let cols = read_u32(&bytes, &mut r) as usize;
        let values = if keep(&name) {
            bytes[r..r + rows * cols].iter().map(|&b| b as i8).collect()
        } else {
            Vec::new()
        };
        r += rows * cols;
        weights.push(WeightMatrix {
            name,
            rows,
            cols,
            values,
        });
    }
    weights
}

fn read_u32(b: &[u8], r: &mut usize) -> u32 {
    let v = u32::from_le_bytes(b[*r..*r + 4].try_into().unwrap());
    *r += 4;
    v
}

fn read_u16(b: &[u8], r: &mut usize) -> u16 {
    let v = u16::from_le_bytes(b[*r..*r + 2].try_into().unwrap());
    *r += 2;
    v
}

/// Load a JSON value from a file (the Python↔Rust interchange format).
pub fn load<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> T {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

/// Save a value to a file as JSON.
pub fn save<T: serde::Serialize>(path: &std::path::Path, v: &T) {
    let bytes = serde_json::to_vec(v).unwrap();
    std::fs::write(path, bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

/// Save a value to a file as bincode (streamed, for large commit tables).
pub fn save_bin<T: serde::Serialize>(path: &std::path::Path, v: &T) {
    let file =
        std::fs::File::create(path).unwrap_or_else(|e| panic!("create {}: {e}", path.display()));
    let mut w = std::io::BufWriter::new(file);
    bincode::serialize_into(&mut w, v).unwrap_or_else(|e| panic!("encode {}: {e}", path.display()));
}

/// Load a value from a bincode file (streamed).
pub fn load_bin<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> T {
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let r = std::io::BufReader::new(file);
    bincode::deserialize_from(r).unwrap_or_else(|e| panic!("decode {}: {e}", path.display()))
}

/// Hex-encode bytes for human-readable output.
pub fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}
