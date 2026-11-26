//! The weights file's table of contents, so a tensor's bytes can be read on demand.
//!
//! The prover needs each tensor's values exactly once — when it commits that tensor's
//! own root — and the claim then reads the retained witness. Holding the whole file (and
//! a copy per tensor) therefore buys nothing: this index records where each tensor lives
//! and [`WeightIndex::tensor`] reads just that one, so peak memory is one tensor instead
//! of the model. The format is the one `python/load_weights.py` writes.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use super::WeightMatrix;

/// One tensor's location in the file.
#[derive(Clone, Copy, Debug)]
struct Entry {
    offset: u64,
    rows: usize,
    cols: usize,
}

/// Name → where the tensor's `int8` values start, plus its shape.
#[derive(Clone, Debug)]
pub struct WeightIndex {
    path: PathBuf,
    entries: HashMap<String, Entry>,
}

impl WeightIndex {
    /// Parse the headers, skipping the values.
    pub fn read(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut r = BufReader::new(file);
        let count = read_u32(&mut r)? as usize;
        let mut entries = HashMap::with_capacity(count);
        for _ in 0..count {
            let name_len = read_u16(&mut r)? as usize;
            let mut name = vec![0u8; name_len];
            r.read_exact(&mut name)?;
            let name = String::from_utf8(name).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
            })?;
            let rows = read_u32(&mut r)? as usize;
            let cols = read_u32(&mut r)? as usize;
            let offset = r.stream_position()?;
            let len = (rows * cols) as i64;
            r.seek(SeekFrom::Current(len))?;
            entries.insert(
                name,
                Entry {
                    offset,
                    rows,
                    cols,
                },
            );
        }
        Ok(Self {
            path: path.to_path_buf(),
            entries,
        })
    }

    /// `(rows, cols)` of a tensor, without reading its values.
    pub fn shape(&self, name: &str) -> Option<(usize, usize)> {
        self.entries.get(name).map(|e| (e.rows, e.cols))
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.entries.keys()
    }

    /// Read one tensor's values.
    pub fn tensor(&self, name: &str) -> std::io::Result<WeightMatrix> {
        let e = *self.entries.get(name).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, format!("no tensor `{name}`"))
        })?;
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(e.offset))?;
        let mut raw = vec![0u8; e.rows * e.cols];
        file.read_exact(&mut raw)?;
        Ok(WeightMatrix {
            name: name.to_string(),
            rows: e.rows,
            cols: e.cols,
            values: raw.into_iter().map(|b| b as i8).collect(),
        })
    }
}

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u16(r: &mut impl Read) -> std::io::Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}
