//! The trace file's table of contents, so a cell can be read on demand.
//!
//! The prover works one segment at a time and the trace is the largest thing it touches
//! (hundreds of MB here, gigabytes at full size), so the file is a flat binary layout —
//! no text, no parsing — with an index of where each cell lives. [`TraceIndex::cell`]
//! reads exactly one cell.
//!
//! Layout, little-endian, written by `python/run_real.py`:
//!
//! ```text
//!   magic  b"FPT1"
//!   u32    number of cells
//!   i64    head token (-1 when absent)
//!   then per cell: u16 name_len, name, u32 rows, u32 cols, i32 × rows·cols
//!
//! Values are `i32`: every traced value is committed, and the field guard requires
//! `|v| < 2^31`, so 32 bits is lossless and half the bytes of `i64`.
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use super::Cell;

const MAGIC: &[u8; 4] = b"FPT1";

#[derive(Clone, Copy, Debug)]
struct Entry {
    offset: u64,
    rows: usize,
    cols: usize,
}

/// Name → where a cell's `i64` values start, plus its shape.
#[derive(Clone, Debug)]
pub struct TraceIndex {
    path: PathBuf,
    head_token: Option<usize>,
    /// Cell names in file order (the spec's cell order).
    order: Vec<String>,
    entries: HashMap<String, Entry>,
}

impl TraceIndex {
    /// Parse the headers, skipping the values.
    pub fn read(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut r = BufReader::new(file);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "not a trace file",
            ));
        }
        let count = read_u32(&mut r)? as usize;
        let head = read_i64(&mut r)?;
        let mut order = Vec::with_capacity(count);
        let mut entries = HashMap::with_capacity(count);
        for _ in 0..count {
            let name_len = read_u16(&mut r)? as usize;
            let mut name = vec![0u8; name_len];
            r.read_exact(&mut name)?;
            let name = String::from_utf8(name)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
            let rows = read_u32(&mut r)? as usize;
            let cols = read_u32(&mut r)? as usize;
            let offset = r.stream_position()?;
            r.seek(SeekFrom::Current((rows * cols * 4) as i64))?;
            order.push(name.clone());
            entries.insert(name, Entry { offset, rows, cols });
        }
        Ok(Self {
            path: path.to_path_buf(),
            head_token: (head >= 0).then_some(head as usize),
            order,
            entries,
        })
    }

    /// The head token, if the trace carries one.
    pub fn head_token(&self) -> Option<usize> {
        self.head_token
    }

    /// Cell names, in file order.
    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.order.iter()
    }

    /// `(rows, cols)` of a cell, without reading it.
    pub fn shape(&self, name: &str) -> Option<(usize, usize)> {
        self.entries.get(name).map(|e| (e.rows, e.cols))
    }

    /// The sequence length the trace was produced with (the embedding's row count).
    pub fn window_len(&self) -> Option<usize> {
        self.order.first().and_then(|n| self.shape(n)).map(|s| s.0)
    }

    /// Read one cell's values.
    pub fn cell(&self, name: &str) -> std::io::Result<Cell> {
        let e = *self.entries.get(name).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, format!("no cell `{name}`"))
        })?;
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(e.offset))?;
        let mut raw = vec![0u8; e.rows * e.cols * 4];
        file.read_exact(&mut raw)?;
        let values = raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().expect("4 bytes")) as i64)
            .collect();
        Ok(Cell {
            name: name.to_string(),
            rows: e.rows,
            cols: e.cols,
            values,
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

fn read_i64(r: &mut impl Read) -> std::io::Result<i64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(i64::from_le_bytes(b))
}
