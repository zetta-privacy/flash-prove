//! The derived-cell planner.
//!
//! A segment's stacked commitment is sealed **before** any claim, so every table
//! an op derives must be planned up front: `y = h·inv_q`, the requant remainder,
//! the digit/bit columns and their LogUp helper cells. [`Plan`] collects them from
//! the trace cells, keyed by name and deduplicated, so the shared shapes live here
//! once instead of being re-derived inside every op.
//!
//! ```text
//!   let mut plan = Plan::new(&trace_cells);
//!   let rq = plan.requant("norm", "y", "ob", k, digits);
//!   let seg = plan.seal(params);                    // ONE root, trace + derived
//!   claim_requant(&mut seg, rq.cells(&seg), (n, k, digits), &mut tr);
//! ```
//!
//! Derived cells are written as **field** values (inverses and shifted digits are
//! not small integers); the trace cells stay `i64`, which is what the op arithmetic
//! reads.

use std::collections::HashMap;

use crate::field::Fp;
use crate::gkr::range::{Decomp, DecompCells};
use crate::gkr::requant::RequantCells;
use crate::gkr::segment::{CellSegment, SegmentCells};
use crate::model::Cell;
use crate::pcs::whir_stacked::Params;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

/// A segment plan: the trace cells plus every derived cell an op needs.
pub struct Plan {
    sc: SegmentCells,
    trace: HashMap<String, Vec<i64>>,
    /// Each cell's logical extents, as the model states them. A planned cell is
    /// stored flat, so the extent is not recoverable from the table height.
    logical: HashMap<String, (usize, usize)>,
    /// `(cell, max |value|)` for cells that do not fit `FLASHPROVE_CHECK_BITS`.
    over: Vec<(String, u64)>,
}

impl Plan {
    /// Start from the segment's trace cells.
    pub fn new(trace: &[Cell]) -> Self {
        let mut sc = SegmentCells::new();
        let mut map = HashMap::new();
        let mut logical = HashMap::new();
        for c in trace {
            map.entry(c.name.clone())
                .or_insert_with(|| c.values.clone());
            logical.entry(c.name.clone()).or_insert((c.rows, c.cols));
            // Flat table, logical extents kept in the metadata.
            sc.cell_logical(&c.name, c.rows, c.cols, c.values.clone());
        }
        let mut this = Self {
            sc,
            trace: map,
            logical,
            over: Vec::new(),
        };
        for c in trace {
            this.note_bound(&c.name, &c.values);
        }
        this
    }

    /// Record a cell that would not fit the checked field width.
    fn note_bound(&mut self, name: &str, values: &[i64]) {
        if let Some(bits) = check_bits() {
            if let Some((n, max)) = bound_violation(name, values, bits) {
                self.over.push((n.to_string(), max));
            }
        }
    }

    /// A cell's logical extents `(rows, cols)`, as the model states them.
    pub fn logical(&self, name: &str) -> Option<(usize, usize)> {
        self.logical.get(name).copied()
    }

    /// A trace cell's integer values (what the op arithmetic reads).
    pub fn values(&self, name: &str) -> &[i64] {
        &self.trace[name]
    }

    /// Add a derived cell. First-writer-wins by name.
    pub fn add(&mut self, name: &str, rows: usize, cols: usize, values: Vec<i64>) -> usize {
        self.note_bound(name, &values);
        // Register in both maps too, so `values()` and `logical()` read it back.
        self.trace
            .entry(name.to_string())
            .or_insert_with(|| values.clone());
        self.logical.entry(name.to_string()).or_insert((rows, cols));
        self.sc.cell(name, rows, cols, values)
    }

    /// Add a derived cell as a **flat** one-column table, keeping `rows × cols` as its
    /// logical shape.
    ///
    /// This is the storage the trace cells use (and what the linear claims read), so a
    /// derived cell that a multi-dimensional claim indexes must be added this way: a
    /// `cols`-wide table has only `log2(rows)` variables, while a flat cell has
    /// `log2(rows·cols)`.
    pub fn add_logical(&mut self, name: &str, rows: usize, cols: usize, values: Vec<i64>) -> usize {
        self.note_bound(name, &values);
        self.trace
            .entry(name.to_string())
            .or_insert_with(|| values.clone());
        self.logical.entry(name.to_string()).or_insert((rows, cols));
        self.sc.cell_logical(name, rows, cols, values)
    }

    /// Seal the whole plan into one segment.
    pub fn seal(self, params: Params) -> CellSegment {
        if let Some(bits) = check_bits() {
            let mut over = self.over.clone();
            over.sort_by(|a, b| b.1.cmp(&a.1));
            if over.is_empty() {
                eprintln!("  field check ({bits} bits): every committed value fits");
            } else {
                eprintln!(
                    "  field check ({bits} bits): {} of {} cells do not fit; worst:",
                    over.len(),
                    self.sc.len()
                );
                for (name, max) in over.iter().take(12) {
                    eprintln!(
                        "    {:<44} max |v| = {:>14} (2^{:.1})",
                        name,
                        max,
                        (*max as f64).log2()
                    );
                }
            }
        }
        self.sc.seal(params)
    }

    /// Plan a LogUp decomposition of `values` under `decomp`.
    pub fn decomp(
        &mut self,
        prefix: &str,
        value_name: &str,
        values: &[i64],
        decomp: Decomp,
    ) -> DecompNames {
        let n = values.len();
        let d = n.next_power_of_two();
        let cols = decomp.columns(values, d);
        let shift = decomp.shift();
        let all_len = n * decomp.digits;
        let all_d = all_len.next_power_of_two();

        let mut all: Vec<Fp> = Vec::with_capacity(all_d);
        for c in &cols {
            for &v in c.iter().take(n) {
                all.push(v + Fp::new(shift as u64));
            }
        }
        all.resize(all_d, Fp::ONE);

        let table = decomp.table();
        let ft_d = table.len().next_power_of_two();
        let mut counts = vec![0usize; table.len()];
        for &v in all.iter().take(all_len) {
            counts[v.as_canonical_u64() as usize - 1] += 1;
        }
        let mut m: Vec<Fp> = counts.iter().map(|&c| Fp::new(c as u64)).collect();
        m.resize(ft_d, Fp::ONE);
        let mut ft = table.clone();
        ft.resize(ft_d, Fp::ONE);
        let ft_inv = crate::backend::batch_invert(&table);
        let mut diff_t: Vec<Fp> = counts
            .iter()
            .zip(&ft_inv)
            .map(|(&c, &t)| Fp::new(c as u64) * t)
            .collect();
        diff_t.resize(ft_d, Fp::ONE);
        let diff_q: Vec<Fp> = crate::backend::batch_invert(&all);

        self.sc.cell_fp(&format!("{prefix}.all"), all_d, 1, all);
        self.sc.cell_fp(&format!("{prefix}.dq"), all_d, 1, diff_q);
        self.sc.cell_fp(&format!("{prefix}.ft"), ft_d, 1, ft);
        self.sc.cell_fp(&format!("{prefix}.dt"), ft_d, 1, diff_t);
        self.sc.cell_fp(&format!("{prefix}.m"), ft_d, 1, m);

        DecompNames {
            value: value_name.to_string(),
            all: format!("{prefix}.all"),
            diff_q: format!("{prefix}.dq"),
            ft: format!("{prefix}.ft"),
            diff_t: format!("{prefix}.dt"),
            m: format!("{prefix}.m"),
        }
    }

    /// Plan `out = requant(x, k)`: the sign/magnitude columns, the remainder, and
    /// both range decompositions.
    ///
    /// The digit width is the *smallest* that reconstructs every value (capped by
    /// `digits`): a narrower decomposition is a strictly tighter range check, so this
    /// only ever costs the prover less.
    pub fn requant(
        &mut self,
        prefix: &str,
        x: &str,
        out: &str,
        k: usize,
        digits: usize,
    ) -> RequantNames {
        let xs = self.values(x).to_vec();
        let os = self.values(out).to_vec();
        let n = xs.len();
        assert_eq!(os.len(), n);

        let s: Vec<i64> = xs.iter().map(|&v| if v < 0 { 1 } else { 0 }).collect();
        let sx: Vec<i64> = (0..n).map(|i| s[i] * xs[i]).collect();
        let sout: Vec<i64> = (0..n).map(|i| s[i] * os[i]).collect();
        let abs_x: Vec<i64> = (0..n).map(|i| xs[i] - 2 * sx[i]).collect();
        let abs_out: Vec<i64> = (0..n).map(|i| os[i] - 2 * sout[i]).collect();
        let half = 1i64 << (k - 1);
        let rem: Vec<i64> = (0..n)
            .map(|i| abs_x[i] - abs_out[i] * (1i64 << k) + half)
            .collect();

        let name = |suffix: &str| format!("{prefix}.{suffix}");
        for (suffix, vals) in [
            ("s", &s),
            ("sx", &sx),
            ("sout", &sout),
            ("abs_x", &abs_x),
            ("abs_out", &abs_out),
            ("rem", &rem),
        ] {
            self.sc.cell(&name(suffix), n, 1, vals.clone());
        }
        let rem_name = name("rem");
        let bits = self.decomp(&format!("{prefix}.bit"), &rem_name, &rem, Decomp::bits(k));
        let dbytes = signed_bytes_needed(&xs, digits);
        let digits_d = self.decomp(&format!("{prefix}.dig"), x, &xs, Decomp::signed_bytes(dbytes));

        RequantNames {
            dbytes,
            x: x.to_string(),
            out: out.to_string(),
            s: name("s"),
            sx: name("sx"),
            sout: name("sout"),
            abs_x: name("abs_x"),
            abs_out: name("abs_out"),
            rem: rem_name,
            bits,
            digits: digits_d,
        }
    }
}

/// With `FLASHPROVE_CHECK_BITS=k` set, every committed integer is checked against a
/// `k`-bit field: signed values must satisfy `|v| < 2^(k-1)`, or the field element that
/// represents them would differ from the integer the model computed.
///
/// This is the instrument for a field migration: it names the exact cells/ops that have to
/// be re-quantised (or limb-split) before the field can shrink.
pub fn check_bits() -> Option<u32> {
    std::env::var("FLASHPROVE_CHECK_BITS")
        .ok()
        .and_then(|v| v.parse().ok())
}

/// Report `(name, max |v|)` when a cell does not fit a `bits`-bit signed representation.
pub fn bound_violation<'a>(name: &'a str, values: &[i64], bits: u32) -> Option<(&'a str, u64)> {
    let limit = 1i128 << (bits - 1);
    let mut max = 0i128;
    for &v in values {
        let a = (v as i128).abs();
        if a > max {
            max = a;
        }
    }
    if max >= limit {
        Some((name, max as u64))
    } else {
        None
    }
}

/// The fewest signed base-256 digits that reconstruct every value, capped by `cap`.
fn signed_bytes_needed(values: &[i64], cap: usize) -> usize {
    let cap = cap.max(1);
    let mut d = 1;
    while d < cap {
        let dec = Decomp::signed_bytes(d);
        if values.iter().all(|&v| {
            let ds = dec.split(v);
            let mut acc = 0i64;
            for (j, &digit) in ds.iter().enumerate() {
                acc += digit * 256i64.pow(j as u32);
            }
            acc == v
        }) {
            return d;
        }
        d += 1;
    }
    cap
}

/// Cell names for one decomposition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecompNames {
    pub value: String,
    pub all: String,
    pub diff_q: String,
    pub ft: String,
    pub diff_t: String,
    pub m: String,
}

impl DecompNames {
    /// Resolve the names on a sealed segment.
    pub fn cells(&self, seg: &CellSegment) -> DecompCells {
        DecompCells {
            value: seg.id(&self.value),
            all_digits: seg.id(&self.all),
            diff_q: seg.id(&self.diff_q),
            ft: seg.id(&self.ft),
            diff_t: seg.id(&self.diff_t),
            m: seg.id(&self.m),
        }
    }
}

/// Cell names for one requant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequantNames {
    /// The signed-byte width actually used for the value decomposition.
    pub dbytes: usize,
    pub x: String,
    pub out: String,
    pub s: String,
    pub sx: String,
    pub sout: String,
    pub abs_x: String,
    pub abs_out: String,
    pub rem: String,
    pub bits: DecompNames,
    pub digits: DecompNames,
}

impl RequantNames {
    /// Resolve the names on a sealed segment.
    pub fn cells(&self, seg: &CellSegment) -> RequantCells {
        RequantCells {
            x: seg.id(&self.x),
            out: seg.id(&self.out),
            s: seg.id(&self.s),
            sx: seg.id(&self.sx),
            sout: seg.id(&self.sout),
            abs_x: seg.id(&self.abs_x),
            abs_out: seg.id(&self.abs_out),
            rem: seg.id(&self.rem),
            bits: self.bits.cells(seg),
            digits: self.digits.cells(seg),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gkr::requant::{claim_requant, verify_requant};
    use crate::gkr::segment::verify;
    use crate::model::spec::requant as requant_i64;
    use crate::pcs::transcript::{new_challenger, Transcript};

    #[test]
    fn planned_requant_is_provable() {
        let k = 1usize;
        let x = vec![5i64, -5, 3, -3];
        let out: Vec<i64> = x.iter().map(|&v| requant_i64(v, k as u32)).collect();
        let trace = vec![
            Cell::new("y", x.len(), 1, x.clone()),
            Cell::new("ob", out.len(), 1, out.clone()),
        ];

        let mut plan = Plan::new(&trace);
        let rq = plan.requant("rq", "y", "ob", k, 1);
        let seg = plan.seal(Params::default());

        let mut seg = seg;
        let com = seg.commitment();
        let mut tr = Transcript::new(&new_challenger());
        let rc = rq.cells(&seg);
        let proof = claim_requant(&mut seg, rc, (x.len(), k, 1), &mut tr);
        let claims = seg.claims().to_vec();
        let wproof = seg.prove(&mut new_challenger());
        assert!(verify(&com, &claims, &wproof, &mut new_challenger()));
        let mut vtr = Transcript::new(&new_challenger());
        assert!(verify_requant(
            &com,
            &claims,
            &proof,
            (x.len(), k, 1),
            &mut vtr
        ));
    }
}

/// Cell names for a two-column (graph) lookup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lookup2Names {
    pub input: String,
    pub output: String,
    pub fp: String,
    pub diff_q: String,
    pub ft: String,
    pub diff_t: String,
    pub m: String,
}

impl Lookup2Names {
    pub fn cells(&self, seg: &CellSegment) -> crate::gkr::logup::Lookup2Cells {
        crate::gkr::logup::Lookup2Cells {
            input: seg.id(&self.input),
            output: seg.id(&self.output),
            fp: seg.id(&self.fp),
            diff_q: seg.id(&self.diff_q),
            ft: seg.id(&self.ft),
            diff_t: seg.id(&self.diff_t),
            m: seg.id(&self.m),
        }
    }
}

impl Plan {
    /// Plan a two-column graph lookup `output = f(input)` against a public table.
    ///
    /// `table` is the padded fingerprint vector; `big_n` is its *true* size, which
    /// the LogUp multiplicity offset needs (padding is filled with `1`).
    pub fn lookup2(
        &mut self,
        prefix: &str,
        input: &str,
        output: &str,
        (alpha, beta): (Fp, Fp),
        table: &[Fp],
        big_n: usize,
    ) -> Lookup2Names {
        let xs = self.values(input).to_vec();
        let os = self.values(output).to_vec();
        let n = xs.len();
        assert_eq!(os.len(), n);
        let d = n.next_power_of_two();

        let name = |suffix: &str| format!("{prefix}.{suffix}");
        let mut fp: Vec<Fp> = (0..n)
            .map(|i| alpha + beta * fp_i64(xs[i]) + beta * beta * fp_i64(os[i]))
            .collect();
        fp.resize(d, Fp::ONE);
        let diff_q: Vec<Fp> = fp.iter().map(|&v| v.inverse()).collect();

        // Multiplicities of each table fingerprint, padding `1`.
        let index: std::collections::HashMap<u64, usize> = table
            .iter()
            .take(big_n)
            .enumerate()
            .map(|(j, v)| (v.as_canonical_u64(), j))
            .collect();
        let mut counts = vec![0usize; big_n];
        for i in 0..n {
            if let Some(&j) = index.get(&fp[i].as_canonical_u64()) {
                counts[j] += 1;
            }
        }
        let ft_d = table.len();
        let mut m: Vec<Fp> = counts.iter().map(|&c| Fp::new(c as u64)).collect();
        m.resize(ft_d, Fp::ONE);
        let mut diff_t: Vec<Fp> = counts
            .iter()
            .zip(table.iter().take(big_n))
            .map(|(&c, &t)| Fp::new(c as u64) * t.inverse())
            .collect();
        diff_t.resize(ft_d, Fp::ONE);

        self.sc.cell_fp(&name("fp"), d, 1, fp);
        self.sc.cell_fp(&name("dq"), d, 1, diff_q);
        self.sc.cell_fp(&name("ft"), ft_d, 1, table.to_vec());
        self.sc.cell_fp(&name("dt"), ft_d, 1, diff_t);
        self.sc.cell_fp(&name("m"), ft_d, 1, m);

        Lookup2Names {
            input: input.to_string(),
            output: output.to_string(),
            fp: name("fp"),
            diff_q: name("dq"),
            ft: name("ft"),
            diff_t: name("dt"),
            m: name("m"),
        }
    }
}

fn fp_i64(x: i64) -> Fp {
    if x < 0 {
        -Fp::new(x.unsigned_abs())
    } else {
        Fp::new(x as u64)
    }
}
