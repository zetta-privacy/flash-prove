//! cp_prove: produce the GKR succinct proof of a model's inference on a request.
//!
//! Loads the spec, integer weights, and a trace (from `python/run_integer.py`),
//! reconstructs the token window from the prompt/output/fill files, derives the
//! lookup bounds, and writes a self-contained `GkrProofFile` (bincode) that
//! `cp_verify` checks without the trace.

#[path = "common/mod.rs"]
mod common;

use std::path::PathBuf;

use clap::Parser;
use flashprove::field::Fp;
use flashprove::gkr::driver::{
    ops_per_segment, DriverError, HeadSpec, LookupBounds, ProvedModel, SegmentBundle,
};
use flashprove::model::spec::ModelSpec;
use flashprove::pcs::whir_stacked::Params;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    spec: PathBuf,
    #[arg(long)]
    weights: PathBuf,
    #[arg(long)]
    trace: PathBuf,
    #[arg(long)]
    prompt: PathBuf, // x
    #[arg(long)]
    output: PathBuf, // y
    #[arg(long)]
    fill: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// Requant digit-decomposition width (bytes per value).
    #[arg(long, default_value_t = 6)]
    digits: usize,
    /// Softmax/reciprocal fixed-point exponent.
    #[arg(long, default_value_t = flashprove::model::spec::RECIP_Q)]
    rec_q: u32,
    /// `eps` for the RMSNorm denominator.
    #[arg(long, default_value_t = 1e-6)]
    eps: f64,
    /// Maximum number of committed tables per segment.
    #[arg(long, default_value_t = 40)]
    segments: usize,
    /// The cell holding the last logits row (the head's input).
    #[arg(long, default_value = "logits")]
    head_cell: String,
    /// Prove only this segment (debugging; the output file holds one segment).
    #[arg(long)]
    only: Option<usize>,
}

fn main() {
    let args = Args::parse();

    let spec: ModelSpec = common::load(&args.spec);
    // Only the trace's headers are read here; each segment's cells are then read on
    // demand, so peak memory is one segment rather than the whole trace.
    let trace = flashprove::model::trace_file::TraceIndex::read(&args.trace).unwrap_or_else(|e| {
        panic!("trace {}: {e}", args.trace.display());
    });
    let x: Vec<usize> = common::load(&args.prompt);
    let y: Vec<usize> = common::load(&args.output);
    let fill: Vec<usize> = common::load(&args.fill);

    // Reconstruct the window the engine actually ran: x + y + fill, zero-padded to
    // the embedding row count (the window length).
    let l = trace
        .window_len()
        .unwrap_or(x.len() + y.len() + fill.len());
    let mut window = Vec::with_capacity(l);
    window.extend_from_slice(&x);
    window.extend_from_slice(&y);
    window.extend_from_slice(&fill);
    window.resize(l, 0);

    let head_token = trace
        .head_token()
        .unwrap_or_else(|| panic!("trace has no head token"));

    let head_max = {
        let logits = trace
            .cell(&args.head_cell)
            .unwrap_or_else(|e| panic!("trace cell {}: {e}", args.head_cell));
        logits.values[(logits.rows - 1) * logits.cols + head_token]
    };
    // The bounds scan a handful of cells; they are read one at a time.
    let bounds = LookupBounds::derive_with(&spec, |name| trace.cell(name).ok());

    // Split the op graph into segments and tie every cross-segment input back to the
    // segment that owns it (see `driver::link_schedule`).
    let segment_cells =
        flashprove::gkr::driver::plan_segments(&spec.ops, &spec.cells, args.segments);
    let per_segment =
        ops_per_segment(&spec.ops, &segment_cells).unwrap_or_else(|e| panic!("{e:?}"));
    let links = flashprove::gkr::driver::link_schedule(&segment_cells, &per_segment);
    let arity = |name: &str| {
        let (rows, cols) = trace
            .shape(name)
            .unwrap_or_else(|| panic!("trace has no `{name}` cell"));
        (rows * cols).next_power_of_two().trailing_zeros() as usize
    };
    let openings = flashprove::gkr::driver::link_openings(&links, segment_cells.len(), arity);

    let head_segment = segment_cells
        .iter()
        .position(|cs| cs.iter().any(|c| c == &args.head_cell))
        .unwrap_or(segment_cells.len() - 1);
    // One weight set for the run: the file's table of contents is parsed once, and a
    // tensor stays committed while segments need it.
    let mut set = flashprove::gkr::driver::WeightSet::from_file(&args.weights, Params::default())
        .unwrap_or_else(|e| panic!("weights {}: {e}", args.weights.display()));
    let mut segments: Vec<SegmentBundle> = Vec::with_capacity(segment_cells.len());
    for s in 0..segment_cells.len() {
        if args.only.is_some_and(|o| o != s) {
            continue;
        }
        let head = if s == head_segment {
            Some(HeadSpec {
                logits: args.head_cell.clone(),
                token: head_token,
                max: head_max,
            })
        } else {
            None
        };
        // Own cells plus a copy of every input an earlier segment owns.
        let read = |name: &str| {
            trace
                .cell(name)
                .unwrap_or_else(|e| panic!("trace cell {name}: {e}"))
        };
        let mut cells: Vec<flashprove::model::Cell> =
            segment_cells[s].iter().map(|n| read(n)).collect();
        for name in flashprove::gkr::driver::external_inputs(&segment_cells[s], &per_segment[s]) {
            if !segment_cells[s].iter().any(|c| c == &name) {
                cells.push(read(&name));
            }
        }
        // Outgoing and incoming links: open our own cells (and the copies we hold) at
        // the points the other segment reads.
        let extra = openings[s].clone();

        eprintln!(
            "segment {s}/{}: {} cells, {} ops, {} links out",
            segment_cells.len(),
            cells.len(),
            per_segment[s].len(),
            extra.len()
        );
        let t0 = std::time::Instant::now();
        let bundle = flashprove::gkr::driver::prove_segment_full(
            &per_segment[s],
            &cells,
            &mut set,
            &spec.dims,
            (args.digits, Fp::new(11), Fp::new(5), args.rec_q, args.eps),
            &bounds,
            Params::default(),
            &window,
            head.as_ref(),
            &extra,
        )
        .unwrap_or_else(|e: DriverError| panic!("segment {s}: {e:?}"));
        eprintln!("segment {s}: proved in {:.1}s", t0.elapsed().as_secs_f64());
        segments.push(bundle);
    }

    let file = ProvedModel {
        segments,
        window,
        bounds,
        head_token,
        head_max,
        head_segment,
    };
    common::save_bin(&args.out, &file);
    println!("proof -> {}", args.out.display());
}
