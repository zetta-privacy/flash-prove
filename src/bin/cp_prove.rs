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
use flashprove::model::Trace;
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
    #[arg(long, default_value_t = 24)]
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
}

fn main() {
    let args = Args::parse();

    let spec: ModelSpec = common::load(&args.spec);
    let weights = common::load_weights_bin(&args.weights);
    let trace: Trace = common::load(&args.trace);
    let x: Vec<usize> = common::load(&args.prompt);
    let y: Vec<usize> = common::load(&args.output);
    let fill: Vec<usize> = common::load(&args.fill);

    // Reconstruct the window the engine actually ran: x + y + fill, zero-padded to
    // the embedding row count (the window length).
    let l = trace.segments[0]
        .cells
        .first()
        .map(|c| c.rows)
        .unwrap_or(x.len() + y.len() + fill.len());
    let mut window = Vec::with_capacity(l);
    window.extend_from_slice(&x);
    window.extend_from_slice(&y);
    window.extend_from_slice(&fill);
    window.resize(l, 0);

    let head_token = match trace.segments[0].head_token {
        Some(h) => h,
        None => panic!("trace has no head token"),
    };

    // One map from name to the trace cell; the trace may arrive as a single segment.
    let by_name: std::collections::HashMap<&str, &flashprove::model::Cell> = trace
        .segments
        .iter()
        .flat_map(|s| s.cells.iter())
        .map(|c| (c.name.as_str(), c))
        .collect();

    let head_max = {
        let logits = by_name
            .get(args.head_cell.as_str())
            .unwrap_or_else(|| panic!("trace has no `{}` cell", args.head_cell));
        logits.values[(logits.rows - 1) * logits.cols + head_token]
    };
    let bounds = LookupBounds::derive(&spec, &trace);

    // Split the op graph into segments and tie every cross-segment input back to the
    // segment that owns it (see `driver::link_schedule`).
    let segment_cells =
        flashprove::gkr::driver::plan_segments(&spec.ops, &spec.cells, args.segments);
    let per_segment =
        ops_per_segment(&spec.ops, &segment_cells).unwrap_or_else(|e| panic!("{e:?}"));
    let links = flashprove::gkr::driver::link_schedule(&segment_cells, &per_segment);
    let arity = |name: &str| {
        let cell = by_name[name];
        (cell.rows * cell.cols).next_power_of_two().trailing_zeros() as usize
    };
    let openings = flashprove::gkr::driver::link_openings(&links, segment_cells.len(), arity);

    let head_segment = segment_cells
        .iter()
        .position(|cs| cs.iter().any(|c| c == &args.head_cell))
        .unwrap_or(segment_cells.len() - 1);
    let mut segments: Vec<SegmentBundle> = Vec::with_capacity(segment_cells.len());
    for s in 0..segment_cells.len() {
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
        let mut cells: Vec<flashprove::model::Cell> = segment_cells[s]
            .iter()
            .map(|n| (*by_name[n.as_str()]).clone())
            .collect();
        for name in flashprove::gkr::driver::external_inputs(&segment_cells[s], &per_segment[s]) {
            if !segment_cells[s].iter().any(|c| c == &name) {
                cells.push((*by_name[name.as_str()]).clone());
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
            &weights,
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
