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
    ops_per_segment, prove_segment_with_head, DriverError, HeadSpec, LookupBounds, ProvedModel,
    SegmentBundle,
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
    let head_max = {
        let seg = trace.segments.last().expect("at least one segment");
        let logits = seg.cells.last().expect("logits cell");
        logits.values[(logits.rows - 1) * logits.cols + head_token]
    };
    let bounds = LookupBounds::derive(&spec, &trace);

    // Assign every op to the segment that owns its output cell.
    let segment_cells: Vec<Vec<String>> = trace
        .segments
        .iter()
        .map(|s| s.cells.iter().map(|c| c.name.clone()).collect())
        .collect();
    let per_segment =
        ops_per_segment(&spec.ops, &segment_cells).unwrap_or_else(|e| panic!("{e:?}"));

    let head_segment = trace.segments.len() - 1;
    let mut segments: Vec<SegmentBundle> = Vec::with_capacity(trace.segments.len());
    for (s, seg) in trace.segments.iter().enumerate() {
        let head = if s == head_segment {
            Some(HeadSpec {
                logits: trace.segments[s]
                    .cells
                    .last()
                    .map(|c| c.name.clone())
                    .unwrap_or_default(),
                token: head_token,
                max: head_max,
            })
        } else {
            None
        };
        eprintln!(
            "segment {s}/{}: {} cells, {} ops",
            trace.segments.len(),
            seg.cells.len(),
            per_segment[s].len()
        );
        let t0 = std::time::Instant::now();
        let bundle = prove_segment_with_head(
            &per_segment[s],
            &seg.cells,
            &weights,
            &spec.dims,
            (args.digits, Fp::new(11), Fp::new(5), args.rec_q, args.eps),
            &bounds,
            Params::default(),
            &window,
            head.as_ref(),
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
