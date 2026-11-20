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
use flashprove::gkr::commit::CommittedConfig;
use flashprove::pcs::fri;
use flashprove::gkr::succinct::prove_gkr_succinct;
use flashprove::model::Trace;
use flashprove::model::spec::ModelSpec;

use common::GkrProofFile;

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
    #[arg(long, default_value_t = 1)]
    log_blowup: usize,
    #[arg(long, default_value_t = 10)]
    num_queries: usize,
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
    let cfg = CommittedConfig::derive(&spec, &trace);

    let mut ctx = fri::Prover::new(args.log_blowup, args.num_queries);
    let proof = prove_gkr_succinct(&mut ctx, &spec, &weights, &trace, &window, &cfg);

    let file = GkrProofFile {
        head_token,
        window,
        cfg,
        proof,
    };
    common::save_bin(&args.out, &file);
    println!("proof -> {}", args.out.display());
}
