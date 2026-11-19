//! gkr_prove: run the GKR op-graph verifier end-to-end on a real model.
//!
//! Loads a `ModelSpec` (JSON), integer weights (binary), and a trace (JSON)
//! emitted by `python/run_integer.py`, reconstructs the token window from the
//! prompt/output/fill files, then proves and verifies in one process, printing
//! `accept` / `reject`.
//!
//! This is the succinct (O(depth·log n)) path: `prove_gkr_succinct` +
//! `verify_gkr_succinct`. The verifier consumes only the proof, not the trace.

#[path = "common/mod.rs"]
mod common;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use flashprove::committed_succinct::CommittedConfig;
use flashprove::fri;
use flashprove::gkr_succinct::{prove_gkr_succinct, verify_gkr_succinct};
use flashprove::model::Trace;
use flashprove::spec::ModelSpec;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    spec: PathBuf,
    #[arg(long)]
    weights: PathBuf,
    #[arg(long)]
    trace: PathBuf,
    /// Token ids `x` (the prompt), as a JSON array.
    #[arg(long)]
    prompt: PathBuf,
    /// Token ids `y` (the greedy continuation), as a JSON array.
    #[arg(long)]
    output: PathBuf,
    /// Optional fill tokens, as a JSON array (may be empty).
    #[arg(long)]
    fill: PathBuf,
    /// Lookup/range-check bounds (JSON `CommittedConfig`). If omitted, the
    /// small-model test bounds are used.
    #[arg(long)]
    cfg: Option<PathBuf>,
    /// FRI blowup (log2) — must match any future verifier deployment.
    #[arg(long, default_value_t = 1)]
    log_blowup: usize,
    /// FRI query count.
    #[arg(long, default_value_t = 10)]
    num_queries: usize,
}

fn main() -> ExitCode {
    let args = Args::parse();

    let spec: ModelSpec = common::load(&args.spec);
    let weights = common::load_weights_bin(&args.weights);
    let trace: Trace = common::load(&args.trace);
    let x: Vec<usize> = common::load(&args.prompt);
    let y: Vec<usize> = common::load(&args.output);
    let fill: Vec<usize> = common::load(&args.fill);

    // Reconstruct the token window the Python engine actually ran: x + y + fill,
    // zero-padded to the window length (the embedding cell's row count).
    let l = trace.segments[0]
        .cells
        .first()
        .map(|c| c.rows)
        .unwrap_or(x.len() + y.len() + fill.len());
    let mut window: Vec<usize> = Vec::with_capacity(l);
    window.extend_from_slice(&x);
    window.extend_from_slice(&y);
    window.extend_from_slice(&fill);
    window.resize(l, 0);

    let head_token = trace.segments[0].head_token;

    let cfg = match &args.cfg {
        Some(p) => common::load::<CommittedConfig>(p),
        None => CommittedConfig::derive(&spec, &trace),
    };

    let mut ctx = fri::Prover::new(args.log_blowup, args.num_queries);
    let proof = prove_gkr_succinct(&mut ctx, &spec, &weights, &trace, &window, &cfg);
    let ok = verify_gkr_succinct(&ctx.pcs, &spec, &window, head_token, &cfg, &proof);

    println!("{}", if ok { "accept" } else { "reject" });
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
