//! cp_prove: produce the request proof (Construction D.12 P_Z). The commit
//! material already lives in the per-segment commit files, so this only checks
//! the request is encodable and carries the trace.

#[path = "common/mod.rs"]
mod common;

use std::path::PathBuf;

use clap::Parser;
use flashprove::model::Trace;
use flashprove::protocol::{prove, Proof};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    trace: PathBuf,
    #[arg(long)]
    prompt: PathBuf, // x
    #[arg(long)]
    output: PathBuf, // y
    #[arg(long)]
    fill: PathBuf, // fill tokens (may be empty)
    #[arg(long)]
    out: PathBuf,
}

fn main() {
    let args = Args::parse();
    let trace: Trace = common::load(&args.trace);
    let x: Vec<usize> = common::load(&args.prompt);
    let y: Vec<usize> = common::load(&args.output);
    let fill: Vec<usize> = common::load(&args.fill);

    let proof: Proof = prove(trace, &x, &y, &fill).expect("request not encodable");
    common::save(&args.out, &proof);
}
