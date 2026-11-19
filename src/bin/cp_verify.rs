//! cp_verify: verify a proof (Construction 4.15 V_Z), decoding each segment's
//! commit one at a time so peak memory stays bounded.

#[path = "common/mod.rs"]
mod common;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use flashprove::protocol::{verify, verify_segment, CommitOutput, Config, Proof};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    /// Prefix for per-segment commit files: `<commit>-<segment>.bin`.
    #[arg(long)]
    commit: PathBuf,
    #[arg(long)]
    proof: PathBuf,
    #[arg(long)]
    prompt: PathBuf, // x
    #[arg(long)]
    output: PathBuf, // y
    #[arg(long)]
    spec: PathBuf,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let config: Config = common::load(&args.config);
    let proof: Proof = common::load(&args.proof);
    let x: Vec<usize> = common::load(&args.prompt);
    let y: Vec<usize> = common::load(&args.output);
    let m = common::load_spec(&args.spec);

    // Decode each segment's commit one at a time, collecting only the digit
    // matrices (each ~a few hundred MB), so peak memory stays bounded.
    let mut matrices = Vec::with_capacity(config.segments.len());
    for (s, sc) in config.segments.iter().enumerate() {
        let stem = args.commit.file_stem().unwrap_or_default();
        let path = args
            .commit
            .with_file_name(format!("{}-{s}.bin", stem.to_string_lossy()));
        let co: CommitOutput = common::load_bin(&path);
        match verify_segment(&sc.layout, &config.dg, &co) {
            Some(a) => matrices.push(a),
            None => {
                println!("reject");
                return ExitCode::FAILURE;
            }
        }
    }

    let ok = verify(m.as_ref(), &config, &x, &y, &proof, &matrices);
    println!("{}", if ok { "accept" } else { "reject" });
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
