//! cp_verify: verify a GKR succinct proof against the pre-shared weight
//! commitments produced by `cp_commit`, without the trace.

#[path = "common/mod.rs"]
mod common;

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use flashprove::gkr::commit::WeightCommit;
use flashprove::pcs::fri;
use flashprove::gkr::succinct::verify_gkr_succinct;
use flashprove::model::spec::ModelSpec;

use common::GkrProofFile;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    spec: PathBuf,
    #[arg(long)]
    commit: PathBuf,
    #[arg(long)]
    proof: PathBuf,
    #[arg(long, default_value_t = 1)]
    log_blowup: usize,
    #[arg(long, default_value_t = 10)]
    num_queries: usize,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let spec: ModelSpec = common::load(&args.spec);
    let commits: Vec<WeightCommit> = common::load_bin(&args.commit);
    let file: GkrProofFile = common::load_bin(&args.proof);

    // Bind: the proof's weight commitments must equal the pre-shared roots.
    let committed: HashMap<String, fri::Commitment> =
        commits.into_iter().map(|c| (c.name, c.com)).collect();
    if committed != file.proof.weight_coms {
        println!("reject (weight commitment mismatch)");
        return ExitCode::FAILURE;
    }

    let (pcs, _) = fri::new_pcs(args.log_blowup, args.num_queries);
    let ok = verify_gkr_succinct(
        &pcs,
        &spec,
        &file.window,
        Some(file.head_token),
        &file.cfg,
        &file.proof,
    );

    println!("{}", if ok { "accept" } else { "reject" });
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
