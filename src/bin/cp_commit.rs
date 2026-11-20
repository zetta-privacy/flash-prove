//! cp_commit: commit the model's weights (offline phase), writing the public
//! commitments to a file. The prove phase re-derives the same (deterministic)
//! commitments; the verify phase binds the proof to these pre-shared roots.

#[path = "common/mod.rs"]
mod common;

use std::path::PathBuf;

use clap::Parser;
use flashprove::gkr::commit::commit_weights_public;
use flashprove::pcs::fri;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    weights: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 1)]
    log_blowup: usize,
    #[arg(long, default_value_t = 10)]
    num_queries: usize,
}

fn main() {
    let args = Args::parse();
    let weights = common::load_weights_bin(&args.weights);

    let (pcs, _) = fri::new_pcs(args.log_blowup, args.num_queries);
    let commits = commit_weights_public(&pcs, &weights);

    common::save_bin(&args.out, &commits);
    println!(
        "committed {} weights -> {}",
        commits.len(),
        args.out.display()
    );
}
