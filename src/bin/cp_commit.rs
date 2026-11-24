//! cp_commit: commit the model's weights (offline phase), writing the public
//! WHIR roots — one per weight tensor — to a file. The prove phase re-derives the
//! same (deterministic) roots; the verify phase binds the proof to these
//! pre-shared roots.

#[path = "common/mod.rs"]
mod common;

use std::path::PathBuf;

use clap::Parser;
use flashprove::gkr::segment::commit_weights_whir;
use flashprove::pcs::whir_stacked::Params;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    weights: PathBuf,
    /// The pre-shared map the verifier consumes: `Vec<(name, SegmentCommitment)>`.
    #[arg(long)]
    out: PathBuf,
}

fn main() {
    let args = Args::parse();
    let weights = common::load_weights_bin(&args.weights);

    let map = commit_weights_whir(&weights, Params::default());
    common::save_bin(&args.out, &map);
    println!("committed {} weights -> {}", map.len(), args.out.display());
}
