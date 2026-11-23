//! cp_commit: commit the model's weights (offline phase), writing the public
//! commitments to a file. The prove phase re-derives the same (deterministic)
//! commitments; the verify phase binds the proof to these pre-shared roots.

#[path = "common/mod.rs"]
mod common;

use std::path::PathBuf;

use clap::Parser;
use flashprove::gkr::commit::{commit_weights_public, commit_weights_whir};
use flashprove::pcs::fri;
use flashprove::pcs::whir_stacked::Params;

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
    /// Also write the WHIR pre-shared map — one root per weight tensor, the form the
    /// stacked-WHIR verifier consumes. The FRI commitment above is left untouched so
    /// the current prove/verify path keeps working until the op graph is ported.
    #[arg(long)]
    whir_out: Option<PathBuf>,
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

    if let Some(path) = &args.whir_out {
        let map = commit_weights_whir(&weights, Params::default());
        common::save_bin(path, &map);
        println!("whir roots for {} tensors -> {}", map.len(), path.display());
    }
}
