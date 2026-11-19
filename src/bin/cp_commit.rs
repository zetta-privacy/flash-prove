//! cp_commit: commit the model's weights, one segment at a time
//! (Construction D.12 Commit_Z), writing one commit file per segment so peak
//! memory stays bounded by the largest segment.

#[path = "common/mod.rs"]
mod common;

use std::path::PathBuf;

use clap::Parser;
use flashprove::model::EXACT_MATCH;
use flashprove::protocol::{commit, gen_coins, Config, SegmentConfig, TableParams};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    weights: PathBuf,
    #[arg(long)]
    spec: PathBuf,
    #[arg(long)]
    seed0: u64,
    #[arg(long)]
    seed1: u64,
    /// Prefix for per-segment commit files: `<out>-<segment>.bin`.
    #[arg(long)]
    out: PathBuf,
}

fn seed32(v: u64) -> [u8; 32] {
    let mut s = [0u8; 32];
    s[..8].copy_from_slice(&v.to_le_bytes());
    s
}

fn main() {
    let args = Args::parse();
    let weights = common::load_weights_bin(&args.weights);
    let m = common::load_spec(&args.spec);

    let n = m.num_segments();

    // Compute the layout per segment (from that segment's digit matrix size) and
    // write the config first.
    let mut segments = Vec::with_capacity(n);
    for s in 0..n {
        let a = m.to_digit_matrix(&weights, s);
        segments.push(SegmentConfig {
            layout: TableParams::flat(a.entries.len(), 1),
        });
    }
    let config = Config {
        segments,
        policy: EXACT_MATCH,
        dg: [0u8; 32],
    };
    common::save(&args.config, &config);

    for s in 0..n {
        let layout = &config.segments[s].layout;
        let coins0 = gen_coins(layout, &seed32(args.seed0));
        let coins1 = gen_coins(layout, &seed32(args.seed1));
        let output = commit(
            m.as_ref(),
            layout,
            &weights,
            &config.dg,
            &coins0,
            &coins1,
            s,
        );
        let stem = args.out.file_stem().unwrap_or_default();
        let path = args
            .out
            .with_file_name(format!("{}-{s}.bin", stem.to_string_lossy()));
        common::save_bin(&path, &output);
        println!("segment {s}: commitment = {}", common::hex(&output.c));
    }
}
