//! cp_verify: verify a GKR succinct proof against the pre-shared weight
//! commitments produced by `cp_commit`, without the trace.

#[path = "common/mod.rs"]
mod common;

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use flashprove::gkr::driver::{
    ops_per_segment, verify_segment_with_head, HeadSpec, ProvedModel,
};
use flashprove::model::spec::ModelSpec;
use flashprove::pcs::whir_stacked::SegmentCommitment;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    spec: PathBuf,
    #[arg(long)]
    commit: PathBuf,
    #[arg(long)]
    proof: PathBuf,
    /// Requant digit-decomposition width (must match the proof).
    #[arg(long, default_value_t = 6)]
    digits: usize,
    /// Softmax/reciprocal fixed-point exponent (must match the proof).
    #[arg(long, default_value_t = 24)]
    rec_q: u32,
    /// `eps` for the RMSNorm denominator (must match the proof).
    #[arg(long, default_value_t = 1e-6)]
    eps: f64,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let spec: ModelSpec = common::load(&args.spec);
    // The pre-shared map: one root per weight tensor, as `cp_commit --whir-out`
    // wrote it.
    let commits: Vec<(String, SegmentCommitment)> = common::load_bin(&args.commit);
    let published: HashMap<String, SegmentCommitment> = commits.into_iter().collect();
    let file: ProvedModel = common::load_bin(&args.proof);

    // Ops are assigned to segments by the cell names each segment committed, so the
    // verifier never needs the trace.
    let segment_cells: Vec<Vec<String>> = file
        .segments
        .iter()
        .map(|b| b.segment.commitment.cells.iter().map(|c| c.name.clone()).collect())
        .collect();
    let Some(per_segment) = ops_per_segment(&spec.ops, &segment_cells).ok() else {
        println!("reject (ops do not map to segments)");
        return ExitCode::FAILURE;
    };

    let mut ok = true;
    for (s, bundle) in file.segments.iter().enumerate() {
        let head = if s == file.head_segment {
            let logits = segment_cells[s].last().cloned().unwrap_or_default();
            Some(HeadSpec {
                logits,
                token: file.head_token,
                max: file.head_max,
            })
        } else {
            None
        };
        ok &= verify_segment_with_head(
            &per_segment[s],
            bundle,
            &published,
            &[],
            &spec.dims,
            (args.digits, flashprove::field::Fp::new(11), flashprove::field::Fp::new(5), args.rec_q, args.eps),
            &file.bounds,
            &file.window,
            head.as_ref(),
        );
    }

    println!("{}", if ok { "accept" } else { "reject" });
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
