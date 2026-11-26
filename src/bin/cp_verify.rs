//! cp_verify: verify a GKR succinct proof against the pre-shared weight
//! commitments produced by `cp_commit`, without the trace.

#[path = "common/mod.rs"]
mod common;

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use flashprove::gkr::driver::{ops_per_segment, verify_segment_with_head, HeadSpec, ProvedModel};
use flashprove::model::spec::ModelSpec;
use flashprove::pcs::whir_stacked::SegmentCommitment;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    spec: PathBuf,
    #[arg(long)]
    commit: PathBuf,
    /// Integer weights: the verifier needs the norm γ vectors (and each embedding's
    /// row count). They are pre-shared with the prover, like the commitments.
    #[arg(long)]
    weights: PathBuf,
    #[arg(long)]
    proof: PathBuf,
    /// Requant digit-decomposition width (must match the proof).
    #[arg(long, default_value_t = 6)]
    digits: usize,
    /// Softmax/reciprocal fixed-point exponent (must match the proof).
    #[arg(long, default_value_t = flashprove::model::spec::RECIP_Q)]
    rec_q: u32,
    /// `eps` for the RMSNorm denominator (must match the proof).
    #[arg(long, default_value_t = 1e-6)]
    eps: f64,
    /// Maximum number of committed tables per segment (must match the proof).
    #[arg(long, default_value_t = 40)]
    segments: usize,
    /// The cell holding the last logits row (the head's input).
    #[arg(long, default_value = "logits")]
    head_cell: String,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let spec: ModelSpec = common::load(&args.spec);
    // The pre-shared map: one root per weight tensor, as `cp_commit` wrote it.
    let commits: Vec<(String, SegmentCommitment)> = common::load_bin(&args.commit);
    let published: HashMap<String, SegmentCommitment> = commits.into_iter().collect();
    // Only the norm γ vectors and the embedding's shape are read; the rest of the
    // 553 MB weight file is skipped.
    let weights = common::load_weights_bin_subset(&args.weights, |name| {
        name.contains("norm") || name == "embed_tokens"
    });
    let file: ProvedModel = common::load_bin(&args.proof);


    // Ops are assigned to segments by the spec's own split, so the verifier derives
    // the same schedule the prover used (including the cross-segment links).
    let plan = flashprove::gkr::driver::plan_segments(&spec.ops, &spec.cells, args.segments);
    if plan.len() != file.segments.len() {
        println!("reject (segment count)");
        return ExitCode::FAILURE;
    }
    let Some(per_segment) = ops_per_segment(&spec.ops, &plan).ok() else {
        println!("reject (ops do not map to segments)");
        return ExitCode::FAILURE;
    };

    // The segments are independent: each replays its own transcript against its own
    // commitment, and `verify_links` ties the cross-segment copies afterwards. So they
    // run in parallel, one thread each; a segment's own checks stay sequential.
    let params = (
        args.digits,
        flashprove::field::Fp::new(11),
        flashprove::field::Fp::new(5),
        args.rec_q,
        args.eps,
    );
    let links_ok = std::cell::Cell::new(true);
    let mut ok = true;
    let outcomes: Vec<(usize, bool, f64)> = std::thread::scope(|scope| {
        // The links only need the commitments, so they run alongside the segments
        // instead of after them.
        let links_handle = {
            let (file, spec) = (&file, &spec);
            let segments = args.segments;
            scope.spawn(move || flashprove::gkr::driver::verify_links(file, spec, segments))
        };
        let mut handles = Vec::with_capacity(file.segments.len());
        for (s, bundle) in file.segments.iter().enumerate() {
            let head = (s == file.head_segment).then(|| HeadSpec {
                logits: args.head_cell.clone(),
                token: file.head_token,
                max: file.head_max,
            });
            let ops = &per_segment[s];
            let published = &published;
            let weights = &weights;
            let dims = &spec.dims;
            let bounds = &file.bounds;
            let window = &file.window;
            handles.push(scope.spawn(move || {
                let t0 = std::time::Instant::now();
                let this = verify_segment_with_head(
                    ops,
                    bundle,
                    published,
                    weights,
                    dims,
                    params,
                    bounds,
                    window,
                    head.as_ref(),
                );
                (s, this, t0.elapsed().as_secs_f64())
            }));
        }
        let mut all: Vec<(usize, bool, f64)> =
            handles.into_iter().map(|h| h.join().unwrap()).collect();
        links_ok.set(links_handle.join().unwrap());
        all.sort_by_key(|t| t.0);
        all
    });
    for (s, this, secs) in outcomes {
        eprintln!(
            "segment {s}: {} in {secs:.1}s",
            if this { "ok" } else { "REJECT" }
        );
        ok &= this;
    }

    // Every cross-segment copy must equal the cell it copies (checked above, in
    // parallel with the segments).
    let links = links_ok.get();
    if !links {
        eprintln!("links: REJECT");
    }
    ok &= links;

    println!("{}", if ok { "accept" } else { "reject" });
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
