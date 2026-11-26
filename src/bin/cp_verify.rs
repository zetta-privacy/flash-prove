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
    /// Verify only this segment (debugging). Cross-segment links are skipped.
    #[arg(long)]
    only: Option<usize>,
    /// Print per-segment proof size and claim counts, then exit without verifying.
    #[arg(long)]
    stats: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let t_load = std::time::Instant::now();
    let tick = |what: &str, t: &std::time::Instant| {
        if std::env::var("FLASHPROVE_PROFILE").is_ok_and(|v| v == "1") {
            eprintln!("  load {what}: {:.2}s", t.elapsed().as_secs_f64());
        }
    };
    let spec: ModelSpec = common::load(&args.spec);
    // The pre-shared map: one root per weight tensor, as `cp_commit` wrote it.
    let commits: Vec<(String, SegmentCommitment)> = common::load_bin(&args.commit);
    let published: HashMap<String, SegmentCommitment> = commits.into_iter().collect();
    tick("commit", &t_load);
    let t_w = std::time::Instant::now();
    // Only the norm γ vectors and the embedding's shape are read; the rest of the
    // 553 MB weight file is skipped.
    let weights = common::load_weights_bin_subset(&args.weights, |name| {
        name.contains("norm") || name == "embed_tokens"
    });
    tick("weights", &t_w);
    let t_p = std::time::Instant::now();
    let file: ProvedModel = common::load_bin(&args.proof);
    tick("proof", &t_p);
    tick("all", &t_load);

    if args.stats {
        for (s, b) in file.segments.iter().enumerate() {
            let cells = b.segment.commitment.cells.len();
            let claims = b.segment.claims.len();
            let values: usize = b
                .segment
                .commitment
                .pcs
                .tables
                .iter()
                .map(|t| 1usize << t.m)
                .sum();
            println!(
                "segment {s}: {cells} cells, {} pcs tables, {values} table rows, {claims} claims, {} ops",
                b.segment.commitment.pcs.tables.len(),
                b.segment.ops.len()
            );
            // Which families of cells account for the tables (and so for PCS time).
            let mut fam: std::collections::HashMap<String, usize> =
                std::collections::HashMap::new();
            for c in &b.segment.commitment.cells {
                let mut parts = c.name.split('.');
                let key = match (parts.next(), parts.next()) {
                    (Some("__norm"), Some(_)) => "__norm.*".to_string(),
                    (Some(k), _) => k.to_string(),
                    (None, _) => "?".to_string(),
                };
                *fam.entry(key).or_default() += 1;
            }
            let mut f: Vec<(String, usize)> = fam.into_iter().collect();
            f.sort_by(|a, b| b.1.cmp(&a.1));
            let top: Vec<String> = f.iter().take(6).map(|(k, v)| format!("{k}={v}")).collect();
            println!("    tables by family: {}", top.join(" "));
        }
        return ExitCode::SUCCESS;
    }

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
    let t_all = std::time::Instant::now();
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
        let links_handle = args.only.is_none().then(|| {
            let (file, spec) = (&file, &spec);
            let segments = args.segments;
            scope.spawn(move || flashprove::gkr::driver::verify_links(file, spec, segments))
        });
        let mut handles = Vec::with_capacity(file.segments.len());
        for (s, bundle) in file.segments.iter().enumerate() {
            if args.only.is_some_and(|o| o != s) {
                continue;
            }
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
        if let Some(h) = links_handle {
            links_ok.set(h.join().unwrap());
        }
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
    if std::env::var("FLASHPROVE_PROFILE").is_ok_and(|v| v == "1") {
        eprintln!(
            "  profile segments: {:.2}s wall",
            t_all.elapsed().as_secs_f64()
        );
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
