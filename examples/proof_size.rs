//! Where do the bytes of `data/proof.bin` go?

#[path = "../src/bin/common/mod.rs"]
mod common;

use flashprove::gkr::driver::ProvedModel;

fn sz<T: serde::Serialize>(v: &T) -> usize {
    bincode::serialized_size(v).unwrap_or(0) as usize
}

fn mb(n: usize) -> String {
    format!("{:.2} MB", n as f64 / 1e6)
}

fn main() {
    let path = std::path::Path::new("data/proof.bin");
    let file: ProvedModel = common::load_bin(path);
    println!("total {}", mb(sz(&file)));
    println!("  window {:?}", sz(&file.window));
    println!("  bounds {}", sz(&file.bounds));
    for (s, b) in file.segments.iter().enumerate() {
        let seg = &b.segment;
        let ops: usize = seg.ops.iter().map(sz).sum();
        println!(
            "segment {s}: commitment {} claims {} pcs-proof {} ops {} head {} | weight proofs {}",
            mb(sz(&seg.commitment)),
            mb(sz(&seg.claims)),
            mb(sz(&seg.proof)),
            mb(ops),
            mb(sz(&seg.head)),
            mb(sz(&b.weights)),
        );
        if s == 0 {
            for (k, o) in seg.ops.iter().enumerate().take(3) {
                println!("    op {k}: {}", mb(sz(o)));
            }
            let mut by_size: Vec<(usize, &str)> = Vec::new();
            for o in seg.ops.iter() {
                by_size.push((sz(o), op_name(o)));
            }
            by_size.sort_by(|a, b| b.0.cmp(&a.0));
            for (n, name) in by_size.iter().take(6) {
                println!("    big op: {name} {}", mb(*n));
            }
        }
    }
}

fn op_name(p: &flashprove::gkr::driver::SegmentOpProof) -> &'static str {
    use flashprove::gkr::driver::SegmentOpProof as P;
    match p {
        P::Add(_) => "add",
        P::AddRq { .. } => "add+requant",
        P::Mul(_) => "mul",
        P::Reshape(_) => "reshape",
        P::MulRq { .. } => "mul+requant",
        P::Matmul { .. } => "matmul",
        P::Silu(_) => "silu",
        P::WeightedSum(_) => "weighted_sum",
        P::Norm(_) => "norm",
        P::Softmax { .. } => "softmax",
        P::Rope(_) => "rope",
        P::Embed(_) => "embed",
        P::TopK { .. } => "top_k",
        P::WeightedV(_) => "weighted_v",
        P::Scores(_) => "scores",
    }
}
