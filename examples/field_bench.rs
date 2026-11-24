//! CPU baseline for the GPU field benchmark: the same kernel shape —
//! `out[i] = a[i]*b[i] + c[i]` over a large array — using the field types the
//! prover actually uses.
//!
//! Run: cargo run --release --example field_bench

use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use std::time::Instant;

fn main() {
    for log_n in [18usize, 20, 22, 24] {
        let n = 1usize << log_n;
        let mut s = 0x1234_5678_9abc_def0u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let a: Vec<Goldilocks> = (0..n).map(|_| Goldilocks::new(rnd())).collect();
        let b: Vec<Goldilocks> = (0..n).map(|_| Goldilocks::new(rnd())).collect();
        let c: Vec<Goldilocks> = (0..n).map(|_| Goldilocks::new(rnd())).collect();
        let mut out = vec![Goldilocks::ZERO; n];

        // warm up + pick a repetition count that runs for ~0.3 s
        let mut reps = 1usize;
        loop {
            let t0 = Instant::now();
            for _ in 0..reps {
                for i in 0..n {
                    out[i] = a[i] * b[i] + c[i];
                }
            }
            if t0.elapsed().as_secs_f64() > 0.3 || reps > 1_000_000 {
                let ns = t0.elapsed().as_nanos() as f64 / reps as f64;
                let per = ns / n as f64;
                println!(
                    "2^{:<2}  Goldilocks  cpu {:8.3} ns/elem   {:6.2} Gop/s   (sink {})",
                    log_n,
                    per,
                    n as f64 / ns,
                    out[0].as_canonical_u64() & 1
                );
                break;
            }
            reps *= 2;
        }
    }
}
