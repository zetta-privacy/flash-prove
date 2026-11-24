//! GPU check: validates and times each backend kernel, CPU against GPU.
//!
//! This is the instrument that decides what is worth porting. It never runs as part of
//! the test suite (it needs a real GPU and the Metal shader compiler).
//!
//! Run: cargo run --release --example gpu_check

use flashprove::backend;
use flashprove::field::{Ext, Fp};
use p3_field::PrimeCharacteristicRing;

use std::time::Instant;

fn main() {
    #[cfg(not(target_os = "macos"))]
    {
        println!("no GPU backend on this platform");
        return;
    }

    #[cfg(target_os = "macos")]
    {
        use flashprove::backend::metal;

        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s % 0xFFFF_FFFF_0000_0001
        };

        println!(
            "{:<6} {:<10} {:>10} {:>10} {:>8}",
            "size", "kernel", "cpu", "gpu", "speedup"
        );
        println!(
            "{:<6} {:<10} {:>10} {:>10} {:>8}",
            "", "", "ns/elem", "ns/elem", ""
        );

        for log_n in [18usize, 20, 22, 24] {
            let n = 1usize << log_n;

            // ---- mle_eval: values (2^n) at a point of n coordinates ----
            let values: Vec<Fp> = (0..n).map(|_| Fp::new(rnd())).collect();
            let point: Vec<Ext> = (0..log_n).map(|_| Ext::from(Fp::new(rnd()))).collect();
        // ---- one-off: compare the per-index terms and the reduced value ----
        if log_n == 18 {
            if let Some(terms) = metal::mle_terms(&values, &point) {
                let want: Vec<Ext> = (0..n)
                    .map(|i| flashprove::field::eq(&point, i) * Ext::from(values[i]))
                    .collect();
                let bad = terms.iter().zip(&want).filter(|(a, b)| a != b).count();
                let sum: Ext = terms.iter().fold(Ext::ZERO, |acc, &t| acc + t);
                eprintln!(
                    "  terms: {bad}/{} differ; gpu sum {:?} vs cpu {:?}",
                    n,
                    <Ext as p3_field::BasedVectorSpace<Fp>>::as_basis_coefficients_slice(&sum)
                        .iter()
                        .map(|v| p3_field::PrimeField64::as_canonical_u64(v))
                        .collect::<Vec<_>>(),
                    <Ext as p3_field::BasedVectorSpace<Fp>>::as_basis_coefficients_slice(&backend::mle_eval_cpu(&values, &point))
                        .iter()
                        .map(|v| p3_field::PrimeField64::as_canonical_u64(v))
                        .collect::<Vec<_>>(),
                );
            } else {
                eprintln!("  terms: gpu unavailable");
            }
        }


            let t0 = Instant::now();
            let cpu = backend::mle_eval_cpu(&values, &point);
            let cpu_ns = t0.elapsed().as_nanos() as f64 / n as f64;

            let t0 = Instant::now();
            let gpu = metal::mle_eval(&values, &point);
            let gpu_ns = t0.elapsed().as_nanos() as f64 / n as f64;

            match gpu {
                Some(v) if v == cpu => println!(
                    "2^{:<4} {:<10} {:>10.3} {:>10.3} {:>7.1}x",
                    log_n,
                    "mle_eval",
                    cpu_ns,
                    gpu_ns,
                    cpu_ns / gpu_ns
                ),
                Some(_) => println!("2^{:<4} {:<10} MISMATCH", log_n, "mle_eval"),
                None => println!("2^{:<4} {:<10} (gpu unavailable)", log_n, "mle_eval"),
            }

            // ---- fold: one sumcheck round over 2^n extension elements ----
            let a: Vec<Ext> = (0..n).map(|_| Ext::from(Fp::new(rnd()))).collect();
            let r = Ext::from(Fp::new(rnd()));

            let mut cpu_a = a.clone();
            let t0 = Instant::now();
            backend::fold_cpu(&mut cpu_a, r);
            let cpu_ns = t0.elapsed().as_nanos() as f64 / n as f64;

            let t0 = Instant::now();
            let gpu = metal::fold(&a, r);
            let gpu_ns = t0.elapsed().as_nanos() as f64 / n as f64;

            match gpu {
                Some(v) if v == cpu_a => println!(
                    "2^{:<4} {:<10} {:>10.3} {:>10.3} {:>7.1}x",
                    log_n,
                    "fold",
                    cpu_ns,
                    gpu_ns,
                    cpu_ns / gpu_ns
                ),
                Some(_) => println!("2^{:<4} {:<10} MISMATCH", log_n, "fold"),
                None => println!("2^{:<4} {:<10} (gpu unavailable)", log_n, "fold"),
            }
        }

        println!(
            "\nA dispatch costs ~180us, so the GPU only wins above ~2^18 elements.\n\
             Wire it in with FLASHPROVE_GPU=1; without it every kernel stays on the CPU."
        );
    }
}
