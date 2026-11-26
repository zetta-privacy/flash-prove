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
                        .map(p3_field::PrimeField64::as_canonical_u64)
                        .collect::<Vec<_>>(),
                    <Ext as p3_field::BasedVectorSpace<Fp>>::as_basis_coefficients_slice(&backend::mle_eval_cpu(&values, &point))
                        .iter()
                        .map(p3_field::PrimeField64::as_canonical_u64)
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

        // ---- linear-claim sumcheck: same rounds and challenges on both sides ----
        {
            use flashprove::field::sumcheck::RoundPoly;
            use flashprove::pcs::transcript::{new_challenger, Transcript};

            for log_n in [18usize, 20, 22] {
                let n = 1usize << log_n;
                let hf: Vec<Ext> = (0..n).map(|_| Ext::from(Fp::new(rnd()))).collect();
                let vf: Vec<Ext> = (0..n).map(|_| Ext::from(Fp::new(rnd()))).collect();

                // CPU reference: the same loop `linear_claim::prove` runs.
                let mut tr = Transcript::new(&new_challenger());
                let t0 = Instant::now();
                let (mut hc, mut vc) = (hf.clone(), vf.clone());
                let mut cpu_rounds: Vec<RoundPoly> = Vec::new();
                let mut cpu_z: Vec<Ext> = Vec::new();
                for _ in 0..log_n {
                    let half = vc.len() / 2;
                    let (mut a, mut bcoef, mut c) = (Ext::ZERO, Ext::ZERO, Ext::ZERO);
                    for j in 0..half {
                        let h0 = hc[2 * j];
                        let dh = hc[2 * j + 1] - h0;
                        let v0 = vc[2 * j];
                        let dv = vc[2 * j + 1] - v0;
                        a += h0 * v0;
                        bcoef += h0 * dv + dh * v0;
                        c += dh * dv;
                    }
                    let g = RoundPoly([a, bcoef, c]);
                    for &coef in &g.0 {
                        tr.observe_ext(coef);
                    }
                    let r = tr.sample_ext();
                    cpu_rounds.push(g);
                    cpu_z.push(r);
                    for j in 0..half {
                        hc[j] = hc[2 * j] + r * (hc[2 * j + 1] - hc[2 * j]);
                        vc[j] = vc[2 * j] + r * (vc[2 * j + 1] - vc[2 * j]);
                    }
                    hc.truncate(half);
                    vc.truncate(half);
                }
                let cpu_s = t0.elapsed().as_secs_f64();

                // GPU: whole claim, operands resident.
                let mut tr2 = Transcript::new(&new_challenger());
                let t0 = Instant::now();
                let gpu = flashprove::backend::metal::linear_sumcheck(&hf, &vf, &mut tr2);
                let gpu_s = t0.elapsed().as_secs_f64();

                match gpu {
                    Some((gr, gz)) if gr == cpu_rounds && gz == cpu_z => println!(
                        "2^{:<4} {:<11} {:>9.2} {:>9.2} {:>7.1}x",
                        log_n,
                        "sumcheck",
                        cpu_s * 1e3,
                        gpu_s * 1e3,
                        cpu_s / gpu_s
                    ),
                    Some((gr, gz)) => {
                        let cb: Vec<u64> = <Ext as p3_field::BasedVectorSpace<Fp>>::as_basis_coefficients_slice(&cpu_rounds[0].0[0]).iter().map(p3_field::PrimeField64::as_canonical_u64).collect();
                        let gb: Vec<u64> = <Ext as p3_field::BasedVectorSpace<Fp>>::as_basis_coefficients_slice(&gr[0].0[0]).iter().map(p3_field::PrimeField64::as_canonical_u64).collect();
                        println!(
                            "2^{:<4} {:<11} MISMATCH rounds_eq={} z_eq={} cpu_A={:?} gpu_A={:?}",
                            log_n,
                            "sumcheck",
                            gr == cpu_rounds,
                            gz == cpu_z,
                            cb,
                            gb
                        );
                    }
                    None => println!("2^{:<4} {:<11} (gpu unavailable)", log_n, "sumcheck"),
                }
            }
        }

        // ---- BLAKE3 leaves (2 field elements -> 1 digest) ----
        {
            use flashprove::pcs::blake3_mmcs::Blake3Hasher;
            use flashprove::pcs::gpu_mmcs::GpuBlake3Hasher;
            use p3_symmetric::CryptographicHasher;

            for log_n in [18usize, 20, 22] {
                let leaves = 1usize << log_n;
                let input: Vec<Fp> = (0..2 * leaves).map(|_| Fp::new(rnd())).collect();

                let mut cpu = vec![[Fp::default(); 2]; leaves];
                let t0 = Instant::now();
                Blake3Hasher.hash_many(&input, &mut cpu);
                let cpu_s = t0.elapsed().as_secs_f64();

                let mut gpu = vec![[Fp::default(); 2]; leaves];
                let t0 = Instant::now();
                GpuBlake3Hasher.hash_many(&input, &mut gpu);
                let gpu_s = t0.elapsed().as_secs_f64();

                println!(
                    "2^{:<4} {:<11} {:>9.3} {:>9.3} {:>7.1}x{}",
                    log_n,
                    "blake3",
                    cpu_s * 1e3,
                    gpu_s * 1e3,
                    cpu_s / gpu_s,
                    if cpu == gpu { "" } else { "   MISMATCH" }
                );
            }
        }

        // ---- raw bandwidth: how fast can this GPU copy 33 MB at all? ----
        {
            let n = 1usize << 22;
            let x: Vec<u64> = (0..n as u64).map(|i| i * 3 + 1).collect();
            let t0 = Instant::now();
            let y = flashprove::backend::metal::copy_probe(&x);
            let s = t0.elapsed().as_secs_f64();
            let ok = y.as_ref().map(|y| y == &x).unwrap_or(false);
            println!(
                "2^22   copy_u64   {:.3} ms round trip   {:.1} GB/s (read+write)   identical={ok}",
                s * 1e3,
                (2.0 * (n * 8) as f64) / s / 1e9
            );
        }

        // ---- the seal's FFT: MetalDft vs Radix2DitParallel, through the trait ----
        {
            use flashprove::pcs::metal_dft::MetalDft;
            use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
            use p3_matrix::dense::RowMajorMatrix;
            use p3_matrix::Matrix;

            for log_h in [18usize, 20, 22] {
                let h = 1usize << log_h;
                let values: Vec<Fp> = (0..h).map(|_| Fp::new(rnd())).collect();
                let mat = RowMajorMatrix::new(values, 1);

                let t0 = Instant::now();
                let cpu = Radix2DitParallel::<Fp>::default()
                    .dft_batch(mat.clone())
                    .to_row_major_matrix()
                    .values;
                let cpu_s = t0.elapsed().as_secs_f64();

                let t0 = Instant::now();
                let gpu = MetalDft
                    .dft_batch(mat.clone())
                    .to_row_major_matrix()
                    .values;
                let gpu_s = t0.elapsed().as_secs_f64();

                println!(
                    "2^{:<4} {:<10} {:>10.3} {:>10.3} {:>7.1}x{}",
                    log_h,
                    "dft_batch",
                    cpu_s * 1e3,
                    gpu_s * 1e3,
                    cpu_s / gpu_s,
                    if gpu == cpu { "" } else { "   MISMATCH" }
                );
            }
        }

        // ---- contraction: p[k] = Σ_n eq[n]·w[n·2^lm + k] ----
        for (ln, lm) in [(11usize, 11usize), (6, 11)] {
            let rows = 1usize << ln;
            let cols = 1usize << lm;
            let wv: Vec<Fp> = (0..rows * cols).map(|_| Fp::new(rnd())).collect();
            let point: Vec<Ext> = (0..ln).map(|_| Ext::from(Fp::new(rnd()))).collect();
            let eq: Vec<Ext> = (0..rows).map(|n| flashprove::field::eq(&point, n)).collect();

            let t0 = Instant::now();
            let mut cpu = vec![Ext::ZERO; cols];
            for (n, rn) in eq.iter().enumerate() {
                let base = n << lm;
                for k in 0..cols {
                    cpu[k] += *rn * Ext::from(wv[base + k]);
                }
            }
            let cpu_ns = t0.elapsed().as_nanos() as f64;

            let t0 = Instant::now();
            let gpu = metal::contract(&wv, &eq, lm);
            let gpu_ns = t0.elapsed().as_nanos() as f64;

            match gpu {
                Some(v) if v == cpu => println!(
                    "{:<10} {:<10} {:>10.3} {:>10.3} {:>7.1}x",
                    format!("2^{ln}x2^{lm}"),
                    "contract",
                    cpu_ns / cols as f64,
                    gpu_ns / cols as f64,
                    cpu_ns / gpu_ns
                ),
                Some(_) => println!("{:<10} {:<10} MISMATCH", format!("2^{ln}x2^{lm}"), "contract"),
                None => println!("{:<10} {:<10} (gpu unavailable)", format!("2^{ln}x2^{lm}"), "contract"),
            }
        }

        println!(
            "\nA dispatch costs ~180us, so the GPU only wins above ~2^18 elements.\n\
             Wire it in with FLASHPROVE_GPU=1; without it every kernel stays on the CPU."
        );
    }
}
