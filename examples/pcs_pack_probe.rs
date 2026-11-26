//! Does the WHIR verifier's cost track the number of *tables* or the number of
//! committed *values*?
//!
//! Builds the same total data twice — once as `N` one-column tables, once as one
//! table with `N` columns — with the same number of opening claims, and times
//! `verify`. If per-table setup dominates, packing cells into columns is the lever.

use std::time::Instant;

use flashprove::field::{Ext, Fp};
use flashprove::pcs::transcript::new_challenger;
use flashprove::pcs::whir_stacked::{verify, Committer, Params};

fn column(m: usize, seed: u64) -> Vec<Fp> {
    let n = 1usize << m;
    (0..n)
        .map(|i| Fp::new((i as u64).wrapping_mul(seed).wrapping_add(seed)))
        .collect()
}

/// `tables` tables of arity `m`, `per_table` columns each, one opening per column.
fn run(tables: usize, per_table: usize, m: usize) -> (f64, usize) {
    let cols: Vec<Vec<Fp>> = (0..per_table).map(|c| column(m, c as u64 + 1)).collect();
    let mut k = Committer::new();
    for _ in 0..tables {
        k.add_table(m, cols.clone());
    }
    let mut prover = k.seal(Params::default());

    let point: Vec<Ext> = (0..m).map(|i| Ext::from(Fp::new(i as u64 + 7))).collect();
    for t in 0..tables {
        for c in 0..per_table {
            prover.claim(t, &[c], &point);
        }
    }
    let commitment = prover.commitment().clone();
    let claims = prover.claims().to_vec();
    let mut ch = new_challenger();
    let proof = prover.prove(&mut ch);
    let bytes = bincode::serialize(&proof).map(|b| b.len()).unwrap_or(0);

    let mut vch = new_challenger();
    let t0 = Instant::now();
    let ok = verify(&commitment, &claims, &proof, &mut vch);
    let dt = t0.elapsed().as_secs_f64();
    assert!(ok, "verification failed");
    println!(
        "  {tables:>3} tables x {per_table} cols x 2^{m}: verify {dt:6.3}s, proof {} KiB, {} claims",
        bytes / 1024,
        claims.len()
    );
    (dt, bytes)
}

fn main() {
    let m = 13;
    // 512 columns of 2^13 values in every layout; one opening per column, so the
    // claim count is identical and only the table layout changes.
    println!("layout A (one column per table):");
    let (a, _) = run(512, 1, m);
    println!("layout B (eight columns per table):");
    let (b, _) = run(64, 8, m);
    println!("layout C (64 columns per table):");
    let (c, _) = run(8, 64, m);
    println!(
        "\nsame 4,194,304 values and 512 openings: A {a:.3}s, B {b:.3}s, C {c:.3}s  =>  packing wins {:.1}x",
        a / c.max(1e-9)
    );
}
