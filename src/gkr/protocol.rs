//! Committed GKR matmul layer: verify `C = A·B` via the GKR/sumcheck reduction,
//! with every MLE evaluation done through the FRI commitment (O(log n) verifier per
//! layer) instead of the O(n) linear-claim/IPA opening collection.
//!
//! The reduction: sample `(r, s)` (row/col points), claim `C̃(r,s) = Σ_k Ã(r,k)·B̃(k,s)`,
//! and run a sumcheck over the `lm` contraction variables. The surviving claim
//! `Ã(r,k*)·B̃(k*,s)` is discharged by three committed MLE evaluations (A, B, C).
//!
//! The round polynomial is the product of the two contraction MLEs evaluated at a
//! partial point (degree 2 per variable), *not* the MLE of the elementwise product
//! table — those differ, and only the former is what GKR needs.
//!
//! Matrices are **row-major** (`flat[i·cols + j] = X[i][j]`), matching the committed
//! cells. The MLE point is `col_bits ‖ row_bits` (col bits in the low positions),
//! since `idx = i·2^lp + j`.

use crate::field::arith::bool_point;
use crate::field::{Ext, Fp};
use crate::pcs::fri::{self, Challenger, Prover};
use crate::lookup::logup::TableCommitment;
use crate::lookup::requant::{prove_requant_given, verify_requant, RequantProof};
use crate::field::sumcheck::{mle_eval, RoundPoly, RoundPoly3};
use crate::pcs::zk::{prove_mle_committed, verify_mle_committed, InnerProductProof, Openings};
use p3_field::{Field, PrimeCharacteristicRing};

fn concat(a: &[Ext], b: &[Ext]) -> Vec<Ext> {
    let mut v = a.to_vec();
    v.extend_from_slice(b);
    v
}

/// MLE of a row-major matrix at `(row_point, col_point)`.
fn matrix_mle(flat: &[Ext], row_point: &[Ext], col_point: &[Ext]) -> Ext {
    mle_eval(flat, &concat(col_point, row_point))
}

fn interpolate_deg2(y0: Ext, y1: Ext, y2: Ext) -> RoundPoly {
    let two = Ext::from(Fp::new(2));
    let bpc = y1 - y0;
    let two_b_four_c = y2 - y0;
    let two_c = two_b_four_c - two * bpc;
    let c = two_c * two.inverse();
    let b = bpc - c;
    RoundPoly([y0, b, c])
}

/// The GKR round polynomial for `Σ_k p(k)·q(k)`: at variable `i` it is
/// `g_i(t) = Σ_{k_{>i}} p̃(r_<i, t, k_>i) · q̃(r_<i, t, k_>i)`.
fn round_poly_product(p: &[Ext], q: &[Ext], lm: usize, i: usize, r_prefix: &[Ext]) -> RoundPoly {
    let suffix = lm - i - 1;
    let mut y = [Ext::ZERO; 3];
    let samples = [Ext::ZERO, Ext::ONE, Ext::from(Fp::new(2))];
    for (idx, t) in samples.iter().enumerate() {
        let mut acc = Ext::ZERO;
        for sb in 0..(1 << suffix) {
            let mut point = vec![Ext::ZERO; lm];
            point[..i].copy_from_slice(r_prefix);
            point[i] = *t;
            for b in 0..suffix {
                point[i + 1 + b] = if (sb >> b) & 1 == 1 {
                    Ext::ONE
                } else {
                    Ext::ZERO
                };
            }
            acc += mle_eval(p, &point) * mle_eval(q, &point);
        }
        y[idx] = acc;
    }
    interpolate_deg2(y[0], y[1], y[2])
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GkrMatmulProof {
    pub ln: usize,
    pub lm: usize,
    pub lp: usize,
    /// Sumcheck round polynomials (one per contraction variable).
    pub rounds: Vec<RoundPoly>,
    pub a_val: Ext,
    pub b_val: Ext,
    pub c_val: Ext,
    pub mle_a: [InnerProductProof; 3],
    pub mle_b: [InnerProductProof; 3],
    pub mle_c: [InnerProductProof; 3],
}

/// Prove `C = A·B` for already-committed `A` (`2^ln × 2^lm`), `B` (`2^lm × 2^lp`),
/// `C` (`2^ln × 2^lp`), all row-major and power-of-two dimensions.
#[allow(clippy::too_many_arguments)]
pub fn prove_gkr_matmul(
    ctx: &mut Prover,
    com_a: &fri::Commitment,
    pd_a: &fri::ProverData,
    dom_a: fri::Domain,
    com_b: &fri::Commitment,
    pd_b: &fri::ProverData,
    dom_b: fri::Domain,
    com_c: &fri::Commitment,
    pd_c: &fri::ProverData,
    dom_c: fri::Domain,
    a: &[Fp],
    b: &[Fp],
    c: &[Fp],
    ln: usize,
    lm: usize,
    lp: usize,
) -> GkrMatmulProof {
    assert_eq!(a.len(), 1 << (ln + lm));
    assert_eq!(b.len(), 1 << (lm + lp));
    assert_eq!(c.len(), 1 << (ln + lp));

    let a_ext: Vec<Ext> = a.iter().map(|&x| Ext::from(x)).collect();
    let b_ext: Vec<Ext> = b.iter().map(|&x| Ext::from(x)).collect();
    let c_ext: Vec<Ext> = c.iter().map(|&x| Ext::from(x)).collect();

    let mut tr = fri::Transcript::new(&ctx.challenger);
    tr.observe(com_a);
    tr.observe(com_b);
    tr.observe(com_c);
    let r: Vec<Ext> = (0..ln).map(|_| tr.sample_ext()).collect();
    let s: Vec<Ext> = (0..lp).map(|_| tr.sample_ext()).collect();

    let c_val = matrix_mle(&c_ext, &r, &s);

    // Contraction MLEs: p[k] = Ã(r, k), q[k] = B̃(k, s), over the Boolean cube of k.
    let mut p = vec![Ext::ZERO; 1 << lm];
    let mut q = vec![Ext::ZERO; 1 << lm];
    for k in 0..(1 << lm) {
        let kb = bool_point(k, lm);
        p[k] = matrix_mle(&a_ext, &r, &kb);
        q[k] = matrix_mle(&b_ext, &kb, &s);
    }
    // Sumcheck over the lm contraction variables. The initial claim is the claimed
    // `C̃(r,s)` value (the prover may be proving a *wrong* `c`; the verifier rejects
    // that by the round checks + the surviving product).
    let mut _claim = c_val;
    let mut kstar = Vec::with_capacity(lm);
    let mut rounds = Vec::with_capacity(lm);
    for i in 0..lm {
        let round = round_poly_product(&p, &q, lm, i, &kstar);
        for coeff in &round.0 {
            tr.observe_ext(*coeff);
        }
        let ri = tr.sample_ext();
        rounds.push(round);
        kstar.push(ri);
        _claim = round.eval(ri);
    }

    // Surviving claim: Ã(r, k*) · B̃(k*, s).
    let a_val = mle_eval(&p, &kstar);
    let b_val = mle_eval(&q, &kstar);

    let mle_a = prove_mle_committed(ctx, com_a, pd_a, dom_a, a, &concat(&kstar, &r));
    let mle_b = prove_mle_committed(ctx, com_b, pd_b, dom_b, b, &concat(&s, &kstar));
    let mle_c = prove_mle_committed(ctx, com_c, pd_c, dom_c, c, &concat(&s, &r));

    GkrMatmulProof {
        ln,
        lm,
        lp,
        rounds,
        a_val,
        b_val,
        c_val,
        mle_a,
        mle_b,
        mle_c,
    }
}

/// Verify `C = A·B` from the committed GKR proof (commitments only, no clear data).
pub fn verify_gkr_matmul(
    challenger: &Challenger,
    com_a: &fri::Commitment,
    dom_a: fri::Domain,
    com_b: &fri::Commitment,
    dom_b: fri::Domain,
    com_c: &fri::Commitment,
    dom_c: fri::Domain,
    proof: &GkrMatmulProof,
    values: &[Ext],
    acc: &mut Openings,
) -> bool {
    let ln = proof.ln;
    let lm = proof.lm;
    let lp = proof.lp;

    let mut tr = fri::Transcript::new(challenger);
    tr.observe(com_a);
    tr.observe(com_b);
    tr.observe(com_c);
    let r: Vec<Ext> = (0..ln).map(|_| tr.sample_ext()).collect();
    let s: Vec<Ext> = (0..lp).map(|_| tr.sample_ext()).collect();

    // Replay the sumcheck.
    let mut claim = proof.c_val;
    let mut kstar = Vec::with_capacity(lm);
    for i in 0..lm {
        let round = proof.rounds[i];
        if round.eval(Ext::ZERO) + round.eval(Ext::ONE) != claim {
            return false;
        }
        for coeff in &round.0 {
            tr.observe_ext(*coeff);
        }
        let ri = tr.sample_ext();
        kstar.push(ri);
        claim = round.eval(ri);
    }
    if claim != proof.a_val * proof.b_val {
        return false;
    }

    // Three committed MLE evaluations.
    verify_mle_committed(
        challenger,
        com_a,
        dom_a,
        1 << (ln + lm),
        &concat(&kstar, &r),
        proof.a_val,
        &proof.mle_a,
        values,
        acc,
    ) && verify_mle_committed(
        challenger,
        com_b,
        dom_b,
        1 << (lm + lp),
        &concat(&s, &kstar),
        proof.b_val,
        &proof.mle_b,
        values,
        acc,
    ) && verify_mle_committed(
        challenger,
        com_c,
        dom_c,
        1 << (ln + lp),
        &concat(&s, &r),
        proof.c_val,
        &proof.mle_c,
        values,
        acc,
    )
}

/// Elementwise matvec `out = W·x` (field arithmetic, correct mod p).
fn compute_matvec(w: &[Fp], x: &[Fp], rows: usize, cols: usize) -> Vec<Fp> {
    let mut out = vec![Fp::ZERO; rows];
    for i in 0..rows {
        let mut acc = Fp::ZERO;
        for j in 0..cols {
            acc += w[i * cols + j] * x[j];
        }
        out[i] = acc;
    }
    out
}

/// Committed GKR for a two-layer linear circuit `Y = W2 · (W1 · X)`. The intermediate
/// `X1 = W1·X` is *not* committed — its MLE is virtualized through the second
/// sumcheck, so the verifier sees O(log n) scalars per layer instead of O(hidden)
/// activations. This is the layer chaining that gives the O(depth·log n) verifier.
pub struct Gkr2LayerProof {
    pub d0: usize,
    pub d1: usize,
    pub d2: usize,
    pub rounds1: Vec<RoundPoly>,
    pub rounds2: Vec<RoundPoly>,
    pub w2_val: Ext,
    pub x1_val: Ext,
    pub w1_val: Ext,
    pub x_val: Ext,
    pub y_val: Ext,
    pub mle_w2: [InnerProductProof; 3],
    pub mle_w1: [InnerProductProof; 3],
    pub mle_x: [InnerProductProof; 3],
    pub mle_y: [InnerProductProof; 3],
}

#[allow(clippy::too_many_arguments)]
pub fn prove_gkr_2layer(
    ctx: &mut Prover,
    com_w2: &fri::Commitment,
    pd_w2: &fri::ProverData,
    dom_w2: fri::Domain,
    com_w1: &fri::Commitment,
    pd_w1: &fri::ProverData,
    dom_w1: fri::Domain,
    com_x: &fri::Commitment,
    pd_x: &fri::ProverData,
    dom_x: fri::Domain,
    com_y: &fri::Commitment,
    pd_y: &fri::ProverData,
    dom_y: fri::Domain,
    w2: &[Fp],
    w1: &[Fp],
    x: &[Fp],
    y: &[Fp],
    d0: usize,
    d1: usize,
    d2: usize,
) -> Gkr2LayerProof {
    assert_eq!(w2.len(), 1 << (d2 + d1));
    assert_eq!(w1.len(), 1 << (d1 + d0));
    assert_eq!(x.len(), 1 << d0);
    assert_eq!(y.len(), 1 << d2);

    let w2_ext: Vec<Ext> = w2.iter().map(|&v| Ext::from(v)).collect();
    let w1_ext: Vec<Ext> = w1.iter().map(|&v| Ext::from(v)).collect();
    let x_ext: Vec<Ext> = x.iter().map(|&v| Ext::from(v)).collect();
    let y_ext: Vec<Ext> = y.iter().map(|&v| Ext::from(v)).collect();
    let x1 = compute_matvec(w1, x, 1 << d1, 1 << d0);
    let x1_ext: Vec<Ext> = x1.iter().map(|&v| Ext::from(v)).collect();

    let mut tr = fri::Transcript::new(&ctx.challenger);
    tr.observe(com_w2);
    tr.observe(com_w1);
    tr.observe(com_x);
    tr.observe(com_y);
    let r: Vec<Ext> = (0..d2).map(|_| tr.sample_ext()).collect();

    let y_val = matrix_mle(&y_ext, &r, &[]);

    // Sumcheck 1: Ỹ(r) = Σ_k W̃2(r,k)·X̃1(k).
    let mut p1 = vec![Ext::ZERO; 1 << d1];
    let mut q1 = vec![Ext::ZERO; 1 << d1];
    for k in 0..(1 << d1) {
        let kb = bool_point(k, d1);
        p1[k] = matrix_mle(&w2_ext, &r, &kb);
        q1[k] = matrix_mle(&x1_ext, &kb, &[]);
    }
    let mut _claim = y_val;
    let mut kstar = Vec::with_capacity(d1);
    let mut rounds1 = Vec::with_capacity(d1);
    for i in 0..d1 {
        let round = round_poly_product(&p1, &q1, d1, i, &kstar);
        for coeff in &round.0 {
            tr.observe_ext(*coeff);
        }
        let ri = tr.sample_ext();
        rounds1.push(round);
        kstar.push(ri);
        _claim = round.eval(ri);
    }
    let w2_val = mle_eval(&p1, &kstar);
    let x1_val = mle_eval(&q1, &kstar);

    // Sumcheck 2: X̃1(k*) = Σ_j W̃1(k*,j)·X̃(j).
    let mut p2 = vec![Ext::ZERO; 1 << d0];
    let mut q2 = vec![Ext::ZERO; 1 << d0];
    for j in 0..(1 << d0) {
        let jb = bool_point(j, d0);
        p2[j] = matrix_mle(&w1_ext, &kstar, &jb);
        q2[j] = matrix_mle(&x_ext, &jb, &[]);
    }
    let mut _claim2 = x1_val;
    let mut jstar = Vec::with_capacity(d0);
    let mut rounds2 = Vec::with_capacity(d0);
    for i in 0..d0 {
        let round = round_poly_product(&p2, &q2, d0, i, &jstar);
        for coeff in &round.0 {
            tr.observe_ext(*coeff);
        }
        let ri = tr.sample_ext();
        rounds2.push(round);
        jstar.push(ri);
        _claim2 = round.eval(ri);
    }
    let w1_val = mle_eval(&p2, &jstar);
    let x_val = mle_eval(&q2, &jstar);

    let mle_w2 = prove_mle_committed(ctx, com_w2, pd_w2, dom_w2, w2, &concat(&kstar, &r));
    let mle_w1 = prove_mle_committed(ctx, com_w1, pd_w1, dom_w1, w1, &concat(&jstar, &kstar));
    let mle_x = prove_mle_committed(ctx, com_x, pd_x, dom_x, x, &jstar);
    let mle_y = prove_mle_committed(ctx, com_y, pd_y, dom_y, y, &r);

    Gkr2LayerProof {
        d0,
        d1,
        d2,
        rounds1,
        rounds2,
        w2_val,
        x1_val,
        w1_val,
        x_val,
        y_val,
        mle_w2,
        mle_w1,
        mle_x,
        mle_y,
    }
}

pub fn verify_gkr_2layer(
    challenger: &Challenger,
    com_w2: &fri::Commitment,
    dom_w2: fri::Domain,
    com_w1: &fri::Commitment,
    dom_w1: fri::Domain,
    com_x: &fri::Commitment,
    dom_x: fri::Domain,
    com_y: &fri::Commitment,
    dom_y: fri::Domain,
    proof: &Gkr2LayerProof,
    values: &[Ext],
    acc: &mut Openings,
) -> bool {
    let d0 = proof.d0;
    let d1 = proof.d1;
    let d2 = proof.d2;

    let mut tr = fri::Transcript::new(challenger);
    tr.observe(com_w2);
    tr.observe(com_w1);
    tr.observe(com_x);
    tr.observe(com_y);
    let r: Vec<Ext> = (0..d2).map(|_| tr.sample_ext()).collect();

    let mut claim = proof.y_val;
    let mut kstar = Vec::with_capacity(d1);
    for i in 0..d1 {
        let round = proof.rounds1[i];
        if round.eval(Ext::ZERO) + round.eval(Ext::ONE) != claim {
            return false;
        }
        for coeff in &round.0 {
            tr.observe_ext(*coeff);
        }
        let ri = tr.sample_ext();
        kstar.push(ri);
        claim = round.eval(ri);
    }
    if claim != proof.w2_val * proof.x1_val {
        return false;
    }

    let mut claim2 = proof.x1_val;
    let mut jstar = Vec::with_capacity(d0);
    for i in 0..d0 {
        let round = proof.rounds2[i];
        if round.eval(Ext::ZERO) + round.eval(Ext::ONE) != claim2 {
            return false;
        }
        for coeff in &round.0 {
            tr.observe_ext(*coeff);
        }
        let ri = tr.sample_ext();
        jstar.push(ri);
        claim2 = round.eval(ri);
    }
    if claim2 != proof.w1_val * proof.x_val {
        return false;
    }

    verify_mle_committed(
        challenger,
        com_w2,
        dom_w2,
        1 << (d2 + d1),
        &concat(&kstar, &r),
        proof.w2_val,
        &proof.mle_w2,
        values,
        acc,
    ) && verify_mle_committed(
        challenger,
        com_w1,
        dom_w1,
        1 << (d1 + d0),
        &concat(&jstar, &kstar),
        proof.w1_val,
        &proof.mle_w1,
        values,
        acc,
    ) && verify_mle_committed(
        challenger,
        com_x,
        dom_x,
        1 << d0,
        &jstar,
        proof.x_val,
        &proof.mle_x,
        values,
        acc,
    ) && verify_mle_committed(
        challenger,
        com_y,
        dom_y,
        1 << d2,
        &r,
        proof.y_val,
        &proof.mle_y,
        values,
        acc,
    )
}

/// Committed GKR for an N-layer linear circuit `Z^{(l+1)} = W^{(l)}·Z^{(l)}`, with
/// `Z^{(0)} = X` and `Z^{(N)} = Y`. Intermediate activations are never committed —
/// each is virtualized through the next sumcheck, so the verifier sees O(log n)
/// scalars per layer (O(N·log n) total, independent of the hidden widths).
pub struct GkrProof {
    /// Log-widths `d_0 … d_N`.
    pub dims: Vec<usize>,
    /// Sumcheck round polynomials, per layer (index 0..N-1).
    pub rounds: Vec<Vec<RoundPoly>>,
    /// `W̃^{(l)}(r_{l+1}, r_l)` for l = 0..N-1.
    pub w_vals: Vec<Ext>,
    /// `Z̃^{(l)}(r_l)` for l = 0..N (index N is `Ỹ(r_N)`).
    pub z_vals: Vec<Ext>,
    pub mle_w: Vec<[InnerProductProof; 3]>,
    pub mle_x: [InnerProductProof; 3],
    pub mle_y: [InnerProductProof; 3],
}

#[allow(clippy::too_many_arguments)]
pub fn prove_gkr(
    ctx: &mut Prover,
    weights: &[&[Fp]],
    coms: &[&fri::Commitment],
    pds: &[&fri::ProverData],
    doms: &[fri::Domain],
    com_x: &fri::Commitment,
    pd_x: &fri::ProverData,
    dom_x: fri::Domain,
    com_y: &fri::Commitment,
    pd_y: &fri::ProverData,
    dom_y: fri::Domain,
    x: &[Fp],
    y: &[Fp],
    dims: &[usize],
) -> GkrProof {
    let n_layers = dims.len() - 1;
    assert_eq!(weights.len(), n_layers);
    assert_eq!(coms.len(), n_layers);
    assert_eq!(pds.len(), n_layers);
    assert_eq!(doms.len(), n_layers);
    for l in 0..n_layers {
        assert_eq!(weights[l].len(), 1 << (dims[l] + dims[l + 1]));
    }
    assert_eq!(x.len(), 1 << dims[0]);
    assert_eq!(y.len(), 1 << dims[n_layers]);

    // Intermediate activations in the clear (prover only).
    let mut zs_fp: Vec<Vec<Fp>> = vec![x.to_vec()];
    for l in 0..n_layers {
        let z_next = compute_matvec(weights[l], &zs_fp[l], 1 << dims[l + 1], 1 << dims[l]);
        zs_fp.push(z_next);
    }
    let zs: Vec<Vec<Ext>> = zs_fp
        .iter()
        .map(|z| z.iter().map(|&v| Ext::from(v)).collect())
        .collect();
    let weights_ext: Vec<Vec<Ext>> = weights
        .iter()
        .map(|w| w.iter().map(|&v| Ext::from(v)).collect())
        .collect();
    let y_ext: Vec<Ext> = y.iter().map(|&v| Ext::from(v)).collect();

    let mut tr = fri::Transcript::new(&ctx.challenger);
    for c in coms {
        tr.observe(c);
    }
    tr.observe(com_x);
    tr.observe(com_y);

    // r[l] = sumcheck challenge of layer l (r[N] is the sampled output point).
    let mut r: Vec<Vec<Ext>> = vec![Vec::new(); n_layers + 1];
    r[n_layers] = (0..dims[n_layers]).map(|_| tr.sample_ext()).collect();

    let mut w_vals = vec![Ext::ZERO; n_layers];
    let mut z_vals = vec![Ext::ZERO; n_layers + 1];
    z_vals[n_layers] = mle_eval(&y_ext, &r[n_layers]);
    let mut rounds_all: Vec<Vec<RoundPoly>> = Vec::with_capacity(n_layers);
    let mut mle_w: Vec<[InnerProductProof; 3]> = Vec::with_capacity(n_layers);

    // Backward through the layers.
    for l in (0..n_layers).rev() {
        let dl = dims[l];
        // Contraction MLEs: p[k] = W̃^{(l)}(r_{l+1}, k), q[k] = Z̃^{(l)}(k).
        let mut p = vec![Ext::ZERO; 1 << dl];
        let mut q = vec![Ext::ZERO; 1 << dl];
        for k in 0..(1 << dl) {
            let kb = bool_point(k, dl);
            p[k] = matrix_mle(&weights_ext[l], &r[l + 1], &kb);
            q[k] = matrix_mle(&zs[l], &kb, &[]);
        }
        let mut _claim = z_vals[l + 1];
        let mut rl = Vec::with_capacity(dl);
        let mut rounds = Vec::with_capacity(dl);
        for i in 0..dl {
            let round = round_poly_product(&p, &q, dl, i, &rl);
            for coeff in &round.0 {
                tr.observe_ext(*coeff);
            }
            let ri = tr.sample_ext();
            rounds.push(round);
            rl.push(ri);
            _claim = round.eval(ri);
        }
        w_vals[l] = mle_eval(&p, &rl);
        z_vals[l] = mle_eval(&q, &rl);
        r[l] = rl;
        rounds_all.push(rounds);
        mle_w.push(prove_mle_committed(
            ctx,
            coms[l],
            pds[l],
            doms[l],
            weights[l],
            &concat(&r[l], &r[l + 1]),
        ));
    }
    // Reverse to layer order (0..N-1).
    rounds_all.reverse();
    mle_w.reverse();

    let mle_x = prove_mle_committed(ctx, com_x, pd_x, dom_x, x, &r[0]);
    let mle_y = prove_mle_committed(ctx, com_y, pd_y, dom_y, y, &r[n_layers]);

    GkrProof {
        dims: dims.to_vec(),
        rounds: rounds_all,
        w_vals,
        z_vals,
        mle_w,
        mle_x,
        mle_y,
    }
}

pub fn verify_gkr(
    challenger: &Challenger,
    coms: &[&fri::Commitment],
    doms: &[fri::Domain],
    com_x: &fri::Commitment,
    dom_x: fri::Domain,
    com_y: &fri::Commitment,
    dom_y: fri::Domain,
    proof: &GkrProof,
    values: &[Ext],
    acc: &mut Openings,
) -> bool {
    let n_layers = proof.dims.len() - 1;
    let mut tr = fri::Transcript::new(challenger);
    for c in coms {
        tr.observe(c);
    }
    tr.observe(com_x);
    tr.observe(com_y);

    let mut r: Vec<Vec<Ext>> = vec![Vec::new(); n_layers + 1];
    r[n_layers] = (0..proof.dims[n_layers]).map(|_| tr.sample_ext()).collect();

    let mut claim = proof.z_vals[n_layers];
    for l in (0..n_layers).rev() {
        let dl = proof.dims[l];
        let rounds = &proof.rounds[l];
        let mut rl = Vec::with_capacity(dl);
        for i in 0..dl {
            let round = rounds[i];
            if round.eval(Ext::ZERO) + round.eval(Ext::ONE) != claim {
                return false;
            }
            for coeff in &round.0 {
                tr.observe_ext(*coeff);
            }
            let ri = tr.sample_ext();
            rl.push(ri);
            claim = round.eval(ri);
        }
        if claim != proof.w_vals[l] * proof.z_vals[l] {
            return false;
        }
        r[l] = rl;
        claim = proof.z_vals[l];
    }

    for l in (0..n_layers).rev() {
        if !verify_mle_committed(
            challenger,
            coms[l],
            doms[l],
            1 << (proof.dims[l] + proof.dims[l + 1]),
            &concat(&r[l], &r[l + 1]),
            proof.w_vals[l],
            &proof.mle_w[l],
            values,
            acc,
        ) {
            return false;
        }
    }
    verify_mle_committed(
        challenger,
        com_x,
        dom_x,
        1 << proof.dims[0],
        &r[0],
        proof.z_vals[0],
        &proof.mle_x,
        values,
        acc,
    ) && verify_mle_committed(
        challenger,
        com_y,
        dom_y,
        1 << proof.dims[n_layers],
        &r[n_layers],
        proof.z_vals[n_layers],
        &proof.mle_y,
        values,
        acc,
    )
}

/// A pointwise `add` op (`out = a + b`) checked via three committed MLE
/// evaluations at one random point.
pub struct GkrAddProof {
    pub a_val: Ext,
    pub b_val: Ext,
    pub out_val: Ext,
    pub mle_a: [InnerProductProof; 3],
    pub mle_b: [InnerProductProof; 3],
    pub mle_out: [InnerProductProof; 3],
}

#[allow(clippy::too_many_arguments)]
pub fn prove_gkr_add(
    ctx: &mut Prover,
    com_a: &fri::Commitment,
    pd_a: &fri::ProverData,
    dom_a: fri::Domain,
    com_b: &fri::Commitment,
    pd_b: &fri::ProverData,
    dom_b: fri::Domain,
    com_out: &fri::Commitment,
    pd_out: &fri::ProverData,
    dom_out: fri::Domain,
    a: &[Fp],
    b: &[Fp],
    out: &[Fp],
    d: usize,
) -> GkrAddProof {
    let a_ext: Vec<Ext> = a.iter().map(|&v| Ext::from(v)).collect();
    let b_ext: Vec<Ext> = b.iter().map(|&v| Ext::from(v)).collect();
    let out_ext: Vec<Ext> = out.iter().map(|&v| Ext::from(v)).collect();
    let mut tr = fri::Transcript::new(&ctx.challenger);
    tr.observe(com_a);
    tr.observe(com_b);
    tr.observe(com_out);
    let r: Vec<Ext> = (0..d).map(|_| tr.sample_ext()).collect();
    let a_val = mle_eval(&a_ext, &r);
    let b_val = mle_eval(&b_ext, &r);
    let out_val = mle_eval(&out_ext, &r);
    let mle_a = prove_mle_committed(ctx, com_a, pd_a, dom_a, a, &r);
    let mle_b = prove_mle_committed(ctx, com_b, pd_b, dom_b, b, &r);
    let mle_out = prove_mle_committed(ctx, com_out, pd_out, dom_out, out, &r);
    GkrAddProof {
        a_val,
        b_val,
        out_val,
        mle_a,
        mle_b,
        mle_out,
    }
}

pub fn verify_gkr_add(
    challenger: &Challenger,
    com_a: &fri::Commitment,
    dom_a: fri::Domain,
    com_b: &fri::Commitment,
    dom_b: fri::Domain,
    com_out: &fri::Commitment,
    dom_out: fri::Domain,
    d: usize,
    proof: &GkrAddProof,
    values: &[Ext],
    acc: &mut Openings,
) -> bool {
    let mut tr = fri::Transcript::new(challenger);
    tr.observe(com_a);
    tr.observe(com_b);
    tr.observe(com_out);
    let r: Vec<Ext> = (0..d).map(|_| tr.sample_ext()).collect();
    if proof.out_val != proof.a_val + proof.b_val {
        return false;
    }
    verify_mle_committed(
        challenger,
        com_a,
        dom_a,
        1 << d,
        &r,
        proof.a_val,
        &proof.mle_a,
        values,
        acc,
    ) && verify_mle_committed(
        challenger,
        com_b,
        dom_b,
        1 << d,
        &r,
        proof.b_val,
        &proof.mle_b,
        values,
        acc,
    ) && verify_mle_committed(
        challenger,
        com_out,
        dom_out,
        1 << d,
        &r,
        proof.out_val,
        &proof.mle_out,
        values,
        acc,
    )
}

fn interpolate_deg3(y0: Ext, y1: Ext, y2: Ext, y3: Ext) -> RoundPoly3 {
    let two = Ext::from(Fp::new(2));
    let three = Ext::from(Fp::new(3));
    let four = Ext::from(Fp::new(4));
    let six = Ext::from(Fp::new(6));
    let a = y0;
    let d1 = y1 - y0;
    let d2 = y2 - y0;
    let d3 = y3 - y0;
    let a_const = (d2 - two * d1) * two.inverse();
    let b_const = (d3 - three * d1) * six.inverse();
    let d = b_const - a_const;
    let c = four * a_const - three * b_const;
    let b = d1 - c - d;
    RoundPoly3([a, b, c, d])
}

/// `eq(r, point)` for an *arbitrary* point (not just Boolean).
fn eq_partial(r: &[Ext], point: &[Ext]) -> Ext {
    let mut acc = Ext::ONE;
    for k in 0..r.len() {
        let p = point[k];
        acc = acc * (r[k] * p + (Ext::ONE - r[k]) * (Ext::ONE - p));
    }
    acc
}

/// Degree-3 GKR round polynomial for a pointwise product `Σ_i eq(r,i)·a[i]·b[i]`:
/// `g_j(t) = Σ_{i_>j} eq(r, r_<j, t, i_>)·ã(r_<j, t, i_>)·b̃(r_<j, t, i_>)`.
fn round_poly3_mul(
    a: &[Ext],
    b: &[Ext],
    r: &[Ext],
    n: usize,
    j: usize,
    r_prefix: &[Ext],
) -> RoundPoly3 {
    let suffix = n - j - 1;
    let samples = [
        Ext::ZERO,
        Ext::ONE,
        Ext::from(Fp::new(2)),
        Ext::from(Fp::new(3)),
    ];
    let mut y = [Ext::ZERO; 4];
    for (idx, t) in samples.iter().enumerate() {
        let mut acc = Ext::ZERO;
        for sb in 0..(1 << suffix) {
            let mut point = vec![Ext::ZERO; n];
            point[..j].copy_from_slice(r_prefix);
            point[j] = *t;
            for b in 0..suffix {
                point[j + 1 + b] = if (sb >> b) & 1 == 1 {
                    Ext::ONE
                } else {
                    Ext::ZERO
                };
            }
            acc += eq_partial(r, &point) * mle_eval(a, &point) * mle_eval(b, &point);
        }
        y[idx] = acc;
    }
    interpolate_deg3(y[0], y[1], y[2], y[3])
}

/// Committed pointwise `out[i] = a[i]·b[i]` via the degree-3 GKR sumcheck.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GkrMulProof {
    pub n: usize,
    pub rounds: Vec<RoundPoly3>,
    pub a_val: Ext,
    pub b_val: Ext,
    pub out_val: Ext,
    pub mle_a: [InnerProductProof; 3],
    pub mle_b: [InnerProductProof; 3],
    pub mle_out: [InnerProductProof; 3],
}

#[allow(clippy::too_many_arguments)]
pub fn prove_gkr_mul(
    ctx: &mut Prover,
    com_a: &fri::Commitment,
    pd_a: &fri::ProverData,
    dom_a: fri::Domain,
    com_b: &fri::Commitment,
    pd_b: &fri::ProverData,
    dom_b: fri::Domain,
    com_out: &fri::Commitment,
    pd_out: &fri::ProverData,
    dom_out: fri::Domain,
    a: &[Fp],
    b: &[Fp],
    out: &[Fp],
    n: usize,
) -> GkrMulProof {
    assert_eq!(a.len(), 1 << n);
    assert_eq!(b.len(), 1 << n);
    assert_eq!(out.len(), 1 << n);
    let a_ext: Vec<Ext> = a.iter().map(|&v| Ext::from(v)).collect();
    let b_ext: Vec<Ext> = b.iter().map(|&v| Ext::from(v)).collect();
    let out_ext: Vec<Ext> = out.iter().map(|&v| Ext::from(v)).collect();
    let mut tr = fri::Transcript::new(&ctx.challenger);
    tr.observe(com_a);
    tr.observe(com_b);
    tr.observe(com_out);
    let r: Vec<Ext> = (0..n).map(|_| tr.sample_ext()).collect();
    let out_val = mle_eval(&out_ext, &r);

    // Sumcheck over f(i) = eq(r,i)·a[i]·b[i], with initial claim out_val = out̃(r).
    let mut _claim = out_val;
    let mut istar = Vec::with_capacity(n);
    let mut rounds = Vec::with_capacity(n);
    for j in 0..n {
        let round = round_poly3_mul(&a_ext, &b_ext, &r, n, j, &istar);
        for coeff in &round.0 {
            tr.observe_ext(*coeff);
        }
        let rj = tr.sample_ext();
        rounds.push(round);
        istar.push(rj);
        _claim = round.eval(rj);
    }
    let a_val = mle_eval(&a_ext, &istar);
    let b_val = mle_eval(&b_ext, &istar);

    let mle_a = prove_mle_committed(ctx, com_a, pd_a, dom_a, a, &istar);
    let mle_b = prove_mle_committed(ctx, com_b, pd_b, dom_b, b, &istar);
    let mle_out = prove_mle_committed(ctx, com_out, pd_out, dom_out, out, &r);
    GkrMulProof {
        n,
        rounds,
        a_val,
        b_val,
        out_val,
        mle_a,
        mle_b,
        mle_out,
    }
}

pub fn verify_gkr_mul(
    challenger: &Challenger,
    com_a: &fri::Commitment,
    dom_a: fri::Domain,
    com_b: &fri::Commitment,
    dom_b: fri::Domain,
    com_out: &fri::Commitment,
    dom_out: fri::Domain,
    proof: &GkrMulProof,
    values: &[Ext],
    acc: &mut Openings,
) -> bool {
    let n = proof.n;
    let mut tr = fri::Transcript::new(challenger);
    tr.observe(com_a);
    tr.observe(com_b);
    tr.observe(com_out);
    let r: Vec<Ext> = (0..n).map(|_| tr.sample_ext()).collect();
    let mut claim = proof.out_val;
    let mut istar = Vec::with_capacity(n);
    for j in 0..n {
        let round = proof.rounds[j];
        if round.eval(Ext::ZERO) + round.eval(Ext::ONE) != claim {
            return false;
        }
        for coeff in &round.0 {
            tr.observe_ext(*coeff);
        }
        let rj = tr.sample_ext();
        istar.push(rj);
        claim = round.eval(rj);
    }
    if claim != eq_partial(&r, &istar) * proof.a_val * proof.b_val {
        return false;
    }
    verify_mle_committed(
        challenger,
        com_a,
        dom_a,
        1 << n,
        &istar,
        proof.a_val,
        &proof.mle_a,
        values,
        acc,
    ) && verify_mle_committed(
        challenger,
        com_b,
        dom_b,
        1 << n,
        &istar,
        proof.b_val,
        &proof.mle_b,
        values,
        acc,
    ) && verify_mle_committed(
        challenger,
        com_out,
        dom_out,
        1 << n,
        &r,
        proof.out_val,
        &proof.mle_out,
        values,
        acc,
    )
}

fn f_i64(x: i64) -> Fp {
    if x < 0 {
        -Fp::new(x.unsigned_abs())
    } else {
        Fp::new(x as u64)
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GkrNormProof {
    pub d: usize,
    /// The proven sum of squares `sum_sq = Σ_i x[i]²` (a field element).
    pub sum_sq: Ext,
    /// The public quantized reciprocal `inv_q = round(2^RECIP_Q / sqrt(mean_sq + eps))`.
    pub inv_q: Ext,
    pub com_x2: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom_x2: fri::Domain,
    pub com_h: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom_h: fri::Domain,
    pub com_y: fri::Commitment,
    #[serde(with = "crate::pcs::fri::serde::domain")]
    pub dom_y: fri::Domain,
    pub mul_x2: GkrMulProof,
    pub mle_x2: [InnerProductProof; 3],
    pub mul_h: GkrMulProof,
    pub h_val: Ext,
    pub y_val: Ext,
    pub mle_h: [InnerProductProof; 3],
    pub mle_y: [InnerProductProof; 3],
    pub requant: RequantProof,
}

/// Prove one committed RMSNorm row: `out = requant(x · gamma · inv_q, RECIP_Q)`.
///
/// Decomposition:
///   1. `x2 = x ⊙ x` (elementwise square, via `prove_gkr_mul`),
///   2. `sum_sq = Σ x2[i] = n · x2̃(1/2,…,1/2)` (one committed MLE at the all-halves point),
///   3. `h = x ⊙ gamma` (via `prove_gkr_mul`),
///   4. `y = h · inv_q` (scalar scale, checked by two MLE evals at a shared `r`),
///   5. `out = requant(y, RECIP_Q)` (via `prove_requant_given`).
#[allow(clippy::too_many_arguments)]
pub fn prove_gkr_norm(
    ctx: &mut Prover,
    com_x: &fri::Commitment,
    pd_x: &fri::ProverData,
    dom_x: fri::Domain,
    com_gamma: &fri::Commitment,
    pd_gamma: &fri::ProverData,
    dom_gamma: fri::Domain,
    x: &[i64],
    gamma: &[i64],
    out: &[i64],
    d: usize,
    eps: f64,
    digit_table: &TableCommitment,
    bit_table: &TableCommitment,
    digits: usize,
) -> GkrNormProof {
    let n = 1 << d;
    assert_eq!(x.len(), n);
    assert_eq!(gamma.len(), n);
    assert_eq!(out.len(), n);

    let x_fp: Vec<Fp> = x.iter().map(|&v| f_i64(v)).collect();
    let gamma_fp: Vec<Fp> = gamma.iter().map(|&v| f_i64(v)).collect();

    // 1. x2 = x ⊙ x.
    let x2: Vec<i64> = x.iter().map(|&v| v * v).collect();
    let x2_fp: Vec<Fp> = x2.iter().map(|&v| f_i64(v)).collect();
    let (com_x2, pd_x2, dom_x2) = fri::commit_vec(&ctx.pcs, &x2_fp);
    let mul_x2 = prove_gkr_mul(
        ctx, com_x, pd_x, dom_x, com_x, pd_x, dom_x, &com_x2, &pd_x2, dom_x2, &x_fp, &x_fp, &x2_fp,
        d,
    );

    // 2. sum_sq = Σ x2[i] = n · x2̃(1/2,…,1/2): every eq factor is 1/2, so the
    // MLE at the all-halves point is the sum scaled by 1/n.
    let half = Ext::from(Fp::new(2)).inverse();
    let all_half: Vec<Ext> = vec![half; d];
    let mle_x2 = prove_mle_committed(ctx, &com_x2, &pd_x2, dom_x2, &x2_fp, &all_half);
    let sum_sq: i64 = x2.iter().sum();
    let sum_sq_ext = Ext::from(Fp::new(sum_sq as u64));

    // 3. h = x ⊙ gamma.
    let h: Vec<i64> = x.iter().zip(gamma).map(|(&a, &b)| a * b).collect();
    let h_fp: Vec<Fp> = h.iter().map(|&v| f_i64(v)).collect();
    let (com_h, pd_h, dom_h) = fri::commit_vec(&ctx.pcs, &h_fp);
    let mul_h = prove_gkr_mul(
        ctx, com_x, pd_x, dom_x, com_gamma, pd_gamma, dom_gamma, &com_h, &pd_h, dom_h, &x_fp,
        &gamma_fp, &h_fp, d,
    );

    // 4. y = h · inv_q (scalar scale).
    let mean_sq = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n as f64;
    let inv_q_i64 = ((1i64 << crate::model::spec::RECIP_Q) as f64 / (mean_sq + eps).sqrt()).round() as i64;
    let inv_q_ext = Ext::from(Fp::new(inv_q_i64 as u64));
    let y: Vec<i64> = h.iter().map(|&v| v * inv_q_i64).collect();
    let y_fp: Vec<Fp> = y.iter().map(|&v| f_i64(v)).collect();
    let (com_y, pd_y, dom_y) = fri::commit_vec(&ctx.pcs, &y_fp);

    let mut tr = fri::Transcript::new(&ctx.challenger);
    tr.observe(&com_h);
    tr.observe(&com_y);
    let r: Vec<Ext> = (0..d).map(|_| tr.sample_ext()).collect();
    let h_ext: Vec<Ext> = h.iter().map(|&v| Ext::from(f_i64(v))).collect();
    let y_ext: Vec<Ext> = y.iter().map(|&v| Ext::from(f_i64(v))).collect();
    let h_val = mle_eval(&h_ext, &r);
    let y_val = mle_eval(&y_ext, &r);
    let mle_h = prove_mle_committed(ctx, &com_h, &pd_h, dom_h, &h_fp, &r);
    let mle_y = prove_mle_committed(ctx, &com_y, &pd_y, dom_y, &y_fp, &r);

    // 5. out = requant(y, RECIP_Q).
    let requant = prove_requant_given(
        ctx,
        &com_y,
        &pd_y,
        dom_y,
        &y,
        out,
        crate::model::spec::RECIP_Q,
        digits,
        digit_table,
        bit_table,
    );

    GkrNormProof {
        d,
        sum_sq: sum_sq_ext,
        inv_q: inv_q_ext,
        com_x2,
        dom_x2,
        com_h,
        dom_h,
        com_y,
        dom_y,
        mul_x2,
        mle_x2,
        mul_h,
        h_val,
        y_val,
        mle_h,
        mle_y,
        requant,
    }
}

pub fn verify_gkr_norm(
    challenger: &Challenger,
    com_x: &fri::Commitment,
    dom_x: fri::Domain,
    com_gamma: &fri::Commitment,
    dom_gamma: fri::Domain,
    digit_table: &TableCommitment,
    bit_table: &TableCommitment,
    proof: &GkrNormProof,
    values: &[Ext],
    acc: &mut Openings,
) -> bool {
    let d = proof.d;
    let n = 1 << d;

    // 1. x2 = x ⊙ x.
    if !verify_gkr_mul(
        challenger,
        com_x,
        dom_x,
        com_x,
        dom_x,
        &proof.com_x2,
        proof.dom_x2,
        &proof.mul_x2,
        values,
        acc,
    ) {
        return false;
    }

    // 2. sum_sq = Σ x2[i] = n · x2̃(1/2,…,1/2).
    let half = Ext::from(Fp::new(2)).inverse();
    let all_half: Vec<Ext> = vec![half; d];
    let inv_n = Ext::from(Fp::new(n as u64)).inverse();
    if !verify_mle_committed(
        challenger,
        &proof.com_x2,
        proof.dom_x2,
        n,
        &all_half,
        proof.sum_sq * inv_n,
        &proof.mle_x2,
        values,
        acc,
    ) {
        return false;
    }

    // 3. h = x ⊙ gamma.
    if !verify_gkr_mul(
        challenger,
        com_x,
        dom_x,
        com_gamma,
        dom_gamma,
        &proof.com_h,
        proof.dom_h,
        &proof.mul_h,
        values,
        acc,
    ) {
        return false;
    }

    // 4. y = h · inv_q: ỹ(r) = inv_q · h̃(r).
    let mut tr = fri::Transcript::new(challenger);
    tr.observe(&proof.com_h);
    tr.observe(&proof.com_y);
    let r: Vec<Ext> = (0..d).map(|_| tr.sample_ext()).collect();
    if proof.y_val != proof.inv_q * proof.h_val {
        return false;
    }
    if !verify_mle_committed(
        challenger,
        &proof.com_h,
        proof.dom_h,
        n,
        &r,
        proof.h_val,
        &proof.mle_h,
        values,
        acc,
    ) {
        return false;
    }
    if !verify_mle_committed(
        challenger,
        &proof.com_y,
        proof.dom_y,
        n,
        &r,
        proof.y_val,
        &proof.mle_y,
        values,
        acc,
    ) {
        return false;
    }

    // 5. out = requant(y).
    verify_requant(
        challenger,
        digit_table,
        bit_table,
        &proof.requant,
        values,
        acc,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lookup::lookup_logup::{
        commit_function_table, prove_lookup2_logup_given, verify_lookup2_logup,
    };
    use crate::lookup::range_check::commit_digit_table;
    use crate::lookup::requant::commit_bit_table;
    use crate::model::spec::silu;

    fn f(x: u64) -> Fp {
        Fp::new(x)
    }

    fn f_i64(x: i64) -> Fp {
        if x < 0 {
            -Fp::new(x.unsigned_abs())
        } else {
            Fp::new(x as u64)
        }
    }

    #[test]
    fn gkr_matmul_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]], C = A·B = [[19,22],[43,50]] (row-major).
        let a = [f(1), f(2), f(3), f(4)];
        let b = [f(5), f(6), f(7), f(8)];
        let c = [f(19), f(22), f(43), f(50)];
        let (com_a, pd_a, dom_a) = fri::commit_vec(&ctx.pcs, &a);
        let (com_b, pd_b, dom_b) = fri::commit_vec(&ctx.pcs, &b);
        let (com_c, pd_c, dom_c) = fri::commit_vec(&ctx.pcs, &c);
        let proof = prove_gkr_matmul(
            &mut ctx, &com_a, &pd_a, dom_a, &com_b, &pd_b, dom_b, &com_c, &pd_c, dom_c, &a, &b, &c,
            1, 1, 1,
        );
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = Openings::new();
        assert!(verify_gkr_matmul(
            &ctx.challenger,
            &com_a,
            dom_a,
            &com_b,
            dom_b,
            &com_c,
            dom_c,
            &proof,
            &values,
            &mut acc,
        ));
        assert!(fri::open_many_verify(
            &ctx.pcs,
            &ctx.challenger,
            &acc,
            &fri_proof
        ));
    }

    #[test]
    fn gkr_matmul_rejects_wrong_c() {
        let mut ctx = fri::Prover::new(1, 10);
        let a = [f(1), f(2), f(3), f(4)];
        let b = [f(5), f(6), f(7), f(8)];
        let c = [f(19), f(22), f(43), f(51)]; // C[1][1] wrong: 50 → 51
        let (com_a, pd_a, dom_a) = fri::commit_vec(&ctx.pcs, &a);
        let (com_b, pd_b, dom_b) = fri::commit_vec(&ctx.pcs, &b);
        let (com_c, pd_c, dom_c) = fri::commit_vec(&ctx.pcs, &c);
        let proof = prove_gkr_matmul(
            &mut ctx, &com_a, &pd_a, dom_a, &com_b, &pd_b, dom_b, &com_c, &pd_c, dom_c, &a, &b, &c,
            1, 1, 1,
        );
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = Openings::new();
        assert!(!verify_gkr_matmul(
            &ctx.challenger,
            &com_a,
            dom_a,
            &com_b,
            dom_b,
            &com_c,
            dom_c,
            &proof,
            &values,
            &mut acc,
        ));
        let _ = fri_proof;
    }

    #[test]
    fn gkr_matmul_larger() {
        // 4×4 × 4×2 = 4×2.
        let mut ctx = fri::Prover::new(1, 10);
        // A = identity-ish, B = random, C = A·B computed in the clear.
        let a: Vec<Fp> = (0..16).map(|i| f(i as u64)).collect();
        let b: Vec<Fp> = (0..8).map(|i| f((i + 3) as u64)).collect();
        let mut c = vec![0u64; 8];
        // Recompute C properly: A[i][k] = (i*4 + k), B[k][j] = (k*2 + j + 3).
        for i in 0usize..4 {
            for j in 0usize..2 {
                let mut acc: u64 = 0;
                for k in 0usize..4 {
                    acc += ((i * 4 + k) as u64) * ((k * 2 + j + 3) as u64);
                }
                c[i * 2 + j] = acc;
            }
        }
        let c_fp: Vec<Fp> = c.iter().map(|&x| f(x)).collect();
        let (com_a, pd_a, dom_a) = fri::commit_vec(&ctx.pcs, &a);
        let (com_b, pd_b, dom_b) = fri::commit_vec(&ctx.pcs, &b);
        let (com_c, pd_c, dom_c) = fri::commit_vec(&ctx.pcs, &c_fp);
        let proof = prove_gkr_matmul(
            &mut ctx, &com_a, &pd_a, dom_a, &com_b, &pd_b, dom_b, &com_c, &pd_c, dom_c, &a, &b,
            &c_fp, 2, 2, 1,
        );
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = Openings::new();
        assert!(verify_gkr_matmul(
            &ctx.challenger,
            &com_a,
            dom_a,
            &com_b,
            dom_b,
            &com_c,
            dom_c,
            &proof,
            &values,
            &mut acc,
        ));
        assert!(fri::open_many_verify(
            &ctx.pcs,
            &ctx.challenger,
            &acc,
            &fri_proof
        ));
    }

    #[test]
    fn gkr_2layer_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        // W1 (2×2), W2 (2×2), X (2), Y = W2·(W1·X).
        let w1 = [f(1), f(2), f(3), f(4)];
        let w2 = [f(5), f(6), f(7), f(8)];
        let x = [f(1), f(1)];
        // X1 = W1·X = [1+2, 3+4] = [3, 7]; Y = W2·X1 = [5·3+6·7, 7·3+8·7] = [57, 77].
        let y = [f(57), f(77)];
        let (com_w2, pd_w2, dom_w2) = fri::commit_vec(&ctx.pcs, &w2);
        let (com_w1, pd_w1, dom_w1) = fri::commit_vec(&ctx.pcs, &w1);
        let (com_x, pd_x, dom_x) = fri::commit_vec(&ctx.pcs, &x);
        let (com_y, pd_y, dom_y) = fri::commit_vec(&ctx.pcs, &y);
        let proof = prove_gkr_2layer(
            &mut ctx, &com_w2, &pd_w2, dom_w2, &com_w1, &pd_w1, dom_w1, &com_x, &pd_x, dom_x,
            &com_y, &pd_y, dom_y, &w2, &w1, &x, &y, 1, 1, 1,
        );
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = Openings::new();
        assert!(verify_gkr_2layer(
            &ctx.challenger,
            &com_w2,
            dom_w2,
            &com_w1,
            dom_w1,
            &com_x,
            dom_x,
            &com_y,
            dom_y,
            &proof,
            &values,
            &mut acc,
        ));
        assert!(fri::open_many_verify(
            &ctx.pcs,
            &ctx.challenger,
            &acc,
            &fri_proof
        ));
    }

    #[test]
    fn gkr_2layer_rejects_wrong_y() {
        let mut ctx = fri::Prover::new(1, 10);
        let w1 = [f(1), f(2), f(3), f(4)];
        let w2 = [f(5), f(6), f(7), f(8)];
        let x = [f(1), f(1)];
        let y = [f(57), f(78)]; // Y[1] wrong: 77 → 78
        let (com_w2, pd_w2, dom_w2) = fri::commit_vec(&ctx.pcs, &w2);
        let (com_w1, pd_w1, dom_w1) = fri::commit_vec(&ctx.pcs, &w1);
        let (com_x, pd_x, dom_x) = fri::commit_vec(&ctx.pcs, &x);
        let (com_y, pd_y, dom_y) = fri::commit_vec(&ctx.pcs, &y);
        let proof = prove_gkr_2layer(
            &mut ctx, &com_w2, &pd_w2, dom_w2, &com_w1, &pd_w1, dom_w1, &com_x, &pd_x, dom_x,
            &com_y, &pd_y, dom_y, &w2, &w1, &x, &y, 1, 1, 1,
        );
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = Openings::new();
        assert!(!verify_gkr_2layer(
            &ctx.challenger,
            &com_w2,
            dom_w2,
            &com_w1,
            dom_w1,
            &com_x,
            dom_x,
            &com_y,
            dom_y,
            &proof,
            &values,
            &mut acc,
        ));
        let _ = fri_proof;
    }

    #[test]
    fn gkr_3layer_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        let w0 = [f(1), f(2), f(3), f(4)];
        let w1 = [f(5), f(6), f(7), f(8)];
        let w2 = [f(9), f(10), f(11), f(12)];
        let x = [f(1), f(1)];
        let z1 = compute_matvec(&w0, &x, 2, 2);
        let z2 = compute_matvec(&w1, &z1, 2, 2);
        let y = compute_matvec(&w2, &z2, 2, 2);
        let (com_w0, pd_w0, dom_w0) = fri::commit_vec(&ctx.pcs, &w0);
        let (com_w1, pd_w1, dom_w1) = fri::commit_vec(&ctx.pcs, &w1);
        let (com_w2, pd_w2, dom_w2) = fri::commit_vec(&ctx.pcs, &w2);
        let (com_x, pd_x, dom_x) = fri::commit_vec(&ctx.pcs, &x);
        let (com_y, pd_y, dom_y) = fri::commit_vec(&ctx.pcs, &y);
        let weights: Vec<&[Fp]> = vec![&w0, &w1, &w2];
        let coms: Vec<&fri::Commitment> = vec![&com_w0, &com_w1, &com_w2];
        let pds: Vec<&fri::ProverData> = vec![&pd_w0, &pd_w1, &pd_w2];
        let doms: Vec<fri::Domain> = vec![dom_w0, dom_w1, dom_w2];
        let proof = prove_gkr(
            &mut ctx,
            &weights,
            &coms,
            &pds,
            &doms,
            &com_x,
            &pd_x,
            dom_x,
            &com_y,
            &pd_y,
            dom_y,
            &x,
            &y,
            &[1, 1, 1, 1],
        );
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = Openings::new();
        assert!(verify_gkr(
            &ctx.challenger,
            &coms,
            &doms,
            &com_x,
            dom_x,
            &com_y,
            dom_y,
            &proof,
            &values,
            &mut acc,
        ));
        assert!(fri::open_many_verify(
            &ctx.pcs,
            &ctx.challenger,
            &acc,
            &fri_proof
        ));
    }

    #[test]
    fn gkr_3layer_rejects_wrong_y() {
        let mut ctx = fri::Prover::new(1, 10);
        let w0 = [f(1), f(2), f(3), f(4)];
        let w1 = [f(5), f(6), f(7), f(8)];
        let w2 = [f(9), f(10), f(11), f(12)];
        let x = [f(1), f(1)];
        let z1 = compute_matvec(&w0, &x, 2, 2);
        let z2 = compute_matvec(&w1, &z1, 2, 2);
        let mut y = compute_matvec(&w2, &z2, 2, 2);
        y[0] = y[0] + Fp::ONE;
        let (com_w0, pd_w0, dom_w0) = fri::commit_vec(&ctx.pcs, &w0);
        let (com_w1, pd_w1, dom_w1) = fri::commit_vec(&ctx.pcs, &w1);
        let (com_w2, pd_w2, dom_w2) = fri::commit_vec(&ctx.pcs, &w2);
        let (com_x, pd_x, dom_x) = fri::commit_vec(&ctx.pcs, &x);
        let (com_y, pd_y, dom_y) = fri::commit_vec(&ctx.pcs, &y);
        let weights: Vec<&[Fp]> = vec![&w0, &w1, &w2];
        let coms: Vec<&fri::Commitment> = vec![&com_w0, &com_w1, &com_w2];
        let pds: Vec<&fri::ProverData> = vec![&pd_w0, &pd_w1, &pd_w2];
        let doms: Vec<fri::Domain> = vec![dom_w0, dom_w1, dom_w2];
        let proof = prove_gkr(
            &mut ctx,
            &weights,
            &coms,
            &pds,
            &doms,
            &com_x,
            &pd_x,
            dom_x,
            &com_y,
            &pd_y,
            dom_y,
            &x,
            &y,
            &[1, 1, 1, 1],
        );
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = Openings::new();
        assert!(!verify_gkr(
            &ctx.challenger,
            &coms,
            &doms,
            &com_x,
            dom_x,
            &com_y,
            dom_y,
            &proof,
            &values,
            &mut acc,
        ));
        let _ = fri_proof;
    }

    #[test]
    fn gkr_silu_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        // W = [[2,0],[0,2]], X = [-1,1] → H = W·X = [-2, 2]; Y = SiLU(H) = [0, 2].
        let w_i64 = [2i64, 0, 0, 2];
        let x_i64 = [-1i64, 1];
        let h_i64 = [-2i64, 2];
        let y_i64 = [silu(-2), silu(2)];
        let w_fp: Vec<Fp> = w_i64.iter().map(|&v| f_i64(v)).collect();
        let x_fp: Vec<Fp> = x_i64.iter().map(|&v| f_i64(v)).collect();
        let h_fp: Vec<Fp> = h_i64.iter().map(|&v| f_i64(v)).collect();
        let y_fp: Vec<Fp> = y_i64.iter().map(|&v| f_i64(v)).collect();
        let (com_w, pd_w, dom_w) = fri::commit_vec(&ctx.pcs, &w_fp);
        let (com_x, pd_x, dom_x) = fri::commit_vec(&ctx.pcs, &x_fp);
        let (com_h, pd_h, dom_h) = fri::commit_vec(&ctx.pcs, &h_fp);
        let (com_y, pd_y, dom_y) = fri::commit_vec(&ctx.pcs, &y_fp);

        // Linear part via GKR: H = W·X (matvec, ln=1, lm=1, lp=0).
        let gkr = prove_gkr_matmul(
            &mut ctx, &com_w, &pd_w, dom_w, &com_x, &pd_x, dom_x, &com_h, &pd_h, dom_h, &w_fp,
            &x_fp, &h_fp, 1, 1, 0,
        );

        // Nonlinear part via a SiLU lookup: Y = SiLU(H).
        let alpha = f(11);
        let beta = f(5);
        let silu_table = commit_function_table(&ctx.pcs, silu, -16, 16, alpha, beta);
        let lookup = prove_lookup2_logup_given(
            &mut ctx,
            &com_h,
            &pd_h,
            dom_h,
            &com_y,
            &pd_y,
            dom_y,
            &h_i64,
            &y_i64,
            &silu_table,
            alpha,
            beta,
        );

        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = Openings::new();
        assert!(verify_gkr_matmul(
            &ctx.challenger,
            &com_w,
            dom_w,
            &com_x,
            dom_x,
            &com_h,
            dom_h,
            &gkr,
            &values,
            &mut acc,
        ));
        assert!(verify_lookup2_logup(
            &ctx.challenger,
            &silu_table.table.com_ft,
            silu_table.table.dom_ft,
            &lookup,
            &values,
            &mut acc,
        ));
        assert!(fri::open_many_verify(
            &ctx.pcs,
            &ctx.challenger,
            &acc,
            &fri_proof
        ));
    }

    // A full two-matmul FFN with a SiLU nonlinearity between them:
    //   H = W1·X, S = SiLU(H), Y = W2·S.
    // The linear parts use GKR; the nonlinearity uses a lookup; all in one FRI proof.
    #[test]
    fn gkr_ffn_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        let w1_i64 = [1i64, 0, 0, 1]; // identity
        let w2_i64 = [1i64, 1, 0, 1];
        let x_i64 = [-1i64, 2];
        let h_i64 = [-1i64, 2]; // H = W1·X
        let s_i64 = [silu(-1), silu(2)]; // S = SiLU(H) = [0, 2]
        let y_i64 = [2i64, 2]; // Y = W2·S = [0+2, 0+2]
        let w1_fp: Vec<Fp> = w1_i64.iter().map(|&v| f_i64(v)).collect();
        let w2_fp: Vec<Fp> = w2_i64.iter().map(|&v| f_i64(v)).collect();
        let x_fp: Vec<Fp> = x_i64.iter().map(|&v| f_i64(v)).collect();
        let h_fp: Vec<Fp> = h_i64.iter().map(|&v| f_i64(v)).collect();
        let s_fp: Vec<Fp> = s_i64.iter().map(|&v| f_i64(v)).collect();
        let y_fp: Vec<Fp> = y_i64.iter().map(|&v| f_i64(v)).collect();
        let (com_w1, pd_w1, dom_w1) = fri::commit_vec(&ctx.pcs, &w1_fp);
        let (com_w2, pd_w2, dom_w2) = fri::commit_vec(&ctx.pcs, &w2_fp);
        let (com_x, pd_x, dom_x) = fri::commit_vec(&ctx.pcs, &x_fp);
        let (com_h, pd_h, dom_h) = fri::commit_vec(&ctx.pcs, &h_fp);
        let (com_s, pd_s, dom_s) = fri::commit_vec(&ctx.pcs, &s_fp);
        let (com_y, pd_y, dom_y) = fri::commit_vec(&ctx.pcs, &y_fp);

        // 1. H = W1·X (GKR matvec).
        let gkr1 = prove_gkr_matmul(
            &mut ctx, &com_w1, &pd_w1, dom_w1, &com_x, &pd_x, dom_x, &com_h, &pd_h, dom_h, &w1_fp,
            &x_fp, &h_fp, 1, 1, 0,
        );
        // 2. S = SiLU(H) (lookup).
        let alpha = f(11);
        let beta = f(5);
        let silu_table = commit_function_table(&ctx.pcs, silu, -16, 16, alpha, beta);
        let lookup = prove_lookup2_logup_given(
            &mut ctx,
            &com_h,
            &pd_h,
            dom_h,
            &com_s,
            &pd_s,
            dom_s,
            &h_i64,
            &s_i64,
            &silu_table,
            alpha,
            beta,
        );
        // 3. Y = W2·S (GKR matvec).
        let gkr2 = prove_gkr_matmul(
            &mut ctx, &com_w2, &pd_w2, dom_w2, &com_s, &pd_s, dom_s, &com_y, &pd_y, dom_y, &w2_fp,
            &s_fp, &y_fp, 1, 1, 0,
        );

        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = Openings::new();
        assert!(verify_gkr_matmul(
            &ctx.challenger,
            &com_w1,
            dom_w1,
            &com_x,
            dom_x,
            &com_h,
            dom_h,
            &gkr1,
            &values,
            &mut acc,
        ));
        assert!(verify_lookup2_logup(
            &ctx.challenger,
            &silu_table.table.com_ft,
            silu_table.table.dom_ft,
            &lookup,
            &values,
            &mut acc,
        ));
        assert!(verify_gkr_matmul(
            &ctx.challenger,
            &com_w2,
            dom_w2,
            &com_s,
            dom_s,
            &com_y,
            dom_y,
            &gkr2,
            &values,
            &mut acc,
        ));
        assert!(fri::open_many_verify(
            &ctx.pcs,
            &ctx.challenger,
            &acc,
            &fri_proof
        ));
    }

    #[test]
    fn gkr_ffn_rejects_wrong_y() {
        let mut ctx = fri::Prover::new(1, 10);
        let w1_i64 = [1i64, 0, 0, 1];
        let w2_i64 = [1i64, 1, 0, 1];
        let x_i64 = [-1i64, 2];
        let h_i64 = [-1i64, 2];
        let s_i64 = [silu(-1), silu(2)];
        let y_i64 = [2i64, 3]; // Y[1] wrong: 2 → 3
        let w1_fp: Vec<Fp> = w1_i64.iter().map(|&v| f_i64(v)).collect();
        let w2_fp: Vec<Fp> = w2_i64.iter().map(|&v| f_i64(v)).collect();
        let x_fp: Vec<Fp> = x_i64.iter().map(|&v| f_i64(v)).collect();
        let h_fp: Vec<Fp> = h_i64.iter().map(|&v| f_i64(v)).collect();
        let s_fp: Vec<Fp> = s_i64.iter().map(|&v| f_i64(v)).collect();
        let y_fp: Vec<Fp> = y_i64.iter().map(|&v| f_i64(v)).collect();
        let (com_w1, pd_w1, dom_w1) = fri::commit_vec(&ctx.pcs, &w1_fp);
        let (com_w2, pd_w2, dom_w2) = fri::commit_vec(&ctx.pcs, &w2_fp);
        let (com_x, pd_x, dom_x) = fri::commit_vec(&ctx.pcs, &x_fp);
        let (com_h, pd_h, dom_h) = fri::commit_vec(&ctx.pcs, &h_fp);
        let (com_s, pd_s, dom_s) = fri::commit_vec(&ctx.pcs, &s_fp);
        let (com_y, pd_y, dom_y) = fri::commit_vec(&ctx.pcs, &y_fp);
        let gkr1 = prove_gkr_matmul(
            &mut ctx, &com_w1, &pd_w1, dom_w1, &com_x, &pd_x, dom_x, &com_h, &pd_h, dom_h, &w1_fp,
            &x_fp, &h_fp, 1, 1, 0,
        );
        let alpha = f(11);
        let beta = f(5);
        let silu_table = commit_function_table(&ctx.pcs, silu, -16, 16, alpha, beta);
        let lookup = prove_lookup2_logup_given(
            &mut ctx,
            &com_h,
            &pd_h,
            dom_h,
            &com_s,
            &pd_s,
            dom_s,
            &h_i64,
            &s_i64,
            &silu_table,
            alpha,
            beta,
        );
        let gkr2 = prove_gkr_matmul(
            &mut ctx, &com_w2, &pd_w2, dom_w2, &com_s, &pd_s, dom_s, &com_y, &pd_y, dom_y, &w2_fp,
            &s_fp, &y_fp, 1, 1, 0,
        );
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = Openings::new();
        verify_gkr_matmul(
            &ctx.challenger,
            &com_w1,
            dom_w1,
            &com_x,
            dom_x,
            &com_h,
            dom_h,
            &gkr1,
            &values,
            &mut acc,
        );
        let _ = verify_lookup2_logup(
            &ctx.challenger,
            &silu_table.table.com_ft,
            silu_table.table.dom_ft,
            &lookup,
            &values,
            &mut acc,
        );
        assert!(!verify_gkr_matmul(
            &ctx.challenger,
            &com_w2,
            dom_w2,
            &com_s,
            dom_s,
            &com_y,
            dom_y,
            &gkr2,
            &values,
            &mut acc,
        ));
        let _ = fri_proof;
    }

    /// Full linear FFN op-graph via GKR: `h1 = W1·x`, `residual = h1 + x`,
    /// `logits = W2·residual` — matmuls via GKR, add via pointwise MLE.
    #[test]
    fn gkr_linear_ffn_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        let w1_i64 = [1i64, 0, 0, 1]; // identity
        let w2_i64 = [1i64, 1, 0, 1];
        let x_i64 = [1i64, 2];
        let h1_i64 = [1i64, 2]; // h1 = W1·x
        let res_i64 = [2i64, 4]; // residual = h1 + x
        let logits_i64 = [6i64, 4]; // logits = W2·residual = [2+4, 0+4]
        let w1_fp: Vec<Fp> = w1_i64.iter().map(|&v| f_i64(v)).collect();
        let w2_fp: Vec<Fp> = w2_i64.iter().map(|&v| f_i64(v)).collect();
        let x_fp: Vec<Fp> = x_i64.iter().map(|&v| f_i64(v)).collect();
        let h1_fp: Vec<Fp> = h1_i64.iter().map(|&v| f_i64(v)).collect();
        let res_fp: Vec<Fp> = res_i64.iter().map(|&v| f_i64(v)).collect();
        let logits_fp: Vec<Fp> = logits_i64.iter().map(|&v| f_i64(v)).collect();
        let (com_w1, pd_w1, dom_w1) = fri::commit_vec(&ctx.pcs, &w1_fp);
        let (com_w2, pd_w2, dom_w2) = fri::commit_vec(&ctx.pcs, &w2_fp);
        let (com_x, pd_x, dom_x) = fri::commit_vec(&ctx.pcs, &x_fp);
        let (com_h1, pd_h1, dom_h1) = fri::commit_vec(&ctx.pcs, &h1_fp);
        let (com_res, pd_res, dom_res) = fri::commit_vec(&ctx.pcs, &res_fp);
        let (com_log, pd_log, dom_log) = fri::commit_vec(&ctx.pcs, &logits_fp);

        let gkr1 = prove_gkr_matmul(
            &mut ctx, &com_w1, &pd_w1, dom_w1, &com_x, &pd_x, dom_x, &com_h1, &pd_h1, dom_h1,
            &w1_fp, &x_fp, &h1_fp, 1, 1, 0,
        );
        let add = prove_gkr_add(
            &mut ctx, &com_h1, &pd_h1, dom_h1, &com_x, &pd_x, dom_x, &com_res, &pd_res, dom_res,
            &h1_fp, &x_fp, &res_fp, 1,
        );
        let gkr2 = prove_gkr_matmul(
            &mut ctx, &com_w2, &pd_w2, dom_w2, &com_res, &pd_res, dom_res, &com_log, &pd_log,
            dom_log, &w2_fp, &res_fp, &logits_fp, 1, 1, 0,
        );

        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = Openings::new();
        assert!(verify_gkr_matmul(
            &ctx.challenger,
            &com_w1,
            dom_w1,
            &com_x,
            dom_x,
            &com_h1,
            dom_h1,
            &gkr1,
            &values,
            &mut acc,
        ));
        assert!(verify_gkr_add(
            &ctx.challenger,
            &com_h1,
            dom_h1,
            &com_x,
            dom_x,
            &com_res,
            dom_res,
            1,
            &add,
            &values,
            &mut acc,
        ));
        assert!(verify_gkr_matmul(
            &ctx.challenger,
            &com_w2,
            dom_w2,
            &com_res,
            dom_res,
            &com_log,
            dom_log,
            &gkr2,
            &values,
            &mut acc,
        ));
        assert!(fri::open_many_verify(
            &ctx.pcs,
            &ctx.challenger,
            &acc,
            &fri_proof
        ));
    }

    #[test]
    fn gkr_mul_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        // a = [1,2,3,4], b = [5,6,7,8], out = a⊙b = [5,12,21,32].
        let a = [f(1), f(2), f(3), f(4)];
        let b = [f(5), f(6), f(7), f(8)];
        let out = [f(5), f(12), f(21), f(32)];
        let (com_a, pd_a, dom_a) = fri::commit_vec(&ctx.pcs, &a);
        let (com_b, pd_b, dom_b) = fri::commit_vec(&ctx.pcs, &b);
        let (com_out, pd_out, dom_out) = fri::commit_vec(&ctx.pcs, &out);
        let proof = prove_gkr_mul(
            &mut ctx, &com_a, &pd_a, dom_a, &com_b, &pd_b, dom_b, &com_out, &pd_out, dom_out, &a,
            &b, &out, 2,
        );
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = Openings::new();
        assert!(verify_gkr_mul(
            &ctx.challenger,
            &com_a,
            dom_a,
            &com_b,
            dom_b,
            &com_out,
            dom_out,
            &proof,
            &values,
            &mut acc,
        ));
        assert!(fri::open_many_verify(
            &ctx.pcs,
            &ctx.challenger,
            &acc,
            &fri_proof
        ));
    }

    #[test]
    fn gkr_mul_rejects_wrong() {
        let mut ctx = fri::Prover::new(1, 10);
        let a = [f(1), f(2), f(3), f(4)];
        let b = [f(5), f(6), f(7), f(8)];
        let out = [f(5), f(12), f(21), f(33)]; // out[3] wrong: 32 → 33
        let (com_a, pd_a, dom_a) = fri::commit_vec(&ctx.pcs, &a);
        let (com_b, pd_b, dom_b) = fri::commit_vec(&ctx.pcs, &b);
        let (com_out, pd_out, dom_out) = fri::commit_vec(&ctx.pcs, &out);
        let proof = prove_gkr_mul(
            &mut ctx, &com_a, &pd_a, dom_a, &com_b, &pd_b, dom_b, &com_out, &pd_out, dom_out, &a,
            &b, &out, 2,
        );
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = Openings::new();
        assert!(!verify_gkr_mul(
            &ctx.challenger,
            &com_a,
            dom_a,
            &com_b,
            dom_b,
            &com_out,
            dom_out,
            &proof,
            &values,
            &mut acc,
        ));
        let _ = fri_proof;
    }

    #[test]
    fn gkr_norm_honest() {
        let mut ctx = fri::Prover::new(1, 10);
        let dt = commit_digit_table(&ctx.pcs);
        let bt = commit_bit_table(&ctx.pcs);
        let d = 2;
        let eps = 1e-6;
        let x = [1i64, -2, 3, 4];
        let gamma = [1i64, 1, 1, 1];
        let sum_sq: i64 = x.iter().map(|&v| v * v).sum();
        let mean_sq = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
        let inv_q = ((1i64 << crate::model::spec::RECIP_Q) as f64 / (mean_sq + eps).sqrt()).round() as i64;
        let y: Vec<i64> = x.iter().zip(&gamma).map(|(&v, &g)| v * g * inv_q).collect();
        let out: Vec<i64> = y
            .iter()
            .map(|&v| crate::model::spec::requant(v, crate::model::spec::RECIP_Q))
            .collect();
        assert!(sum_sq > 0);

        let x_fp: Vec<Fp> = x.iter().map(|&v| f_i64(v)).collect();
        let gamma_fp: Vec<Fp> = gamma.iter().map(|&v| f_i64(v)).collect();
        let (com_x, pd_x, dom_x) = fri::commit_vec(&ctx.pcs, &x_fp);
        let (com_gamma, pd_gamma, dom_gamma) = fri::commit_vec(&ctx.pcs, &gamma_fp);

        let proof = prove_gkr_norm(
            &mut ctx, &com_x, &pd_x, dom_x, &com_gamma, &pd_gamma, dom_gamma, &x, &gamma, &out, d,
            eps, &dt, &bt, 4,
        );
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = Openings::new();
        assert!(verify_gkr_norm(
            &ctx.challenger,
            &com_x,
            dom_x,
            &com_gamma,
            dom_gamma,
            &dt,
            &bt,
            &proof,
            &values,
            &mut acc,
        ));
        assert!(fri::open_many_verify(
            &ctx.pcs,
            &ctx.challenger,
            &acc,
            &fri_proof
        ));
    }

    #[test]
    fn gkr_norm_rejects_wrong_out() {
        let mut ctx = fri::Prover::new(1, 10);
        let dt = commit_digit_table(&ctx.pcs);
        let bt = commit_bit_table(&ctx.pcs);
        let d = 2;
        let eps = 1e-6;
        let x = [1i64, -2, 3, 4];
        let gamma = [1i64, 1, 1, 1];
        let mean_sq = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
        let inv_q = ((1i64 << crate::model::spec::RECIP_Q) as f64 / (mean_sq + eps).sqrt()).round() as i64;
        let y: Vec<i64> = x.iter().zip(&gamma).map(|(&v, &g)| v * g * inv_q).collect();
        let mut out: Vec<i64> = y
            .iter()
            .map(|&v| crate::model::spec::requant(v, crate::model::spec::RECIP_Q))
            .collect();
        out[0] += 1; // wrong output

        let x_fp: Vec<Fp> = x.iter().map(|&v| f_i64(v)).collect();
        let gamma_fp: Vec<Fp> = gamma.iter().map(|&v| f_i64(v)).collect();
        let (com_x, pd_x, dom_x) = fri::commit_vec(&ctx.pcs, &x_fp);
        let (com_gamma, pd_gamma, dom_gamma) = fri::commit_vec(&ctx.pcs, &gamma_fp);

        let proof = prove_gkr_norm(
            &mut ctx, &com_x, &pd_x, dom_x, &com_gamma, &pd_gamma, dom_gamma, &x, &gamma, &out, d,
            eps, &dt, &bt, 4,
        );
        let fri_proof = ctx.open_all();
        let values = ctx.values.clone();
        let mut acc = Openings::new();
        assert!(!verify_gkr_norm(
            &ctx.challenger,
            &com_x,
            dom_x,
            &com_gamma,
            dom_gamma,
            &dt,
            &bt,
            &proof,
            &values,
            &mut acc,
        ));
        let _ = fri_proof;
    }
}
