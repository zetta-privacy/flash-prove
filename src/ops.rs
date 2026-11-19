//! Remaining nonlinear/attention/router ops — softmax and top-k — arithmetized
//! on top of the LogUp lookup ([`crate::lookup`]). The exp/selection nonlinearity
//! is a 2-column lookup; the max/count/normalize structure is a set of linear and
//! range identities.

use crate::field::Ext;
use crate::lookup::{i64_to_ext, multiplicities, verify_lookup, verify_requant};
use crate::spec::exp_q15;

/// Verify one (non-causal) softmax row:
///
/// ```text
///   m   = max(scores)
///   e[t] = exp_q15(scores[t] − m)
///   S   = Σ e[t]
///   probs[t] = e[t] · 2^15 / S   (floor)
/// ```
///
/// The `exp_q15` step is checked as a 2-column LogUp lookup `(d[t], e[t])` into the
/// public exp table `{(x, exp_q15(x)) : x ∈ [exp_lo, 0]}`; the max and the
/// normalize are checked by the linear/range identities (`scores ≤ m`, and the
/// floor-division reconstruction). `exp_lo` must be ≤ `min(scores − m)`.
pub fn verify_softmax(scores: &[i64], probs: &[i64], exp_lo: i64, alpha: Ext, beta: Ext) -> bool {
    let l = scores.len();
    if probs.len() != l || l == 0 {
        return false;
    }
    let max = *scores.iter().max().unwrap();
    // max dominates every score.
    if scores.iter().any(|&s| s > max) {
        return false;
    }
    let d: Vec<i64> = scores.iter().map(|&s| s - max).collect();
    let e: Vec<i64> = d.iter().map(|&x| exp_q15(x)).collect();

    // exp lookup: (d[t], e[t]) ∈ exp table.
    let table: Vec<Vec<Ext>> = (exp_lo..=0)
        .map(|x| vec![i64_to_ext(x), i64_to_ext(exp_q15(x))])
        .collect();
    let q: Vec<Vec<Ext>> = d
        .iter()
        .zip(&e)
        .map(|(&x, &y)| vec![i64_to_ext(x), i64_to_ext(y)])
        .collect();
    let m = multiplicities(&q, &table);
    if !verify_lookup(&q, &table, &m, alpha, beta) {
        return false;
    }

    // normalize: probs[t] = floor(e[t] · 2^15 / S).
    let sum: i64 = e.iter().sum();
    if sum == 0 {
        return false;
    }
    for t in 0..l {
        if probs[t] != e[t] * 32768 / sum {
            return false;
        }
    }
    true
}

/// Verify one top-k selection: `choice[i] = 1` iff `i` is among the `k` largest
/// `scores` (ties broken by lower index, matching [`crate::spec`]'s `TopK`).
///
/// - `choice[i] ∈ {0,1}` (1-column LogUp lookup into `{0,1}`),
/// - `Σ_i choice[i] = k` (count),
/// - dominance: `choice[i]=1, choice[j]=0 ⟹ scores[i] > scores[j]` or
///   (`scores[i] == scores[j]` and `i < j`).
pub fn verify_topk(scores: &[i64], choice: &[i64], k: usize, alpha: Ext, beta: Ext) -> bool {
    let e = scores.len();
    if choice.len() != e {
        return false;
    }
    // choice ∈ {0,1}.
    let table01 = vec![vec![i64_to_ext(0)], vec![i64_to_ext(1)]];
    let q01: Vec<Vec<Ext>> = choice.iter().map(|&c| vec![i64_to_ext(c)]).collect();
    let m01 = multiplicities(&q01, &table01);
    if !verify_lookup(&q01, &table01, &m01, alpha, beta) {
        return false;
    }
    // exactly k selected.
    if choice.iter().sum::<i64>() != k as i64 {
        return false;
    }
    // dominance with lower-index tie-breaking.
    for i in 0..e {
        for j in 0..e {
            if choice[i] == 1 && choice[j] == 0 {
                if scores[i] < scores[j] || (scores[i] == scores[j] && i > j) {
                    return false;
                }
            }
        }
    }
    true
}

/// Verify RoPE on one head vector: `out[i] = requant(rotation(pos, θ)·in[i], 15)`
/// per pair, with the rotation
///
/// ```text
///   rot_a = a·c − b·s,  rot_b = a·s + b·c,  (a,b) = (in[2j], in[2j+1])
///   c = round(cos(angle)·2^15), s = round(sin(angle)·2^15), angle = pos·θ^(−i/d)
/// ```
///
/// The trig coefficients `(c, s)` are public (they depend only on `pos`, `θ`, `d`),
/// so the rotation is a deterministic bilinear map of the committed input; the
/// only nonlinearity is the `requant(·, 15)` rounding, checked as a 2-column LogUp
/// lookup into the requant table over `[req_lo, req_hi]` (which must bound every
/// `rot` value).
pub fn verify_rope(
    inp: &[i64],
    out: &[i64],
    pos: usize,
    theta: f64,
    req_lo: i64,
    req_hi: i64,
    alpha: Ext,
    beta: Ext,
) -> bool {
    let d = inp.len();
    if out.len() != d || d % 2 != 0 {
        return false;
    }
    let pairs = d / 2;
    let mut rot = Vec::with_capacity(d);
    for j in 0..pairs {
        let i = 2 * j;
        let angle = (pos as f64) * theta.powf(-(i as f64) / d as f64);
        let c = (angle.cos() * 32768.0).round() as i64;
        let s = (angle.sin() * 32768.0).round() as i64;
        let (a, b) = (inp[i], inp[i + 1]);
        rot.push(a * c - b * s);
        rot.push(a * s + b * c);
    }
    verify_requant(&rot, out, 15, req_lo, req_hi, alpha, beta)
}

/// Verify RMSNorm `out[i] = round(x[i]·gamma[i]/sqrt(mean_sq+eps))` with
/// `mean_sq = Σ x²/d`.
///
/// The mean-square is a degree-2 reduction (a sum of squares, checked succinctly
/// by a degree-2 sumcheck); the `1/sqrt` normalization is a public function of the
/// bounded `mean_sq` and becomes a lookup in the fully-quantized model. This
/// arithmetic-level check recomputes the float normalization to stay bit-exact with
/// [`crate::spec::rmsnorm_vec`].
pub fn verify_norm(x: &[i64], gamma: &[i8], out: &[i64], eps: f64) -> bool {
    let d = x.len();
    if gamma.len() != d || out.len() != d {
        return false;
    }
    let sum_sq: i64 = x.iter().map(|&v| v * v).sum();
    let mean_sq = sum_sq as f64 / d as f64;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    for i in 0..d {
        if out[i] != (x[i] as f64 * inv * gamma[i] as f64).round() as i64 {
            return false;
        }
    }
    true
}

/// Verify one causal-softmax row (position `l` over the sequence `t`):
///
/// ```text
///   m   = max_{t≤l} scores[t]
///   e[t] = exp_q15(scores[t] − m)   (t ≤ l)
///   S   = Σ_{t≤l} e[t]
///   probs[t] = e[t]·2^15 / S (t ≤ l),  0 (t > l)
/// ```
///
/// The `exp_q15` step is a 2-column LogUp lookup into the public exp table over
/// `[exp_lo, 0]`; the running max, causal mask, and normalize are linear/range
/// identities.
pub fn verify_causal_softmax_row(
    scores: &[i64],
    probs: &[i64],
    l: usize,
    exp_lo: i64,
    alpha: Ext,
    beta: Ext,
) -> bool {
    let n = scores.len();
    if probs.len() != n || l >= n {
        return false;
    }
    let m = *scores[..=l].iter().max().unwrap();
    let mut d = Vec::with_capacity(l + 1);
    let mut e = Vec::with_capacity(l + 1);
    for t in 0..=l {
        let dt = scores[t] - m;
        d.push(dt);
        e.push(exp_q15(dt));
    }
    // exp lookup: (d[t], e[t]) ∈ exp table over [exp_lo, 0].
    if !crate::lookup::verify_function_lookup(&d, &e, exp_q15, exp_lo, 0, alpha, beta) {
        return false;
    }
    // causal mask + normalize.
    let s: i64 = e.iter().sum();
    for t in 0..n {
        if t > l {
            if probs[t] != 0 {
                return false;
            }
        } else if s == 0 {
            if probs[t] != 0 {
                return false;
            }
        } else if probs[t] != e[t] * 32768 / s {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Fp;
    use p3_field::PrimeCharacteristicRing;

    fn a() -> Ext {
        Ext::from(Fp::new(11))
    }
    fn b() -> Ext {
        Ext::from(Fp::new(5))
    }

    #[test]
    fn softmax_honest() {
        let scores = [3, 1, 5, 2];
        // max = 5, d = [-2,-4,0,-3], e = exp_q15(d), S, probs.
        let max = 5i64;
        let e: Vec<i64> = scores.iter().map(|&s| exp_q15(s - max)).collect();
        let sum: i64 = e.iter().sum();
        let probs: Vec<i64> = e.iter().map(|&v| v * 32768 / sum).collect();
        assert!(verify_softmax(&scores, &probs, -8, a(), b()));
    }

    #[test]
    fn softmax_rejects_wrong_prob() {
        let scores = [3, 1, 5, 2];
        let max = 5i64;
        let e: Vec<i64> = scores.iter().map(|&s| exp_q15(s - max)).collect();
        let sum: i64 = e.iter().sum();
        let mut probs: Vec<i64> = e.iter().map(|&v| v * 32768 / sum).collect();
        probs[0] += 1;
        assert!(!verify_softmax(&scores, &probs, -8, a(), b()));
    }

    #[test]
    fn softmax_rejects_out_of_domain() {
        // A score so low that scores[t] − max is below exp_lo = -4.
        let scores = [0, 100];
        let max = 100i64;
        let e: Vec<i64> = scores.iter().map(|&s| exp_q15(s - max)).collect();
        let sum: i64 = e.iter().sum();
        let probs: Vec<i64> = e.iter().map(|&v| v * 32768 / sum).collect();
        // exp_lo = -4, but d[0] = -100 < -4 → the lookup must reject.
        assert!(!verify_softmax(&scores, &probs, -4, a(), b()));
    }

    #[test]
    fn topk_honest() {
        let scores = [10, 30, 20, 40];
        // top-2 = indices {3 (40), 1 (30)}.
        let choice = [0, 1, 0, 1];
        assert!(verify_topk(&scores, &choice, 2, a(), b()));
    }

    #[test]
    fn topk_rejects_wrong_choice() {
        let scores = [10, 30, 20, 40];
        // picks {2,3} = {20,40} instead of {1,3} = {30,40}.
        let choice = [0, 0, 1, 1];
        assert!(!verify_topk(&scores, &choice, 2, a(), b()));
    }

    #[test]
    fn topk_rejects_wrong_count() {
        let scores = [10, 30, 20, 40];
        let choice = [0, 1, 0, 0]; // only one selected
        assert!(!verify_topk(&scores, &choice, 2, a(), b()));
    }

    #[test]
    fn topk_tie_break_by_index() {
        // Ties at 30 for indices 1 and 2; lower index (1) wins the second slot.
        let scores = [10, 30, 30, 40];
        let choice = [0, 1, 0, 1]; // {1 (30), 3 (40)}
        assert!(verify_topk(&scores, &choice, 2, a(), b()));
        let wrong = [0, 0, 1, 1]; // {2 (30), 3 (40)} — higher-index tie chosen
        assert!(!verify_topk(&scores, &wrong, 2, a(), b()));
    }

    #[test]
    fn rope_honest() {
        let inp = [1, 2];
        let mut out = inp.to_vec();
        crate::spec::rope_apply(&mut out, 3, 10000.0);
        assert!(verify_rope(
            &inp,
            &out,
            3,
            10000.0,
            -200_000,
            200_000,
            a(),
            b()
        ));
    }

    #[test]
    fn rope_rejects_wrong() {
        let inp = [1, 2];
        let mut out = inp.to_vec();
        crate::spec::rope_apply(&mut out, 3, 10000.0);
        out[0] += 1;
        assert!(!verify_rope(
            &inp,
            &out,
            3,
            10000.0,
            -200_000,
            200_000,
            a(),
            b()
        ));
    }

    #[test]
    fn norm_honest() {
        let x = [1, 2, 3, 4];
        let gamma = [1i8, 1, 1, 1];
        let out = crate::spec::rmsnorm_vec(&x, &gamma, 1e-6);
        assert!(verify_norm(&x, &gamma, &out, 1e-6));
    }

    #[test]
    fn norm_rejects_wrong() {
        let x = [1, 2, 3, 4];
        let gamma = [1i8, 1, 1, 1];
        let mut out = crate::spec::rmsnorm_vec(&x, &gamma, 1e-6);
        out[0] += 1;
        assert!(!verify_norm(&x, &gamma, &out, 1e-6));
    }

    #[test]
    fn causal_softmax_row_honest() {
        let scores = [3, 1, 5, 2];
        let l = 2; // causal: only t <= 2
        let m = *scores[..=l].iter().max().unwrap(); // 5
        let mut e = vec![0i64; 4];
        for t in 0..=l {
            e[t] = exp_q15(scores[t] - m);
        }
        let s: i64 = e.iter().sum();
        let mut probs = vec![0i64; 4];
        for t in 0..=l {
            probs[t] = e[t] * 32768 / s;
        }
        assert!(verify_causal_softmax_row(
            &scores,
            &probs,
            l,
            -100,
            a(),
            b()
        ));
    }

    #[test]
    fn causal_softmax_row_rejects_wrong() {
        let scores = [3, 1, 5, 2];
        let l = 2;
        let m = *scores[..=l].iter().max().unwrap();
        let mut e = vec![0i64; 4];
        for t in 0..=l {
            e[t] = exp_q15(scores[t] - m);
        }
        let s: i64 = e.iter().sum();
        let mut probs = vec![0i64; 4];
        for t in 0..=l {
            probs[t] = e[t] * 32768 / s;
        }
        probs[1] += 1; // tamper
        assert!(!verify_causal_softmax_row(
            &scores,
            &probs,
            l,
            -100,
            a(),
            b()
        ));
    }

    #[test]
    fn causal_softmax_row_rejects_unmasked() {
        let scores = [3, 1, 5, 2];
        let l = 1; // only t <= 1
        let m = *scores[..=l].iter().max().unwrap();
        let mut e = vec![0i64; 4];
        for t in 0..=l {
            e[t] = exp_q15(scores[t] - m);
        }
        let s: i64 = e.iter().sum();
        let mut probs = vec![0i64; 4];
        for t in 0..=l {
            probs[t] = e[t] * 32768 / s;
        }
        probs[3] = 1; // t=3 > l should be 0
        assert!(!verify_causal_softmax_row(
            &scores,
            &probs,
            l,
            -100,
            a(),
            b()
        ));
    }
}
