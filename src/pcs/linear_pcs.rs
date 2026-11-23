//! Linear-code PCS following **Ligero** (§4.1 `Test-Interleaved`, §4.2
//! `Test-Linear-Constraints-IRS`), over a **roots-of-unity** RS domain so the
//! encoder / interpolator / product are NTT (`O(N log)`).
//!
//! Message `v ∈ F^{n_rows · msg_len}` → `n_rows` rows of `msg_len` values at the
//! `msg_len`-th roots of unity (`ζ_c = ω_msg^c`). Each row is RS-encoded: its
//! degree-`< msg_len` polynomial is evaluated at the `code_len`-th roots
//! (`η_j = ω_code^j`, `code_len = blowup·msg_len`), giving codeword `U`
//! (`n_rows × code_len`, and `ζ_c = η_{c·stride}` so the message is a subset).
//! `U` is Merkle-committed **column-wise**.
//!
//! Linear claim `⟨h, v⟩ = b`:
//!  - **Proximity:** `r ← F^{n_rows}`, `w = rᵀU`; check `w ∈ L` (degree `< msg_len`)
//!    and `w_j = Σ_i r_i U_ij` on `t` sampled columns.
//!  - **Linear:** `r_i(·)` = interpolant of `h`'s row `i`, `q = Σ_i r_i·p_i`
//!    (degree `< 2·msg_len−1`); check (a) `Σ_c q(ζ_c)=b` — which is
//!    `msg_len·(q_0 + q_{msg_len}) = b` for roots of unity — and
//!    (b) `Σ_i r_i(η_j) U_ij = q(η_j)` on sampled columns.
//!
//! Verifier compute is `O(N)` (the `r_i(η_j)` evaluations / sampled-column work);
//! the `O(√N)` win is proof size, not verifier compute.

use crate::field::Fp;
use crate::pcs::blake3_mmcs::{Blake3Compress, Blake3Hasher};
use p3_blake3::Blake3;
use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Dimensions;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::CryptographicHasher;

type Val = Fp;
const P: u64 = 0xFFFF_FFFF_0000_0001;

pub type RowMmcs = MerkleTreeMmcs<Val, Val, Blake3Hasher, Blake3Compress, 2, 2>;
pub type Commitment = <RowMmcs as Mmcs<Val>>::Commitment;
pub type ProverData = <RowMmcs as Mmcs<Val>>::ProverData<RowMajorMatrix<Val>>;
pub type ColOpening = BatchOpening<Val, RowMmcs>;

pub const NUM_QUERIES: usize = 64;

fn challenge_bytes(parts: &[&[u8]]) -> [u8; 32] {
    let mut all = Vec::new();
    for p in parts {
        all.extend_from_slice(p);
    }
    Blake3.hash_slice(&all)
}

fn derive_u64(seed: &[u8; 32], idx: usize) -> u64 {
    let mut h = Vec::with_capacity(40);
    h.extend_from_slice(seed);
    h.extend_from_slice(&(idx as u64).to_le_bytes());
    u64::from_le_bytes(Blake3.hash_slice(&h)[0..8].try_into().unwrap())
}

fn fp_bytes(x: Val) -> [u8; 8] {
    use p3_field::PrimeField64;
    x.as_canonical_u64().to_le_bytes()
}

fn val_of(x: u64) -> Val {
    Val::new(x % P)
}

fn bit_reverse(mut x: usize, log_n: usize) -> usize {
    let mut r = 0;
    for _ in 0..log_n {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    r
}

/// In-place natural-order radix-2 NTT (bit-reversed input, natural output).
fn ntt_inplace(a: &mut [Val], log_n: usize, inverse: bool) {
    let n = 1usize << log_n;
    for i in 0..n {
        let j = bit_reverse(i, log_n);
        if i < j {
            a.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let log_len = len.trailing_zeros() as usize;
        let mut w_len = Val::two_adic_generator(log_len);
        if inverse {
            w_len = w_len.inverse();
        }
        let half = len / 2;
        let mut base = 0;
        while base < n {
            let mut w = Val::ONE;
            for j in 0..half {
                let u = a[base + j];
                let v = a[base + j + half] * w;
                a[base + j] = u + v;
                a[base + j + half] = u - v;
                w *= w_len;
            }
            base += len;
        }
        len <<= 1;
    }
    if inverse {
        let n_inv = Val::new(n as u64).inverse();
        for x in a.iter_mut() {
            *x *= n_inv;
        }
    }
}

fn ntt(coeffs: &[Val], log_n: usize) -> Vec<Val> {
    let mut a = coeffs.to_vec();
    a.resize(1 << log_n, Val::ZERO);
    ntt_inplace(&mut a, log_n, false);
    a
}

fn intt(evals: &[Val], log_n: usize) -> Vec<Val> {
    let mut a = evals.to_vec();
    a.resize(1 << log_n, Val::ZERO);
    ntt_inplace(&mut a, log_n, true);
    a
}

fn eval_poly(c: &[Val], x: Val) -> Val {
    let mut acc = Val::ZERO;
    for &coef in c.iter().rev() {
        acc = acc * x + coef;
    }
    acc
}

#[derive(Clone)]
pub struct LinearProof {
    pub w: Vec<Val>,
    pub q: Vec<Val>,
    pub queries: Vec<(usize, ColOpening)>,
}

#[derive(Clone)]
pub struct LinearPcs {
    mmcs: RowMmcs,
    n_rows: usize,
    msg_len: usize,
    code_len: usize,
    log_msg: usize,
    log_code: usize,
}

impl LinearPcs {
    pub fn new(n: usize, blowup: usize) -> Self {
        assert!(n >= 1 && blowup.is_power_of_two());
        let n_rows = (n as f64).sqrt().ceil() as usize;
        let msg_len = n.div_ceil(n_rows).next_power_of_two();
        let code_len = (msg_len * blowup).next_power_of_two();
        assert!(code_len > msg_len);
        let log_msg = msg_len.trailing_zeros() as usize;
        let log_code = code_len.trailing_zeros() as usize;
        // Goldilocks has 2-adicity 32.
        assert!(log_code <= 32, "code domain too large for Goldilocks");
        Self {
            mmcs: RowMmcs::new(Blake3Hasher, Blake3Compress, 0),
            n_rows,
            msg_len,
            code_len,
            log_msg,
            log_code,
        }
    }

    pub fn shape(&self) -> (usize, usize, usize) {
        (self.n_rows, self.msg_len, self.code_len)
    }

    fn omega_code(&self) -> Val {
        Val::two_adic_generator(self.log_code)
    }

    /// `n_rows × code_len` codeword, row-major: message → INTT → pad → NTT.
    pub fn encode(&self, v: &[Val]) -> Vec<Val> {
        let mut out = Vec::with_capacity(self.n_rows * self.code_len);
        for i in 0..self.n_rows {
            let lo = i * self.msg_len;
            let hi = (lo + self.msg_len).min(v.len());
            let mut msg = vec![Val::ZERO; self.msg_len];
            if lo < hi {
                msg[..hi - lo].copy_from_slice(&v[lo..hi]);
            }
            let coeffs = intt(&msg, self.log_msg);
            out.extend(ntt(&coeffs, self.log_code));
        }
        out
    }

    pub fn commit(&self, v: &[Val]) -> (Commitment, ProverData) {
        let w = self.encode(v);
        let mut col_major = Vec::with_capacity(self.n_rows * self.code_len);
        for j in 0..self.code_len {
            for i in 0..self.n_rows {
                col_major.push(w[i * self.code_len + j]);
            }
        }
        self.mmcs
            .commit(vec![RowMajorMatrix::new(col_major, self.n_rows)])
    }

    pub fn dimensions(&self) -> Dimensions {
        Dimensions {
            height: self.code_len,
            width: self.n_rows,
        }
    }

    fn open_column(&self, j: usize, pd: &ProverData) -> ColOpening {
        self.mmcs.open_batch(j, pd)
    }

    fn verify_column(&self, com: &Commitment, j: usize, op: &ColOpening) -> Option<Vec<Val>> {
        self.mmcs
            .verify_batch(
                com,
                &[self.dimensions()],
                j,
                BatchOpeningRef::new(&op.opened_values, &op.opening_proof),
            )
            .is_ok()
            .then(|| op.opened_values[0].clone())
    }

    fn com_bytes(com: &Commitment) -> Vec<u8> {
        bincode::serialize(com).unwrap()
    }

    fn proximity_weights(&self, com: &Commitment) -> Vec<Val> {
        let seed = challenge_bytes(&[&Self::com_bytes(com), b"prox"]);
        (0..self.n_rows)
            .map(|i| val_of(derive_u64(&seed, i)))
            .collect()
    }

    fn query_indices(&self, com: &Commitment, w: &[Val], q: &[Val]) -> Vec<usize> {
        let cb = Self::com_bytes(com);
        let mut wb = Vec::new();
        for x in w {
            wb.extend_from_slice(&fp_bytes(*x));
        }
        let mut qb = Vec::new();
        for x in q {
            qb.extend_from_slice(&fp_bytes(*x));
        }
        let seed = challenge_bytes(&[&cb, b"q", &wb, &qb]);
        (0..NUM_QUERIES)
            .map(|t| (derive_u64(&seed, t) as usize) % self.code_len)
            .collect()
    }

    fn r_i_coeffs(&self, h: &[Val]) -> Vec<Vec<Val>> {
        (0..self.n_rows)
            .map(|i| intt(&h[i * self.msg_len..(i + 1) * self.msg_len], self.log_msg))
            .collect()
    }

    pub fn prove_linear(&self, v: &[Val], h: &[Val]) -> LinearProof {
        assert_eq!(h.len(), self.n_rows * self.msg_len);
        let u = self.encode(v);
        let (com, pd) = self.commit(v);

        let rp = self.proximity_weights(&com);
        let mut w = vec![Val::ZERO; self.code_len];
        for i in 0..self.n_rows {
            for j in 0..self.code_len {
                w[j] += rp[i] * u[i * self.code_len + j];
            }
        }

        // q(η_j) = Σ_i r_i(η_j)·U_ij; then q's coefficients via INTT.
        let rcoeff = self.r_i_coeffs(h);
        let mut sum = vec![Val::ZERO; self.code_len];
        for i in 0..self.n_rows {
            let r_evals = ntt(&rcoeff[i], self.log_code);
            for j in 0..self.code_len {
                sum[j] += r_evals[j] * u[i * self.code_len + j];
            }
        }
        let qall = intt(&sum, self.log_code);
        let q = qall[..2 * self.msg_len - 1].to_vec();

        let queries = self
            .query_indices(&com, &w, &q)
            .into_iter()
            .map(|j| (j, self.open_column(j, &pd)))
            .collect();
        LinearProof { w, q, queries }
    }

    pub fn verify_linear(&self, com: &Commitment, h: &[Val], b: Val, proof: &LinearProof) -> bool {
        assert_eq!(h.len(), self.n_rows * self.msg_len);
        if proof.w.len() != self.code_len || proof.q.len() != 2 * self.msg_len - 1 {
            return false;
        }
        // Proximity: w must be a codeword of degree < msg_len.
        let wcoeff = intt(&proof.w, self.log_code);
        if wcoeff[self.msg_len..].iter().any(|&c| c != Val::ZERO) {
            return false;
        }
        // (a) Σ_c q(ζ_c) = msg_len·(q_0 + q_{msg_len}) = b.
        if Val::new(self.msg_len as u64) * (proof.q[0] + proof.q[self.msg_len]) != b {
            return false;
        }

        let rp = self.proximity_weights(com);
        let rcoeff = self.r_i_coeffs(h);
        let omega = self.omega_code();
        let expected = self.query_indices(com, &proof.w, &proof.q);
        if proof.queries.len() != NUM_QUERIES {
            return false;
        }
        for (t, (j, op)) in proof.queries.iter().enumerate() {
            if *j != expected[t] {
                return false;
            }
            let Some(col) = self.verify_column(com, *j, op) else {
                return false;
            };
            if col.len() != self.n_rows {
                return false;
            }
            let xj = omega.exp_u64(*j as u64);
            // proximity consistency: Σ_i r_i U_ij = w_j
            let ws: Val = (0..self.n_rows).map(|i| rp[i] * col[i]).sum();
            if ws != proof.w[*j] {
                return false;
            }
            // (b) Σ_i r_i(η_j) U_ij = q(η_j)
            let lhs: Val = (0..self.n_rows)
                .map(|i| eval_poly(&rcoeff[i], xj) * col[i])
                .sum();
            if lhs != eval_poly(&proof.q, xj) {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    fn rand_vec(rng: &mut StdRng, n: usize) -> Vec<Val> {
        (0..n).map(|_| val_of(rng.random::<u64>())).collect()
    }
    fn msg_len(pcs: &LinearPcs) -> usize {
        let (r, m, _) = pcs.shape();
        r * m
    }
    fn inner(a: &[Val], b: &[Val]) -> Val {
        a.iter().zip(b).map(|(&x, &y)| x * y).sum()
    }

    #[test]
    fn ntt_roundtrip() {
        let mut rng = StdRng::seed_from_u64(0);
        for log in 3..10 {
            let n = 1usize << log;
            let v = rand_vec(&mut rng, n);
            assert_eq!(intt(&ntt(&v, log), log), v, "ntt/intt roundtrip log={log}");
        }
    }

    #[test]
    fn encode_matches_message_at_roots() {
        let pcs = LinearPcs::new(64, 4);
        let mut rng = StdRng::seed_from_u64(1);
        let v = rand_vec(&mut rng, msg_len(&pcs));
        let u = pcs.encode(&v);
        let (n_rows, m, code_len) = pcs.shape();
        let stride = code_len / m;
        for i in 0..n_rows {
            for c in 0..m {
                assert_eq!(
                    u[i * code_len + c * stride],
                    v[i * m + c],
                    "ζ_c = η_{{c·stride}}"
                );
            }
        }
    }

    #[test]
    fn linear_claim_honest() {
        let pcs = LinearPcs::new(64, 4);
        let mut rng = StdRng::seed_from_u64(2);
        let v = rand_vec(&mut rng, msg_len(&pcs));
        let h = rand_vec(&mut rng, msg_len(&pcs));
        let b = inner(&h, &v);
        let (com, _pd) = pcs.commit(&v);
        let proof = pcs.prove_linear(&v, &h);
        assert!(pcs.verify_linear(&com, &h, b, &proof));
    }

    #[test]
    fn linear_claim_wrong_b_rejected() {
        let pcs = LinearPcs::new(64, 4);
        let mut rng = StdRng::seed_from_u64(3);
        let v = rand_vec(&mut rng, msg_len(&pcs));
        let h = rand_vec(&mut rng, msg_len(&pcs));
        let b = inner(&h, &v) + Val::ONE;
        let (com, _pd) = pcs.commit(&v);
        let proof = pcs.prove_linear(&v, &h);
        assert!(!pcs.verify_linear(&com, &h, b, &proof));
    }

    #[test]
    fn linear_claim_wrong_v_rejected() {
        let pcs = LinearPcs::new(64, 4);
        let mut rng = StdRng::seed_from_u64(4);
        let v = rand_vec(&mut rng, msg_len(&pcs));
        let other = rand_vec(&mut rng, msg_len(&pcs));
        let h = rand_vec(&mut rng, msg_len(&pcs));
        let b = inner(&h, &v);
        let (com, _pd) = pcs.commit(&other);
        let proof = pcs.prove_linear(&v, &h);
        assert!(!pcs.verify_linear(&com, &h, b, &proof));
    }

    #[test]
    fn tampered_w_rejected() {
        let pcs = LinearPcs::new(64, 4);
        let mut rng = StdRng::seed_from_u64(5);
        let v = rand_vec(&mut rng, msg_len(&pcs));
        let h = rand_vec(&mut rng, msg_len(&pcs));
        let b = inner(&h, &v);
        let (com, _pd) = pcs.commit(&v);
        let mut proof = pcs.prove_linear(&v, &h);
        proof.w[0] += Val::ONE;
        assert!(!pcs.verify_linear(&com, &h, b, &proof));
    }

    #[test]
    fn tampered_q_rejected() {
        let pcs = LinearPcs::new(64, 4);
        let mut rng = StdRng::seed_from_u64(6);
        let v = rand_vec(&mut rng, msg_len(&pcs));
        let h = rand_vec(&mut rng, msg_len(&pcs));
        let b = inner(&h, &v);
        let (com, _pd) = pcs.commit(&v);
        let mut proof = pcs.prove_linear(&v, &h);
        proof.q[1] += Val::ONE;
        assert!(!pcs.verify_linear(&com, &h, b, &proof));
    }
}
