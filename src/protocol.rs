//! The three IOPs (key / link / req) and the compiled verifier V_Z
//! (paper Definition 4.6, Definition 4.7, Construction 4.15).

use crate::encoding::{encode, RowParams};
use crate::field::{Ext, Fp};
use crate::merkle::{commit as merkle_commit, Digest, Salt, Table};
use crate::model::{admissible, build, DigitMatrix, Model, Policy, Trace, Weights};
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField64};
use sha2::{Digest as _, Sha256};

// --- Out-of-domain samples (Definition 4.3) ---------------------------------

/// ⌈log2 n⌉ — the Boolean-hypercube dimension of a length-`n` message.
fn log2_ceil(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    (usize::BITS - (n - 1).leading_zeros()) as usize
}

/// The equality polynomial `eq(z, j) = ∏_k (z_k·j_k + (1−z_k)(1−j_k))`
/// (Definition 4.3), with `j_k` the k-th bit of `j`, lowest first.
///
/// It is the multilinear-extension basis: `eq(z, j) = 1` when `z` equals the
/// Boolean point of `j`, and `0` at every other Boolean point.
pub fn eq(z: &[Ext], j: usize) -> Ext {
    let mut acc = Ext::ONE;
    for (k, &zk) in z.iter().enumerate() {
        let bit = Ext::from(Fp::new(((j >> k) & 1) as u64));
        // bit=1 → zk; bit=0 → 1−zk.
        let factor = zk * bit + (Ext::ONE - zk) * (Ext::ONE - bit);
        acc = acc * factor;
    }
    acc
}

/// An out-of-domain seed: `s` points `z_i ∈ E^⌈log2 ℓ⌉` and factors `β_i ∈ E*`
/// (Definition 4.3).
#[derive(serde::Serialize, serde::Deserialize)]
pub struct OodSeed {
    /// `z_i`, each a vector of length ⌈log2 ℓ⌉.
    pub points: Vec<Vec<Ext>>,
    /// `β_i`, each nonzero.
    pub betas: Vec<Ext>,
}

impl OodSeed {
    /// The number `s` of samples.
    pub fn num_samples(&self) -> usize {
        self.points.len()
    }
}

/// Out-of-domain answers `a = (a_{r,i})_{r<2^v, i<s}` (Definition 4.3),
/// row-major.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct OodAnswers {
    pub rows: usize,
    pub s: usize,
    /// `a[row * s + i]`
    pub a: Vec<Ext>,
}

impl OodAnswers {
    pub fn get(&self, row: usize, i: usize) -> Ext {
        self.a[row * self.s + i]
    }
}

/// One OOD answer `a_{r,i} = Σ_j eq(z_i, j)·f[j] + β_i·pad_i` (Definition 4.3):
/// the multilinear extension of `f` at `z_i`, plus a masked pad.
///
/// `msg` is the row message `f_r ∈ F^ℓ`; `pad` is `pad_{r,i}`.
///
/// NOTE: the paper places the pads at the row message's last `3s` entries
/// (Definition 4.2), but with `s = 1` there are 3 pads and 1 answer. Which pad
/// index the single answer uses was an image lost in the markdown — verify
/// against the release spec. Here the caller supplies the pad explicitly, so the
/// ambiguity is pushed to the call site, not baked into this function.
pub fn ood_answer(z: &[Ext], beta: &Ext, msg: &[Fp], pad: Fp) -> Ext {
    let mut acc = Ext::ZERO;
    for (j, &fj) in msg.iter().enumerate() {
        acc = acc + eq(z, j) * Ext::from(fj);
    }
    acc + *beta * Ext::from(pad)
}

/// Compute all OOD answers for a table's row messages (Definition 4.3).
///
/// `msgs[r]` is `f_r ∈ F^ℓ`; `pads[r]` is the row's pad vector (length `3s`,
/// entry `i` is `pad_{r,i}`).
pub fn ood_answers(seed: &OodSeed, msgs: &[Vec<Fp>], pads: &[Vec<Fp>]) -> OodAnswers {
    let rows = msgs.len();
    let s = seed.num_samples();
    assert_eq!(pads.len(), rows, "one pad vector per row");
    let mut a = Vec::with_capacity(rows * s);
    for r in 0..rows {
        for i in 0..s {
            a.push(ood_answer(
                &seed.points[i],
                &seed.betas[i],
                &msgs[r],
                pads[r][i],
            ));
        }
    }
    OodAnswers { rows, s, a }
}

// --- Seed derivation (Construction 4.15 step 7, Construction D.11) -----------

/// Derive an OOD seed from a Merkle root: two hash queries, then expand the
/// bytes into `s` points `z_i ∈ E^⌈log2 ℓ⌉` and nonzero factors `β_i ∈ E*`.
pub fn derive_ood_seed(
    rt: &Digest,
    dg_z: &Digest,
    sizes: &[u8],
    log_l: usize,
    s: usize,
) -> OodSeed {
    let mut h1 = Sha256::new();
    h1.update(b"ood-query-1");
    h1.update(rt);
    h1.update(dg_z);
    h1.update(sizes);
    let chain: [u8; 32] = h1.finalize().into();

    let mut h2 = Sha256::new();
    h2.update(b"ood-query-2");
    h2.update(chain);
    let seed: [u8; 32] = h2.finalize().into();

    // Each point is 3·log_l field elements, each factor 3.
    let elems = expand_seed(&seed, s * (3 * log_l + 3));

    let mut points = Vec::with_capacity(s);
    let mut betas = Vec::with_capacity(s);
    let mut idx = 0;
    for _ in 0..s {
        let mut z = Vec::with_capacity(log_l);
        for _ in 0..log_l {
            z.push(Ext::new([elems[idx], elems[idx + 1], elems[idx + 2]]));
            idx += 3;
        }
        let beta = Ext::new([elems[idx], elems[idx + 1], elems[idx + 2]]);
        idx += 3;
        let beta = if beta == Ext::ZERO { Ext::ONE } else { beta };
        points.push(z);
        betas.push(beta);
    }
    OodSeed { points, betas }
}

/// Expand a 32-byte seed into `n` base-field elements (counter-mode hashing).
fn expand_seed(seed: &[u8; 32], n: usize) -> Vec<Fp> {
    let mut out = Vec::with_capacity(n);
    let mut counter = 0u64;
    while out.len() < n {
        let mut h = Sha256::new();
        h.update(seed);
        h.update(counter.to_le_bytes());
        let d: [u8; 32] = h.finalize().into();
        for chunk in d.chunks(8) {
            let v = u64::from_le_bytes(chunk.try_into().unwrap());
            out.push(Fp::new(v % Fp::ORDER_U64));
            if out.len() >= n {
                break;
            }
        }
        counter += 1;
    }
    out
}

// --- Decoder (Definition 4.5) ------------------------------------------------

/// The full layout of a committed table (Definition 4.2, Table 2).
///
/// The digit matrix `A` has `d` rows × `2^n` columns, with `n = n0 + v`. It is
/// distributed across the `2^v` table rows: table row `r` holds the sub-matrix
/// `A[:, r·2^n0 .. (r+1)·2^n0]`, flattened row-major, in the first `d·2^n0`
/// entries of its row message `f_r` (a concrete instantiation of the placement
/// rule; the paper's own rule is deferred, Remark D.10, but "need not be
/// unique", Definition 4.1).
#[derive(serde::Serialize, serde::Deserialize)]
pub struct TableParams {
    pub row: RowParams, // ℓ, t, |D|
    pub d: usize,       // matrix rows (3,076,096)
    pub n0: usize,      // log2 columns per table row (7)
    pub v: usize,       // log2 table rows (11)
    pub s: usize,       // OOD answers per row (1)
}

impl TableParams {
    /// log2 of matrix columns (n = n0 + v).
    pub fn n(&self) -> usize {
        self.n0 + self.v
    }

    /// Number of table rows (2^v).
    pub fn num_table_rows(&self) -> usize {
        1 << self.v
    }

    /// Matrix columns held by one table row (2^n0).
    pub fn cols_per_row(&self) -> usize {
        1 << self.n0
    }

    /// Plain entries a row message uses for `A` (d · 2^n0).
    pub fn matrix_entries_per_row(&self) -> usize {
        self.d * (1 << self.n0)
    }

    /// ℓ_0 = ℓ − 3s (Definition 4.2).
    pub fn plain_len(&self) -> usize {
        self.row.message_len - 3 * self.s
    }

    /// Well-formed iff d·2^n0 ≤ ℓ_0 (Definition 4.2).
    pub fn is_well_formed(&self) -> bool {
        self.matrix_entries_per_row() <= self.plain_len()
    }

    /// A flat layout (n0 = v = 0) for a digit matrix of `d` entries: one table
    /// row holds the whole matrix, `s` OOD answers per row.
    pub fn flat(d: usize, s: usize) -> Self {
        let message_len = d + 3 * s;
        let num_masks = 4;
        let mut log_domain_size = 0usize;
        while (1usize << log_domain_size) < message_len + num_masks {
            log_domain_size += 1;
        }
        Self {
            row: RowParams {
                message_len,
                num_masks,
                log_domain_size,
            },
            d,
            n0: 0,
            v: 0,
            s,
        }
    }
}

/// Map a field element to a signed digit in `[-2^7, 2^7)`, or `None` if out of
/// range (Definition 4.1). Digits are stored modulo `p`: small non-negative
/// values as themselves, negative values as `p − |d|`.
fn field_to_digit(f: Fp) -> Option<i8> {
    let p = Fp::ORDER_U64 as u128;
    let v = f.as_canonical_u64() as u128;
    if v < 128 {
        Some(v as i8)
    } else if v >= p - 127 {
        Some((v as i128 - p as i128) as i8)
    } else {
        None
    }
}

/// The inverse of [`field_to_digit`]: encode a digit into the field.
fn digit_to_field(d: i8) -> Fp {
    if d >= 0 {
        Fp::new(d as u64)
    } else {
        Fp::new(Fp::ORDER_U64 - (d.unsigned_abs() as u64))
    }
}

/// The result of decoding (Definition 4.5): a digit matrix `A` in range, or ⊥.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecodeResult {
    /// The digit matrix `A` (row-major, `d × 2^n`), in range.
    Matrix(Vec<i8>),
    /// ⊥
    Bottom,
}

/// Extract the digit matrix `A` from the decoded row messages (the inverse of
/// the placement in `TableParams`), or `None` if any entry is out of range
/// (Definition 4.1, Definition 4.5 item 3).
fn extract_matrix(layout: &TableParams, msgs: &[Vec<Fp>]) -> Option<Vec<i8>> {
    let n = layout.n();
    let mut a = vec![0i8; layout.d * (1 << n)];
    for r in 0..layout.num_table_rows() {
        let col0 = r * layout.cols_per_row();
        for i in 0..layout.d {
            for c in 0..layout.cols_per_row() {
                let digit = field_to_digit(msgs[r][i * layout.cols_per_row() + c])?;
                a[i * (1 << n) + col0 + c] = digit;
            }
        }
    }
    Some(a)
}

/// Decode a word `T*` that is an *exact* table of codewords (Definition 4.5,
/// the distance-0 case the honest prover always satisfies).
///
/// Steps, mirroring Definition 4.5:
/// 1. No collision: a word over `F` at distance 0 has exactly one codeword
///    within radius `e` (the next codeword is ≥ the minimum distance, which
///    exceeds `e`), so `Coll = false`.
/// 2. `T' = T*` (it is nearest and in the list).
/// 3. Return `A` iff its entries are in range.
///
/// `table[r]` is the row-`r` codeword (length `|D|`); `pads[r]` is the row's
/// pad vector (length `3s`, entry `i` is `pad_{r,i}`, i.e. `f_r[ℓ_0 + i]`).
///
/// This is correct for completeness. The *below-Johnson list decoder* — finding
/// all codewords within radius `e` of a word that is *not* an exact codeword —
/// is the soundness-side component (Lemma 5.13 / BCHKS25) and is not implemented
/// here; see the module note.
pub fn decode_exact(
    layout: &TableParams,
    table: &[Vec<Fp>],
    seed: &OodSeed,
    answers: &OodAnswers,
    pads: &[Vec<Fp>],
) -> DecodeResult {
    if !layout.is_well_formed() {
        return DecodeResult::Bottom;
    }
    let rows = layout.num_table_rows();
    if table.len() != rows || pads.len() != rows || answers.rows != rows || answers.s != layout.s {
        return DecodeResult::Bottom;
    }
    let (t, l) = (layout.row.num_masks, layout.row.message_len);

    // 1 + 2. Verify each row is a valid codeword and read its message.
    let mut msgs = Vec::with_capacity(rows);
    for r in 0..rows {
        let coeffs = Radix2Dit::<Fp>::default().idft(table[r].clone());
        if coeffs[t + l..].iter().any(|&c| c != Fp::new(0)) {
            return DecodeResult::Bottom; // degree > t + l − 1
        }
        msgs.push(coeffs[t..t + l].to_vec());
    }

    // Verify the OOD answers (Definition 4.3).
    for r in 0..rows {
        for i in 0..layout.s {
            let expected = ood_answer(&seed.points[i], &seed.betas[i], &msgs[r], pads[r][i]);
            if expected != answers.get(r, i) {
                return DecodeResult::Bottom;
            }
        }
    }

    // 3. Extract A and check in range.
    match extract_matrix(layout, &msgs) {
        Some(a) => DecodeResult::Matrix(a),
        None => DecodeResult::Bottom,
    }
}

/// Collision test (Definition 4.5). In the exact-codeword regime this is always
/// `false` (see `decode_exact`); the list-decoding collision check is deferred.
pub fn collision(_layout: &TableParams, _table: &[Vec<Fp>], _seed: &OodSeed) -> bool {
    false
}

/// Build the row messages from a digit matrix `A` and pads (the committer's
/// placement, the inverse of `extract_matrix`, Definition 4.2). Each message
/// `f_r ∈ F^ℓ` holds `A`'s slice in the first `d·2^n0` entries, zeros up to
/// `ℓ_0`, and the row's pads in the last `3s` entries.
pub fn place(layout: &TableParams, a: &[i8], pads: &[Vec<Fp>]) -> Vec<Vec<Fp>> {
    assert!(layout.is_well_formed());
    let n = layout.n();
    assert_eq!(a.len(), layout.d * (1 << n), "matrix length");
    assert_eq!(
        pads.len(),
        layout.num_table_rows(),
        "one pad vector per row"
    );

    let mut msgs = Vec::with_capacity(layout.num_table_rows());
    for r in 0..layout.num_table_rows() {
        let mut f = vec![Fp::new(0); layout.row.message_len];
        let col0 = r * layout.cols_per_row();
        for i in 0..layout.d {
            for c in 0..layout.cols_per_row() {
                f[i * layout.cols_per_row() + c] = digit_to_field(a[i * (1 << n) + col0 + c]);
            }
        }
        // pads at [ℓ_0, ℓ_0 + 3s)
        for (k, &p) in pads[r].iter().enumerate() {
            f[layout.plain_len() + k] = p;
        }
        msgs.push(f);
    }
    msgs
}
// --- The argument Π_Z and verifier V_Z (Construction 4.15) -------------------

/// A key `k = (rt0, σ0)` (Construction 4.15).
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Key {
    pub rt0: Digest,
    pub seed0: OodSeed,
    pub answers0: OodAnswers,
}

/// A copy `j` and its material (Definition 4.6).
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Copy {
    pub index: usize,
    pub rtj: Digest,
    pub seedj: OodSeed,
    pub answersj: OodAnswers,
}

/// A key proof (non-succinct): the committed table and the material to verify
/// it (leaf-salt seed and pads).
#[derive(serde::Serialize, serde::Deserialize)]
pub struct KeyProof {
    pub table: Table,
    pub salt_seed: Salt,
    #[serde(with = "crate::field::fp_vec_vec")]
    pub pads: Vec<Vec<Fp>>, // 3s per row
}

/// A copy proof (non-succinct): the copy's table and its verification material.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct CopyProof {
    pub table: Table,
    pub salt_seed: Salt,
    #[serde(with = "crate::field::fp_vec_vec")]
    pub pads: Vec<Vec<Fp>>,
}

/// A request proof (non-succinct): the inference trace. The digit matrix `A` is
/// recovered by decoding the key, not shipped here.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ReqProof {
    pub trace: Trace,
}

/// A proof `π = (fill, π_req)` (Construction 4.15). The per-segment key/copy and
/// their proofs live in the commit files (each segment is committed and shipped
/// separately), so the request proof carries only the trace + fill.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Proof {
    pub fill: Vec<usize>,
    pub pi_req: Option<ReqProof>,
}

/// Flatten a table's row-major data into per-row vectors.
fn table_rows(table: &Table) -> Vec<Vec<Fp>> {
    table.data.chunks(table.cols).map(|r| r.to_vec()).collect()
}

/// The verifier `V_Z` (Construction 4.15), non-succinct and exactly sound.
///
/// Every check is exact: the Merkle root binds the shipped table to the
/// committed root, `decode_exact` verifies the codeword + OOD answers + range,
/// the copy decodes to the *same* matrix, and the arithmetization checks the
/// trace against that matrix. No proximity test or list decoding is needed
/// here (those are for the succinct version).
/// Verify one segment's commitment and decode its digit matrix (steps 5, 7, 8, 9
/// of `V_Z`). Returns `None` if any check fails.
pub fn verify_segment(
    layout: &TableParams,
    dg_z: &Digest,
    co: &CommitOutput,
) -> Option<DigitMatrix> {
    // 5. H(code(k, dg_Z)) == c.
    if commit_key(&co.key.rt0, &co.key.seed0, dg_z) != co.c {
        return None;
    }
    // 7. σ0 = Seed(rt0), σj = Seed(rtj).
    let sizes = table_sizes_bytes(layout);
    let log_l = log2_ceil(layout.row.message_len);
    let seed0 = derive_ood_seed(&co.key.rt0, dg_z, &sizes, log_l, layout.s);
    if seed0.points != co.key.seed0.points || seed0.betas != co.key.seed0.betas {
        return None;
    }
    let seedj = derive_ood_seed(&co.copy.rtj, dg_z, &sizes, log_l, layout.s);
    if seedj.points != co.copy.seedj.points || seedj.betas != co.copy.seedj.betas {
        return None;
    }

    // 8. Key: Merkle root binds the table; decode_exact verifies the codeword,
    //    OOD answers, and range, yielding this segment's A.
    if merkle_commit(&co.pi_key.table, &co.pi_key.salt_seed).0 != co.key.rt0 {
        return None;
    }
    let a_s = match decode_exact(
        layout,
        &table_rows(&co.pi_key.table),
        &co.key.seed0,
        &co.key.answers0,
        &co.pi_key.pads,
    ) {
        DecodeResult::Matrix(entries) => DigitMatrix::new(layout.d, layout.n(), entries),
        DecodeResult::Bottom => return None,
    };

    // 8. Link: the copy decodes to the same matrix A.
    if merkle_commit(&co.pi_link.table, &co.pi_link.salt_seed).0 != co.copy.rtj {
        return None;
    }
    match decode_exact(
        layout,
        &table_rows(&co.pi_link.table),
        &co.copy.seedj,
        &co.copy.answersj,
        &co.pi_link.pads,
    ) {
        DecodeResult::Matrix(entries) => {
            if entries != a_s.entries {
                return None;
            }
        }
        DecodeResult::Bottom => return None,
    }

    // 9. The copy names itself (its index is the copy number).
    Some(a_s)
}

/// The verifier `V_Z` (Construction 4.15), non-succinct and exactly sound, taking
/// the already-decoded per-segment digit matrices (the caller decodes each
/// segment via [`verify_segment`] to keep peak memory bounded).
pub fn verify<M: Model + ?Sized>(
    m: &M,
    cfg: &Config,
    x: &[usize],
    y: &[usize],
    proof: &Proof,
    matrices: &[DigitMatrix],
) -> bool {
    // 1. Scope test (Definition 3.8).
    if !admissible(&cfg.policy) {
        return false;
    }
    if matrices.len() != cfg.segments.len() || proof.pi_req.is_none() {
        return false;
    }
    // 4. build(x, y, fill, L) must be defined (Definition 4.7); L is read from
    //    the trace's first cell (the embedding, whose row count is the window
    //    length), so the prover and verifier agree on L without a separate field.
    let window_len = match proof
        .pi_req
        .as_ref()
        .and_then(|r| r.trace.segments.first())
        .and_then(|s| s.cells.first())
    {
        Some(c) => c.rows,
        None => return false,
    };
    let xi = match build(x, y, &proof.fill, window_len) {
        Some(xi) => xi,
        None => return false,
    };

    // 8. Req: the trace is an honest forward pass of A, and the greedy token
    //    matches the request's output.
    let rp = proof.pi_req.as_ref().unwrap();
    if !m.check(matrices, &xi, &rp.trace) {
        return false;
    }
    let head = rp.trace.segments.last().and_then(|s| s.head_token);
    let expected = if xi.i1 < xi.window.len() {
        Some(xi.window[xi.i1])
    } else {
        Some(xi.y_plus)
    };
    head == expected
}

/// Serialize the table sizes for the seed derivation (step 7).
fn table_sizes_bytes(layout: &TableParams) -> Vec<u8> {
    let mut b = Vec::new();
    for v in [
        layout.d as u64,
        layout.n0 as u64,
        layout.v as u64,
        layout.s as u64,
        layout.row.message_len as u64,
        layout.row.num_masks as u64,
        layout.row.log_domain_size as u64,
    ] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

/// Compute `c = H(code(k, dg_Z))` (Construction 4.15 step 5). `code` is deferred
/// (Remark D.10); here it is a simple injective byte encoding (root ‖ seed points
/// and betas ‖ dg_Z) hashed with SHA-256 as the random oracle.
pub fn commit_key(rt0: &Digest, seed0: &OodSeed, dg_z: &Digest) -> Digest {
    let mut h = Sha256::new();
    h.update(rt0);
    for (z, beta) in seed0.points.iter().zip(&seed0.betas) {
        for zk in z {
            for part in <Ext as BasedVectorSpace<Fp>>::as_basis_coefficients_slice(zk) {
                h.update(part.as_canonical_u64().to_le_bytes());
            }
        }
        for part in <Ext as BasedVectorSpace<Fp>>::as_basis_coefficients_slice(beta) {
            h.update(part.as_canonical_u64().to_le_bytes());
        }
    }
    h.update(dg_z);
    h.finalize().into()
}

/// The prover's coins for one committed table (Definition 5.16): the leaf-salt
/// seed, the masking coefficients τ (one per row), and the pads (3s per row).
pub struct Coins {
    pub salt_seed: Salt,
    pub masks: Vec<Vec<Fp>>, // num_table_rows × t
    pub pads: Vec<Vec<Fp>>,  // num_table_rows × 3s
}

/// Generate deterministic coins from a secret seed (the coin map Γ,
/// Definition 5.16, instantiated with a hash expansion).
pub fn gen_coins(layout: &TableParams, seed: &[u8; 32]) -> Coins {
    let rows = layout.num_table_rows();
    let t = layout.row.num_masks;
    let s3 = 3 * layout.s;
    let elems = expand_seed(seed, rows * (t + s3));
    let mut it = elems.into_iter();
    let mut masks = Vec::with_capacity(rows);
    let mut pads = Vec::with_capacity(rows);
    for _ in 0..rows {
        masks.push(it.by_ref().take(t).collect());
        pads.push(it.by_ref().take(s3).collect());
    }
    let mut h = Sha256::new();
    h.update(b"coins-salt");
    h.update(seed);
    let salt_seed: Salt = h.finalize().into();
    Coins {
        salt_seed,
        masks,
        pads,
    }
}

/// Encode the row messages into a committed table.
fn encode_table(layout: &TableParams, msgs: &[Vec<Fp>], masks: &[Vec<Fp>]) -> Table {
    let mut data = Vec::with_capacity(layout.num_table_rows() * layout.row.domain_size());
    for r in 0..layout.num_table_rows() {
        let codeword = encode(&layout.row, &msgs[r], &masks[r]);
        data.extend_from_slice(&codeword);
    }
    Table::new(layout.num_table_rows(), layout.row.domain_size(), data)
}

/// The output of committing the weights (Construction 4.15, D.12): the
/// commitment `c`, the key, one copy, and their proofs.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct CommitOutput {
    pub c: Digest,
    pub key: Key,
    pub copy: Copy,
    pub pi_key: KeyProof,
    pub pi_link: CopyProof,
}

/// The protocol configuration shared by the committer and verifier: the
/// per-segment table layouts, the acceptance policy, and the configuration
/// digest `dg_Z`.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Config {
    pub segments: Vec<SegmentConfig>,
    pub policy: Policy,
    pub dg: Digest,
}

/// One segment's table layout.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct SegmentConfig {
    pub layout: TableParams,
}

/// The committer `Commit_Z` (Construction D.12): place the weights, mask-encode,
/// Merkle-commit, and produce the OOD sample, for the first table and one copy.
pub fn commit<M: Model + ?Sized>(
    m: &M,
    layout: &TableParams,
    weights: &Weights,
    dg_z: &Digest,
    coins0: &Coins,
    coinsj: &Coins,
    segment: usize,
) -> CommitOutput {
    let a = m.to_digit_matrix(weights, segment);

    // First table T0.
    let msgs0 = place(layout, &a.entries, &coins0.pads);
    let table0 = encode_table(layout, &msgs0, &coins0.masks);
    let (rt0, _) = merkle_commit(&table0, &coins0.salt_seed);
    let sizes = table_sizes_bytes(layout);
    let log_l = log2_ceil(layout.row.message_len);
    let seed0 = derive_ood_seed(&rt0, dg_z, &sizes, log_l, layout.s);
    let answers0 = ood_answers(&seed0, &msgs0, &coins0.pads);
    drop(msgs0); // free the large message before building the copy

    // Copy Tj (same matrix, fresh coins).
    let msgsj = place(layout, &a.entries, &coinsj.pads);
    let tablej = encode_table(layout, &msgsj, &coinsj.masks);
    let (rtj, _) = merkle_commit(&tablej, &coinsj.salt_seed);
    let seedj = derive_ood_seed(&rtj, dg_z, &sizes, log_l, layout.s);
    let answersj = ood_answers(&seedj, &msgsj, &coinsj.pads);
    drop(msgsj);

    let c = commit_key(&rt0, &seed0, dg_z);

    CommitOutput {
        c,
        key: Key {
            rt0,
            seed0,
            answers0,
        },
        copy: Copy {
            index: 1,
            rtj,
            seedj,
            answersj,
        },
        pi_key: KeyProof {
            table: table0,
            salt_seed: coins0.salt_seed,
            pads: coins0.pads.clone(),
        },
        pi_link: CopyProof {
            table: tablej,
            salt_seed: coinsj.salt_seed,
            pads: coinsj.pads.clone(),
        },
    }
}

/// The prover `P_Z` (Construction D.12): assemble the request proof. The commit
/// material already lives in the per-segment commit files, so this only checks
/// the request is encodable and carries the trace.
pub fn prove(trace: Trace, x: &[usize], y: &[usize], fill: &[usize]) -> Option<Proof> {
    let window_len = trace.segments.first()?.cells.first()?.rows;
    build(x, y, fill, window_len)?; // the request must be encodable (Definition 4.7)
    Some(Proof {
        fill: fill.to_vec(),
        pi_req: Some(ReqProof { trace }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eq_is_boolean_indicator() {
        let j0 = 5usize; // bits 101
        let z: Vec<Ext> = (0..3)
            .map(|k| Ext::from(Fp::new(((j0 >> k) & 1) as u64)))
            .collect();
        assert_eq!(eq(&z, j0), Ext::ONE);
        for j in 0..8 {
            if j != j0 {
                assert_eq!(eq(&z, j), Ext::ZERO);
            }
        }
    }

    #[test]
    fn ood_answer_is_mle_plus_masked_pad() {
        // ℓ = 8 (log2 = 3). Message f = [1..8], pad = 9.
        let msg: Vec<Fp> = (1..=8u64).map(Fp::new).collect();
        let pad = Fp::new(9);

        // β = 1, z = Boolean point of j0 = 5.
        let j0 = 5usize;
        let z: Vec<Ext> = (0..3)
            .map(|k| Ext::from(Fp::new(((j0 >> k) & 1) as u64)))
            .collect();
        let beta = Ext::ONE;

        // a = f[5] + β·pad = 6 + 9 = 15.
        let a = ood_answer(&z, &beta, &msg, pad);
        assert_eq!(a, Ext::from(Fp::new(15)));
    }

    #[test]
    fn field_to_digit_roundtrips() {
        assert_eq!(field_to_digit(Fp::new(5)), Some(5));
        assert_eq!(field_to_digit(Fp::new(127)), Some(127));
        assert_eq!(field_to_digit(Fp::new(200)), None); // out of range
                                                        // −3 is stored as p − 3.
        let neg3 = Fp::new(Fp::ORDER_U64 - 3);
        assert_eq!(field_to_digit(neg3), Some(-3));
        assert_eq!(digit_to_field(-3), neg3);
    }

    #[test]
    fn decode_exact_recovers_matrix() {
        // |D| = 16, ℓ = 8, t = 4, d = 2, n0 = 1, v = 1, s = 1.
        let row = RowParams {
            message_len: 8,
            num_masks: 4,
            log_domain_size: 4,
        };
        let layout = TableParams {
            row,
            d: 2,
            n0: 1,
            v: 1,
            s: 1,
        };
        assert!(layout.is_well_formed());

        // Digit matrix A (2 × 4), entries 1..8, all in range.
        let a: Vec<i8> = (1i8..=8).collect();

        // Build messages. Row 0 holds A[:, 0..2] = [1,2,5,6]; row 1 holds
        // A[:, 2..4] = [3,4,7,8]; then a 0 (padding), then 3 pads at [5..8).
        let pads = [Fp::new(101), Fp::new(102), Fp::new(103)];
        let vals = [[1i8, 2, 5, 6], [3, 4, 7, 8]];
        let mut msgs = Vec::new();
        for r in 0..2 {
            let mut f = vec![Fp::new(0); 8];
            for (k, &v) in vals[r].iter().enumerate() {
                f[k] = digit_to_field(v);
            }
            f[5] = pads[0];
            f[6] = pads[1];
            f[7] = pads[2];
            msgs.push(f);
        }

        // Encode each row with fixed masks.
        let mut table = Vec::new();
        for r in 0..2 {
            let masks: Vec<Fp> = (0..4).map(|i| Fp::new(50 + i as u64)).collect();
            table.push(crate::encoding::encode(&row, &msgs[r], &masks));
        }

        // OOD seed: s = 1, z ∈ E^3, β = 1. pad_{r,0} = pads[0] = f_r[5].
        let z = vec![
            Ext::from(Fp::new(1)),
            Ext::from(Fp::new(2)),
            Ext::from(Fp::new(3)),
        ];
        let seed = OodSeed {
            points: vec![z],
            betas: vec![Ext::ONE],
        };
        let pad_vecs = vec![vec![pads[0]], vec![pads[0]]];
        let answers = ood_answers(&seed, &msgs, &pad_vecs);

        assert_eq!(
            decode_exact(&layout, &table, &seed, &answers, &pad_vecs),
            DecodeResult::Matrix(a)
        );
    }

    #[test]
    fn decode_exact_rejects_tampered_table() {
        let row = RowParams {
            message_len: 8,
            num_masks: 4,
            log_domain_size: 4,
        };
        let layout = TableParams {
            row,
            d: 2,
            n0: 1,
            v: 1,
            s: 1,
        };
        let pads = [Fp::new(101), Fp::new(102), Fp::new(103)];
        let vals = [[1i8, 2, 5, 6], [3, 4, 7, 8]];
        let mut msgs = Vec::new();
        for r in 0..2 {
            let mut f = vec![Fp::new(0); 8];
            for (k, &v) in vals[r].iter().enumerate() {
                f[k] = digit_to_field(v);
            }
            f[5] = pads[0];
            f[6] = pads[1];
            f[7] = pads[2];
            msgs.push(f);
        }
        let mut table = Vec::new();
        for r in 0..2 {
            let masks: Vec<Fp> = (0..4).map(|i| Fp::new(50 + i as u64)).collect();
            table.push(crate::encoding::encode(&row, &msgs[r], &masks));
        }
        // Corrupt one codeword symbol.
        table[0][0] = table[0][0] + Fp::new(1);

        let z = vec![
            Ext::from(Fp::new(1)),
            Ext::from(Fp::new(2)),
            Ext::from(Fp::new(3)),
        ];
        let seed = OodSeed {
            points: vec![z],
            betas: vec![Ext::ONE],
        };
        let pad_vecs = vec![vec![pads[0]], vec![pads[0]]];
        let answers = ood_answers(&seed, &msgs, &pad_vecs);

        assert_eq!(
            decode_exact(&layout, &table, &seed, &answers, &pad_vecs),
            DecodeResult::Bottom
        );
    }

    #[test]
    fn place_and_extract_matrix_roundtrip() {
        let row = RowParams {
            message_len: 8,
            num_masks: 4,
            log_domain_size: 4,
        };
        let layout = TableParams {
            row,
            d: 2,
            n0: 1,
            v: 1,
            s: 1,
        };
        let a: Vec<i8> = (1i8..=8).collect();
        let pads = vec![vec![Fp::new(101), Fp::new(102), Fp::new(103)]; 2];
        let msgs = place(&layout, &a, &pads);
        assert_eq!(extract_matrix(&layout, &msgs), Some(a));
    }
}
