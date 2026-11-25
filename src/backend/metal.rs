//! Metal implementation of the MLE reduction.
//!
//! The kernel is compiled from source at runtime, so no `xcrun metal` toolchain is
//! needed. Goldilocks arithmetic is done with 32-bit limbs (`mulhi`), because Apple
//! GPUs are 32-bit ALUs: a 64×64→128 multiply is emulated, and a native 32×32→64 is
//! the only cheap primitive available.
//!
//! Layout: `values` is `2^m` Goldilocks elements (8 bytes each), `point` is `m`
//! extension elements as 3 consecutive Goldilocks words, `Ext` in MSL is the same
//! three words.

use std::sync::OnceLock;

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField64};

use crate::field::{Ext, Fp};

const MSL: &str = r#"
#include <metal_stdlib>

#include <metal_stdlib>
using namespace metal;

constant ulong GL_ORDER = 0xFFFFFFFF00000001ul;
constant ulong GL_EPS   = 0x00000000FFFFFFFFul;

// p = 2^64 - 2^32 + 1, so 2^64 = EPS (mod p).
static inline ulong gl_add(ulong a, ulong b) {
    ulong t = a + b;
    if (t < a) { t += GL_EPS; }
    else if (t >= GL_ORDER) { t -= GL_ORDER; }
    return t;
}

static inline ulong gl_mul(ulong a, ulong b) {
    uint a0 = (uint)a, a1 = (uint)(a >> 32);
    uint b0 = (uint)b, b1 = (uint)(b >> 32);

    uint p00 = a0 * b0, p00h = mulhi(a0, b0);
    uint p01 = a0 * b1, p01h = mulhi(a0, b1);
    uint p10 = a1 * b0, p10h = mulhi(a1, b0);
    uint p11 = a1 * b1, p11h = mulhi(a1, b1);

    ulong t0 = ((ulong)p00h << 32) | (ulong)p00;
    ulong t2 = ((ulong)p11h << 32) | (ulong)p11;

    uint  s_lo = p01 + p10;
    uint  s_carry = (s_lo < p01) ? 1u : 0u;
    ulong s_hi = (ulong)p01h + (ulong)p10h + (ulong)s_carry;

    ulong lo = t0 + ((ulong)s_lo << 32);
    ulong carry = (lo < t0) ? 1ul : 0ul;
    ulong hi = t2 + s_hi + carry;

    uint  x_hi_hi = (uint)(hi >> 32);
    uint  x_hi_lo = (uint)hi;
    ulong r = lo - (ulong)x_hi_hi;
    if (lo < (ulong)x_hi_hi) { r -= GL_EPS; }
    ulong t1 = ((ulong)mulhi(x_hi_lo, (uint)GL_EPS) << 32) | (ulong)(x_hi_lo * (uint)GL_EPS);
    ulong t = r + t1;
    if (t < r) { t += GL_EPS; }
    else if (t >= GL_ORDER) { t -= GL_ORDER; }
    return t;
}

static inline ulong gl_neg(ulong a) { return (a == 0ul) ? 0ul : (GL_ORDER - a); }

// Extension arithmetic as macros over three separate `ulong`s: nothing here returns a
// struct, which is what tripped the compiler when the helpers returned `Ext`.
// x^3 = x + 1.
#define EXT_ADD(r0, r1, r2, a0, a1, a2, b0, b1, b2) do { \
    r0 = gl_add(a0, b0); r1 = gl_add(a1, b1); r2 = gl_add(a2, b2); \
} while (0)

#define EXT_SUB(r0, r1, r2, a0, a1, a2, b0, b1, b2) do { \
    r0 = gl_add(a0, gl_neg(b0)); \
    r1 = gl_add(a1, gl_neg(b1)); \
    r2 = gl_add(a2, gl_neg(b2)); \
} while (0)

#define EXT_MUL(r0, r1, r2, a0, a1, a2, b0, b1, b2) do { \
    ulong _t0, _t1, _t2, _c, _d, _e2; \
    _c = gl_mul(a1, b2); _d = gl_mul(a2, b1); _e2 = gl_mul(a2, b2); \
    _t0 = gl_add(_c, _d); \
    _t1 = gl_add(_t0, _e2); \
    _t2 = gl_add(gl_mul(a0, b2), gl_add(gl_mul(a1, b1), gl_mul(a2, b0))); \
    r0 = gl_add(gl_mul(a0, b0), _t0); \
    r1 = gl_add(gl_add(gl_mul(a0, b1), gl_mul(a1, b0)), _t1); \
    r2 = gl_add(_t2, _e2); \
} while (0)

// One sumcheck round: `out[j] = a[2j] + r*(a[2j+1] - a[2j])`, three scalars per element.
kernel void fold_ext(device const ulong *a [[buffer(0)]],
                     device ulong *out     [[buffer(1)]],
                     device const ulong *r [[buffer(2)]],
                     uint gid [[thread_position_in_grid]])
{
    ulong a00 = a[6u * gid],      a01 = a[6u * gid + 1u], a02 = a[6u * gid + 2u];
    ulong a10 = a[6u * gid + 3u], a11 = a[6u * gid + 4u], a12 = a[6u * gid + 5u];
    ulong r0 = r[0], r1 = r[1], r2 = r[2];

    ulong d0, d1, d2, m0, m1, m2;
    EXT_SUB(d0, d1, d2, a10, a11, a12, a00, a01, a02);
    EXT_MUL(m0, m1, m2, r0, r1, r2, d0, d1, d2);
    ulong o0, o1, o2;
    EXT_ADD(o0, o1, o2, a00, a01, a02, m0, m1, m2);

    out[3u * gid] = o0;
    out[3u * gid + 1u] = o1;
    out[3u * gid + 2u] = o2;
}

// MLE reduction: `Σ_i eq(point, i)·values[i]`, one partial per threadgroup.
kernel void mle_reduce(device const ulong *values [[buffer(0)]],
                       device const ulong *point  [[buffer(1)]],
                       device ulong *out          [[buffer(2)]],
                       device const uint *args    [[buffer(3)]],
                       uint gid   [[thread_position_in_grid]],
                       uint lid   [[thread_position_in_threadgroup]],
                       uint gsize [[threads_per_threadgroup]])
{
    threadgroup ulong partial[3072];

    uint m = args[0];
    uint n = args[1];

    ulong e0 = 1ul, e1 = 0ul, e2 = 0ul;
    if (gid < n) {
        ulong e0i = 1ul, e1i = 0ul, e2i = 0ul;
        for (uint k = 0; k < m; ++k) {
            ulong p0 = point[3u * k], p1 = point[3u * k + 1u], p2 = point[3u * k + 2u];
            bool bit = ((gid >> k) & 1u) != 0u;
            // The factor is `p` when the bit is 1 and `1 - p` when it is 0.
            ulong f0, f1, f2;
            if (bit) { f0 = p0; f1 = p1; f2 = p2; }
            else {
                f0 = gl_add(1ul, gl_neg(p0));
                f1 = gl_neg(p1);
                f2 = gl_neg(p2);
            }
            ulong n0, n1, n2;
            EXT_MUL(n0, n1, n2, e0i, e1i, e2i, f0, f1, f2);
            e0i = n0; e1i = n1; e2i = n2;
        }
        // Multiply by the base-field value: three scalar multiplies.
        ulong v = values[gid];
        e0 = gl_mul(e0i, v);
        e1 = gl_mul(e1i, v);
        e2 = gl_mul(e2i, v);
    }

    partial[3u * lid] = e0;
    partial[3u * lid + 1u] = e1;
    partial[3u * lid + 2u] = e2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = gsize / 2u; st > 0u; st >>= 1u) {
        if (lid < st) {
            partial[3u * lid] = gl_add(partial[3u * lid], partial[3u * (lid + st)]);
            partial[3u * lid + 1u] = gl_add(partial[3u * lid + 1u], partial[3u * (lid + st) + 1u]);
            partial[3u * lid + 2u] = gl_add(partial[3u * lid + 2u], partial[3u * (lid + st) + 2u]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
        ulong base = 3ul * (gid / gsize);
        out[base] = partial[0];
        out[base + 1ul] = partial[1];
        out[base + 2ul] = partial[2];
    }
}

// ---- BLAKE3, single-block root hash ----
// Our Merkle leaves are 2 field elements (16 B) and nodes 2 digests (32 B), so both are
// one BLAKE3 block: one compression, `CHUNK_START | CHUNK_END | ROOT`, counter 0.

constant uint B3_IV0 = 0x6A09E667u, B3_IV1 = 0xBB67AE85u, B3_IV2 = 0x3C6EF372u, B3_IV3 = 0xA54FF53Au;
constant uint B3_IV4 = 0x510E527Fu, B3_IV5 = 0x9B05688Cu, B3_IV6 = 0x1F83D9ABu, B3_IV7 = 0x5BE0CD19u;

static inline uint b3_rotr(uint x, uint n) { return (x >> n) | (x << (32u - n)); }

#define B3_G(a, b, c, d, x, y) do { \
    a = a + b + x; d = b3_rotr(d ^ a, 16u); \
    c = c + d;     b = b3_rotr(b ^ c, 12u); \
    a = a + b + y; d = b3_rotr(d ^ a, 8u);  \
    c = c + d;     b = b3_rotr(b ^ c, 7u);  \
} while (0)

// Compresses one block; `words` is the block length in bytes (<= 64).
static inline void b3_compress(ulong in0, ulong in1, ulong in2, ulong in3,
                               ulong in4, ulong in5, ulong in6, ulong in7,
                               uint block_len, uint flags, thread ulong &out0, thread ulong &out1)
{
    uint m[16];
    ulong packed[8] = {in0, in1, in2, in3, in4, in5, in6, in7};
    for (int i = 0; i < 8; ++i) {
        m[2 * i] = (uint)packed[i];
        m[2 * i + 1] = (uint)(packed[i] >> 32);
    }

    uint v[16];
    v[0] = B3_IV0; v[1] = B3_IV1; v[2] = B3_IV2; v[3] = B3_IV3;
    v[4] = B3_IV4; v[5] = B3_IV5; v[6] = B3_IV6; v[7] = B3_IV7;
    v[8] = B3_IV0; v[9] = B3_IV1; v[10] = B3_IV2; v[11] = B3_IV3;
    v[12] = 0u; v[13] = 0u; v[14] = block_len; v[15] = flags;

    for (int r = 0; r < 7; ++r) {
        B3_G(v[0], v[4], v[8],  v[12], m[0],  m[1]);
        B3_G(v[1], v[5], v[9],  v[13], m[2],  m[3]);
        B3_G(v[2], v[6], v[10], v[14], m[4],  m[5]);
        B3_G(v[3], v[7], v[11], v[15], m[6],  m[7]);
        B3_G(v[0], v[5], v[10], v[15], m[8],  m[9]);
        B3_G(v[1], v[6], v[11], v[12], m[10], m[11]);
        B3_G(v[2], v[7], v[8],  v[13], m[12], m[13]);
        B3_G(v[3], v[4], v[9],  v[14], m[14], m[15]);
        // Message permutation.
        uint t0 = m[2], t1 = m[6], t2 = m[3], t3 = m[10], t4 = m[7], t5 = m[0],
             t6 = m[4], t7 = m[13], t8 = m[1], t9 = m[11], t10 = m[12], t11 = m[5],
             t12 = m[9], t13 = m[14], t14 = m[15], t15 = m[8];
        m[0] = t0; m[1] = t1; m[2] = t2; m[3] = t3; m[4] = t4; m[5] = t5;
        m[6] = t6; m[7] = t7; m[8] = t8; m[9] = t9; m[10] = t10; m[11] = t11;
        m[12] = t12; m[13] = t13; m[14] = t14; m[15] = t15;
    }

    uint o[8];
    for (int i = 0; i < 8; ++i) {
        o[i] = v[i] ^ v[i + 8];
    }
    out0 = ((ulong)o[1] << 32) | (ulong)o[0];
    out1 = ((ulong)o[3] << 32) | (ulong)o[2];
}

// ---- GPU-resident sumcheck ----
// The operand arrays stay on the device for the whole claim; the host is in the loop only
// to derive each round's challenge (Fiat-Shamir is sequential).
//   round poly: A = sum a0*b0, B = sum (a0*db + da*b0), C = sum da*db   (da = a[2j+1]-a[2j])
//   fold:       a[j] = a0 + r*da                                        (same for b)
// The three coefficient reductions reuse `reduce_ext` (one call each): a threadgroup array
// and field arithmetic cannot share a kernel on this GPU.

kernel void sc_terms(device const ulong *a    [[buffer(0)]],
                     device const ulong *b    [[buffer(1)]],
                     device ulong *out        [[buffer(2)]],
                     device const uint *args  [[buffer(3)]],
                     uint gid [[thread_position_in_grid]])
{
    uint npair = args[0];
    if (gid >= npair) { return; }

    ulong a00 = a[6u * gid],      a01 = a[6u * gid + 1u], a02 = a[6u * gid + 2u];
    ulong a10 = a[6u * gid + 3u], a11 = a[6u * gid + 4u], a12 = a[6u * gid + 5u];
    ulong b00 = b[6u * gid],      b01 = b[6u * gid + 1u], b02 = b[6u * gid + 2u];
    ulong b10 = b[6u * gid + 3u], b11 = b[6u * gid + 4u], b12 = b[6u * gid + 5u];

    ulong da0, da1, da2, db0, db1, db2;
    EXT_SUB(da0, da1, da2, a10, a11, a12, a00, a01, a02);
    EXT_SUB(db0, db1, db2, b10, b11, b12, b00, b01, b02);

    ulong A0, A1, A2;
    EXT_MUL(A0, A1, A2, a00, a01, a02, b00, b01, b02);
    ulong p0, p1, p2, q0, q1, q2, B0, B1, B2;
    EXT_MUL(p0, p1, p2, a00, a01, a02, db0, db1, db2);
    EXT_MUL(q0, q1, q2, da0, da1, da2, b00, b01, b02);
    EXT_ADD(B0, B1, B2, p0, p1, p2, q0, q1, q2);
    ulong C0, C1, C2;
    EXT_MUL(C0, C1, C2, da0, da1, da2, db0, db1, db2);

    out[3u * gid] = A0;
    out[3u * gid + 1u] = A1;
    out[3u * gid + 2u] = A2;
    out[3u * (npair + gid)] = B0;
    out[3u * (npair + gid) + 1u] = B1;
    out[3u * (npair + gid) + 2u] = B2;
    out[3u * (2u * npair + gid)] = C0;
    out[3u * (2u * npair + gid) + 1u] = C1;
    out[3u * (2u * npair + gid) + 2u] = C2;
}

kernel void sc_fold(device const ulong *a    [[buffer(0)]],
                    device const ulong *b    [[buffer(1)]],
                    device const ulong *r    [[buffer(2)]],
                    device ulong *oa         [[buffer(3)]],
                    device ulong *ob         [[buffer(4)]],
                    device const uint *args  [[buffer(5)]],
                    uint gid [[thread_position_in_grid]])
{
    uint npair = args[0];
    if (gid >= npair) { return; }
    ulong r0 = r[0], r1 = r[1], r2 = r[2];

    ulong a00 = a[6u * gid],      a01 = a[6u * gid + 1u], a02 = a[6u * gid + 2u];
    ulong a10 = a[6u * gid + 3u], a11 = a[6u * gid + 4u], a12 = a[6u * gid + 5u];
    ulong d0, d1, d2, m0, m1, m2, s0, s1, s2;
    EXT_SUB(d0, d1, d2, a10, a11, a12, a00, a01, a02);
    EXT_MUL(m0, m1, m2, r0, r1, r2, d0, d1, d2);
    EXT_ADD(s0, s1, s2, a00, a01, a02, m0, m1, m2);
    oa[3u * gid] = s0; oa[3u * gid + 1u] = s1; oa[3u * gid + 2u] = s2;

    ulong b00 = b[6u * gid],      b01 = b[6u * gid + 1u], b02 = b[6u * gid + 2u];
    ulong b10 = b[6u * gid + 3u], b11 = b[6u * gid + 4u], b12 = b[6u * gid + 5u];
    ulong e0, e1, e2, n0, n1, n2, t0, t1, t2;
    EXT_SUB(e0, e1, e2, b10, b11, b12, b00, b01, b02);
    EXT_MUL(n0, n1, n2, r0, r1, r2, e0, e1, e2);
    EXT_ADD(t0, t1, t2, b00, b01, b02, n0, n1, n2);
    ob[3u * gid] = t0; ob[3u * gid + 1u] = t1; ob[3u * gid + 2u] = t2;
}

// One digest per thread. `args = [n_digests, words_per_digest]`.
kernel void blake3_leaves(device const ulong *in  [[buffer(0)]],
                          device ulong *out        [[buffer(1)]],
                          device const uint *args  [[buffer(2)]],
                          uint gid [[thread_position_in_grid]])
{
    uint n = args[0];
    uint words = args[1];
    if (gid >= n) { return; }

    ulong w[8] = {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul};
    uint base = gid * words;
    for (uint i = 0; i < words; ++i) { w[i] = in[base + i]; }

    ulong a, b;
    uint block_len = words * 8u;
    // flags: CHUNK_START (1) | CHUNK_END (2) | ROOT (8)
    b3_compress(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7],
                block_len, 11u, a, b);

    out[2u * gid] = a % GL_ORDER;
    out[2u * gid + 1u] = b % GL_ORDER;
}

// One DIF layer of a radix-2 FFT. Natural input order; after all layers the output is
// in bit-reversed order, which is exactly the layout p3's `BitReversedMatrixView` wraps.
// No threadgroup memory: one thread per butterfly, so nothing needs to synchronise.
// `tw` is this layer's compact table: `tw[j] = root^(j * stride)`, read contiguously.
kernel void ntt_layer(device ulong *buf [[buffer(0)]],
                      device const ulong *tw [[buffer(1)]],
                      device const uint *args [[buffer(2)]],
                      uint gid [[thread_position_in_grid]])
{
    // `half` is a reserved type name in MSL; `hw` is the half-block width.
    uint hw = args[0];
    uint w = args[1];

    uint per_block = hw * w;
    uint blk = gid / per_block;
    uint rem = gid - blk * per_block;
    uint j = rem / w;
    uint c = rem - j * w;

    uint a = (blk * 2u * hw + j) * w + c;
    uint b = a + hw * w;

    ulong u = buf[a];
    ulong v = buf[b];
    buf[a] = gl_add(u, v);
    buf[b] = gl_mul(gl_add(u, gl_neg(v)), tw[j]);
}

// `dst[j] = src[2j]`: the next layer's compact twiddle table is the even subsequence
// of this one (its stride doubles). The read is strided by two, which costs half a
// cache line — versus a stride of `2^l` when gathering from the full table.
kernel void decimate_tw(device const ulong *src [[buffer(0)]],
                        device ulong *dst      [[buffer(1)]],
                        device const uint *args [[buffer(2)]],
                        uint gid [[thread_position_in_grid]])
{
    uint n = args[0];
    if (gid >= n) { return; }
    dst[gid] = src[2u * gid];
}

// `p[k] = Σ_n eq_r[n]·w[n·2^lm + k]` — the weight/activation contraction of a
// matmul claim. One thread per `k`, so threads read `w` contiguously.
kernel void contract_p(device const ulong *w       [[buffer(0)]],
                       device const ulong *eq_r    [[buffer(1)]],
                       device ulong *out           [[buffer(2)]],
                       device const uint *args     [[buffer(3)]],
                       uint gid [[thread_position_in_grid]])
{
    uint ln = args[0];
    uint lm = args[1];
    uint rows = 1u << ln;

    ulong p0 = 0ul, p1 = 0ul, p2 = 0ul;
    for (uint n = 0; n < rows; ++n) {
        ulong v = w[(n << lm) + gid];
        ulong e0 = eq_r[3u * n], e1 = eq_r[3u * n + 1u], e2 = eq_r[3u * n + 2u];
        p0 = gl_add(p0, gl_mul(e0, v));
        p1 = gl_add(p1, gl_mul(e1, v));
        p2 = gl_add(p2, gl_mul(e2, v));
    }
    out[3u * gid] = p0;
    out[3u * gid + 1u] = p1;
    out[3u * gid + 2u] = p2;
}

// Pass 2 of the MLE reduction: sum Ext triples, one partial per threadgroup.
// Kept arithmetic-free on purpose: a threadgroup array together with heavy field
// arithmetic in the same kernel does not run on this GPU (it completes with no
// writes), so the two are split.
kernel void reduce_ext(device const ulong *terms [[buffer(0)]],
                       device ulong *out         [[buffer(1)]],
                       device const uint *args   [[buffer(2)]],
                       uint gid   [[thread_position_in_grid]],
                       uint lid   [[thread_position_in_threadgroup]],
                       uint gsize [[threads_per_threadgroup]])
{
    threadgroup ulong partial[1024];
    ulong v0 = terms[3u * gid], v1 = terms[3u * gid + 1u], v2 = terms[3u * gid + 2u];

    partial[lid] = v0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = gsize / 2u; st > 0u; st >>= 1u) {
        if (lid < st) { partial[lid] = gl_add(partial[lid], partial[lid + st]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) { out[3u * (gid / gsize)] = partial[0]; }

    // Components 1 and 2 of this threadgroup, reduced the same way.
    threadgroup ulong p1[1024];
    p1[lid] = v1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = gsize / 2u; st > 0u; st >>= 1u) {
        if (lid < st) { p1[lid] = gl_add(p1[lid], p1[lid + st]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) { out[3u * (gid / gsize) + 1u] = p1[0]; }

    threadgroup ulong p2[1024];
    p2[lid] = v2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = gsize / 2u; st > 0u; st >>= 1u) {
        if (lid < st) { p2[lid] = gl_add(p2[lid], p2[lid + st]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) { out[3u * (gid / gsize) + 2u] = p2[0]; }
}

// ---- bisect kernels ----

kernel void bis_copy(device const ulong *a [[buffer(0)]],
                     device const ulong *b [[buffer(1)]],
                     device ulong *out     [[buffer(2)]],
                     device const uint *args [[buffer(3)]],
                     uint gid [[thread_position_in_grid]])
{
    out[gid] = a[gid];
}

kernel void bis_read_b(device const ulong *a [[buffer(0)]],
                       device const ulong *b [[buffer(1)]],
                       device ulong *out     [[buffer(2)]],
                       device const uint *args [[buffer(3)]],
                       uint gid [[thread_position_in_grid]])
{
    out[gid] = b[gid];
}

kernel void bis_mul(device const ulong *a [[buffer(0)]],
                    device const ulong *b [[buffer(1)]],
                    device ulong *out     [[buffer(2)]],
                    device const uint *args [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    out[gid] = gl_mul(a[gid], b[gid]);
}

kernel void bis_eq(device const ulong *a [[buffer(0)]],
                   device const ulong *b [[buffer(1)]],
                   device ulong *out     [[buffer(2)]],
                   device const uint *args [[buffer(3)]],
                   uint gid [[thread_position_in_grid]])
{
    ulong p0 = b[gid], p1 = 0ul, p2 = 0ul;
    ulong f0, f1, f2;
    if ((gid & 1u) != 0u) { f0 = p0; f1 = p1; f2 = p2; }
    else { f0 = gl_add(1ul, gl_neg(p0)); f1 = gl_neg(p1); f2 = gl_neg(p2); }
    ulong v = a[gid];
    out[3u * gid] = gl_mul(f0, v);
    out[3u * gid + 1u] = gl_mul(f1, v);
    out[3u * gid + 2u] = gl_mul(f2, v);
}

kernel void bis_args(device const ulong *a [[buffer(0)]],
                     device const ulong *b [[buffer(1)]],
                     device ulong *out     [[buffer(2)]],
                     device const uint *args [[buffer(3)]],
                     uint gid [[thread_position_in_grid]])
{
    if (gid == 0u) {
        out[0] = (ulong)args[0];
        out[1] = (ulong)args[1];
    }
}

kernel void bis_reduce(device const ulong *a [[buffer(0)]],
                       device const ulong *b [[buffer(1)]],
                       device ulong *out     [[buffer(2)]],
                       device const uint *args [[buffer(3)]],
                       uint gid [[thread_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]],
                       uint gsize [[threads_per_threadgroup]])
{
    threadgroup ulong partial[1024];
    partial[lid] = a[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = gsize / 2u; st > 0u; st >>= 1u) {
        if (lid < st) { partial[lid] = gl_add(partial[lid], partial[lid + st]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) { out[0] = partial[0]; out[1] = 0ul; out[2] = 0ul; }
}

// Three eq factors, fully scalar, for the bisect.
kernel void mle_e_only(device const ulong *values [[buffer(0)]],
                       device const ulong *point  [[buffer(1)]],
                       device ulong *out          [[buffer(2)]],
                       device const uint *args    [[buffer(3)]],
                       uint gid [[thread_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]],
                       uint gsize [[threads_per_threadgroup]])
{
    threadgroup ulong partial[1024];

    ulong e0 = 1ul, e1 = 0ul, e2 = 0ul;
    // Unrolled: three explicit factors, no loop.
    ulong pa0 = point[0], pa1 = point[1], pa2 = point[2];
    ulong pb0 = point[3], pb1 = point[4], pb2 = point[5];
    ulong pc0 = point[6], pc1 = point[7], pc2 = point[8];

    ulong fa0, fa1, fa2, fb0, fb1, fb2, fc0, fc1, fc2;
    if ((gid & 1u) != 0u) { fa0 = pa0; fa1 = pa1; fa2 = pa2; }
    else { fa0 = gl_add(1ul, gl_neg(pa0)); fa1 = gl_neg(pa1); fa2 = gl_neg(pa2); }
    if ((gid & 2u) != 0u) { fb0 = pb0; fb1 = pb1; fb2 = pb2; }
    else { fb0 = gl_add(1ul, gl_neg(pb0)); fb1 = gl_neg(pb1); fb2 = gl_neg(pb2); }
    if ((gid & 4u) != 0u) { fc0 = pc0; fc1 = pc1; fc2 = pc2; }
    else { fc0 = gl_add(1ul, gl_neg(pc0)); fc1 = gl_neg(pc1); fc2 = gl_neg(pc2); }

    ulong t0, t1, t2;
    EXT_MUL(t0, t1, t2, e0, e1, e2, fa0, fa1, fa2);
    ulong u0, u1, u2;
    EXT_MUL(u0, u1, u2, t0, t1, t2, fb0, fb1, fb2);
    ulong w0, w1, w2;
    EXT_MUL(w0, w1, w2, u0, u1, u2, fc0, fc1, fc2);
    e0 = w0; e1 = w1; e2 = w2;

    // No threadgroup array: write one Ext per thread (host reduces).
    out[3u * gid] = e0;
    out[3u * gid + 1u] = e1;
    out[3u * gid + 2u] = e2;
}

// The value alone, summed: the reduction without any eq.
kernel void mle_v_only(device const ulong *values [[buffer(0)]],
                       device const ulong *point  [[buffer(1)]],
                       device ulong *out          [[buffer(2)]],
                       device const uint *args    [[buffer(3)]],
                       uint gid [[thread_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]],
                       uint gsize [[threads_per_threadgroup]])
{
    threadgroup ulong partial[1024];
    partial[lid] = values[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = gsize / 2u; st > 0u; st >>= 1u) {
        if (lid < st) { partial[lid] = gl_add(partial[lid], partial[lid + st]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) { out[0] = partial[0]; out[1] = 0ul; out[2] = 0ul; }
}

// Per-thread terms, host-summed (no reduction).
kernel void mle_terms(device const ulong *values [[buffer(0)]],
                      device const ulong *point  [[buffer(1)]],
                      device ulong *out          [[buffer(2)]],
                      device const uint *args    [[buffer(3)]],
                      uint gid [[thread_position_in_grid]])
{
    uint m = args[0];
    ulong e0 = 1ul, e1 = 0ul, e2 = 0ul;
    for (uint k = 0; k < m; ++k) {
        ulong p0 = point[3u * k], p1 = point[3u * k + 1u], p2 = point[3u * k + 2u];
        ulong f0, f1, f2;
        if (((gid >> k) & 1u) != 0u) { f0 = p0; f1 = p1; f2 = p2; }
        else { f0 = gl_add(1ul, gl_neg(p0)); f1 = gl_neg(p1); f2 = gl_neg(p2); }
        ulong n0, n1, n2;
        EXT_MUL(n0, n1, n2, e0, e1, e2, f0, f1, f2);
        e0 = n0; e1 = n1; e2 = n2;
    }
    ulong v = values[gid];
    out[3u * gid] = gl_mul(e0, v);
    out[3u * gid + 1u] = gl_mul(e1, v);
    out[3u * gid + 2u] = gl_mul(e2, v);
}

kernel void copy_u64(device const ulong *in [[buffer(0)]],
                     device ulong *out      [[buffer(1)]],
                     uint gid               [[thread_position_in_grid]])
{
    out[gid] = in[gid];
}

"#;

struct Gpu {
    device: Device,
    queue: CommandQueue,
    /// One sumcheck fold.
    fold_pipe: ComputePipelineState,
    /// Per-thread terms.
    terms_pipe: ComputePipelineState,
    /// Reduction of Ext triples.
    reduce_pipe: ComputePipelineState,
    /// Weight/activation contraction.
    contract_pipe: ComputePipelineState,
    /// One radix-2 DIF layer.
    ntt_pipe: ComputePipelineState,
    /// Compact-twiddle decimation.
    decimate_pipe: ComputePipelineState,
    /// BLAKE3 single-block hasher.
    blake3_pipe: ComputePipelineState,
    /// Sumcheck round-poly terms.
    sc_terms_pipe: ComputePipelineState,
    /// Sumcheck fold.
    sc_fold_pipe: ComputePipelineState,
    /// Trivial copy (debug path).
    copy_pipe: ComputePipelineState,
    bis: Vec<(&'static str, ComputePipelineState)>,
}

static GPU: OnceLock<Option<Gpu>> = OnceLock::new();

fn init() -> Option<&'static Gpu> {
    GPU.get_or_init(|| {
        let device = Device::system_default()?;
        let opts = CompileOptions::new();
        // Pin a modern MSL version: the default in metal-rs 0.33 is old, and 64-bit
        // integer arithmetic in loops mis-compiles under it (kernels run but write
        // nothing).
        opts.set_language_version(metal::MTLLanguageVersion::V3_0);
        opts.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(MSL, &opts)
            .map_err(|e| eprintln!("flashprove: Metal shader compile failed: {e}"))
            .ok()?;
        let fold_fn = library
            .get_function("fold_ext", None)
            .map_err(|e| eprintln!("flashprove: Metal function lookup failed: {e}"))
            .ok()?;
        let fold_pipe = device
            .new_compute_pipeline_state_with_function(&fold_fn)
            .map_err(|e| eprintln!("flashprove: Metal pipeline failed: {e}"))
            .ok()?;
        let terms_fn = library
            .get_function("mle_terms", None)
            .map_err(|e| eprintln!("flashprove: Metal function lookup failed: {e}"))
            .ok()?;
        let reduce_fn = library
            .get_function("reduce_ext", None)
            .map_err(|e| eprintln!("flashprove: reduce_ext: {e}"))
            .ok()?;
        let reduce_pipe = device
            .new_compute_pipeline_state_with_function(&reduce_fn)
            .map_err(|e| eprintln!("flashprove: reduce_ext pipeline: {e}"))
            .ok()?;
        let contract_fn = library
            .get_function("contract_p", None)
            .map_err(|e| eprintln!("flashprove: contract_p: {e}"))
            .ok()?;
        let contract_pipe = device
            .new_compute_pipeline_state_with_function(&contract_fn)
            .map_err(|e| eprintln!("flashprove: contract_p pipeline: {e}"))
            .ok()?;
        let ntt_fn = library
            .get_function("ntt_layer", None)
            .map_err(|e| eprintln!("flashprove: ntt_layer: {e}"))
            .ok()?;
        let ntt_pipe = device
            .new_compute_pipeline_state_with_function(&ntt_fn)
            .map_err(|e| eprintln!("flashprove: ntt_layer pipeline: {e}"))
            .ok()?;
        let dec_fn = library
            .get_function("decimate_tw", None)
            .map_err(|e| eprintln!("flashprove: decimate_tw: {e}"))
            .ok()?;
        let decimate_pipe = device
            .new_compute_pipeline_state_with_function(&dec_fn)
            .map_err(|e| eprintln!("flashprove: decimate_tw pipeline: {e}"))
            .ok()?;
        let b3_fn = library
            .get_function("blake3_leaves", None)
            .map_err(|e| eprintln!("flashprove: blake3_leaves: {e}"))
            .ok()?;
        let blake3_pipe = device
            .new_compute_pipeline_state_with_function(&b3_fn)
            .map_err(|e| eprintln!("flashprove: blake3_leaves pipeline: {e}"))
            .ok()?;
        let mut sc_terms_pipe = None;
        let mut sc_fold_pipe = None;
        for (name, slot) in [
            ("sc_terms", &mut sc_terms_pipe),
            ("sc_fold", &mut sc_fold_pipe),
        ] {
            let f = library
                .get_function(name, None)
                .map_err(|e| eprintln!("flashprove: {name}: {e}"))
                .ok()?;
            let p = device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| eprintln!("flashprove: {name} pipeline: {e}"))
                .ok()?;
            *slot = Some(p);
        }
        let sc_terms_pipe = sc_terms_pipe?;
        let sc_fold_pipe = sc_fold_pipe?;
        let terms_pipe = device
            .new_compute_pipeline_state_with_function(&terms_fn)
            .map_err(|e| eprintln!("flashprove: Metal pipeline failed: {e}"))
            .ok()?;
        let copy_fn = library
            .get_function("copy_u64", None)
            .map_err(|e| eprintln!("flashprove: Metal function lookup failed: {e}"))
            .ok()?;
        let copy_pipe = device
            .new_compute_pipeline_state_with_function(&copy_fn)
            .map_err(|e| eprintln!("flashprove: Metal pipeline failed: {e}"))
            .ok()?;
        let mut bis = Vec::new();
        for name in ["bis_copy", "bis_read_b", "bis_mul", "bis_eq", "bis_args", "bis_reduce", "mle_v_only", "mle_e_only", "mle_reduce"] {
            let f = library
                .get_function(name, None)
                .map_err(|e| eprintln!("flashprove: {name}: {e}"))
                .ok()?;
            let p = device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| eprintln!("flashprove: {name}: pipeline: {e}"))
                .ok()?;
            bis.push((name, p));
        }
        let queue = device.new_command_queue();
        let name = device.name().to_string();
        let gpu = Gpu {
            device,
            queue,
            fold_pipe,
            terms_pipe,
            reduce_pipe,
            contract_pipe,
            ntt_pipe,
            decimate_pipe,
            blake3_pipe,
            sc_terms_pipe,
            sc_fold_pipe,
            copy_pipe,
            bis,
        };
        // Self-check against the CPU reference; fall back for good if they disagree.
        let values: Vec<Fp> = (0..1024u64).map(|i| Fp::new(i * 7 + 3)).collect();
        let point: Vec<Ext> = (0..3u64).map(|i| Ext::from(Fp::new(11 + i))).collect();
        let want = crate::backend::mle_eval_cpu(&values, &point);
        let got = gpu.mle_eval_raw(&values, &point);
        let force = std::env::var("FLASHPROVE_GPU_FORCE").is_ok();
        if got != want && !force {
            let cb: &[Fp] = <Ext as BasedVectorSpace<Fp>>::as_basis_coefficients_slice(&want);
            let gb: &[Fp] = <Ext as BasedVectorSpace<Fp>>::as_basis_coefficients_slice(&got);
            eprintln!(
                "flashprove: Metal self-check failed — staying on the CPU\n  cpu {:?}\n  gpu {:?}",
                cb.iter().map(|v| v.as_canonical_u64()).collect::<Vec<_>>(),
                gb.iter().map(|v| v.as_canonical_u64()).collect::<Vec<_>>(),
            );
            return None;
        }
        eprintln!("flashprove: Metal backend ready ({name})");
        Some(gpu)
    })
    .as_ref()
}

impl Gpu {
    fn mle_eval_raw(&self, values: &[Fp], point: &[Ext]) -> Ext {
        let n = values.len();
        let m = point.len();
        let vals: Vec<u64> = values.iter().map(|v| v.as_canonical_u64()).collect();
        let mut pts: Vec<u64> = Vec::with_capacity(3 * m);
        for e in point {
            for c in <Ext as BasedVectorSpace<Fp>>::as_basis_coefficients_slice(e) {
                pts.push(c.as_canonical_u64());
            }
        }

        const GROUP: u64 = 1024;
        let groups = (n as u64).div_ceil(GROUP);
        let opts = MTLResourceOptions::StorageModeShared;

        let vbuf = self.device.new_buffer_with_data(
            vals.as_ptr() as *const std::ffi::c_void,
            (n * 8) as u64,
            opts,
        );
        let pbuf = self.device.new_buffer_with_data(
            pts.as_ptr() as *const std::ffi::c_void,
            (pts.len() * 8) as u64,
            opts,
        );
        let args: [u32; 2] = [m as u32, n as u32];
        let argsbuf = self.device.new_buffer_with_data(
            args.as_ptr() as *const std::ffi::c_void,
            8,
            opts,
        );
        // Pass 1: one Ext term per index (no threadgroup memory in this kernel).
        let terms = self.device.new_buffer((n * 24) as u64, opts);
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.terms_pipe);
        enc.set_buffer(0, Some(&vbuf), 0);
        enc.set_buffer(1, Some(&pbuf), 0);
        enc.set_buffer(2, Some(&terms), 0);
        enc.set_buffer(3, Some(&argsbuf), 0);
        enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(GROUP, 1, 1));
        enc.end_encoding();

        // Pass 2: reduce the terms, one partial per threadgroup.
        let out = self.device.new_buffer(groups * 24, opts);
        let enc2 = cmd.new_compute_command_encoder();
        enc2.set_compute_pipeline_state(&self.reduce_pipe);
        enc2.set_buffer(0, Some(&terms), 0);
        enc2.set_buffer(1, Some(&out), 0);
        enc2.set_buffer(2, Some(&argsbuf), 0);
        enc2.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(GROUP, 1, 1));
        enc2.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        if std::env::var("FLASHPROVE_GPU_DEBUG").is_ok() {
            eprintln!(
                "  [gpu] n={n} m={m} groups={groups} status={:?}",
                cmd.status()
            );
        }

        // Sum the per-group partials on the host (a few thousand at most).
        let raw = unsafe {
            std::slice::from_raw_parts(out.contents() as *const u64, (groups as usize) * 3)
        };
        let mut acc = Ext::ZERO;
        for g in 0..groups as usize {
            let coeffs = [
                Fp::new(raw[3 * g]),
                Fp::new(raw[3 * g + 1]),
                Fp::new(raw[3 * g + 2]),
            ];
            acc += <Ext as BasedVectorSpace<Fp>>::from_basis_coefficients_slice(&coeffs).unwrap();
        }
        acc
    }

    /// `out[j] = a[2j] + r·(a[2j+1] − a[2j])` for `j < n/2`.
    fn fold_raw(&self, a: &[Ext], r: Ext) -> Vec<Ext> {
        let n = a.len();
        let half = n / 2;
        let opts = MTLResourceOptions::StorageModeShared;
        let abuf = self.device.new_buffer_with_data(
            a.as_ptr() as *const std::ffi::c_void,
            (n * 24) as u64,
            opts,
        );
        let obuf = self.device.new_buffer((half * 24) as u64, opts);
        let rb: &[Fp] = <Ext as BasedVectorSpace<Fp>>::as_basis_coefficients_slice(&r);
        let rw: [u64; 3] = [
            rb[0].as_canonical_u64(),
            rb[1].as_canonical_u64(),
            rb[2].as_canonical_u64(),
        ];
        let rbuf = self.device.new_buffer_with_data(
            rw.as_ptr() as *const std::ffi::c_void,
            24,
            opts,
        );

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.fold_pipe);
        enc.set_buffer(0, Some(&abuf), 0);
        enc.set_buffer(1, Some(&obuf), 0);
        enc.set_buffer(2, Some(&rbuf), 0);
        enc.dispatch_threads(MTLSize::new(half as u64, 1, 1), MTLSize::new(1024, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let out = unsafe {
            std::slice::from_raw_parts(obuf.contents() as *const u64, half * 3)
        };
        (0..half)
            .map(|j| {
                let coeffs = [Fp::new(out[3 * j]), Fp::new(out[3 * j + 1]), Fp::new(out[3 * j + 2])];
                <Ext as BasedVectorSpace<Fp>>::from_basis_coefficients_slice(&coeffs).unwrap()
            })
            .collect()
    }
}

/// The bisect: run each kernel that adds one thing to the working copy, and report
/// the first step that stops producing the expected values.
pub fn bisect() -> Vec<(&'static str, String)> {
    let Some(gpu) = init() else {
        return vec![("init", "gpu unavailable".into())];
    };
    let n = 1024usize;
    let a: Vec<u64> = (0..n as u64).map(|i| i * 3 + 1).collect();
    let b: Vec<u64> = (0..n as u64).map(|i| i * 5 + 2).collect();
    let opts = MTLResourceOptions::StorageModeShared;
    let abuf = gpu
        .device
        .new_buffer_with_data(a.as_ptr() as *const std::ffi::c_void, (n * 8) as u64, opts);
    let bbuf = gpu
        .device
        .new_buffer_with_data(b.as_ptr() as *const std::ffi::c_void, (n * 8) as u64, opts);
    let args: [u32; 2] = [3, n as u32];
    let argsbuf = gpu
        .device
        .new_buffer_with_data(args.as_ptr() as *const std::ffi::c_void, 8, opts);
    // Big enough for any step: n triples of u64.
    let obuf = gpu.device.new_buffer((n * 24) as u64, opts);

    let read = |k: usize| -> Vec<u64> {
        unsafe { std::slice::from_raw_parts(obuf.contents() as *const u64, k) }.to_vec()
    };

    let prime = |buf: &Buffer, bytes: u64| {
        unsafe {
            std::ptr::write_bytes(buf.contents() as *mut u8, 0, bytes as usize);
        }
    };

    let mut out = Vec::new();
    for (i, (name, pipe)) in gpu.bis.iter().enumerate() {
        prime(&obuf, (n * 24) as u64);
        let cmd = gpu.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        // Step 0 binds only two buffers, to test whether extra bindings are the issue.
        enc.set_buffer(0, Some(&abuf), 0);
        if i >= 1 || *name == "bis_copy" {
            // bind b and args from step 1 on (bis_copy is run twice, see below)
        }
        enc.set_buffer(2, Some(&obuf), 0);
        if *name != "bis_copy" {
            enc.set_buffer(1, Some(&bbuf), 0);
            enc.set_buffer(3, Some(&argsbuf), 0);
        }
        enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(1024, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let got = read(6);
        let st = cmd.status();
        let ok = match *name {
            "bis_copy" => got[0] == a[0],
            "bis_read_b" => got[0] == b[0],
            "bis_mul" => got[0] == crate::backend::mul_cpu(a[0], b[0]),
            "bis_eq" => {
                // term 0: eq over a single coordinate p = b[0], index bit 0 = 0
                let p = Fp::new(b[0]);
                let w0 = Ext::ONE - Ext::from(p);
                let want = w0 * Ext::from(Fp::new(a[0]));
                let wc: &[Fp] = <Ext as BasedVectorSpace<Fp>>::as_basis_coefficients_slice(&want);
                got[0] == wc[0].as_canonical_u64()
            }
            "bis_args" => got[0] == 3 && got[1] == n as u64,
            "mle_v_only" => got[0] == a.iter().sum::<u64>() % crate::backend::ORDER,
            "mle_e_only" => got[0] == 128 * 1,
            "mle_reduce" => {
                let pt: Vec<Ext> = (0..3).map(|k| Ext::from(Fp::new(b[k]))).collect();
                let vals: Vec<Fp> = a.iter().map(|&v| Fp::new(v)).collect();
                let want = crate::backend::mle_eval_cpu(&vals, &pt);
                let wc: &[Fp] = <Ext as BasedVectorSpace<Fp>>::as_basis_coefficients_slice(&want);
                got[0] == wc[0].as_canonical_u64()
            }
            "bis_reduce" => got[0] == a.iter().sum::<u64>() % crate::backend::ORDER,
            _ => false,
        };
        out.push((
            *name,
            format!(
                "{} first={:?} status={:?} err={:?}",
                if ok { "OK" } else { "FAIL" },
                got,
                st,
                "n/a"
            ),
        ));
    }
    // The same *simple* kernel at 2^18, i.e. 256 threadgroups instead of 1.
    {
        let big = 1usize << 18;
        let a2: Vec<u64> = (0..big as u64).map(|i| i * 3 + 1).collect();
        let abuf2 = gpu.device.new_buffer_with_data(
            a2.as_ptr() as *const std::ffi::c_void,
            (big * 8) as u64,
            opts,
        );
        let obuf2 = gpu.device.new_buffer((big * 24) as u64, opts);
        let args2: [u32; 2] = [3, big as u32];
        let argsbuf2 = gpu.device.new_buffer_with_data(
            args2.as_ptr() as *const std::ffi::c_void,
            8,
            opts,
        );
        let cmd = gpu.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&gpu.bis[5].1); // mle_v_only
        enc.set_buffer(0, Some(&abuf2), 0);
        enc.set_buffer(2, Some(&obuf2), 0);
        enc.set_buffer(3, Some(&argsbuf2), 0);
        enc.dispatch_threads(MTLSize::new(big as u64, 1, 1), MTLSize::new(1024, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let raw = unsafe { std::slice::from_raw_parts(obuf2.contents() as *const u64, 3) };
        let want = a2.iter().sum::<u64>() % crate::backend::ORDER;
        out.push((
            "mle_v_only@2^18",
            format!(
                "{} got={:?} want={} status={:?}",
                if raw[0] == want { "OK" } else { "FAIL" },
                raw,
                want,
                cmd.status()
            ),
        ));
    }

    // bis_copy with all four buffers bound.
    {
        prime(&obuf, (n * 24) as u64);
        let cmd = gpu.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&gpu.bis[0].1);
        enc.set_buffer(0, Some(&abuf), 0);
        enc.set_buffer(1, Some(&bbuf), 0);
        enc.set_buffer(2, Some(&obuf), 0);
        enc.set_buffer(3, Some(&argsbuf), 0);
        enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(1024, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let got = read(3);
        out.push((
            "bis_copy+4buf",
            format!("{} first={:?}", if got[0] == a[0] { "OK" } else { "FAIL" }, got),
        ));
    }
    out
}

impl Gpu {
    /// `dft_batch` on the GPU: `log2(h)` DIF layers over `h × w`, output bit-reversed.
    ///
    /// `tw` holds `root^0 … root^(h/2-1)` for the transform's root of order `h`.
    fn dft_raw(&self, values: &[Fp], h: usize, w: usize, tw: &[Fp]) -> Vec<Fp> {
        let opts = MTLResourceOptions::StorageModeShared;
        let vals: Vec<u64> = values.iter().map(|v| v.as_canonical_u64()).collect();
        let tws: Vec<u64> = tw.iter().map(|v| v.as_canonical_u64()).collect();

        let buf = self.device.new_buffer_with_data(
            vals.as_ptr() as *const std::ffi::c_void,
            (vals.len() * 8) as u64,
            opts,
        );
        let tw_a = self.device.new_buffer_with_data(
            tws.as_ptr() as *const std::ffi::c_void,
            (tws.len() * 8) as u64,
            opts,
        );
        let tw_b = self.device.new_buffer(((h / 4).max(1) * 8) as u64, opts);

        // Every layer's args up front, bound with a per-layer offset: allocating a small
        // buffer per layer cost more than the transform itself.
        let n_disp = h.trailing_zeros() as usize;
        let mut args: Vec<u32> = Vec::with_capacity(2 * n_disp);
        let mut half = h / 2;
        for _ in 0..n_disp {
            args.push(half as u32);
            args.push(w as u32);
            half /= 2;
        }
        let argbuf = self.device.new_buffer_with_data(
            args.as_ptr() as *const std::ffi::c_void,
            (args.len() * 4) as u64,
            opts,
        );

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        let mut half = h / 2;
        let mut src_is_a = true;
        for layer in 0..n_disp {
            let off = (layer * 8) as u64;
            if layer > 0 {
                enc.set_compute_pipeline_state(&self.decimate_pipe);
                enc.set_buffer(0, Some(if src_is_a { &tw_a } else { &tw_b }), 0);
                enc.set_buffer(1, Some(if src_is_a { &tw_b } else { &tw_a }), 0);
                enc.set_buffer(2, Some(&argbuf), off);
                enc.dispatch_threads(MTLSize::new(half as u64, 1, 1), MTLSize::new(256, 1, 1));
                src_is_a = !src_is_a;
            }

            enc.set_compute_pipeline_state(&self.ntt_pipe);
            enc.set_buffer(0, Some(&buf), 0);
            enc.set_buffer(1, Some(if src_is_a { &tw_a } else { &tw_b }), 0);
            enc.set_buffer(2, Some(&argbuf), off);
            let threads = (h / 2) * w;
            enc.dispatch_threads(MTLSize::new(threads as u64, 1, 1), MTLSize::new(256, 1, 1));

            half /= 2;
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let raw = unsafe { std::slice::from_raw_parts(buf.contents() as *const u64, vals.len()) };
        raw.iter().map(|&v| Fp::new(v)).collect()
    }

    /// One BLAKE3 digest per `words` field elements (≤ 8, i.e. one 64-byte block).
    fn hash_many_raw(&self, input: &[Fp], words: usize) -> Vec<[Fp; 2]> {
        let n = input.len() / words;
        let opts = MTLResourceOptions::StorageModeShared;
        let vals: Vec<u64> = input.iter().map(|v| v.as_canonical_u64()).collect();
        let ibuf = self.device.new_buffer_with_data(
            vals.as_ptr() as *const std::ffi::c_void,
            (vals.len() * 8) as u64,
            opts,
        );
        let obuf = self.device.new_buffer((n * 16) as u64, opts);
        let args: [u32; 2] = [n as u32, words as u32];
        let abuf = self.device.new_buffer_with_data(
            args.as_ptr() as *const std::ffi::c_void,
            8,
            opts,
        );
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.blake3_pipe);
        enc.set_buffer(0, Some(&ibuf), 0);
        enc.set_buffer(1, Some(&obuf), 0);
        enc.set_buffer(2, Some(&abuf), 0);
        enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let raw = unsafe { std::slice::from_raw_parts(obuf.contents() as *const u64, n * 2) };
        (0..n)
            .map(|i| [Fp::new(raw[2 * i]), Fp::new(raw[2 * i + 1])])
            .collect()
    }

    /// `p[k] = Σ_n eq[n]·w[n·2^lm + k]`.
    #[allow(dead_code)]
    fn contract_raw(&self, w: &[Fp], eq: &[Ext], lm: usize) -> Vec<Ext> {
        let rows = eq.len();
        let cols = 1usize << lm;
        let wv: Vec<u64> = w.iter().map(|v| v.as_canonical_u64()).collect();
        let mut eqv: Vec<u64> = Vec::with_capacity(3 * rows);
        for e in eq {
            for c in <Ext as BasedVectorSpace<Fp>>::as_basis_coefficients_slice(e) {
                eqv.push(c.as_canonical_u64());
            }
        }
        let opts = MTLResourceOptions::StorageModeShared;
        let wbuf = self.device.new_buffer_with_data(
            wv.as_ptr() as *const std::ffi::c_void,
            (wv.len() * 8) as u64,
            opts,
        );
        let ebuf = self.device.new_buffer_with_data(
            eqv.as_ptr() as *const std::ffi::c_void,
            (eqv.len() * 8) as u64,
            opts,
        );
        let obuf = self.device.new_buffer((cols * 24) as u64, opts);
        let args: [u32; 2] = [rows.trailing_zeros(), lm as u32];
        let abuf = self.device.new_buffer_with_data(
            args.as_ptr() as *const std::ffi::c_void,
            8,
            opts,
        );

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.contract_pipe);
        enc.set_buffer(0, Some(&wbuf), 0);
        enc.set_buffer(1, Some(&ebuf), 0);
        enc.set_buffer(2, Some(&obuf), 0);
        enc.set_buffer(3, Some(&abuf), 0);
        enc.dispatch_threads(MTLSize::new(cols as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let raw = unsafe { std::slice::from_raw_parts(obuf.contents() as *const u64, cols * 3) };
        (0..cols)
            .map(|k| {
                let c = [
                    Fp::new(raw[3 * k]),
                    Fp::new(raw[3 * k + 1]),
                    Fp::new(raw[3 * k + 2]),
                ];
                <Ext as BasedVectorSpace<Fp>>::from_basis_coefficients_slice(&c).unwrap()
            })
            .collect()
    }
}

/// The whole linear-claim sumcheck on the GPU: operands stay resident, the host only
/// relays challenges through the transcript (identical observation order to the CPU path).
pub fn linear_sumcheck(
    hf: &[Ext],
    vf: &[Ext],
    tr: &mut crate::pcs::transcript::Transcript,
) -> Option<(Vec<crate::field::sumcheck::RoundPoly>, Vec<Ext>)> {
    let m = hf.len().trailing_zeros() as usize;
    let mut sc = SumcheckGpu::new(hf, vf)?;
    let mut rounds = Vec::with_capacity(m);
    let mut z = Vec::with_capacity(m);
    for _ in 0..m {
        let (a, bcoef, c) = sc.round();
        let g = crate::field::sumcheck::RoundPoly([a, bcoef, c]);
        for &coef in &g.0 {
            tr.observe_ext(coef);
        }
        let r = tr.sample_ext();
        rounds.push(g);
        z.push(r);
        sc.fold(r);
    }
    Some((rounds, z))
}

/// A sumcheck whose two operand arrays stay resident on the GPU for the whole claim.
///
/// The host stays in the loop to sample each round's challenge (Fiat–Shamir is
/// sequential), but never touches the data: per round it reads back three extension
/// elements and writes one.
pub struct SumcheckGpu {
    gpu: &'static Gpu,
    /// Ping-pong pairs of operand buffers: pair 0 = (a0, b0), pair 1 = (a1, b1).
    bufs: [Buffer; 4],
    /// Round-poly terms for the three coefficients.
    terms: Buffer,
    /// Per-group partial sums for the three coefficients.
    partials: Buffer,
    /// `[npair]` for the current round.
    args: Buffer,
    /// `r` for the fold.
    rbuf: Buffer,
    /// Current operand length (`2^m`, halved each round).
    n: usize,
    /// Which ping-pong pair holds the live operands.
    cur: usize,
}

impl SumcheckGpu {
    /// `None` when the GPU is off, unavailable, or the arrays are too small to pay for it.
    pub fn new(a: &[Ext], b: &[Ext]) -> Option<Self> {
        assert_eq!(a.len(), b.len());
        let n = a.len();
        // Measured: 0.4x at 2^18, 1.2x at 2^20, 1.8x at 2^22 — only 2^21 and up pays.
        if n < (1 << 21) || !n.is_power_of_two() {
            return None;
        }
        let gpu = init()?;
        let opts = MTLResourceOptions::StorageModeShared;
        let flat = |x: &[Ext]| -> Vec<u64> {
            let mut v = Vec::with_capacity(3 * x.len());
            for e in x {
                for c in <Ext as BasedVectorSpace<Fp>>::as_basis_coefficients_slice(e) {
                    v.push(c.as_canonical_u64());
                }
            }
            v
        };
        let bytes = (n * 24) as u64;
        let mk = |vals: &[Ext]| {
            let f = flat(vals);
            gpu.device
                .new_buffer_with_data(f.as_ptr() as *const std::ffi::c_void, bytes, opts)
        };
        let bufs = [
            mk(a),
            mk(b),
            gpu.device.new_buffer(bytes, opts),
            gpu.device.new_buffer(bytes, opts),
        ];
        // Three coefficients, one Ext each per pair.
        let terms = gpu.device.new_buffer((n / 2 * 3 * 24) as u64, opts);
        let groups = (n / 2).div_ceil(1024);
        let partials = gpu.device.new_buffer((groups * 3 * 24) as u64, opts);
        let args = gpu
            .device
            .new_buffer_with_data([(n / 2) as u32].as_ptr() as *const std::ffi::c_void, 4, opts);
        let rbuf = gpu.device.new_buffer(24, opts);
        Some(Self {
            gpu,
            bufs,
            terms,
            partials,
            args,
            rbuf,
            n,
            cur: 0,
        })
    }

    /// The current round's polynomial coefficients `(A, B, C)`.
    pub fn round(&mut self) -> (Ext, Ext, Ext) {
        let npair = self.n / 2;
        let groups = npair.div_ceil(1024) as usize;
        let cmd = self.gpu.queue.new_command_buffer();

        // 1. Terms.
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.gpu.sc_terms_pipe);
        enc.set_buffer(0, Some(&self.bufs[2 * self.cur]), 0);
        enc.set_buffer(1, Some(&self.bufs[2 * self.cur + 1]), 0);
        enc.set_buffer(2, Some(&self.terms), 0);
        enc.set_buffer(3, Some(&self.args), 0);
        enc.dispatch_threads(MTLSize::new(npair as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();

        // 2. One reduction per coefficient, by binding offsets into the terms/partials.
        for k in 0..3usize {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.gpu.reduce_pipe);
            enc.set_buffer(0, Some(&self.terms), (k * npair * 24) as u64);
            enc.set_buffer(1, Some(&self.partials), (k * groups * 24) as u64);
            enc.set_buffer(2, Some(&self.args), 0);
            enc.dispatch_threads(MTLSize::new(npair as u64, 1, 1), MTLSize::new(1024, 1, 1));
            enc.end_encoding();
        }
        cmd.commit();
        cmd.wait_until_completed();

        let raw = unsafe {
            std::slice::from_raw_parts(self.partials.contents() as *const u64, groups * 3 * 3)
        };
        let mut out = [Ext::ZERO; 3];
        for k in 0..3usize {
            let mut acc = Ext::ZERO;
            for g in 0..groups {
                let base = k * groups * 3 + g * 3;
                let c = [
                    Fp::new(raw[base]),
                    Fp::new(raw[base + 1]),
                    Fp::new(raw[base + 2]),
                ];
                acc += <Ext as BasedVectorSpace<Fp>>::from_basis_coefficients_slice(&c).unwrap();
            }
            out[k] = acc;
        }
        (out[0], out[1], out[2])
    }

    /// Fold the low variable with challenge `r`.
    pub fn fold(&mut self, r: Ext) {
        let npair = self.n / 2;
        let w: [u64; 3] = {
            let c: &[Fp] = <Ext as BasedVectorSpace<Fp>>::as_basis_coefficients_slice(&r);
            [
                c[0].as_canonical_u64(),
                c[1].as_canonical_u64(),
                c[2].as_canonical_u64(),
            ]
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                w.as_ptr(),
                self.rbuf.contents() as *mut u64,
                3,
            );
        }
        let dst = 1 - self.cur;
        let cmd = self.gpu.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.gpu.sc_fold_pipe);
        enc.set_buffer(0, Some(&self.bufs[2 * self.cur]), 0);
        enc.set_buffer(1, Some(&self.bufs[2 * self.cur + 1]), 0);
        enc.set_buffer(2, Some(&self.rbuf), 0);
        enc.set_buffer(3, Some(&self.bufs[2 * dst]), 0);
        enc.set_buffer(4, Some(&self.bufs[2 * dst + 1]), 0);
        enc.set_buffer(5, Some(&self.args), 0);
        enc.dispatch_threads(MTLSize::new(npair as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        self.cur = dst;
        self.n = npair;
        // `args` carries the half-length for the next round.
        let next = [(self.n / 2).max(1) as u32];
        unsafe {
            std::ptr::copy_nonoverlapping(
                next.as_ptr(),
                self.args.contents() as *mut u32,
                1,
            );
        }
    }
}

/// One BLAKE3 digest per `words` field elements; `None` when the GPU is unavailable or
/// the digest input would not fit in a single BLAKE3 block.
pub fn hash_many(input: &[Fp], words: usize) -> Option<Vec<[Fp; 2]>> {
    if words == 0 || words > 8 || input.len() % words != 0 {
        return None;
    }
    let gpu = init()?;
    Some(gpu.hash_many_raw(input, words))
}

/// `Some(values)` when the GPU produced them, `None` when it is unavailable.
pub fn dft(values: &[Fp], h: usize, w: usize, tw: &[Fp]) -> Option<Vec<Fp>> {
    let gpu = init()?;
    Some(gpu.dft_raw(values, h, w, tw))
}

/// `Some(p)` when the GPU produced it, `None` when it is unavailable.
pub fn contract(w: &[Fp], eq: &[Ext], lm: usize) -> Option<Vec<Ext>> {
    let gpu = init()?;
    Some(gpu.contract_raw(w, eq, lm))
}

/// Debug: copy `x` through the GPU, for plumbing checks.
pub fn copy_probe(x: &[u64]) -> Option<Vec<u64>> {
    let gpu = init()?;
    let n = x.len();
    let opts = MTLResourceOptions::StorageModeShared;
    let ibuf = gpu
        .device
        .new_buffer_with_data(x.as_ptr() as *const std::ffi::c_void, (n * 8) as u64, opts);
    let obuf = gpu.device.new_buffer((n * 8) as u64, opts);
    let cmd = gpu.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&gpu.copy_pipe);
    enc.set_buffer(0, Some(&ibuf), 0);
    enc.set_buffer(1, Some(&obuf), 0);
    enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(1024, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let raw = unsafe { std::slice::from_raw_parts(obuf.contents() as *const u64, n) };
    Some(raw.to_vec())
}

/// Debug: the per-index terms `eq(point,i)·values[i]`, summed on the host.
pub fn mle_terms(values: &[Fp], point: &[Ext]) -> Option<Vec<Ext>> {
    let gpu = init()?;
    let n = values.len();
    let m = point.len();
    let vals: Vec<u64> = values.iter().map(|v| v.as_canonical_u64()).collect();
    let mut pts: Vec<u64> = Vec::with_capacity(3 * m);
    for e in point {
        for c in <Ext as BasedVectorSpace<Fp>>::as_basis_coefficients_slice(e) {
            pts.push(c.as_canonical_u64());
        }
    }
    let opts = MTLResourceOptions::StorageModeShared;
    let vbuf = gpu
        .device
        .new_buffer_with_data(vals.as_ptr() as *const std::ffi::c_void, (n * 8) as u64, opts);
    let pbuf = gpu.device.new_buffer_with_data(
        pts.as_ptr() as *const std::ffi::c_void,
        (pts.len() * 8) as u64,
        opts,
    );
    let obuf = gpu.device.new_buffer((n * 24) as u64, opts);
    let cmd = gpu.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&gpu.terms_pipe);
    enc.set_buffer(0, Some(&vbuf), 0);
    enc.set_buffer(1, Some(&pbuf), 0);
    enc.set_buffer(2, Some(&obuf), 0);
    let args: [u32; 2] = [m as u32, n as u32];
    let args_buf = gpu
        .device
        .new_buffer_with_data(args.as_ptr() as *const std::ffi::c_void, 8, opts);
    enc.set_buffer(3, Some(&args_buf), 0);
    enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(1024, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let raw = unsafe { std::slice::from_raw_parts(obuf.contents() as *const u64, n * 3) };
    Some(
        (0..n)
            .map(|i| {
                let c = [Fp::new(raw[3 * i]), Fp::new(raw[3 * i + 1]), Fp::new(raw[3 * i + 2])];
                <Ext as BasedVectorSpace<Fp>>::from_basis_coefficients_slice(&c).unwrap()
            })
            .collect(),
    )
}

/// `Some(value)` when the GPU produced a result, `None` when it is unavailable.
pub fn mle_eval(values: &[Fp], point: &[Ext]) -> Option<Ext> {
    let gpu = init()?;
    Some(gpu.mle_eval_raw(values, point))
}

/// `Some(folded)` when the GPU produced a result, `None` when it is unavailable.
pub fn fold(a: &[Ext], r: Ext) -> Option<Vec<Ext>> {
    let gpu = init()?;
    Some(gpu.fold_raw(a, r))
}
