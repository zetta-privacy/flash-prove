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
using namespace metal;

struct Ext { ulong a; ulong b; ulong c; };

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

// x^3 = x + 1
static inline Ext ext_mul(Ext x, Ext y) {
    Ext out;
    ulong a1b2 = gl_mul(x.b, y.c);
    ulong a2b1 = gl_mul(x.c, y.b);
    ulong a2b2 = gl_mul(x.c, y.c);
    ulong t = gl_add(a1b2, a2b1);
    out.a = gl_add(gl_mul(x.a, y.a), t);
    out.c = gl_add(gl_add(gl_mul(x.a, y.c), gl_add(gl_mul(x.b, y.b), gl_mul(x.c, y.a))), a2b2);
    out.b = gl_add(gl_add(gl_mul(x.a, y.b), gl_mul(x.b, y.a)), gl_add(t, a2b2));
    return out;
}

static inline Ext ext_add(Ext x, Ext y) {
    Ext out;
    out.a = gl_add(x.a, y.a);
    out.b = gl_add(x.b, y.b);
    out.c = gl_add(x.c, y.c);
    return out;
}

static inline Ext ext_neg(Ext x) {
    Ext out;
    out.a = (x.a == 0ul) ? 0ul : (GL_ORDER - x.a);
    out.b = (x.b == 0ul) ? 0ul : (GL_ORDER - x.b);
    out.c = (x.c == 0ul) ? 0ul : (GL_ORDER - x.c);
    return out;
}

static inline Ext ext_sub(Ext x, Ext y) { return ext_add(x, ext_neg(y)); }

static inline Ext ext_load(device const ulong *p) {
    Ext e = {p[0], p[1], p[2]};
    return e;
}
static inline void ext_store(device ulong *p, Ext e) {
    p[0] = e.a; p[1] = e.b; p[2] = e.c;
}

// One sumcheck round: `out[j] = a[2j] + r*(a[2j+1] - a[2j])`.
// Everything is passed as flat `ulong` triples: no struct layout is shared with the host.
kernel void fold_ext(device const ulong *a   [[buffer(0)]],
                     device ulong *out       [[buffer(1)]],
                     device const ulong *r   [[buffer(2)]],
                     uint gid                [[thread_position_in_grid]])
{
    Ext a0 = ext_load(a + 6u * gid);
    Ext a1 = ext_load(a + 6u * gid + 3u);
    Ext rr = ext_load(r);
    ext_store(out + 3u * gid, ext_add(a0, ext_mul(rr, ext_sub(a1, a0))));
}

// ---- bisect kernels: each adds one thing to the working copy kernel ----

// 1: copy, 4-buffer signature (host may leave buffers unbound).
kernel void bis_copy(device const ulong *a [[buffer(0)]],
                     device const ulong *b [[buffer(1)]],
                     device ulong *out     [[buffer(2)]],
                     device const uint *args [[buffer(3)]],
                     uint gid [[thread_position_in_grid]])
{
    out[gid] = a[gid];
}

// 2: read the *second* input buffer.
kernel void bis_read_b(device const ulong *a [[buffer(0)]],
                       device const ulong *b [[buffer(1)]],
                       device ulong *out     [[buffer(2)]],
                       device const uint *args [[buffer(3)]],
                       uint gid [[thread_position_in_grid]])
{
    out[gid] = b[gid];
}

// 3: one Goldilocks multiply of two inputs.
kernel void bis_mul(device const ulong *a [[buffer(0)]],
                    device const ulong *b [[buffer(1)]],
                    device ulong *out     [[buffer(2)]],
                    device const uint *args [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    out[gid] = gl_mul(a[gid], b[gid]);
}

// 4: the eq loop, with a single point coordinate from `b`.
kernel void bis_eq(device const ulong *a [[buffer(0)]],
                   device const ulong *b [[buffer(1)]],
                   device ulong *out     [[buffer(2)]],
                   device const uint *args [[buffer(3)]],
                   uint gid [[thread_position_in_grid]])
{
    Ext p = {b[gid], 0ul, 0ul};
    Ext one = {1ul, 0ul, 0ul};
    Ext one_minus_p = ext_sub(one, p);
    bool bit = (gid & 1u) != 0u;
    Ext bt = bit ? one : (Ext){0ul, 0ul, 0ul};
    Ext ob = bit ? (Ext){0ul, 0ul, 0ul} : one;
    Ext e = ext_add(ext_mul(p, bt), ext_mul(one_minus_p, ob));
    Ext ve = {a[gid], 0ul, 0ul};
    Ext r = ext_mul(e, ve);
    out[3u * gid] = r.a;
    out[3u * gid + 1u] = r.b;
    out[3u * gid + 2u] = r.c;
}

// 4b: read the small args buffer.
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

// 5: threadgroup reduction to out[0].
kernel void bis_reduce(device const ulong *a [[buffer(0)]],
                       device const ulong *b [[buffer(1)]],
                       device ulong *out     [[buffer(2)]],
                       device const uint *args [[buffer(3)]],
                       uint gid [[thread_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]],
                       uint gsize [[threads_per_threadgroup]])
{
    threadgroup Ext partial[1024];
    partial[lid] = (Ext){a[gid], 0ul, 0ul};
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = gsize / 2u; st > 0u; st >>= 1u) {
        if (lid < st) { partial[lid] = ext_add(partial[lid], partial[lid + st]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
        out[0] = partial[0].a;
        out[1] = partial[0].b;
        out[2] = partial[0].c;
    }
}

// Debug: trivial copy, to isolate buffer plumbing from arithmetic.
kernel void copy_u64(device const ulong *in [[buffer(0)]],
                     device ulong *out      [[buffer(1)]],
                     uint gid               [[thread_position_in_grid]])
{
    out[gid] = in[gid];
}

// Debug: one Ext per thread, no reduction (host sums them).
kernel void mle_terms(device const ulong *values [[buffer(0)]],
                      device const ulong *point  [[buffer(1)]],
                      device ulong *out          [[buffer(2)]],
                      device const uint *args    [[buffer(3)]],
                      uint gid [[thread_position_in_grid]])
{
    uint m = args[0];
    Ext e = {1ul, 0ul, 0ul};
    for (uint k = 0; k < m; ++k) {
        Ext p = {point[3u * k], point[3u * k + 1u], point[3u * k + 2u]};
        Ext one = {1ul, 0ul, 0ul};
        Ext one_minus_p = ext_sub(one, p);
        bool bit = ((gid >> k) & 1u) != 0u;
        Ext bt = bit ? one : (Ext){0ul, 0ul, 0ul};
        Ext ob = bit ? (Ext){0ul, 0ul, 0ul} : one;
        e = ext_mul(e, ext_add(ext_mul(p, bt), ext_mul(one_minus_p, ob)));
    }
    Ext ve = {values[gid], 0ul, 0ul};
    ext_store(out + 3u * gid, ext_mul(e, ve));
}

// The eq factor alone, summed: must be 1 (Σ_i eq(point,i) = 1).
kernel void mle_e_only(device const ulong *values [[buffer(0)]],
                       device const ulong *point  [[buffer(1)]],
                       device ulong *out          [[buffer(2)]],
                       device const uint *args    [[buffer(3)]],
                       uint gid  [[thread_position_in_grid]],
                       uint lid  [[thread_position_in_threadgroup]],
                       uint gsize[[threads_per_threadgroup]])
{
    threadgroup Ext partial[1024];
    Ext zero = {0ul, 0ul, 0ul};
    Ext one = {1ul, 0ul, 0ul};

    // Fully unrolled, three factors, no array, no loop.
    Ext p0 = {point[0], point[1], point[2]};
    Ext p1 = {point[3], point[4], point[5]};
    Ext p2 = {point[6], point[7], point[8]};

    Ext f0 = ((gid & 1u) != 0u) ? p0 : ext_sub(one, p0);
    Ext f1 = ((gid & 2u) != 0u) ? p1 : ext_sub(one, p1);
    Ext f2 = ((gid & 4u) != 0u) ? p2 : ext_sub(one, p2);

    Ext e = ext_mul(ext_mul(f0, f1), f2);
    if (gid == 0u) { out[3] = e.a; out[4] = f0.a; out[5] = f1.a; }
    partial[lid] = e;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = gsize / 2u; st > 0u; st >>= 1u) {
        if (lid < st) { partial[lid] = ext_add(partial[lid], partial[lid + st]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
        out[0] = partial[0].a;
        out[1] = partial[0].b;
        out[2] = partial[0].c;
    }
}

// The value alone, summed: must be Σ values.
kernel void mle_v_only(device const ulong *values [[buffer(0)]],
                       device const ulong *point  [[buffer(1)]],
                       device ulong *out          [[buffer(2)]],
                       device const uint *args    [[buffer(3)]],
                       uint gid  [[thread_position_in_grid]],
                       uint lid  [[thread_position_in_threadgroup]],
                       uint gsize[[threads_per_threadgroup]])
{
    threadgroup Ext partial[1024];
    partial[lid] = (Ext){values[gid], 0ul, 0ul};
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = gsize / 2u; st > 0u; st >>= 1u) {
        if (lid < st) { partial[lid] = ext_add(partial[lid], partial[lid + st]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
        out[0] = partial[0].a;
        out[1] = partial[0].b;
        out[2] = partial[0].c;
    }
}

// mle_reduce with `m` and the guard hardcoded, to isolate arg-reading vs the loop.
kernel void mle_fixed(device const ulong *values [[buffer(0)]],
                      device const ulong *point  [[buffer(1)]],
                      device ulong *out          [[buffer(2)]],
                      device const uint *args    [[buffer(3)]],
                      uint gid  [[thread_position_in_grid]],
                      uint lid  [[thread_position_in_threadgroup]],
                      uint gsize[[threads_per_threadgroup]])
{
    threadgroup Ext partial[1024];
    Ext acc = {0ul, 0ul, 0ul};
    Ext zero = {0ul, 0ul, 0ul};
    Ext one = {1ul, 0ul, 0ul};
    Ext e = {1ul, 0ul, 0ul};
    for (uint k = 0; k < 3u; ++k) {
        Ext p = {point[3u * k], point[3u * k + 1u], point[3u * k + 2u]};
        Ext one_minus_p = ext_sub(one, p);
        Ext bt = zero;
        Ext ob = one;
        if (((gid >> k) & 1u) != 0u) { bt = one; ob = zero; }
        e = ext_mul(e, ext_add(ext_mul(p, bt), ext_mul(one_minus_p, ob)));
    }
    Ext ve = {values[gid], 0ul, 0ul};
    acc = ext_mul(e, ve);

    partial[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = gsize / 2u; st > 0u; st >>= 1u) {
        if (lid < st) { partial[lid] = ext_add(partial[lid], partial[lid + st]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
        ulong base = 3ul * (gid / gsize);
        out[base] = partial[0].a;
        out[base + 1ul] = partial[0].b;
        out[base + 2ul] = partial[0].c;
    }
}

kernel void mle_reduce(device const ulong *values [[buffer(0)]],
                       device const ulong *point  [[buffer(1)]],
                       device ulong *out          [[buffer(2)]],
                       device const uint *args    [[buffer(3)]],
                       uint gid  [[thread_position_in_grid]],
                       uint lid  [[thread_position_in_threadgroup]],
                       uint gsize[[threads_per_threadgroup]])
{
    threadgroup Ext partial[1024];

    uint m = args[0];
    uint n = args[1];
    Ext acc = {0ul, 0ul, 0ul};
    if (gid < n) {
        // eq(point, gid) = prod_k (p_k*b_k + (1-p_k)(1-b_k))
        Ext zero = {0ul, 0ul, 0ul};
        Ext one = {1ul, 0ul, 0ul};
        Ext e = {1ul, 0ul, 0ul};
        for (uint k = 0; k < m; ++k) {
            Ext p = {point[3u * k], point[3u * k + 1u], point[3u * k + 2u]};
            Ext one_minus_p = ext_sub(one, p);
            Ext bt = zero;
            Ext ob = one;
            if (((gid >> k) & 1u) != 0u) { bt = one; ob = zero; }
            e = ext_mul(e, ext_add(ext_mul(p, bt), ext_mul(one_minus_p, ob)));
        }
        Ext ve = {values[gid], 0ul, 0ul};
        acc = ext_mul(e, ve);
    }

    partial[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = gsize / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            partial[lid] = ext_add(partial[lid], partial[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
        ulong base = 3ul * (gid / gsize);
        out[base] = partial[0].a;
        out[base + 1ul] = partial[0].b;
        out[base + 2ul] = partial[0].c;
    }
}
"#;

struct Gpu {
    device: Device,
    queue: CommandQueue,
    /// MLE reduction kernel.
    pipe: ComputePipelineState,
    /// One sumcheck fold.
    fold_pipe: ComputePipelineState,
    /// Per-thread terms (debug path).
    terms_pipe: ComputePipelineState,
    /// Trivial copy (debug path).
    copy_pipe: ComputePipelineState,
    bis: Vec<(&'static str, ComputePipelineState)>,
}

static GPU: OnceLock<Option<Gpu>> = OnceLock::new();

fn init() -> Option<&'static Gpu> {
    GPU.get_or_init(|| {
        let device = Device::system_default()?;
        let opts = CompileOptions::new();
        let library = device
            .new_library_with_source(MSL, &opts)
            .map_err(|e| eprintln!("flashprove: Metal shader compile failed: {e}"))
            .ok()?;
        let func = library
            .get_function("mle_reduce", None)
            .map_err(|e| eprintln!("flashprove: Metal function lookup failed: {e}"))
            .ok()?;
        let pipe = device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| eprintln!("flashprove: Metal pipeline failed: {e}"))
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
        for name in ["bis_copy", "bis_read_b", "bis_mul", "bis_eq", "bis_args", "bis_reduce", "mle_v_only", "mle_e_only", "mle_fixed", "mle_reduce"] {
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
            pipe,
            fold_pipe,
            terms_pipe,
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
        let vbuf: Buffer = self.device.new_buffer_with_data(
            vals.as_ptr() as *const std::ffi::c_void,
            (n * 8) as u64,
            opts,
        );
        let pbuf: Buffer = self.device.new_buffer_with_data(
            pts.as_ptr() as *const std::ffi::c_void,
            (pts.len() * 8) as u64,
            opts,
        );
        let obuf = self.device.new_buffer(groups * 24, opts);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipe);
        enc.set_buffer(0, Some(&vbuf), 0);
        enc.set_buffer(1, Some(&pbuf), 0);
        enc.set_buffer(2, Some(&obuf), 0);
        let args: [u32; 2] = [m as u32, n as u32];
        let args_buf = self.device.new_buffer_with_data(
            args.as_ptr() as *const std::ffi::c_void,
            8,
            opts,
        );
        enc.set_buffer(3, Some(&args_buf), 0);
        enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(GROUP, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // Sum the per-group partials on the host (there are only a few thousand).
        if std::env::var("FLASHPROVE_GPU_DEBUG").is_ok() {
            let raw = unsafe {
                std::slice::from_raw_parts(obuf.contents() as *const u64, (groups as usize) * 3)
            };
            eprintln!(
                "  [gpu] n={n} m={m} groups={groups} status={:?} partial0={:?}",
                cmd.status(),
                &raw[..3.min(raw.len())]
            );
        }
        let out = unsafe {
            std::slice::from_raw_parts(obuf.contents() as *const u64, (groups as usize) * 3)
        };
        let mut acc = Ext::ZERO;
        for g in 0..groups as usize {
            let coeffs = [
                Fp::new(out[3 * g]),
                Fp::new(out[3 * g + 1]),
                Fp::new(out[3 * g + 2]),
            ];
            acc += <Ext as BasedVectorSpace<Fp>>::from_basis_coefficients_slice(&coeffs).unwrap();
        }
        acc
    }
}

impl Gpu {
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
            "mle_e_only" => got[0] == 1,
            "mle_fixed" | "mle_reduce" => {
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
