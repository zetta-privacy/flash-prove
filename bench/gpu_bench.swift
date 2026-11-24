// Field-arithmetic GPU benchmark for the FlashProve prover.
//
// The single hottest loop in the prover is `out[i] = a[i]*b[i] + c[i]` over large
// field arrays (elementwise products in the sumchecks, folds, eq tables, and the
// 9 base-field multiplies of one cubic-extension multiply). This measures exactly
// that loop, on the CPU and on the GPU, for both candidate fields:
//
//   Goldilocks  p = 2^64 - 2^32 + 1   (64-bit: emulated on the GPU's 32-bit ALUs)
//   BabyBear    p = 2^31 - 2^27 + 1   (31-bit: one native 32x32->64 multiply)
//
// Build:  swiftc -O -o bench/gpu_bench bench/gpu_bench.swift
// Run:    ./bench/gpu_bench

import Foundation
import Metal

// MARK: - Metal source

let msl = """
#include <metal_stdlib>
using namespace metal;

// ---------------- Goldilocks: p = 2^64 - 2^32 + 1 ----------------
constant ulong GL_ORDER = 0xFFFFFFFF00000001ul;
constant ulong GL_EPS   = 0x00000000FFFFFFFFul;

static inline ulong gl_mul(ulong a, ulong b) {
    uint a0 = (uint)a, a1 = (uint)(a >> 32);
    uint b0 = (uint)b, b1 = (uint)(b >> 32);

    uint p00 = a0 * b0, p00h = mulhi(a0, b0);
    uint p01 = a0 * b1, p01h = mulhi(a0, b1);
    uint p10 = a1 * b0, p10h = mulhi(a1, b0);
    uint p11 = a1 * b1, p11h = mulhi(a1, b1);

    ulong t0 = ((ulong)p00h << 32) | (ulong)p00;       // a0*b0
    ulong t2 = ((ulong)p11h << 32) | (ulong)p11;       // a1*b1

    // (a0*b1 + a1*b0) as a 64-bit-ish sum: both halves matter.
    uint  s_lo = p01 + p10;
    uint  s_carry = (s_lo < p01) ? 1u : 0u;
    ulong s_hi = (ulong)p01h + (ulong)p10h + (ulong)s_carry;

    ulong lo = t0 + ((ulong)s_lo << 32);               // bits  0..63
    ulong carry = (lo < t0) ? 1ul : 0ul;
    ulong hi = t2 + s_hi + carry;                      // bits 64..127

    // p3's reduce128: 2^64 = EPS (mod p)
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

static inline ulong gl_add(ulong a, ulong b) {
    ulong t = a + b;
    if (t < a) { t += GL_EPS; }
    else if (t >= GL_ORDER) { t -= GL_ORDER; }
    return t;
}

kernel void gl_mul_add(device const ulong *a [[buffer(0)]],
                       device const ulong *b [[buffer(1)]],
                       device const ulong *c [[buffer(2)]],
                       device ulong *out      [[buffer(3)]],
                       constant uint &n       [[buffer(4)]],
                       uint gid               [[thread_position_in_grid]]) {
    if (gid >= n) return;
    out[gid] = gl_add(gl_mul(a[gid], b[gid]), c[gid]);
}

// ---------------- BabyBear (Montgomery, 32-bit) ----------------
constant uint BB_P = 0x78000001u;
constant uint BB_M = 0x77FFFFFFu;   // -p^{-1} mod 2^32

static inline uint bb_mul(uint a, uint b) {
    uint lo = a * b;
    uint hi = mulhi(a, b);
    uint m  = lo * BB_M;
    uint t  = m * BB_P;
    uint t_hi = mulhi(m, BB_P);
    uint lo2 = lo + t;
    uint u = hi + t_hi + ((lo2 < lo) ? 1u : 0u);
    if (u >= BB_P) { u -= BB_P; }
    return u;
}

static inline uint bb_add(uint a, uint b) {
    uint t = a + b;
    if (t >= BB_P) { t -= BB_P; }
    return t;
}

kernel void bb_mul_add(device const uint *a [[buffer(0)]],
                       device const uint *b [[buffer(1)]],
                       device const uint *c [[buffer(2)]],
                       device uint *out       [[buffer(3)]],
                       constant uint &n       [[buffer(4)]],
                       uint gid               [[thread_position_in_grid]]) {
    if (gid >= n) return;
    out[gid] = bb_add(bb_mul(a[gid], b[gid]), c[gid]);
}
"""

// MARK: - CPU reference implementations

let GL_ORDER: UInt64 = 0xFFFF_FFFF_0000_0001
let GL_EPS: UInt64 = 0x0000_0000_FFFF_FFFF
let BB_P: UInt32 = 0x7800_0001
let BB_M: UInt32 = 0x77FF_FFFF

print("checking BabyBear Montgomery constant...")
do {
    var inv: UInt32 = 1
    for _ in 0..<5 { inv = inv &* (2 &- BB_P &* inv) } // Newton, mod 2^32
    let negInv = (0 &- inv)
    print("  -p^-1 mod 2^32 = 0x\(String(negInv, radix: 16))  (expected 0x\(String(BB_M, radix: 16)))")
    precondition(negInv == BB_M, "hardcoded Montgomery constant is wrong")
}

var cpuSink64: UInt64 = 0
var cpuSink32: UInt32 = 0

@inline(__always) func glMul(_ a: UInt64, _ b: UInt64) -> UInt64 {
    let p = a.multipliedFullWidth(by: b)
    let lo = p.low, hi = p.high
    let xHiHi = UInt64(hi >> 32)
    let xHiLo = UInt64(UInt32(truncatingIfNeeded: hi))
    var r = lo &- xHiHi
    if lo < xHiHi { r = r &- GL_EPS }
    let t1 = xHiLo &* GL_EPS
    var t = r &+ t1
    if t < r { t = t &+ GL_EPS } else if t >= GL_ORDER { t = t &- GL_ORDER }
    return t
}

@inline(__always) func glAdd(_ a: UInt64, _ b: UInt64) -> UInt64 {
    var t = a &+ b
    if t < a { t = t &+ GL_EPS } else if t >= GL_ORDER { t = t &- GL_ORDER }
    return t
}

@inline(__always) func bbMul(_ a: UInt32, _ b: UInt32) -> UInt32 {
    let lo = a &* b
    let hi = a.multipliedFullWidth(by: b).high
    let m = lo &* BB_M
    let t = m &* BB_P
    let tHi = m.multipliedFullWidth(by: BB_P).high
    let lo2 = lo &+ t
    var u = hi &+ tHi &+ (lo2 < lo ? 1 : 0)
    if u >= BB_P { u = u &- BB_P }
    return u
}

@inline(__always) func bbAdd(_ a: UInt32, _ b: UInt32) -> UInt32 {
    var t = a &+ b
    if t >= BB_P { t = t &- BB_P }
    return t
}

// MARK: - Host

setvbuf(stdout, nil, _IONBF, 0)  // unbuffered: report progress as it happens
func err(_ m: String) { FileHandle.standardError.write((m + "\n").data(using: .utf8)!) }

let device = MTLCreateSystemDefaultDevice()!
print("device: \(device.name)  (unified memory: \(device.hasUnifiedMemory))")

let library: MTLLibrary
do {
    library = try device.makeLibrary(source: msl, options: nil)
} catch {
    fatalError("Metal shader compile failed: \(error)")
}

func pipeline(_ name: String) -> MTLComputePipelineState {
    try! device.makeComputePipelineState(function: library.makeFunction(name: name)!)
}

let glPipe = pipeline("gl_mul_add")
let bbPipe = pipeline("bb_mul_add")

func gpuTime(_ pipe: MTLComputePipelineState, n: Int, reps: Int,
             a: MTLBuffer, b: MTLBuffer, c: MTLBuffer, out: MTLBuffer) -> (perOp: Double, latencyNs: Double) {
    var nn = UInt32(n)
    let grid = MTLSize(width: n, height: 1, depth: 1)
    let group = MTLSize(width: 256, height: 1, depth: 1)
    let queue = device.makeCommandQueue()!

    func encode(_ cb: MTLCommandBuffer) {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipe)
        enc.setBuffer(a, offset: 0, index: 0)
        enc.setBuffer(b, offset: 0, index: 1)
        enc.setBuffer(c, offset: 0, index: 2)
        enc.setBuffer(out, offset: 0, index: 3)
        enc.setBytes(&nn, length: 4, index: 4)
        enc.dispatchThreads(grid, threadsPerThreadgroup: group)
        enc.endEncoding()
    }

    // warm up
    for _ in 0..<3 {
        let cb = queue.makeCommandBuffer()!
        encode(cb)
        cb.commit()
        cb.waitUntilCompleted()
    }

    // throughput: commit all, wait on the last (queues are ordered)
    let t0 = DispatchTime.now().uptimeNanoseconds
    var last: MTLCommandBuffer?
    for _ in 0..<reps {
        let cb = queue.makeCommandBuffer()!
        encode(cb)
        cb.commit()
        last = cb
    }
    last!.waitUntilCompleted()
    let ns = DispatchTime.now().uptimeNanoseconds - t0

    // latency: one dispatch, measured alone
    let t1 = DispatchTime.now().uptimeNanoseconds
    let cb = queue.makeCommandBuffer()!
    encode(cb)
    cb.commit()
    cb.waitUntilCompleted()
    let lat = DispatchTime.now().uptimeNanoseconds - t1

    return (Double(ns) / Double(reps), Double(lat))
}

/// CPU baseline: the same kernel shape — read a,b,c, write out — so the comparison is
/// honest. `out` is allocated by the caller and the sink keeps the writes alive.
func cpuTime64(_ n: Int, reps: Int, _ a: [UInt64], _ b: [UInt64], _ c: [UInt64], _ out: inout [UInt64]) -> Double {
    var ns: UInt64 = 0
    a.withUnsafeBufferPointer { pa in b.withUnsafeBufferPointer { pb in c.withUnsafeBufferPointer { pc in
        out.withUnsafeMutableBufferPointer { po in
            let t0 = DispatchTime.now().uptimeNanoseconds
            for _ in 0..<reps {
                for i in 0..<n { po[i] = glAdd(glMul(pa[i], pb[i]), pc[i]) }
            }
            ns = DispatchTime.now().uptimeNanoseconds - t0
        }
    }}}
    cpuSink64 &+= out[0]
    return Double(ns) / Double(reps)
}

func cpuTime32(_ n: Int, reps: Int, _ a: [UInt32], _ b: [UInt32], _ c: [UInt32], _ out: inout [UInt32]) -> Double {
    var ns: UInt64 = 0
    a.withUnsafeBufferPointer { pa in b.withUnsafeBufferPointer { pb in c.withUnsafeBufferPointer { pc in
        out.withUnsafeMutableBufferPointer { po in
            let t0 = DispatchTime.now().uptimeNanoseconds
            for _ in 0..<reps {
                for i in 0..<n { po[i] = bbAdd(bbMul(pa[i], pb[i]), pc[i]) }
            }
            ns = DispatchTime.now().uptimeNanoseconds - t0
        }
    }}}
    cpuSink32 &+= out[0]
    return Double(ns) / Double(reps)
}

// MARK: - Run

let sizes: [Int] = [1 << 18, 1 << 20, 1 << 22, 1 << 24]
print("""

size        field       GPU ns/elem  GPU Gop/s   CPU ns/elem  CPU Gop/s  speedup  latency-us
""")

for n in sizes {
    // ---- Goldilocks ----
    var ga = [UInt64](repeating: 0, count: n)
    var gb = [UInt64](repeating: 0, count: n)
    var gc = [UInt64](repeating: 0, count: n)
    for i in 0..<n {
        ga[i] = UInt64.random(in: 0..<GL_ORDER)
        gb[i] = UInt64.random(in: 0..<GL_ORDER)
        gc[i] = UInt64.random(in: 0..<GL_ORDER)
    }
    let gBufA = device.makeBuffer(bytes: ga, length: n * 8, options: .storageModeShared)!
    let gBufB = device.makeBuffer(bytes: gb, length: n * 8, options: .storageModeShared)!
    let gBufC = device.makeBuffer(bytes: gc, length: n * 8, options: .storageModeShared)!
    let gBufO = device.makeBuffer(length: n * 8, options: .storageModeShared)!

    // correctness: GPU vs CPU on the first 256 elements
    var nn = UInt32(n)
    let queue = device.makeCommandQueue()!
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(glPipe)
    enc.setBuffer(gBufA, offset: 0, index: 0); enc.setBuffer(gBufB, offset: 0, index: 1)
    enc.setBuffer(gBufC, offset: 0, index: 2); enc.setBuffer(gBufO, offset: 0, index: 3)
    enc.setBytes(&nn, length: 4, index: 4)
    enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    let gpuOut = gBufO.contents().bindMemory(to: UInt64.self, capacity: n)
    var bad = 0
    for i in 0..<min(n, 256) {
        let want = glAdd(glMul(ga[i], gb[i]), gc[i])
        if gpuOut[i] != want {
            bad += 1
            if bad <= 3 { err("  GL MISMATCH at \(i): gpu=\(gpuOut[i]) cpu=\(want)") }
        }
    }
    if bad > 0 { err("  goldilocks kernel wrong (\(bad)/256)"); exit(1) }

    var reps = min(4096, max(1, (1 << 26) / n))
    let glGpu = gpuTime(glPipe, n: n, reps: reps, a: gBufA, b: gBufB, c: gBufC, out: gBufO)
    FileHandle.standardError.write("   gl gpu done\n".data(using: .utf8)!)
    var glOut = [UInt64](repeating: 0, count: n)
    let glCpu = cpuTime64(n, reps: min(reps, 16), ga, gb, gc, &glOut)
    FileHandle.standardError.write("   gl cpu done\n".data(using: .utf8)!)
    print(String(format: "2^%-4d Goldilocks %9.3f %10.2f %11.3f %10.2f %8.1fx %10.1f",
                 Int(log2(Double(n))),
                 glGpu.perOp / Double(n), Double(n) / glGpu.perOp / 1.0,
                 glCpu / Double(n), Double(n) / glCpu, glCpu / glGpu.perOp,
                 glGpu.latencyNs / 1000))

    // ---- BabyBear ----
    var ba = [UInt32](repeating: 0, count: n)
    var bb = [UInt32](repeating: 0, count: n)
    var bc = [UInt32](repeating: 0, count: n)
    for i in 0..<n {
        ba[i] = UInt32.random(in: 0..<BB_P)
        bb[i] = UInt32.random(in: 0..<BB_P)
        bc[i] = UInt32.random(in: 0..<BB_P)
    }
    let bBufA = device.makeBuffer(bytes: ba, length: n * 4, options: .storageModeShared)!
    let bBufB = device.makeBuffer(bytes: bb, length: n * 4, options: .storageModeShared)!
    let bBufC = device.makeBuffer(bytes: bc, length: n * 4, options: .storageModeShared)!
    let bBufO = device.makeBuffer(length: n * 4, options: .storageModeShared)!

    let cb2 = queue.makeCommandBuffer()!
    let enc2 = cb2.makeComputeCommandEncoder()!
    enc2.setComputePipelineState(bbPipe)
    enc2.setBuffer(bBufA, offset: 0, index: 0); enc2.setBuffer(bBufB, offset: 0, index: 1)
    enc2.setBuffer(bBufC, offset: 0, index: 2); enc2.setBuffer(bBufO, offset: 0, index: 3)
    enc2.setBytes(&nn, length: 4, index: 4)
    enc2.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    enc2.endEncoding(); cb2.commit(); cb2.waitUntilCompleted()
    let bbOut = bBufO.contents().bindMemory(to: UInt32.self, capacity: n)
    var bad2 = 0
    for i in 0..<min(n, 256) {
        let want = bbAdd(bbMul(ba[i], bb[i]), bc[i])
        if bbOut[i] != want {
            bad2 += 1
            if bad2 <= 3 { err("  BB MISMATCH at \(i): gpu=\(bbOut[i]) cpu=\(want)") }
        }
    }
    if bad2 > 0 { err("  babybear kernel wrong (\(bad2)/256)"); exit(1) }

    let bbGpu = gpuTime(bbPipe, n: n, reps: reps, a: bBufA, b: bBufB, c: bBufC, out: bBufO)
    var bbOutCpu = [UInt32](repeating: 0, count: n)
    let bbCpu = cpuTime32(n, reps: min(reps, 16), ba, bb, bc, &bbOutCpu)
    print(String(format: "2^%-4d BabyBear   %9.3f %10.2f %11.3f %10.2f %8.1fx %10.1f",
                 Int(log2(Double(n))),
                 bbGpu.perOp / Double(n), Double(n) / bbGpu.perOp,
                 bbCpu / Double(n), Double(n) / bbCpu, bbCpu / bbGpu.perOp,
                 bbGpu.latencyNs / 1000))
    reps = max(reps, 1)
}

// Dispatch overhead: how big must an array be before the GPU wins?
do {
    let n = 4096
    let a = device.makeBuffer(length: n * 8, options: .storageModeShared)!
    let b = device.makeBuffer(length: n * 8, options: .storageModeShared)!
    let c = device.makeBuffer(length: n * 8, options: .storageModeShared)!
    let o = device.makeBuffer(length: n * 8, options: .storageModeShared)!
    let r = gpuTime(glPipe, n: n, reps: 64, a: a, b: b, c: c, out: o)
    print(String(format: "\none empty dispatch of n=4096: %.0f us  (that is the floor for any per-claim work)",
                 r.latencyNs / 1000))
}

print("""

Notes
  * "Gop/s" = field multiply-adds per nanosecond (1e9 ops/s).
  * One cubic-extension multiply (Ext, used in every sumcheck) is 9 base
    multiply-adds; a quartic one is 16.
  * CPU column is a single core, scalar, same algorithm.
""")
