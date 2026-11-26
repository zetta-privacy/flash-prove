#!/usr/bin/env python3
"""Run the integer model described by a ModelSpec, emitting the trace.

This mirrors `src/spec.rs` exactly: round-half-away-from-zero, q15 fixed-point,
and the same silu/softmax/rope/rmsnorm lookups, so the emitted trace is accepted
by the Rust verifier. The op graph is read from a spec (data), so this is
model-agnostic.

Every op that produces a cell may carry an integer right shift (a *requant*), so
all committed values stay small (`SHIFT_TARGET` bits). Shifts are per *role*
(`layer.*.expert.*.gate_proj`), never per op instance: the terms of a
`weighted_sum` must share a scale or the sum is meaningless.
"""

import argparse
import json
import math
import struct

import numpy as np


def round_half_away_scalar(x):
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def round_half_away(x):
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5)).astype(np.int64)


# Fixed-point bits for the quantized reciprocal (matches `src/spec.rs::RECIP_Q`).
RECIP_Q = 14

# Bits every committed cell is calibrated to (see `python/calibrate.py`).
SHIFT_TARGET = 11


def exp_f64(x):
    # Rust f64::exp returns +inf on overflow (IEEE 754); math.exp raises instead.
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


def requant(x, k):
    half = 1 << (k - 1)
    return np.where(x >= 0, (x + half) >> k, -(((-x) + half) >> k)).astype(np.int64)


def rq(x, k):
    """`requant`, tolerating the no-op shift `k <= 0`."""
    return x if k <= 0 else requant(x, k)


def _norm(name: str) -> str:
    """`layer.0.expert.12.gate_proj` -> `layer.*.expert.*.gate_proj`.

    Only whole numeric path *components* are wildcarded: `residual_1` and
    `residual_2` keep separate roles, because their pre-shift magnitudes differ by
    2^6 in an uncalibrated model and one shared shift makes the calibration
    oscillate.
    """
    return ".".join("*" if part.isdigit() else part for part in name.split("."))


def op_role(op: dict) -> str:
    """The shift-sharing role of an op: ops in one role use one shift.

    `add` is the exception: the residual stream accumulates across layers, so a single
    shift for all of them cannot keep every layer in range. Each layer keeps its own
    (the layer index is part of the role; the expert index still collapses).
    """
    kind = op["op"]
    if kind == "matmul":
        return "mm:" + _norm(op["weight"])
    if kind == "mul":
        return "mul:" + _norm(op["output"])
    if kind == "add":
        parts = op["output"].split(".")
        kept = [p if (p.isdigit() and i == 1) else ("*" if p.isdigit() else p)
                for i, p in enumerate(parts)]
        return "add:" + ".".join(kept)
    if kind == "scores":
        return "scores"
    return kind


# --- precomputed transcendental tables (bit-identical to Rust's f64 libm) ------

_silu: np.ndarray = np.empty(0, dtype=np.int64)
_exp: np.ndarray = np.empty(0, dtype=np.int64)


def _init_tables():
    global _silu, _exp
    if _silu.size > 0:
        return
    lo = -(1 << 15)
    # silu over the clamped domain [-2^15, 2^15 - 1].
    xs = np.arange(lo, (1 << 15), dtype=np.int64)
    _silu = np.array(
        [round_half_away_scalar(float(v) / (1.0 + exp_f64(-float(v)))) for v in xs],
        dtype=np.int64,
    )
    # exp_q15 over the clamped domain [-2^15, 0].
    xe = np.arange(lo, 1, dtype=np.int64)
    _exp = np.array(
        [round_half_away_scalar(math.exp(float(v)) * 32768.0) for v in xe],
        dtype=np.int64,
    )


def silu(x):
    _init_tables()
    xc = np.clip(x, -(1 << 15), (1 << 15) - 1).astype(np.int64)
    return _silu[xc + (1 << 15)]


def exp_q15(x):
    _init_tables()
    xc = np.clip(x, -(1 << 15), 0).astype(np.int64)
    return _exp[xc + (1 << 15)]


_rope_cache = {}


def rope_tables(L, d, theta):
    key = (L, d, theta)
    t = _rope_cache.get(key)
    if t is not None:
        return t
    pairs = d // 2
    C = np.empty((L, pairs), dtype=np.int64)
    S = np.empty((L, pairs), dtype=np.int64)
    for li in range(L):
        for j in range(pairs):
            i = 2 * j
            angle = li * math.pow(theta, -i / d)
            C[li, j] = round_half_away_scalar(math.cos(angle) * 32768.0)
            S[li, j] = round_half_away_scalar(math.sin(angle) * 32768.0)
    _rope_cache[key] = (C, S)
    return C, S


def rmsnorm(x, gamma, eps):
    gamma = np.asarray(gamma, dtype=np.int64).reshape(-1)
    d = x.shape[-1]
    flat = x.reshape(-1, d)
    out = np.empty(flat.shape, dtype=np.int64)
    for b in range(flat.shape[0]):
        row = flat[b]
        # sequential integer sum, matching Rust's `Iterator::sum::<i64>()`.
        sum_sq = 0
        for v in row:
            sum_sq += int(v) * int(v)
        mean_sq = sum_sq / d
        inv_q = round_half_away_scalar((1 << RECIP_Q) / math.sqrt(mean_sq + eps))
        out[b] = requant(row.astype(np.int64) * gamma * inv_q, RECIP_Q)
    return out.reshape(x.shape)


def write_binary_trace(trace, path):
    """Flat binary layout, read on demand by `src/model/trace_file.rs`.

    magic "FPT1", u32 cell count, i64 head token, then per cell
    u16 name_len, name, u32 rows, u32 cols, i32 * rows * cols.

    Values are `i32`: every traced value is committed and the field guard requires
    `|v| < 2^31` (BabyBear/Goldilocks signed range), so 32 bits is lossless and half
    the bytes of `i64`.
    """
    seg = trace["segments"][0]
    cells = seg["cells"]
    with open(path, "wb") as f:
        f.write(b"FPT1")
        f.write(struct.pack("<I", len(cells)))
        f.write(struct.pack("<q", seg["head_token"] if seg["head_token"] is not None else -1))
        for c in cells:
            name = c["name"].encode("utf-8")
            f.write(struct.pack("<H", len(name)))
            f.write(name)
            f.write(struct.pack("<II", int(c["rows"]), int(c["cols"])))
            vals = np.asarray(c["values"])
            if vals.size and int(np.abs(vals).max()) >= (1 << 31):
                raise ValueError(f"{c['name']}: value does not fit the 31-bit field")
            f.write(vals.astype("<i4").tobytes())


def run(spec, weights, window, output_pos=None, stats=None) -> dict:
    """Run the op graph. With `stats`, record each role's largest pre-shift value
    (used by `calibrate.py`; `stats["scores"]` holds the largest safe score shift).
    """
    env = {}
    selections = {}
    if stats is None:
        stats = {}

    def get(name):
        return env[name]

    def note(role, values):
        m = float(np.abs(values).max()) if np.size(values) else 0.0
        # Always insert: a role whose values are all zero still needs a shift in the
        # table, or `gen_spec` rejects the table as a typo on the next round.
        stats[role] = max(stats.get(role, 0.0), m)

    for op in spec["ops"]:
        kind = op["op"]
        role = op_role(op)
        if kind == "embed":
            t = weights[op["weight"]]  # [V, hidden]
            env[op["output"]] = t[window].astype(np.int64)
        elif kind == "norm":
            env[op["output"]] = rmsnorm(get(op["input"]), weights[op["weight"]], op["eps"])
        elif kind == "matmul":
            x = get(op["input"])
            w = weights[op["weight"]]  # [out, in] int8
            # float64 matmul is BLAS-fast and exact: products/accumulations stay
            # far below 2^53, so the float64 result is the exact integer sum.
            acc = x.astype(np.float64) @ w.astype(np.float64).T
            note(role, acc)
            env[op["output"]] = rq(round_half_away(acc).astype(np.int64), int(op["shift"]))
        elif kind == "rope":
            v = get(op["input"]).astype(np.int64)  # [L, H, d] (or [L, d])
            L, d = v.shape[0], v.shape[-1]
            C, S = rope_tables(L, d, op["theta"])
            ve = v[..., 0::2]
            vo = v[..., 1::2]
            c = C[:, None, :] if v.ndim == 3 else C
            s = S[:, None, :] if v.ndim == 3 else S
            out = np.empty(v.shape, dtype=np.int64)
            out[..., 0::2] = requant(ve * c - vo * s, 15)
            out[..., 1::2] = requant(ve * s + vo * c, 15)
            env[op["output"]] = out
        elif kind == "reshape":
            x = get(op["input"])
            shape = list(op["shape"])
            if -1 in shape:
                known = 1
                for s in shape:
                    if s != -1:
                        known *= s
                shape[shape.index(-1)] = x.size // known
            env[op["output"]] = x.reshape(shape)
        elif kind == "scores":
            q = get(op["q"])  # [L, H, d]
            k = get(op["k"])
            dot = np.einsum("lhd,thd->lht", q.astype(np.float64), k.astype(np.float64))
            shift = int(op.get("shift", RECIP_Q))
            inv_q = round_half_away_scalar((1 << shift) / op["scale"])
            prod = np.round(dot).astype(np.int64) * inv_q
            note(role, prod)
            m = float(np.abs(dot).max()) if dot.size else 1.0
            safe = math.floor(math.log2((1 << 29) * op["scale"] / max(m, 1.0)))
            safe = max(1, min(RECIP_Q, safe))
            stats[role] = min(stats.get(role, RECIP_Q), safe)
            env[op["output"]] = rq(prod, shift)
        elif kind == "causal_softmax":
            s = get(op["input"]).astype(np.int64)  # [L, H, L]
            L = s.shape[0]
            runmax: np.ndarray = np.maximum.accumulate(s, axis=-1)  # running max over t
            m = runmax[np.arange(L), :, np.arange(L)]  # [L, H]: max over t <= li
            diff = np.clip(s - m[..., None], -(1 << 15), 0)
            exps = exp_q15(diff)  # [L, H, L]
            mask = np.tril(np.ones((L, L), dtype=bool))
            exps = np.where(mask[:, None, :], exps, 0)
            total = exps.sum(axis=-1)  # [L, H]
            inv_q = round_half_away(
                (1 << (RECIP_Q + 15)) / np.where(total != 0, total, 1).astype(np.float64)
            )
            inv_q = np.where(total != 0, inv_q, 0)
            out = requant(exps * inv_q[..., None], RECIP_Q)
            out = np.where(total[..., None] != 0, out, 0)
            env[op["output"]] = out.astype(np.int64)
        elif kind == "weighted_v":
            w = get(op["weights"]).astype(np.int64)  # [L, H, L]
            v = get(op["v"]).astype(np.int64)  # [L, H, d]
            # out[l,h,j] = sum_t w[l,h,t] v[t,h,j]
            acc = np.einsum("lht,thd->lhd", w, v)
            note(role, acc)
            env[op["output"]] = requant(acc, 15)
        elif kind == "silu":
            env[op["output"]] = silu(get(op["input"]))
        elif kind == "mul":
            prod = (get(op["a"]) * get(op["b"])).astype(np.int64)
            assert "shift" in op, "calibrate the spec first: `mul` needs a shift"
            note(role, prod)
            env[op["output"]] = rq(prod, int(op["shift"] or 0))
        elif kind == "add":
            tot = get(op["inputs"][0]).copy()
            for rest in op["inputs"][1:]:
                tot = tot + get(rest)
            note(role, tot)
            env[op["output"]] = rq(tot, int(op.get("shift") or 0))
        elif kind == "top_k":
            s = get(op["input"])  # [..., E]
            E = s.shape[-1]
            k = min(op["k"], E)
            top = np.argsort(-s, axis=-1, kind="stable")[..., :k]  # largest k, least idx on tie
            choice = np.zeros_like(s)
            np.put_along_axis(choice, top, 1, axis=-1)
            env[op["output"]] = s
            env[op["choice"]] = choice
        elif kind == "gather":
            # The rows of `input` where this expert is selected, in increasing order.
            ch = get(op["choice"])  # [L, E]
            e = int(op["expert"])
            rows = np.flatnonzero(ch[:, e]).astype(np.int64)
            src = get(op["input"])  # [L, hidden]
            # The arithmetization needs a power-of-two height; pad with zeros. Every op
            # downstream maps a zero row to a zero row, and the combine reads only the
            # real rows, so the padding is inert.
            pad = max(1, int(2 ** np.ceil(np.log2(max(len(rows), 1)))))
            out = np.zeros((pad, src.shape[1]), dtype=np.int64)
            if len(rows):
                out[: len(rows)] = src[rows]
            env[op["output"]] = out
            selections[op["output"]] = [int(r) for r in rows]
        elif kind == "moe_combine":
            # out[p] = sum over the experts whose gathered row p came from position p.
            ch = get(op["choice"])
            terms = [get(t) for t in op["terms"]]
            acc = np.zeros((ch.shape[0], terms[0].shape[1]), dtype=np.int64)
            for e, term in enumerate(terms):
                rows = np.flatnonzero(ch[:, e]).astype(np.int64)
                if rows.size:
                    # The gathered cell is padded to a power of two; only the real rows
                    # belong to positions.
                    acc[rows] += term[: rows.size]
            env[op["output"]] = acc
        elif kind == "weighted_sum":
            ch = get(op["choice"])  # [L, E]
            terms = [get(t) for t in op["terms"]]  # each [L, hidden]
            acc = np.zeros_like(terms[0])
            for e in range(ch.shape[-1]):
                acc = acc + terms[e] * ch[:, e][:, None]
            env[op["output"]] = acc
        else:
            raise ValueError(f"unknown op {kind}")

    cells = []
    for name in spec["cells"]:
        t = env[name]
        if t.ndim == 1:
            rows, cols = 1, t.shape[0]
        elif t.ndim == 2:
            rows, cols = t.shape
        else:
            rows, cols = t.shape[0], t.size // t.shape[0]
        # Keep the array: `tolist()` would materialise ~28 bytes of Python object per
        # value, which for the full model is gigabytes of pure waste.
        cells.append({"name": name, "rows": int(rows), "cols": int(cols),
                      "values": np.ascontiguousarray(t.flatten())})

    if output_pos is None:
        output_pos = len(window) - 1
    logits = env["logits"]  # [L, vocab]
    head_token = int(np.argmax(logits[output_pos]))
    return {
        "segments": [{"head_token": head_token, "cells": cells}],
        # Per gathered cell: the source rows it holds, in order. Public data the prover
        # hands to the verifier, which checks it against the committed choice.
        "selections": selections,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", default="data/olmoe.json")
    p.add_argument("--weights", default="data/weights.json")
    p.add_argument("--prompt", default="1 2 3 4")
    p.add_argument("--out", default="data/trace.bin")
    args = p.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)
    with open(args.weights) as f:
        weights = {w["name"]: np.array(w["values"], dtype=np.int64).reshape(w["rows"], w["cols"])
                   for w in json.load(f)}

    prompt = [int(t) for t in args.prompt.split()]
    trace = run(spec, weights, prompt)
    write_binary_trace(trace, args.out)
    print(f"greedy token = {trace['segments'][0]['head_token']}, "
          f"{len(trace['segments'][0]['cells'])} cells -> {args.out}")


if __name__ == "__main__":
    main()
