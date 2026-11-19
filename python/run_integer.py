#!/usr/bin/env python3
"""Run the integer model described by a ModelSpec, emitting the trace.

This mirrors `src/spec.rs` exactly: round-half-away-from-zero, q15 fixed-point,
and the same silu/softmax/rope/rmsnorm lookups, so the emitted trace is accepted
by the Rust verifier. The op graph is read from a spec (data), so this is
model-agnostic.

The transcendental functions are precomputed into lookup tables over the (bounded,
clamped) integer domains. Every table entry is computed with `math.*` (C libm),
which is bit-identical to Rust's `f64::{exp,cos,sin,powf,sqrt}` on the same
platform, so this stays bit-exact while avoiding the per-element Python loop.
"""

import argparse
import json
import math

import numpy as np


def round_half_away_scalar(x):
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def round_half_away(x):
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5)).astype(np.int64)


def exp_f64(x):
    # Rust f64::exp returns +inf on overflow (IEEE 754); math.exp raises instead.
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


def requant(x, k):
    half = 1 << (k - 1)
    return np.where(x >= 0, (x + half) >> k, -(((-x) + half) >> k)).astype(np.int64)


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
    gamma = np.asarray(gamma, dtype=np.float64).reshape(-1)
    d = x.shape[-1]
    flat = x.reshape(-1, d)
    out = np.empty(flat.shape, dtype=np.int64)
    for b in range(flat.shape[0]):
        row = flat[b]
        # sequential f64 sum, matching Rust's `Iterator::sum::<f64>()`.
        mean_sq = 0.0
        for v in row:
            mean_sq += float(v) * float(v)
        mean_sq /= d
        inv = 1.0 / math.sqrt(mean_sq + eps)
        out[b] = round_half_away(row.astype(np.float64) * inv * gamma)
    return out.reshape(x.shape)


def run(spec, weights, window, output_pos=None) -> dict:
    env = {}

    def get(name):
        return env[name]

    for op in spec["ops"]:
        kind = op["op"]
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
            env[op["output"]] = round_half_away(acc / (1 << op["shift"])).astype(np.int64)
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
            env[op["output"]] = round_half_away(
                np.einsum("lhd,thd->lht", q.astype(np.float64), k.astype(np.float64)) / op["scale"]
            )
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
            safe = np.where(total[..., None] != 0, total[..., None], 1)
            out = exps * (1 << 15) // safe
            out = np.where(total[..., None] != 0, out, 0)
            env[op["output"]] = out.astype(np.int64)
        elif kind == "weighted_v":
            w = get(op["weights"]).astype(np.int64)  # [L, H, L]
            v = get(op["v"]).astype(np.int64)  # [L, H, d]
            # out[l,h,j] = sum_t w[l,h,t] v[t,h,j]
            env[op["output"]] = requant(np.einsum("lht,thd->lhd", w, v), 15)
        elif kind == "silu":
            env[op["output"]] = silu(get(op["input"]))
        elif kind == "mul":
            env[op["output"]] = (get(op["a"]) * get(op["b"])).astype(np.int64)
        elif kind == "add":
            env[op["output"]] = get(op["inputs"][0]).copy()
            for rest in op["inputs"][1:]:
                env[op["output"]] = env[op["output"]] + get(rest)
        elif kind == "top_k":
            s = get(op["input"])  # [..., E]
            E = s.shape[-1]
            k = min(op["k"], E)
            top = np.argsort(-s, axis=-1, kind="stable")[..., :k]  # largest k, least idx on tie
            choice = np.zeros_like(s)
            np.put_along_axis(choice, top, 1, axis=-1)
            env[op["output"]] = s
            env[op["choice"]] = choice
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
        cells.append({"name": name, "rows": int(rows), "cols": int(cols),
                      "values": t.flatten().tolist()})

    if output_pos is None:
        output_pos = len(window) - 1
    logits = env["logits"]  # [L, vocab]
    head_token = int(np.argmax(logits[output_pos]))
    return {"segments": [{"head_token": head_token, "cells": cells}]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", default="data/olmoe.json")
    p.add_argument("--weights", default="data/weights.json")
    p.add_argument("--prompt", default="1 2 3 4")
    p.add_argument("--out", default="data/trace.json")
    args = p.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)
    with open(args.weights) as f:
        weights = {w["name"]: np.array(w["values"], dtype=np.int64).reshape(w["rows"], w["cols"])
                   for w in json.load(f)}

    prompt = [int(t) for t in args.prompt.split()]
    trace = run(spec, weights, prompt)
    with open(args.out, "w") as f:
        json.dump(trace, f)
    print(f"greedy token = {trace['segments'][0]['head_token']}, "
          f"{len(trace['segments'][0]['cells'])} cells -> {args.out}")


if __name__ == "__main__":
    main()
