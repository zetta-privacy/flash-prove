#!/usr/bin/env python3
"""Generate the data-driven ModelSpec from config.json.

Emits the weight layout, the op graph, and the cell list. Every op that produces a
cell carries a requant shift taken from a *per-role* table shared by the prover,
the verifier and the reference engine (`python/calibrate.py` produces it).
"""

import argparse
import json
import os
import sys

from run_integer import RECIP_Q, op_role

# Starting shift for an uncalibrated matmul role (the old global constant).
SEED_MM = 4


def gen_spec(cfg: dict, shifts: dict) -> dict:
    hidden = cfg["hidden_size"]
    layers = cfg["num_hidden_layers"]
    heads = cfg["num_attention_heads"]
    head_dim = hidden // heads
    experts = cfg["num_experts"]
    active = cfg["num_experts_per_tok"]
    intermediate = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]
    theta = cfg["rope_theta"]

    dims = {
        "hidden": hidden,
        "vocab": vocab,
        "layers": layers,
        "heads": heads,
        "head_dim": head_dim,
        "experts": experts,
        "active": active,
        "intermediate": intermediate,
    }

    weights = []
    weights.append({"name": "embed_tokens", "rows": vocab, "cols": hidden})
    weights.append({"name": "lm_head", "rows": vocab, "cols": hidden})
    weights.append({"name": "norm", "rows": 1, "cols": hidden})

    for l in range(layers):
        p = f"layer.{l}"
        weights.append({"name": f"{p}.q_proj", "rows": hidden, "cols": hidden})
        weights.append({"name": f"{p}.k_proj", "rows": hidden, "cols": hidden})
        weights.append({"name": f"{p}.v_proj", "rows": hidden, "cols": hidden})
        weights.append({"name": f"{p}.o_proj", "rows": hidden, "cols": hidden})
        weights.append({"name": f"{p}.q_norm", "rows": 1, "cols": hidden})
        weights.append({"name": f"{p}.k_norm", "rows": 1, "cols": hidden})
        weights.append({"name": f"{p}.input_layernorm", "rows": 1, "cols": hidden})
        weights.append({"name": f"{p}.post_attention_layernorm", "rows": 1, "cols": hidden})
        weights.append({"name": f"{p}.router", "rows": experts, "cols": hidden})
        for e in range(experts):
            weights.append({"name": f"{p}.expert.{e}.gate_proj", "rows": intermediate, "cols": hidden})
            weights.append({"name": f"{p}.expert.{e}.up_proj", "rows": intermediate, "cols": hidden})
            weights.append({"name": f"{p}.expert.{e}.down_proj", "rows": hidden, "cols": intermediate})

    ops = []
    cells = []

    ops.append({"op": "embed", "weight": "embed_tokens", "output": "embedding"})
    cells.append("embedding")

    prev = "embedding"
    for l in range(layers):
        p = f"layer.{l}"
        ops.append({"op": "norm", "input": prev, "weight": f"{p}.input_layernorm",
                    "output": f"{p}.norm_in", "eps": 1e-6})
        cells.append(f"{p}.norm_in")

        # Attention (multi-head).
        for proj in ("q", "k", "v"):
            ops.append({"op": "matmul", "input": f"{p}.norm_in",
                        "weight": f"{p}.{proj}_proj", "output": f"{p}.{proj}"})
            cells.append(f"{p}.{proj}")
        ops.append({"op": "norm", "input": f"{p}.q", "weight": f"{p}.q_norm",
                    "output": f"{p}.q_normed", "eps": 1e-6})
        ops.append({"op": "norm", "input": f"{p}.k", "weight": f"{p}.k_norm",
                    "output": f"{p}.k_normed", "eps": 1e-6})
        cells += [f"{p}.q_normed", f"{p}.k_normed"]
        for proj in ("q_normed", "k_normed", "v"):
            ops.append({"op": "reshape", "input": f"{p}.{proj}",
                        "output": f"{p}.{proj}_heads", "shape": [-1, heads, head_dim]})
            cells.append(f"{p}.{proj}_heads")
        ops.append({"op": "rope", "input": f"{p}.q_normed_heads", "output": f"{p}.q_rot", "theta": theta})
        ops.append({"op": "rope", "input": f"{p}.k_normed_heads", "output": f"{p}.k_rot", "theta": theta})
        cells += [f"{p}.q_rot", f"{p}.k_rot"]
        ops.append({"op": "scores", "q": f"{p}.q_rot", "k": f"{p}.k_rot",
                    "output": f"{p}.scores", "scale": head_dim ** 0.5})
        ops.append({"op": "causal_softmax", "input": f"{p}.scores", "output": f"{p}.attn_weights"})
        ops.append({"op": "weighted_v", "weights": f"{p}.attn_weights", "v": f"{p}.v_heads",
                    "output": f"{p}.attn_heads"})
        ops.append({"op": "reshape", "input": f"{p}.attn_heads",
                    "output": f"{p}.attn", "shape": [-1, hidden]})
        ops.append({"op": "matmul", "input": f"{p}.attn", "weight": f"{p}.o_proj",
                    "output": f"{p}.attn_out"})
        cells += [f"{p}.scores", f"{p}.attn_weights", f"{p}.attn_heads", f"{p}.attn", f"{p}.attn_out"]
        ops.append({"op": "add", "inputs": [prev, f"{p}.attn_out"], "output": f"{p}.residual_1"})
        cells.append(f"{p}.residual_1")

        # MoE.
        ops.append({"op": "norm", "input": f"{p}.residual_1",
                    "weight": f"{p}.post_attention_layernorm", "output": f"{p}.norm_moe", "eps": 1e-6})
        cells.append(f"{p}.norm_moe")
        ops.append({"op": "matmul", "input": f"{p}.norm_moe", "weight": f"{p}.router",
                    "output": f"{p}.router_logits"})
        ops.append({"op": "top_k", "input": f"{p}.router_logits", "output": f"{p}.router_logits",
                    "choice": f"{p}.expert_choice", "k": active})
        cells += [f"{p}.router_logits", f"{p}.expert_choice"]

        terms = []
        for e in range(experts):
            e_ = f"{p}.expert.{e}"
            ops.append({"op": "matmul", "input": f"{p}.norm_moe",
                        "weight": f"{e_}.gate_proj", "output": f"{e_}.gate"})
            ops.append({"op": "matmul", "input": f"{p}.norm_moe",
                        "weight": f"{e_}.up_proj", "output": f"{e_}.up"})
            ops.append({"op": "silu", "input": f"{e_}.gate", "output": f"{e_}.act"})
            ops.append({"op": "mul", "a": f"{e_}.act", "b": f"{e_}.up", "output": f"{e_}.h"})
            ops.append({"op": "matmul", "input": f"{e_}.h", "weight": f"{e_}.down_proj",
                        "output": f"{e_}.down"})
            terms.append(f"{e_}.down")
            cells += [f"{e_}.gate", f"{e_}.up", f"{e_}.act", f"{e_}.h", f"{e_}.down"]
        ops.append({"op": "weighted_sum", "choice": f"{p}.expert_choice",
                    "terms": terms, "output": f"{p}.moe"})
        cells.append(f"{p}.moe")

        ops.append({"op": "add", "inputs": [f"{p}.residual_1", f"{p}.moe"],
                    "output": f"{p}.residual_2"})
        cells.append(f"{p}.residual_2")
        prev = f"{p}.residual_2"

    ops.append({"op": "norm", "input": prev, "weight": "norm", "output": "final_norm", "eps": 1e-6})
    cells.append("final_norm")
    ops.append({"op": "matmul", "input": "final_norm", "weight": "lm_head",
                "output": "logits"})
    cells.append("logits")

    # Bind every op to its role's shift. An empty table is the calibration seed
    # (every matmul at `SEED_MM`); a *non-empty* table that misses a role is a typo.
    missing: list[str] = []
    for op in ops:
        role = op_role(op)
        kind = op["op"]
        if kind == "matmul":
            if role in shifts:
                op["shift"] = int(shifts[role])
            else:
                missing.append(role)
                op["shift"] = SEED_MM
        elif kind in ("mul", "add"):
            k = int(shifts.get(role, 0))
            op["shift"] = k if k > 0 else None
        elif kind == "scores":
            op["shift"] = int(shifts.get(role, RECIP_Q))

    if missing and shifts:
        sys.exit(f"no calibrated shift for {sorted(set(missing))}; re-run python/calibrate.py")

    return {"dims": dims, "weights": weights, "ops": ops, "cells": cells}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="olmoe/config.json")
    parser.add_argument("--shifts", default="data/shifts.json")
    parser.add_argument("--out", default="data/olmoe.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    shifts = {}
    if os.path.exists(args.shifts):
        with open(args.shifts) as f:
            shifts = json.load(f)
    spec = gen_spec(cfg, shifts)
    with open(args.out, "w") as f:
        json.dump(spec, f, indent=1)
    print(f"wrote {args.out}: {len(spec['weights'])} weights, "
          f"{len(spec['ops'])} ops, {len(spec['cells'])} cells")
