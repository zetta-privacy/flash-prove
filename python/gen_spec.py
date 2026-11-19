#!/usr/bin/env python3
"""Generate `olmoe.json` (the data-driven ModelSpec) from config.json.

Emits the full weight layout, the op graph, and the cell list for OLMoE-1B-7B.
The attention *scores → causal softmax → V* part needs matrix (2-D) ops, which
the generic interpreter does not model yet; it is marked as deferred below and
omitted from the ops, so this spec currently covers the MoE / FFN / embedding /
output path end-to-end.
"""

import argparse
import json


def gen_spec(cfg: dict) -> dict:
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
                        "weight": f"{p}.{proj}_proj", "output": f"{p}.{proj}", "shift": 4})
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
                    "output": f"{p}.attn_out", "shift": 4})
        cells += [f"{p}.scores", f"{p}.attn_weights", f"{p}.attn_heads", f"{p}.attn", f"{p}.attn_out"]
        ops.append({"op": "add", "inputs": [prev, f"{p}.attn_out"], "output": f"{p}.residual_1"})
        cells.append(f"{p}.residual_1")

        # MoE.
        ops.append({"op": "norm", "input": f"{p}.residual_1",
                    "weight": f"{p}.post_attention_layernorm", "output": f"{p}.norm_moe", "eps": 1e-6})
        cells.append(f"{p}.norm_moe")
        ops.append({"op": "matmul", "input": f"{p}.norm_moe", "weight": f"{p}.router",
                    "output": f"{p}.router_logits", "shift": 4})
        ops.append({"op": "top_k", "input": f"{p}.router_logits", "output": f"{p}.router_logits",
                    "choice": f"{p}.expert_choice", "k": active})
        cells += [f"{p}.router_logits", f"{p}.expert_choice"]

        terms = []
        for e in range(experts):
            e_ = f"{p}.expert.{e}"
            ops.append({"op": "matmul", "input": f"{p}.norm_moe",
                        "weight": f"{e_}.gate_proj", "output": f"{e_}.gate", "shift": 4})
            ops.append({"op": "matmul", "input": f"{p}.norm_moe",
                        "weight": f"{e_}.up_proj", "output": f"{e_}.up", "shift": 4})
            ops.append({"op": "silu", "input": f"{e_}.gate", "output": f"{e_}.act"})
            ops.append({"op": "mul", "a": f"{e_}.act", "b": f"{e_}.up", "output": f"{e_}.h"})
            ops.append({"op": "matmul", "input": f"{e_}.h", "weight": f"{e_}.down_proj",
                        "output": f"{e_}.down", "shift": 4})
            terms.append(f"{e_}.down")
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
                "output": "logits", "shift": 4})
    cells.append("logits")

    return {"dims": dims, "weights": weights, "ops": ops, "cells": cells}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="olmoe/config.json")
    parser.add_argument("--out", default="data/olmoe.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    spec = gen_spec(cfg)
    with open(args.out, "w") as f:
        json.dump(spec, f, indent=1)
    print(f"wrote {args.out}: {len(spec['weights'])} weights, "
          f"{len(spec['ops'])} ops, {len(spec['cells'])} cells")


if __name__ == "__main__":
    main()
