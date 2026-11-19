#!/usr/bin/env python3
"""Run OLMoE-1B-7B and emit the trace as the named-cell layout the Rust
verifier expects (see the FlashProve cell-layout contract).

Cell order per token (one segment):
  embedding, then per layer l: layer.{l}.attn_out, layer.{l}.moe_out,
  layer.{l}.residual, layer.{l}.router_logits, layer.{l}.expert_choice,
  then logits. head_token = argmax(logits).
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--out", default="data/trace.json")
    parser.add_argument("--4bit", dest="use_4bit", action="store_true", help="load in 4-bit (needs CUDA)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_kwargs: dict = {"device_map": "auto"}
    if args.use_4bit:
        load_kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(args.repo, **load_kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.repo)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    captured: dict[str, torch.Tensor] = {}

    def hook_into(name: str):
        def hook(_module, _inp, out):
            # Some module outputs are tuples (hidden_states, ...); take the tensor.
            t = out[0] if isinstance(out, tuple) else out
            captured[name] = t.detach().cpu()
        return hook

    # Register hooks by name pattern. Verify against model.named_modules() if
    # the OLMoE attribute names differ in your transformers version.
    for name, mod in model.named_modules():
        if name.endswith("embed_tokens"):
            mod.register_forward_hook(hook_into("embedding"))
        elif ".self_attn" in name and name.count(".") == 2:
            mod.register_forward_hook(hook_into("layer." + name.split(".")[1] + ".attn_out"))
        elif ".mlp" in name and name.count(".") == 2:
            mod.register_forward_hook(hook_into("layer." + name.split(".")[1] + ".moe_out"))
        elif name.endswith("lm_head"):
            mod.register_forward_hook(hook_into("logits"))

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1].cpu()
    greedy = int(logits.argmax())

    # Emit cells in the contract order (residual/router/choice still TODO below).
    cells = []
    if "embedding" in captured:
        t = captured["embedding"][0]
        cells.append({"name": "embedding", "rows": t.shape[0], "cols": t.shape[-1], "values": t.tolist()})
    for name in sorted(k for k in captured if k.startswith("layer.")):
        t = captured[name][0]
        cells.append({"name": name, "rows": t.shape[0], "cols": t.shape[-1], "values": t.tolist()})
    if "logits" in captured:
        t = captured["logits"][0, -1]
        cells.append({"name": "logits", "rows": 1, "cols": t.shape[-1], "values": t.tolist()})

    trace = {"segments": [{"head_token": greedy, "cells": cells}]}

    with open(args.out, "w") as f:
        json.dump(trace, f)

    print(f"greedy token = {greedy}")
    print(f"emitted {len(cells)} named cells -> {args.out}")


if __name__ == "__main__":
    main()
