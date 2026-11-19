#!/usr/bin/env python3
"""Generate a tiny end-to-end model (spec + weights.bin + trace.json + prompt/output/fill)
for exercising the `gkr_prove` binary."""
import json
import struct
import numpy as np
from run_integer import run

VOCAB = 10  # intentionally non-power-of-two to exercise padding
HIDDEN = 4
rng = np.random.default_rng(12345)

spec = {
    "dims": {"hidden": HIDDEN, "vocab": VOCAB, "layers": 1, "heads": 1,
             "head_dim": HIDDEN, "experts": 1, "active": 1, "intermediate": HIDDEN},
    "weights": [
        {"name": "embed_tokens", "rows": VOCAB, "cols": HIDDEN},
        {"name": "lm_head", "rows": VOCAB, "cols": HIDDEN},
    ],
    "ops": [
        {"op": "embed", "weight": "embed_tokens", "output": "embedding"},
        {"op": "matmul", "input": "embedding", "weight": "lm_head", "output": "logits", "shift": 4},
    ],
    "cells": ["embedding", "logits"],
}

weights = []
for w in spec["weights"]:
    vals = (rng.integers(-8, 8, size=(w["rows"], w["cols"]))).astype(np.int8)
    weights.append({"name": w["name"], "rows": w["rows"], "cols": w["cols"],
                    "values": vals.ravel().tolist()})

prompt = [1, 3, 7]
window = prompt + [0] * (len(prompt) - 1)  # L = 3 (non-power-of-two too)
L = len(window)

wmap = {w["name"]: np.array(w["values"], dtype=np.int8).reshape(w["rows"], w["cols"]) for w in weights}
trace = run(spec, wmap, window)

json.dump(spec, open("data/tiny_spec.json", "w"))
json.dump(prompt, open("data/prompt.json", "w"))
json.dump([], open("data/output.json", "w"))
json.dump([], open("data/fill.json", "w"))
json.dump(trace, open("data/trace.json", "w"))

# weights.bin: u32 count, then per tensor u16 name_len, name, u32 rows, u32 cols, int8 vals
buf = bytearray()
buf += struct.pack("<I", len(weights))
for w in weights:
    name = w["name"].encode()
    buf += struct.pack("<H", len(name))
    buf += name
    buf += struct.pack("<II", w["rows"], w["cols"])
    buf += bytes(np.array(w["values"], dtype=np.int8).tobytes())
open("data/weights.bin", "wb").write(buf)

print("wrote tiny model: vocab=%d hidden=%d L=%d" % (VOCAB, HIDDEN, L))
print("head_token =", trace["segments"][0]["head_token"])
