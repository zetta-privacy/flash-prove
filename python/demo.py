#!/usr/bin/env python3
"""Generate synthetic weights + trace + request files for the demo smoke test,
then the Rust `cp_commit`/`cp_prove`/`cp_verify` round-trip it.

The demo uses `specs/demo.json` (a tiny FFN) with a fixed 1024-token window, so
the trace and the Rust verifier agree on the window length.
"""

import argparse
import json
import struct

import numpy as np

from run_integer import run

WINDOW_LEN = 1024


def gen_weights(spec, seed=12345):
    rng = np.random.default_rng(seed)
    weights = []
    for w in spec["weights"]:
        n = w["rows"] * w["cols"]
        vals = rng.integers(-4, 5, size=n).astype(np.int64)
        weights.append({
            "name": w["name"], "rows": w["rows"], "cols": w["cols"],
            "values": vals.tolist(),
        })
    return weights


def write_bin(weights, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(weights)))
        for w in weights:
            nb = w["name"].encode("utf-8")
            f.write(struct.pack("<H", len(nb)))
            f.write(nb)
            f.write(struct.pack("<II", w["rows"], w["cols"]))
            f.write(np.asarray(w["values"], dtype=np.int8).tobytes())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", default="specs/demo.json")
    args = p.parse_args()
    with open(args.spec) as f:
        spec = json.load(f)

    weights = gen_weights(spec)
    write_bin(weights, "data/weights.bin")

    wdict = {
        w["name"]: np.array(w["values"], dtype=np.int64).reshape(w["rows"], w["cols"])
        for w in weights
    }

    x = [1, 2, 3]
    output_pos = len(x) - 1  # position whose logits predict y[0]

    # First pass: prompt + padding, to read the greedy token.
    w1 = x + [0] * (WINDOW_LEN - len(x))
    greedy = run(spec, wdict, w1, output_pos)["segments"][0]["head_token"]

    y = [greedy]
    fill = []
    window = (x + y + fill) + [0] * (WINDOW_LEN - len(x) - len(y) - len(fill))
    trace = run(spec, wdict, window, output_pos)
    with open("data/trace.json", "w") as f:
        json.dump(trace, f)

    for name, val in [("data/prompt.json", x), ("data/output.json", y), ("data/fill.json", fill)]:
        with open(name, "w") as f:
            json.dump(val, f)

    print(f"weights={len(weights)}, head_token={trace['segments'][0]['head_token']}")


if __name__ == "__main__":
    main()
