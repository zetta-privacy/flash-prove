#!/usr/bin/env python3
"""Run the real OLMoE integer model: read the binary weights, run the spec,
emit the trace + request files."""

import argparse
import json

from run_integer import run, write_binary_trace
from weights import WeightStore



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", default="data/olmoe.json")
    p.add_argument("--weights", default="data/weights.bin")
    p.add_argument("--prompt", default="1 2 3")
    p.add_argument("--window", type=int, default=1024, help="window length L")
    p.add_argument("--out", default="data/trace.bin")
    args = p.parse_args()

    spec = json.load(open(args.spec))
    # Mapped, not read: each tensor is a view, loaded when an op touches it.
    weights = WeightStore(args.weights)

    x = [int(t) for t in args.prompt.split()]
    output_pos = len(x) - 1
    L = args.window

    w1 = x + [0] * (L - len(x))
    greedy = run(spec, weights, w1, output_pos)["segments"][0]["head_token"]

    y = [greedy]
    fill = []
    window = (x + y + fill) + [0] * (L - len(x) - len(y) - len(fill))
    trace = run(spec, weights, window, output_pos)

    write_binary_trace(trace, args.out)
    json.dump(x, open("data/prompt.json", "w"))
    json.dump(y, open("data/output.json", "w"))
    json.dump(fill, open("data/fill.json", "w"))
    print(f"greedy={greedy}, {len(trace['segments'][0]['cells'])} cells -> {args.out}")


if __name__ == "__main__":
    main()
