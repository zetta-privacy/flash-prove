#!/usr/bin/env python3
"""Run the real OLMoE integer model: read the binary weights, run the spec,
emit the trace + request files."""

import argparse
import json
import struct

import numpy as np

from run_integer import run



def read_bin(path):
    data = open(path, "rb").read()
    r = 0
    n = struct.unpack_from("<I", data, r)[0]
    r += 4
    weights = {}
    for _ in range(n):
        name_len = struct.unpack_from("<H", data, r)[0]
        r += 2
        name = data[r:r + name_len].decode("utf-8")
        r += name_len
        rows, cols = struct.unpack_from("<II", data, r)
        r += 8
        vals = np.frombuffer(data, dtype=np.int8, count=rows * cols, offset=r)
        r += rows * cols
        weights[name] = vals.reshape(rows, cols)
    return weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", default="data/olmoe.json")
    p.add_argument("--weights", default="data/weights.bin")
    p.add_argument("--prompt", default="1 2 3")
    p.add_argument("--window", type=int, default=1024, help="window length L")
    p.add_argument("--out", default="data/trace.json")
    args = p.parse_args()

    spec = json.load(open(args.spec))
    weights = read_bin(args.weights)

    x = [int(t) for t in args.prompt.split()]
    output_pos = len(x) - 1
    L = args.window

    w1 = x + [0] * (L - len(x))
    greedy = run(spec, weights, w1, output_pos)["segments"][0]["head_token"]

    y = [greedy]
    fill = []
    window = (x + y + fill) + [0] * (L - len(x) - len(y) - len(fill))
    trace = run(spec, weights, window, output_pos)

    json.dump(trace, open(args.out, "w"))
    json.dump(x, open("data/prompt.json", "w"))
    json.dump(y, open("data/output.json", "w"))
    json.dump(fill, open("data/fill.json", "w"))
    print(f"greedy={greedy}, {len(trace['segments'][0]['cells'])} cells -> {args.out}")


if __name__ == "__main__":
    main()
