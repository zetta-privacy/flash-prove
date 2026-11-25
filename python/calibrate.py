#!/usr/bin/env python3
"""Calibrate the per-role requant shift table.

Loops `gen_spec` -> integer forward pass until the shift table stops moving, then
writes the table and the spec. Matmul/mul/add shifts are monotone (they only grow,
so the loop terminates); the score shift is a one-shot bound (it cannot influence
the dot products it depends on).

    python/calibrate.py --config olmoe/config_small.json \
        --weights data/weights.bin --window 64 \
        --out-spec data/olmoe_small.json --out-shifts data/shifts.json
"""

import argparse
import json
import math

import numpy as np

from gen_spec import gen_spec
from run_integer import RECIP_Q, SHIFT_TARGET, op_role, run
from run_real import read_bin


def want(raw: float) -> int:
    """The shift that brings `raw` down to `SHIFT_TARGET` bits (`requant` needs k>=1)."""
    if raw <= 1.0:
        return 1
    return max(1, int(math.ceil(math.log2(raw))) - SHIFT_TARGET)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="olmoe/config.json")
    p.add_argument("--weights", default="data/weights.bin")
    p.add_argument("--prompt", default="1 2 3")
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--out-spec", default="data/olmoe.json")
    p.add_argument("--out-shifts", default="data/shifts.json")
    args = p.parse_args()

    cfg = json.load(open(args.config))
    weights = read_bin(args.weights)
    x = [int(t) for t in args.prompt.split()]
    window = (x + [0] * (args.window - len(x)))[: args.window]
    pos = min(len(x), len(window)) - 1

    shifts: dict = {}
    token = None
    for it in range(args.rounds):
        spec = gen_spec(cfg, shifts)
        stats: dict = {}
        trace = run(spec, weights, window, pos, stats=stats)
        token = trace["segments"][0]["head_token"]

        new = dict(shifts)
        for role, raw in stats.items():
            # Only the kinds that carry a shift are calibrated; `weighted_v` is
            # reported for context and ignored by `gen_spec`.
            if role.startswith(("mm:", "mul:", "add:")):
                new[role] = want(raw)
            elif role == "scores":
                # A bound, not a fixed point: the score shift cannot influence the
                # dot products it is derived from.
                new[role] = int(min(RECIP_Q, raw))

        changed = sum(1 for r in set(new) | set(shifts) if new.get(r) != shifts.get(r))
        print(f"round {it}: token={token} roles={len(new)} changed={changed} "
              f"scores={new.get('scores')}")
        if not changed:
            break
        shifts = new
    else:
        print(f"warning: no fixed point after {args.rounds} rounds; using the last table")
        shifts = new

    # Report the headroom the calibration bought.
    print("role shifts and pre-shift magnitudes (log2):")
    for role in sorted(stats):
        k = new.get(role, 0)
        print(f"  {role:<44} shift={k:>2} raw=2^{math.log2(max(stats[role], 1)):5.1f}")
    worst = max(((float(np.abs(np.asarray(c["values"])).max()), c["name"]) for c in
                 trace["segments"][0]["cells"]), default=(0.0, "-"))
    print(f"largest committed value: {worst[0]:.0f} (2^{math.log2(max(worst[0], 1)):.1f}) "
          f"in {worst[1]}")

    json.dump(shifts, open(args.out_shifts, "w"), indent=1)
    json_to_write = gen_spec(cfg, shifts)
    json.dump(json_to_write, open(args.out_spec, "w"), indent=1)
    print(f"wrote {args.out_shifts} and {args.out_spec}; greedy={token}")


if __name__ == "__main__":
    main()
