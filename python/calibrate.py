#!/usr/bin/env python3
"""Calibrate the per-role requant shift table.

Loops `gen_spec` -> integer forward pass until the shift table stops moving, then
writes the table and the spec.

Each role's shift is left alone while its post-shift magnitude sits inside a **band**
(`BAND_LO..BAND_HI` bits) and only re-targeted when it leaves it. Without the band the
loop is a hair-trigger: aiming every role's maximum at exactly one bit count means a
one-bit change cascades through the model and the table two-cycles instead of settling —
which is exactly what a sixteen-layer residual stream does. The score shift is monotone
(a bound, and it cannot influence the dot products it is derived from).

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
from weights import WeightStore


# Post-shift magnitudes, in bits, that a role is happy with. Inside the band the
# shift is not touched; outside it is re-aimed at `SHIFT_TARGET`.
BAND_LO = 8
BAND_HI = 12


def want(raw: float, current) -> int:
    """The shift to use for `raw`, keeping `current` while the value stays in band."""
    # A value of zero carries no information: the shift may itself be what zeroed it
    # (a first, uncalibrated round can hand a residual a shift of 20+ bits and collapse
    # the rest of the network), so reset to the smallest rather than freeze.
    if raw <= 1.0:
        return 1
    bits = math.ceil(math.log2(raw))
    if current is None:
        return max(1, bits - SHIFT_TARGET)
    if BAND_LO <= bits - int(current) <= BAND_HI:
        return int(current)
    return max(1, bits - SHIFT_TARGET)


def seed_table(seed: dict, layers: int) -> dict:
    """Expand a table's `layer.*` roles to one entry per layer."""
    out = {}
    for role, k in seed.items():
        # Only the residual additions are per-layer; every other role is shared by all
        # layers (a normalisation is scale-invariant, a matmul's shift follows it).
        if role.startswith("add:") and "layer.*" in role:
            for l in range(layers):
                out[role.replace("layer.*", f"layer.{l}")] = k
        else:
            out[role] = k
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="olmoe/config.json")
    p.add_argument("--weights", default="data/weights.bin")
    p.add_argument("--prompt", default="1 2 3")
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--out-spec", default="data/olmoe.json")
    p.add_argument("--out-shifts", default="data/shifts.json")
    p.add_argument("--seed", default=None,
                   help="an existing shift table to start from (layer.* roles expand "
                        "to every layer)")
    args = p.parse_args()

    cfg = json.load(open(args.config))
    weights = WeightStore(args.weights)
    x = [int(t) for t in args.prompt.split()]
    window = (x + [0] * (args.window - len(x)))[: args.window]
    pos = min(len(x), len(window)) - 1

    shifts: dict = seed_table(json.load(open(args.seed)), cfg["num_hidden_layers"]) \
        if args.seed else {}
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
                new[role] = want(raw, shifts.get(role))
            elif role == "scores":
                # A bound: `raw` is the largest safe shift. Never raise it, so the loop
                # cannot oscillate; start at the most precise one.
                current = shifts.get(role)
                new[role] = int(min(RECIP_Q, raw)) if current is None else int(
                    min(int(current), raw)
                )

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
    print("role shifts and magnitudes (log2), pre-shift -> post-shift:")
    for role in sorted(stats):
        k = shifts.get(role, 0)
        raw = max(stats[role], 1.0)
        print(f"  {role:<44} shift={k:>2}  2^{math.log2(raw):5.1f} -> 2^{math.log2(raw) - k:5.1f}")
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
