#!/usr/bin/env python3
"""Load OLMoE weights from a sharded safetensors checkpoint, rename to the spec's
tensor names, quantize to int8, and write a compact binary file.

Format (all little-endian):
  u32 num_tensors
  per tensor: u16 name_len, name (utf-8), u32 rows, u32 cols, rows*cols int8
"""

import argparse
import json
import os
import struct

import numpy as np


def read_shard(path):
    """The shard's tensor directory: `(header, data_start)`.

    Nothing is read eagerly: each tensor's bytes are fetched when it is quantised, so a
    5 GB shard is never materialised as a whole — as `float32` it would be 10 GB.
    """
    f = open(path, "rb")
    header_len = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(header_len))
    return f, header, 8 + header_len


def read_tensor(f, data_start, name, info):
    """One tensor as `float32` (handles F32/F16/BF16 without touching the rest)."""
    dtype = info["dtype"]
    start, end = info["data_offsets"]
    f.seek(data_start + start)
    raw = f.read(end - start)
    shape = info["shape"]
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    if dtype == "BF16":
        # Exact: BF16 is the top 16 bits of an F32.
        u16 = np.frombuffer(raw, dtype=np.uint16)
        return (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)
    raise ValueError(f"unsupported dtype {dtype} for {name}")


def rename(key: str) -> str:
    """Map a safetensors key to the spec's tensor name."""
    if key == "model.embed_tokens.weight":
        return "embed_tokens"
    if key == "model.norm.weight":
        return "norm"
    if key == "lm_head.weight":
        return "lm_head"
    parts = key.split(".")
    if parts[0] == "model" and parts[1] == "layers":
        l = parts[2]
        rest = parts[3:]
        if rest[0] == "self_attn":
            return f"layer.{l}." + ".".join(rest[1:-1])
        if rest[0] in ("input_layernorm", "post_attention_layernorm"):
            return f"layer.{l}." + rest[0]
        if rest[0] == "mlp":
            m = rest[1]
            if m == "gate":
                return f"layer.{l}.router"
            if m == "experts":
                return f"layer.{l}.expert.{rest[2]}.{rest[3]}"
    raise ValueError(f"unmapped key {key}")


def quantize(t, bits=7):
    w = np.asarray(t, dtype=np.float32)
    max_abs = float(np.abs(w).max())
    if max_abs == 0:
        return np.zeros(w.shape, dtype=np.int8)
    scale = max_abs / (2**bits - 1)
    return np.round(w / scale).astype(np.int8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="olmoe")
    p.add_argument("--out", default="data/weights.bin")
    p.add_argument("--spec", default=None, help="optional spec JSON to filter tensors to")
    args = p.parse_args()

    spec_names = None
    spec_shapes = {}
    if args.spec:
        with open(args.spec) as f:
            spec_weights = json.load(f)["weights"]
        spec_names = {w["name"] for w in spec_weights}
        spec_shapes = {w["name"]: (w["rows"], w["cols"]) for w in spec_weights}

    index_path = os.path.join(args.dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    shards = sorted(set(index["weight_map"].values()))

    # Stream: one tensor is read, quantised and written at a time, so peak memory is the
    # largest tensor (`lm_head`, ~0.4 GB) rather than the model. The count at the head of
    # the file is patched once the number is known.
    count = 0
    with open(args.out, "wb") as out:
        out.write(struct.pack("<I", 0))
        for shard in shards:
            f, header, data_start = read_shard(os.path.join(args.dir, shard))
            try:
                for key, info in header.items():
                    if key == "__metadata__":
                        continue
                    name = rename(key)
                    if spec_names is not None and name not in spec_names:
                        continue
                    vals = quantize(read_tensor(f, data_start, name, info))
                    if vals.ndim == 1:
                        vals = vals.reshape(1, -1)
                    # The spec fixes each tensor's shape (e.g. a reduced test vocab), so
                    # trim the checkpoint tensor to it and drop everything else.
                    if name in spec_shapes:
                        rows, cols = spec_shapes[name]
                        if vals.shape[0] < rows:
                            # The vocabulary is padded to a power of two; the extra rows
                            # are zero and never selected.
                            padded = np.zeros((rows, vals.shape[1]), dtype=vals.dtype)
                            padded[: vals.shape[0]] = vals
                            vals = padded
                        vals = vals[:rows, :cols]
                    nb = name.encode("utf-8")
                    rows, cols = vals.shape
                    out.write(struct.pack("<H", len(nb)))
                    out.write(nb)
                    out.write(struct.pack("<II", rows, cols))
                    out.write(vals.astype(np.int8).tobytes())
                    count += 1
            finally:
                f.close()
        out.seek(0)
        out.write(struct.pack("<I", count))
    print(f"wrote {count} tensors -> {args.out}")


if __name__ == "__main__":
    main()
