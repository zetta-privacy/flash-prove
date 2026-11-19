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


def load_safetensors(path):
    """Read a safetensors file without torch, returning {name: float32 ndarray}.
    Handles F32/F16/BF16 (BF16 is reinterpreted exactly as the top 16 bits)."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        out = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype = info["dtype"]
            shape = info["shape"]
            start, end = info["data_offsets"]
            f.seek(8 + header_len + start)
            raw = f.read(end - start)
            if dtype == "F32":
                arr = np.frombuffer(raw, dtype=np.float32)
            elif dtype == "F16":
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
            elif dtype == "BF16":
                u16 = np.frombuffer(raw, dtype=np.uint16)
                arr = (u16.astype(np.uint32) << 16).view(np.float32)
            else:
                raise ValueError(f"unsupported dtype {dtype} for {name}")
            out[name] = arr.reshape(shape)
        return out


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
    if args.spec:
        with open(args.spec) as f:
            spec_names = {w["name"] for w in json.load(f)["weights"]}

    index_path = os.path.join(args.dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    shards = sorted(set(index["weight_map"].values()))

    tensors = []
    for shard in shards:
        state = load_safetensors(os.path.join(args.dir, shard))
        for key, tensor in state.items():
            name = rename(key)
            if spec_names is not None and name not in spec_names:
                continue
            tensors.append((name, quantize(tensor)))
        del state

    with open(args.out, "wb") as f:
        f.write(struct.pack("<I", len(tensors)))
        for name, vals in tensors:
            if vals.ndim == 1:
                rows, cols = 1, vals.shape[0]
            else:
                rows, cols = vals.shape
            nb = name.encode("utf-8")
            f.write(struct.pack("<H", len(nb)))
            f.write(nb)
            f.write(struct.pack("<II", rows, cols))
            f.write(vals.astype(np.int8).tobytes())

    print(f"wrote {len(tensors)} tensors -> {args.out}")


if __name__ == "__main__":
    main()
