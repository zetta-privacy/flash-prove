#!/usr/bin/env python3
"""Read the integer weights without holding them all.

`python/load_weights.py` writes a flat file: `u32` count, then per tensor a `u16` name
length, the name, `u32` rows, `u32` cols, and `rows*cols` `int8` values.

A whole-file `read()` would pin the model in memory (6.9 GB for OLMoE-1B-7B). Instead
the file is mapped and each tensor is a zero-copy view of the mapping, so pages are
loaded when an op touches them and the kernel can evict them afterwards. Reads are
read-only; nothing writes to the mapping.
"""

import mmap
import struct

import numpy as np


class WeightStore:
    """Name → `int8` matrix, backed by the weights file itself."""

    def __init__(self, path):
        self._file = open(path, "rb")
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        r = 0
        count = struct.unpack_from("<I", self._map, r)[0]
        r += 4
        self.index = {}
        for _ in range(count):
            name_len = struct.unpack_from("<H", self._map, r)[0]
            r += 2
            name = self._map[r : r + name_len].decode("utf-8")
            r += name_len
            rows, cols = struct.unpack_from("<II", self._map, r)
            r += 8
            self.index[name] = (r, rows, cols)
            r += rows * cols
        self.size = r

    def __contains__(self, name):
        return name in self.index

    def __getitem__(self, name):
        offset, rows, cols = self.index[name]
        return np.frombuffer(
            self._map, dtype=np.int8, count=rows * cols, offset=offset
        ).reshape(rows, cols)

    def get(self, name, default=None):
        return self[name] if name in self.index else default

    def names(self):
        return self.index.keys()

    def close(self):
        self._map.close()
        self._file.close()
