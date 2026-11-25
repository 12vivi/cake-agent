"""Memory helpers of the simulated GPU: swizzled shared-memory tiles and global boxes."""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence, Tuple

import numpy as np

SWIZZLES = ("none", "32B", "64B", "128B")
SWIZZLE_SPAN = {"none": 0, "32B": 32, "64B": 64, "128B": 128}
# Base-address alignment a swizzled tile needs (the swizzle atom: 8 rows x span bytes).
SWIZZLE_ALIGN = {"none": 128, "32B": 256, "64B": 512, "128B": 1024}
# Shared-memory bandwidth penalty of each layout for MMA operand fetch / generic access.
MMA_SWIZZLE_FACTOR = {"none": 1.6, "32B": 1.25, "64B": 1.1, "128B": 1.0}
SMEM_CONFLICT_FACTOR = {"none": 4.0, "32B": 2.0, "64B": 1.3, "128B": 1.0}


@lru_cache(maxsize=256)
def chunk_offsets(nbytes: int, swizzle: str) -> np.ndarray:
    """Physical byte offset of every logical 16-byte chunk of a tile.

    Implements the XOR swizzle ``phys = lin ^ (((lin >> 7) & mask) << 4)`` (the
    CUTLASS ``Swizzle<B,4,3>`` family) on byte offsets relative to the tile base.
    """
    lin = np.arange(nbytes // 16, dtype=np.int64) * 16
    span = SWIZZLE_SPAN[swizzle]
    if span == 0:
        return lin
    mask = {32: 1, 64: 3, 128: 7}[span]
    return lin ^ (((lin >> 7) & mask) << 4)


def _gather_index(nbytes: int, swizzle: str) -> np.ndarray:
    offs = chunk_offsets(nbytes, swizzle)
    return (offs[:, None] + np.arange(16, dtype=np.int64)[None, :]).reshape(-1)


_GATHER_CACHE = {}


def gather_index(nbytes: int, swizzle: str) -> np.ndarray:
    key = (nbytes, swizzle)
    idx = _GATHER_CACHE.get(key)
    if idx is None:
        idx = _gather_index(nbytes, swizzle)
        if len(_GATHER_CACHE) < 256:
            _GATHER_CACHE[key] = idx
    return idx


def write_tile(smem: np.ndarray, addr: int, tile: np.ndarray, swizzle: str) -> None:
    raw = np.ascontiguousarray(tile).view(np.uint8).reshape(-1)
    idx = gather_index(raw.size, swizzle)
    smem[addr + idx] = raw


def read_tile(smem: np.ndarray, addr: int, shape: Tuple[int, int], storage: np.dtype,
              swizzle: str) -> np.ndarray:
    nbytes = int(np.prod(shape)) * np.dtype(storage).itemsize
    idx = gather_index(nbytes, swizzle)
    raw = smem[addr + idx]
    return raw.view(storage).reshape(shape)


def tile_ranges(addr: int, nbytes: int) -> Tuple[int, int]:
    return addr, addr + nbytes


def read_box(arr: np.ndarray, coords: Sequence[int], box: Sequence[int]) -> Tuple[np.ndarray, bool]:
    """Read ``box`` elements at ``coords`` with zero fill outside the tensor.

    Returns the box and whether any element was out of bounds.
    """
    out = np.zeros(tuple(box), dtype=arr.dtype)
    src, dst = [], []
    oob = False
    for c, b, n in zip(coords, box, arr.shape):
        lo, hi = max(c, 0), min(c + b, n)
        if lo > c or hi < c + b:
            oob = True
        if hi <= lo:
            return out, True
        src.append(slice(lo, hi))
        dst.append(slice(lo - c, hi - c))
    out[tuple(dst)] = arr[tuple(src)]
    return out, oob


def box_in_bounds(shape: Sequence[int], coords: Sequence[int], box: Sequence[int]) -> bool:
    return all(c >= 0 and c + b <= n for c, b, n in zip(coords, box, shape))


def write_box(arr: np.ndarray, coords: Sequence[int], data: np.ndarray) -> None:
    sl = tuple(slice(c, c + b) for c, b in zip(coords, data.shape))
    arr[sl] = data
