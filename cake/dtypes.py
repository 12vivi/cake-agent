"""Element types and their storage encodings.

NumPy has no bfloat16/fp8, so the simulator stores them as raw bit patterns
(uint16 / uint8) and decodes to float32 for arithmetic, which is also what the
tensor cores do (BF16 x BF16 products accumulated in FP32).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class DType:
    name: str
    bits: int
    kind: str  # "float" | "int"

    @property
    def nbytes(self) -> int:
        return self.bits // 8

    @property
    def storage(self) -> np.dtype:
        return np.dtype({"bf16": np.uint16, "f16": np.float16, "f32": np.float32,
                         "i32": np.int32, "fp8": np.uint8}[self.name])

    @property
    def compute(self) -> np.dtype:
        return np.dtype(np.int32 if self.kind == "int" else np.float32)

    def __repr__(self) -> str:
        return f"lm.{self.name}"


bf16 = DType("bf16", 16, "float")
f16 = DType("f16", 16, "float")
f32 = DType("f32", 32, "float")
i32 = DType("i32", 32, "int")
fp8 = DType("fp8", 8, "float")  # e4m3

DTYPES: Dict[str, DType] = {d.name: d for d in (bf16, f16, f32, i32, fp8)}


def get(name) -> DType:
    if isinstance(name, DType):
        return name
    return DTYPES[str(name)]


# ---------------------------------------------------------------------------
# bf16
def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 -> bf16 bit pattern."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    rounded = u + 0x7FFF + ((u >> 16) & 1)
    return ((rounded >> 16) & 0xFFFF).astype(np.uint16)


def bf16_bits_to_f32(b: np.ndarray) -> np.ndarray:
    b = np.ascontiguousarray(b, dtype=np.uint16)
    return (b.astype(np.uint32) << 16).view(np.float32)


# ---------------------------------------------------------------------------
# fp8 e4m3 (finite variant, max 448) via a 256-entry table
def _fp8_table() -> np.ndarray:
    vals = np.zeros(256, dtype=np.float32)
    for code in range(256):
        s, e, m = (code >> 7) & 1, (code >> 3) & 0xF, code & 7
        if e == 0:
            v = (m / 8.0) * 2.0 ** -6
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        if e == 15 and m == 7:
            v = np.nan
        vals[code] = -v if s else v
    return vals


_FP8_VALUES = _fp8_table()
_FP8_POS = _FP8_VALUES[:127]  # codes 0..126: non-negative finite, ascending


def f32_to_fp8_bits(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    a = np.minimum(np.abs(x), 448.0)
    idx = np.searchsorted(_FP8_POS, a)
    idx = np.clip(idx, 1, 126)
    lo, hi = _FP8_POS[idx - 1], _FP8_POS[idx]
    code = np.where(a - lo <= hi - a, idx - 1, idx).astype(np.uint8)
    return np.where(np.signbit(x) & (code != 0), code | 0x80, code).astype(np.uint8)


def fp8_bits_to_f32(b: np.ndarray) -> np.ndarray:
    return _FP8_VALUES[np.asarray(b, dtype=np.uint8)]


# ---------------------------------------------------------------------------
def encode(x: np.ndarray, dt: DType) -> np.ndarray:
    """float/int values -> storage array for ``dt``."""
    if dt.name == "bf16":
        return f32_to_bf16_bits(x)
    if dt.name == "fp8":
        return f32_to_fp8_bits(x)
    return np.asarray(x).astype(dt.storage)


def decode(a: np.ndarray, dt: DType) -> np.ndarray:
    """storage array -> float32 (or int32) compute values."""
    if dt.name == "bf16":
        return bf16_bits_to_f32(a)
    if dt.name == "fp8":
        return fp8_bits_to_f32(a)
    return np.asarray(a).astype(dt.compute)


def quantize(x: np.ndarray, dt: DType) -> np.ndarray:
    """Round float values to the precision of ``dt`` (returned as float32)."""
    return decode(encode(x, dt), dt)
