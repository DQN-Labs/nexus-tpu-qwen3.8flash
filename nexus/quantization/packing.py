"""Groupwise symmetric quantization with dense sub-byte packing (INT4/3/2/1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SUPPORTED_BITS = (1, 2, 3, 4, 8)


@dataclass
class QuantizedArray:
    packed: np.ndarray
    scales: np.ndarray
    bits: int
    group_size: int
    shape: tuple[int, ...]
    signed: bool = True

    @property
    def logical_elements(self) -> int:
        return int(np.prod(self.shape))

    @property
    def total_bytes(self) -> int:
        return int(self.packed.nbytes + self.scales.nbytes)

    @property
    def bits_per_weight(self) -> float:
        return self.total_bytes * 8 / max(1, self.logical_elements)


def _code_range(bits: int, signed: bool) -> tuple[int, int]:
    if not signed:
        return 0, (1 << bits) - 1
    if bits == 1:
        return -1, 0
    half = 1 << (bits - 1)
    return -half, half - 1


def _pack_bits(codes: np.ndarray, bits: int) -> np.ndarray:
    flat = codes.astype(np.uint32).ravel()
    bit_matrix = ((flat[:, None] >> np.arange(bits, dtype=np.uint32)[None, :]) & 1).astype(np.uint8)
    stream = bit_matrix.reshape(-1)
    pad = (-stream.size) % 8
    if pad:
        stream = np.concatenate([stream, np.zeros(pad, dtype=np.uint8)])
    stream = stream.reshape(-1, 8)
    weights = (1 << np.arange(8, dtype=np.uint8))[None, :]
    return (stream * weights).sum(axis=1).astype(np.uint8)


def _unpack_bits(packed: np.ndarray, bits: int, count: int) -> np.ndarray:
    bit_stream = ((packed[:, None] >> np.arange(8, dtype=np.uint8)[None, :]) & 1).astype(np.uint8)
    bit_stream = bit_stream.reshape(-1)[: count * bits].reshape(count, bits)
    weights = (1 << np.arange(bits, dtype=np.uint32))[None, :]
    return (bit_stream.astype(np.uint32) * weights).sum(axis=1)


def quantize_symmetric(
    array: np.ndarray, bits: int, group_size: int = 128, signed: bool = True
) -> QuantizedArray:
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"unsupported bit width {bits}")
    arr = np.asarray(array, dtype=np.float32)
    shape = arr.shape
    last = shape[-1]
    if last % group_size:
        pad = group_size - (last % group_size)
        arr = np.pad(arr, [(0, 0)] * (arr.ndim - 1) + [(0, pad)])
        last = arr.shape[-1]
    n_groups = last // group_size
    grouped = arr.reshape(*shape[:-1], n_groups, group_size)
    lo, hi = _code_range(bits, signed)
    max_abs = np.max(np.abs(grouped), axis=-1, keepdims=True)
    scale = np.where(max_abs > 0, max_abs / max(1, hi), 1.0).astype(np.float32)
    codes = np.clip(np.round(grouped / scale), lo, hi).astype(np.int32)
    unsigned = (codes - lo).astype(np.uint32)
    packed = _pack_bits(unsigned, bits)
    return QuantizedArray(
        packed=packed,
        scales=scale.reshape(*shape[:-1], n_groups).astype(np.float32),
        bits=bits,
        group_size=group_size,
        shape=shape,
        signed=signed,
    )


def dequantize(q: QuantizedArray) -> np.ndarray:
    lo, _ = _code_range(q.bits, q.signed)
    last = q.shape[-1]
    padded_last = q.scales.shape[-1] * q.group_size
    count = int(np.prod(q.shape[:-1])) * padded_last
    unsigned = _unpack_bits(q.packed, q.bits, count)
    signed_codes = unsigned.astype(np.int32) + lo
    grouped = signed_codes.reshape(*q.shape[:-1], q.scales.shape[-1], q.group_size)
    return (grouped.astype(np.float32) * q.scales[..., None]).reshape(*q.shape[:-1], padded_last)[
        ..., :last
    ]


def estimate_storage_gib(
  *,
  backbone_params: int = 125_000_000_000,
  ngram_params: int = 51_000_000_000,
  backbone_bits: int = 4,
  ngram_bits: int = 4,
  metadata_overhead: float = 1.08,
) -> dict[str, float]:
    """Theoretical raw storage from the mission brief (not measured HBM)."""
    backbone = backbone_params * backbone_bits / 8
    ngram = ngram_params * ngram_bits / 8
    total = (backbone + ngram) * metadata_overhead
    return {
        "backbone_gib": backbone / 1024**3 * metadata_overhead,
        "ngram_gib": ngram / 1024**3 * metadata_overhead,
        "total_gib": total / 1024**3,
        "backbone_bits": backbone_bits,
        "ngram_bits": ngram_bits,
        "source": "theoretical",
    }
