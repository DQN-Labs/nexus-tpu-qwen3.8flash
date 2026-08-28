"""PLE / n-gram embedding addressing — faithful port of vLLM ``ple_layer.py``.

Uses SplitMix64 multipliers and prime-based per-head vocabulary sizes from the official
implementation. Lookup tables are sharded across the TPU mesh; this module handles
deterministic index computation only.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB


def splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime_64(value: int) -> bool:
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    exponent = value - 1
    shifts = 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, exponent, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def nth_prime_after(start: int, count: int) -> int:
    prime = int(start)
    for _ in range(count):
        candidate = prime + 1
        if candidate <= 2:
            prime = 2
            continue
        if candidate % 2 == 0:
            candidate += 1
        while not _is_prime_64(candidate):
            candidate += 2
        prime = candidate
    return prime


def build_ple_tables_metadata(
    *,
    ple_dense_layer_id: int,
    ngram_size: int,
    heads_per_ngram: int,
    ngram_vocab_size_base: int,
    make_divisible_by: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build layer multipliers and per-head vocab metadata (checkpoint-compatible)."""
    ngram_heads = (ngram_size - 1) * heads_per_ngram
    multipliers = []
    for index in range(ngram_size):
        value = splitmix64((ple_dense_layer_id + 1) * 1_000_003 + index)
        half_bound = 1 << 63
        multipliers.append(2 * (value % half_bound) + 1)
    layer_multipliers = np.array(multipliers, dtype=np.uint64)

    sizes = []
    offsets = []
    for local_head in range(ngram_heads):
        global_head = ple_dense_layer_id * ngram_heads + local_head
        size = nth_prime_after(ngram_vocab_size_base - 1, global_head + 1)
        if make_divisible_by > 0:
            size = ((size + make_divisible_by - 1) // make_divisible_by) * make_divisible_by
        sizes.append(size)
        offsets.append(sum(sizes[:-1]) if sizes else 0)
    # Recompute offsets as cumulative (vLLM stores per-head mod sizes, not global offsets)
    sizes_arr = np.array(sizes, dtype=np.int64)
    offsets_arr = np.zeros_like(sizes_arr)
    return layer_multipliers, sizes_arr, offsets_arr


def shift_tokens(tokens: Array, shift: int, eos_token_id: int) -> Array:
    if shift == 0:
        return tokens
    batch, length = tokens.shape
    idx = jnp.arange(length) - shift
    gather = jnp.clip(idx, 0)
    shifted = jnp.take(tokens, gather, axis=1)
    valid = idx[None, :] >= 0
    return jnp.where(valid, shifted, jnp.full_like(shifted, eos_token_id))


def compute_ngram_ids(
    tokens: Array,
    *,
    layer_multipliers: Array,
    head_vocab_sizes: Array,
    ngram_size: int,
    heads_per_ngram: int,
    eos_token_id: int,
) -> Array:
    """Compute n-gram table indices for each token position (XOR mix, mod prime size)."""
    shifted = [tokens]
    for s in range(1, ngram_size):
        shifted.append(shift_tokens(tokens, s, eos_token_id))
    id_blocks = []
    for ngram in range(2, ngram_size + 1):
        start = (ngram - 2) * heads_per_ngram
        end = start + heads_per_ngram
        mixed = shifted[0].astype(jnp.uint64) * jnp.uint64(layer_multipliers[0])
        for index in range(1, ngram):
            mixed = jnp.bitwise_xor(mixed, shifted[index].astype(jnp.uint64) * jnp.uint64(layer_multipliers[index]))
        sizes = head_vocab_sizes[start:end]
        ids = jnp.remainder(mixed[..., None], sizes.astype(jnp.uint64))
        id_blocks.append(ids.astype(jnp.int32))
    return jnp.concatenate(id_blocks, axis=-1)


def ple_lookup(
    ngram_ids: Array,
    embedding_tables: Array,
) -> Array:
    """Gather n-gram embeddings and flatten head dimensions."""
    # embedding_tables: [total_rows, head_dim] or sharded [heads, vocab, dim]
    if embedding_tables.ndim == 3:
        # Per-head tables: [n_heads, vocab, dim]
        head_dim = embedding_tables.shape[-1]
        n_heads = embedding_tables.shape[0]
        flat_ids = ngram_ids.reshape(*ngram_ids.shape[:-1], n_heads)
        out = jnp.stack(
            [
                embedding_tables[h][flat_ids[..., h]]
                for h in range(n_heads)
            ],
            axis=-2,
        )
        return out.reshape(*ngram_ids.shape[:-1], n_heads * head_dim)
    return jnp.take(embedding_tables, ngram_ids, axis=0).reshape(
        *ngram_ids.shape[:-1], -1
    )
