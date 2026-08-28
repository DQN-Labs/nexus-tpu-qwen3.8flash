"""Qwen Sparse Attention (QSA) — bounded sparse attention (K=2048, r=4)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import Array, lax


def repeat_kv(x: Array, num_heads: int) -> Array:
    repeats = num_heads // x.shape[-2]
    return jnp.repeat(x, repeats, axis=-2)


def expand_qsa_indices(
    block_indices: Array,
    query_positions: Array,
    sequence_lengths: Array,
    compress_ratio: int,
    token_topk: int,
) -> Array:
    rows = block_indices.shape[0]
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    offsets = jnp.arange(compress_ratio)
    blocks = block_indices.astype(jnp.int32)
    if blocks.shape[-1] < block_topk:
        blocks = jnp.pad(blocks, ((0, 0), (0, block_topk - blocks.shape[-1])), constant_values=-1)
    expanded = blocks[..., None] * compress_ratio + offsets
    expanded = jnp.where(blocks[..., None] >= 0, expanded, -1).reshape(rows, block_topk * compress_ratio)
    expanded = expanded[:, :token_topk]
    expanded = jnp.where(
        (expanded >= 0) & (expanded < sequence_lengths[:, None]),
        expanded,
        -1,
    )
    tail_offsets = jnp.arange(compress_ratio - 1)
    visible_tokens = query_positions + 1
    tail_start = (visible_tokens // compress_ratio) * compress_ratio
    tail = tail_start[:, None] + tail_offsets[None, :]
    tail_count = (visible_tokens - tail_start)[:, None]
    tail_valid = (tail_offsets[None, :] < tail_count) & (tail < sequence_lengths[:, None])
    tail = jnp.where(tail_valid, tail, -1)
    result = jnp.concatenate([expanded, tail], axis=1)
    order = jnp.broadcast_to(jnp.arange(output_width), result.shape)
    sort_key = jnp.where(result >= 0, order, order + output_width)
    return jnp.take_along_axis(result, jnp.argsort(sort_key, axis=1), axis=1).astype(jnp.int32)


def qsa_select_blocks(
    q: Array,
    k_blocks: Array,
    query_positions: Array,
    sequence_lengths: Array,
    token_budget: int,
    block_size: int,
) -> Array:
    batch, length, num_heads, head_dim = q.shape
    n_blocks = k_blocks.shape[1]
    compressed = jnp.mean(k_blocks.astype(jnp.float32), axis=2)
    compressed = repeat_kv(compressed, num_heads)
    block_topk = token_budget // block_size

    def score_row(query: Array, position: Array, seq_len: Array) -> Array:
        scores = jnp.einsum("hd,nhd->nh", query.astype(jnp.float32), compressed[0])
        scores = jnp.maximum(scores, 0.0).sum(axis=-1) / math.sqrt(head_dim)
        block_ids = jnp.arange(n_blocks)
        visible = jnp.minimum((position + 1) // block_size, seq_len // block_size)
        scores = jnp.where(block_ids <= visible, scores, -jnp.inf)
        _, top = lax.top_k(scores, min(n_blocks, block_topk))
        return expand_qsa_indices(
            top[None, :],
            position[None],
            seq_len[None],
            block_size,
            token_budget,
        )[0]

    return jax.vmap(lambda pos: score_row(q[0, pos], pos, sequence_lengths[0]))(jnp.arange(length))


def qsa_prefill(
    x: Array,
    q_proj_w: Array,
    k_proj_w: Array,
    v_proj_w: Array,
    o_proj_w: Array,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    token_budget: int,
) -> tuple[Array, dict]:
    batch, length, _hidden = x.shape
    padded = ((length + block_size - 1) // block_size) * block_size
    n_blocks = padded // block_size
    pad = padded - length
    q = (x @ q_proj_w).reshape(batch, length, num_heads, head_dim)
    k = (x @ k_proj_w).reshape(batch, length, num_kv_heads, head_dim)
    v = (x @ v_proj_w).reshape(batch, length, num_kv_heads, head_dim)
    kp = jnp.pad(k, ((0, 0), (0, pad), (0, 0), (0, 0)))
    k_blocks = kp.reshape(batch, n_blocks, block_size, num_kv_heads, head_dim)
    seq_lens = jnp.array([length] * batch)
    indices = qsa_select_blocks(q, k_blocks, jnp.arange(length), seq_lens, token_budget, block_size)
    k_flat = repeat_kv(k, num_heads)
    v_flat = repeat_kv(v, num_heads)

    def attend_one(query: Array, idx: Array) -> Array:
        valid = idx >= 0
        safe = jnp.clip(jnp.where(valid, idx, 0), 0, length - 1)
        gk = jnp.take(k_flat[0], safe, axis=0)
        gv = jnp.take(v_flat[0], safe, axis=0)
        logits = jnp.einsum("hd,thd->ht", query.astype(jnp.float32), gk.astype(jnp.float32))
        logits = jnp.where(valid[None, :], logits, -jnp.inf)
        logits = jnp.where(jnp.any(valid), logits, jnp.zeros_like(logits))
        probs = jax.nn.softmax(logits / math.sqrt(head_dim), axis=-1)
        return jnp.einsum("ht,thd->hd", probs, gv.astype(jnp.float32))

    output = jax.vmap(attend_one, in_axes=(0, 0), out_axes=0)(q[0], indices)
    output = output[None, ...]
    output = output.reshape(batch, length, -1) @ o_proj_w
    return output.astype(x.dtype), {"k_blocks": k_blocks, "indices": indices}
