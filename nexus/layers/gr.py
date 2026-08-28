"""Gated Residual (GR) — faithful JAX port of vLLM ``GatedResidual``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array


def grouped_gemma_rms_norm(
    x: Array,
    weight: Array,
    eps: float,
    hc_count: int,
    hidden_size: int,
) -> Array:
    grouped = x.reshape(*x.shape[:-1], hc_count, hidden_size)
    variance = jnp.mean(grouped.astype(jnp.float32) ** 2, axis=-1, keepdims=True)
    normalized = grouped * jax.lax.rsqrt(variance + eps)
    w = weight.reshape(hc_count, hidden_size)
    return (normalized * (1.0 + w)).reshape(*x.shape[:-1], hc_count * hidden_size).astype(x.dtype)


class GatedResidual(nnx.Module):
    def __init__(
        self,
        hc_count: int,
        hidden_size: int,
        hc_lowrank: int,
        rms_norm_eps: float,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ) -> None:
        self.hc_count = hc_count
        self.hidden_size = hidden_size
        self.hyper_hidden_size = hc_count * hidden_size
        self.hc_lowrank = hc_lowrank
        self.rms_norm_eps = rms_norm_eps
        self.norm_weight = nnx.Param(jnp.zeros((hc_count * hidden_size,), dtype=dtype))
        self.input_mix_weight_down = nnx.Linear(
            self.hyper_hidden_size,
            hc_lowrank,
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs,
        )
        self.input_mix_weight_up = nnx.Linear(
            hc_lowrank,
            self.hyper_hidden_size,
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs,
        )
        self.block_inject_weight = nnx.Linear(
            self.hyper_hidden_size,
            hc_count,
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs,
        )

    def mix(self, hyper_input: Array) -> tuple[Array, tuple[Array, Array]]:
        normed = grouped_gemma_rms_norm(
            hyper_input,
            self.norm_weight.value,
            self.rms_norm_eps,
            self.hc_count,
            self.hidden_size,
        )
        gate = jax.nn.silu(self.input_mix_weight_down(normed) / self.hc_count)
        gate = jax.nn.sigmoid(self.input_mix_weight_up(gate)).reshape(
            *hyper_input.shape[:-1], self.hc_count, self.hidden_size
        )
        branches = normed.reshape(*hyper_input.shape[:-1], self.hc_count, self.hidden_size)
        return jnp.mean(gate * branches, axis=-2).astype(hyper_input.dtype), (hyper_input, normed)

    def combine(self, block_output: Array, residuals: tuple[Array, Array]) -> Array:
        hyper_input, hyper_input_normed = residuals
        residual = hyper_input.reshape(*hyper_input.shape[:-1], self.hc_count, self.hidden_size)
        injection = 2.0 * jax.nn.sigmoid(self.block_inject_weight(hyper_input_normed) / self.hc_count)
        output = residual + block_output[..., None, :] * injection[..., :, None]
        return output.reshape(*hyper_input.shape[:-1], self.hyper_hidden_size).astype(hyper_input.dtype)
