"""HBM planning and live device memory reporting."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import jax

from nexus.config import FlashNextConfig, RuntimeConfig
from nexus.quantization.packing import estimate_storage_gib


@dataclass(frozen=True)
class MemoryEstimate:
    weights: int
    ngram: int
    state: int
    activations: int
    reserve: int
    source: str = "estimated"

    @property
    def total(self) -> int:
        return self.weights + self.ngram + self.state + self.activations + self.reserve

    def as_gib(self) -> dict[str, float | str]:
        values = {
            k: getattr(self, k) / 1024**3
            for k in ("weights", "ngram", "state", "activations", "reserve")
        }
        return {**values, "total": self.total / 1024**3, "source": self.source}


def estimate_runtime_memory(
    model: FlashNextConfig,
    runtime: RuntimeConfig,
    sessions: int,
    context: int,
) -> MemoryEstimate:
    theory = estimate_storage_gib(
        backbone_bits=runtime.backbone_bits,
        ngram_bits=runtime.ngram_bits,
    )
    weights = int(theory["backbone_gib"] * 1024**3)
    ngram = int(theory["ngram_gib"] * 1024**3)
    gdn_layers = sum(t == "gdn" for t in model.layer_types)
    qsa_layers = sum(t == "qsa" for t in model.layer_types)
    gdn_state = (
        sessions
        * gdn_layers
        * model.linear_num_key_heads
        * model.linear_key_head_dim
        * model.linear_value_head_dim
        * 4
    )
    blocks = (context + model.qsa_block_size - 1) // model.qsa_block_size
    qsa_kv = (
        sessions
        * qsa_layers
        * blocks
        * model.qsa_block_size
        * model.num_key_value_heads
        * model.head_dim
        * 2
        * 2
    )
    gr_state = sessions * model.hc_count * model.hidden_size * 2 * 2
    state = gdn_state + qsa_kv + gr_state
    bucket = runtime.bucket_for(min(context, runtime.max_context))
    activations = sessions * bucket * model.hidden_size * model.hc_count * 2
    return MemoryEstimate(weights, ngram, state, activations, runtime.hbm_reserve_bytes)


def live_device_memory() -> list[dict]:
    reports = []
    for device in jax.devices():
        stats = device.memory_stats() or {}
        reports.append(
            {
                "device": str(device),
                "platform": str(device.platform),
                "source": "measured" if stats else "unavailable",
                **{k: int(v) for k, v in stats.items() if isinstance(v, (int, float))},
            }
        )
    return reports


def assert_fits(estimate: MemoryEstimate, runtime: RuntimeConfig) -> None:
    if estimate.total > runtime.hbm_limit_bytes:
        raise MemoryError(
            f"estimated {estimate.total / 1024**3:.2f} GiB exceeds "
            f"configured limit {runtime.hbm_limit_bytes / 1024**3:.2f} GiB"
        )


def cli() -> None:
    parser = argparse.ArgumentParser(description="NEXUS memory estimate / device report")
    parser.add_argument("--sessions", type=int, default=16)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--backbone-bits", type=int, default=4)
    parser.add_argument("--ngram-bits", type=int, default=4)
    args = parser.parse_args()
    runtime = RuntimeConfig(backbone_bits=args.backbone_bits, ngram_bits=args.ngram_bits)
    est = estimate_runtime_memory(FlashNextConfig.tiny(), runtime, args.sessions, args.context)
    print(json.dumps({"estimate_gib": est.as_gib(), "devices": live_device_memory()}, indent=2))


if __name__ == "__main__":
    cli()
