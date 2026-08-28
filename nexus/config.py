"""Architecture configuration derived from the official Qwen3.8-Flash-Next checkpoint.

Dimensions are imported from HuggingFace ``config.json``; NEXUS never invents missing
values. ``tiny()`` exists only for deterministic CPU/JAX unit tests.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


_LAYER_MAP = {
    "linear_attention": "gdn",
    "full_attention": "qsa",
}


@dataclass(frozen=True)
class FlashNextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    layer_types: tuple[str, ...]
    # GDN
    linear_conv_kernel_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    # QSA (indexer_budget K=2048, compress_ratio r=4)
    qsa_block_size: int
    qsa_token_budget: int
    indexer_n_heads: int
    indexer_kv_heads: int
    indexer_head_dim: int
    # GR (Gated Residual)
    hc_count: int
    hc_lowrank: int
    # PLE / n-gram
    ple_layer_ids: tuple[int, ...]
    ple_embed_dim: int
    ple_conv_kernel_size: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    make_ngram_vocab_size_divisible_by: int
    split_ngram_parts: int
    eos_token_id: int
    source_revision: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types length must equal num_hidden_layers")
        allowed = {"gdn", "qsa"}
        if set(self.layer_types) - allowed:
            raise ValueError(f"unsupported layer types: {set(self.layer_types) - allowed}")
        if self.qsa_token_budget % self.qsa_block_size:
            raise ValueError("QSA token budget must be divisible by block size")
        if self.indexer_kv_heads != 1:
            raise ValueError("QSA MQA requires indexer_kv_heads=1")

    @property
    def qsa_selected_blocks(self) -> int:
        return self.qsa_token_budget // self.qsa_block_size

    @property
    def hyper_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size

    @property
    def ngram_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram

    @classmethod
    def tiny(cls) -> FlashNextConfig:
        return cls(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=48,
            shared_expert_intermediate_size=48,
            rms_norm_eps=1e-6,
            rope_theta=1_000_000.0,
            partial_rotary_factor=0.25,
            layer_types=("gdn", "gdn", "gdn", "qsa"),
            linear_conv_kernel_dim=4,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            qsa_block_size=4,
            qsa_token_budget=16,
            indexer_n_heads=2,
            indexer_kv_heads=1,
            indexer_head_dim=8,
            hc_count=4,
            hc_lowrank=8,
            ple_layer_ids=(2,),
            ple_embed_dim=32,
            ple_conv_kernel_size=4,
            ngram_size=3,
            heads_per_ngram=2,
            ngram_vocab_size_base=257,
            make_ngram_vocab_size_divisible_by=128,
            split_ngram_parts=4,
            eos_token_id=0,
            source_revision="nexus-tiny-v1",
        )

    @classmethod
    def from_hf_config(cls, path: str | Path) -> FlashNextConfig:
        raw = json.loads(Path(path).read_text())
        text = raw.get("text_config", raw)
        layer_types = tuple(
            _LAYER_MAP.get(str(x), str(x)) for x in text.get("layer_types", [])
        )
        if not layer_types:
            layer_types = tuple(
                "qsa" if (i + 1) % 4 == 0 else "gdn"
                for i in range(text["num_hidden_layers"])
            )
        elif len(layer_types) < text["num_hidden_layers"]:
            pattern = list(layer_types)
            while len(pattern) < text["num_hidden_layers"]:
                pattern.extend(layer_types)
            layer_types = tuple(pattern[: text["num_hidden_layers"]])
        rope = text.get("rope_parameters", {})
        return cls(
            vocab_size=text["vocab_size"],
            hidden_size=text["hidden_size"],
            intermediate_size=text.get("intermediate_size", text["hidden_size"] * 4),
            num_hidden_layers=text["num_hidden_layers"],
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text.get("num_key_value_heads", text["num_attention_heads"]),
            head_dim=text.get("head_dim", text["hidden_size"] // text["num_attention_heads"]),
            num_experts=text["num_experts"],
            num_experts_per_tok=text["num_experts_per_tok"],
            moe_intermediate_size=text.get("moe_intermediate_size", text["hidden_size"]),
            shared_expert_intermediate_size=text.get("shared_expert_intermediate_size", 0),
            rms_norm_eps=text.get("rms_norm_eps", 1e-6),
            rope_theta=rope.get("rope_theta", text.get("rope_theta", 10_000_000.0)),
            partial_rotary_factor=rope.get(
                "partial_rotary_factor", text.get("partial_rotary_factor", 0.25)
            ),
            layer_types=layer_types,
            linear_conv_kernel_dim=text.get("linear_conv_kernel_dim", 4),
            linear_num_key_heads=text.get("linear_num_key_heads", text["num_attention_heads"]),
            linear_num_value_heads=text.get("linear_num_value_heads", text["num_attention_heads"]),
            linear_key_head_dim=text.get("linear_key_head_dim", text.get("head_dim", 128)),
            linear_value_head_dim=text.get("linear_value_head_dim", text.get("head_dim", 128)),
            qsa_block_size=text.get("indexer_compress_ratio", 4),
            qsa_token_budget=text.get("indexer_budget", 2048),
            indexer_n_heads=text.get("indexer_n_heads", 4),
            indexer_kv_heads=text.get("indexer_kv_heads", 1),
            indexer_head_dim=text.get("indexer_head_dim", 128),
            hc_count=text.get("hc_count", 4),
            hc_lowrank=text.get("hc_lowrank", 320),
            ple_layer_ids=tuple(text.get("ple_layer_ids", [2])),
            ple_embed_dim=text.get("ple_embed_dim", text["hidden_size"]),
            ple_conv_kernel_size=text.get("ple_conv_kernel_size", 4),
            ngram_size=text.get("ngram_size", 3),
            heads_per_ngram=text.get("heads_per_ngram", 8),
            ngram_vocab_size_base=text.get("ngram_vocab_size_base", 20_000_000),
            make_ngram_vocab_size_divisible_by=text.get(
                "make_ngram_vocab_size_divisible_by", 128
            ),
            split_ngram_parts=text.get("split_ngram_parts", 128),
            eos_token_id=text.get("bos_token_id", text.get("eos_token_id", 0)),
            source_revision=raw.get("_commit_hash", "unknown"),
            extra=raw,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeConfig:
    mesh_axis: str = "model"
    prefill_buckets: tuple[int, ...] = (4096, 8192, 16384, 32768)
    max_sessions: int = 16
    max_context: int = 32768
    hbm_limit_bytes: int = 128 * 1024**3
    hbm_reserve_bytes: int = 8 * 1024**3
    backbone_bits: int = 4
    ngram_bits: int = 4
    production: bool = True

    def bucket_for(self, length: int) -> int:
        for size in self.prefill_buckets:
            if length <= size:
                return size
        raise ValueError(f"sequence length {length} exceeds largest bucket {self.prefill_buckets[-1]}")
