import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from nexus.config import FlashNextConfig, RuntimeConfig
from nexus.layers.ple import build_ple_tables_metadata, compute_ngram_ids, splitmix64
from nexus.layers.qsa import expand_qsa_indices, qsa_prefill
from nexus.quantization.packing import dequantize, quantize_symmetric


def test_official_config_parses(tmp_path: Path):
    cfg = {
        "text_config": {
            "vocab_size": 248320,
            "hidden_size": 2560,
            "num_hidden_layers": 48,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "hc_count": 4,
            "ple_layer_ids": [2],
        }
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    model = FlashNextConfig.from_hf_config(path)
    assert model.qsa_token_budget == 2048
    assert model.qsa_block_size == 4
    assert model.qsa_selected_blocks == 512
    assert model.hc_count == 4


def test_splitmix_is_deterministic():
    assert splitmix64(42) == splitmix64(42)
    assert splitmix64(42) != splitmix64(43)


def test_ple_metadata_shapes():
    mult, sizes, offsets = build_ple_tables_metadata(
        ple_dense_layer_id=0,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=257,
        make_divisible_by=128,
    )
    assert mult.shape == (3,)
    assert sizes.shape == (4,)
    assert offsets.shape == (4,)


def test_ngram_ids_bounded():
    tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    mult, sizes, _ = build_ple_tables_metadata(
        ple_dense_layer_id=0,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=257,
        make_divisible_by=128,
    )
    ids = compute_ngram_ids(
        tokens,
        layer_multipliers=jnp.asarray(mult),
        head_vocab_sizes=jnp.asarray(sizes),
        ngram_size=3,
        heads_per_ngram=2,
        eos_token_id=0,
    )
    assert ids.shape == (1, 4, 4)
    assert bool(jnp.all(ids >= 0))


def test_expand_qsa_indices_includes_tail():
    blocks = jnp.array([[3, 1, -1, -1]], dtype=jnp.int32)
    out = expand_qsa_indices(
        blocks,
        jnp.array([9]),
        jnp.array([20]),
        compress_ratio=4,
        token_topk=16,
    )
    assert out.shape[1] == 16 + 4 - 1
    assert bool(jnp.any(out[0] >= 0))


def test_qsa_prefill_no_nan():
    cfg = FlashNextConfig.tiny()
    rng = np.random.default_rng(0)
    h = cfg.hidden_size
    x = jnp.asarray(rng.normal(size=(1, 8, h)), dtype=jnp.bfloat16)
    wq = jnp.asarray(rng.normal(size=(h, cfg.num_attention_heads * cfg.head_dim)) * 0.02)
    wk = jnp.asarray(rng.normal(size=(h, cfg.num_key_value_heads * cfg.head_dim)) * 0.02)
    wv = jnp.asarray(rng.normal(size=(h, cfg.num_key_value_heads * cfg.head_dim)) * 0.02)
    wo = jnp.asarray(rng.normal(size=(cfg.num_attention_heads * cfg.head_dim, h)) * 0.02)
    out, cache = qsa_prefill(
        x, wq, wk, wv, wo,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.qsa_block_size,
        cfg.qsa_token_budget,
    )
    assert out.shape == x.shape
    assert not bool(jnp.any(jnp.isnan(out)))
    assert "indices" in cache


@pytest.mark.parametrize("bits", [4, 3, 2, 1])
def test_quant_roundtrip(bits: int):
    arr = np.random.randn(4, 128).astype(np.float32)
    q = quantize_symmetric(arr, bits=bits, group_size=64)
    restored = dequantize(q)
    assert restored.shape == arr.shape
    corr = np.corrcoef(arr.ravel(), restored.ravel())[0, 1]
    assert corr > 0.5 if bits >= 2 else corr > 0.0


def test_memory_estimate_fits_tiny_on_cpu():
    est = __import__("nexus.memory.planner", fromlist=["estimate_runtime_memory"]).estimate_runtime_memory(
        FlashNextConfig.tiny(),
        RuntimeConfig(hbm_limit_bytes=128 * 1024**3, production=False),
        sessions=2,
        context=64,
    )
    assert est.total > 0
