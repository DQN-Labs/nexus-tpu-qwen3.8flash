# NEXUS — Qwen3.8-Flash-Next on vLLM-TPU

NEXUS adds **production-oriented** [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) support to the **vLLM → tpu-inference → TPU/XLA** stack. It is **not** a standalone inference engine.

```
upstream vLLM (qwen4_exp model, PR #53896)
        ↓
tpu-inference (JAX/XLA execution, GDN kernels, scheduler, KV/state)
        ↓
NEXUS overlay (QSA, GR, PLE/n-gram sharding, INT4/3/2/1 n-gram experiments)
        ↓
Kaggle TPU v5e-8 (8 chips, ~128 GB HBM)
```

## What is implemented

| Component | Status | Integration point |
|-----------|--------|-------------------|
| 3× GDN + 1× QSA layer schedule | Config + model skeleton | Official `layer_types` from HF config |
| Gated Residual (GR) | JAX `GatedResidual` | vLLM `hyperconnection.py` parity |
| QSA (K=2048, r=4) | Bounded sparse attention | vLLM QSA reference semantics |
| PLE / n-gram addressing | SplitMix64 + prime vocab sizes | vLLM `ple_layer.py` |
| GDN recurrence | Delegates to tpu-inference GDN bridge | Existing `gdn_attention_op.py` |
| MoE | `JaxRoutedExperts` | tpu-inference MoE path |
| Quantization | INT4 backbone + INT4/3/2/1 n-gram packing | `nexus.quantization` |
| Model registration | `nexus.integration.register` | tpu-inference `model_loader` |

## What requires TPU + checkpoint qualification

- Full 125B + 51B checkpoint load and tensor-name mapping
- Measured per-device HBM (not theoretical GiB tables)
- Compilation vs steady-state benchmarks
- 16 × 32K concurrent session target
- INT3/INT2/INT1 n-gram quality validation

NEXUS does **not** fabricate TPU benchmark numbers when hardware is unavailable.

## Quick start (development)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

Install pinned upstream (vLLM Flash-Next PR + tpu-inference + overlay):

```bash
chmod +x scripts/install_upstream.sh
./scripts/install_upstream.sh
nexus-register
```

Serve on TPU (after install, with checkpoint mounted):

```bash
export JAX_PLATFORMS=tpu
vllm serve Qwen/Qwen3.8-Flash-Next \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --max-num-seqs 16
```

## Quantization configurations

| Config | Approx. raw storage |
|--------|---------------------|
| INT4 backbone + INT4 n-gram | ~88.0 GB |
| INT4 backbone + INT3 n-gram | ~81.6 GB |
| INT4 backbone + INT2 n-gram | ~75.25 GB |
| INT4 backbone + INT1 n-gram | ~68.9 GB |

These are **theoretical** parameter bytes. Runtime HBM includes scales, XLA buffers, activations, recurrent state, and sharding overhead. Use `nexus-memory-report` and the Kaggle notebook for measurement on real hardware.

## Kaggle qualification

Open [`nexus_flash_next_kaggle.ipynb`](nexus_flash_next_kaggle.ipynb) on a **TPU v5e-8** runtime. The notebook:

1. Installs pinned dependencies
2. Verifies 8 TPU devices
3. Registers the NEXUS model
4. Runs CPU-safe smoke tests
5. Attempts model load / generation when a checkpoint is attached
6. Reports measured HBM where available
7. Exports JSON benchmark artifacts

## Upstream pins

See [UPSTREAM.md](UPSTREAM.md).

## Tests

```bash
pytest tests/ -q
ruff check nexus tests
```

Tests cover config parsing, PLE addressing, QSA bounded attention, quantization round-trips, and memory estimation. Full-model parity against vLLM CUDA kernels requires the upstream vLLM test suite on GPU; TPU qualification is separate.
