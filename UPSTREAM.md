# Upstream pins

NEXUS is a focused integration layer on top of upstream vLLM and tpu-inference. These commits are the authoritative bases for development and qualification.

| Component | Repository | Commit | Notes |
|-----------|------------|--------|-------|
| vLLM (Flash-Next model) | https://github.com/vllm-project/vllm | `2a4cd640ff1a61b66124ddbaaf02a73781f7295a` | PR #53896 head (`qwen4_exp` model tree) |
| tpu-inference | https://github.com/vllm-project/tpu-inference | `c5c8a055edfa7853fe1cb9e8873c027d931ab490` | JAX/XLA backend, GDN kernels, model loader |
| Official checkpoint config | https://huggingface.co/Qwen/Qwen3.8-Flash-Next | `main` config.json | `Qwen4ExpForConditionalGeneration` / `qwen4_exp_text` |

Install with:

```bash
./scripts/install_upstream.sh
```

This clones the pinned commits, applies the NEXUS tpu-inference overlay from `patches/tpu-inference/`, and installs the editable `nexus` package.
