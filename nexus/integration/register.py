"""Register Qwen4Exp with tpu-inference at runtime (no fork edit required)."""

from __future__ import annotations


def register_qwen4_exp() -> None:
    try:
        from tpu_inference.models.common import model_loader
    except ImportError as exc:
        raise RuntimeError(
            "tpu-inference is not installed. Run ./scripts/install_upstream.sh first."
        ) from exc
    try:
        from tpu_inference.models.jax.qwen4_exp import Qwen4ExpForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "Qwen4Exp JAX model not found. Ensure the NEXUS overlay was applied to tpu-inference."
        ) from exc

    model_loader._MODEL_REGISTRY["Qwen4ExpForConditionalGeneration"] = Qwen4ExpForCausalLM
    model_loader._MODEL_REGISTRY["Qwen4ExpForCausalLM"] = Qwen4ExpForCausalLM


def main() -> None:
    register_qwen4_exp()
    print("Registered Qwen4ExpForConditionalGeneration with tpu-inference.")


if __name__ == "__main__":
    main()
