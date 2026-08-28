# NEXUS overlay patch for tpu_inference.models.common.model_loader
# Applied by scripts/install_upstream.sh via rsync.

# Add to _get_model_architecture imports and registry:
#
#     from tpu_inference.models.jax.qwen4_exp import Qwen4ExpForCausalLM
#     _MODEL_REGISTRY["Qwen4ExpForConditionalGeneration"] = Qwen4ExpForCausalLM
#     _MODEL_REGISTRY["Qwen4ExpForCausalLM"] = Qwen4ExpForCausalLM

PATCH_IMPORT = "from tpu_inference.models.jax.qwen4_exp import Qwen4ExpForCausalLM"
PATCH_LINES = """
    _MODEL_REGISTRY[\"Qwen4ExpForConditionalGeneration\"] = Qwen4ExpForCausalLM
    _MODEL_REGISTRY[\"Qwen4ExpForCausalLM\"] = Qwen4ExpForCausalLM
"""
