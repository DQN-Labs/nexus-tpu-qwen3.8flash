#!/usr/bin/env bash
# Install pinned upstream vLLM (Flash-Next PR) and tpu-inference with NEXUS overlay.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${NEXUS_BUILD_DIR:-${ROOT}/.upstream}"
VLLM_COMMIT="2a4cd640ff1a61b66124ddbaaf02a73781f7295a"
TPU_COMMIT="c5c8a055edfa7853fe1cb9e8873c027d931ab490"

mkdir -p "${BUILD}"

clone_or_checkout() {
  local url="$1" dir="$2" commit="$3"
  if [[ ! -d "${dir}/.git" ]]; then
    git clone --depth 1 "${url}" "${dir}"
  fi
  git -C "${dir}" fetch --depth 1 origin "${commit}" 2>/dev/null || git -C "${dir}" fetch origin
  git -C "${dir}" checkout -q "${commit}"
}

clone_or_checkout "https://github.com/vllm-project/vllm.git" "${BUILD}/vllm" "${VLLM_COMMIT}"
clone_or_checkout "https://github.com/vllm-project/tpu-inference.git" "${BUILD}/tpu-inference" "${TPU_COMMIT}"

# Apply NEXUS overlay onto tpu-inference (model registration + qwen4_exp JAX tree).
rsync -a --delete "${ROOT}/patches/tpu-inference/" "${BUILD}/tpu-inference/"

python -m pip install -e "${ROOT}[dev]"
python -m pip install -e "${BUILD}/vllm"
python -m pip install -e "${BUILD}/tpu-inference"

echo "NEXUS upstream install complete."
echo "  vLLM:          ${VLLM_COMMIT}"
echo "  tpu-inference: ${TPU_COMMIT}"
