#!/usr/bin/env bash

# Install pinned upstream vLLM (Flash-Next PR) and tpu-inference with NEXUS overlay.

set -euo pipefail

export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-tpu}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${NEXUS_BUILD_DIR:-${ROOT}/.upstream}"

VLLM_COMMIT="2a4cd640ff1a61b66124ddbaaf02a73781f7295a"
TPU_COMMIT="c5c8a055edfa7853fe1cb9e8873c027d931ab490"

mkdir -p "${BUILD}"

clone_or_checkout() {
    local url="$1"
    local dir="$2"
    local commit="$3"

    if [[ ! -d "${dir}/.git" ]]; then
        echo "Cloning ${url}..."
        git clone --depth 1 "${url}" "${dir}"
    else
        echo "Using existing repository: ${dir}"
    fi

    git -C "${dir}" fetch --depth 1 origin "${commit}" 2>/dev/null \
        || git -C "${dir}" fetch origin

    git -C "${dir}" checkout -q "${commit}"
}

# Reuse existing clones when available.
clone_or_checkout \
    "https://github.com/vllm-project/vllm.git" \
    "${BUILD}/vllm" \
    "${VLLM_COMMIT}"

clone_or_checkout \
    "https://github.com/vllm-project/tpu-inference.git" \
    "${BUILD}/tpu-inference" \
    "${TPU_COMMIT}"

# Apply NEXUS overlay onto tpu-inference.
rsync -a --delete \
    "${ROOT}/patches/tpu-inference/" \
    "${BUILD}/tpu-inference/"

# Install build requirements into the actual environment.
python -m pip install --no-build-isolation -q \
    setuptools \
    setuptools-rust \
    wheel \
    ninja \
    cmake

# Install NEXUS itself.
python -m pip install --no-build-isolation -e "${ROOT}[dev]"

# Build/install vLLM explicitly for TPU.
# --no-build-isolation is important because an isolated environment
# was previously pulling in a CUDA-dependent PyTorch build.
VLLM_TARGET_DEVICE=tpu \
python -m pip install --no-build-isolation -e "${BUILD}/vllm"

# Install TPU inference after vLLM.
python -m pip install --no-build-isolation -e "${BUILD}/tpu-inference"

echo
echo "NEXUS upstream install complete."
echo "  VLLM_TARGET_DEVICE: ${VLLM_TARGET_DEVICE}"
echo "  vLLM:               ${VLLM_COMMIT}"
echo "  tpu-inference:      ${TPU_COMMIT}"
