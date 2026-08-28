#!/usr/bin/env bash

# Install pinned upstream vLLM (Flash-Next PR) and tpu-inference
# with the NEXUS overlay.
#
# Target: Kaggle TPU v5e-8
# Build host: Linux / Python 3.12
#
# IMPORTANT:
# - Do NOT use pip build isolation for vLLM.
# - The Kaggle TPU environment intentionally has no CUDA toolkit.
# - VLLM_TARGET_DEVICE must remain "tpu".

set -euo pipefail

export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-tpu}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${NEXUS_BUILD_DIR:-${ROOT}/.upstream}"

VLLM_COMMIT="2a4cd640ff1a61b66124ddbaaf02a73781f7295a"
TPU_COMMIT="c5c8a055edfa7853fe1cb9e8873c027d931ab490"

mkdir -p "${BUILD}"

echo "========================================"
echo "NEXUS upstream installation"
echo "========================================"
echo "ROOT:               ${ROOT}"
echo "BUILD:              ${BUILD}"
echo "VLLM_TARGET_DEVICE: ${VLLM_TARGET_DEVICE}"
echo "vLLM commit:        ${VLLM_COMMIT}"
echo "TPU commit:         ${TPU_COMMIT}"
echo

# ---------------------------------------------------------------------------
# Clone / reuse pinned upstream repositories
# ---------------------------------------------------------------------------

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

    echo "Checking out ${commit}..."

    # Try shallow fetch first. If the commit cannot be fetched shallowly,
    # fall back to a normal fetch.
    git -C "${dir}" fetch --depth 1 origin "${commit}" 2>/dev/null \
        || git -C "${dir}" fetch origin

    git -C "${dir}" checkout -q "${commit}"
}

clone_or_checkout \
    "https://github.com/vllm-project/vllm.git" \
    "${BUILD}/vllm" \
    "${VLLM_COMMIT}"

clone_or_checkout \
    "https://github.com/vllm-project/tpu-inference.git" \
    "${BUILD}/tpu-inference" \
    "${TPU_COMMIT}"

# ---------------------------------------------------------------------------
# Apply NEXUS overlay
# ---------------------------------------------------------------------------

echo
echo "Applying NEXUS TPU overlay..."

rsync -a --delete \
    "${ROOT}/patches/tpu-inference/" \
    "${BUILD}/tpu-inference/"

# ---------------------------------------------------------------------------
# Install build tooling
# ---------------------------------------------------------------------------

echo
echo "Installing build dependencies..."

python -m pip install \
    --no-build-isolation \
    -q \
    --upgrade \
    setuptools \
    setuptools-rust \
    setuptools-scm \
    wheel \
    ninja \
    cmake

# ---------------------------------------------------------------------------
# Install NEXUS
# ---------------------------------------------------------------------------

echo
echo "Installing NEXUS..."

python -m pip install \
    --no-build-isolation \
    -e "${ROOT}[dev]"

# ---------------------------------------------------------------------------
# Install vLLM
# ---------------------------------------------------------------------------
#
# --no-build-isolation is REQUIRED here.
#
# vLLM's build process imports the active PyTorch installation and inspects
# the target device. Pip build isolation previously created a separate
# environment containing a CUDA-dependent PyTorch, causing:
#
#   AssertionError: CUDA_HOME is not set
#
# The active Kaggle environment uses CPU PyTorch + TPU/JAX, so we deliberately
# build against the current environment instead.
#
# setuptools-rust and setuptools-scm were installed above because vLLM's
# setup/build metadata imports both during editable installation.

echo
echo "Installing vLLM for TPU..."

VLLM_TARGET_DEVICE=tpu \
python -m pip install \
    --no-build-isolation \
    -e "${BUILD}/vllm"

# ---------------------------------------------------------------------------
# Install TPU inference
# ---------------------------------------------------------------------------

echo
echo "Installing tpu-inference..."

python -m pip install \
    --no-build-isolation \
    -e "${BUILD}/tpu-inference"

# ---------------------------------------------------------------------------
# Final verification
# ---------------------------------------------------------------------------

echo
echo "========================================"
echo "NEXUS upstream install complete"
echo "========================================"
echo "VLLM_TARGET_DEVICE: ${VLLM_TARGET_DEVICE}"
echo "vLLM:               ${VLLM_COMMIT}"
echo "tpu-inference:      ${TPU_COMMIT}"

python - <<'PY'
import os

print()
print("Build environment:")
print("  VLLM_TARGET_DEVICE =", os.environ.get("VLLM_TARGET_DEVICE"))

try:
    import torch
    print("  PyTorch             =", torch.__version__)
    print("  CUDA available      =", torch.cuda.is_available())
except Exception as exc:
    print("  PyTorch check failed:", exc)

try:
    import jax
    print("  JAX                 =", jax.__version__)
    print("  JAX devices         =", jax.devices())
except Exception as exc:
    print("  JAX check failed    =", exc)
PY
