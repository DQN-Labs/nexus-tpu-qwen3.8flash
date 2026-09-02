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

# Use the same Python installation that is running the Kaggle environment.
PYTHON="${PYTHON:-$(command -v python)}"

VLLM_COMMIT="2a4cd640ff1a61b66124ddbaaf02a73781f7295a"
TPU_COMMIT="c5c8a055edfa7853fe1cb9e8873c027d931ab490"

mkdir -p "${BUILD}"

echo "========================================"
echo "NEXUS upstream installation"
echo "========================================"
echo "ROOT:                ${ROOT}"
echo "BUILD:               ${BUILD}"
echo "PYTHON:              ${PYTHON}"
echo "VLLM_TARGET_DEVICE:  ${VLLM_TARGET_DEVICE}"
echo "vLLM commit:         ${VLLM_COMMIT}"
echo "TPU commit:          ${TPU_COMMIT}"
echo

# ---------------------------------------------------------------------------
# Clone / reuse pinned upstream repositories
# ---------------------------------------------------------------------------

clone_or_checkout() {
    local url="$1"
    local dir="$2"
    local commit="$3"

    # If something exists at this path but isn't a Git repository,
    # remove it so we can create a clean checkout.
    if [[ -e "${dir}" && ! -d "${dir}/.git" ]]; then
        echo "Removing invalid checkout: ${dir}"
        rm -rf "${dir}"
    fi

    if [[ ! -d "${dir}/.git" ]]; then
        echo "Cloning ${url}..."
        git clone "${url}" "${dir}"
    else
        echo "Using existing repository: ${dir}"
    fi

    echo "Fetching ${commit}..."

    git -C "${dir}" fetch origin "${commit}"

    # Make the checkout deterministic. Any modifications from a previous
    # failed install are removed before applying the NEXUS overlay.
    git -C "${dir}" reset --hard "${commit}"
    git -C "${dir}" clean -fd

    echo "Checked out ${commit}"
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

# IMPORTANT: DO NOT use --delete here.
#
# The upstream tpu-inference repository contains packaging files
# (pyproject.toml, setup metadata, etc.) that are not present in the
# NEXUS patch directory. --delete would destroy those files and make
# the upstream checkout impossible to install.
rsync -a \
    "${ROOT}/patches/tpu-inference/" \
    "${BUILD}/tpu-inference/"

# ---------------------------------------------------------------------------
# Install build tooling
# ---------------------------------------------------------------------------

echo
echo "Installing build dependencies..."

"${PYTHON}" -m pip install \
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

"${PYTHON}" -m pip install \
    --no-build-isolation \
    -e "${ROOT}[dev]"

# ---------------------------------------------------------------------------
# Install vLLM
# ---------------------------------------------------------------------------

echo
echo "Installing vLLM for TPU..."

VLLM_TARGET_DEVICE=tpu \
"${PYTHON}" -m pip install \
    --no-build-isolation \
    -e "${BUILD}/vllm"

# ---------------------------------------------------------------------------
# Install TPU inference
# ---------------------------------------------------------------------------

echo
echo "Installing tpu-inference..."

"${PYTHON}" -m pip install \
    --no-build-isolation \
    -e "${BUILD}/tpu-inference"

# ---------------------------------------------------------------------------
# Final verification
# ---------------------------------------------------------------------------

echo
echo "========================================"
echo "Verifying installation"
echo "========================================"

"${PYTHON}" - <<'PY'
import os

print()
print("Build environment:")
print("  Python              =", __import__("sys").executable)
print("  VLLM_TARGET_DEVICE  =", os.environ.get("VLLM_TARGET_DEVICE"))

try:
    import torch

    print("  PyTorch             =", torch.__version__)
    print("  CUDA available      =", torch.cuda.is_available())

except Exception as exc:
    print("  PyTorch check failed:", exc)
    raise

try:
    import jax

    print("  JAX                 =", jax.__version__)
    print("  JAX devices         =", jax.devices())

except Exception as exc:
    print("  JAX check failed    =", exc)
    raise

print()
print("Package imports:")

import nexus
import vllm
import tpu_inference

print("  NEXUS               =", nexus.__file__)
print("  vLLM                =", vllm.__file__)
print("  tpu-inference       =", tpu_inference.__file__)

print()
print("ALL IMPORTS OK")
PY

echo
echo "========================================"
echo "NEXUS upstream install complete"
echo "========================================"
echo "Python:              ${PYTHON}"
echo "VLLM_TARGET_DEVICE:  ${VLLM_TARGET_DEVICE}"
echo "vLLM:                ${VLLM_COMMIT}"
echo "tpu-inference:       ${TPU_COMMIT}"
