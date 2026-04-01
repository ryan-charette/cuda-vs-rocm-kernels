#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-$ROOT_DIR/build}"

if command -v nvidia-smi >/dev/null 2>&1; then
  CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')"
  if [[ -n "${CC}" ]]; then
    export CUDAARCHS="${CC}"
    echo "Detected CUDAARCHS=${CUDAARCHS}"
  fi
fi

if command -v nvcc >/dev/null 2>&1; then
  cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DPGKL_ENABLE_CUDA=ON -DPGKL_ENABLE_HIP=OFF
else
  cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DPGKL_ENABLE_CUDA=OFF -DPGKL_ENABLE_HIP=OFF
fi

cmake --build "$BUILD_DIR" -j
