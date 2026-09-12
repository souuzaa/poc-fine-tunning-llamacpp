#!/usr/bin/env bash
# Stage 0 -- vendor and build llama.cpp with CUDA for this machine only.
#
# Pinned to a specific commit (ADR 0011): a converter change must never silently alter
# results between runs. CMAKE_CUDA_ARCHITECTURES=86 targets the RTX 3060 alone, which
# cuts build time substantially versus compiling every architecture.
set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-vendor/llama.cpp}"
LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}"
CUDA_ARCH="${CUDA_ARCH:-86}"

# cmake is installed into the project venv (dev extra) so no system package is needed.
if [ -z "${CMAKE:-}" ]; then
  if [ -x ".venv/bin/cmake" ]; then
    CMAKE="$(pwd)/.venv/bin/cmake"
  elif command -v cmake >/dev/null 2>&1; then
    CMAKE="$(command -v cmake)"
  else
    echo "FATAL: cmake not found. Run: uv sync --extra dev" >&2
    exit 1
  fi
fi
echo ">> using cmake at $CMAKE ($("$CMAKE" --version | head -1))"

if [ ! -d "$LLAMA_DIR/.git" ]; then
  echo ">> cloning llama.cpp into $LLAMA_DIR"
  mkdir -p "$(dirname "$LLAMA_DIR")"
  if [ -n "${LLAMA_COMMIT:-}" ]; then
    git init -q "$LLAMA_DIR"
    git -C "$LLAMA_DIR" remote add origin "$LLAMA_REPO"
    git -C "$LLAMA_DIR" fetch --depth 1 origin "$LLAMA_COMMIT"
    git -C "$LLAMA_DIR" checkout -q FETCH_HEAD
  else
    git clone --depth 1 "$LLAMA_REPO" "$LLAMA_DIR"
  fi
elif [ -n "${LLAMA_COMMIT:-}" ]; then
  echo ">> checking out pinned commit $LLAMA_COMMIT"
  git -C "$LLAMA_DIR" fetch --depth 1 origin "$LLAMA_COMMIT"
  git -C "$LLAMA_DIR" checkout -q FETCH_HEAD
fi

HEAD_SHA="$(git -C "$LLAMA_DIR" rev-parse HEAD)"
echo ">> llama.cpp at $HEAD_SHA"
if [ -z "${LLAMA_COMMIT:-}" ]; then
  echo ">> LLAMA_COMMIT not set. PIN IT for reproducible builds:"
  echo ">>   export LLAMA_COMMIT=$HEAD_SHA"
fi

echo ">> configuring (CUDA, sm_$CUDA_ARCH)"
"$CMAKE" -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF

echo ">> building"
"$CMAKE" --build "$LLAMA_DIR/build" --config Release -j "$(nproc)" \
  --target llama-server llama-quantize llama-perplexity llama-cli

for binary in llama-server llama-quantize llama-perplexity; do
  path="$LLAMA_DIR/build/bin/$binary"
  [ -x "$path" ] || { echo "FATAL: $path missing after build" >&2; exit 1; }
  echo ">> ok: $path"
done

[ -f "$LLAMA_DIR/convert_hf_to_gguf.py" ] || {
  echo "FATAL: convert_hf_to_gguf.py missing" >&2; exit 1; }

echo ">> llama.cpp ready"
