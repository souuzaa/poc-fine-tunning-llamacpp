#!/usr/bin/env bash
# Stage 4 -- serve a GGUF with full GPU offload.
#
# Only ONE model at a time: two Q4_K_M 8B models exceed the ~9.5GB free VRAM (ADR 0010).
set -euo pipefail

MODEL="${1:?usage: 04_serve.sh <path-to-gguf> [port]}"
PORT="${2:-8080}"
LLAMA_DIR="${LLAMA_DIR:-vendor/llama.cpp}"

[ -f "$MODEL" ] || { echo "FATAL: $MODEL not found" >&2; exit 1; }

exec "$LLAMA_DIR/build/bin/llama-server" \
  -m "$MODEL" \
  -ngl 99 \
  -c 4096 \
  -fa on \
  --host 127.0.0.1 \
  --port "$PORT"
