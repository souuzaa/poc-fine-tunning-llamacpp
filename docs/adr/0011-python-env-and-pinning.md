# ADR 0011 — Isolated uv environment on Python 3.12, fully pinned

- **Status:** Accepted
- **Date:** 2026-09-12

## Context

The host runs **Python 3.14.4**, for which no torch, Unsloth or bitsandbytes wheel exists.
Installing into it is not an option.

Separately, the Unsloth / torch / transformers / trl / peft stack moves fast and its
components are tightly coupled. An unpinned environment that works today can fail to
resolve next week — and the failure typically appears *after* the model has downloaded,
minutes into what was meant to be a 4-hour run.

The host CUDA toolkit is 13.3 with driver 595.84; the GPU is Ampere `sm_86`.

## Decision

A project-local **`uv` virtual environment on Python 3.12**, with every dependency pinned
in `uv.lock`, committed to the repo. The system Python is not modified.

torch comes from the **cu128** wheel index — `sm_86` is fully supported, and the 595
driver is backward compatible with the CUDA 12.8 runtime, so the host's 13.3 toolkit is
irrelevant to the Python side.

llama.cpp is vendored under `vendor/llama.cpp` at a pinned commit and built with
`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86` against the host toolkit.

## Consequences

- The environment is reproducible: `uv sync` reconstructs it exactly.
- Dependency breakage is a deliberate act (re-locking), never a surprise mid-run.
- `CMAKE_CUDA_ARCHITECTURES=86` targets only this GPU, cutting llama.cpp build time
  substantially versus compiling every architecture.
- Two separate toolchains to set up (uv for Python, cmake for llama.cpp). `make setup`
  wraps both and verifies CUDA is visible from torch before declaring success.
- Upgrading Unsloth later is an explicit, testable change with a diff to review.

## Alternatives considered

- **conda/mamba.** Works, but heavier, slower to resolve, and `uv` handles the PyTorch
  index URLs cleanly.
- **Docker.** Best isolation, but adds GPU passthrough setup and an image rebuild to every
  iteration — friction the PoC does not need on a single-user workstation.
- **Unpinned `pip install unsloth`.** The documented quickstart, and the most common cause
  of "it worked yesterday".
