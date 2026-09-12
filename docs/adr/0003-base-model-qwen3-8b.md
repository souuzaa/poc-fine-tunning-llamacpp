# ADR 0003 — Base model: Qwen3-8B

- **Status:** Accepted
- **Date:** 2026-09-12

## Context

The target task is generating rich Brazilian-Portuguese persona narratives. The base
model must (a) already speak natural pt-BR, so the fine-tune spends its capacity on the
task rather than on the language, (b) fit 4-bit QLoRA on 12GB, (c) be supported by both
Unsloth and llama.cpp's `convert_hf_to_gguf.py`, and (d) have a permissive licence.

Four candidates were evaluated and presented:

| Model | pt-BR | Licence | Notes |
|---|---|---|---|
| Qwen2.5-7B-Instruct | Strong | Apache-2.0 | Safest, smallest, ungated |
| **Qwen3-8B** | **Excellent** | **Apache-2.0** | **Newer, ~1.5GB more VRAM, hybrid thinking mode** |
| Llama-3.1-8B-Instruct | Strong | Llama Community | Gated repo, needs token + licence acceptance |
| Mistral-7B-Instruct-v0.3 | Weak | Apache-2.0 | Would waste capacity learning pt-BR |

## Decision

**Qwen3-8B**, Apache-2.0, loaded 4-bit via Unsloth.

## Consequences

- Best pt-BR quality of the candidates, and no gated download or licence acceptance.
- ~1.5GB more VRAM than a true 7B. Still inside budget (ADR 0002), but it is the reason
  batch size is 2 rather than 4.
- **Hybrid thinking mode must be explicitly disabled** at every point where a prompt is
  rendered, or the model emits `<think>` blocks that were never in the training targets.
  This is the single highest-risk detail in the project — see ADR 0006.
- 8B (not 7B) — the PoC description should say "8B" to stay accurate.

## Alternatives considered

See table above. Qwen2.5-7B-Instruct remains the documented fallback if Qwen3 support in
the pinned Unsloth version proves unstable: the task formulation, data pipeline and eval
harness are model-agnostic, so switching is a config change plus a retrain.
