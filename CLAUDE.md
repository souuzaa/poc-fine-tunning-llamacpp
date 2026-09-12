# poc-fine-tunning-llamacpp

Fine-tune **Qwen3-8B** with **Unsloth QLoRA** on NVIDIA's synthetic Brazilian persona
corpus, export to **GGUF**, serve and evaluate with **llama.cpp** — all on a single
RTX 3060 (12GB).

**Task:** demographic attributes in, six-section Brazilian-Portuguese persona narrative out.

## Read first

- [`docs/superpowers/specs/2026-09-12-persona-finetune-design.md`](docs/superpowers/specs/2026-09-12-persona-finetune-design.md) — the design
- [`docs/adr/README.md`](docs/adr/README.md) — why each choice was made

## Hardware and environment

| | |
|---|---|
| GPU | RTX 3060 **12GB**, ~9.5GB free (GNOME holds ~2.7GB) |
| RAM / disk | 30GB / 304GB free |
| Driver / CUDA | 595.84 / toolkit 13.3, Ampere `sm_86` |
| Python | host is **3.14 (unusable for torch)** — always use the project `uv` venv on 3.12 |

Never `pip install` into the host Python. Use `uv run` / `uv sync`; dependencies are
pinned in the committed `uv.lock` (ADR 0011).

## Pipeline

```
make setup → make data → make train → make export → make serve → make eval → make report
```

`make smoke` runs the whole chain on a 200-row slice in minutes. Run it before any
multi-hour training run.

## Invariants — breaking these silently invalidates results

1. **One prompt renderer.** Every prompt comes from `src/personas/prompt.py`. Never build a
   prompt string anywhere else. `tests/test_prompt_parity.py` enforces this (ADR 0006).
2. **Thinking mode off everywhere.** Always `apply_chat_template(..., enable_thinking=False)`.
   Any `<think>` in a generation is a bug and is scored as a format failure.
3. **Raw `/completion` only.** Never evaluate through `/v1/chat/completions` — llama-server
   re-applies its own Jinja template and breaks train/inference parity (ADR 0006, 0007).
4. **Base and tuned never run concurrently.** Two Q4_K_M 8B models do not fit in 9.5GB.
   Eval is sequential; the Makefile owns server lifecycle (ADR 0010).
5. **Base and tuned share a quantisation lineage.** Both go through the same merge →
   convert → quantise steps so the A/B measures the LoRA and nothing else (ADR 0008).
6. **Changing the system prompt invalidates the adapter.** Bump `PROMPT_VERSION` in
   `prompt.py`; eval checks it against the dataset manifest.
7. **LangChain lives only in stages 5-6.** It must never become a training dependency —
   its release cadence cannot be allowed to break a 4-hour GPU run (ADR 0007).

## VRAM

Training budget is ~7.7-8.3GB of ~9.5GB free. If it OOMs, in order: batch 1 with
grad-accum 16 → `max_seq_length` 1536 → stop the GNOME session to reclaim 2.7GB.

## Data note

The dataset is CC-BY-4.0 and **fully synthetic** — no real PII, despite being "person
data". Generated samples can be shared freely.
