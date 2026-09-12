# ADR 0007 — LangChain scoped to the evaluation and serving layer

- **Status:** Accepted
- **Date:** 2026-09-12

## Context

The question was raised whether LangChain should orchestrate this pipeline. The pipeline
has two halves with very different shapes:

- **Stages 1-3 (data prep, QLoRA training, GGUF export)** are a batch ETL-and-compute
  chain: HuggingFace `datasets` to `SFTTrainer` to `convert_hf_to_gguf.py`. LangChain has
  no abstraction for any step.
- **Stages 5-6 (evaluation, reporting)** are exactly what LangChain is for: fan out many
  prompts against an LLM endpoint, with retries, bounded concurrency and swappable
  backends.

## Decision

LangChain is used **only in stages 5-6**, and it wraps llama-server's **raw `/completion`**
endpoint through a thin custom `Runnable` (`src/personas/llm.py`), not `ChatOpenAI`.

Everything upstream is orchestrated by `make` plus plain Python scripts.

## Consequences

- We get `.batch()` with bounded concurrency over the held-out set, a declarative retry
  and timeout policy, and base-versus-tuned A/B as a single swapped `base_url` — without
  hand-rolling an asyncio pool.
- The custom `Runnable` posts our exact pre-rendered string, so ADR 0006's parity
  guarantee survives contact with the framework. Using `ChatOpenAI` would have handed
  prompt construction to the server.
- ~30 lines of adapter code we own, instead of a well-tested library integration. This is
  a deliberate trade of convention for control.
- No LangChain dependency in the training environment, so LangChain's release cadence
  cannot break a 4-hour GPU run.
- A LangChain demo app over the fine-tuned model is a natural stage 7, deliberately left
  out of scope for now.

## Alternatives considered

- **LangChain over `/v1/chat/completions`.** Idiomatic and portable, but reintroduces
  server-side templating — see ADR 0006.
- **LangChain orchestrating the whole pipeline.** Ceremony with no payoff; there is no
  chain-shaped work in stages 1-3.
- **No LangChain (httpx + asyncio).** ~40 lines, zero dependency, and entirely viable. The
  PoC includes LangChain because exercising it against a local fine-tuned model is one of
  the things the PoC is meant to demonstrate.
