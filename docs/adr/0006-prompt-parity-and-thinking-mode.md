# ADR 0006 — One shared prompt renderer; thinking mode disabled everywhere

- **Status:** Accepted
- **Date:** 2026-09-12

## Context

The most common way a fine-tune "mysteriously fails" is that the prompt at inference time
differs from the prompt at training time — a stray space, a different system message, a
chat template applied twice. The model is fine; the harness is lying.

Qwen3 makes this sharper than usual. It ships a **hybrid thinking mode**, and its chat
template branches on `enable_thinking`. With thinking enabled the assistant turn begins
with a `<think>` block; with it disabled the template inserts an empty
`<think>\n\n</think>\n\n` preamble. Those are three different token sequences for what
looks like "the same prompt".

Compounding this, `llama-server`'s `/v1/chat/completions` endpoint **re-applies its own
Jinja template** to the messages it receives. A client that sends structured messages has
no direct control over the final token sequence.

## Decision

**1. A single renderer.** `src/personas/prompt.py` exposes `render_prompt(attrs)` and
`render_target(row)`. Data prep (stage 1), evaluation (stage 5) and the report (stage 6)
all import them. There is no second place where a prompt string is built.

**2. Thinking mode off at every boundary.** All rendering goes through
`tokenizer.apply_chat_template(..., enable_thinking=False)`, and training targets contain
exactly what that produces — including the empty think preamble if the pinned template
emits one. We train on the literal string; we never hand-craft it.

**3. Raw `/completion`, not `/v1/chat/completions`.** Inference posts the pre-rendered
string to llama-server's raw completion endpoint, bypassing server-side templating
entirely. See ADR 0007 for how LangChain wraps this.

**4. A test enforces it.** `tests/test_prompt_parity.py` asserts that the prompt written
into `train.jsonl` is byte-for-byte identical to the prompt the eval harness sends for
the same row. Drift becomes a failing test, not a bad eval run.

## Consequences

- Prompt drift is structurally prevented rather than watched for.
- `<think>` leakage becomes a measurable eval metric (any occurrence fails format
  validity) instead of an invisible quality tax.
- We give up llama-server's built-in chat UI for evaluation, since it templates its own
  prompts. Acceptable: the UI stays available for manual poking, it is just not the
  measurement path.
- Changing the system prompt invalidates the trained adapter. `prompt.py` carries a
  `PROMPT_VERSION` constant that is written into the dataset manifest and checked at
  eval time.

## Alternatives considered

- **`/v1/chat/completions` with `--jinja` and `chat_template_kwargs`.** More idiomatic and
  more portable, but parity depends on the server's template matching the training-time
  template, which must be verified by diffing logged prompts — and re-verified on every
  llama.cpp upgrade. Rejected for a project whose whole point is trusting the measurement.
- **Hand-writing the ChatML string.** Removes the tokenizer as source of truth; silently
  breaks on a template update.
