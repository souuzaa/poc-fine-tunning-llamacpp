# ADR 0008 — Explicit merge, convert and quantise for GGUF export

- **Status:** Accepted
- **Date:** 2026-09-12

## Context

QLoRA training produces a LoRA adapter over 4-bit base weights (ADR 0002). llama.cpp
needs a single GGUF file. Unsloth offers a one-call convenience path,
`model.save_pretrained_gguf(...)`, which performs merge, convert and quantise internally.

That call is a frequent source of failures: it shells out to a llama.cpp checkout it may
build itself, and when it breaks the error surfaces several layers from the actual cause.

## Decision

Three explicit, independently re-runnable steps:

1. `model.save_pretrained_merged("outputs/merged-16bit", tokenizer, save_method="merged_16bit")`
2. `python vendor/llama.cpp/convert_hf_to_gguf.py outputs/merged-16bit --outfile outputs/gguf/personas-f16.gguf --outtype f16`
3. `llama-quantize personas-f16.gguf personas-q4_k_m.gguf Q4_K_M`

The **base model is converted through the identical three steps** (with an untrained
adapter, i.e. a straight merge of nothing) so that base-versus-tuned comparison differs
only by the LoRA weights — not by quantisation lineage or converter version.

## Consequences

- Each step fails in isolation with a legible error, and can be re-run without repeating
  the others.
- Disk cost ~37GB per model lineage (16GB merged + 16GB f16 GGUF + 5GB Q4_K_M). Against
  304GB free this is comfortable; intermediates are deleted after a verified quantise.
- RAM peak ~17GB during the merge, against 30GB total. Safe, but the merge must not run
  concurrently with anything large.
- llama.cpp is vendored at a **pinned commit**, so a converter change cannot silently
  alter results between runs.
- Q8_0 is also produced as a quality reference, to separate "quantisation hurt it" from
  "the fine-tune hurt it" if results disappoint.

## Alternatives considered

- **`save_pretrained_gguf`.** One line, but opaque when it fails, and it obscures which
  llama.cpp version did the conversion.
- **Downloading a pre-built base GGUF from the Hub.** Convenient, but it would be
  quantised by a different tool version than our fine-tune, making the A/B comparison
  measure two variables instead of one.
- **Serving the adapter unmerged.** llama.cpp supports GGUF LoRA adapters, but the merged
  path is what a deployment would ship, and it is what the PoC should validate.
