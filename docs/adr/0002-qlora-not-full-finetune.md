# ADR 0002 — 4-bit QLoRA, not a full fine-tune

- **Status:** Accepted
- **Date:** 2026-09-12

## Context

The stated goal was to train a ~7B model "full GPU" on an RTX 3060 (12GB, ~9.5GB free
with the GNOME desktop resident).

A full-parameter fine-tune of an 8B model in bf16 needs roughly:

| Component | Memory |
|---|---|
| Weights (bf16) | ~16 GB |
| Gradients (bf16) | ~16 GB |
| Adam optimizer states (fp32 m + v) | ~65 GB |
| Activations | several GB |
| **Total** | **~100 GB** |

That is an 8xA100 job. It is not a tuning problem, it is an arithmetic one.

## Decision

Fine-tune with **QLoRA**: base weights frozen in NF4 4-bit, trainable LoRA adapters in
bf16, 8-bit Adam. Everything resident on the GPU — no CPU offload, no gradient offload,
no DeepSpeed ZeRO. "Full GPU" is honoured in the sense that matters: the whole training
step runs on the 3060.

Budget at seq 2048, batch 2, `use_gradient_checkpointing="unsloth"`:

| Component | Memory |
|---|---|
| NF4 base weights (8.2B) | ~5.4 GB |
| LoRA params + 8-bit Adam states | ~0.5 GB |
| Activations | ~1.2-1.8 GB |
| CUDA/cuBLAS context | ~0.6 GB |
| **Total** | **~7.7-8.3 GB** |

## Consequences

- Fits in ~9.5GB free with modest headroom. Documented fallbacks if OOM: batch 1 +
  grad-accum 16 (~-0.7GB), seq 1536, or stopping the GNOME session to reclaim 2.7GB.
- Only adapter weights are trained, so a training run produces a ~160MB adapter rather
  than a 16GB checkpoint — cheap to keep many experiments.
- Requires a merge step before GGUF export (see ADR 0008).
- Some quality is left on the table versus a full fine-tune. Irrelevant at PoC scale:
  the task is format- and register-learning, which LoRA handles well.

## Alternatives considered

- **Full fine-tune with CPU offload.** Technically bootable, ~50-100x slower. A 4-hour
  run becomes weeks.
- **LoRA on bf16 base (no quantisation).** 16GB of weights alone — does not fit.
- **A smaller model (1.5B-3B) full fine-tuned.** Would fit, but abandons the "7B-class
  model" requirement that motivated the PoC.
