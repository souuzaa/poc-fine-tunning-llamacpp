# ADR 0012 — Revised training budget: 10k examples, one epoch

- **Status:** Accepted
- **Date:** 2026-09-12
- **Supersedes:** [ADR 0009](0009-training-budget.md)

## Context

ADR 0009 chose 20,000 examples on an estimated throughput of 1000-1400 tokens/sec,
projecting a 3-5 hour run. That estimate was never measured.

A 160-row smoke run on the actual hardware measured **39.2 s/step**, steady state, at
effective batch 16 — converging down from 43.4s as kernels finished compiling, so this is
the real rate and not warmup. That is roughly 570 tokens/sec: the original estimate was
optimistic by about 2.4x.

At the measured rate the original budget reprices badly:

| Rows | Steps | Measured wall clock |
|---|---|---|
| 5,000 | 312 | ~3.4h |
| 7,000 | 437 | ~4.8h |
| **10,000** | **625** | **~6.8h** |
| 20,000 | 1250 | ~13.6h |

## Decision

**10,000 training examples, 1 epoch** — 625 optimizer steps, ~6.8 hours. Validation and
test splits are unchanged at 500 and 200 rows.

## Consequences

- The run fits in a single overnight window rather than spanning a night and most of the
  following day. A bad hyperparameter costs one night, not a full day.
- Still 60x the smoke run, and well past the point where the model learns output format,
  pt-BR register and attribute grounding — the three things evaluation measures.
- The 1M-row corpus remains barely touched, so a longer run later needs only a config
  change and `make data`.
- ADR 0009's reasoning about preferring one epoch over unique examples to several over
  fewer still holds; only the row count changes.

## Related finding: VRAM headroom is thinner than ADR 0002 projected

The smoke run peaked at **9.13 GB** reserved by torch — above the 7.7-8.3GB projection,
because evaluation runs alongside the training allocation. With ~2.1GB held by the GNOME
desktop that is ~11.2GB of 11.6GB total. It completes, but there is almost no margin:
opening a browser mid-run can OOM it. Run with the desktop light, or from a TTY.

`prediction_loss_only` and `per_device_eval_batch_size=1` are what brought evaluation
inside the budget at all; before those, eval OOMed a GPU that had just trained cleanly.

## Alternatives considered

- **Keep 20,000 (~13.6h).** Best quality, rejected on iteration cost for a PoC.
- **7,000 (~4.8h).** Honours the original 3-5h intent, but gives up quality for an hour
  that a single overnight run does not actually need.
- **20,000 at `max_seq_length` 1536 (~10-11h).** Buys ~20% speed and ~1GB of VRAM back by
  truncating roughly 1% of examples (measured p99 = 1424 tokens). Worth revisiting if the
  VRAM margin proves too thin in practice.
