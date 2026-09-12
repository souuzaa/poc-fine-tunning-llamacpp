# ADR 0009 — Training budget: 20k examples, one epoch

- **Status:** Accepted
- **Date:** 2026-09-12

## Context

The source dataset holds 1M rows. On an RTX 3060 the training set size is purely a
time-versus-quality dial. Estimated throughput for Qwen3-8B QLoRA at seq 2048 is roughly
1000-1400 tokens/sec.

| Budget | Tokens | Wall clock | Outcome |
|---|---|---|---|
| 5k, 1 epoch | ~4.5M | ~1h | Format learned, content generic |
| **20k, 1 epoch** | **~18M** | **~3.5-5h** | **Format, register and grounding all learned** |
| 50k+, 1-2 epochs | ~45M+ | ~12-24h | Best achievable here; a bad hyperparameter costs a day |

## Decision

**20,000 training examples, 1 epoch**, with 500 validation and 200 test rows held out and
disjoint by `uuid`.

At batch 2 x grad-accum 8 (effective batch 16) this is ~1250 optimizer steps.

## Consequences

- A run fits in a workday or overnight, so a mistake costs hours rather than a day.
- One epoch over unique examples, rather than several over fewer, is the better use of a
  1M-row corpus: it minimises memorisation and maximises coverage of occupations, regions
  and education levels.
- The 200-row test split is never seen during training, so eval (ADR 0010) is honest.
- Training reports a measured ETA after the first 50 steps, so a bad throughput estimate
  is caught in minutes rather than discovered at hour four.
- Checkpoints every 250 steps with `save_total_limit=3` make the run resumable.

## Alternatives considered

- **5k first as a smoke test, then 20k.** Genuinely prudent, and rejected only because the
  stage scripts are independently runnable — stages 1-6 can be validated end to end on a
  200-row slice in minutes via `make smoke`, which gets the same de-risking for less time.
- **50k+.** Worth doing once the pipeline is proven, not before.
