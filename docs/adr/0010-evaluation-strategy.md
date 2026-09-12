# ADR 0010 — Evaluation: automatic metrics plus side-by-side comparison

- **Status:** Accepted
- **Date:** 2026-09-12

## Context

The PoC must answer one question defensibly: **did fine-tuning help?** Manual poking at a
chat UI cannot answer it. An LLM judge can, but needs an API key and turns a local,
offline PoC into one with an external dependency.

Because the task has real reference text (ADR 0004), cheaper objective metrics are
available here than for most generation tasks.

## Decision

Four automatic metrics plus a human-readable comparison, all local, no API keys.

**1. Held-out perplexity.** `llama-perplexity` over the 200-row test split, base versus
tuned, on the same Q4_K_M quantisation (ADR 0008). The headline number.

**2. Format validity** (% of generations): all six sections present, in the required
order, no extra sections, and **no `<think>` leakage** (ADR 0006).

**3. Grounding recall** (per field): does the output actually mention the given name,
municipality, state, occupation keywords and age? This is the check that the model is
conditioning on its input rather than producing fluent generic prose.

**4. Language and length.** pt-BR heuristic via stopword ratio, and generated length
distribution against reference length distribution.

**5. Side-by-side report.** 30 test attribute sets rendered as three columns — ground
truth, base, tuned — as an HTML artifact for direct reading.

Decoding parameters are **identical** for base and tuned (temp 0.7, top_p 0.8, top_k 20,
repeat_penalty 1.05, `n_predict` 1024, fixed seed), so the only variable is the weights.

## Consequences

- Every number is reproducible offline on this machine.
- Grounding recall is the metric most likely to move dramatically, and the most
  convincing evidence for a non-specialist reader.
- Base and tuned servers **cannot run concurrently**: two Q4_K_M 8B models at ~5GB each
  plus KV cache exceed the ~9.5GB free. Eval therefore runs sequentially — base pass,
  teardown, tuned pass — with the Makefile owning server lifecycle. This is a hard
  constraint, not a preference.
- Heuristic metrics can be gamed by a degenerate model (e.g. one that parrots input
  attributes scores perfectly on grounding). The side-by-side report is the guard against
  trusting a number that is technically true and substantively wrong.

## Alternatives considered

- **LLM-as-judge.** More sensitive to quality differences the heuristics miss, at the cost
  of an API key and per-run spend. Deliberately left as an easy add-on: the eval harness
  already produces the paired generations a judge would score.
- **Manual testing only.** No defensible answer to the central question.
- **BLEU/ROUGE against references.** Nearly meaningless for open-ended generation, where
  many good outputs share few n-grams with the reference.
