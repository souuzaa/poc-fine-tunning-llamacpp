# Results — Qwen3-8B QLoRA persona fine-tune

**Date:** 2026-09-13 · **Hardware:** single RTX 3060 12GB · **Config:** `configs/qwen3-8b-personas.yaml`

## Training

| | |
|---|---|
| Rows / steps | 10,000 / 625 (exactly one epoch, ADR 0012) |
| Runtime | 7.42h at 41.6 s/step |
| Peak VRAM | 9.07 GB of 11.63 GB |
| Trainable params | 87,293,952 of 8,278,029,312 (1.05%) |
| Train loss | 1.719 → 0.9073 |

Eval loss fell monotonically at every checkpoint, with train and eval tracking
together — convergence without overfitting:

| Step | 200 | 300 | 400 | 500 | 600 | 625 |
|---|---|---|---|---|---|---|
| Eval loss | 0.9454 | 0.8900 | 0.8603 | 0.8417 | 0.8308 | **0.8267** |

## Process metrics

Every stage, measured end to end on one RTX 3060 12GB.

| Stage | Wall clock | Notes |
|---|---|---|
| Environment + llama.cpp CUDA build | ~25 min | download-bound; `sm_86` only |
| Data preparation | ~3 min | streamed 10,708 rows, 0.1% dropped |
| **Fine-tuning** | **7h 25m** | 625 steps at 42.7 s/step |
| GGUF export (per model) | **6m 26s** | merge → f16 GGUF → Q4_K_M + Q8_0 |
| Evaluation | **46m 19s** | 200 rows × 2 models, sequential |
| Report | < 1s | |
| **Total** | **≈ 9h** | |

Evaluation breakdown:

| | Base | Fine-tuned |
|---|---|---|
| 200 generations | 1050 s (5.3 s/row) | 1610 s (8.1 s/row) |
| Mean output | 2093 chars | 3072 chars |

The tuned model is ~50% slower per row because it produces ~50% more text — it fills all six
sections rather than stopping early. Single-stream generation is ~58 tok/s at `-ngl 99`.

### Estimate vs measurement

| | Estimated | Measured | |
|---|---|---|---|
| Throughput | 1000-1400 tok/s | ~570 tok/s | 2.4× optimistic |
| Training (20k rows) | 3-5h | ~13.6h | budget cut to 10k ([ADR 0012](adr/0012-revised-training-budget.md)) |
| Peak VRAM | 7.7-8.3 GB | 9.07 GB | evaluation runs alongside the training allocation |

The throughput estimate in ADR 0009 was never measured; a 160-row smoke run disproved it in
7 minutes, and the budget was re-derived before committing a night of GPU time.

## Base vs fine-tuned

200 held-out rows, Q4_K_M through the same quantisation lineage, identical decoding
(temp 0.7, top_p 0.8, top_k 20, repeat_penalty 1.05, seed 3407), served sequentially by
llama.cpp.

| Metric | Base | Fine-tuned | |
|---|---|---|---|
| **Format validity** | 0.000 | **1.000** | 200/200 rows, six sections in order |
| **Perplexity** (held-out) | 4.7283 | **2.7733** | −41% |
| Think leak rate | 0.000 | 0.000 | ADR 0006 held |
| Portuguese rate | 1.000 | 1.000 | |
| Mean chars | 2093 | 3072 | reference: 3491 |
| Grounding recall | 1.000 | 0.903 | see below |

**Format validity is the headline.** The base model never once produced the required
structure; the fine-tune produced it on every single row.

### The grounding number is not a regression

Grounding recall reads 1.000 → 0.903, which looks like the fine-tune got worse at using
its input. Scoring the **ground truth** with the same function settles it:

| Field | Base | Fine-tuned | Ground truth |
|---|---|---|---|
| name | 1.000 | 0.995 | 1.000 |
| municipality | 1.000 | 0.995 | 1.000 |
| state | 1.000 | 0.890 | 0.815 |
| occupation | 1.000 | 0.890 | 0.870 |
| age | 1.000 | 0.745 | 0.650 |
| **recall** | **1.000** | **0.903** | **0.867** |

The human-written reference scores 0.867. The fine-tuned model scores **higher** than the
reference on state, occupation and age, and matches it on name and municipality. The base
model's perfect 1.000 comes from echoing the attribute block back verbatim in an
"Identificação Pessoal" section — a degenerate strategy that scores perfectly while
producing worse text.

This is exactly the saturated metric ADR 0010 anticipated, caught by comparing against the
reference rather than trusting the number. **Interpretation: grounding is a pass/fail
sanity check, not a ranking metric.** A future run should score it against the reference
as a baseline rather than against 1.0.

## Qualitative

Same input, both models:

> `José Pereira | Masculino | 49 | Diretor ou gerente | São Paulo, São Paulo`

**Fine-tuned** — correct structure, natural pt-BR, invents coherent specifics:
> `## Síntese`
> José Pereira é um diretor de 49 anos que equilibra liderança estratégica, fé evangélica
> ativa e paixão por futebol e música, mas costuma adiar a conclusão da graduação em
> Administração enquanto luta contra a procrastinação.

**Base** — wrong structure, restates the input as a form:
> `**1. Identificação Pessoal:**`
> José Pereira é um homem de 49 anos, solteiro, residente no município de São Paulo,
> estado de São Paulo, Brasil.

## Reproducing

```bash
make setup && make data && make train && make export && make eval && make report
```

Artifacts (gitignored): `outputs/adapter/` (333MB), `outputs/gguf/personas-tuned-q4_k_m.gguf`
(4.68GB), `outputs/eval/report.html`.
