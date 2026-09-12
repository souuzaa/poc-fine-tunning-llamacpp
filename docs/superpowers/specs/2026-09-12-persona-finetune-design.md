# Design — Brazilian persona generator: Qwen3-8B QLoRA to llama.cpp

- **Date:** 2026-09-12
- **Status:** Approved
- **Decisions:** see [`docs/adr/`](../../adr/README.md)

## 1. Goal

Fine-tune Qwen3-8B on NVIDIA's synthetic Brazilian persona corpus so that, given a block
of demographic attributes, it writes a rich six-section persona narrative in Brazilian
Portuguese. Train with Unsloth QLoRA entirely on one RTX 3060 (12GB), export to GGUF, serve
with llama.cpp, and demonstrate with measurements that the fine-tune beat the base model.

**Success is:** the tuned model beats base on held-out perplexity, produces valid
six-section output on >95% of test prompts with zero `<think>` leakage, and grounds
noticeably better in the supplied attributes — all reproducible offline via `make`.

**Out of scope:** multi-turn persona role-play, a production serving stack, a LangChain
demo application (a possible stage 7), training beyond a single 3060.

## 2. Data

`nvidia/Nemotron-Personas-Brazil` — 1M rows, 21 columns, 2.5GB parquet, **CC-BY-4.0**.
Fully synthetic: despite being "person data" it contains no real PII, so there is no
privacy constraint on local training or on sharing generated samples.

**Input attributes:** `sex, age, marital_status, education_level, occupation,
municipality, state, country`, plus a **name extracted from the narrative text** (ADR 0005).

**Target sections** (ADR 0004): `persona`, `professional_persona`, `cultural_background`,
`skills_and_expertise`, `hobbies_and_interests`, `career_goals_and_ambitions`, rendered as
fixed-order markdown headings. The four lifestyle facets and both `*_list` columns are
excluded.

**Splits, disjoint by `uuid`:** 20,000 train / 500 validation / 200 test.

## 3. Environment (ADR 0011)

| Component | Choice |
|---|---|
| Python | 3.12 in a project-local `uv` venv (host is 3.14, unsupported) |
| torch | cu128 wheels, `sm_86` |
| Training stack | unsloth, trl, peft, transformers, datasets, bitsandbytes — pinned in `uv.lock` |
| Eval stack | langchain-core, httpx |
| llama.cpp | vendored at a pinned commit, `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86` |

Hardware: RTX 3060 12GB (~9.5GB free with GNOME resident), 30GB RAM, 304GB disk,
driver 595.84 / CUDA 13.3.

## 4. Pipeline

Six stages, each a standalone resumable script behind one `Makefile` target and one YAML
config (`configs/qwen3-8b-personas.yaml`).

| # | Target | Does | Output |
|---|---|---|---|
| 0 | `make setup` | uv venv; clone and build llama.cpp; verify torch sees CUDA | `.venv/`, `vendor/llama.cpp/build/bin/` |
| 1 | `make data` | stream parquet, extract name, render pairs, token histogram, filter, split | `data/{train,val,test}.jsonl` + `manifest.json` |
| 2 | `make train` | Unsloth QLoRA, response-only loss, checkpointed | `outputs/adapter/`, loss curves |
| 3 | `make export` | merge to fp16, convert to GGUF, quantise (tuned **and** base) | `outputs/gguf/*.gguf` |
| 4 | `make serve` | `llama-server -ngl 99` | endpoint on :8080 |
| 5 | `make eval` | LangChain `.batch()`, sequential base then tuned | `outputs/eval/results.json` |
| 6 | `make report` | side-by-side HTML | published artifact |

`make smoke` runs stages 1-6 on a 200-row slice to validate the chain end to end in
minutes before committing to a multi-hour run.

## 5. Layout

```
├── Makefile
├── pyproject.toml / uv.lock
├── configs/qwen3-8b-personas.yaml    # single source of truth for hyperparameters
├── src/personas/
│   ├── prompt.py      # render_prompt / render_target — THE shared contract (ADR 0006)
│   ├── schema.py      # attribute and section dataclasses
│   ├── grounding.py   # attribute-recall checks
│   └── llm.py         # LangChain Runnable over llama-server /completion (ADR 0007)
├── scripts/0X_*.py
├── tests/test_prompt_parity.py       # train prompt == eval prompt, byte for byte
├── data/ · outputs/ · vendor/        # gitignored
└── docs/adr/ · docs/superpowers/specs/
```

`src/personas/prompt.py` is the linchpin: stages 1, 5 and 6 all import it, so prompt drift
between training and inference is structurally impossible rather than merely watched for.
It carries a `PROMPT_VERSION` constant, written into the dataset manifest and checked at
eval time — a changed system prompt invalidates the trained adapter, and that must fail
loudly.

## 6. Training configuration (ADR 0002, 0009)

**Model:** Qwen3-8B, `load_in_4bit=True`, `max_seq_length=2048`.

**LoRA:** `r=32`, `lora_alpha=32`, `lora_dropout=0`, `bias="none"`, targeting
`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`,
`use_gradient_checkpointing="unsloth"`, `random_state=3407`.

**Trainer (TRL `SFTTrainer`):**

| Parameter | Value |
|---|---|
| `per_device_train_batch_size` | 2 |
| `gradient_accumulation_steps` | 8 (effective batch 16) |
| `num_train_epochs` | 1 (~1250 optimizer steps) |
| `learning_rate` | 2e-4, cosine, `warmup_ratio=0.03` |
| `optim` | `adamw_8bit` |
| `weight_decay` | 0.01 |
| precision | bf16 (Ampere native) |
| `packing` | False — response-only loss requires unpacked sequences |
| eval / save | eval every 100 steps, save every 250, `save_total_limit=3` |

Loss is computed on the assistant response only, via `train_on_responses_only` with Qwen3
markers `<|im_start|>user\n` and `<|im_start|>assistant\n`.

**VRAM budget:** ~7.7-8.3GB against ~9.5GB free (breakdown in ADR 0002). Documented OOM
fallbacks, in order: batch 1 with grad-accum 16; `max_seq_length` 1536; stop the GNOME
session to reclaim 2.7GB.

**Throughput:** a measured ETA is printed after 50 steps, so a wrong estimate surfaces in
minutes rather than at hour four.

## 7. Export, serving, evaluation

**Export (ADR 0008):** merge to fp16, `convert_hf_to_gguf.py` to f16 GGUF, `llama-quantize`
to Q4_K_M (plus Q8_0 as a quality reference). The base model goes through the identical
three steps, so base-versus-tuned differs only by LoRA weights.

**Serving:** `llama-server -m <gguf> -ngl 99 -c 4096 -fa --host 127.0.0.1 --port 8080`.
Full GPU offload; Q4_K_M 8B is ~5GB plus KV cache.

**Hard constraint:** base and tuned **cannot be served concurrently** — two models exceed
the free VRAM. Eval runs sequentially with the Makefile owning server lifecycle (ADR 0010).

**Metrics (ADR 0010):** held-out perplexity via `llama-perplexity`; format validity
(six sections, correct order, no `<think>` leakage); per-field grounding recall (name,
municipality, state, occupation, age); pt-BR heuristic and length distribution; plus a
30-row three-column side-by-side (ground truth / base / tuned) HTML report.

Decoding is identical across both models: temp 0.7, top_p 0.8, top_k 20,
repeat_penalty 1.05, `n_predict` 1024, fixed seed.

## 8. Risks

| Risk | Mitigation |
|---|---|
| `<think>` leakage silently corrupts targets | Single renderer with `enable_thinking=False`; parity test; leakage is a scored metric (ADR 0006) |
| OOM at batch 2 | Documented fallback ladder; ~1.2GB headroom by design |
| Unsloth/torch version churn breaks a long run | Everything pinned in committed `uv.lock` (ADR 0011) |
| Name extraction fails on many rows | Prep reports the drop rate and fails loudly above 5% (ADR 0005) |
| Model learns format but ignores attributes | Grounding recall is a first-class metric, not an afterthought |
| Quantisation, not the fine-tune, degrades output | Q8_0 reference build isolates the two effects |
| Throughput estimate wrong by 3x | ETA measured and printed at step 50 |

## 9. Open questions

None blocking. Deliberately deferred: LLM-as-judge scoring (the harness already emits the
paired generations it would need), the four lifestyle facets, a LangChain demo app, and a
50k-example run once the pipeline is proven.
