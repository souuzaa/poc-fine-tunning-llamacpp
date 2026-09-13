# Brazilian persona generator — Qwen3-8B QLoRA on one RTX 3060

Fine-tunes Qwen3-8B with Unsloth QLoRA on `nvidia/Nemotron-Personas-Brazil`, exports to
GGUF, and evaluates against the base model through llama.cpp. Everything runs on a single
12GB GPU.

**Task:** demographic attributes in, six-section Brazilian-Portuguese persona out.

| | Base Qwen3-8B | After fine-tuning |
|---|---|---|
| **Format validity** | 0% | **100%** |
| **Perplexity** (held-out) | 4.73 | **2.77** |
| Output | a demographic *report about* the person | a persona *of* the person |

Measured on 200 held-out rows, identical decoding parameters, same Q4_K_M quantisation.
Full numbers in [`docs/RESULTS.md`](docs/RESULTS.md).

---

## Before and after

Both models get the **same prompt** — only the attributes, nothing else:

```
Nome: Gilcimara Souza | Sexo: Feminino | Idade: 19 | Estado civil: Solteiro
Escolaridade: Fundamental completo e médio incompleto
Ocupação: Ocupação elementar
Município: Rio de Janeiro | Estado: Rio de Janeiro | País: Brasil
```

### ❌ Before — base Qwen3-8B

> **1. Identificação:**
> Gilcimara Souza, uma jovem brasileira de 19 anos, residente no Rio de Janeiro, estado do
> Rio de Janeiro. Solteira e com 19 anos, sua vida está em fase de formação pessoal e profissional.
>
> **2. Características Demográficas:**
> Gilcimara tem 19 anos, está solteira e reside na cidade do Rio de Janeiro. Sua escolaridade
> é fundamental completa e médio incompleto, o que **indica que** ela ainda está em processo de
> educação formal. Sua ocupação é considerada elementar, **sugerindo que** ela está envolvida em
> atividades de baixa complexidade…
>
> **3. Contexto Social e Econômico:**
> Como residente do Rio de Janeiro, Gilcimara vive em uma das cidades mais populosas…

It restates the input as a numbered form and hedges everything it cannot know
("indica que", "sugerindo que"). It reads like a case file, not a person. **None of the
six required sections appear.**

### ✅ After — fine-tuned

> ## Síntese
> Gilcimara Souza é uma jovem carioca prática e sociável que combina trabalho no varejo, fé
> ativa e paixão por música ao vivo, buscando estabilidade financeira enquanto cultiva sua
> criatividade.
>
> ## Perfil profissional
> Gilcimara Souza, 19, trabalha como atendente em um comércio de bairro no Rio, onde utiliza
> seu domínio de caixa e negociação para lidar com o fluxo intenso de clientes, enquanto sua
> praticidade e curiosidade a impulsionam a buscar uma posição de supervisão…
>
> ## Origem cultural
> Gilcimara nasceu e foi criada no **bairro da Tijuca**, em um lar de classe trabalhadora onde
> a família valorizava o esforço e a solidariedade. Cresceu ouvindo **samba e pagode** nas ruas,
> participando das festas de **Carnaval e dos blocos de rua**… nos fins de semana costuma
> reunir-se com amigos na **praia de Copacabana** ou no **bar da Lapa**, onde a música ao vivo anima…

Correct structure, confident prose, and invented-but-plausible regional detail — Tijuca,
samba, blocos de rua, Lapa. It committed to a specific person.

### 📋 What the dataset actually contains (ground truth)

> ## Síntese
> Gilcimara Souza, 19 anos, combina energia sociável, organização prática e fé católica ao
> trabalhar no varejo, curtir a praia, criar artesanato e planejar seu futuro empreendedor.

The fine-tune landed in the right register — retail work, faith, beach, creative
side-interests — on a row it had never seen.

<details>
<summary><b>Second example — Pedro Joaquim Alves, 52, construction worker, Eunápolis/BA</b></summary>

```
Nome: Pedro Joaquim Alves | Sexo: Masculino | Idade: 52 | Estado civil: Solteiro
Escolaridade: Médio completo e superior incompleto
Ocupação: Trabalhador qualificado, operário ou artesão da construção…
Município: Eunápolis | Estado: Bahia | País: Brasil
```

**❌ Before:**

> **1. Identificação:**
> Pedro Joaquim Alves, um brasileiro masculino de 52 anos, solteiro, residente na cidade de
> Eunápolis, estado da Bahia…
>
> **2. Características Demográficas:**
> Pedro pertence à geração que viveu a transição para uma sociedade mais moderna… A idade de
> 52 anos o **posiciona como** um adulto maduro… Sua localização no interior do estado da Bahia
> **indica uma possível ligação** com a cultura e tradições rurais ou semiurbanas.

**✅ After:**

> ## Síntese
> Pedro Joaquim Alves, 52 anos, é um mestre da construção que combina disciplina e
> criatividade para liderar obras sustentáveis em Eunápolis, enquanto cultiva a fé católica,
> o futebol de várzea e o cuidado com a família.
>
> ## Perfil profissional
> Pedro Joaquim Alves, 52 anos, é mestre da construção civil que combina sua experiência em
> alvenaria, carpintaria e soldagem com um planejamento meticuloso e liderança assertiva,
> buscando coordenar obras sustentáveis e abrir sua própria empresa de reformas em Eunápolis.

</details>

Thirty more side-by-side comparisons are generated into `outputs/eval/report.html` by
`make report`.

---

## Process metrics

Measured end to end on one RTX 3060 12GB (Ryzen host, 30GB RAM, NVMe).

| Stage | Command | Wall clock | Produces |
|---|---|---|---|
| Environment + llama.cpp CUDA build | `make setup` | ~25 min* | venv + `llama-server`/`quantize`/`perplexity` |
| Data preparation | `make data` | ~3 min* | 10k/500/200 JSONL splits |
| **Fine-tuning (QLoRA)** | `make train` | **7h 25m** | 344 MB LoRA adapter |
| GGUF export (per model) | `make export` | **6 min 26 s** | 4.7 GB Q4_K_M + 8.1 GB Q8_0 |
| Evaluation (200 rows × 2 models) | `make eval` | **46 min 19 s** | `results.json`, `generations.json` |
| Report | `make report` | < 1 s | `report.html` |
| **Total** | | **≈ 9 h** | |

<sub>* approximate — dominated by downloads (torch/CUDA wheels, 15.3 GB base model, dataset shards) and therefore bandwidth-dependent. Starred rows are wall clock on a first run; everything else is measured precisely from logs.</sub>

### Training detail

| | |
|---|---|
| Rows / epochs | 10,000 / 1 (exactly one pass) |
| Optimizer steps | 625 (effective batch 16 = 2 × 8 grad-accum) |
| Throughput | **42.7 s/step** wall, ≈ 570 tokens/s |
| Peak VRAM | **9.07 GB** of 11.63 GB |
| Trainable params | 87,293,952 of 8,278,029,312 (**1.05 %**) |
| In-training validation | 5 passes × 150 rows, ~2 min each |

The 3060 is the bottleneck, not the code: at 4-bit with gradient checkpointing the step
time is dominated by memory bandwidth. An initial estimate of 3-5 h proved **2.4× optimistic**;
the schedule was re-derived from a measured smoke run — see
[ADR 0012](docs/adr/0012-revised-training-budget.md), which supersedes the guess.

### Inference / evaluation detail

| | Base | Fine-tuned |
|---|---|---|
| 200 generations | 1050 s | 1610 s |
| Per row | 5.3 s | 8.1 s |
| Mean output | 2093 chars | 3072 chars |

The tuned model takes ~50 % longer per row because it writes ~50 % more text — it fills all
six sections instead of stopping early. Single-stream generation runs at **~58 tok/s** on the
3060 with full GPU offload (`-ngl 99`).

Base and tuned are evaluated **sequentially, never concurrently**: two Q4_K_M 8B models need
~10 GB plus KV cache and do not fit in 12 GB together
([ADR 0010](docs/adr/0010-evaluation-strategy.md)).

---

## Quickstart

```bash
make setup    # uv venv (Python 3.12) + CUDA build of llama.cpp
make smoke    # 200-row end-to-end check — run this first
make data     # 10k/500/200 splits
make train    # ~7.4h QLoRA on the 3060 (measured)
make export   # GGUF for tuned and base
make eval     # sequential base-vs-tuned scoring
make report   # outputs/eval/report.html
```

## Requirements

RTX 3060 12GB or better · ~9.5GB free VRAM · 30GB RAM · 60GB free disk ·
NVIDIA driver 535+ · CUDA toolkit and a C++ compiler for the llama.cpp build.

Host Python is not used — `uv` manages a project-local 3.12 environment, and `cmake`
is installed into it, so no system packages beyond a compiler and CUDA are needed.

## Pinning llama.cpp

After the first `make setup`, record the SHA and export it so builds stay reproducible:

```bash
git -C vendor/llama.cpp rev-parse HEAD
export LLAMA_COMMIT=<sha>
```

## If training runs out of memory

In order: batch 1 with grad-accum 16 → `max_seq_length` 1536 → stop the GNOME session
(`sudo systemctl isolate multi-user.target`) to reclaim ~2.7GB.

## Documentation

- `docs/superpowers/specs/2026-09-12-persona-finetune-design.md` — the design
- `docs/superpowers/plans/2026-09-12-persona-finetune.md` — the implementation plan
- `docs/adr/README.md` — why each decision was made
- `CLAUDE.md` — invariants that must not be broken

## Data

`nvidia/Nemotron-Personas-Brazil` is CC-BY-4.0 and fully synthetic. No real PII.
