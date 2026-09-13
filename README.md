# Brazilian persona generator — Qwen3-8B QLoRA on one RTX 3060

Fine-tunes Qwen3-8B with Unsloth QLoRA on `nvidia/Nemotron-Personas-Brazil`, exports to
GGUF, and evaluates against the base model through llama.cpp. Everything runs on a single
12GB GPU.

**Task:** demographic attributes in, six-section Brazilian-Portuguese persona out.

```
Nome: Vânia da Rocha | Sexo: Feminino | Idade: 63 | Estado civil: Casado
Escolaridade: Sem instrução e fundamental incompleto
Ocupação: Ocupação elementar
Município: Anápolis | Estado: Goiás | País: Brasil
```
→ `## Síntese` · `## Perfil profissional` · `## Origem cultural` · `## Habilidades` ·
`## Hobbies e interesses` · `## Objetivos de carreira`

## Quickstart

```bash
make setup    # uv venv (Python 3.12) + CUDA build of llama.cpp
make smoke    # 200-row end-to-end check — run this first
make data     # 20k/500/200 splits
make train    # ~3-5h QLoRA on the 3060
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
