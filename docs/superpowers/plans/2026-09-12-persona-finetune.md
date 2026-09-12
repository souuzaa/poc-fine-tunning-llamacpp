# Brazilian Persona Fine-Tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune Qwen3-8B with Unsloth QLoRA on synthetic Brazilian persona data, export to GGUF, and prove with measurements that it beats the base model when served by llama.cpp — all on one RTX 3060.

**Architecture:** Six independently runnable stages behind a `Makefile`, sharing one YAML config and one prompt renderer. Data prep emits JSONL; Unsloth trains a LoRA adapter; an explicit merge/convert/quantise chain produces GGUF for both the tuned and base model; a LangChain `Runnable` over llama-server's raw `/completion` drives evaluation; a static HTML report renders the comparison.

**Tech Stack:** Python 3.12 (uv), torch cu128, unsloth, trl, peft, datasets, bitsandbytes, langchain-core, httpx, llama.cpp (CUDA, `sm_86`), pytest.

**Spec:** `docs/superpowers/specs/2026-09-12-persona-finetune-design.md`

## Global Constraints

Copied verbatim from the spec and ADRs. Every task's requirements implicitly include these.

- **Python 3.12 only.** Host Python is 3.14 and cannot run torch. Never `pip install` into host Python; always `uv run` / `uv sync`.
- **All dependencies pinned** in a committed `uv.lock`.
- **Base model:** `unsloth/Qwen3-8B` loaded with `load_in_4bit=True`, `max_seq_length=2048`.
- **One prompt renderer.** Every prompt string in the project comes from `src/personas/prompt.py`. No exceptions.
- **Thinking mode off everywhere:** always `apply_chat_template(..., enable_thinking=False)`. Any `<think>` in a generation is a format failure.
- **Raw `/completion` only** for evaluation. Never `/v1/chat/completions`.
- **Base and tuned models never run concurrently** — two Q4_K_M 8B models exceed the ~9.5GB free VRAM.
- **Base and tuned share a quantisation lineage** — identical merge/convert/quantise steps.
- **LangChain is never a training dependency.** It belongs to the `eval` dependency group only.
- **Six target sections, fixed order:** `## Síntese`, `## Perfil profissional`, `## Origem cultural`, `## Habilidades`, `## Hobbies e interesses`, `## Objetivos de carreira`.
- **Splits disjoint by `uuid`:** 20,000 train / 500 val / 200 test.
- **LoRA:** `r=32`, `lora_alpha=32`, `lora_dropout=0`, `bias="none"`, `use_gradient_checkpointing="unsloth"`, `random_state=3407`.
- **Trainer:** batch 2 × grad-accum 8, 1 epoch, LR 2e-4 cosine, `warmup_ratio=0.03`, `adamw_8bit`, `weight_decay=0.01`, bf16, `packing=False`.
- **Decoding, identical for base and tuned:** temp 0.7, top_p 0.8, top_k 20, repeat_penalty 1.05, `n_predict` 1024, seed 3407.
- **llama.cpp** vendored at a pinned commit, built `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` / `uv.lock` | Pinned Python 3.12 environment; `train` and `eval` dependency groups |
| `configs/qwen3-8b-personas.yaml` | Single source of truth for every hyperparameter and path |
| `src/personas/config.py` | Loads and validates the YAML into a typed object |
| `src/personas/schema.py` | `PersonaAttributes` and `PersonaSections` dataclasses; the six section headings |
| `src/personas/prompt.py` | `PROMPT_VERSION`, `SYSTEM_PROMPT`, `render_attributes`, `render_target`, `render_prompt`, `parse_sections` |
| `src/personas/names.py` | `extract_name` — pulls the person's name out of the `persona` narrative |
| `src/personas/grounding.py` | `score_grounding`, `check_format`, `is_portuguese` |
| `src/personas/llm.py` | `LlamaCppCompletion` — LangChain `Runnable` over llama-server `/completion` |
| `scripts/00_setup_llamacpp.sh` | Clone llama.cpp at a pinned commit, build with CUDA, verify binaries |
| `scripts/01_prepare_data.py` | Stream parquet, extract names, render pairs, token histogram, split, manifest |
| `scripts/02_train.py` | Unsloth QLoRA training with response-only loss and measured ETA |
| `scripts/03_export_gguf.py` | Merge, convert, quantise — for tuned and base alike |
| `scripts/04_serve.sh` | Launch `llama-server` with full GPU offload |
| `scripts/05_evaluate.py` | Generate over the test split via LangChain, score, write `results.json` |
| `scripts/06_report.py` | Render the side-by-side HTML report |
| `tests/test_prompt_parity.py` | Training prompt equals eval prompt, byte for byte |
| `tests/test_names.py` | Name extraction behaviour |
| `tests/test_grounding.py` | Grounding, format and language scorers |
| `Makefile` | `setup data train export serve eval report smoke test` |

---

## Task 1: Project scaffolding and environment

**Files:**
- Create: `pyproject.toml`, `configs/qwen3-8b-personas.yaml`, `src/personas/__init__.py`, `src/personas/config.py`, `Makefile`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `personas.config.load_config(path: str | Path) -> Config`. `Config` exposes `.model`, `.data`, `.lora`, `.train`, `.decode`, `.paths` as attribute-accessible nested objects.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "personas"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "pyyaml>=6.0.2",
]

[project.optional-dependencies]
train = [
    "torch==2.8.0",
    "unsloth==2025.9.1",
    "unsloth-zoo==2025.9.1",
    "trl==0.21.0",
    "peft==0.17.0",
    "transformers==4.55.2",
    "datasets==4.0.0",
    "bitsandbytes==0.47.0",
    "accelerate==1.10.0",
    "huggingface-hub>=0.34.0",
    "pyarrow>=17.0.0",
]
eval = [
    "langchain-core==0.3.75",
    "httpx==0.28.1",
    "jinja2>=3.1.4",
]
dev = [
    "pytest>=8.3.0",
]

[tool.uv]
# torch cu128 wheels: sm_86 supported, driver 595 is backward compatible
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/personas"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

Pinned versions are a starting point. If `uv sync` cannot resolve them, relax the
Unsloth/trl/transformers trio together (they are coupled), re-lock, and record the
working set — do **not** switch to unpinned installs (ADR 0011).

- [ ] **Step 2: Create `configs/qwen3-8b-personas.yaml`**

```yaml
model:
  base_id: unsloth/Qwen3-8B
  max_seq_length: 2048
  load_in_4bit: true

data:
  hf_dataset: nvidia/Nemotron-Personas-Brazil
  n_train: 20000
  n_val: 500
  n_test: 200
  seed: 3407
  max_name_drop_rate: 0.05

lora:
  r: 32
  lora_alpha: 32
  lora_dropout: 0.0
  bias: none
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  use_gradient_checkpointing: unsloth
  random_state: 3407

train:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  num_train_epochs: 1
  learning_rate: 2.0e-4
  lr_scheduler_type: cosine
  warmup_ratio: 0.03
  optim: adamw_8bit
  weight_decay: 0.01
  logging_steps: 10
  eval_steps: 100
  save_steps: 250
  save_total_limit: 3
  seed: 3407

decode:
  temperature: 0.7
  top_p: 0.8
  top_k: 20
  repeat_penalty: 1.05
  n_predict: 1024
  seed: 3407

paths:
  data_dir: data
  outputs_dir: outputs
  llamacpp_dir: vendor/llama.cpp
```

- [ ] **Step 3: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path
from personas.config import load_config

CONFIG = Path(__file__).parent.parent / "configs" / "qwen3-8b-personas.yaml"


def test_load_config_exposes_nested_attributes():
    cfg = load_config(CONFIG)
    assert cfg.model.base_id == "unsloth/Qwen3-8B"
    assert cfg.model.max_seq_length == 2048
    assert cfg.lora.r == 32
    assert cfg.train.per_device_train_batch_size == 2
    assert cfg.decode.top_k == 20


def test_effective_batch_size_is_sixteen():
    cfg = load_config(CONFIG)
    effective = cfg.train.per_device_train_batch_size * cfg.train.gradient_accumulation_steps
    assert effective == 16


def test_splits_are_the_budget_from_adr_0009():
    cfg = load_config(CONFIG)
    assert (cfg.data.n_train, cfg.data.n_val, cfg.data.n_test) == (20000, 500, 200)
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `uv run --extra dev pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'personas.config'`

- [ ] **Step 5: Write `src/personas/config.py`**

```python
"""Loads the YAML config into a dotted-access object.

One config file is the source of truth for every hyperparameter (see the spec);
scripts must never hardcode a value that lives here.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "qwen3-8b-personas.yaml"


def _namespacify(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _namespacify(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_namespacify(v) for v in value]
    return value


def load_config(path: str | Path = DEFAULT_CONFIG) -> SimpleNamespace:
    """Read the YAML config and return it with attribute access."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config at {path} must be a mapping, got {type(raw).__name__}")
    for section in ("model", "data", "lora", "train", "decode", "paths"):
        if section not in raw:
            raise ValueError(f"config at {path} is missing the '{section}' section")
    return _namespacify(raw)
```

Also create an empty `src/personas/__init__.py`.

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run --extra dev pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 7: Create the `Makefile`**

```makefile
CONFIG ?= configs/qwen3-8b-personas.yaml
UV     ?= uv

.PHONY: setup data train export serve eval report smoke test clean

setup:
	$(UV) sync --extra train --extra eval --extra dev
	$(UV) run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible to torch'; print('torch', torch.__version__, '|', torch.cuda.get_device_name(0))"
	bash scripts/00_setup_llamacpp.sh

data:
	$(UV) run python scripts/01_prepare_data.py --config $(CONFIG)

train:
	$(UV) run python scripts/02_train.py --config $(CONFIG)

export:
	$(UV) run python scripts/03_export_gguf.py --config $(CONFIG) --which tuned
	$(UV) run python scripts/03_export_gguf.py --config $(CONFIG) --which base

serve:
	bash scripts/04_serve.sh $(MODEL)

eval:
	$(UV) run python scripts/05_evaluate.py --config $(CONFIG)

report:
	$(UV) run python scripts/06_report.py --config $(CONFIG)

test:
	$(UV) run --extra dev pytest -v

smoke:
	$(UV) run python scripts/01_prepare_data.py --config $(CONFIG) --smoke
	@echo "Smoke data ready. Run: make train export eval report"

clean:
	rm -rf outputs/merged-16bit outputs/gguf/*-f16.gguf
```

- [ ] **Step 8: Verify the environment builds**

Run: `make setup` (the llama.cpp line will fail until Task 4 — that is expected; confirm `uv sync` and the torch CUDA assertion both pass first)
Expected: `torch 2.8.0 | NVIDIA GeForce RTX 3060`

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock configs/ src/personas/__init__.py src/personas/config.py tests/test_config.py Makefile
git commit -m "feat: scaffold pinned uv environment, config loader and Makefile"
```

---

## Task 2: Schema, name extraction and the shared prompt renderer

This is the most important task in the plan. Everything downstream depends on these
functions producing exactly one canonical string per prompt (ADR 0006).

**Files:**
- Create: `src/personas/schema.py`, `src/personas/names.py`, `src/personas/prompt.py`
- Test: `tests/test_names.py`, `tests/test_prompt_parity.py`

**Interfaces:**
- Consumes: `personas.config.load_config` (Task 1).
- Produces:
  - `personas.schema.PersonaAttributes` — frozen dataclass with fields `name, sex, age, marital_status, education_level, occupation, municipality, state, country`; classmethod `from_row(row: dict, name: str) -> PersonaAttributes`.
  - `personas.schema.SECTIONS: tuple[tuple[str, str], ...]` — six `(heading, source_column)` pairs in order.
  - `personas.names.extract_name(persona_text: str) -> str | None`
  - `personas.prompt.PROMPT_VERSION: str`, `SYSTEM_PROMPT: str`
  - `personas.prompt.render_attributes(attrs: PersonaAttributes) -> str`
  - `personas.prompt.render_target(row: dict) -> str`
  - `personas.prompt.render_prompt(attrs, tokenizer) -> str` — inference-time string
  - `personas.prompt.render_training_text(attrs, row, tokenizer) -> str` — full train-time string
  - `personas.prompt.parse_sections(text: str) -> dict[str, str]`

- [ ] **Step 1: Write `src/personas/schema.py`**

```python
"""Typed view over the Nemotron-Personas-Brazil columns we use.

Six target sections, fixed order (ADR 0004). The order is load-bearing: evaluation
scores section presence *and* ordering, and the model is trained to reproduce it.
"""

from __future__ import annotations

from dataclasses import dataclass

SECTIONS: tuple[tuple[str, str], ...] = (
    ("Síntese", "persona"),
    ("Perfil profissional", "professional_persona"),
    ("Origem cultural", "cultural_background"),
    ("Habilidades", "skills_and_expertise"),
    ("Hobbies e interesses", "hobbies_and_interests"),
    ("Objetivos de carreira", "career_goals_and_ambitions"),
)

ATTRIBUTE_COLUMNS: tuple[str, ...] = (
    "sex",
    "age",
    "marital_status",
    "education_level",
    "occupation",
    "municipality",
    "state",
    "country",
)


@dataclass(frozen=True)
class PersonaAttributes:
    name: str
    sex: str
    age: int
    marital_status: str
    education_level: str
    occupation: str
    municipality: str
    state: str
    country: str

    @classmethod
    def from_row(cls, row: dict, name: str) -> "PersonaAttributes":
        return cls(
            name=name,
            sex=row["sex"],
            age=int(row["age"]),
            marital_status=row["marital_status"],
            education_level=row["education_level"],
            occupation=row["occupation"],
            municipality=row["municipality"],
            state=row["state"],
            country=row["country"],
        )
```

- [ ] **Step 2: Write the failing test for name extraction**

```python
# tests/test_names.py
import pytest
from personas.names import extract_name


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Marcos Antunes é um operador técnico organizado.", "Marcos Antunes"),
        ("Ana Paula dos Santos trabalha como enfermeira.", "Ana Paula dos Santos"),
        ("João da Silva mora em Recife.", "João da Silva"),
        ("Luíza Gonçalves, 34 anos, é professora.", "Luíza Gonçalves"),
    ],
)
def test_extracts_leading_proper_name(text, expected):
    assert extract_name(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "uma pessoa comum que trabalha muito",
        "Ele é um operador técnico.",
    ],
)
def test_returns_none_when_no_name_present(text):
    assert extract_name(text) is None


def test_does_not_swallow_the_rest_of_the_sentence():
    name = extract_name("Carlos Eduardo Ferreira Lima Souza Pereira é engenheiro.")
    assert name is not None
    assert len(name.split()) <= 4
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run --extra dev pytest tests/test_names.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'personas.names'`

- [ ] **Step 4: Write `src/personas/names.py`**

```python
"""Extracts the person's name from the narrative text.

The dataset has no `name` column — the name lives inside the prose (ADR 0005). Without
it the model must invent a name and every reference-based comparison becomes noise.
"""

from __future__ import annotations

import re

# A capitalised token, allowing Portuguese accents, hyphens and apostrophes.
_TOKEN = r"[A-ZÁÀÂÃÉÊÍÓÔÕÚÜÇ][a-zàáâãéêíóôõúüç'\-]+"
# Lowercase connectors that appear inside Brazilian names: "dos Santos", "da Silva".
_CONNECTOR = r"(?:d[aeo]s?|e)"

_NAME_RE = re.compile(
    rf"^({_TOKEN}(?:\s+(?:{_CONNECTOR}\s+)?{_TOKEN}){{1,3}})\b"
)

# Words that start a sentence but are not names.
_STOPWORDS = {"Ele", "Ela", "Eles", "Elas", "Uma", "Um", "Esse", "Essa", "Este", "Esta"}


def extract_name(persona_text: str) -> str | None:
    """Return the leading proper name, or None if the text does not start with one.

    Requires at least two tokens (given name + surname) so that a sentence beginning
    with a capitalised common word is not mistaken for a name.
    """
    if not persona_text:
        return None
    match = _NAME_RE.match(persona_text.strip())
    if match is None:
        return None
    name = match.group(1)
    if name.split()[0] in _STOPWORDS:
        return None
    return name
```

- [ ] **Step 5: Run it to verify it passes**

Run: `uv run --extra dev pytest tests/test_names.py -v`
Expected: 8 passed

- [ ] **Step 6: Write `src/personas/prompt.py`**

`PROMPT_VERSION` must be bumped whenever `SYSTEM_PROMPT` or either renderer changes —
stage 1 writes it into the manifest and stage 5 refuses to evaluate a mismatch.

```python
"""The single source of every prompt string in this project (ADR 0006).

Data prep, evaluation and reporting all import from here. If you find yourself building
a prompt string anywhere else, that is the bug.
"""

from __future__ import annotations

import re

from personas.schema import SECTIONS, PersonaAttributes

PROMPT_VERSION = "1.0.0"

SYSTEM_PROMPT = (
    "Você é um especialista em criar perfis demográficos detalhados de brasileiros. "
    "A partir dos atributos fornecidos, escreva um perfil completo em português do Brasil, "
    "usando exatamente as seis seções indicadas, na ordem dada, sem adicionar outras seções."
)

_SECTION_TEMPLATE = "## {heading}\n{body}"

_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def render_attributes(attrs: PersonaAttributes) -> str:
    """The user turn: a compact pt-BR attribute block."""
    return (
        f"Nome: {attrs.name} | Sexo: {attrs.sex} | Idade: {attrs.age} | "
        f"Estado civil: {attrs.marital_status}\n"
        f"Escolaridade: {attrs.education_level}\n"
        f"Ocupação: {attrs.occupation}\n"
        f"Município: {attrs.municipality} | Estado: {attrs.state} | País: {attrs.country}\n\n"
        "Escreva o perfil completo nas seis seções."
    )


def render_target(row: dict) -> str:
    """The assistant turn: six fixed sections built from the source columns."""
    parts = [
        _SECTION_TEMPLATE.format(heading=heading, body=str(row[column]).strip())
        for heading, column in SECTIONS
    ]
    return "\n\n".join(parts)


def _messages(attrs: PersonaAttributes) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_attributes(attrs)},
    ]


def render_prompt(attrs: PersonaAttributes, tokenizer) -> str:
    """Inference-time string. Thinking mode is disabled unconditionally."""
    return tokenizer.apply_chat_template(
        _messages(attrs),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def render_training_text(attrs: PersonaAttributes, row: dict, tokenizer) -> str:
    """Full train-time string: prompt plus the target assistant turn.

    Invariant enforced by tests/test_prompt_parity.py: this string always starts with
    exactly what render_prompt() produces for the same attributes.
    """
    messages = _messages(attrs) + [
        {"role": "assistant", "content": render_target(row)}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def parse_sections(text: str) -> dict[str, str]:
    """Split generated markdown into {heading: body}, preserving encounter order."""
    sections: dict[str, str] = {}
    matches = list(_HEADING_RE.finditer(text))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group(1)] = text[match.end():end].strip()
    return sections
```

- [ ] **Step 7: Write the parity test**

This test is the whole point of ADR 0006. It needs the real tokenizer, so it is marked
`slow` and skipped when transformers is unavailable.

```python
# tests/test_prompt_parity.py
import pytest

from personas.prompt import (
    PROMPT_VERSION,
    parse_sections,
    render_attributes,
    render_prompt,
    render_target,
    render_training_text,
)
from personas.schema import SECTIONS, PersonaAttributes

ATTRS = PersonaAttributes(
    name="Marcos Antunes",
    sex="Masculino",
    age=22,
    marital_status="Solteiro",
    education_level="Fundamental completo e médio incompleto",
    occupation="Operador de instalação ou máquina ou montador",
    municipality="São Pedro de Alcântara",
    state="Santa Catarina",
    country="Brasil",
)

ROW = {column: f"texto de {column}" for _, column in SECTIONS}


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained("unsloth/Qwen3-8B")


def test_prompt_version_is_set():
    assert PROMPT_VERSION


def test_attributes_block_contains_every_attribute():
    rendered = render_attributes(ATTRS)
    for value in ("Marcos Antunes", "Masculino", "22", "Solteiro",
                  "São Pedro de Alcântara", "Santa Catarina", "Brasil"):
        assert value in rendered


def test_target_has_six_sections_in_order():
    parsed = parse_sections(render_target(ROW))
    assert list(parsed) == [heading for heading, _ in SECTIONS]


@pytest.mark.slow
def test_training_text_starts_with_the_inference_prompt(tokenizer):
    """The model must see at inference exactly the prefix it was trained on."""
    prompt = render_prompt(ATTRS, tokenizer)
    training_text = render_training_text(ATTRS, ROW, tokenizer)
    assert training_text.startswith(prompt), (
        "Train/inference prompt drift.\n"
        f"prompt tail:   {prompt[-120:]!r}\n"
        f"training text: {training_text[len(prompt) - 120:len(prompt) + 60]!r}"
    )


@pytest.mark.slow
def test_no_thinking_block_is_requested(tokenizer):
    prompt = render_prompt(ATTRS, tokenizer)
    # enable_thinking=False may emit an EMPTY think block; it must never be an open one.
    assert "<think>" not in prompt or "<think>\n\n</think>" in prompt
```

Register the marker in `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
markers = ["slow: requires downloading the tokenizer"]
```

- [ ] **Step 8: Run the tests**

Run: `uv run --extra dev --extra train pytest tests/ -v`
Expected: all pass. If `test_training_text_starts_with_the_inference_prompt` fails, **stop
and fix it before any other work** — every downstream measurement depends on it.

- [ ] **Step 9: Commit**

```bash
git add src/personas/schema.py src/personas/names.py src/personas/prompt.py tests/test_names.py tests/test_prompt_parity.py pyproject.toml
git commit -m "feat: add schema, name extraction and the shared prompt renderer"
```

---

## Task 3: Data preparation

**Files:**
- Create: `scripts/01_prepare_data.py`
- Test: `tests/test_prepare_data.py`

**Interfaces:**
- Consumes: `personas.config.load_config`, `personas.names.extract_name`, `personas.prompt.render_prompt/render_training_text/PROMPT_VERSION`, `personas.schema.PersonaAttributes`.
- Produces: `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`, `data/manifest.json`. Each JSONL line is `{"uuid", "attributes", "prompt", "target", "text", "n_tokens"}` where `text` is the full training string and `prompt` is the inference prefix. Also `scripts/01_prepare_data.py:build_record(row, tokenizer) -> dict | None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prepare_data.py
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "01_prepare_data.py"
spec = importlib.util.spec_from_file_location("prepare_data", SCRIPT)
prepare_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_data)

ROW = {
    "uuid": "abc123",
    "persona": "Marcos Antunes é um operador técnico organizado.",
    "professional_persona": "Marcos Antunes opera máquinas CNC.",
    "cultural_background": "Marcos cresceu em São Pedro de Alcântara.",
    "skills_and_expertise": "Marcos possui experiência com máquinas.",
    "hobbies_and_interests": "Marcos gosta de futebol.",
    "career_goals_and_ambitions": "Marcos pretende concluir o ensino médio.",
    "sex": "Masculino",
    "age": 22,
    "marital_status": "Solteiro",
    "education_level": "Fundamental completo e médio incompleto",
    "occupation": "Operador de instalação ou máquina ou montador",
    "municipality": "São Pedro de Alcântara",
    "state": "Santa Catarina",
    "country": "Brasil",
}


class FakeTokenizer:
    """Stands in for the Qwen3 tokenizer: same interface, trivial template."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            enable_thinking=True, **kwargs):
        assert enable_thinking is False, "thinking mode must always be disabled"
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def __call__(self, text, **kwargs):
        return {"input_ids": text.split()}


def test_build_record_produces_a_prompt_that_prefixes_the_training_text():
    record = prepare_data.build_record(ROW, FakeTokenizer())
    assert record is not None
    assert record["text"].startswith(record["prompt"])


def test_build_record_injects_the_extracted_name():
    record = prepare_data.build_record(ROW, FakeTokenizer())
    assert record["attributes"]["name"] == "Marcos Antunes"
    assert "Marcos Antunes" in record["prompt"]


def test_build_record_returns_none_when_the_name_cannot_be_extracted():
    row = dict(ROW, persona="uma pessoa comum que trabalha muito")
    assert prepare_data.build_record(row, FakeTokenizer()) is None


def test_build_record_returns_none_on_a_missing_section():
    row = dict(ROW, hobbies_and_interests="")
    assert prepare_data.build_record(row, FakeTokenizer()) is None


def test_split_indices_are_disjoint():
    train, val, test = prepare_data.split_indices(total=100, n_train=60, n_val=20,
                                                  n_test=10, seed=3407)
    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))
    assert (len(train), len(val), len(test)) == (60, 20, 10)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --extra dev pytest tests/test_prepare_data.py -v`
Expected: FAIL — the script does not exist yet.

- [ ] **Step 3: Write `scripts/01_prepare_data.py`**

```python
#!/usr/bin/env python
"""Stage 1 — build train/val/test JSONL from nvidia/Nemotron-Personas-Brazil.

Streams the parquet so the 2.5GB corpus is never fully materialised. For every row:
extract the name (ADR 0005), render the prompt and target through the shared renderer
(ADR 0006), measure token length, and keep it only if it fits max_seq_length.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from personas.config import load_config  # noqa: E402
from personas.names import extract_name  # noqa: E402
from personas.prompt import (  # noqa: E402
    PROMPT_VERSION,
    render_prompt,
    render_training_text,
)
from personas.schema import SECTIONS, PersonaAttributes  # noqa: E402


def build_record(row: dict, tokenizer) -> dict | None:
    """Turn a source row into a training record, or None if it is unusable."""
    name = extract_name(str(row.get("persona") or ""))
    if name is None:
        return None
    for _, column in SECTIONS:
        if not str(row.get(column) or "").strip():
            return None

    attrs = PersonaAttributes.from_row(row, name)
    prompt = render_prompt(attrs, tokenizer)
    text = render_training_text(attrs, row, tokenizer)
    if not text.startswith(prompt):
        raise RuntimeError(
            "prompt/training-text drift detected — see ADR 0006 and "
            "tests/test_prompt_parity.py"
        )
    return {
        "uuid": row["uuid"],
        "attributes": attrs.__dict__,
        "prompt": prompt,
        "target": text[len(prompt):],
        "text": text,
        "n_tokens": len(tokenizer(text)["input_ids"]),
    }


def split_indices(total: int, n_train: int, n_val: int, n_test: int, seed: int):
    """Disjoint index splits over `total` usable records."""
    needed = n_train + n_val + n_test
    if total < needed:
        raise ValueError(f"only {total} usable rows, need {needed}")
    order = list(range(total))
    random.Random(seed).shuffle(order)
    return (
        order[:n_train],
        order[n_train:n_train + n_val],
        order[n_train + n_val:needed],
    )


def histogram(lengths: list[int]) -> str:
    if not lengths:
        return "(no records)"
    ordered = sorted(lengths)

    def pct(p: float) -> int:
        return ordered[min(len(ordered) - 1, int(len(ordered) * p))]

    buckets = Counter(min(length // 256 * 256, 3072) for length in lengths)
    lines = [
        f"  n={len(lengths)} min={ordered[0]} p50={pct(0.5)} "
        f"p90={pct(0.9)} p99={pct(0.99)} max={ordered[-1]}"
    ]
    for bucket in sorted(buckets):
        bar = "#" * max(1, round(40 * buckets[bucket] / len(lengths)))
        lines.append(f"  {bucket:>5}+ {bar} {buckets[bucket]}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3-8b-personas.yaml")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny splits for an end-to-end pipeline check")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_train, n_val, n_test = cfg.data.n_train, cfg.data.n_val, cfg.data.n_test
    if args.smoke:
        n_train, n_val, n_test = 160, 20, 20

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_id)
    stream = load_dataset(cfg.data.hf_dataset, split="train", streaming=True)

    # Oversample: some rows are dropped for a missing name or an over-long target.
    target_count = n_train + n_val + n_test
    scan_limit = int(target_count * 1.6) + 500

    records, seen, dropped_name, dropped_empty, dropped_long = [], 0, 0, 0, 0
    lengths: list[int] = []

    for row in stream:
        seen += 1
        if seen > scan_limit or len(records) >= target_count:
            break
        if extract_name(str(row.get("persona") or "")) is None:
            dropped_name += 1
            continue
        record = build_record(row, tokenizer)
        if record is None:
            dropped_empty += 1
            continue
        lengths.append(record["n_tokens"])
        if record["n_tokens"] > cfg.model.max_seq_length:
            dropped_long += 1
            continue
        records.append(record)

    drop_rate = dropped_name / max(1, seen)
    print(f"scanned={seen} usable={len(records)} "
          f"dropped: name={dropped_name} ({drop_rate:.1%}) "
          f"empty={dropped_empty} too_long={dropped_long}")
    print("token length distribution (full training text):")
    print(histogram(lengths))

    if drop_rate > cfg.data.max_name_drop_rate:
        print(f"\nFATAL: name extraction failed on {drop_rate:.1%} of rows, "
              f"limit is {cfg.data.max_name_drop_rate:.1%}. "
              f"Fix the regex in src/personas/names.py (ADR 0005).", file=sys.stderr)
        return 1

    train_idx, val_idx, test_idx = split_indices(
        len(records), n_train, n_val, n_test, cfg.data.seed
    )

    data_dir = Path(cfg.paths.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for split_name, indices in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        path = data_dir / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for index in indices:
                handle.write(json.dumps(records[index], ensure_ascii=False) + "\n")
        print(f"wrote {path} ({len(indices)} rows)")

    uuids = {name: {records[i]["uuid"] for i in idx}
             for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx))}
    assert not (uuids["train"] & uuids["val"]), "train/val overlap"
    assert not (uuids["train"] & uuids["test"]), "train/test overlap"
    assert not (uuids["val"] & uuids["test"]), "val/test overlap"

    manifest = {
        "prompt_version": PROMPT_VERSION,
        "base_id": cfg.model.base_id,
        "dataset": cfg.data.hf_dataset,
        "max_seq_length": cfg.model.max_seq_length,
        "counts": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "scanned": seen,
        "dropped": {"name": dropped_name, "empty": dropped_empty, "too_long": dropped_long},
        "smoke": args.smoke,
    }
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {data_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the unit tests**

Run: `uv run --extra dev pytest tests/test_prepare_data.py -v`
Expected: 5 passed

- [ ] **Step 5: Run the real smoke preparation**

Run: `make smoke`
Expected: name drop rate well under 5%; a printed token histogram; `data/train.jsonl`
(160 rows), `data/val.jsonl` (20), `data/test.jsonl` (20), `data/manifest.json`.

**Read the histogram before continuing.** If p99 exceeds 2048, `max_seq_length` is
truncating real examples — either raise it (costs VRAM) or revisit the section scope
(ADR 0005). This is the measurement that decision was deferred to.

- [ ] **Step 6: Run the full preparation**

Run: `make data`
Expected: 20000 / 500 / 200 rows written. Takes a few minutes; the stream downloads only
the shards it needs.

- [ ] **Step 7: Commit**

```bash
git add scripts/01_prepare_data.py tests/test_prepare_data.py
git commit -m "feat: add data preparation with name extraction and token histogram"
```

---

## Task 4: Build llama.cpp with CUDA

**Files:**
- Create: `scripts/00_setup_llamacpp.sh`

**Interfaces:**
- Consumes: nothing.
- Produces: `vendor/llama.cpp/build/bin/{llama-server,llama-quantize,llama-perplexity}` and `vendor/llama.cpp/convert_hf_to_gguf.py`.

- [ ] **Step 1: Write `scripts/00_setup_llamacpp.sh`**

```bash
#!/usr/bin/env bash
# Stage 0 — vendor and build llama.cpp with CUDA for this machine only.
#
# Pinned to a specific commit (ADR 0011): a converter change must never silently alter
# results between runs. CMAKE_CUDA_ARCHITECTURES=86 targets the RTX 3060 alone, which
# cuts build time substantially versus compiling every architecture.
set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-vendor/llama.cpp}"
LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}"
CUDA_ARCH="${CUDA_ARCH:-86}"

if [ ! -d "$LLAMA_DIR/.git" ]; then
  echo ">> cloning llama.cpp into $LLAMA_DIR"
  git clone "$LLAMA_REPO" "$LLAMA_DIR"
fi

if [ -n "${LLAMA_COMMIT:-}" ]; then
  echo ">> checking out pinned commit $LLAMA_COMMIT"
  git -C "$LLAMA_DIR" fetch --all --quiet
  git -C "$LLAMA_DIR" checkout --quiet "$LLAMA_COMMIT"
else
  echo ">> LLAMA_COMMIT not set; using current checkout"
  echo ">> PIN IT: record the SHA below in configs/ and export LLAMA_COMMIT"
fi
echo ">> llama.cpp at $(git -C "$LLAMA_DIR" rev-parse HEAD)"

echo ">> configuring (CUDA, sm_$CUDA_ARCH)"
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DLLAMA_CURL=OFF

echo ">> building"
cmake --build "$LLAMA_DIR/build" --config Release -j "$(nproc)" \
  --target llama-server llama-quantize llama-perplexity llama-cli

for binary in llama-server llama-quantize llama-perplexity; do
  path="$LLAMA_DIR/build/bin/$binary"
  [ -x "$path" ] || { echo "FATAL: $path missing after build" >&2; exit 1; }
  echo ">> ok: $path"
done

[ -f "$LLAMA_DIR/convert_hf_to_gguf.py" ] || {
  echo "FATAL: convert_hf_to_gguf.py missing" >&2; exit 1; }

echo ">> llama.cpp ready"
```

- [ ] **Step 2: Build it**

Run: `bash scripts/00_setup_llamacpp.sh`
Expected: three `>> ok:` lines and `>> llama.cpp ready`. Takes 5-15 minutes.

- [ ] **Step 3: Verify CUDA is actually compiled in**

Run: `vendor/llama.cpp/build/bin/llama-server --version 2>&1 | head -5`
Expected: output mentioning CUDA and the RTX 3060. If it does not, the build fell back to
CPU and every later timing number will be wrong — fix it now.

- [ ] **Step 4: Pin the commit**

Run: `git -C vendor/llama.cpp rev-parse HEAD`
Record the SHA in `configs/qwen3-8b-personas.yaml` under a new `llamacpp.commit` key and
document exporting `LLAMA_COMMIT` in the README.

- [ ] **Step 5: Commit**

```bash
git add scripts/00_setup_llamacpp.sh configs/qwen3-8b-personas.yaml
git commit -m "feat: add pinned CUDA build of llama.cpp"
```

---

## Task 5: QLoRA training

**Files:**
- Create: `scripts/02_train.py`

**Interfaces:**
- Consumes: `data/{train,val}.jsonl`, `data/manifest.json`, `personas.config.load_config`.
- Produces: `outputs/adapter/` (LoRA weights + tokenizer), `outputs/train_log.json` (loss curve and final eval loss).

- [ ] **Step 1: Write `scripts/02_train.py`**

`import unsloth` must come before `transformers`/`trl` — Unsloth patches them on import
and warns loudly if it loses the race.

```python
#!/usr/bin/env python
"""Stage 2 — QLoRA fine-tune Qwen3-8B on a single RTX 3060 (ADR 0002, 0009).

Trains on the assistant response only, so the loss never rewards reproducing the prompt.
Prints a measured ETA after 50 steps: a wrong throughput estimate should surface in
minutes, not at hour four.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import unsloth  # noqa: F401  # must precede transformers/trl
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from personas.config import load_config  # noqa: E402
from personas.prompt import PROMPT_VERSION  # noqa: E402


class EtaCallback:
    """Prints a measured ETA once throughput is known."""

    def __init__(self, total_steps: int, warmup_steps: int = 50):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.start: float | None = None
        self.reported = False

    def __call__(self, step: int) -> None:
        if step == 1:
            self.start = time.time()
        elif step == self.warmup_steps and self.start and not self.reported:
            per_step = (time.time() - self.start) / (self.warmup_steps - 1)
            remaining = (self.total_steps - step) * per_step
            print(f"\n>> measured {per_step:.2f}s/step -> "
                  f"ETA {remaining / 3600:.1f}h for {self.total_steps} steps\n")
            self.reported = True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3-8b-personas.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    data_dir = Path(cfg.paths.data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["prompt_version"] != PROMPT_VERSION:
        print(f"FATAL: data was prepared with prompt version "
              f"{manifest['prompt_version']}, code is at {PROMPT_VERSION}. "
              f"Re-run `make data` (ADR 0006).", file=sys.stderr)
        return 1

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.base_id,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,
        dtype=None,  # auto-detect: bf16 on Ampere
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        target_modules=list(cfg.lora.target_modules),
        use_gradient_checkpointing=cfg.lora.use_gradient_checkpointing,
        random_state=cfg.lora.random_state,
    )

    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer

    splits = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "val.jsonl"),
        },
    )

    outputs = Path(cfg.paths.outputs_dir)
    (outputs / "adapter").mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=splits["train"],
        eval_dataset=splits["validation"],
        dataset_text_field="text",
        max_seq_length=cfg.model.max_seq_length,
        packing=False,  # response-only loss requires unpacked sequences
        args=TrainingArguments(
            output_dir=str(outputs / "checkpoints"),
            per_device_train_batch_size=cfg.train.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
            num_train_epochs=cfg.train.num_train_epochs,
            learning_rate=cfg.train.learning_rate,
            lr_scheduler_type=cfg.train.lr_scheduler_type,
            warmup_ratio=cfg.train.warmup_ratio,
            optim=cfg.train.optim,
            weight_decay=cfg.train.weight_decay,
            bf16=True,
            fp16=False,
            logging_steps=cfg.train.logging_steps,
            eval_strategy="steps",
            eval_steps=cfg.train.eval_steps,
            save_steps=cfg.train.save_steps,
            save_total_limit=cfg.train.save_total_limit,
            seed=cfg.train.seed,
            report_to="none",
        ),
    )

    # Loss on the assistant turn only. These markers are Qwen3's ChatML delimiters.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    total_steps = int(trainer.state.max_steps or 0) or len(splits["train"]) // (
        cfg.train.per_device_train_batch_size * cfg.train.gradient_accumulation_steps
    )
    print(f">> {len(splits['train'])} train rows, ~{total_steps} optimizer steps")

    import torch
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    result = trainer.train()
    elapsed = time.time() - started

    peak_gb = torch.cuda.max_memory_reserved() / 1024**3
    print(f">> finished in {elapsed / 3600:.2f}h, peak VRAM {peak_gb:.2f} GB")

    model.save_pretrained(str(outputs / "adapter"))
    tokenizer.save_pretrained(str(outputs / "adapter"))

    (outputs / "train_log.json").write_text(
        json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "base_id": cfg.model.base_id,
                "train_runtime_hours": elapsed / 3600,
                "peak_vram_gb": peak_gb,
                "final_train_loss": result.training_loss,
                "log_history": trainer.state.log_history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f">> adapter saved to {outputs / 'adapter'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Free VRAM before the real run**

Run: `nvidia-smi --query-gpu=memory.used,memory.total --format=csv`
Expected: under ~3GB used. Close browsers and heavy GUI apps first. If the run OOMs, apply
the fallback ladder in order — batch 1 with grad-accum 16, then `max_seq_length` 1536,
then stop the GNOME session.

- [ ] **Step 3: Train on the smoke split first**

Run: `make smoke && make train`
Expected: completes in a few minutes; `outputs/adapter/` exists; the printed peak VRAM is
under 9GB. This validates the whole training path cheaply.

**Read the peak VRAM number.** It is the empirical check on ADR 0002's ~7.7-8.3GB estimate.

- [ ] **Step 4: Train for real**

Run: `make data && make train`
Expected: the ETA line appears within a few minutes and should read 3-5h. If it reads
substantially more, stop and reduce `n_train` rather than discovering it at hour eight.

- [ ] **Step 5: Check that the model actually learned**

Run: `uv run python -c "import json; h=json.load(open('outputs/train_log.json'))['log_history']; e=[x for x in h if 'eval_loss' in x]; print([round(x['eval_loss'],4) for x in e])"`
Expected: a decreasing sequence. A flat or rising curve means something is wrong — check
that response-only masking applied, and inspect a decoded batch before spending more GPU
time.

- [ ] **Step 6: Commit**

```bash
git add scripts/02_train.py
git commit -m "feat: add Unsloth QLoRA training with response-only loss"
```

---

## Task 6: GGUF export for tuned and base

**Files:**
- Create: `scripts/03_export_gguf.py`

**Interfaces:**
- Consumes: `outputs/adapter/`, `vendor/llama.cpp/`.
- Produces: `outputs/gguf/personas-tuned-q4_k_m.gguf`, `outputs/gguf/personas-tuned-q8_0.gguf`, `outputs/gguf/personas-base-q4_k_m.gguf`.

- [ ] **Step 1: Write `scripts/03_export_gguf.py`**

```python
#!/usr/bin/env python
"""Stage 3 — merge, convert and quantise (ADR 0008).

Three explicit steps rather than Unsloth's one-call GGUF helper: each fails in isolation
with a legible error and can be re-run without repeating the others.

The base model goes through the IDENTICAL chain so that base-vs-tuned differs only by the
LoRA weights, not by quantisation lineage or converter version.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from personas.config import load_config  # noqa: E402

QUANTS = ("Q4_K_M", "Q8_0")


def run(command: list[str]) -> None:
    print(">>", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def merge_tuned(cfg, merged_dir: Path) -> None:
    """Merge the LoRA adapter into fp16 weights. Peak RAM ~17GB."""
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    adapter = Path(cfg.paths.outputs_dir) / "adapter"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter),
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=True,
    )
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")


def materialise_base(cfg, merged_dir: Path) -> None:
    """Write the untuned base in fp16 through the same save path as the merge."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_id, dtype=torch.float16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_id)
    model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3-8b-personas.yaml")
    parser.add_argument("--which", choices=("tuned", "base"), required=True)
    parser.add_argument("--keep-intermediates", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)

    outputs = Path(cfg.paths.outputs_dir)
    llamacpp = Path(cfg.paths.llamacpp_dir)
    gguf_dir = outputs / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = outputs / f"merged-16bit-{args.which}"
    f16_path = gguf_dir / f"personas-{args.which}-f16.gguf"

    # Step 1 — fp16 weights on disk
    if not (merged_dir / "config.json").exists():
        print(f">> materialising fp16 weights for '{args.which}' (~16GB, RAM peak ~17GB)")
        if args.which == "tuned":
            merge_tuned(cfg, merged_dir)
        else:
            materialise_base(cfg, merged_dir)
    else:
        print(f">> reusing {merged_dir}")

    # Step 2 — HF to GGUF f16
    if not f16_path.exists():
        run([
            sys.executable,
            str(llamacpp / "convert_hf_to_gguf.py"),
            str(merged_dir),
            "--outfile", str(f16_path),
            "--outtype", "f16",
        ])
    else:
        print(f">> reusing {f16_path}")

    # Step 3 — quantise
    quantize = llamacpp / "build" / "bin" / "llama-quantize"
    for quant in QUANTS:
        out = gguf_dir / f"personas-{args.which}-{quant.lower()}.gguf"
        if out.exists():
            print(f">> reusing {out}")
            continue
        run([str(quantize), str(f16_path), str(out), quant])
        size_gb = out.stat().st_size / 1024**3
        print(f">> {out.name}: {size_gb:.2f} GB")

    if not args.keep_intermediates:
        print(">> removing intermediates (re-runnable; pass --keep-intermediates to keep)")
        f16_path.unlink(missing_ok=True)

    print(">> export complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Export the tuned model**

Run: `uv run python scripts/03_export_gguf.py --which tuned`
Expected: `personas-tuned-q4_k_m.gguf` at roughly 4.7-5.1GB, plus a Q8_0 around 8.7GB.

- [ ] **Step 3: Export the base model**

Run: `uv run python scripts/03_export_gguf.py --which base`
Expected: `personas-base-q4_k_m.gguf` at a near-identical size to the tuned one. A large
size difference means the two went through different paths — investigate before evaluating.

- [ ] **Step 4: Smoke-test generation**

```bash
vendor/llama.cpp/build/bin/llama-cli \
  -m outputs/gguf/personas-tuned-q4_k_m.gguf -ngl 99 -no-cnv \
  -p "<|im_start|>user
Nome: Ana Paula dos Santos | Sexo: Feminino | Idade: 34<|im_end|>
<|im_start|>assistant
" -n 128
```
Expected: coherent Brazilian Portuguese. This only checks the file loads and generates —
real measurement comes in Task 8.

- [ ] **Step 5: Commit**

```bash
git add scripts/03_export_gguf.py
git commit -m "feat: add explicit GGUF export for tuned and base models"
```

---

## Task 7: Serving and the LangChain completion runnable

**Files:**
- Create: `scripts/04_serve.sh`, `src/personas/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `personas.config.load_config`.
- Produces: `personas.llm.LlamaCppCompletion(base_url: str, decode: object, timeout: float = 300.0)` — a LangChain `Runnable` mapping `str -> str`, with `.invoke()`, `.batch()`; and `personas.llm.wait_for_server(base_url: str, timeout: float = 300.0) -> None`.

- [ ] **Step 1: Write `scripts/04_serve.sh`**

```bash
#!/usr/bin/env bash
# Stage 4 — serve a GGUF with full GPU offload.
#
# Only ONE model at a time: two Q4_K_M 8B models exceed the ~9.5GB free VRAM (ADR 0010).
set -euo pipefail

MODEL="${1:?usage: 04_serve.sh <path-to-gguf> [port]}"
PORT="${2:-8080}"
LLAMA_DIR="${LLAMA_DIR:-vendor/llama.cpp}"

[ -f "$MODEL" ] || { echo "FATAL: $MODEL not found" >&2; exit 1; }

exec "$LLAMA_DIR/build/bin/llama-server" \
  -m "$MODEL" \
  -ngl 99 \
  -c 4096 \
  -fa on \
  --host 127.0.0.1 \
  --port "$PORT"
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_llm.py
import json

import pytest

from personas.llm import LlamaCppCompletion


class Decode:
    temperature = 0.7
    top_p = 0.8
    top_k = 20
    repeat_penalty = 1.05
    n_predict = 1024
    seed = 3407


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeClient:
    """Captures the posted body so we can assert on the exact request."""

    def __init__(self):
        self.posts = []

    def post(self, url, json=None, timeout=None):  # noqa: A002
        self.posts.append((url, json))
        return FakeResponse({"content": "## Síntese\ntexto gerado"})

    def close(self):
        return None


def test_posts_the_prompt_verbatim_to_completion():
    client = FakeClient()
    llm = LlamaCppCompletion("http://localhost:8080", Decode(), client=client)
    out = llm.invoke("<|im_start|>user\nNome: Ana<|im_end|>\n")
    assert out == "## Síntese\ntexto gerado"
    url, body = client.posts[0]
    assert url.endswith("/completion"), "must use raw /completion, never /v1 (ADR 0006)"
    assert body["prompt"] == "<|im_start|>user\nNome: Ana<|im_end|>\n"


def test_sends_the_configured_decoding_parameters():
    client = FakeClient()
    llm = LlamaCppCompletion("http://localhost:8080", Decode(), client=client)
    llm.invoke("hello")
    _, body = client.posts[0]
    assert body["temperature"] == 0.7
    assert body["top_p"] == 0.8
    assert body["top_k"] == 20
    assert body["repeat_penalty"] == 1.05
    assert body["n_predict"] == 1024
    assert body["seed"] == 3407
    assert body["cache_prompt"] is False


def test_batch_returns_one_result_per_prompt():
    client = FakeClient()
    llm = LlamaCppCompletion("http://localhost:8080", Decode(), client=client)
    results = llm.batch(["a", "b", "c"])
    assert len(results) == 3
    assert len(client.posts) == 3
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run --extra dev pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'personas.llm'`

- [ ] **Step 4: Write `src/personas/llm.py`**

```python
"""LangChain Runnable over llama-server's raw /completion endpoint (ADR 0007).

Deliberately NOT ChatOpenAI: /v1/chat/completions makes llama-server re-apply its own
Jinja template, which would hand prompt construction to the server and break the
train/inference parity guarantee in ADR 0006. Here we post our exact pre-rendered string.

In exchange for ~30 lines we keep .batch() with bounded concurrency, a retry policy, and
base-vs-tuned A/B as a single swapped base_url.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from langchain_core.runnables import Runnable, RunnableConfig

STOP = ["<|im_end|>", "<|endoftext|>"]


class LlamaCppCompletion(Runnable[str, str]):
    def __init__(self, base_url: str, decode: Any, timeout: float = 300.0,
                 max_retries: int = 3, client: Any | None = None):
        self.base_url = base_url.rstrip("/")
        self.decode = decode
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = client or httpx.Client(timeout=timeout)

    def _body(self, prompt: str) -> dict:
        return {
            "prompt": prompt,
            "temperature": self.decode.temperature,
            "top_p": self.decode.top_p,
            "top_k": self.decode.top_k,
            "repeat_penalty": self.decode.repeat_penalty,
            "n_predict": self.decode.n_predict,
            "seed": self.decode.seed,
            "stop": STOP,
            # Prompts share a long system prefix; caching it across DIFFERENT prompts
            # would make results order-dependent, so it stays off for reproducibility.
            "cache_prompt": False,
        }

    def invoke(self, input: str, config: RunnableConfig | None = None, **kwargs) -> str:  # noqa: A002
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.post(
                    f"{self.base_url}/completion",
                    json=self._body(input),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()["content"]
            except Exception as error:  # noqa: BLE001
                last_error = error
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(
            f"llama-server at {self.base_url} failed after {self.max_retries} attempts"
        ) from last_error

    def close(self) -> None:
        self._client.close()


def wait_for_server(base_url: str, timeout: float = 300.0) -> None:
    """Block until llama-server reports ready, or raise."""
    deadline = time.time() + timeout
    url = f"{base_url.rstrip('/')}/health"
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=5.0).status_code == 200:
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2.0)
    raise RuntimeError(f"llama-server at {base_url} did not become ready in {timeout}s")
```

`Runnable` supplies `.batch()` on top of `.invoke()`, so no batch implementation is needed;
concurrency is controlled by the caller via `config={"max_concurrency": N}`.

- [ ] **Step 5: Run the tests**

Run: `uv run --extra dev --extra eval pytest tests/test_llm.py -v`
Expected: 3 passed

- [ ] **Step 6: Verify against a real server**

```bash
bash scripts/04_serve.sh outputs/gguf/personas-tuned-q4_k_m.gguf 8080 &
sleep 30
curl -s http://127.0.0.1:8080/health
```
Expected: `{"status":"ok"}`. Then `kill %1` — leaving it running would block Task 8's
sequential server lifecycle.

- [ ] **Step 7: Commit**

```bash
git add scripts/04_serve.sh src/personas/llm.py tests/test_llm.py
git commit -m "feat: add llama-server launcher and LangChain completion runnable"
```

---

## Task 8: Scoring and evaluation

**Files:**
- Create: `src/personas/grounding.py`, `scripts/05_evaluate.py`
- Test: `tests/test_grounding.py`

**Interfaces:**
- Consumes: `personas.llm.LlamaCppCompletion/wait_for_server`, `personas.prompt.parse_sections/PROMPT_VERSION`, `personas.schema.SECTIONS`, `data/test.jsonl`.
- Produces: `personas.grounding.check_format(text) -> dict`, `personas.grounding.score_grounding(text, attributes) -> dict`, `personas.grounding.is_portuguese(text) -> bool`; and `outputs/eval/results.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grounding.py
from personas.grounding import check_format, is_portuguese, score_grounding
from personas.schema import SECTIONS

VALID = "\n\n".join(f"## {heading}\nconteúdo de {heading}." for heading, _ in SECTIONS)

ATTRS = {
    "name": "Marcos Antunes",
    "sex": "Masculino",
    "age": 22,
    "marital_status": "Solteiro",
    "education_level": "Fundamental completo",
    "occupation": "Operador de máquina",
    "municipality": "São Pedro de Alcântara",
    "state": "Santa Catarina",
    "country": "Brasil",
}


def test_valid_output_passes_every_format_check():
    result = check_format(VALID)
    assert result["valid"] is True
    assert result["has_all_sections"] is True
    assert result["order_correct"] is True
    assert result["think_leak"] is False


def test_missing_section_fails():
    truncated = VALID.split("## Habilidades")[0]
    result = check_format(truncated)
    assert result["valid"] is False
    assert result["has_all_sections"] is False


def test_out_of_order_sections_fail():
    headings = [h for h, _ in SECTIONS]
    swapped = [headings[1], headings[0], *headings[2:]]
    text = "\n\n".join(f"## {h}\ncorpo." for h in swapped)
    assert check_format(text)["order_correct"] is False


def test_think_leak_is_detected_and_fails_validity():
    result = check_format("<think>hmm</think>\n\n" + VALID)
    assert result["think_leak"] is True
    assert result["valid"] is False


def test_grounding_finds_mentioned_attributes():
    text = ("## Síntese\nMarcos Antunes, 22 anos, mora em São Pedro de Alcântara, "
            "Santa Catarina, e trabalha como operador de máquina.")
    scores = score_grounding(text, ATTRS)
    assert scores["name"] is True
    assert scores["municipality"] is True
    assert scores["state"] is True
    assert scores["age"] is True
    assert scores["occupation"] is True


def test_grounding_misses_absent_attributes():
    scores = score_grounding("## Síntese\nUma pessoa genérica qualquer.", ATTRS)
    assert scores["name"] is False
    assert scores["municipality"] is False
    assert scores["recall"] == 0.0


def test_portuguese_detection():
    assert is_portuguese("Ele trabalha na fábrica e mora com a família dele.") is True
    assert is_portuguese("He works at the factory and lives with his family.") is False
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --extra dev pytest tests/test_grounding.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'personas.grounding'`

- [ ] **Step 3: Write `src/personas/grounding.py`**

```python
"""Scorers for generated personas (ADR 0010).

Grounding recall is the metric that distinguishes a model conditioning on its input from
one producing fluent generic prose. It is also the one a degenerate model could game by
parroting attributes — which is why the side-by-side report exists alongside it.
"""

from __future__ import annotations

import re
import unicodedata

from personas.prompt import parse_sections
from personas.schema import SECTIONS

_EXPECTED = [heading for heading, _ in SECTIONS]

# Frequent Portuguese function words, chosen to be rare in English.
_PT_MARKERS = {
    "de", "da", "do", "dos", "das", "que", "não", "com", "uma", "para",
    "seu", "sua", "ele", "ela", "mais", "como", "por", "em", "no", "na",
}

_GROUNDED_FIELDS = ("name", "municipality", "state", "occupation", "age")


def _normalise(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def check_format(text: str) -> dict:
    """Six sections, right order, no thinking-mode leakage."""
    think_leak = "<think>" in text and "<think>\n\n</think>" not in text
    found = list(parse_sections(text))
    has_all = all(heading in found for heading in _EXPECTED)
    order_correct = [h for h in found if h in _EXPECTED] == _EXPECTED
    no_extra = all(heading in _EXPECTED for heading in found)
    return {
        "has_all_sections": has_all,
        "order_correct": order_correct,
        "no_extra_sections": no_extra,
        "think_leak": think_leak,
        "n_sections": len(found),
        "valid": bool(has_all and order_correct and no_extra and not think_leak),
    }


def score_grounding(text: str, attributes: dict) -> dict:
    """Per-field: does the generation actually mention this input attribute?"""
    haystack = _normalise(text)
    scores: dict[str, object] = {}

    scores["name"] = _normalise(str(attributes["name"])) in haystack
    scores["municipality"] = _normalise(str(attributes["municipality"])) in haystack
    scores["state"] = _normalise(str(attributes["state"])) in haystack
    scores["age"] = bool(re.search(rf"\b{int(attributes['age'])}\b", text))

    # Occupation strings are long and formal ("Operador de instalação ou máquina ou
    # montador"), so require overlap on content words rather than an exact match.
    content_words = [
        word for word in _normalise(str(attributes["occupation"])).split()
        if len(word) > 3 and word not in {"para", "como", "ou"}
    ]
    scores["occupation"] = bool(content_words) and any(
        word in haystack for word in content_words
    )

    hits = sum(1 for field in _GROUNDED_FIELDS if scores[field])
    scores["recall"] = hits / len(_GROUNDED_FIELDS)
    return scores


def is_portuguese(text: str, threshold: float = 0.08) -> bool:
    """Heuristic: share of tokens that are common Portuguese function words."""
    words = re.findall(r"\w+", text.lower())
    if not words:
        return False
    return sum(1 for word in words if word in _PT_MARKERS) / len(words) >= threshold
```

- [ ] **Step 4: Run the tests**

Run: `uv run --extra dev pytest tests/test_grounding.py -v`
Expected: 7 passed

- [ ] **Step 5: Write `scripts/05_evaluate.py`**

```python
#!/usr/bin/env python
"""Stage 5 — generate over the test split with base and tuned, then score (ADR 0010).

Runs the two models SEQUENTIALLY. Two Q4_K_M 8B models do not fit in ~9.5GB of free
VRAM, so this script owns server lifecycle: start, wait, generate, tear down, repeat.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from personas.grounding import check_format, is_portuguese, score_grounding  # noqa: E402
from personas.config import load_config  # noqa: E402
from personas.llm import LlamaCppCompletion, wait_for_server  # noqa: E402
from personas.prompt import PROMPT_VERSION  # noqa: E402

PORT = 8080
BASE_URL = f"http://127.0.0.1:{PORT}"


def load_test_rows(data_dir: Path) -> list[dict]:
    with (data_dir / "test.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def generate_with(model_path: Path, prompts: list[str], cfg, concurrency: int = 2):
    """Start llama-server, generate, always tear it down."""
    print(f">> starting llama-server for {model_path.name}")
    server = subprocess.Popen(
        ["bash", "scripts/04_serve.sh", str(model_path), str(PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_server(BASE_URL)
        print(f">> ready; generating {len(prompts)} completions")
        llm = LlamaCppCompletion(BASE_URL, cfg.decode)
        started = time.time()
        outputs = llm.batch(prompts, config={"max_concurrency": concurrency})
        print(f">> done in {time.time() - started:.0f}s")
        llm.close()
        return outputs
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        print(">> server stopped")
        time.sleep(5)  # let the driver release VRAM before the next model loads


def score(rows: list[dict], generations: list[str]) -> dict:
    per_row = []
    for row, generated in zip(rows, generations):
        fmt = check_format(generated)
        grounding = score_grounding(generated, row["attributes"])
        per_row.append({
            "uuid": row["uuid"],
            "format": fmt,
            "grounding": grounding,
            "portuguese": is_portuguese(generated),
            "chars": len(generated),
        })
    return {
        "format_validity": mean(r["format"]["valid"] for r in per_row),
        "think_leak_rate": mean(r["format"]["think_leak"] for r in per_row),
        "grounding_recall": mean(r["grounding"]["recall"] for r in per_row),
        "grounding_by_field": {
            field: mean(r["grounding"][field] for r in per_row)
            for field in ("name", "municipality", "state", "occupation", "age")
        },
        "portuguese_rate": mean(r["portuguese"] for r in per_row),
        "mean_chars": mean(r["chars"] for r in per_row),
        "per_row": per_row,
    }


def perplexity(model_path: Path, corpus: Path, cfg) -> float | None:
    """Held-out perplexity via llama-perplexity. Returns None if it fails."""
    binary = Path(cfg.paths.llamacpp_dir) / "build" / "bin" / "llama-perplexity"
    try:
        result = subprocess.run(
            [str(binary), "-m", str(model_path), "-f", str(corpus),
             "-ngl", "99", "-c", "2048", "--chunks", "40"],
            capture_output=True, text=True, timeout=3600, check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(f">> perplexity failed for {model_path.name}: {error}", file=sys.stderr)
        return None
    import re
    matches = re.findall(r"Final estimate: PPL = ([\d.]+)", result.stdout + result.stderr)
    return float(matches[-1]) if matches else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3-8b-personas.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    data_dir = Path(cfg.paths.data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["prompt_version"] != PROMPT_VERSION:
        print(f"FATAL: prompt version mismatch — data {manifest['prompt_version']}, "
              f"code {PROMPT_VERSION} (ADR 0006)", file=sys.stderr)
        return 1

    rows = load_test_rows(data_dir)
    if args.limit:
        rows = rows[:args.limit]
    prompts = [row["prompt"] for row in rows]
    print(f">> evaluating on {len(rows)} held-out rows")

    gguf_dir = Path(cfg.paths.outputs_dir) / "gguf"
    models = {
        "base": gguf_dir / "personas-base-q4_k_m.gguf",
        "tuned": gguf_dir / "personas-tuned-q4_k_m.gguf",
    }
    for name, path in models.items():
        if not path.exists():
            print(f"FATAL: {name} model missing at {path}. Run `make export`.",
                  file=sys.stderr)
            return 1

    # Reference corpus for perplexity: the held-out targets.
    eval_dir = Path(cfg.paths.outputs_dir) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    corpus = eval_dir / "test_corpus.txt"
    corpus.write_text("\n\n".join(row["target"] for row in rows), encoding="utf-8")

    results: dict = {"prompt_version": PROMPT_VERSION, "n_rows": len(rows), "models": {}}
    generations: dict[str, list[str]] = {}

    for name, path in models.items():  # sequential — see module docstring
        outputs = generate_with(path, prompts, cfg)
        generations[name] = outputs
        results["models"][name] = score(rows, outputs)
        results["models"][name]["perplexity"] = perplexity(path, corpus, cfg)

    (eval_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (eval_dir / "generations.json").write_text(
        json.dumps(
            [
                {
                    "uuid": row["uuid"],
                    "attributes": row["attributes"],
                    "reference": row["target"],
                    "base": generations["base"][i],
                    "tuned": generations["tuned"][i],
                }
                for i, row in enumerate(rows)
            ],
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== RESULTS ===")
    header = f"{'metric':<24}{'base':>12}{'tuned':>12}"
    print(header)
    print("-" * len(header))
    for metric in ("perplexity", "format_validity", "grounding_recall",
                   "think_leak_rate", "portuguese_rate", "mean_chars"):
        base_value = results["models"]["base"][metric]
        tuned_value = results["models"]["tuned"][metric]
        fmt = lambda v: "n/a" if v is None else f"{v:.4f}"  # noqa: E731
        print(f"{metric:<24}{fmt(base_value):>12}{fmt(tuned_value):>12}")
    print(f"\nwrote {eval_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run evaluation on a few rows first**

Run: `uv run python scripts/05_evaluate.py --limit 5`
Expected: both servers start and stop cleanly; a results table prints. Watch
`nvidia-smi` in another terminal to confirm only one model is resident at a time.

- [ ] **Step 7: Run the full evaluation**

Run: `make eval`
Expected: tuned beats base on format validity and grounding recall, with lower
perplexity. `think_leak_rate` should be 0.0000 for the tuned model — anything above zero
means ADR 0006 was violated somewhere.

- [ ] **Step 8: Commit**

```bash
git add src/personas/grounding.py scripts/05_evaluate.py tests/test_grounding.py
git commit -m "feat: add grounding scorers and sequential base-vs-tuned evaluation"
```

---

## Task 9: Side-by-side report and README

**Files:**
- Create: `scripts/06_report.py`, `README.md`

**Interfaces:**
- Consumes: `outputs/eval/results.json`, `outputs/eval/generations.json`, `outputs/train_log.json`.
- Produces: `outputs/eval/report.html`.

- [ ] **Step 1: Write `scripts/06_report.py`**

```python
#!/usr/bin/env python
"""Stage 6 — render the base-vs-tuned comparison as a standalone HTML page.

The metrics table answers "did fine-tuning help?"; the three-column comparison is the
guard against trusting a number that is technically true and substantively wrong.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from personas.config import load_config  # noqa: E402

N_EXAMPLES = 30

STYLE = """
:root { --bg:#fbfbfa; --fg:#1a1a19; --muted:#6b6b68; --line:#e3e3e0;
        --good:#1a7f4b; --bad:#b4341f; --card:#fff; }
@media (prefers-color-scheme: dark) { :root:not([data-theme=light]) {
  --bg:#16161a; --fg:#eceae6; --muted:#9a9a95; --line:#2e2e33; --card:#1e1e23; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.6
  ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
.wrap { max-width:1400px; margin:0 auto; padding:40px 24px 80px; }
h1 { font-size:26px; margin:0 0 4px; letter-spacing:-.02em; }
.sub { color:var(--muted); margin:0 0 32px; }
table { width:100%; border-collapse:collapse; margin-bottom:40px; }
th,td { text-align:left; padding:10px 14px; border-bottom:1px solid var(--line); }
th { font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
.win { color:var(--good); font-weight:600; } .lose { color:var(--bad); }
.ex { background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:18px; margin-bottom:20px; }
.attrs { font-size:13px; color:var(--muted); margin-bottom:14px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; white-space:pre-wrap; }
.cols { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }
@media (max-width:900px) { .cols { grid-template-columns:1fr; } }
.col h4 { margin:0 0 8px; font-size:12px; text-transform:uppercase;
  letter-spacing:.06em; color:var(--muted); }
.col div.body { font-size:13px; white-space:pre-wrap; max-height:380px;
  overflow-y:auto; border-left:2px solid var(--line); padding-left:12px; }
"""


def metric_row(label: str, base, tuned, lower_is_better: bool = False) -> str:
    def cell(value, other):
        if value is None:
            return '<td class="num">n/a</td>'
        if other is None:
            return f'<td class="num">{value:.4f}</td>'
        better = value < other if lower_is_better else value > other
        css = "win" if better else ("lose" if value != other else "")
        return f'<td class="num {css}">{value:.4f}</td>'

    return (f"<tr><td>{html.escape(label)}</td>"
            f"{cell(base, tuned)}{cell(tuned, base)}</tr>")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3-8b-personas.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    eval_dir = Path(cfg.paths.outputs_dir) / "eval"
    results = json.loads((eval_dir / "results.json").read_text(encoding="utf-8"))
    generations = json.loads((eval_dir / "generations.json").read_text(encoding="utf-8"))

    base, tuned = results["models"]["base"], results["models"]["tuned"]
    rows = [
        metric_row("Perplexity (held-out)", base["perplexity"], tuned["perplexity"], True),
        metric_row("Format validity", base["format_validity"], tuned["format_validity"]),
        metric_row("Grounding recall", base["grounding_recall"], tuned["grounding_recall"]),
        metric_row("Think leak rate", base["think_leak_rate"], tuned["think_leak_rate"], True),
        metric_row("Portuguese rate", base["portuguese_rate"], tuned["portuguese_rate"]),
    ]
    for field in ("name", "municipality", "state", "occupation", "age"):
        rows.append(metric_row(
            f"  grounding: {field}",
            base["grounding_by_field"][field],
            tuned["grounding_by_field"][field],
        ))

    examples = []
    for item in generations[:N_EXAMPLES]:
        attrs = item["attributes"]
        header = (f"{attrs['name']} | {attrs['sex']} | {attrs['age']} | "
                  f"{attrs['occupation']} | {attrs['municipality']}, {attrs['state']}")
        columns = "".join(
            f'<div class="col"><h4>{title}</h4>'
            f'<div class="body">{html.escape(item[key])}</div></div>'
            for title, key in (("Ground truth", "reference"),
                               ("Base", "base"), ("Fine-tuned", "tuned"))
        )
        examples.append(
            f'<div class="ex"><div class="attrs">{html.escape(header)}</div>'
            f'<div class="cols">{columns}</div></div>'
        )

    page = f"""<title>Persona Fine-Tune Results</title>
<style>{STYLE}</style>
<div class="wrap">
<h1>Qwen3-8B persona fine-tune — base vs tuned</h1>
<p class="sub">{results['n_rows']} held-out rows · prompt version
{html.escape(results['prompt_version'])} · Q4_K_M on RTX 3060 · identical decoding
parameters</p>
<table>
<thead><tr><th>Metric</th><th style="text-align:right">Base</th>
<th style="text-align:right">Fine-tuned</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
<h1>Side-by-side generations</h1>
<p class="sub">First {min(N_EXAMPLES, len(generations))} held-out attribute sets.</p>
{''.join(examples)}
</div>"""

    out = eval_dir / "report.html"
    out.write_text(page, encoding="utf-8")
    print(f">> wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Generate the report**

Run: `make report`
Expected: `outputs/eval/report.html` written. Open it and **actually read several
comparisons** — this is the check that the metrics are measuring something real.

- [ ] **Step 3: Write `README.md`**

```markdown
# Brazilian persona generator — Qwen3-8B QLoRA on one RTX 3060

Fine-tunes Qwen3-8B with Unsloth QLoRA on `nvidia/Nemotron-Personas-Brazil`, exports to
GGUF, and evaluates against the base model through llama.cpp. Everything runs on a single
12GB GPU.

**Task:** demographic attributes in, six-section Brazilian-Portuguese persona out.

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
NVIDIA driver 535+ · CUDA toolkit for the llama.cpp build.

Host Python is not used — `uv` manages a project-local 3.12 environment.

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
- `docs/adr/README.md` — why each decision was made
- `CLAUDE.md` — invariants that must not be broken

## Data

`nvidia/Nemotron-Personas-Brazil` is CC-BY-4.0 and fully synthetic. No real PII.
```

- [ ] **Step 4: Run the whole test suite**

Run: `make test`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/06_report.py README.md
git commit -m "feat: add side-by-side HTML report and README"
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: §3 environment → Task 1 and Task 4;
§2 data → Task 3; §5 layout → Tasks 1-2; §6 training config → Task 5; §7 export/serving →
Tasks 6-7; §7 evaluation → Task 8; the report → Task 9. The ADR invariants each have an
enforcing mechanism: prompt parity is a test (Task 2) *and* a runtime assertion in data
prep (Task 3); `PROMPT_VERSION` is checked in Tasks 5 and 8; sequential serving is
structural in Task 8's `generate_with`; the shared quantisation lineage is Task 6's
single code path.

**Placeholders.** None. Every code step carries runnable code; every run step names the
exact command and expected output.

**Type consistency.** `PersonaAttributes` is constructed only via `from_row` and
serialised via `__dict__`, which is what `score_grounding` and the report read.
`SECTIONS` is the single ordering used by `render_target`, `parse_sections` and
`check_format`. `LlamaCppCompletion` takes the `cfg.decode` namespace directly, so
decoding parameters cannot drift between base and tuned.

**Known risks carried forward.**
1. The pinned dependency versions in Task 1 may not resolve together. Task 1 Step 1 says
   to relax the Unsloth/trl/transformers trio as a unit and re-lock, never to unpin.
2. `train_on_responses_only` markers are Qwen3-specific. If the pinned Unsloth version
   changes its signature, Task 5 Step 5's decreasing eval-loss check is what catches it.
3. `llama-perplexity` on a plain-text corpus measures continuation perplexity over
   concatenated targets, not conditional perplexity given each prompt. It is directionally
   valid for comparing two models on identical input, and Task 8 degrades to `None`
   rather than failing if the binary errors.
