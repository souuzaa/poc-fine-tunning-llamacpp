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
    messages = _messages(attrs) + [{"role": "assistant", "content": render_target(row)}]
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
