# ADR 0004 — Task formulation: attributes to persona narrative

- **Status:** Accepted
- **Date:** 2026-09-12

## Context

`nvidia/Nemotron-Personas-Brazil` (1M rows, 21 columns, 2.5GB parquet, CC-BY-4.0) pairs
structured demographic attributes with several free-text pt-BR narratives. Four task
framings were considered: persona role-play chat, attributes-to-narrative generation,
narrative-to-JSON extraction, and a bare pipeline smoke test.

## Decision

**Attributes to persona narrative.** Given a structured attribute block, generate a
fixed-section pt-BR markdown document.

**Input** (user turn):

```
Nome: Marcos Antunes | Sexo: Masculino | Idade: 22 | Estado civil: Solteiro
Escolaridade: Fundamental completo e médio incompleto
Ocupação: Operador de instalação ou máquina ou montador
Município: São Pedro de Alcântara | Estado: Santa Catarina
```

**Output** — six sections, always this order:

| Section heading | Source column |
|---|---|
| `## Síntese` | `persona` |
| `## Perfil profissional` | `professional_persona` |
| `## Origem cultural` | `cultural_background` |
| `## Habilidades` | `skills_and_expertise` |
| `## Hobbies e interesses` | `hobbies_and_interests` |
| `## Objetivos de carreira` | `career_goals_and_ambitions` |

A single fixed system prompt is used, identical at training and inference time.

## Consequences

- The dataset maps onto the task with no synthetic relabelling — every training pair is
  ground truth straight from the source columns.
- A held-out split gives real reference text, so evaluation can be objective rather than
  vibes-based (ADR 0010).
- Fixed section order makes the output trivially parseable, enabling per-section
  scoring and a hard format-validity metric.
- The model learns a rigid output shape. This is a deliberate PoC trade: it makes success
  measurable. A conversational persona model would be a different project.

## Alternatives considered

- **Persona role-play chat.** The most engaging demo, but there is no ground truth for a
  generated conversation, so "did fine-tuning help?" becomes unanswerable without a
  human study or a judge model.
- **Narrative to JSON.** Cleanest metrics of all, but it trains extraction rather than
  pt-BR generation — it would barely exercise the language ability that motivated the
  base-model choice.
- **Smoke test only.** Rejected: proves the toolchain, produces nothing worth keeping.
