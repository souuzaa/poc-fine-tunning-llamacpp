# ADR 0001 — Record architecture decisions

- **Status:** Accepted
- **Date:** 2026-09-12

## Context

This PoC fine-tunes an 8B model on synthetic Brazilian persona data and validates it
through llama.cpp. Most of the decisions (base model, quantisation strategy, task
formulation, tooling) are one-way-ish doors: reversing them mid-project costs hours of
GPU time. A future reader — including us in three weeks — needs to know *why* the 3060's
12GB ruled certain options out, not just what was chosen.

## Decision

We record every consequential decision as a numbered ADR in `docs/adr/`, using this
lightweight format: Context, Decision, Consequences, Alternatives considered.

ADRs are immutable once Accepted. A reversal is a new ADR that supersedes the old one;
the old file gains a `Superseded by ADR-XXXX` line and stays in the repo.

## Consequences

- Decisions are reviewable in isolation, without reading the full design spec.
- `CLAUDE.md` stays a short index and pointer file rather than growing into a changelog.
- Small overhead per decision, paid back the first time a hyperparameter choice is questioned.

## Alternatives considered

- **Comments in code.** Invisible to anyone planning work; lost when code is refactored.
- **One monolithic design doc.** Already exists (`docs/superpowers/specs/`), but it
  describes the target state, not the reasoning or the rejected branches.
