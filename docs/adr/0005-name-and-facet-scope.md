# ADR 0005 — Inject the name as an input attribute; drop the lifestyle facets

- **Status:** Accepted
- **Date:** 2026-09-12

## Context

Two problems surfaced when mapping the dataset schema onto the task in ADR 0004.

**1. There is no `name` column.** The person's name exists only *inside* the narrative
text ("Marcos Antunes é um operador técnico..."). If the model receives only
demographics, it must invent a name — and then every sentence of the generated narrative
disagrees with the reference for a reason that has nothing to do with model quality.
Grounding metrics and side-by-side reading both become noise.

**2. Ten narrative columns is too much output.** The dataset also carries
`sports_persona`, `arts_persona`, `travel_persona`, `culinary_persona` and two
string-encoded `*_list` columns. Including everything pushes targets past ~1500 tokens.

## Decision

**Extract the name** from the `persona` column during data prep with a leading-proper-noun
regex, and pass it as the first input attribute. Rows where extraction fails are dropped
and counted — the drop rate is reported, and prep fails loudly if it exceeds 5%.

**Target the six substantive sections only** (ADR 0004). Drop the four lifestyle facets
and both `*_list` columns.

## Consequences

- The task becomes well-posed: name, demographics and location are all given, so the
  model is scored on narrative quality, not on guessing a proper noun.
- Targets stay near ~2048 total tokens, so `max_seq_length=2048` truncates few examples
  and training time roughly halves versus the ten-section variant.
- Enables an exact-match grounding check on the name — a cheap, unambiguous signal that
  the model is conditioning on its input at all.
- The lifestyle facets are the most formulaic content in the dataset; excluding them
  costs little signal. They remain available for a follow-up run.
- Data prep must print the real token-length histogram so this can be revisited with
  measurements rather than the estimate above.

## Alternatives considered

- **Let the model invent names.** Free, but destroys reference-based evaluation.
- **Use the `uuid` as an identifier.** Grounds nothing — the narratives never mention it.
- **Keep all ten sections.** Roughly doubles training time and forces truncation at
  seq 2048, for the least informative columns in the dataset.
