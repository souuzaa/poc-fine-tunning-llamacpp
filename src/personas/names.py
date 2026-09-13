"""Extracts the person's name from the narrative text.

The dataset has no `name` column -- the name lives inside the prose (ADR 0005). Without
it the model must invent a name and every reference-based comparison becomes noise.

Three shapes appear in the corpus, in rough order of frequency:

1. `"Marcos Antunes é um operador..."`   -- a full name opens the sentence.
2. `"Irani é um operador logístico..."`  -- a mononym opens the sentence.
3. `"Mestre de obras calmo, ..., Carlos une..."` or `"Um bombeiro militar metódico..."`
   -- the name is buried mid-sentence or absent from this column entirely.

Cases 2 and 3 are resolved by reading the other narrative columns. A real name repeats
across the facets ("Marcos cresceu...", "Marcos possui..."); an adjective that happens to
open one sentence does not. That repetition is what confirms a single-token name.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

# A capitalised token, allowing Portuguese accents, hyphens and apostrophes.
_TOKEN = r"[A-ZÁÀÂÃÉÊÍÓÔÕÚÜÇ][a-zàáâãéêíóôõúüç'\-]+"
# Lowercase connectors that appear inside Brazilian names: "dos Santos", "da Silva".
_CONNECTOR = r"(?:d[aeo]s?|e)"

# Two to four tokens: a full name.
_MULTI_RE = re.compile(rf"^({_TOKEN}(?:\s+(?:{_CONNECTOR}\s+)?{_TOKEN}){{1,3}})\b")
# Exactly one token, followed by something that is not another capitalised token.
_SINGLE_RE = re.compile(rf"^({_TOKEN})\b")

# Capitalised words that open a sentence without being names.
_STOPWORDS = {
    "Ele", "Ela", "Eles", "Elas", "Uma", "Um", "Esse", "Essa", "Este", "Esta",
    "Trabalha", "Nascido", "Nascida", "Natural", "Aos", "Com", "Nas", "Nos",
    "Como", "Depois", "Desde", "Durante", "Entre", "Sempre", "Atualmente",
}

# How many columns must open with the same single token before it counts as a name.
_MONONYM_CONFIRMATIONS = 2


def _multi_token_name(text: str) -> str | None:
    match = _MULTI_RE.match(text.strip())
    if match is None:
        return None
    name = match.group(1)
    if name.split()[0] in _STOPWORDS:
        return None
    return name


def _single_token_name(text: str) -> str | None:
    match = _SINGLE_RE.match(text.strip())
    if match is None:
        return None
    name = match.group(1)
    if name in _STOPWORDS:
        return None
    return name


def extract_name(persona_text: str, other_texts: Iterable[str] = ()) -> str | None:
    """Return the person's name, or None if no column yields one.

    `other_texts` are the remaining narrative columns for the same row. They are used
    both as fallbacks and to confirm single-token names.
    """
    texts = [t.strip() for t in (persona_text or "", *other_texts) if t and t.strip()]
    if not texts:
        return None

    # A full name anywhere is the strongest signal -- take the first one.
    for text in texts:
        name = _multi_token_name(text)
        if name is not None:
            return name

    # Otherwise accept a mononym, but only if several columns agree on it.
    leading = Counter(
        name for name in (_single_token_name(text) for text in texts) if name
    )
    if leading:
        candidate, count = leading.most_common(1)[0]
        if count >= _MONONYM_CONFIRMATIONS:
            return candidate
    return None
