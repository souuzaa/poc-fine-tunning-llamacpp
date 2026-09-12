"""Extracts the person's name from the narrative text.

The dataset has no `name` column -- the name lives inside the prose (ADR 0005). Without
it the model must invent a name and every reference-based comparison becomes noise.
"""

from __future__ import annotations

import re

# A capitalised token, allowing Portuguese accents, hyphens and apostrophes.
_TOKEN = r"[A-ZÁÀÂÃÉÊÍÓÔÕÚÜÇ][a-zàáâãéêíóôõúüç'\-]+"
# Lowercase connectors that appear inside Brazilian names: "dos Santos", "da Silva".
_CONNECTOR = r"(?:d[aeo]s?|e)"

_NAME_RE = re.compile(rf"^({_TOKEN}(?:\s+(?:{_CONNECTOR}\s+)?{_TOKEN}){{1,3}})\b")

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
