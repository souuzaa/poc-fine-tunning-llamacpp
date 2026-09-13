"""Scorers for generated personas (ADR 0010).

Grounding recall is the metric that distinguishes a model conditioning on its input from
one producing fluent generic prose. It is also the one a degenerate model could game by
parroting attributes -- which is why the side-by-side report exists alongside it.
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
