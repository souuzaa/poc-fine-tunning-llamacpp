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

REQUIRED_SECTIONS = ("model", "data", "lora", "train", "decode", "paths")


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
    for section in REQUIRED_SECTIONS:
        if section not in raw:
            raise ValueError(f"config at {path} is missing the '{section}' section")
    return _namespacify(raw)
