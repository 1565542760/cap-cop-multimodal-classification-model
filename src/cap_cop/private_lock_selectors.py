"""Resolve model-artifact identifiers; this mapping contains no patient identifiers."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SELECTOR_MAP = PROJECT_ROOT / "configs" / "artifact_selectors.json"


def load_selectors(path: Path | None = None) -> dict[str, str]:
    path = Path(path) if path is not None else SELECTOR_MAP
    if not path.is_file():
        raise RuntimeError(
            f"Model-artifact selector configuration is missing: {path}"
        )
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or not mapping or any(
        not isinstance(k, str) or not isinstance(v, str) or not v.strip()
        for k, v in mapping.items()
    ):
        raise ValueError("Artifact selectors must be non-empty string-to-string entries")
    return mapping


def get_selector(name: str) -> str:
    mapping = load_selectors()
    try:
        return mapping[name]
    except KeyError as exc:
        raise KeyError(f"No governed lock selector named {name!r}") from exc
