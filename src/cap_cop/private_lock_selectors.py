"""Resolve governed historical lock selectors without exposing them in public code."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_MAP = PROJECT_ROOT / "configs" / "private_locked_pair_ids.json"


def get_selector(name: str) -> str:
    """Return a local lock selector or fail clearly when governed artifacts are absent."""
    if not PRIVATE_MAP.is_file():
        raise RuntimeError(
            "This operation compares private historical locks and requires the governed "
            f"selector file: {PRIVATE_MAP}"
        )
    mapping = json.loads(PRIVATE_MAP.read_text(encoding="utf-8"))
    try:
        return str(mapping[name])
    except KeyError as exc:
        raise KeyError(f"No governed lock selector named {name!r}") from exc
