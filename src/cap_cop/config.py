from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import resolve_project_path


@dataclass(frozen=True)
class ModelConfig:
    name: str
    paper_role: str
    clinical_strategy: str
    imaging_strategy: str
    clinical_weight: float
    imaging_weight: float
    threshold: float
    status: str
    blocker: str | None = None


def load_config(path: str | Path) -> dict[str, Any]:
    resolved = resolve_project_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_model_configs(path: str | Path = "configs/models.json") -> list[ModelConfig]:
    data = load_config(path)
    return [ModelConfig(**item) for item in data["models"]]
