"""Semantic interface for the locked alternate clinical configuration.

The exact feature contract is read from the local lock file. The associated
replay uses the locked outer-fold family-selection registry and keeps the
heterogeneous selected-family composition explicit.
"""

from __future__ import annotations

import json
from pathlib import Path

from cap_cop.paths import project_root


MODEL_NAME = "alternate_clinical_configuration"
DEFAULT_FEATURES = (
    "age",
    "height_cm",
    "weight_kg",
    "body_mass_index",
    "fever",
    "cough",
    "sputum_category",
    "chest_tightness",
    "dyspnoea",
    "haemoptysis",
    "weight_loss",
    "lymphocyte_count",
    "erythrocyte_sedimentation_rate",
    "c_reactive_protein",
    "high_sensitivity_c_reactive_protein",
    "procalcitonin",
)
LOCK_PATH = (
    project_root()
    / "models"
    / "clinical"
    / "alternate_configuration"
    / "clinical_model_lock.json"
)


def load_contract() -> dict:
    if LOCK_PATH.is_file():
        return json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    return {
        "exact_features": list(DEFAULT_FEATURES),
        "artifact_status": "governed_lock_artifact_required_for_prediction",
    }


def feature_names() -> tuple[str, ...]:
    return tuple(str(value) for value in load_contract()["exact_features"])


FEATURES = feature_names()
STATUS = "locked_strict_family_replay_available"
SELECTION_BOUNDARY = (
    "Outer-fold selected families are retained from the lock; this is not a single-family RF claim."
)


__all__ = [
    "MODEL_NAME",
    "DEFAULT_FEATURES",
    "LOCK_PATH",
    "FEATURES",
    "STATUS",
    "SELECTION_BOUNDARY",
    "load_contract",
    "feature_names",
]
