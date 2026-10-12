"""Semantic interface for the exploratory structural-missingness ensemble.

The deployment artifact is intentionally exposed as an exploratory, locked
15-member ensemble. Its lock explicitly says that the strict procedure is not
qualified and that the artifact does not supersede the main clinical branch.
This module therefore makes the verified replay callable without presenting it
as a newly re-derived primary clinical contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cap_cop.clinical.full_structured_model import FEATURES as FULL_FEATURES
from cap_cop.clinical.structural_missingness_model import (
    PermanentImputerIndicator,
    RFStructuralModel,
    load_locked_artifact,
)


MODEL_NAME = "structural_missingness_clinical"
FEATURES = tuple(FULL_FEATURES)
INDICATOR_FEATURE_COUNT = len(FEATURES) * 2
ENSEMBLE_MEMBER_COUNT = 15
STATUS = "exploratory_locked_15_member_artifact_replay"
STRICT_PROCEDURE_STATUS = "CLINICAL_V2B_STRICT_PROCEDURE_NOT_QUALIFIED"


def predict_equal_weight_ensemble(
    model_paths: list[str | Path], frame: pd.DataFrame
) -> np.ndarray:
    """Return the equal-weight probability mean from the locked members."""
    if len(model_paths) != ENSEMBLE_MEMBER_COUNT:
        raise ValueError(
            f"Expected {ENSEMBLE_MEMBER_COUNT} locked members, got {len(model_paths)}"
        )
    probabilities = [
        load_locked_artifact(path).predict_proba(frame) for path in model_paths
    ]
    return np.mean(np.vstack(probabilities), axis=0)


__all__ = [
    "MODEL_NAME",
    "FEATURES",
    "INDICATOR_FEATURE_COUNT",
    "ENSEMBLE_MEMBER_COUNT",
    "STATUS",
    "STRICT_PROCEDURE_STATUS",
    "PermanentImputerIndicator",
    "RFStructuralModel",
    "load_locked_artifact",
    "predict_equal_weight_ensemble",
]
