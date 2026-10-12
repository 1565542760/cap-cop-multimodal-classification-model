"""Semantic public interface for the locked availability-gated branch.

The gate and both ensemble routes are implemented in the validated shared
clinical replay module.  Re-exporting that implementation keeps the public
model surface semantic without creating a second, potentially divergent
definition of the clinical contract.
"""

from __future__ import annotations

from cap_cop.clinical.core_and_availability_gated_model import (
    AVAILABILITY_THRESHOLD,
    CORE_FEATURES,
    FULL_FEATURES,
    FULL_ONLY_FEATURES,
    FrameCleaner,
    load_locked_artifact,
    predict_availability_gated,
)


MODEL_NAME = "availability_gated_clinical"

__all__ = [
    "MODEL_NAME",
    "AVAILABILITY_THRESHOLD",
    "CORE_FEATURES",
    "FULL_FEATURES",
    "FULL_ONLY_FEATURES",
    "FrameCleaner",
    "load_locked_artifact",
    "predict_availability_gated",
]
