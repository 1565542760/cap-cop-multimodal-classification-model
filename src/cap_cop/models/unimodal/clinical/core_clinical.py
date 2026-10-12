"""Semantic public interface for the locked core clinical branch.

The implementation is shared with the validated full/core availability-gated
replay module.  This file intentionally contains no alternate preprocessing or
model definition: it exposes the verified feature contract and prediction
function under the public semantic model name.
"""

from __future__ import annotations

from cap_cop.clinical.core_and_availability_gated_model import (
    CORE_FEATURES,
    FrameCleaner,
    load_locked_artifact,
    predict_core,
)


MODEL_NAME = "core_clinical"

__all__ = [
    "MODEL_NAME",
    "CORE_FEATURES",
    "FrameCleaner",
    "load_locked_artifact",
    "predict_core",
]
