from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


FULL_FEATURES = [
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
]
CORE_FEATURES = [
    "age",
    "body_mass_index",
    "fever",
    "cough",
    "sputum_category",
    "weight_loss",
    "lymphocyte_count",
]
FULL_ONLY_FEATURES = [
    "height_cm",
    "weight_kg",
    "chest_tightness",
    "dyspnoea",
    "haemoptysis",
    "erythrocyte_sedimentation_rate",
    "c_reactive_protein",
    "high_sensitivity_c_reactive_protein",
    "procalcitonin",
]
AVAILABILITY_THRESHOLD = 6


class FrameCleaner(BaseEstimator, TransformerMixin):
    """Byte-compatible copy of the cleaner used by the locked pipelines."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for column in ["height_cm", "weight_kg", "body_mass_index"]:
            if column in X.columns:
                X[column] = pd.to_numeric(X[column], errors="coerce")
                X.loc[X[column] == 0, column] = np.nan
        if {"height_cm", "weight_kg", "body_mass_index"} <= set(X.columns):
            valid_bmi = (
                X["body_mass_index"].isna()
                & X["height_cm"].notna()
                & X["weight_kg"].notna()
                & (X["height_cm"] > 0)
            )
            X.loc[valid_bmi, "body_mass_index"] = X.loc[valid_bmi, "weight_kg"] / (
                (X.loc[valid_bmi, "height_cm"] / 100.0) ** 2
            )
        return X


def load_locked_artifact(path: str | Path) -> dict:
    # These locked files were serialized from a script entry point, so pickle
    # recorded FrameCleaner under __main__. Re-register the exact class before
    # loading; this changes no fitted parameter or prediction logic.
    main_module = sys.modules["__main__"]
    if not hasattr(main_module, "FrameCleaner"):
        setattr(main_module, "FrameCleaner", FrameCleaner)
    return joblib.load(path)


def _member_mean(artifact: dict, frame: pd.DataFrame) -> np.ndarray:
    features = list(artifact["features"])
    probabilities = [
        member.predict_proba(frame[features])[:, 1]
        for member in artifact["members"].values()
    ]
    return np.mean(np.vstack(probabilities), axis=0)


def predict_core(artifact: dict, frame: pd.DataFrame) -> np.ndarray:
    if artifact.get("model_id") != "core_ensemble":
        raise ValueError("Expected a locked core clinical ensemble artifact.")
    return _member_mean(artifact, frame)


def predict_full(artifact: dict, frame: pd.DataFrame) -> np.ndarray:
    if artifact.get("model_id") != "full_ensemble":
        raise ValueError("Expected a locked full clinical ensemble artifact.")
    return _member_mean(artifact, frame)


def predict_availability_gated(artifact: dict, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if artifact.get("model_id") != "availability_gated_model":
        raise ValueError("Expected a locked availability-gated artifact.")
    available = frame[FULL_ONLY_FEATURES].notna().sum(axis=1).to_numpy(dtype=int)
    full_probability = _member_mean(artifact["full_ensemble"], frame)
    core_probability = _member_mean(artifact["core_ensemble"], frame)
    use_full = available >= AVAILABILITY_THRESHOLD
    probability = np.where(use_full, full_probability, core_probability)
    return probability, available, use_full
