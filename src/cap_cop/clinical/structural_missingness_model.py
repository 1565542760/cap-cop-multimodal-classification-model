from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class PermanentImputerIndicator:
    def __init__(self, features: list[str]):
        self.features = list(features)
        self.medians_: pd.Series | None = None

    def fit(self, X: pd.DataFrame):
        self.medians_ = X[self.features].apply(pd.to_numeric, errors="coerce").median(axis=0)
        self.medians_ = self.medians_.fillna(0.0)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.medians_ is None:
            raise RuntimeError("Transformer not fit")
        raw = X[self.features].apply(pd.to_numeric, errors="coerce")
        miss = raw.isna().astype(float)
        imp = raw.fillna(self.medians_)
        return np.concatenate([imp.to_numpy(dtype=float), miss.to_numpy(dtype=float)], axis=1)

    @property
    def output_dim(self) -> int:
        return len(self.features) * 2


class RFStructuralModel:
    def __init__(self, features: list[str], params: dict[str, Any], seed: int):
        self.features = list(features)
        self.params = dict(params)
        self.seed = int(seed)
        self.t = PermanentImputerIndicator(self.features)
        self.model = RandomForestClassifier(
            n_estimators=int(self.params["n_estimators"]),
            max_depth=self.params["max_depth"],
            min_samples_leaf=int(self.params["min_samples_leaf"]),
            max_features=self.params["max_features"],
            class_weight="balanced",
            random_state=self.seed,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.t.fit(X)
        self.model.fit(self.t.transform(X), y.to_numpy(dtype=int))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.t.transform(X))[:, 1]


def load_locked_artifact(path: str | Path) -> RFStructuralModel:
    """Load a locked artifact while preserving its historical pickle module."""
    sys.modules.setdefault("rf_structuralmissing_model_v2b", sys.modules[__name__])
    return joblib.load(path)
