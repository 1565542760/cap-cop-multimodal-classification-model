from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


FEATURES = [
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

CATEGORICAL = {
    "fever",
    "cough",
    "sputum_category",
    "chest_tightness",
    "dyspnoea",
    "haemoptysis",
    "weight_loss",
}

LAB_FEATURES = {
    "c_reactive_protein",
    "high_sensitivity_c_reactive_protein",
    "procalcitonin",
    "erythrocyte_sedimentation_rate",
}

DEFAULT_SEEDS = [20260908, 20260909, 20260910]
LOCKED_THRESHOLD = 0.443125


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[FEATURES].copy()
    for column in ["height_cm", "weight_kg", "body_mass_index"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        result.loc[result[column] == 0, column] = np.nan
    valid_bmi = (
        result["body_mass_index"].isna()
        & result["height_cm"].notna()
        & result["weight_kg"].notna()
        & (result["height_cm"] > 0)
    )
    result.loc[valid_bmi, "body_mass_index"] = (
        result.loc[valid_bmi, "weight_kg"]
        / (result.loc[valid_bmi, "height_cm"] / 100.0) ** 2
    )
    return result


class FoldImputer:
    def __init__(self, features: list[str], seed: int):
        self.features = list(features)
        self.seed = int(seed)

    def fit(self, frame: pd.DataFrame):
        values = clean(frame).apply(pd.to_numeric, errors="coerce")
        self.lower = {
            feature: float(values[feature].quantile(0.01))
            if values[feature].notna().any()
            else 0.0
            for feature in self.features
        }
        self.upper = {
            feature: float(values[feature].quantile(0.99))
            if values[feature].notna().any()
            else 0.0
            for feature in self.features
        }
        self.categories = {}
        for feature in self.features:
            if feature in CATEGORICAL:
                observed = sorted(values[feature].dropna().unique().tolist())
                self.categories[feature] = observed or [0.0]
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.imputer.fit(values.to_numpy(dtype=float, copy=True))
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        original = clean(frame)
        values = original.apply(pd.to_numeric, errors="coerce")
        imputed = self.imputer.transform(values.to_numpy(dtype=float, copy=True))
        result = pd.DataFrame(imputed, columns=self.features, index=frame.index)
        for feature in self.features:
            result[feature] = result[feature].clip(
                lower=self.lower[feature],
                upper=self.upper[feature],
            )
            if feature in self.categories:
                categories = np.asarray(self.categories[feature])
                result[feature] = result[feature].map(
                    lambda value: float(categories[np.argmin(np.abs(categories - value))])
                )
        return result


def make_design(
    imputed: pd.DataFrame,
    original: pd.DataFrame,
) -> pd.DataFrame:
    result = imputed.copy()
    missing = original[FEATURES].isna()
    for feature in FEATURES:
        result[f"__missing_{feature}"] = missing[feature].astype(float)
    clinical = ["fever", "cough", "sputum_category", "weight_loss"]
    labs = [feature for feature in LAB_FEATURES if feature in FEATURES]
    anthropometric = ["height_cm", "weight_kg", "body_mass_index"]
    result["__missing_count_selected"] = missing.sum(axis=1).astype(float)
    result["__missing_count_clinical"] = missing[clinical].sum(axis=1).astype(float)
    result["__missing_count_laboratory"] = missing[labs].sum(axis=1).astype(float)
    result["__missing_count_anthropometric"] = (
        missing[anthropometric].sum(axis=1).astype(float)
    )
    return result


def balanced_bootstrap(
    design: pd.DataFrame,
    labels: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=int)
    class_indices = {
        cls: np.flatnonzero(labels == cls)
        for cls in np.unique(labels)
    }
    target_n = max(len(indices) for indices in class_indices.values())
    rows = [design.copy()]
    output_labels = [labels.copy()]
    for cls, indices in sorted(class_indices.items()):
        if len(indices) >= target_n:
            continue
        sampled = rng.choice(indices, size=target_n - len(indices), replace=True)
        rows.append(design.iloc[sampled].copy().reset_index(drop=True))
        output_labels.append(np.full(target_n - len(indices), cls, dtype=int))
    combined = pd.concat(rows, ignore_index=True)
    combined_labels = np.concatenate(output_labels)
    order = rng.permutation(len(combined))
    return combined.iloc[order].reset_index(drop=True), combined_labels[order]


def encode_design(
    design: pd.DataFrame,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    encoded = pd.get_dummies(
        design,
        columns=[feature for feature in FEATURES if feature in CATEGORICAL],
        drop_first=False,
        dtype=float,
    )
    if columns is not None:
        encoded = encoded.reindex(columns=columns, fill_value=0.0)
    return encoded, encoded.columns.tolist()


def fit_single(
    internal: pd.DataFrame,
    labels: np.ndarray,
    seed: int,
) -> dict:
    original = clean(internal)
    imputer = FoldImputer(FEATURES, seed).fit(internal)
    imputed = imputer.transform(internal)
    design = make_design(imputed, original)
    augmented_design, augmented_labels = balanced_bootstrap(
        design,
        labels,
        seed + 7919,
    )
    encoded, columns = encode_design(augmented_design)
    model = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(encoded, augmented_labels)
    return {
        "features": FEATURES,
        "seed": int(seed),
        "imputer": imputer,
        "model": model,
        "encoded_columns": columns,
    }


def fit_ensemble(
    internal: pd.DataFrame,
    labels: np.ndarray,
    seeds: list[int] | None = None,
) -> dict:
    seeds = list(seeds or DEFAULT_SEEDS)
    return {
        "model_name": "v34_full16_balanced_bootstrap_random_forest",
        "positive_class": "COP",
        "negative_class": "CAP",
        "features": FEATURES,
        "threshold": LOCKED_THRESHOLD,
        "members": [fit_single(internal, labels, seed) for seed in seeds],
    }


def predict_proba(artifact: dict, frame: pd.DataFrame) -> np.ndarray:
    probabilities = []
    original = clean(frame)
    for member in artifact["members"]:
        imputed = member["imputer"].transform(frame)
        design = make_design(imputed, original)
        encoded, _ = encode_design(design, member["encoded_columns"])
        probabilities.append(member["model"].predict_proba(encoded)[:, 1])
    return np.mean(np.vstack(probabilities), axis=0)


def save_artifact(artifact: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path, compress=3)


def load_artifact(path: str | Path) -> dict:
    # The locked joblib predates the semantic public module name. Registering
    # the historical import name preserves byte-identical artifact loading
    # without exposing that name as a public source filename.
    sys.modules.setdefault("capcop_v34_model", sys.modules[__name__])
    return joblib.load(path)
