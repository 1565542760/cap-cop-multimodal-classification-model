from __future__ import annotations

import hashlib
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "transportability_v3_20260908"
INTERNAL_PATH = ROOT / "data" / "local_internal_clinical.csv"
EXTERNAL_PATH = ROOT / "data" / "local_external_clinical.csv"

SEEDS = [20260908, 20260909, 20260910]
N_SPLITS = 5
N_BOOT = 5000
BOOT_SEED = 20260908

ALL_FEATURES = [
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
CORE7 = [
    "age",
    "body_mass_index",
    "fever",
    "cough",
    "sputum_category",
    "weight_loss",
    "lymphocyte_count",
]
CORE6_NO_SPUTUM = [x for x in CORE7 if x != "sputum_category"]
CORE7_HSCRP_PCT = CORE7 + [
    "high_sensitivity_c_reactive_protein",
    "procalcitonin",
]
TRANSPORTABLE11 = [
    "age",
    "height_cm",
    "weight_kg",
    "body_mass_index",
    "fever",
    "cough",
    "sputum_category",
    "weight_loss",
    "lymphocyte_count",
    "high_sensitivity_c_reactive_protein",
    "procalcitonin",
]

FEATURE_SETS = {
    "core7": CORE7,
    "core6_no_sputum": CORE6_NO_SPUTUM,
    "core7_hscrp_pct": CORE7_HSCRP_PCT,
    "transportable11": TRANSPORTABLE11,
    "full16": ALL_FEATURES,
}

LAB_FEATURES = {
    "c_reactive_protein",
    "high_sensitivity_c_reactive_protein",
    "procalcitonin",
    "erythrocyte_sedimentation_rate",
}


class FrameCleaner(BaseEstimator, TransformerMixin):
    """Apply only label-blind, deterministic data cleaning inside each fold."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in ["height_cm", "weight_kg", "body_mass_index"]:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")
                X.loc[X[col] == 0, col] = np.nan
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


class FoldWinsorizer(BaseEstimator, TransformerMixin):
    """Clip numeric inputs to training-fold quantiles; never uses labels."""

    def __init__(self, columns=None, lower=0.01, upper=0.99):
        self.columns = columns
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = X.copy()
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        for col in self.columns or []:
            values = pd.to_numeric(X[col], errors="coerce")
            self.lower_bounds_[col] = values.quantile(self.lower)
            self.upper_bounds_[col] = values.quantile(self.upper)
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns or []:
            if col not in X.columns:
                continue
            X[col] = pd.to_numeric(X[col], errors="coerce").clip(
                self.lower_bounds_[col], self.upper_bounds_[col]
            )
        return X


class FoldLog1p(BaseEstimator, TransformerMixin):
    """Apply log1p to non-negative laboratory variables after clipping."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns or []:
            if col in X.columns:
                values = pd.to_numeric(X[col], errors="coerce")
                X[col] = np.log1p(values.clip(lower=0))
        return X


class FixedMissingness(BaseEstimator, TransformerMixin):
    """Create label-independent missingness stress scenarios."""

    def __init__(self, columns=None, rate=0.0, random_state=0):
        self.columns = columns
        self.rate = rate
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.rate <= 0:
            return X
        rng = np.random.default_rng(self.random_state)
        for col in self.columns or []:
            if col in X.columns:
                X.loc[rng.random(len(X)) < self.rate, col] = np.nan
        return X


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_types(features: list[str], sputum_mode: str):
    categorical = [
        x
        for x in features
        if x in {"fever", "cough", "chest_tightness", "dyspnoea", "haemoptysis", "weight_loss"}
    ]
    if "sputum_category" in features and sputum_mode == "categorical":
        categorical.append("sputum_category")
    numeric = [x for x in features if x not in categorical]
    return numeric, categorical


def make_preprocessor(features, scaler_name, sputum_mode, clip_outliers, log_labs):
    numeric, categorical = split_types(features, sputum_mode)
    numeric_steps = []
    if clip_outliers:
        numeric_steps.append(("clip", FoldWinsorizer(columns=numeric)))
    if log_labs:
        numeric_steps.append(
            (
                "log1p_labs",
                FoldLog1p(columns=[x for x in numeric if x in LAB_FEATURES]),
            )
        )
    numeric_steps.append(("imputer", SimpleImputer(strategy="median", add_indicator=True)))
    if scaler_name == "standard":
        numeric_steps.append(("scaler", StandardScaler()))
    elif scaler_name == "robust":
        numeric_steps.append(("scaler", RobustScaler()))
    cat_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            ("categorical", Pipeline(cat_steps), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model(spec_name, features, seed):
    specs = {
        "logistic_standard": {
            "model": LogisticRegression(C=0.3, solver="liblinear", max_iter=5000),
            "scale": "standard",
            "sputum": "categorical",
            "clip": False,
            "log": False,
        },
        "logistic_robust": {
            "model": LogisticRegression(C=0.3, solver="liblinear", max_iter=5000),
            "scale": "robust",
            "sputum": "categorical",
            "clip": False,
            "log": False,
        },
        "logistic_clip_log": {
            "model": LogisticRegression(C=0.3, solver="liblinear", max_iter=5000),
            "scale": "robust",
            "sputum": "categorical",
            "clip": True,
            "log": True,
        },
        "logistic_ordinal_sputum": {
            "model": LogisticRegression(C=0.3, solver="liblinear", max_iter=5000),
            "scale": "robust",
            "sputum": "ordinal",
            "clip": True,
            "log": True,
        },
        "svm_ordinal": {
            "model": SVC(C=1.0, gamma="scale", probability=True, random_state=seed),
            "scale": "robust",
            "sputum": "ordinal",
            "clip": True,
            "log": True,
        },
        "random_forest_clip_log": {
            "model": RandomForestClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=5,
                random_state=seed,
                n_jobs=1,
            ),
            "scale": None,
            "sputum": "categorical",
            "clip": True,
            "log": True,
        },
    }
    if spec_name not in specs:
        raise ValueError(spec_name)
    spec = specs[spec_name]
    return Pipeline(
        [
            ("cleaner", FrameCleaner()),
            (
                "preprocess",
                make_preprocessor(
                    features,
                    spec["scale"],
                    spec["sputum"],
                    spec["clip"],
                    spec["log"],
                ),
            ),
            ("model", spec["model"]),
        ]
    )


MODEL_SPECS = [
    "logistic_standard",
    "logistic_robust",
    "logistic_clip_log",
    "logistic_ordinal_sputum",
    "svm_ordinal",
    "random_forest_clip_log",
]

SCENARIOS = {
    "normal": ([], 0.0, 0),
    "labs_30pct": (
        ["erythrocyte_sedimentation_rate", "c_reactive_protein",
         "high_sensitivity_c_reactive_protein", "procalcitonin"],
        0.30,
        301,
    ),
    "clinical_30pct": (
        ["cough", "sputum_category", "fever", "weight_loss"],
        0.30,
        302,
    ),
    "labs_and_clinical_30pct": (
        ["erythrocyte_sedimentation_rate", "c_reactive_protein",
         "high_sensitivity_c_reactive_protein", "procalcitonin",
         "cough", "sputum_category", "fever", "weight_loss"],
        0.30,
        303,
    ),
}


def apply_scenario(X: pd.DataFrame, scenario: str):
    columns, rate, random_state = SCENARIOS[scenario]
    if rate == 0:
        return X.copy()
    return FixedMissingness(
        columns=columns, rate=rate, random_state=random_state
    ).transform(X)


def metric_point(y, p, threshold):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "LogLoss": float(log_loss(y, p, labels=[0, 1])),
        "ACC": float(accuracy_score(y, pred)),
        "SE": float(tp / (tp + fn)) if tp + fn else np.nan,
        "SP": float(tn / (tn + fp)) if tn + fp else np.nan,
        "F1": float(f1_score(y, pred, zero_division=0)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def bootstrap_ci(y, p, threshold, seed):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in ["AUROC", "AUPRC", "Brier", "ACC", "SE", "SP", "F1"]}
    for _ in range(N_BOOT):
        idx = np.concatenate(
            [
                rng.choice(pos, size=len(pos), replace=True),
                rng.choice(neg, size=len(neg), replace=True),
            ]
        )
        vals = metric_point(y[idx], p[idx], threshold)
        for name in draws:
            draws[name].append(vals[name])
    out = {}
    for name, values in draws.items():
        arr = np.asarray(values, dtype=float)
        out[f"{name}_CI_low"] = float(np.nanpercentile(arr, 2.5))
        out[f"{name}_CI_high"] = float(np.nanpercentile(arr, 97.5))
    return out


def youden_threshold(y, p):
    fpr, tpr, thresholds = roc_curve(y, p)
    j = tpr - fpr
    candidates = thresholds[np.flatnonzero(j == np.nanmax(j))]
    finite = candidates[np.isfinite(candidates)]
    return float(finite[np.argmin(np.abs(finite - 0.5))]) if len(finite) else 0.5


def cross_fitted_predictions(X, y, features, model_spec, seed, scenario):
    from sklearn.model_selection import StratifiedKFold

    X_scenario = apply_scenario(X, scenario)
    p = np.full(len(X), np.nan, dtype=float)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_scenario, y)):
        pipe = build_model(model_spec, features, seed * 100 + fold)
        pipe.fit(X_scenario.iloc[train_idx][features], y[train_idx])
        p[val_idx] = pipe.predict_proba(
            X_scenario.iloc[val_idx][features]
        )[:, 1]
    return p


def fit_full_ensemble(X, y, X_external, features, model_spec):
    preds = []
    for seed in SEEDS:
        pipe = build_model(model_spec, features, seed)
        pipe.fit(X[features], y)
        preds.append(pipe.predict_proba(X_external[features])[:, 1])
    return np.mean(np.vstack(preds), axis=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    internal = pd.read_csv(INTERNAL_PATH)
    external = pd.read_csv(EXTERNAL_PATH)
    y_internal = internal["label_binary"].astype(int).to_numpy()
    y_external = external["label"].astype(int).to_numpy()
    if len(internal) != 375 or y_internal.sum() != 125:
        raise ValueError("Unexpected internal cohort")
    if len(external) != 33 or y_external.sum() != 14:
        raise ValueError("Unexpected external cohort")

    candidate_rows = []
    pooled_normal = {}
    external_predictions = {}
    oof_records = []
    for feature_set_name, features in FEATURE_SETS.items():
        for model_spec in MODEL_SPECS:
            scenario_auroc = {}
            scenario_brier = {}
            scenario_predictions = {}
            for scenario in SCENARIOS:
                seed_predictions = []
                for seed in SEEDS:
                    p = cross_fitted_predictions(
                        internal,
                        y_internal,
                        features,
                        model_spec,
                        seed,
                        scenario,
                    )
                    seed_predictions.append(p)
                pooled = np.mean(np.vstack(seed_predictions), axis=0)
                scenario_predictions[scenario] = pooled
                scenario_auroc[scenario] = roc_auc_score(y_internal, pooled)
                scenario_brier[scenario] = brier_score_loss(y_internal, pooled)
                if scenario == "normal":
                    pooled_normal[(feature_set_name, model_spec)] = pooled
                    for idx, prob in enumerate(pooled):
                        oof_records.append(
                            {
                                "feature_set": feature_set_name,
                                "model_spec": model_spec,
                                "scenario": scenario,
                                "row_index": idx,
                                "case_id_public": internal.iloc[idx]["case_id_public"],
                                "label": int(y_internal[idx]),
                                "prob_COP": float(prob),
                            }
                        )
            candidate_rows.append(
                {
                    "feature_set": feature_set_name,
                    "n_features": len(features),
                    "features": ";".join(features),
                    "model_spec": model_spec,
                    "normal_AUROC": scenario_auroc["normal"],
                    "normal_AUPRC": average_precision_score(
                        y_internal, scenario_predictions["normal"]
                    ),
                    "normal_Brier": scenario_brier["normal"],
                    "mean_stress_AUROC": float(np.mean(list(scenario_auroc.values()))),
                    "worst_stress_AUROC": float(np.min(list(scenario_auroc.values()))),
                    "mean_stress_Brier": float(np.mean(list(scenario_brier.values()))),
                    "labs30_AUROC": scenario_auroc["labs_30pct"],
                    "clinical30_AUROC": scenario_auroc["clinical_30pct"],
                    "labs_clinical30_AUROC": scenario_auroc["labs_and_clinical_30pct"],
                }
            )
            external_predictions[(feature_set_name, model_spec)] = fit_full_ensemble(
                internal, y_internal, external, features, model_spec
            )

    candidate_table = pd.DataFrame(candidate_rows).sort_values(
        ["mean_stress_AUROC", "worst_stress_AUROC", "normal_AUROC", "mean_stress_Brier"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    selected = candidate_table.iloc[0]
    selected_key = (str(selected["feature_set"]), str(selected["model_spec"]))
    selected_internal = pooled_normal[selected_key]
    selected_external = external_predictions[selected_key]
    threshold = round(youden_threshold(y_internal, selected_internal), 6)

    performance = pd.DataFrame(
        [
            {
                "cohort": "internal_OOF_normal",
                **metric_point(y_internal, selected_internal, threshold),
                **bootstrap_ci(y_internal, selected_internal, threshold, BOOT_SEED + 1),
            },
            {
                "cohort": "external_frozen_all",
                **metric_point(y_external, selected_external, threshold),
                **bootstrap_ci(y_external, selected_external, threshold, BOOT_SEED + 2),
            },
        ]
    )

    # These restrictions are secondary, label-blind analyses. They do not
    # alter selection, the fitted model, or the locked threshold.
    restriction_rows = []
    for fraction in [0.50, 0.75, 1.00]:
        min_obs = int(np.ceil(len(FEATURE_SETS[selected_key[0]]) * fraction))
        observed = external[FEATURE_SETS[selected_key[0]]].notna().sum(axis=1)
        keep = observed >= min_obs
        row = {
            "feature_set": selected_key[0],
            "model_spec": selected_key[1],
            "fraction": fraction,
            "minimum_observed": min_obs,
            "external_n_total": len(external),
            "external_n_included": int(keep.sum()),
            "external_n_excluded": int((~keep).sum()),
            "included_CAP_n": int(((y_external == 0) & keep).sum()),
            "included_COP_n": int(((y_external == 1) & keep).sum()),
        }
        if keep.sum() and len(np.unique(y_external[keep])) == 2:
            row.update(metric_point(y_external[keep], selected_external[keep], threshold))
            row.update(
                bootstrap_ci(
                    y_external[keep],
                    selected_external[keep],
                    threshold,
                    BOOT_SEED + int(fraction * 100),
                )
            )
        else:
            row.update({name: np.nan for name in [
                "AUROC", "AUPRC", "Brier", "LogLoss", "ACC", "SE", "SP", "F1",
            ]})
        restriction_rows.append(row)

    quality_audit = pd.read_csv(
        ROOT / "outputs" / "sensitivity_20260908" / "data_quality_record_audit.csv"
    )
    external_flagged_ids = set(
        quality_audit.loc[
            quality_audit["source"].eq("external_aligned_csv"), "record_id"
        ]
    )
    keep_quality = ~external["patient_id"].isin(external_flagged_ids)
    quality_row = {
        "analysis": "external_extreme_values_removed_secondary_only",
        "rule": "remove only four previously flagged records; not used for selection",
        "external_n_included": int(keep_quality.sum()),
        "included_CAP_n": int(((y_external == 0) & keep_quality).sum()),
        "included_COP_n": int(((y_external == 1) & keep_quality).sum()),
    }
    if len(np.unique(y_external[keep_quality])) == 2:
        quality_row.update(
            metric_point(y_external[keep_quality], selected_external[keep_quality], threshold)
        )
    else:
        quality_row.update({name: np.nan for name in [
            "AUROC", "AUPRC", "Brier", "LogLoss", "ACC", "SE", "SP", "F1",
        ]})

    external_pred_table = pd.DataFrame(
        {
            "patient_id": external["patient_id"],
            "label": y_external,
            "prob_COP": selected_external,
            "flagged_extreme_value": external["patient_id"].isin(external_flagged_ids),
            "prediction_locked_threshold": (selected_external >= threshold).astype(int),
        }
    )

    candidate_table.to_csv(OUT / "candidate_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(oof_records).to_csv(
        OUT / "selected_internal_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    performance.to_csv(OUT / "selected_performance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(restriction_rows).to_csv(
        OUT / "external_missingness_restrictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([quality_row]).to_csv(
        OUT / "external_quality_flag_restriction.csv",
        index=False,
        encoding="utf-8-sig",
    )
    external_pred_table.to_csv(
        OUT / "selected_external_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "analysis_date": "2026-09-08",
        "iteration": "transportability_v3",
        "internal_source": str(INTERNAL_PATH),
        "external_source": str(EXTERNAL_PATH),
        "internal_n": len(internal),
        "external_n": len(external),
        "positive_class": "COP",
        "negative_class": "CAP",
        "candidate_feature_sets": FEATURE_SETS,
        "candidate_model_specs": MODEL_SPECS,
        "scenarios": SCENARIOS,
        "selection_rule": (
            "internal repeated 5-fold OOF only: mean AUROC across normal and "
            "three predefined 30% label-blind missingness stress scenarios, "
            "then worst stress AUROC, normal AUROC, and mean stress Brier"
        ),
        "external_labels_used_for_selection": False,
        "external_labels_used_for_threshold": False,
        "external_covariates_used_for_candidate_availability_rules": True,
        "selected_feature_set": selected_key[0],
        "selected_model_spec": selected_key[1],
        "selected_features": FEATURE_SETS[selected_key[0]],
        "locked_threshold": threshold,
        "bootstrap_replicates": N_BOOT,
        "internal_sha256": sha256_file(INTERNAL_PATH),
        "external_sha256": sha256_file(EXTERNAL_PATH),
        "selected_internal_metrics": performance.iloc[0].to_dict(),
        "selected_external_metrics": performance.iloc[1].to_dict(),
        "flagged_external_quality_records": sorted(external_flagged_ids),
        "quality_flag_restriction_policy": (
            "secondary only; records retained in primary analysis; exclusion is "
            "not used for selection and requires data-dictionary confirmation"
        ),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (OUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))
    print("\nTop candidate configurations:")
    print(candidate_table.head(15).to_string(index=False))
    print("\nSelected performance:")
    print(performance[["cohort", "AUROC", "AUPRC", "Brier", "ACC", "SE", "SP", "F1"]].to_string(index=False))
    print("\nExternal missingness restrictions:")
    print(pd.DataFrame(restriction_rows)[
        ["fraction", "external_n_included", "included_CAP_n", "included_COP_n", "AUROC", "AUPRC", "Brier", "ACC", "SE", "SP", "F1"]
    ].to_string(index=False))
    print("\nExternal quality flag restriction:")
    print(pd.DataFrame([quality_row]).to_string(index=False))
    print(f"\nOUTPUT_DIR={OUT}")


if __name__ == "__main__":
    main()
