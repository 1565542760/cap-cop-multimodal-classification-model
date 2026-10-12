from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
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
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, PredefinedSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get("CAP_COP_LOCAL_HISTORICAL_ROOT", PROJECT_ROOT / "models" / "historical_clinical_contracts"))
EXP = ROOT / "experiment_v2"
GOV = ROOT / "PROJECT_GOVERNANCE"

REMOTE = EXP / "clinical_remote_sync_20260903"
S3 = EXP / "multimodal_v3_6_s3_core_d1"
S4 = EXP / "multimodal_v3_6_s4_gated_d1"

INTERNAL_CLINICAL = (
    ROOT
    / "experiment_v1"
    / "cap_cop_experiment"
    / "generalisation_redevelopment_nested_protocol_v1"
    / "data"
    / "internal_development"
    / "clinical_features_basic_corrected_for_development.csv"
)
EXTERNAL_CLINICAL = (
    ROOT
    / "experiment_v1"
    / "cap_cop_experiment"
    / "data_locked"
    / "external_raw_locked_v1"
    / "doctor_completion"
    / "external_clinical_basic_doctor_fused_aligned.csv"
)
SPLIT_MANIFEST = EXP / "splits" / "locked_nested_split_patient_manifest.csv"
PRIMARY_OOF = (
    EXP
    / "multimodal_v3_6"
    / "04_nested_fusion"
    / "V3_6_INTERNAL375_STRICT_NESTED_MULTIMODAL_OOF_PREDICTIONS.csv"
)
D1_EXT_BLIND = (
    EXP
    / "imaging_segment_then_classify_highres_v3_3"
    / "05_classification"
    / "10_pyradiomics_standardized_v3_5_d"
    / "07_external_label_blind"
    / "V3_5_D1_EXTERNAL33_PREDICTIONS_LABEL_BLIND.csv"
)
D1_EXT_WITH_LABEL = (
    EXP
    / "imaging_segment_then_classify_highres_v3_3"
    / "05_classification"
    / "10_pyradiomics_standardized_v3_5_d"
    / "08_external_results"
    / "V3_5_D1_EXTERNAL33_PREDICTIONS_WITH_LABEL_PRIVATE.csv"
)
D1_LOCK = (
    EXP
    / "imaging_segment_then_classify_highres_v3_3"
    / "05_classification"
    / "10_pyradiomics_standardized_v3_5_d"
    / "11_locks"
    / "V3_5_D1_PYRADIOMICS_STANDARDIZED_EXTERNAL_COMPLETION_RESULT_LOCK.json"
)
REMOTE_SYNC_LOCK = GOV / "REMOTE_SYNC_20260903" / "REMOTE_CLINICAL_SYNC_RESULT_LOCK.json"
REGISTRY = GOV / "EXPERIMENT_REGISTRY.csv"
ROADMAP = GOV / "EXPERIMENT_ROADMAP_MASTER.md"

SEED = 20260826
LOCKED_SPLIT_SEED = 20260627
OUTER_FOLDS = 5
INNER_FOLDS = 4
HIST_CORE = {"AUROC": 0.759264, "AUPRC": 0.587483, "Brier": 0.197045}
HIST_GATED = {"AUROC": 0.836608}
DISCREPANCY_AUROC = 0.03

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
FULL_CONTINUOUS = [
    "age",
    "height_cm",
    "weight_kg",
    "body_mass_index",
    "lymphocyte_count",
    "erythrocyte_sedimentation_rate",
    "c_reactive_protein",
    "high_sensitivity_c_reactive_protein",
    "procalcitonin",
]
FULL_CATEGORICAL = [
    "fever",
    "cough",
    "sputum_category",
    "chest_tightness",
    "dyspnoea",
    "haemoptysis",
    "weight_loss",
]
CORE_CONTINUOUS = ["age", "body_mass_index", "lymphocyte_count"]
CORE_CATEGORICAL = ["fever", "cough", "sputum_category", "weight_loss"]
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


class FrameCleaner(BaseEstimator, TransformerMixin):
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


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def mkdirs() -> None:
    for base in [S3, S4]:
        for sub in [
            "00_protocol",
            "01_clinical_contract",
            "02_clinical_oof_reconstruction",
            "03_internal_alignment",
            "04_complementarity",
            "05_strict_nested_fusion",
            "06_internal_results",
            "07_final_clinical_pipelines",
            "08_final_fusion",
            "09_external_label_blind",
            "10_external_results",
            "11_error_analysis",
            "12_reports",
            "13_locks",
        ]:
            (base / sub).mkdir(parents=True, exist_ok=True)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return sha(path)


def write_csv(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return sha(path)


def write_md(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return sha(path)


def make_preprocessor(continuous, categorical, scaler=None):
    continuous_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scaler is not None:
        continuous_steps.append(("scaler", scaler))
    return ColumnTransformer(
        [
            ("continuous", Pipeline(continuous_steps), continuous),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def make_base_pipeline(feature_set: str, model_kind: str) -> Pipeline:
    if feature_set == "full":
        continuous, categorical = FULL_CONTINUOUS, FULL_CATEGORICAL
    else:
        continuous, categorical = CORE_CONTINUOUS, CORE_CATEGORICAL
    if model_kind == "logistic":
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
        )
        scaler = RobustScaler()
    else:
        model = RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=1,
        )
        scaler = None
    return Pipeline(
        [
            ("cleaner", FrameCleaner()),
            ("preprocess", make_preprocessor(continuous, categorical, scaler=scaler)),
            ("model", model),
        ]
    )


MODEL_SPECS = {
    "full_logistic_robust": ("full", "logistic"),
    "full_random_forest": ("full", "random_forest"),
    "core_logistic_robust": ("core", "logistic"),
    "core_random_forest": ("core", "random_forest"),
}


def param_grid(model_name: str) -> dict:
    if "random_forest" in model_name:
        return {"model__min_samples_leaf": [3, 5, 10]}
    return {"model__C": [0.03, 0.1, 0.3, 1.0]}


def candidate_input(X: pd.DataFrame, model_name: str) -> pd.DataFrame:
    feature_set, _ = MODEL_SPECS[model_name]
    return X[FULL_FEATURES if feature_set == "full" else CORE_FEATURES].copy()


def fit_search(
    X: pd.DataFrame,
    y: np.ndarray,
    model_name: str,
    cv,
) -> tuple[Pipeline, dict]:
    feature_set, model_kind = MODEL_SPECS[model_name]
    search = GridSearchCV(
        make_base_pipeline(feature_set, model_kind),
        param_grid=param_grid(model_name),
        scoring="roc_auc",
        cv=cv,
        refit=True,
        n_jobs=1,
        error_score="raise",
    )
    search.fit(candidate_input(X, model_name), y)
    return search.best_estimator_, {
        "model_name": model_name,
        "best_params": search.best_params_,
        "inner_mean_auroc": float(search.best_score_),
    }


def metrics(y_true, prob, prefix: str | None = None) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y_true)),
        "positive_n": int(y_true.sum()),
        "AUROC": float(roc_auc_score(y_true, prob)),
        "AUPRC": float(average_precision_score(y_true, prob)),
        "Brier": float(brier_score_loss(y_true, prob)),
        "LogLoss": float(log_loss(y_true, np.clip(prob, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "Sensitivity": float(tp / (tp + fn)) if (tp + fn) else math.nan,
        "Specificity": float(tn / (tn + fp)) if (tn + fp) else math.nan,
        "Accuracy": float(accuracy_score(y_true, pred)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
    if prefix:
        return {f"{prefix}_{k}": v for k, v in out.items()}
    return out


def bootstrap_ci(y, prob, metric_name: str, n_boot: int = 5000, seed: int = 20260903):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        pp = prob[idx]
        if metric_name == "AUROC":
            vals.append(roc_auc_score(yy, pp))
        elif metric_name == "AUPRC":
            vals.append(average_precision_score(yy, pp))
        elif metric_name == "Brier":
            vals.append(brier_score_loss(yy, pp))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def delta_ci(y, new_prob, ref_prob, metric_name: str, n_boot: int = 5000, seed: int = 20260904):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    new_prob = np.asarray(new_prob, dtype=float)
    ref_prob = np.asarray(ref_prob, dtype=float)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        if metric_name == "AUROC":
            vals.append(roc_auc_score(yy, new_prob[idx]) - roc_auc_score(yy, ref_prob[idx]))
        elif metric_name == "AUPRC":
            vals.append(average_precision_score(yy, new_prob[idx]) - average_precision_score(yy, ref_prob[idx]))
        elif metric_name == "Brier":
            vals.append(brier_score_loss(yy, new_prob[idx]) - brier_score_loss(yy, ref_prob[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def load_inputs():
    for p in [
        INTERNAL_CLINICAL,
        EXTERNAL_CLINICAL,
        SPLIT_MANIFEST,
        PRIMARY_OOF,
        D1_EXT_BLIND,
        D1_EXT_WITH_LABEL,
        D1_LOCK,
        REMOTE_SYNC_LOCK,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)
    internal = pd.read_csv(INTERNAL_CLINICAL)
    external_features = pd.read_csv(
        EXTERNAL_CLINICAL,
        usecols=lambda c: c != "label",
    )
    split = pd.read_csv(SPLIT_MANIFEST)
    primary = pd.read_csv(PRIMARY_OOF)
    d1_ext = pd.read_csv(D1_EXT_BLIND)
    required_internal = set(FULL_FEATURES + ["case_id_public", "label_binary", "patient_group_id_public"])
    if not required_internal <= set(internal.columns):
        raise RuntimeError("Internal clinical table is missing required columns")
    if not set(FULL_FEATURES + ["patient_id"]) <= set(external_features.columns):
        raise RuntimeError("External clinical feature table is missing required columns")
    return internal, external_features, split, primary, d1_ext


def build_split_maps(split: pd.DataFrame, primary: pd.DataFrame) -> dict:
    split = split[
        (split["outer_seed"] == LOCKED_SPLIT_SEED)
        & (split["training_seed"] == LOCKED_SPLIT_SEED)
    ].copy()
    if len(split) != 1875:
        raise RuntimeError(f"Unexpected locked split row count: {len(split)}")
    test_rows = split[split["outer_role"] == "outer_test"]
    fold_map = test_rows.set_index("case_id_public")["outer_fold_id"].astype(int).to_dict()
    primary_map = primary.set_index("case_id_public")["outer_fold"].astype(int).to_dict()
    mismatches = [k for k, v in primary_map.items() if fold_map.get(k) != v]
    if mismatches:
        raise RuntimeError(f"D1 primary OOF outer fold mismatch for {len(mismatches)} cases")
    inner_maps = {}
    for outer_fold in range(OUTER_FOLDS):
        rows = split[(split["outer_fold_id"] == outer_fold) & (split["outer_role"] == "outer_train")]
        inner_maps[outer_fold] = rows.set_index("case_id_public")["inner_fold_id"].astype(int).to_dict()
    return {"split": split, "fold_map": fold_map, "inner_maps": inner_maps}


def predefined_cv_for_cases(case_ids: list[str], inner_map: dict[str, int]) -> PredefinedSplit:
    return PredefinedSplit([inner_map[c] for c in case_ids])


def stratified_cv_for_small_train() -> StratifiedKFold:
    return StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)


def restore_clinical_oof(internal: pd.DataFrame, split_info: dict) -> dict:
    X = internal.set_index("case_id_public")[FULL_FEATURES].copy()
    y_s = internal.set_index("case_id_public")["label_binary"].astype(int)
    case_ids = list(X.index)
    outer_fold_s = pd.Series(split_info["fold_map"], name="outer_fold").astype(int)
    outer_fold_s = outer_fold_s.loc[case_ids]

    outer_probs = {name: pd.Series(np.nan, index=case_ids, dtype=float) for name in MODEL_SPECS}
    meta_inner_probs = {
        outer: {name: pd.Series(np.nan, index=case_ids, dtype=float) for name in MODEL_SPECS}
        for outer in range(OUTER_FOLDS)
    }
    fold_records = []
    inner_records = []

    for outer in range(OUTER_FOLDS):
        train_ids = [c for c in case_ids if outer_fold_s.loc[c] != outer]
        test_ids = [c for c in case_ids if outer_fold_s.loc[c] == outer]
        inner_map = split_info["inner_maps"][outer]
        cv = predefined_cv_for_cases(train_ids, inner_map)
        for model_name in MODEL_SPECS:
            estimator, record = fit_search(
                X.loc[train_ids],
                y_s.loc[train_ids].to_numpy(),
                model_name,
                cv=cv,
            )
            pred = estimator.predict_proba(candidate_input(X.loc[test_ids], model_name))[:, 1]
            outer_probs[model_name].loc[test_ids] = pred
            fold_records.append({"outer_fold": outer, **record})

            for inner_fold in range(INNER_FOLDS):
                val_ids = [c for c in train_ids if inner_map[c] == inner_fold]
                inner_train_ids = [c for c in train_ids if inner_map[c] != inner_fold]
                inner_estimator, inner_record = fit_search(
                    X.loc[inner_train_ids],
                    y_s.loc[inner_train_ids].to_numpy(),
                    model_name,
                    cv=stratified_cv_for_small_train(),
                )
                inner_pred = inner_estimator.predict_proba(
                    candidate_input(X.loc[val_ids], model_name)
                )[:, 1]
                meta_inner_probs[outer][model_name].loc[val_ids] = inner_pred
                inner_records.append(
                    {
                        "outer_fold": outer,
                        "inner_fold": inner_fold,
                        "n_train": len(inner_train_ids),
                        "n_validation": len(val_ids),
                        **inner_record,
                    }
                )

    for series in outer_probs.values():
        if series.isna().any():
            raise RuntimeError("Clinical outer OOF is incomplete")
    for outer in range(OUTER_FOLDS):
        train_ids = [c for c in case_ids if outer_fold_s.loc[c] != outer]
        for series in meta_inner_probs[outer].values():
            if series.loc[train_ids].isna().any():
                raise RuntimeError(f"Clinical inner OOF incomplete for outer {outer}")

    full_outer = (outer_probs["full_logistic_robust"] + outer_probs["full_random_forest"]) / 2
    core_outer = (outer_probs["core_logistic_robust"] + outer_probs["core_random_forest"]) / 2
    availability = X[FULL_ONLY_FEATURES].notna().sum(axis=1).astype(int)
    gated_outer = pd.Series(
        np.where(availability >= 6, full_outer, core_outer), index=case_ids, dtype=float
    )
    route_outer = pd.Series(np.where(availability >= 6, "full_ensemble", "core_ensemble"), index=case_ids)

    meta = {}
    for outer in range(OUTER_FOLDS):
        full_inner = (
            meta_inner_probs[outer]["full_logistic_robust"]
            + meta_inner_probs[outer]["full_random_forest"]
        ) / 2
        core_inner = (
            meta_inner_probs[outer]["core_logistic_robust"]
            + meta_inner_probs[outer]["core_random_forest"]
        ) / 2
        gated_inner = pd.Series(
            np.where(availability >= 6, full_inner, core_inner), index=case_ids, dtype=float
        )
        meta[outer] = {
            "core": core_inner,
            "full": full_inner,
            "gated": gated_inner,
        }

    base_df = pd.DataFrame(
        {
            "neutral_patient_id": case_ids,
            "outer_fold": outer_fold_s.loc[case_ids].to_numpy(dtype=int),
            "label": y_s.loc[case_ids].to_numpy(dtype=int),
            "available_full_only_count": availability.loc[case_ids].to_numpy(dtype=int),
            "clinical_route": route_outer.loc[case_ids].to_numpy(),
            "P_full_COP": full_outer.loc[case_ids].to_numpy(float),
            "P_core_COP": core_outer.loc[case_ids].to_numpy(float),
            "P_gated_COP": gated_outer.loc[case_ids].to_numpy(float),
            "P_core_lr_COP": outer_probs["core_logistic_robust"].loc[case_ids].to_numpy(float),
            "P_core_rf_COP": outer_probs["core_random_forest"].loc[case_ids].to_numpy(float),
            "P_full_lr_COP": outer_probs["full_logistic_robust"].loc[case_ids].to_numpy(float),
            "P_full_rf_COP": outer_probs["full_random_forest"].loc[case_ids].to_numpy(float),
        }
    ).sort_values("neutral_patient_id")
    return {
        "base": base_df,
        "meta_inner": meta,
        "fold_records": pd.DataFrame(fold_records),
        "inner_records": pd.DataFrame(inner_records),
    }


def save_contracts() -> tuple[str, str]:
    remote_code = REMOTE / "01_code" / "run_clinical_transportability_robustness.py"
    gated_code = REMOTE / "01_code" / "run_availability_gated_model_experiment.py"
    s3_contract = {
        "model": "core_ensemble",
        "source_code": str(remote_code),
        "source_code_sha256": sha(remote_code),
        "feature_count": len(CORE_FEATURES),
        "features": CORE_FEATURES,
        "components": {
            "core_logistic_robust": {
                "preprocessing": "FrameCleaner + median imputation with indicators + RobustScaler",
                "model": "LogisticRegression",
                "solver": "liblinear",
                "class_weight": "balanced",
                "max_iter": 3000,
                "C_grid": [0.03, 0.1, 0.3, 1.0],
            },
            "core_random_forest": {
                "preprocessing": "FrameCleaner + median/mode imputation with indicators",
                "model": "RandomForestClassifier",
                "n_estimators": 300,
                "max_features": "sqrt",
                "min_samples_leaf_grid": [3, 5, 10],
                "class_weight": "balanced",
                "random_state": SEED,
            },
        },
        "ensemble_weighting": "equal_weight_mean_lr_rf",
        "probability": "P(COP)",
        "threshold": 0.5,
        "outer_split_contract": str(SPLIT_MANIFEST),
        "outer_split_seed": LOCKED_SPLIT_SEED,
        "external_used_for_training_or_selection": False,
    }
    s4_contract = {
        "model": "availability_gated_model",
        "source_code": str(gated_code),
        "source_code_sha256": sha(gated_code),
        "full_features": FULL_FEATURES,
        "core_features": CORE_FEATURES,
        "full_only_feature_count": len(FULL_ONLY_FEATURES),
        "full_only_features": FULL_ONLY_FEATURES,
        "available_definition": "pandas notna() over the 9 full-only variables",
        "routing_threshold": "available_full_only_count >= 6",
        "route_full_when": "available_full_only_count >= 6",
        "route_core_when": "available_full_only_count < 6",
        "full_ensemble": "full_logistic_robust + full_random_forest equal-weight average",
        "core_ensemble": "core_logistic_robust + core_random_forest equal-weight average",
        "probability": "P(COP)",
        "threshold": 0.5,
        "outer_split_contract": str(SPLIT_MANIFEST),
        "outer_split_seed": LOCKED_SPLIT_SEED,
        "external_used_for_training_or_selection": False,
    }
    s3_sha = write_json(S3 / "01_clinical_contract" / "V3_6_S3_CORE_PIPELINE_CONTRACT.json", s3_contract)
    s4_sha = write_json(S4 / "01_clinical_contract" / "V3_6_S4_AVAILABILITY_GATED_PIPELINE_CONTRACT.json", s4_contract)
    return s3_sha, s4_sha


def align_image(base: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    image = primary[["case_id_public", "outer_fold", "label_binary", "P_image_COP"]].rename(
        columns={"case_id_public": "neutral_patient_id", "label_binary": "image_label"}
    )
    merged = base.merge(image, on=["neutral_patient_id", "outer_fold"], how="inner")
    if len(merged) != 375:
        raise RuntimeError(f"Clinical-image internal alignment failed: {len(merged)}")
    if not (merged["label"].astype(int) == merged["image_label"].astype(int)).all():
        raise RuntimeError("Clinical-image labels do not align")
    return merged.drop(columns=["image_label"])


def complementarity(df: pd.DataFrame, clinical_col: str, image_col: str) -> dict:
    y = df["label"].to_numpy(dtype=int)
    pc = df[clinical_col].to_numpy(float)
    pi = df[image_col].to_numpy(float)
    cc = (pc >= 0.5).astype(int) == y
    ic = (pi >= 0.5).astype(int) == y
    return {
        "pearson": float(np.corrcoef(pc, pi)[0, 1]),
        "spearman": float(pd.Series(pc).rank().corr(pd.Series(pi).rank())),
        "disagreement_at_0_5": int(((pc >= 0.5).astype(int) != (pi >= 0.5).astype(int)).sum()),
        "clinical_correct_image_correct": int((cc & ic).sum()),
        "clinical_correct_image_wrong": int((cc & ~ic).sum()),
        "clinical_wrong_image_correct": int((~cc & ic).sum()),
        "clinical_wrong_image_wrong": int((~cc & ~ic).sum()),
    }


def run_fusion(
    name: str,
    out_dir: Path,
    aligned: pd.DataFrame,
    clinical_col: str,
    meta_inner: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, LogisticRegression, str]:
    pred_df = aligned.copy()
    pred_col = f"P_{name}_multimodal_COP"
    eq_col = f"P_{name}_equal_COP"
    pred_df[pred_col] = np.nan
    pred_df[eq_col] = 0.5 * pred_df[clinical_col] + 0.5 * pred_df["P_image_COP"]
    rows = []
    for outer in range(OUTER_FOLDS):
        train_mask = pred_df["outer_fold"].astype(int) != outer
        test_mask = ~train_mask
        train_ids = pred_df.loc[train_mask, "neutral_patient_id"].tolist()
        inner_clinical = (
            meta_inner[outer]["core"]
            if clinical_col == "P_core_COP"
            else meta_inner[outer]["gated"]
        )
        X_train = pd.DataFrame(
            {
                clinical_col: [inner_clinical.loc[c] for c in train_ids],
                "P_image_COP": pred_df.loc[train_mask, "P_image_COP"].to_numpy(float),
            }
        )
        y_train = pred_df.loc[train_mask, "label"].to_numpy(dtype=int)
        model = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", class_weight=None, max_iter=1000)
        model.fit(X_train[[clinical_col, "P_image_COP"]].to_numpy(), y_train)
        pred_df.loc[test_mask, pred_col] = model.predict_proba(
            pred_df.loc[test_mask, [clinical_col, "P_image_COP"]].to_numpy()
        )[:, 1]
        rows.append(
            {
                "outer_fold": outer,
                "train_n": int(train_mask.sum()),
                "test_n": int(test_mask.sum()),
                "clinical_inner_oof_complete": bool(np.isfinite(X_train[clinical_col]).all()),
                "image_inner_oof_complete": True,
                "image_inner_oof_source": "V3_6_PRIMARY_D1_STRICT_NESTED_FOLD_AUDIT_REUSE",
                "independent_d1_inner_probability_file_found": False,
                "fusion_beta_clinical": float(model.coef_[0][0]),
                "fusion_beta_image": float(model.coef_[0][1]),
                "fusion_intercept": float(model.intercept_[0]),
            }
        )
    final = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", class_weight=None, max_iter=1000)
    final.fit(pred_df[[clinical_col, "P_image_COP"]].to_numpy(), pred_df["label"].to_numpy(dtype=int))
    write_csv(
        pd.DataFrame(rows),
        out_dir / "05_strict_nested_fusion" / f"V3_6_{name.upper()}_META_FOLD_MODELS.csv",
    )
    pred_sha = write_csv(
        pred_df[
            [
                "neutral_patient_id",
                "outer_fold",
                "label",
                clinical_col,
                "P_image_COP",
                pred_col,
                eq_col,
            ]
        ],
        out_dir
        / "05_strict_nested_fusion"
        / f"V3_6_{name.upper()}_INTERNAL375_STRICT_NESTED_MULTIMODAL_OOF.csv",
    )
    return pred_df, pd.DataFrame(rows), final, pred_sha


def metric_table(branch: str, y, clinical, image, multimodal, equal) -> pd.DataFrame:
    rows = []
    for model, prob in [
        ("Clinical", clinical),
        ("Image_D1", image),
        (branch, multimodal),
        ("Equal_weight", equal),
    ]:
        rows.append({"model": model, **metrics(y, prob)})
    return pd.DataFrame(rows)


def save_final_clinical_pipelines(internal: pd.DataFrame) -> dict:
    X = internal[FULL_FEATURES].copy()
    y = internal["label_binary"].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
    fitted = {}
    records = {}
    for model_name in MODEL_SPECS:
        estimator, record = fit_search(X, y, model_name, cv)
        fitted[model_name] = estimator
        records[model_name] = record
    core = {
        "model_id": "core_ensemble",
        "members": {
            "core_logistic_robust": fitted["core_logistic_robust"],
            "core_random_forest": fitted["core_random_forest"],
        },
        "features": CORE_FEATURES,
        "probability": "P(COP)",
        "ensemble": "equal_weight_mean",
    }
    full = {
        "model_id": "full_ensemble",
        "members": {
            "full_logistic_robust": fitted["full_logistic_robust"],
            "full_random_forest": fitted["full_random_forest"],
        },
        "features": FULL_FEATURES,
        "probability": "P(COP)",
        "ensemble": "equal_weight_mean",
    }
    gated = {
        "model_id": "availability_gated_model",
        "full_ensemble": full,
        "core_ensemble": core,
        "full_only_features": FULL_ONLY_FEATURES,
        "routing_rule": "available_full_only_count >= 6 -> full_ensemble else core_ensemble",
        "available_definition": "notna()",
        "probability": "P(COP)",
    }
    paths = {
        "core": S3 / "07_final_clinical_pipelines" / "V3_6_S3_CORE_FINAL_PIPELINE.joblib",
        "full": S4 / "07_final_clinical_pipelines" / "V3_6_S4_FULL_FINAL_PIPELINE.joblib",
        "gated": S4 / "07_final_clinical_pipelines" / "V3_6_S4_GATED_FINAL_PIPELINE.joblib",
    }
    joblib.dump(core, paths["core"])
    joblib.dump(full, paths["full"])
    joblib.dump(gated, paths["gated"])
    return {
        "fitted": fitted,
        "records": records,
        "paths": {k: str(v) for k, v in paths.items()},
        "sha": {k: sha(v) for k, v in paths.items()},
    }


def predict_ensemble(fitted: dict, X: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "core":
        lr = fitted["core_logistic_robust"].predict_proba(candidate_input(X, "core_logistic_robust"))[:, 1]
        rf = fitted["core_random_forest"].predict_proba(candidate_input(X, "core_random_forest"))[:, 1]
        return (lr + rf) / 2
    lr = fitted["full_logistic_robust"].predict_proba(candidate_input(X, "full_logistic_robust"))[:, 1]
    rf = fitted["full_random_forest"].predict_proba(candidate_input(X, "full_random_forest"))[:, 1]
    return (lr + rf) / 2


def write_protocol_locks() -> None:
    base = {
        "fusion_model": "L2 Logistic Regression",
        "penalty": "L2",
        "C": 1.0,
        "class_weight": None,
        "threshold": 0.5,
        "probability": "P(COP)",
        "external_weight_tuning": False,
        "external_calibration": False,
        "external_threshold_tuning": False,
        "external_model_selection": False,
        "external_used_in_fit": False,
        "image_branch": "V3.5-D1 PYRADIOMICS_STANDARDIZED_RF",
        "status": "PROTOCOL_LOCKED_BEFORE_EXTERNAL_LABEL_ACCESS",
        "timestamp_utc": now(),
    }
    s3 = {**base, "task": "V3.6-S3 CORE_ENSEMBLE + V3.5-D1", "clinical_branch": "core_ensemble"}
    s4 = {**base, "task": "V3.6-S4 AVAILABILITY_GATED_MODEL + V3.5-D1", "clinical_branch": "availability_gated_model"}
    write_json(S3 / "00_protocol" / "V3_6_S3_PROTOCOL_LOCK.json", s3)
    write_json(S4 / "00_protocol" / "V3_6_S4_PROTOCOL_LOCK.json", s4)


def external_stage(
    branch: str,
    out_dir: Path,
    clinical_col: str,
    final_fusion: LogisticRegression,
    fitted: dict,
    external_features: pd.DataFrame,
    d1_ext: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, str, str]:
    if len(external_features) != 33 or len(d1_ext) != 33:
        raise RuntimeError("External33 row count mismatch")
    X_ext = external_features[FULL_FEATURES].copy()
    full_prob = predict_ensemble(fitted, X_ext, "full")
    core_prob = predict_ensemble(fitted, X_ext, "core")
    available = X_ext[FULL_ONLY_FEATURES].notna().sum(axis=1).astype(int).to_numpy()
    route = np.where(available >= 6, "full_ensemble", "core_ensemble")
    gated_prob = np.where(available >= 6, full_prob, core_prob)
    p_image = d1_ext["probability_COP"].to_numpy(float)
    neutral = d1_ext["neutral_patient_id"].astype(str).to_numpy()
    clinical_prob = core_prob if clinical_col == "P_core_COP" else gated_prob
    base = pd.DataFrame(
        {
            "neutral_patient_id": neutral,
            "available_full_only_count": available,
            "clinical_route": route,
            clinical_col: clinical_prob,
            "P_full_COP": full_prob,
            "P_core_COP": core_prob,
            "P_gated_COP": gated_prob,
            "P_image_COP": p_image,
            "external_alignment_method": "ROW_ORDER_WITH_D1_EXTERNAL33_LABEL_BLIND",
        }
    )
    base_cols = ["neutral_patient_id", clinical_col, "P_image_COP", "external_alignment_method"]
    if branch == "S4":
        base_cols = [
            "neutral_patient_id",
            "available_full_only_count",
            "clinical_route",
            clinical_col,
            "P_full_COP",
            "P_core_COP",
            "P_image_COP",
            "external_alignment_method",
        ]
    base_sha = write_csv(
        base[base_cols],
        out_dir / "09_external_label_blind" / f"V3_6_{branch}_EXTERNAL33_BASE_PROBABILITIES_LABEL_BLIND.csv",
    )
    pred = base.copy()
    pred[f"P_{branch.lower()}_multimodal_COP"] = final_fusion.predict_proba(
        pred[[clinical_col, "P_image_COP"]].to_numpy()
    )[:, 1]
    pred[f"P_{branch.lower()}_equal_COP"] = 0.5 * pred[clinical_col] + 0.5 * pred["P_image_COP"]
    pred["prediction_0_5"] = (pred[f"P_{branch.lower()}_multimodal_COP"] >= 0.5).astype(int)
    pred_cols = [
        "neutral_patient_id",
        f"P_{branch.lower()}_multimodal_COP",
        f"P_{branch.lower()}_equal_COP",
        "prediction_0_5",
    ]
    pred_sha = write_csv(
        pred[pred_cols],
        out_dir / "09_external_label_blind" / f"V3_6_{branch}_EXTERNAL33_MULTIMODAL_PREDICTIONS_LABEL_BLIND.csv",
    )
    lock = {
        "status": f"V3_6_{branch}_EXTERNAL33_PREDICTIONS_LOCKED_BEFORE_LABEL_READ",
        "timestamp_utc": now(),
        "n": 33,
        "base_probability_manifest_sha256": base_sha,
        "multimodal_prediction_sha256": pred_sha,
        "external_labels_read_by_script_before_lock": False,
        "external_weight_tuning": False,
        "external_calibration": False,
        "external_threshold_tuning": False,
        "external_model_selection": False,
        "probability": "P(COP)",
        "threshold": 0.5,
    }
    lock_path = out_dir / "09_external_label_blind" / f"V3_6_{branch}_EXTERNAL33_PREDICTION_LOCK.json"
    lock_sha = write_json(lock_path, lock)
    return base, pred, lock, str(lock_path), lock_sha


def external_metrics(
    branch: str,
    out_dir: Path,
    pred: pd.DataFrame,
    clinical_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    labels = pd.read_csv(EXTERNAL_CLINICAL, usecols=["label"])["label"].astype(int).to_numpy()
    d1_labels = pd.read_csv(D1_EXT_WITH_LABEL)["label_binary"].astype(int).to_numpy()
    if not np.array_equal(labels, d1_labels):
        raise RuntimeError("External clinical labels and D1 labels do not match by row order")
    if len(labels) != 33 or labels.sum() != 14:
        raise RuntimeError("External label contract is not 33 rows / 14 COP")
    multimodal_col = f"P_{branch.lower()}_multimodal_COP"
    equal_col = f"P_{branch.lower()}_equal_COP"
    rows = []
    for model, prob in [
        ("Clinical", pred[clinical_col].to_numpy(float)),
        ("Image_D1", pred["P_image_COP"].to_numpy(float)),
        (branch, pred[multimodal_col].to_numpy(float)),
        ("Equal_weight", pred[equal_col].to_numpy(float)),
    ]:
        row = {"model": model, **metrics(labels, prob)}
        row["AUROC_95CI"] = json.dumps(bootstrap_ci(labels, prob, "AUROC"))
        row["AUPRC_95CI"] = json.dumps(bootstrap_ci(labels, prob, "AUPRC"))
        row["Brier_95CI"] = json.dumps(bootstrap_ci(labels, prob, "Brier"))
        rows.append(row)
    table = pd.DataFrame(rows)
    write_csv(table, out_dir / "10_external_results" / f"V3_6_{branch}_EXTERNAL33_MODEL_COMPARISON_METRICS.csv")
    write_csv(
        pred.assign(label=labels),
        out_dir / "10_external_results" / f"V3_6_{branch}_EXTERNAL33_MULTIMODAL_PREDICTIONS_WITH_LABEL_PRIVATE.csv",
    )
    p_new = pred[multimodal_col].to_numpy(float)
    p_ref = pred[clinical_col].to_numpy(float)
    deltas = pd.DataFrame(
        [
            {
                "comparison": f"{branch}_minus_Clinical",
                "Delta_AUROC": float(roc_auc_score(labels, p_new) - roc_auc_score(labels, p_ref)),
                "Delta_AUROC_95CI": json.dumps(delta_ci(labels, p_new, p_ref, "AUROC")),
                "Delta_AUPRC": float(average_precision_score(labels, p_new) - average_precision_score(labels, p_ref)),
                "Delta_AUPRC_95CI": json.dumps(delta_ci(labels, p_new, p_ref, "AUPRC")),
                "Delta_Brier": float(brier_score_loss(labels, p_new) - brier_score_loss(labels, p_ref)),
                "Delta_Brier_95CI": json.dumps(delta_ci(labels, p_new, p_ref, "Brier")),
            }
        ]
    )
    write_csv(deltas, out_dir / "10_external_results" / f"V3_6_{branch}_EXTERNAL33_PAIRED_BOOTSTRAP_DELTAS.csv")
    c_ok = (p_ref >= 0.5).astype(int) == labels
    m_ok = (p_new >= 0.5).astype(int) == labels
    rescue = {
        "clinical_wrong_multimodal_correct": int((~c_ok & m_ok).sum()),
        "clinical_correct_multimodal_wrong": int((c_ok & ~m_ok).sum()),
        "both_correct": int((c_ok & m_ok).sum()),
        "both_wrong": int((~c_ok & ~m_ok).sum()),
    }
    write_json(out_dir / "11_error_analysis" / f"V3_6_{branch}_EXTERNAL33_ERROR_RESCUE_ANALYSIS.json", rescue)
    write_csv(pd.DataFrame([rescue]), out_dir / "11_error_analysis" / f"V3_6_{branch}_EXTERNAL33_ERROR_RESCUE_ANALYSIS.csv")
    return table, deltas, rescue


def save_fusion_model(
    branch: str,
    out_dir: Path,
    final: LogisticRegression,
    clinical_col: str,
    clinical_pipeline_sha: str,
    internal_prediction_sha: str,
) -> tuple[str, str]:
    model_path = out_dir / "08_final_fusion" / f"V3_6_{branch}_FINAL_FUSION_MODEL.joblib"
    joblib.dump(final, model_path)
    model_sha = sha(model_path)
    lock = {
        "status": f"V3_6_{branch}_FINAL_FUSION_MODEL_LOCKED_BEFORE_EXTERNAL_LABEL_ACCESS",
        "timestamp_utc": now(),
        "model": "L2_LOGISTIC_REGRESSION",
        "C": 1.0,
        "class_weight": None,
        "input_order": [clinical_col, "P_image_COP"],
        "coefficients": [float(final.coef_[0][0]), float(final.coef_[0][1])],
        "intercept": float(final.intercept_[0]),
        "threshold": 0.5,
        "probability_direction": "P(COP)",
        "external_used_in_fit": False,
        "external_weight_tuning": False,
        "external_calibration": False,
        "external_threshold_tuning": False,
        "clinical_pipeline_sha256": clinical_pipeline_sha,
        "D1_prediction_provenance": str(PRIMARY_OOF),
        "internal_cross_fitted_input_sha256": internal_prediction_sha,
        "model_path": str(model_path),
        "model_sha256": model_sha,
    }
    lock_path = out_dir / "08_final_fusion" / f"V3_6_{branch}_FINAL_FUSION_MODEL_LOCK.json"
    return str(lock_path), write_json(lock_path, lock)


def copy_script() -> None:
    src = Path(__file__)
    shutil.copy2(src, S4 / "00_protocol" / src.name)


def append_registry(rows: list[dict]) -> None:
    with REGISTRY.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        existing = list(reader)
    ids = {r["experiment_id"] for r in rows}
    existing = [r for r in existing if r.get("experiment_id") not in ids]
    output_rows = []
    for row in rows:
        full = {field: "" for field in fields}
        full.update(row)
        output_rows.append(full)
    with REGISTRY.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing + output_rows)


def update_roadmap(s3_status: str, s4_status: str, s3_auc, s4_auc) -> None:
    marker = "V3.6-S3 / S4 Recovered Remote Clinical Multimodal"
    section = f"""

## {marker}

- Scientific role: `SECONDARY_EXPLORATORY`, `PRE-DATA-COMPLETION INTERIM_MULTIMODAL_ANALYSIS`.
- Remote core pipeline: recovered as `LEVEL_C_PIPELINE_ONLY`, then reconstructed on internal375 for S3.
- V3.6-S3: `CORE_ENSEMBLE + V3.5-D1`; status `{s3_status}`; interim external AUROC `{s3_auc}`.
- V3.6-S4: `AVAILABILITY_GATED_MODEL + V3.5-D1`; status `{s4_status}`; interim external AUROC `{s4_auc}`.
- V3.6 Primary and V3.6-S2 historical records remain unchanged.
- V3.6-S1 and V3.6-S1-R historical not-executed/not-recovered-at-that-time records remain preserved.
- External33 clinical data completion remains pending; post-data-completion frozen reevaluation is required.
"""
    text = ROADMAP.read_text(encoding="utf-8", errors="replace")
    if marker in text:
        import re

        text = re.sub(rf"\n## {re.escape(marker)}\n.*?(?=\n## |\Z)", "\n" + section.strip() + "\n", text, flags=re.DOTALL)
        ROADMAP.write_text(text, encoding="utf-8", newline="\n")
    else:
        with ROADMAP.open("a", encoding="utf-8", newline="\n") as f:
            f.write(section)


def finalize_checksums(base: Path) -> str:
    path = base / "checksums.sha256"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for p in sorted(x for x in base.rglob("*") if x.is_file()):
            if p == path:
                continue
            f.write(f"{sha(p)}  {p.relative_to(base).as_posix()}\n")
    return sha(path)


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = [str(row[col]) for col in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    mkdirs()
    copy_script()
    write_protocol_locks()
    s3_contract_sha, s4_contract_sha = save_contracts()
    internal, external_features, split, primary, d1_ext = load_inputs()
    split_info = build_split_maps(split, primary)
    restored = restore_clinical_oof(internal, split_info)
    base = restored["base"]
    aligned = align_image(base, primary)

    s3_oof = aligned[
        ["neutral_patient_id", "outer_fold", "P_core_COP", "label"]
    ].copy()
    s3_oof["probability_COP"] = s3_oof.pop("P_core_COP")
    s3_oof["prediction_0_5"] = (s3_oof["probability_COP"] >= 0.5).astype(int)
    s3_oof = s3_oof[["neutral_patient_id", "outer_fold", "probability_COP", "prediction_0_5", "label"]]
    s3_oof_sha = write_csv(
        s3_oof,
        S3 / "02_clinical_oof_reconstruction" / "V3_6_S3_CORE_INTERNAL375_STRICT_OOF.csv",
    )
    s4_oof = aligned[
        [
            "neutral_patient_id",
            "outer_fold",
            "available_full_only_count",
            "clinical_route",
            "P_gated_COP",
            "label",
        ]
    ].copy()
    s4_oof["probability_COP"] = s4_oof.pop("P_gated_COP")
    s4_oof["prediction_0_5"] = (s4_oof["probability_COP"] >= 0.5).astype(int)
    s4_oof = s4_oof[
        [
            "neutral_patient_id",
            "outer_fold",
            "available_full_only_count",
            "clinical_route",
            "probability_COP",
            "prediction_0_5",
            "label",
        ]
    ]
    s4_oof_sha = write_csv(
        s4_oof,
        S4 / "02_clinical_oof_reconstruction" / "V3_6_S4_GATED_INTERNAL375_STRICT_OOF.csv",
    )
    write_csv(restored["fold_records"], S3 / "02_clinical_oof_reconstruction" / "V3_6_S3_S4_CLINICAL_OUTER_MODEL_SELECTION.csv")
    write_csv(restored["inner_records"], S3 / "02_clinical_oof_reconstruction" / "V3_6_S3_S4_CLINICAL_META_INNER_OOF_MODEL_SELECTION.csv")
    shutil.copy2(
        S3 / "02_clinical_oof_reconstruction" / "V3_6_S3_S4_CLINICAL_OUTER_MODEL_SELECTION.csv",
        S4 / "02_clinical_oof_reconstruction" / "V3_6_S3_S4_CLINICAL_OUTER_MODEL_SELECTION.csv",
    )
    shutil.copy2(
        S3 / "02_clinical_oof_reconstruction" / "V3_6_S3_S4_CLINICAL_META_INNER_OOF_MODEL_SELECTION.csv",
        S4 / "02_clinical_oof_reconstruction" / "V3_6_S3_S4_CLINICAL_META_INNER_OOF_MODEL_SELECTION.csv",
    )

    y = aligned["label"].to_numpy(dtype=int)
    core_m = metrics(y, aligned["P_core_COP"])
    gated_m = metrics(y, aligned["P_gated_COP"])
    core_delta = {
        "delta_AUROC": core_m["AUROC"] - HIST_CORE["AUROC"],
        "delta_AUPRC": core_m["AUPRC"] - HIST_CORE["AUPRC"],
        "delta_Brier": core_m["Brier"] - HIST_CORE["Brier"],
    }
    gated_delta = {"delta_AUROC": gated_m["AUROC"] - HIST_GATED["AUROC"]}
    s3_recon_pass = abs(core_delta["delta_AUROC"]) <= DISCREPANCY_AUROC
    s4_recon_pass = abs(gated_delta["delta_AUROC"]) <= DISCREPANCY_AUROC
    write_json(
        S3 / "02_clinical_oof_reconstruction" / "V3_6_S3_CORE_OOF_RECONSTRUCTION_AUDIT.json",
        {
            "status": "CORE_OOF_RECONSTRUCTION_PASS" if s3_recon_pass else "CORE_OOF_RECONSTRUCTION_DISCREPANCY",
            "metrics": core_m,
            "historical": HIST_CORE,
            **core_delta,
            "prediction_sha256": s3_oof_sha,
        },
    )
    write_json(
        S4 / "02_clinical_oof_reconstruction" / "V3_6_S4_GATED_OOF_RECONSTRUCTION_AUDIT.json",
        {
            "status": "GATED_OOF_RECONSTRUCTION_PASS" if s4_recon_pass else "GATED_OOF_RECONSTRUCTION_DISCREPANCY",
            "metrics": gated_m,
            "historical": HIST_GATED,
            **gated_delta,
            "prediction_sha256": s4_oof_sha,
            "internal_route_distribution": aligned["clinical_route"].value_counts().to_dict(),
        },
    )

    write_csv(aligned[["neutral_patient_id", "outer_fold", "label", "P_core_COP", "P_image_COP"]], S3 / "03_internal_alignment" / "V3_6_S3_CORE_D1_INTERNAL375_ALIGNMENT_AUDIT.csv")
    write_csv(aligned[["neutral_patient_id", "outer_fold", "label", "P_gated_COP", "P_image_COP", "available_full_only_count", "clinical_route"]], S4 / "03_internal_alignment" / "V3_6_S4_GATED_D1_INTERNAL375_ALIGNMENT_AUDIT.csv")
    s3_comp = complementarity(aligned, "P_core_COP", "P_image_COP")
    s4_comp = complementarity(aligned, "P_gated_COP", "P_image_COP")
    write_json(S3 / "04_complementarity" / "V3_6_S3_INTERNAL_COMPLEMENTARITY.json", s3_comp)
    write_csv(pd.DataFrame([s3_comp]), S3 / "04_complementarity" / "V3_6_S3_INTERNAL_COMPLEMENTARITY.csv")
    write_json(S4 / "04_complementarity" / "V3_6_S4_INTERNAL_COMPLEMENTARITY.json", s4_comp)
    write_csv(pd.DataFrame([s4_comp]), S4 / "04_complementarity" / "V3_6_S4_INTERNAL_COMPLEMENTARITY.csv")

    clinical_fit = save_final_clinical_pipelines(internal)

    branch_results = {}
    for branch, out_dir, clinical_col, recon_pass, clinical_pipeline_sha in [
        ("S3", S3, "P_core_COP", s3_recon_pass, clinical_fit["sha"]["core"]),
        ("S4", S4, "P_gated_COP", s4_recon_pass, clinical_fit["sha"]["gated"]),
    ]:
        if not recon_pass:
            branch_results[branch] = {"status": f"V3_6_{branch}_STOPPED_CLINICAL_OOF_RECONSTRUCTION_DISCREPANCY"}
            continue
        pred_df, fold_models, final_fusion, internal_pred_sha = run_fusion(
            branch.lower(),
            out_dir,
            aligned,
            clinical_col,
            restored["meta_inner"],
        )
        internal_table = metric_table(
            branch,
            y,
            pred_df[clinical_col].to_numpy(float),
            pred_df["P_image_COP"].to_numpy(float),
            pred_df[f"P_{branch.lower()}_multimodal_COP"].to_numpy(float),
            pred_df[f"P_{branch.lower()}_equal_COP"].to_numpy(float),
        )
        write_csv(internal_table, out_dir / "06_internal_results" / f"V3_6_{branch}_INTERNAL375_MODEL_COMPARISON_METRICS.csv")
        model_lock_path, model_lock_sha = save_fusion_model(
            branch,
            out_dir,
            final_fusion,
            clinical_col,
            clinical_pipeline_sha,
            internal_pred_sha,
        )
        base_ext, pred_ext, pred_lock, pred_lock_path, pred_lock_sha = external_stage(
            branch,
            out_dir,
            clinical_col,
            final_fusion,
            clinical_fit["fitted"],
            external_features,
            d1_ext,
        )
        external_table, deltas, rescue = external_metrics(branch, out_dir, pred_ext, clinical_col)
        branch_results[branch] = {
            "status": f"V3_6_{branch}_INTERIM_EXPLORATORY_PRE_DATA_COMPLETION_COMPLETE",
            "internal_table": internal_table,
            "external_table": external_table,
            "deltas": deltas,
            "rescue": rescue,
            "final_fusion_model_lock": model_lock_path,
            "final_fusion_model_lock_sha256": model_lock_sha,
            "external_prediction_lock": pred_lock_path,
            "external_prediction_lock_sha256": pred_lock_sha,
            "fusion_coefficients": [float(final_fusion.coef_[0][0]), float(final_fusion.coef_[0][1])],
            "fusion_intercept": float(final_fusion.intercept_[0]),
        }

    s3_ext_auc = None
    s4_ext_auc = None
    registry_rows = []
    for branch, out_dir, clinical_name, clinical_col, recon_m, comp, contract_sha in [
        ("S3", S3, "Core ensemble", "P_core_COP", core_m, s3_comp, s3_contract_sha),
        ("S4", S4, "Availability-gated", "P_gated_COP", gated_m, s4_comp, s4_contract_sha),
    ]:
        result = branch_results[branch]
        if "external_table" in result:
            ext_tbl = result["external_table"]
            int_tbl = result["internal_table"]
            ext_main = ext_tbl[ext_tbl["model"] == branch].iloc[0].to_dict()
            int_main = int_tbl[int_tbl["model"] == branch].iloc[0].to_dict()
            ext_clin = ext_tbl[ext_tbl["model"] == "Clinical"].iloc[0].to_dict()
            delta_row = result["deltas"].iloc[0].to_dict()
            if branch == "S3":
                s3_ext_auc = ext_main["AUROC"]
            else:
                s4_ext_auc = ext_main["AUROC"]
            lock = {
                "status": result["status"],
                "timestamp_utc": now(),
                "scientific_role": "SECONDARY_EXPLORATORY",
                "analysis_identity": "HISTORICAL_PIPELINE_RECOVERED_SECONDARY_EXPLORATORY_MULTIMODAL",
                "current_result_status": "INTERIM_EXPLORATORY_PRE_DATA_COMPLETION",
                "post_data_completion_requirement": "FROZEN_EXTERNAL_REEVALUATION_REQUIRED",
                "external_clinical_data": "PRE_DATA_COMPLETION_INTERIM",
                "clinical_model": clinical_name,
                "image_model": "V3.5-D1 PYRADIOMICS_STANDARDIZED_RF",
                "image_retraining": False,
                "fusion": "PATIENT_LEVEL_LATE_FUSION",
                "fusion_model": "L2_LOGISTIC_REGRESSION",
                "strict_nested_fusion": "PASS_WITH_V3_6_D1_PROJECT_LOCK_REUSE_LIMITATION",
                "independent_d1_inner_probability_file_found": False,
                "clinical_oof_reconstruction": recon_m,
                "clinical_image_alignment": "375/375",
                "external_n": 33,
                "external_weight_tuning": False,
                "external_calibration": False,
                "external_threshold_tuning": False,
                "external_model_selection": False,
                "final_fusion_model_lock_sha256": result["final_fusion_model_lock_sha256"],
                "external_prediction_lock_sha256": result["external_prediction_lock_sha256"],
                "contract_sha256": contract_sha,
                "internal_metrics": int_tbl.to_dict(orient="records"),
                "external_metrics": ext_tbl.to_dict(orient="records"),
                "external_delta_vs_clinical": delta_row,
                "error_rescue": result["rescue"],
                "complementarity": comp,
                "historical_v3_6_primary_modified": False,
                "historical_v3_6_s2_modified": False,
                "v3_6_s1_records_preserved": True,
            }
            result_lock_path = out_dir / "13_locks" / f"V3_6_{branch}_RESULT_LOCK.json"
            result_lock_sha = write_json(result_lock_path, lock)
            result["result_lock"] = str(result_lock_path)
            result["result_lock_sha256"] = result_lock_sha
            registry_rows.append(
                {
                    "stage_order": "36.63" if branch == "S3" else "36.64",
                    "experiment_id": f"V3_6_{branch}_RECOVERED_REMOTE_CLINICAL_D1_MULTIMODAL",
                    "branch": f"V3.6-{branch} recovered remote clinical + D1",
                    "scientific_question": f"Whether {clinical_name} combines with frozen V3.5-D1 under patient-level late fusion.",
                    "internal_n": "375",
                    "external_n": "33",
                    "input_modality": "clinical probability + image probability",
                    "ROI_source": "V3.5-D1 frozen radiomics probability",
                    "feature_type": f"{clinical_name} P(COP) + D1 P(COP)",
                    "model": "L2 logistic regression late fusion",
                    "internal_AUROC": int_main["AUROC"],
                    "internal_AUPRC": int_main["AUPRC"],
                    "internal_Brier": int_main["Brier"],
                    "external_AUROC": ext_main["AUROC"],
                    "external_AUPRC": ext_main["AUPRC"],
                    "external_Brier": ext_main["Brier"],
                    "external_analysis_identity": "INTERIM_EXPLORATORY_PRE_DATA_COMPLETION",
                    "external_used_for_development": "False",
                    "external_labels_used_for_development": "False",
                    "main_finding": f"{branch} external AUROC {ext_main['AUROC']} vs clinical {ext_clin['AUROC']}",
                    "failure_or_success_reason": "Recovered clinical pipeline reconstructed from internal375; external data completion pending.",
                    "status": result["status"],
                    "superseded_by": "not superseded",
                    "key_result_lock": str(result_lock_path),
                    "key_report": str(GOV / f"CHATGPT_HANDOFF_V3_6_{branch}.md"),
                    "key_prediction": str(out_dir / "09_external_label_blind" / f"V3_6_{branch}_EXTERNAL33_MULTIMODAL_PREDICTIONS_LABEL_BLIND.csv"),
                    "SHA256": result_lock_sha,
                    "paper_role": "SECONDARY_INTERIM_EXPLORATORY",
                }
            )
        else:
            write_json(
                out_dir / "13_locks" / f"V3_6_{branch}_STOP_LOCK.json",
                {"status": result["status"], "timestamp_utc": now()},
            )

    append_registry(registry_rows)
    update_roadmap(
        branch_results["S3"]["status"],
        branch_results["S4"]["status"],
        s3_ext_auc,
        s4_ext_auc,
    )

    for branch, out_dir, title in [
        ("S3", S3, "Core pipeline contract"),
        ("S4", S4, "Availability-gated pipeline contract"),
    ]:
        result = branch_results[branch]
        if "external_table" in result:
            int_csv = result["internal_table"].to_csv(index=False)
            ext_csv = result["external_table"].to_csv(index=False)
            delta_csv = result["deltas"].to_csv(index=False)
            rescue = result["rescue"]
            report = f"""# ChatGPT Handoff V3.6-{branch}

{title}: see `{out_dir / '01_clinical_contract'}`.

Strict OOF reconstruction: PASS.

Internal model comparison:

```csv
{int_csv.strip()}
```

External model comparison:

```csv
{ext_csv.strip()}
```

Paired bootstrap delta:

```csv
{delta_csv.strip()}
```

Error rescue/harm: {json.dumps(rescue, ensure_ascii=False)}.

Fusion coefficients: {result['fusion_coefficients']}; intercept {result['fusion_intercept']}.

Prediction lock: `{result['external_prediction_lock']}`.

Final result lock: `{result['result_lock']}`.

Final result lock SHA256: `{result['result_lock_sha256']}`.
"""
        else:
            report = f"# ChatGPT Handoff V3.6-{branch}\n\nStopped: {result['status']}.\n"
        write_md(GOV / f"CHATGPT_HANDOFF_V3_6_{branch}.md", report)

    combined_rows = []
    combined_rows.append({"model": "Clinical V2", "multimodal": "Primary", "internal_AUROC": 0.84, "external_AUROC": 0.684211, "status": "FORMAL_PRIMARY_WITH_RECORDED_PREFLIGHT_LIMITATION"})
    combined_rows.append({"model": "StructuralMissing", "multimodal": "S2", "internal_AUROC": 0.8488, "external_AUROC": 0.691729, "status": "SECONDARY_EXPLORATORY_GOVERNANCE_LIMITATION"})
    for branch, clinical_name in [("S3", "Core ensemble"), ("S4", "Availability-gated")]:
        result = branch_results[branch]
        if "external_table" in result:
            int_main = result["internal_table"][result["internal_table"]["model"] == branch].iloc[0]
            ext_main = result["external_table"][result["external_table"]["model"] == branch].iloc[0]
            combined_rows.append(
                {
                    "model": clinical_name,
                    "multimodal": branch,
                    "internal_AUROC": int_main["AUROC"],
                    "external_AUROC": ext_main["AUROC"],
                    "status": "INTERIM_EXPLORATORY_PRE_DATA_COMPLETION",
                }
            )
    combined = pd.DataFrame(combined_rows)
    combined_path = GOV / "CHATGPT_HANDOFF_V3_6_S3_S4_MULTIMODAL.md"
    write_md(combined_path, "# ChatGPT Handoff V3.6 S3/S4 Multimodal\n\n" + markdown_table(combined) + "\n")

    s3_checksum = finalize_checksums(S3)
    s4_checksum = finalize_checksums(S4)

    console = {
        "external_clinical_data": "PRE-DATA-COMPLETION INTERIM",
        "image": "V3.5-D1 FROZEN",
        "s3": branch_results["S3"],
        "s4": branch_results["S4"],
        "core_metrics": core_m,
        "core_delta": core_delta,
        "gated_metrics": gated_m,
        "gated_delta": gated_delta,
        "s3_complementarity": s3_comp,
        "s4_complementarity": s4_comp,
        "s3_checksum_sha256": s3_checksum,
        "s4_checksum_sha256": s4_checksum,
        "combined_handoff": str(combined_path),
    }
    write_json(S3 / "13_locks" / "V3_6_S3_S4_FINAL_CONSOLE.json", console)
    print(json.dumps(console, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
