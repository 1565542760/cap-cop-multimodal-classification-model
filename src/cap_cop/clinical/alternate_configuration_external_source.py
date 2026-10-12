from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path(
    os.environ.get(
        "CAP_COP_LOCAL_EXPERIMENT_V2",
        PROJECT_ROOT / "models" / "historical_clinical_contracts" / "experiment_v2",
    )
)
V1 = Path(
    os.environ.get(
        "CAP_COP_LOCAL_EXPERIMENT_V1",
        PROJECT_ROOT / "models" / "historical_clinical_contracts" / "experiment_v1",
    )
)
OUT = ROOT / "external_evaluation" / "clinical_v2_transportability_exploratory"
REPORT = OUT
LOCK_FILE = ROOT / "locked_models" / "CLINICAL_BRANCH_LOCK_V2.json"
EXTERNAL_SOURCE = (
    V1
    / "cap_cop_experiment"
    / "data_locked"
    / "external_raw_locked_v1"
    / "doctor_completion"
    / "external_clinical_basic_doctor_fused_aligned.csv"
)
EXPECTED_EXTERNAL_SHA256 = "ace098cf65a05b330256161ed06ec0a012c6bd1eb18143ce6d586b713e55eccf"
INTERNAL_DEV = (
    V1
    / "cap_cop_experiment"
    / "generalisation_redevelopment_nested_protocol_v1"
    / "data"
    / "internal_development"
    / "clinical_features_basic_corrected_for_development.csv"
)
NESTED_SPLIT = ROOT / "splits" / "locked_nested_split_patient_manifest.csv"
CHECKSUMS = ROOT / "checksums.sha256"
SCRIPT_PATH = Path(__file__)

SEEDS = [20260627, 20260628, 20260629]
FOLDS = [0, 1, 2, 3, 4]
BOOTSTRAP_SEED = 20260816
N_BOOT = 5000
RF_PARAMS = {
    "missingness": "median_indicator",
    "n_estimators": 300,
    "max_depth": None,
    "max_features": "sqrt",
    "min_samples_leaf": 3,
    "class_weight": "balanced",
}

GENERATED: list[Path] = []
LABEL_READ_COUNT = 0


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    GENERATED.append(path)


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")
    GENERATED.append(path)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    GENERATED.append(path)


def append_checksums(paths: list[Path]) -> None:
    existing = CHECKSUMS.read_text(encoding="utf-8", errors="ignore") if CHECKSUMS.exists() else ""
    lines = []
    for p in paths:
        if p.exists() and p.is_file():
            try:
                rel = p.relative_to(ROOT).as_posix()
            except ValueError:
                rel = str(p)
            line = f"{sha256_file(p)}  {rel}"
            if line not in existing:
                lines.append(line)
    if lines:
        with CHECKSUMS.open("a", encoding="utf-8", newline="\n") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            for line in lines:
                f.write(line + "\n")


def stop_report(filename: str, reason: str, details: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "STOPPED",
        "stage": "CLINICAL_V2_EXTERNAL_TRANSPORTABILITY_EXPLORATORY_EVALUATION",
        "reason": reason,
        "timestamp": now_iso(),
        "details": details,
        "label_read_count": LABEL_READ_COUNT,
        "model_modification_counts": model_modification_counts(),
    }
    path = ROOT / filename
    write_md(
        path,
        f"""# {filename}

Status: STOPPED

Reason: {reason}

```json
{json.dumps(payload, indent=2, ensure_ascii=False)}
```
""",
    )
    append_checksums(GENERATED + [SCRIPT_PATH])
    raise SystemExit(2)


def model_modification_counts() -> dict[str, int]:
    return {
        "retraining_after_external_label_read": 0,
        "recalibration_on_external": 0,
        "threshold_tuning_on_external": 0,
        "feature_selection_after_external": 0,
        "hyperparameter_tuning_after_external": 0,
        "algorithm_selection_after_external": 0,
        "multimodal_training_in_this_stage": 0,
        "imaging_training_in_this_stage": 0,
        "clinical_branch_modifications": 0,
    }


def load_lock() -> tuple[dict[str, Any], str]:
    if not LOCK_FILE.exists():
        stop_report("STOP_REPORT_CLINICAL_EXTERNAL_LOCK_INVALID.md", "clinical_branch_lock_missing", {"path": str(LOCK_FILE)})
    lock_hash = sha256_file(LOCK_FILE)
    lock = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    failures = []
    if lock.get("status") != "CLINICAL_BRANCH_V2_LOCKED":
        failures.append("status_not_CLINICAL_BRANCH_V2_LOCKED")
    if lock.get("selected_branch_algorithm") != "RF":
        failures.append("selected_branch_not_RF")
    for key in ["exact_features", "missingness_strategy", "calibration_strategy", "split_hash", "source_code_hash", "training_data_hash"]:
        if not lock.get(key):
            failures.append(f"missing_{key}")
    if failures:
        stop_report(
            "STOP_REPORT_CLINICAL_EXTERNAL_LOCK_INVALID.md",
            "clinical_branch_lock_invalid",
            {"failures": failures, "lock_sha256": lock_hash},
        )
    return lock, lock_hash


def verify_external_hash() -> str:
    if not EXTERNAL_SOURCE.exists():
        stop_report(
            "STOP_REPORT_EXTERNAL_CLINICAL_HASH_MISMATCH.md",
            "external_clinical_source_missing",
            {"path": str(EXTERNAL_SOURCE)},
        )
    got = sha256_file(EXTERNAL_SOURCE)
    if got.lower() != EXPECTED_EXTERNAL_SHA256:
        stop_report(
            "STOP_REPORT_EXTERNAL_CLINICAL_HASH_MISMATCH.md",
            "external_clinical_hash_mismatch",
            {"path": str(EXTERNAL_SOURCE), "expected": EXPECTED_EXTERNAL_SHA256, "observed": got},
        )
    return got


def read_external_header() -> list[str]:
    with EXTERNAL_SOURCE.open("r", encoding="utf-8-sig", newline="") as f:
        return next(csv.reader(f))


def make_rf_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=RF_PARAMS["n_estimators"],
                    max_depth=RF_PARAMS["max_depth"],
                    min_samples_leaf=RF_PARAMS["min_samples_leaf"],
                    max_features=RF_PARAMS["max_features"],
                    class_weight=RF_PARAMS["class_weight"],
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ]
    )


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt_from_seed_oof(raw: pd.Series, y: pd.Series) -> tuple[LogisticRegression, dict[str, Any]]:
    cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal.fit(logit(raw.to_numpy()).reshape(-1, 1), y.loc[raw.index].to_numpy())
    return cal, {
        "calibrator": "PlattLogisticRegressionOnInternal5FoldOOFRawProbabilities",
        "n_internal_oof_patients": int(len(raw)),
        "coef": float(cal.coef_[0][0]),
        "intercept": float(cal.intercept_[0]),
        "raw_oof_brier": float(brier_score_loss(y.loc[raw.index], raw.to_numpy())),
    }


def apply_calibrator(cal: LogisticRegression, raw_prob: np.ndarray) -> np.ndarray:
    return cal.predict_proba(logit(raw_prob).reshape(-1, 1))[:, 1]


def safe_auroc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr.fit(logit(np.asarray(p)).reshape(-1, 1), np.asarray(y, dtype=int))
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def metric_dict(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    intercept, slope = calibration_intercept_slope(y, p)
    return {
        "AUROC": safe_auroc(y, p),
        "AUPRC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "sensitivity_at_0_5": sens,
        "specificity_at_0_5": spec,
        "f1_at_0_5": float(f1_score(y, pred, zero_division=0)),
        "ppv_at_0_5": ppv,
        "npv_at_0_5": npv,
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(y)
    metrics = ["AUROC", "AUPRC", "Brier", "sensitivity_at_0_5", "specificity_at_0_5", "f1_at_0_5", "calibration_intercept", "calibration_slope"]
    values = {m: [] for m in metrics}
    skipped = CounterDict()
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        pp = p[idx]
        md = metric_dict(yy, pp)
        for m in metrics:
            if np.isfinite(md[m]):
                values[m].append(md[m])
            else:
                skipped[m] += 1
    rows = []
    point = metric_dict(y, p)
    for m in metrics:
        arr = np.asarray(values[m], dtype=float)
        rows.append(
            {
                "metric": m,
                "point_estimate": point[m],
                "ci_low_95": float(np.percentile(arr, 2.5)) if arr.size else float("nan"),
                "ci_high_95": float(np.percentile(arr, 97.5)) if arr.size else float("nan"),
                "bootstrap_resamples": N_BOOT,
                "valid_resamples": int(arr.size),
                "skipped_resamples": int(skipped[m]),
                "bootstrap_unit": "patient",
                "seed": BOOTSTRAP_SEED,
                "note": "calibration CI exploratory due to small external sample size" if m.startswith("calibration") else "",
            }
        )
    return pd.DataFrame(rows)


class CounterDict(dict):
    def __missing__(self, key):
        self[key] = 0
        return 0


def parse_label_value(value: Any) -> int | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s in {"0", "0.0"}:
        return 0
    if s in {"1", "1.0"}:
        return 1
    u = s.upper()
    if u == "CAP":
        return 0
    if u == "COP":
        return 1
    return None


def feature_availability(external_x: pd.DataFrame, internal_x: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    type_rules = {
        "age": (0, 120),
        "height_cm": (100, 230),
        "weight_kg": (20, 250),
        "body_mass_index": (10, 80),
        "fever": (0, 1),
        "cough": (0, 1),
        "chest_tightness": (0, 1),
        "dyspnoea": (0, 1),
        "haemoptysis": (0, 1),
        "weight_loss": (0, 1),
    }
    for f in features:
        x = pd.to_numeric(external_x[f], errors="coerce")
        internal = pd.to_numeric(internal_x[f], errors="coerce")
        nonmiss = x.dropna()
        impossible = 0
        if f in type_rules:
            lo, hi = type_rules[f]
            impossible = int(((nonmiss < lo) | (nonmiss > hi)).sum())
        dev_min, dev_max = float(internal.min()), float(internal.max())
        out_range = int(((nonmiss < dev_min) | (nonmiss > dev_max)).sum())
        rows.append(
            {
                "feature": f,
                "n_available": int(nonmiss.shape[0]),
                "n_missing": int(x.isna().sum()),
                "missing_rate": float(x.isna().mean()),
                "external_min": float(nonmiss.min()) if len(nonmiss) else float("nan"),
                "external_max": float(nonmiss.max()) if len(nonmiss) else float("nan"),
                "development_min": dev_min,
                "development_max": dev_max,
                "impossible_value_count": impossible,
                "out_of_development_range_count": out_range,
                "dtype_convertible": bool(x.notna().sum() + external_x[f].isna().sum() == len(external_x)),
                "used_for_model_modification": 0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    start = now_iso()
    access_log_rows = []

    lock, lock_hash = load_lock()
    external_hash = verify_external_hash()
    features = list(lock["exact_features"])
    if lock.get("selected_branch_algorithm") != "RF":
        stop_report("STOP_REPORT_CLINICAL_EXTERNAL_LOCK_INVALID.md", "only_RF_allowed", {"selected": lock.get("selected_branch_algorithm")})

    header = read_external_header()
    required_stage1 = ["patient_id", *features]
    missing = [c for c in required_stage1 if c not in header]
    if missing:
        stop_report("STOP_REPORT_EXTERNAL_FEATURE_CONTRACT_MISMATCH.md", "external_required_feature_missing", {"missing": missing})
    if "label" not in header:
        stop_report("STOP_REPORT_EXTERNAL_LABEL_CONTRACT_MISMATCH.md", "external_label_column_missing_for_stage2", {"columns": header})

    write_md(
        OUT / "00_EXTERNAL_EVALUATION_PROTOCOL.md",
        f"""# External Transportability Exploratory Evaluation Protocol

Formal name: `CLINICAL_V2_EXTERNAL_TRANSPORTABILITY_EXPLORATORY_EVALUATION`.

Scientific positioning: external-cohort-informed exploratory evaluation of the locked clinical model.

This is not a new model-development experiment and is not described as independent external validation. Only the locked RF clinical branch is evaluated. External results must not modify RF features, hyperparameters, missingness, calibration, threshold, clinical branch selection, imaging, or multimodal plans.

Two-stage blind workflow:

1. Stage 1 reads only `patient_id` and locked RF predictor columns, fits the fixed full-development RF ensemble on internal 375 patients, generates and hashes blinded external probabilities.
2. Stage 2 occurs only after prediction lock hashing. It performs one planned external label-read event and computes metrics once.
""",
    )

    # STAGE 1: label-blind prediction lock.
    external_raw = pd.read_csv(EXTERNAL_SOURCE, usecols=required_stage1, dtype=str, encoding="utf-8-sig")
    access_log_rows.append(
        {
            "timestamp": now_iso(),
            "stage": "STAGE_1_LABEL_BLIND_PREDICTION_LOCK",
            "file": str(EXTERNAL_SOURCE),
            "usecols": json.dumps(required_stage1, ensure_ascii=False),
            "label_access": 0,
            "prediction_access": 0,
            "metric_access": 0,
            "model_modified_after_access": 0,
        }
    )
    if len(external_raw) != 33:
        stop_report("STOP_REPORT_EXTERNAL_PATIENT_CONTRACT_FAILED.md", "external_patient_count_not_33_before_label_read", {"n": int(len(external_raw))})
    if external_raw["patient_id"].duplicated().any():
        stop_report(
            "STOP_REPORT_EXTERNAL_PATIENT_CONTRACT_FAILED.md",
            "external_patient_id_duplicate_before_label_read",
            {"duplicates": external_raw.loc[external_raw["patient_id"].duplicated(), "patient_id"].map(sha256_text).tolist()},
        )
    external_raw = external_raw.reset_index(drop=True)
    external_raw["neutral_patient_id"] = [f"EXTV2P{i:03d}" for i in range(1, len(external_raw) + 1)]
    external_x = external_raw[features].apply(pd.to_numeric, errors="coerce")

    internal = pd.read_csv(INTERNAL_DEV)
    nested = pd.read_csv(NESTED_SPLIT)
    internal_x = internal.set_index("case_id_public", drop=False)[features].apply(pd.to_numeric, errors="coerce")
    y = internal.set_index("case_id_public", drop=False)["label_binary"].astype(int)
    if len(internal) != 375 or int((y == 0).sum()) != 250 or int((y == 1).sum()) != 125:
        stop_report(
            "STOP_REPORT_INTERNAL_DEVELOPMENT_CONTRACT_FAILED.md",
            "internal_development_contract_failed",
            {"n": int(len(internal)), "counts": y.value_counts().to_dict()},
        )

    availability = feature_availability(external_x, internal_x, features)
    write_csv(OUT / "external_feature_availability_audit.csv", availability)

    seed_probs = {}
    calibrator_info = []
    oof_registry_rows = []
    for seed in SEEDS:
        raw_oof = pd.Series(index=internal_x.index, dtype=float)
        split_seed = nested[nested["training_seed"].eq(seed)]
        for fold in FOLDS:
            df = split_seed[split_seed["outer_fold_id"].eq(fold)]
            train_ids = df.loc[df["outer_role"].eq("outer_train"), "case_id_public"].tolist()
            test_ids = df.loc[df["outer_role"].eq("outer_test"), "case_id_public"].tolist()
            if set(train_ids) & set(test_ids):
                stop_report("STOP_REPORT_INTERNAL_SPLIT_CONTRACT_FAILED.md", "outer_train_test_overlap", {"seed": seed, "fold": fold})
            model = make_rf_pipeline(seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(internal_x.loc[train_ids], y.loc[train_ids])
            raw_oof.loc[test_ids] = model.predict_proba(internal_x.loc[test_ids])[:, 1]
            oof_registry_rows.append({"training_seed": seed, "outer_fold_id": fold, "train_n": len(train_ids), "test_n": len(test_ids)})
        if raw_oof.isna().any():
            stop_report("STOP_REPORT_DEPLOYMENT_STRATEGY_FAILED.md", "internal_5fold_raw_oof_incomplete", {"seed": seed})
        cal, info = fit_platt_from_seed_oof(raw_oof, y)
        info["training_seed"] = seed
        calibrator_info.append(info)
        full_model = make_rf_pipeline(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full_model.fit(internal_x, y)
        raw_ext = full_model.predict_proba(external_x)[:, 1]
        prob_ext = apply_calibrator(cal, raw_ext)
        if np.any((prob_ext < 0) | (prob_ext > 1)) or np.any(~np.isfinite(prob_ext)):
            stop_report("STOP_REPORT_EXTERNAL_PROBABILITY_CONTRACT_FAILED.md", "external_probability_out_of_range", {"seed": seed})
        seed_probs[seed] = prob_ext

    pred = pd.DataFrame({"neutral_patient_id": external_raw["neutral_patient_id"]})
    for seed in SEEDS:
        pred[f"probability_COP_seed_{seed}"] = seed_probs[seed]
    seed_cols = [f"probability_COP_seed_{seed}" for seed in SEEDS]
    pred["probability_COP_mean"] = pred[seed_cols].mean(axis=1)
    pred["available_feature_count"] = external_x.notna().sum(axis=1).astype(int)
    pred["missing_feature_count"] = external_x.isna().sum(axis=1).astype(int)
    pred["missing_feature_rate"] = external_x.isna().mean(axis=1).astype(float)
    pred["outcome_label_included"] = 0
    if len(pred) != 33:
        stop_report("STOP_REPORT_EXTERNAL_PREDICTION_CONTRACT_FAILED.md", "prediction_not_33", {"n": int(len(pred))})
    pred_path = OUT / "external_clinical_v2_rf_predictions_BLINDED_PRIVATE.csv"
    write_csv(pred_path, pred)
    pred_hash = sha256_file(pred_path)

    patient_audit = pd.DataFrame(
        {
            "neutral_patient_id": external_raw["neutral_patient_id"],
            "patient_id_sha256": external_raw["patient_id"].map(sha256_text),
            "source_rows_collapsed_to_patient": 1,
            "unique_patient": 1,
            "label_read_in_stage1": 0,
        }
    )
    write_csv(OUT / "external_patient_matching_audit_PRIVATE.csv", patient_audit)
    write_csv(OUT / "deployment_internal_oof_registry.csv", pd.DataFrame(oof_registry_rows))

    prediction_lock = {
        "status": "EXTERNAL_CLINICAL_V2_PREDICTION_LOCKED",
        "formal_name": "CLINICAL_V2_EXTERNAL_TRANSPORTABILITY_EXPLORATORY_EVALUATION",
        "created_at": now_iso(),
        "model_lock_hash": lock_hash,
        "clinical_source_hash": external_hash,
        "training_data_hash": sha256_file(INTERNAL_DEV),
        "nested_split_hash": sha256_file(NESTED_SPLIT),
        "source_code_hash": sha256_file(SCRIPT_PATH),
        "deployment_strategy": "fixed full-development RF ensemble fallback authorized by protocol",
        "rf_params": RF_PARAMS,
        "features": features,
        "calibrator_provenance": calibrator_info,
        "prediction_file": str(pred_path),
        "prediction_file_hash": pred_hash,
        "patient_count": 33,
        "label_access_before_prediction_lock": 0,
        "external_statistics_used_for_imputation": 0,
        "external_statistics_used_for_calibration": 0,
    }
    lock_path = OUT / "EXTERNAL_CLINICAL_V2_PREDICTION_LOCK.json"
    write_json(lock_path, prediction_lock)
    prediction_lock_hash = sha256_file(lock_path)

    write_md(
        OUT / "01_LOCKED_MODEL_DEPLOYMENT_AUDIT.md",
        f"""# Locked Model Deployment Audit

Status: PASS

- Clinical branch lock status: `CLINICAL_BRANCH_V2_LOCKED`
- Selected branch: RF
- Clinical branch lock SHA256: `{lock_hash}`
- External clinical source SHA256: `{external_hash}`
- Deployment strategy: fixed full-development RF ensemble, three seeds `{SEEDS}`
- RF config: `median_indicator`, `n_estimators=300`, `max_depth=None`, `max_features=sqrt`, `min_samples_leaf=3`, `class_weight=balanced`
- Calibration: Platt logistic calibrator fitted on internal 5-fold cross-fitted raw probabilities only
- External label access before prediction lock: 0
""",
    )
    write_md(
        OUT / "02_EXTERNAL_PATIENT_MATCHING_AUDIT.md",
        f"""# External Patient Matching Audit

Status before label read: PASS

- External rows read without label: 33
- Unique patient IDs: 33
- Source A/B treated as independent patients: no
- Prediction rows: 33
- Private matching audit: `{OUT / 'external_patient_matching_audit_PRIVATE.csv'}`
""",
    )
    top_missing = availability.sort_values(["missing_rate", "feature"], ascending=[False, True]).head(6)
    write_md(
        OUT / "03_EXTERNAL_FEATURE_AVAILABILITY_REPORT.md",
        "# External Feature Availability Report\n\n"
        "Status: label-blind descriptive audit only. These results were not used for model modification.\n\n"
        + top_missing.to_markdown(index=False),
    )
    write_md(
        OUT / "04_BLINDED_PREDICTION_LOCK_REPORT.md",
        f"""# Blinded Prediction Lock Report

Status: EXTERNAL_CLINICAL_V2_PREDICTION_LOCKED

- Prediction file: `{pred_path}`
- Prediction SHA256: `{pred_hash}`
- Prediction lock file: `{lock_path}`
- Prediction lock SHA256: `{prediction_lock_hash}`
- External labels included in blinded file: no
- Probability range valid: yes
""",
    )

    # Dry-run metrics before one planned label read.
    mock_y = y.iloc[:33].to_numpy()
    _ = metric_dict(mock_y, pred["probability_COP_mean"].to_numpy())

    # STAGE 2: one-time planned label read.
    global LABEL_READ_COUNT
    label_df = pd.read_csv(EXTERNAL_SOURCE, usecols=["patient_id", "label"], dtype=str, encoding="utf-8-sig")
    LABEL_READ_COUNT = 1
    access_log_rows.append(
        {
            "timestamp": now_iso(),
            "stage": "STAGE_2_ONE_TIME_LABEL_READ_AND_EVALUATION",
            "file": str(EXTERNAL_SOURCE),
            "usecols": json.dumps(["patient_id", "label"], ensure_ascii=False),
            "label_access": 1,
            "prediction_access": 0,
            "metric_access": 0,
            "model_modified_after_access": 0,
        }
    )
    if label_df["patient_id"].duplicated().any() or len(label_df) != 33:
        stop_report("STOP_REPORT_EXTERNAL_LABEL_CONTRACT_FAILED.md", "label_patient_contract_failed", {"n": int(len(label_df))})
    label_df["label_binary"] = label_df["label"].map(parse_label_value)
    if label_df["label_binary"].isna().any():
        stop_report("STOP_REPORT_EXTERNAL_LABEL_CONTRACT_FAILED.md", "unparseable_external_labels", {})
    counts = label_df["label_binary"].astype(int).value_counts().to_dict()
    if counts.get(0) != 19 or counts.get(1) != 14:
        stop_report("STOP_REPORT_EXTERNAL_LABEL_CONTRACT_FAILED.md", "external_label_counts_not_19_14", {"counts": counts})

    neutral_map = external_raw[["patient_id", "neutral_patient_id"]]
    labelled = label_df.merge(neutral_map, on="patient_id", how="inner").merge(pred, on="neutral_patient_id", how="inner")
    if len(labelled) != 33:
        stop_report("STOP_REPORT_EXTERNAL_LABEL_CONTRACT_FAILED.md", "label_prediction_merge_not_33", {"n": int(len(labelled))})
    labelled["disease_label"] = np.where(labelled["label_binary"].astype(int).eq(1), "COP", "CAP")
    with_label = labelled[
        ["neutral_patient_id", "label_binary", "disease_label", *seed_cols, "probability_COP_mean", "available_feature_count", "missing_feature_count", "missing_feature_rate"]
    ].copy()
    write_csv(OUT / "external_clinical_v2_rf_predictions_WITH_LABEL_PRIVATE.csv", with_label)

    y_ext = with_label["label_binary"].astype(int).to_numpy()
    p_ext = with_label["probability_COP_mean"].astype(float).to_numpy()
    metrics = metric_dict(y_ext, p_ext)
    metrics_df = pd.DataFrame([{"evaluation_name": "CLINICAL_V2_EXTERNAL_TRANSPORTABILITY_EXPLORATORY_EVALUATION", **metrics}])
    write_csv(OUT / "external_clinical_v2_metrics.csv", metrics_df)
    ci_df = bootstrap_ci(y_ext, p_ext)
    write_csv(OUT / "external_clinical_v2_bootstrap_ci.csv", ci_df)

    cap_probs = with_label.loc[with_label["label_binary"].eq(0), "probability_COP_mean"]
    cop_probs = with_label.loc[with_label["label_binary"].eq(1), "probability_COP_mean"]
    distribution_df = pd.DataFrame(
        [
            {"group": "overall", "n": 33, "median_probability_COP": float(np.median(p_ext)), "iqr_probability_COP": float(np.percentile(p_ext, 75) - np.percentile(p_ext, 25))},
            {"group": "CAP", "n": int(len(cap_probs)), "median_probability_COP": float(np.median(cap_probs)), "iqr_probability_COP": float(np.percentile(cap_probs, 75) - np.percentile(cap_probs, 25))},
            {"group": "COP", "n": int(len(cop_probs)), "median_probability_COP": float(np.median(cop_probs)), "iqr_probability_COP": float(np.percentile(cop_probs, 75) - np.percentile(cop_probs, 25))},
        ]
    )
    write_csv(OUT / "external_probability_distribution.csv", distribution_df)

    label_audit = {
        "status": "PASS",
        "label_read_event": "one planned label-read evaluation event",
        "label_read_count": LABEL_READ_COUNT,
        "n": 33,
        "CAP": int(counts.get(0, 0)),
        "COP": int(counts.get(1, 0)),
        "label_mapping": {"CAP": 0, "COP": 1},
        "prediction_lock_hash_before_label_read": prediction_lock_hash,
        "prediction_file_hash_before_label_read": pred_hash,
        "model_modified_after_label_read": 0,
    }
    write_json(OUT / "label_read_audit.json", label_audit)
    write_json(OUT / "model_modification_audit.json", model_modification_counts())
    write_csv(OUT / "external_access_log.csv", pd.DataFrame(access_log_rows))

    internal_dev_metrics = lock["current_RF_development_selected_OOF_metrics"]
    strict_metrics = lock["strict_family_nested_procedure_metrics"]
    context_rows = []
    for metric in ["AUROC", "AUPRC", "Brier"]:
        context_rows.append(
            {
                "metric": metric,
                "external_exploratory": metrics[metric],
                "development_selected_RF_OOF": internal_dev_metrics[metric],
                "external_minus_development_selected_RF_OOF": metrics[metric] - internal_dev_metrics[metric],
                "strict_family_nested_procedure_OOF": strict_metrics[metric],
                "external_minus_strict_family_nested_procedure_OOF": metrics[metric] - strict_metrics[metric],
            }
        )
    context_df = pd.DataFrame(context_rows)
    write_csv(OUT / "external_internal_context_metrics.csv", context_df)

    auroc_ci = ci_df[ci_df["metric"].eq("AUROC")].iloc[0]
    auprc_ci = ci_df[ci_df["metric"].eq("AUPRC")].iloc[0]
    brier_ci = ci_df[ci_df["metric"].eq("Brier")].iloc[0]
    write_md(
        OUT / "05_CLINICAL_V2_EXTERNAL_RESULTS.md",
        f"""# Clinical V2 External Results

Status: CLINICAL_V2_EXTERNAL_EXPLORATORY_EVALUATED

This is an external-cohort-informed exploratory evaluation of the locked clinical RF model, not independent external validation.

## Metrics

- AUROC: {metrics['AUROC']:.6f} (95% bootstrap CI {auroc_ci['ci_low_95']:.6f} to {auroc_ci['ci_high_95']:.6f})
- AUPRC: {metrics['AUPRC']:.6f} (95% bootstrap CI {auprc_ci['ci_low_95']:.6f} to {auprc_ci['ci_high_95']:.6f})
- Brier: {metrics['Brier']:.6f} (95% bootstrap CI {brier_ci['ci_low_95']:.6f} to {brier_ci['ci_high_95']:.6f})
- Calibration intercept: {metrics['calibration_intercept']:.6f}
- Calibration slope: {metrics['calibration_slope']:.6f}
- Sensitivity@0.5: {metrics['sensitivity_at_0_5']:.6f}
- Specificity@0.5: {metrics['specificity_at_0_5']:.6f}
- F1@0.5: {metrics['f1_at_0_5']:.6f}
- PPV@0.5: {metrics['ppv_at_0_5']:.6f}
- NPV@0.5: {metrics['npv_at_0_5']:.6f}

No threshold tuning was performed.
""",
    )
    write_md(
        OUT / "06_CLINICAL_INTERNAL_EXTERNAL_CONTEXT.md",
        "# Clinical Internal External Context\n\n"
        "Internal evidence pathways remain separate. External point estimates are descriptive and were not used to modify RF.\n\n"
        + context_df.to_markdown(index=False),
    )

    completed = {
        "status": "CLINICAL_V2_EXTERNAL_EXPLORATORY_EVALUATED",
        "start_time": start,
        "end_time": now_iso(),
        "deployment_model_strategy": "fixed full-development RF ensemble",
        "external_patient_count": 33,
        "external_CAP": 19,
        "external_COP": 14,
        "prediction_lock_sha256": prediction_lock_hash,
        "blinded_prediction_sha256": pred_hash,
        "label_read_count": LABEL_READ_COUNT,
        "metrics": metrics,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": N_BOOT,
        "model_modification_counts": model_modification_counts(),
        "external_access": {
            "prediction_pre_label_label_access": 0,
            "planned_label_read_events": LABEL_READ_COUNT,
            "historical_external_predictions_read": 0,
            "historical_external_metrics_read": 0,
        },
        "output_directory": str(OUT),
        "source_code_hash": sha256_file(SCRIPT_PATH),
    }
    write_json(OUT / "CLINICAL_V2_EXTERNAL_RUN_COMPLETED.json", completed)
    append_checksums(GENERATED + [SCRIPT_PATH])

    print("deployment_model_strategy=fixed full-development RF ensemble")
    print("external_patient_count=33")
    print(f"blinded_prediction_lock_SHA256={prediction_lock_hash}")
    print(f"label_read_count={LABEL_READ_COUNT}")
    print(f"AUROC={metrics['AUROC']:.6f} 95%CI={auroc_ci['ci_low_95']:.6f},{auroc_ci['ci_high_95']:.6f}")
    print(f"AUPRC={metrics['AUPRC']:.6f} 95%CI={auprc_ci['ci_low_95']:.6f},{auprc_ci['ci_high_95']:.6f}")
    print(f"Brier={metrics['Brier']:.6f} 95%CI={brier_ci['ci_low_95']:.6f},{brier_ci['ci_high_95']:.6f}")
    print(f"calibration_intercept={metrics['calibration_intercept']:.6f} calibration_slope={metrics['calibration_slope']:.6f}")
    print(f"sensitivity_specificity_F1_at_0.5={metrics['sensitivity_at_0_5']:.6f},{metrics['specificity_at_0_5']:.6f},{metrics['f1_at_0_5']:.6f}")
    print("external_feature_availability_main_risk=" + ";".join(top_missing["feature"].astype(str).tolist()))
    print("model_modification_counts_all_zero=1")
    print(f"output_directory={OUT}")
    print("final_status=CLINICAL_V2_EXTERNAL_EXPLORATORY_EVALUATED")


if __name__ == "__main__":
    main()
