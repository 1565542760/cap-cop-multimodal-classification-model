from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
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
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
    XGBOOST_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    XGBClassifier = None
    XGBOOST_AVAILABLE = False
    XGBOOST_IMPORT_ERROR = repr(exc)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get("CAP_COP_LOCAL_EXPERIMENT_V2", PROJECT_ROOT / "models" / "historical_clinical_contracts" / "experiment_v2"))
V1 = Path(os.environ.get("CAP_COP_LOCAL_EXPERIMENT_V1", PROJECT_ROOT / "models" / "historical_clinical_contracts" / "experiment_v1"))
V1_COHORT = V1 / "cohort_and_schema"
V1_PROJECT = V1 / "cap_cop_experiment"
V1_NESTED = V1_PROJECT / "generalisation_redevelopment_nested_protocol_v1"

OUT = {
    "governance": ROOT / "governance",
    "data_contract": ROOT / "data_contract",
    "splits": ROOT / "splits",
    "src": ROOT / "src",
    "configs": ROOT / "configs",
    "runs": ROOT / "runs" / "clinical",
    "locked": ROOT / "locked_models",
    "reports": ROOT / "reports",
    "audit": ROOT / "audit",
    "external": ROOT / "external_evaluation",
    "public": ROOT / "public_release",
}

START_TIME = datetime.now(timezone.utc)
TRAINING_SEEDS = [20260627, 20260628, 20260629]
OUTER_FOLDS = [0, 1, 2, 3, 4]
BOOTSTRAP_SEED = 20260815
N_BOOT = 1000
STRESS_SEEDS = {"S1": 70101, "S2": 70102, "S3": 70103, "S4": 70104}

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

FEATURE_MAP = {
    "age": "age",
    "height_cm": "height_cm",
    "weight_kg": "weight_kg",
    "bmi": "body_mass_index",
    "fever": "fever",
    "cough": "cough",
    "sputum_category": "sputum_category",
    "chest_distress": "chest_tightness",
    "dyspnea": "dyspnoea",
    "hemoptysis": "haemoptysis",
    "weight_loss": "weight_loss",
    "lymphocyte_count": "lymphocyte_count",
    "esr": "erythrocyte_sedimentation_rate",
    "crp": "c_reactive_protein",
    "hscrp": "high_sensitivity_c_reactive_protein",
    "procalcitonin": "procalcitonin",
}

FORBIDDEN_PATTERNS = [
    "patient",
    "case_uid",
    "case_id",
    "disease",
    "label",
    "hospital",
    "site",
    "file",
    "path",
    "source",
    "split",
    "fold",
    "cap",
    "cop",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    for path in OUT.values():
        path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    tmp = df.copy()
    for col in tmp.columns:
        tmp[col] = tmp[col].map(lambda v: "" if pd.isna(v) else f"{v:.6g}" if isinstance(v, float) else str(v))
    cols = list(tmp.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in tmp.iterrows():
        vals = [str(row[c]).replace("|", "\\|") for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def stop_run(reason: str, details: dict[str, Any] | None = None) -> None:
    ensure_dirs()
    report = {
        "status": "STOPPED",
        "reason": reason,
        "details": details or {},
        "timestamp": now_iso(),
    }
    write_json(OUT["audit"] / "RUN_COMPLETED.json", report)
    write_md(
        ROOT / "STOP_REPORT.md",
        f"""
# STOP_REPORT

Status: STOPPED

Reason: {reason}

Details:

```json
{json.dumps(details or {}, indent=2, ensure_ascii=False)}
```
""",
    )
    print("STATUS: STOPPED")
    print(f"REASON: {reason}")
    print(f"STOP_REPORT: {ROOT / 'STOP_REPORT.md'}")
    sys.exit(2)


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = [
        V1_COHORT / "case_master_v1.csv",
        V1_COHORT / "clinical_feature_matrix_draft.csv",
        V1_COHORT / "feature_schema_v1.json",
        V1_NESTED / "aggregate" / "outer_inner_fold_patient_manifest.csv",
        V1_NESTED / "data" / "internal_development" / "clinical_features_basic_corrected_for_development.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        stop_run("required_v1_data_contract_files_missing", {"missing": missing})
    case_master = pd.read_csv(required[0])
    original_clinical = pd.read_csv(required[1])
    nested = pd.read_csv(required[3])
    clinical = pd.read_csv(required[4])
    with required[2].open("r", encoding="utf-8-sig") as f:
        schema = json.load(f)
    return case_master, original_clinical, nested, clinical, schema


def copy_locked_split_sources() -> dict[str, str]:
    copied: dict[str, str] = {}
    for seed in TRAINING_SEEDS:
        src = V1_COHORT / f"split_patient_cv5_seed{seed}.csv"
        if not src.exists():
            stop_run("locked_outer_split_missing", {"missing": str(src)})
        dst = OUT["splits"] / src.name
        shutil.copy2(src, dst)
        copied[str(dst)] = sha256_file(dst)
    nested_src = V1_NESTED / "aggregate" / "outer_inner_fold_patient_manifest.csv"
    nested_dst = OUT["splits"] / "locked_nested_split_patient_manifest.csv"
    shutil.copy2(nested_src, nested_dst)
    copied[str(nested_dst)] = sha256_file(nested_dst)
    rows = [{"file": k, "sha256": v} for k, v in copied.items()]
    pd.DataFrame(rows).to_csv(OUT["splits"] / "locked_split_hashes.csv", index=False, encoding="utf-8-sig")
    return copied


def make_governance_docs(split_hashes: dict[str, str]) -> None:
    sources = [
        {
            "source_file": str(V1_COHORT / "cohort_v1_summary.md"),
            "contract_items": "Internal cohort size 375, CAP 250, COP 125, excluded case count, residual duplicate risk summary.",
        },
        {
            "source_file": str(V1_COHORT / "cohort_schema_split_report.md"),
            "contract_items": "CAP=0, COP=1 label mapping, 5-fold patient-level split distribution, schema summary, fold-level CAP/COP balance.",
        },
        {
            "source_file": str(V1_COHORT / "case_master_v1.csv"),
            "contract_items": "Private patient-level manifest fields, raw image/label relative paths, clinical file/sheet/row, matching status, patient_group_key.",
        },
        {
            "source_file": str(V1_COHORT / "clinical_feature_matrix_draft.csv"),
            "contract_items": "Original cleaned clinical feature matrix and raw/clean value pairs for the 375 internal patients.",
        },
        {
            "source_file": str(V1_COHORT / "feature_schema_v1.json"),
            "contract_items": "Clinical variable definitions, original clinical columns, allowed codings, feature groups, leakage notes.",
        },
        {
            "source_file": str(V1_COHORT / "freeze_cohort_schema_v1.py"),
            "contract_items": "Original data root, clinical data path fallback, patient matching and normalization implementation, feature cleaning rules.",
        },
        {
            "source_file": str(V1_NESTED / "protocols" / "locked_nested_validation_protocol.md"),
            "contract_items": "Locked nested validation protocol, training seeds, fold-scope preprocessing and calibration constraints, external exclusion rule.",
        },
        {
            "source_file": str(V1_NESTED / "aggregate" / "outer_inner_fold_patient_manifest.csv"),
            "contract_items": "Patient-level outer train/test roles and 4-fold inner validation roles for each training seed and outer fold.",
        },
        {
            "source_file": str(V1_NESTED / "data" / "internal_development" / "clinical_features_basic_corrected_for_development.csv"),
            "contract_items": "Deidentified 16-variable clinical_basic development table with documented BMI correction; used as v2 clinical feature source.",
        },
    ]
    for row in sources:
        p = Path(row["source_file"])
        row["exists"] = p.exists()
        row["sha256"] = sha256_file(p) if p.exists() and p.is_file() else ""
    pd.DataFrame(sources).to_csv(
        OUT["governance"] / "v1_data_contract_sources.csv", index=False, encoding="utf-8-sig"
    )
    split_lines = "\n".join([f"- `{k}`: `{v}`" for k, v in split_hashes.items()])
    write_md(
        OUT["governance"] / "V1_TO_V2_DATA_CONTRACT.md",
        f"""
# V1 To V2 Data Contract

This v2 run used v1 files only as read-only data-contract sources. Historical v1 AUROC, AUPRC, Brier, external leaderboard, and external labels were not used for v2 algorithm choice, hyperparameter design, calibration, thresholding, stopping, or model selection.

## Source Items

| Source | Contract extracted |
|---|---|
{os.linesep.join([f"| `{r['source_file']}` | {r['contract_items']} |" for r in sources])}

## Locked Split Hashes Copied Into V2

{split_lines}

## Clinical Development Choice

The v2 clinical stage uses the 16-variable `clinical_basic` feature contract only. Structured radiological variables and tumor-marker sensitivity variables are not predictors in this stage. This avoids starting imaging, mask-guided, or multimodal work.

## External Isolation

External labels, predictions, metrics, and historical external ranking information were not read. `audit/external_access_log.csv` records zero label, prediction, and metric access.
""",
    )


def public_release_readme() -> None:
    write_md(
        OUT["public"] / "README.md",
        """
# Public Release Rules

This folder is initialized for future public code only. No candidate model code is copied here during the clinical v2 stage.

Future public folders may use only algorithm/model names such as:

- `RF`
- `ElasticNetLogit`
- `ResNet3D`
- `ResNet3D_AuxMask`
- `mul_RF_ResNet3D`
- `mul_RF_ResNet3D_AuxMask`

Forbidden folder names include:

- `phase1`
- `phase2`
- `v2_best`
- `experiment17`
- `best_external`
- `final_after_external`

Public release content must never include patient IDs, labels, raw clinical tables, CT, masks, private paths, patient-level predictions, private checkpoints, or external results.

Only a final scientific model that is locked and passes reproducibility testing may contribute a source snapshot to this folder.
""",
    )


def external_access_log() -> None:
    pd.DataFrame(
        [
            {
                "timestamp": now_iso(),
                "process": "clinical_v2_initialization",
                "file": "",
                "reason": "external cohort deliberately not accessed during clinical development",
                "label_access": 0,
                "prediction_access": 0,
                "metric_access": 0,
                "model_modified_after_access": 0,
            }
        ]
    ).to_csv(OUT["audit"] / "external_access_log.csv", index=False, encoding="utf-8-sig")


def validate_data(
    case_master: pd.DataFrame, original_clinical: pd.DataFrame, nested: pd.DataFrame, clinical: pd.DataFrame
) -> dict[str, Any]:
    failures: list[str] = []
    counts = clinical["label_binary"].value_counts().to_dict()
    disease_counts = clinical["disease_label"].value_counts().to_dict()
    if len(clinical) != 375:
        failures.append("internal_patient_count_not_375")
    if counts.get(0, 0) != 250 or counts.get(1, 0) != 125:
        failures.append("label_binary_counts_not_250_125")
    if disease_counts.get("CAP", 0) != 250 or disease_counts.get("COP", 0) != 125:
        failures.append("disease_label_counts_not_250_125")
    if set(clinical["label_binary"].unique()) != {0, 1}:
        failures.append("label_mapping_not_CAP0_COP1")
    mapping_bad = clinical.groupby("disease_label")["label_binary"].nunique().to_dict()
    if clinical.loc[clinical["disease_label"].eq("CAP"), "label_binary"].ne(0).any():
        failures.append("CAP_not_mapped_to_0")
    if clinical.loc[clinical["disease_label"].eq("COP"), "label_binary"].ne(1).any():
        failures.append("COP_not_mapped_to_1")
    for col in ["case_id_public", "patient_group_id_public"]:
        if clinical[col].duplicated().any():
            failures.append(f"duplicate_{col}")
    if len(case_master) != 375 or len(original_clinical) != 375:
        failures.append("case_master_or_original_clinical_not_375")
    if case_master["patient_group_key"].duplicated().any():
        failures.append("duplicate_private_patient_group_key")
    if original_clinical["patient_group_key"].duplicated().any():
        failures.append("duplicate_original_clinical_patient_group_key")
    if not set(FEATURES).issubset(clinical.columns):
        failures.append("missing_required_clinical_features")

    for seed in TRAINING_SEEDS:
        df_seed = nested[nested["training_seed"].eq(seed)]
        if len(df_seed) != 1875:
            failures.append(f"nested_seed_{seed}_row_count_not_1875")
        test_counts = (
            df_seed[df_seed["outer_role"].eq("outer_test")]
            .groupby("case_id_public")["outer_fold_id"]
            .nunique()
        )
        if len(test_counts) != 375 or not (test_counts == 1).all():
            failures.append(f"seed_{seed}_patients_not_once_outer_test")
        for fold in OUTER_FOLDS:
            df_fold = df_seed[df_seed["outer_fold_id"].eq(fold)]
            train_ids = set(df_fold.loc[df_fold["outer_role"].eq("outer_train"), "case_id_public"])
            test_ids = set(df_fold.loc[df_fold["outer_role"].eq("outer_test"), "case_id_public"])
            if len(train_ids) != 300 or len(test_ids) != 75:
                failures.append(f"seed_{seed}_fold_{fold}_train_test_size_bad")
            if train_ids & test_ids:
                failures.append(f"seed_{seed}_fold_{fold}_outer_overlap")
            inner = df_fold[df_fold["outer_role"].eq("outer_train")]
            if not set(inner["inner_fold_id"].dropna().astype(int).unique()) == {0, 1, 2, 3}:
                failures.append(f"seed_{seed}_fold_{fold}_inner_folds_not_0_3")
            for inner_fold in range(4):
                val_ids = set(inner.loc[inner["inner_fold_id"].astype(int).eq(inner_fold), "case_id_public"])
                if not val_ids or not val_ids.issubset(train_ids):
                    failures.append(f"seed_{seed}_fold_{fold}_inner_not_subset_outer_train")

    split_patient_ids = set(nested["case_id_public"])
    clinical_patient_ids = set(clinical["case_id_public"])
    if split_patient_ids != clinical_patient_ids:
        failures.append("nested_split_patient_ids_do_not_match_clinical")

    predictor_set = set(FEATURES)
    forbidden_used = []
    for f in FEATURES:
        fl = f.lower()
        reasons = [p for p in FORBIDDEN_PATTERNS if p in fl]
        if reasons:
            forbidden_used.append({"feature": f, "reason": ",".join(reasons)})
    if forbidden_used:
        failures.append("forbidden_feature_entered_matrix")

    if failures:
        stop_run("data_integrity_or_split_validation_failed", {"failures": failures})

    return {
        "patient_counts": {
            "total": int(len(clinical)),
            "CAP": int(disease_counts.get("CAP", 0)),
            "COP": int(disease_counts.get("COP", 0)),
            "label_0": int(counts.get(0, 0)),
            "label_1": int(counts.get(1, 0)),
        },
        "label_mapping": {"CAP": 0, "COP": 1, "group_label_nunique": mapping_bad},
        "split_validation": "passed",
        "failures": failures,
    }


def write_data_audits(
    case_master: pd.DataFrame,
    clinical: pd.DataFrame,
    nested: pd.DataFrame,
    schema: dict[str, Any],
    validation: dict[str, Any],
) -> None:
    manifest = clinical[["case_id_public", "patient_group_id_public", "disease_label", "label_binary"]].copy()
    cm = case_master.reset_index(drop=True)
    manifest["v1_case_uid_private"] = cm["case_uid"].values
    manifest["v1_patient_group_key_private"] = cm["patient_group_key"].values
    manifest["v1_study_date"] = cm["study_date"].values
    manifest["v1_clinical_file"] = cm["clinical_file"].values
    manifest["v1_clinical_sheet"] = cm["clinical_sheet"].values
    manifest["v1_clinical_row_index"] = cm["clinical_row_index"].values
    manifest["v1_match_status"] = cm["match_status"].values
    manifest.to_csv(
        OUT["data_contract"] / "internal_patient_manifest.csv", index=False, encoding="utf-8-sig"
    )

    rows = []
    variables = {v["canonical_name"]: v for v in schema["variables"]}
    for canonical, v2_name in FEATURE_MAP.items():
        item = variables.get(canonical, {})
        rows.append(
            {
                "v2_feature": v2_name,
                "v1_canonical_name": canonical,
                "variable_type": item.get("variable_type", "numeric"),
                "feature_group": item.get("feature_group", ""),
                "preprocessing_rule": "numeric_ordinal; fold-internal median imputation; optional missingness indicator",
                "included_in_v2_clinical_basic": True,
                "missing_rate_CAP_v1": item.get("missing_rate_CAP", ""),
                "missing_rate_COP_v1": item.get("missing_rate_COP", ""),
                "leakage_risk_v1": item.get("leakage_risk", ""),
                "notes": item.get("notes", ""),
            }
        )
    pd.DataFrame(rows).to_csv(
        OUT["data_contract"] / "clinical_feature_dictionary.csv", index=False, encoding="utf-8-sig"
    )

    audit_rows = []
    for seed in TRAINING_SEEDS:
        df_seed = nested[nested["training_seed"].eq(seed)]
        for fold in OUTER_FOLDS:
            df_fold = df_seed[df_seed["outer_fold_id"].eq(fold)]
            for role in ["outer_train", "outer_test"]:
                sub = df_fold[df_fold["outer_role"].eq(role)]
                audit_rows.append(
                    {
                        "training_seed": seed,
                        "outer_seed_from_locked_manifest": int(sub["outer_seed"].iloc[0]),
                        "outer_fold_id": fold,
                        "role": role,
                        "n": len(sub),
                        "CAP": int((sub["label_binary"] == 0).sum()),
                        "COP": int((sub["label_binary"] == 1).sum()),
                        "unique_patients": sub["case_id_public"].nunique(),
                    }
                )
            inner = df_fold[df_fold["outer_role"].eq("outer_train")].copy()
            inner["inner_fold_id"] = inner["inner_fold_id"].astype(int)
            for inner_fold in range(4):
                sub = inner[inner["inner_fold_id"].eq(inner_fold)]
                audit_rows.append(
                    {
                        "training_seed": seed,
                        "outer_seed_from_locked_manifest": int(sub["outer_seed"].iloc[0]),
                        "outer_fold_id": fold,
                        "role": f"inner_validation_{inner_fold}",
                        "n": len(sub),
                        "CAP": int((sub["label_binary"] == 0).sum()),
                        "COP": int((sub["label_binary"] == 1).sum()),
                        "unique_patients": sub["case_id_public"].nunique(),
                    }
                )
    pd.DataFrame(audit_rows).to_csv(
        OUT["splits"] / "locked_split_audit.csv", index=False, encoding="utf-8-sig"
    )

    all_cols = list(clinical.columns)
    forbidden_rows = []
    for col in all_cols:
        lower = col.lower()
        reasons = [p for p in FORBIDDEN_PATTERNS if p in lower]
        forbidden_rows.append(
            {
                "column": col,
                "used_as_predictor": col in FEATURES,
                "forbidden_pattern_hits": ";".join(reasons),
                "allowed_in_feature_matrix": col in FEATURES and not reasons,
            }
        )
    pd.DataFrame(forbidden_rows).to_csv(
        OUT["audit"] / "forbidden_feature_audit.csv", index=False, encoding="utf-8-sig"
    )

    write_json(
        OUT["audit"] / "patient_leakage_audit.json",
        {
            "status": "passed",
            "patient_level_unit": "case_id_public and patient_group_id_public",
            "clinical_rows": int(len(clinical)),
            "unique_case_id_public": int(clinical["case_id_public"].nunique()),
            "unique_patient_group_id_public": int(clinical["patient_group_id_public"].nunique()),
            "outer_train_test_overlap": 0,
            "inner_outside_outer_train": 0,
            "forbidden_features_used": [],
        },
    )

    write_md(
        OUT["audit"] / "data_integrity_report.md",
        f"""
# Data Integrity Report

Status: PASSED

- Internal patients: {validation['patient_counts']['total']}
- CAP: {validation['patient_counts']['CAP']}
- COP: {validation['patient_counts']['COP']}
- Label mapping: CAP=0, COP=1
- Patient IDs duplicated: no
- Clinical rows matched one-to-one: yes
- Locked nested split coverage: all 375 patients for each training seed
- Outer train/test overlap: none
- Inner folds subset of outer-training only: yes
- Forbidden predictors in matrix: none

The locked nested manifest has a fixed `outer_seed=20260627` and three `training_seed` values: {TRAINING_SEEDS}. V2 treats each training seed as one seed-specific OOF run and averages the three probabilities per patient for consensus OOF.
""",
    )


def candidate_registry() -> dict[str, Any]:
    registry = {
        "created_utc": now_iso(),
        "feature_set": "clinical_basic_v2",
        "features": FEATURES,
        "tuning_metric": "inner_cv_auroc",
        "probability_output_contract": "predict_proba class COP=1, Platt-calibrated with outer-training cross-fitted raw probabilities; clipped only for numerical logit stability",
        "label_contract": {"CAP": 0, "COP": 1},
        "random_seeds": TRAINING_SEEDS,
        "missingness_strategies": [
            "training-fold median/simple imputation",
            "training-fold median/simple imputation plus missingness indicators",
        ],
        "calibration_strategy": "For every selected outer-fold model, fit Platt logistic calibrator on outer-training cross-fitted raw probabilities from the locked 4 inner folds; outer-test never enters calibration.",
        "xgboost_available": XGBOOST_AVAILABLE,
        "xgboost_import_error": XGBOOST_IMPORT_ERROR,
        "candidate_families": {
            "ElasticNetLogit": {
                "algorithm_family": "ElasticNetLogit",
                "preprocessing": ["median_impute_scale", "median_impute_indicator_scale"],
                "hyperparameter_search_space": {
                    "C": [0.05, 0.2, 1.0],
                    "l1_ratio": [0.1, 0.5, 0.9],
                    "class_weight": ["balanced"],
                },
                "missingness_strategy": "median and median_plus_indicator compared inside training fold",
            },
            "RF": {
                "algorithm_family": "RF",
                "preprocessing": ["median_impute", "median_impute_indicator"],
                "hyperparameter_search_space": {
                    "n_estimators": [300],
                    "max_depth": [3, None],
                    "min_samples_leaf": [3, 8],
                    "max_features": ["sqrt", 0.5],
                    "class_weight": ["balanced"],
                },
                "missingness_strategy": "median and median_plus_indicator compared inside training fold",
            },
            "GradientBoosting": {
                "algorithm_family": "GradientBoosting",
                "implementation": "XGBoost" if XGBOOST_AVAILABLE else "HistGradientBoosting fallback not executed unless XGBoost unavailable is explicitly recorded",
                "preprocessing": ["median_impute", "median_impute_indicator", "xgboost_native_missing"],
                "hyperparameter_search_space": {
                    "n_estimators": [75, 150],
                    "max_depth": [2, 3],
                    "learning_rate": [0.03, 0.1],
                    "subsample": [0.9],
                    "colsample_bytree": [0.9],
                    "reg_lambda": [1.0, 5.0],
                },
                "missingness_strategy": "median, median_plus_indicator, and XGBoost native NaN handling compared inside training fold",
            },
        },
    }
    if not XGBOOST_AVAILABLE:
        stop_run("xgboost_unavailable_and_no_silent_algorithm_substitution_allowed", registry)
    write_json(OUT["configs"] / "clinical_candidate_registry.json", registry)
    return registry


def build_estimator(family: str, params: dict[str, Any], seed: int) -> Any:
    if family == "ElasticNetLogit":
        return LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=params["C"],
            l1_ratio=params["l1_ratio"],
            class_weight="balanced",
            max_iter=10000,
            random_state=seed,
            n_jobs=1,
        )
    if family == "RF":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    if family == "GradientBoosting":
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_lambda=params["reg_lambda"],
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
    raise ValueError(family)


def make_pipeline(family: str, missingness: str, params: dict[str, Any], seed: int) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    if missingness == "median":
        steps.append(("imputer", SimpleImputer(strategy="median", add_indicator=False)))
    elif missingness == "median_indicator":
        steps.append(("imputer", SimpleImputer(strategy="median", add_indicator=True)))
    elif missingness == "native":
        if family != "GradientBoosting":
            raise ValueError("native missingness only allowed for GradientBoosting/XGBoost")
    else:
        raise ValueError(missingness)
    if family == "ElasticNetLogit":
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", build_estimator(family, params, seed)))
    return Pipeline(steps)


def param_candidates(family: str) -> list[dict[str, Any]]:
    if family == "ElasticNetLogit":
        grid = {"C": [0.05, 0.2, 1.0], "l1_ratio": [0.1, 0.5, 0.9]}
        missingness = ["median", "median_indicator"]
    elif family == "RF":
        grid = {
            "n_estimators": [300],
            "max_depth": [3, None],
            "min_samples_leaf": [3, 8],
            "max_features": ["sqrt", 0.5],
        }
        missingness = ["median", "median_indicator"]
    elif family == "GradientBoosting":
        grid = {
            "n_estimators": [75, 150],
            "max_depth": [2, 3],
            "learning_rate": [0.03, 0.1],
            "subsample": [0.9],
            "colsample_bytree": [0.9],
            "reg_lambda": [1.0, 5.0],
        }
        missingness = ["median", "median_indicator", "native"]
    else:
        raise ValueError(family)
    out = []
    for m in missingness:
        for params in ParameterGrid(grid):
            out.append({"missingness": m, "params": dict(params)})
    return out


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def metrics_at_05(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "sensitivity_at_0_5": sens,
        "specificity_at_0_5": spec,
        "f1_at_0_5": float(f1_score(y, pred, zero_division=0)),
        "ppv_at_0_5": ppv,
        "npv_at_0_5": npv,
    }


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000)
        lr.fit(x, y)
        return float(lr.intercept_[0]), float(lr.coef_[0][0])
    except Exception:
        return float("nan"), float("nan")


def compute_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    intercept, slope = calibration_intercept_slope(y, p)
    out = {
        "AUROC": safe_auc(y, p),
        "AUPRC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }
    out.update(metrics_at_05(y, p))
    return out


def get_split(nested: pd.DataFrame, seed: int, fold: int) -> tuple[list[str], list[str], dict[str, int]]:
    df = nested[(nested["training_seed"] == seed) & (nested["outer_fold_id"] == fold)]
    train_ids = df.loc[df["outer_role"].eq("outer_train"), "case_id_public"].tolist()
    test_ids = df.loc[df["outer_role"].eq("outer_test"), "case_id_public"].tolist()
    inner_map = (
        df.loc[df["outer_role"].eq("outer_train"), ["case_id_public", "inner_fold_id"]]
        .assign(inner_fold_id=lambda x: x["inner_fold_id"].astype(int))
        .set_index("case_id_public")["inner_fold_id"]
        .to_dict()
    )
    return train_ids, test_ids, inner_map


def tune_family(
    family: str,
    X: pd.DataFrame,
    y: pd.Series,
    train_ids: list[str],
    inner_map: dict[str, int],
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    y_train_all = y.loc[train_ids].to_numpy()
    candidates = param_candidates(family)
    for cand_idx, cand in enumerate(candidates):
        oof = pd.Series(index=train_ids, dtype=float)
        fold_scores = []
        for inner_fold in range(4):
            val_ids = [pid for pid in train_ids if inner_map[pid] == inner_fold]
            tr_ids = [pid for pid in train_ids if inner_map[pid] != inner_fold]
            model = make_pipeline(family, cand["missingness"], cand["params"], seed + cand_idx + inner_fold)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X.loc[tr_ids], y.loc[tr_ids])
            prob = model.predict_proba(X.loc[val_ids])[:, 1]
            oof.loc[val_ids] = prob
            fold_scores.append(safe_auc(y.loc[val_ids].to_numpy(), prob))
        if oof.isna().any():
            stop_run("inner_cv_oof_incomplete", {"family": family, "seed": seed, "candidate": cand})
        score = safe_auc(y_train_all, oof.loc[train_ids].to_numpy())
        rows.append(
            {
                "family": family,
                "candidate_index": cand_idx,
                "missingness": cand["missingness"],
                "params_json": json.dumps(cand["params"], sort_keys=True),
                "inner_cv_auroc": score,
                "inner_fold_aurocs_json": json.dumps(fold_scores),
            }
        )
    df = pd.DataFrame(rows).sort_values(["inner_cv_auroc", "candidate_index"], ascending=[False, True])
    best_row = df.iloc[0].to_dict()
    best = {
        "family": family,
        "missingness": best_row["missingness"],
        "params": json.loads(best_row["params_json"]),
        "inner_cv_auroc": float(best_row["inner_cv_auroc"]),
        "candidate_index": int(best_row["candidate_index"]),
    }
    return best, rows


def fit_platt_calibrator(
    family: str,
    best: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    train_ids: list[str],
    inner_map: dict[str, int],
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    raw = pd.Series(index=train_ids, dtype=float)
    for inner_fold in range(4):
        val_ids = [pid for pid in train_ids if inner_map[pid] == inner_fold]
        tr_ids = [pid for pid in train_ids if inner_map[pid] != inner_fold]
        model = make_pipeline(family, best["missingness"], best["params"], seed + 900 + inner_fold)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X.loc[tr_ids], y.loc[tr_ids])
        raw.loc[val_ids] = model.predict_proba(X.loc[val_ids])[:, 1]
    if raw.isna().any():
        stop_run("calibration_crossfit_incomplete", {"family": family, "seed": seed, "best": best})
    x = np.log(np.clip(raw.to_numpy(), 1e-6, 1 - 1e-6) / np.clip(1 - raw.to_numpy(), 1e-6, 1))
    cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal.fit(x.reshape(-1, 1), y.loc[train_ids].to_numpy())
    info = {
        "calibration_scope": "outer_training_cross_fitted_inner_predictions_only",
        "calibrator": "PlattLogisticRegressionOnLogitRawProbability",
        "n_calibration_patients": len(train_ids),
        "coef": float(cal.coef_[0][0]),
        "intercept": float(cal.intercept_[0]),
        "raw_oof_brier": float(brier_score_loss(y.loc[train_ids], raw.loc[train_ids])),
    }
    return cal, info


def apply_calibrator(cal: Any, raw_prob: np.ndarray) -> np.ndarray:
    raw_prob = np.clip(np.asarray(raw_prob, dtype=float), 1e-6, 1 - 1e-6)
    x = np.log(raw_prob / (1 - raw_prob)).reshape(-1, 1)
    return cal.predict_proba(x)[:, 1]


def stress_copy(X_test: pd.DataFrame, scenario: str, seed: int, fold: int, family: str) -> pd.DataFrame:
    Xs = X_test.copy()
    rng_seed = STRESS_SEEDS.get(scenario, 0) + seed * 13 + fold * 101 + abs(hash(family)) % 1000
    rng = np.random.default_rng(rng_seed)
    if scenario == "S0":
        return Xs
    if scenario in {"S1", "S2", "S3"}:
        rate = {"S1": 0.10, "S2": 0.20, "S3": 0.30}[scenario]
        mask = rng.random(Xs.shape) < rate
        return Xs.mask(mask)
    if scenario == "S4":
        lab_features = [
            "lymphocyte_count",
            "erythrocyte_sedimentation_rate",
            "c_reactive_protein",
            "high_sensitivity_c_reactive_protein",
            "procalcitonin",
        ]
        patient_mask = rng.random(len(Xs)) < 0.20
        Xs.loc[patient_mask, lab_features] = np.nan
        return Xs
    raise ValueError(scenario)


def run_nested(nested: pd.DataFrame, clinical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clinical = clinical.set_index("case_id_public", drop=False)
    X = clinical[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = clinical["label_binary"].astype(int)
    families = ["ElasticNetLogit", "RF", "GradientBoosting"]
    pred_rows: list[dict[str, Any]] = []
    stress_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    preprocessing_audit: list[dict[str, Any]] = []
    probability_failures: list[dict[str, Any]] = []

    for seed in TRAINING_SEEDS:
        for fold in OUTER_FOLDS:
            train_ids, test_ids, inner_map = get_split(nested, seed, fold)
            for family in families:
                best, tune_rows = tune_family(family, X, y, train_ids, inner_map, seed)
                for r in tune_rows:
                    r.update({"training_seed": seed, "outer_fold_id": fold})
                tuning_rows.extend(tune_rows)
                calibrator, cal_info = fit_platt_calibrator(family, best, X, y, train_ids, inner_map, seed)
                model = make_pipeline(family, best["missingness"], best["params"], seed + 777)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X.loc[train_ids], y.loc[train_ids])

                fold_pred = []
                for scenario in ["S0", "S1", "S2", "S3", "S4"]:
                    X_eval = stress_copy(X.loc[test_ids], scenario, seed, fold, family)
                    raw = model.predict_proba(X_eval)[:, 1]
                    prob = apply_calibrator(calibrator, raw)
                    if np.any((prob < 0) | (prob > 1)) or np.any(~np.isfinite(prob)):
                        probability_failures.append(
                            {"family": family, "seed": seed, "fold": fold, "scenario": scenario}
                        )
                    for pid, raw_p, p in zip(test_ids, raw, prob):
                        row = {
                            "family": family,
                            "training_seed": seed,
                            "outer_fold_id": fold,
                            "case_id_public": pid,
                            "patient_group_id_public": clinical.loc[pid, "patient_group_id_public"],
                            "label_binary": int(y.loc[pid]),
                            "disease_label": clinical.loc[pid, "disease_label"],
                            "scenario": scenario,
                            "raw_probability_COP": float(raw_p),
                            "probability_COP": float(p),
                            "selected_missingness": best["missingness"],
                            "selected_params_json": json.dumps(best["params"], sort_keys=True),
                            "inner_cv_auroc": best["inner_cv_auroc"],
                            "calibration_coef": cal_info["coef"],
                            "calibration_intercept_param": cal_info["intercept"],
                        }
                        if scenario == "S0":
                            pred_rows.append(row)
                            fold_pred.append(row)
                        stress_rows.append(row)

                y_fold = np.array([r["label_binary"] for r in fold_pred])
                p_fold = np.array([r["probability_COP"] for r in fold_pred])
                metrics = compute_metrics(y_fold, p_fold)
                fold_metric_rows.append(
                    {
                        "family": family,
                        "training_seed": seed,
                        "outer_fold_id": fold,
                        "n_test": len(test_ids),
                        "selected_missingness": best["missingness"],
                        "selected_params_json": json.dumps(best["params"], sort_keys=True),
                        "inner_cv_auroc": best["inner_cv_auroc"],
                        **metrics,
                    }
                )
                preprocessing_audit.append(
                    {
                        "family": family,
                        "training_seed": seed,
                        "outer_fold_id": fold,
                        "preprocessing_fit_scope": "outer_training_only",
                        "hyperparameter_selection_scope": "outer_training_locked_inner_4fold_only",
                        "calibration_scope": cal_info["calibration_scope"],
                        "outer_test_used_for_fit_tune_calibration_threshold": False,
                        "selected_missingness": best["missingness"],
                        "selected_params": best["params"],
                    }
                )

    if probability_failures:
        stop_run("probability_contract_failed", {"failures": probability_failures[:10]})
    pred = pd.DataFrame(pred_rows)
    stress = pd.DataFrame(stress_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    pd.DataFrame(tuning_rows).to_csv(
        OUT["runs"] / "clinical_inner_tuning_registry.csv", index=False, encoding="utf-8-sig"
    )
    write_json(
        OUT["audit"] / "preprocessing_leakage_audit.json",
        {
            "status": "passed",
            "records": preprocessing_audit,
            "outer_test_used_for_preprocessing_tuning_calibration_threshold": False,
        },
    )
    write_json(
        OUT["audit"] / "probability_contract_audit.json",
        {
            "status": "passed",
            "probability_column": "probability_COP",
            "min_probability": float(pred["probability_COP"].min()),
            "max_probability": float(pred["probability_COP"].max()),
            "nonfinite_count": int((~np.isfinite(pred["probability_COP"])).sum()),
            "decision_function_used_as_probability": False,
        },
    )
    return pred, stress, fold_metrics


def summarize_predictions(pred: pd.DataFrame, stress: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred.to_csv(
        OUT["runs"] / "clinical_patient_oof_predictions_PRIVATE.csv", index=False, encoding="utf-8-sig"
    )
    seed_checks = []
    for (family, seed), sub in pred.groupby(["family", "training_seed"]):
        counts = sub.groupby("case_id_public").size()
        seed_checks.append(
            {
                "family": family,
                "training_seed": seed,
                "patients": counts.size,
                "min_predictions_per_patient": int(counts.min()),
                "max_predictions_per_patient": int(counts.max()),
            }
        )
        if counts.size != 375 or counts.min() != 1 or counts.max() != 1:
            stop_run("OOF_not_complete_or_duplicate", {"family": family, "seed": int(seed)})
    pd.DataFrame(seed_checks).to_csv(
        OUT["audit"] / "oof_completeness_audit.csv", index=False, encoding="utf-8-sig"
    )

    metric_rows = []
    consensus_rows = []
    for family, fam in pred.groupby("family"):
        for seed, sub in fam.groupby("training_seed"):
            m = compute_metrics(sub["label_binary"].to_numpy(), sub["probability_COP"].to_numpy())
            metric_rows.append({"family": family, "level": "seed_specific", "training_seed": seed, **m})
        cons = (
            fam.groupby(["case_id_public", "patient_group_id_public", "label_binary", "disease_label"], as_index=False)[
                "probability_COP"
            ]
            .mean()
            .rename(columns={"probability_COP": "consensus_probability_COP"})
        )
        for _, row in cons.iterrows():
            consensus_rows.append({"family": family, **row.to_dict()})
        m = compute_metrics(cons["label_binary"].to_numpy(), cons["consensus_probability_COP"].to_numpy())
        metric_rows.append({"family": family, "level": "mean_consensus", "training_seed": "mean_3_seeds", **m})
    metrics = pd.DataFrame(metric_rows)
    consensus = pd.DataFrame(consensus_rows)
    metrics.to_csv(OUT["runs"] / "clinical_consensus_metrics.csv", index=False, encoding="utf-8-sig")

    stress_metrics = []
    for (family, scenario), sub in stress.groupby(["family", "scenario"]):
        cons = (
            sub.groupby(["case_id_public", "patient_group_id_public", "label_binary", "disease_label"], as_index=False)[
                "probability_COP"
            ]
            .mean()
            .rename(columns={"probability_COP": "consensus_probability_COP"})
        )
        m = compute_metrics(cons["label_binary"].to_numpy(), cons["consensus_probability_COP"].to_numpy())
        stress_metrics.append({"family": family, "scenario": scenario, **m})
    stress_metrics_df = pd.DataFrame(stress_metrics)
    stress_metrics_df.to_csv(OUT["runs"] / "clinical_stress_metrics.csv", index=False, encoding="utf-8-sig")
    return metrics, consensus, stress_metrics_df


def bootstrap_ci(y: np.ndarray, p: np.ndarray, metric: str, rng: np.random.Generator) -> tuple[float, float]:
    vals = []
    n = len(y)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        pp = p[idx]
        if len(np.unique(yy)) < 2:
            continue
        if metric == "AUROC":
            vals.append(roc_auc_score(yy, pp))
        elif metric == "AUPRC":
            vals.append(average_precision_score(yy, pp))
        elif metric == "Brier":
            vals.append(brier_score_loss(yy, pp))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap_diff(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, metric: str, rng: np.random.Generator
) -> tuple[float, float, float]:
    def metric_value(yy: np.ndarray, pp: np.ndarray) -> float:
        if metric == "AUROC":
            return roc_auc_score(yy, pp)
        if metric == "AUPRC":
            return average_precision_score(yy, pp)
        if metric == "Brier":
            return brier_score_loss(yy, pp)
        raise ValueError(metric)

    observed = metric_value(y, p_a) - metric_value(y, p_b)
    vals = []
    n = len(y)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(metric_value(yy, p_a[idx]) - metric_value(yy, p_b[idx]))
    return observed, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_reports(consensus: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    diff_rows = []
    wide = {}
    y_ref = None
    for family, sub in consensus.groupby("family"):
        sub = sub.sort_values("case_id_public")
        y = sub["label_binary"].to_numpy()
        p = sub["consensus_probability_COP"].to_numpy()
        y_ref = y
        wide[family] = p
        for metric in ["AUROC", "AUPRC", "Brier"]:
            lo, hi = bootstrap_ci(y, p, metric, rng)
            rows.append({"family": family, "metric": metric, "ci_low": lo, "ci_high": hi, "n_boot": N_BOOT})
    for a, b in combinations(sorted(wide), 2):
        for metric in ["AUROC", "AUPRC", "Brier"]:
            diff, lo, hi = paired_bootstrap_diff(y_ref, wide[a], wide[b], metric, rng)
            diff_rows.append(
                {
                    "family_a": a,
                    "family_b": b,
                    "metric": metric,
                    "difference_a_minus_b": diff,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_boot": N_BOOT,
                }
            )
    ci = pd.DataFrame(rows)
    diffs = pd.DataFrame(diff_rows)
    ci.to_csv(OUT["runs"] / "clinical_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    diffs.to_csv(OUT["runs"] / "clinical_paired_differences.csv", index=False, encoding="utf-8-sig")
    return ci


def select_model(metrics: pd.DataFrame, stress_metrics: pd.DataFrame) -> dict[str, Any]:
    clean = metrics[metrics["level"].eq("mean_consensus")].copy()
    clean = clean.sort_values("AUROC", ascending=False)
    best_auc = float(clean.iloc[0]["AUROC"])
    retained = clean[clean["AUROC"] >= best_auc - 0.02]["family"].tolist()
    stress_sub = stress_metrics[stress_metrics["family"].isin(retained)]
    worst = stress_sub.groupby("family")["AUROC"].min().reset_index(name="worst_case_AUROC")
    merged = clean.merge(worst, on="family", how="left")
    merged = merged[merged["family"].isin(retained)].copy()
    merged["slope_distance"] = (merged["calibration_slope"] - 1).abs()
    merged["intercept_distance"] = merged["calibration_intercept"].abs()
    complexity = {"ElasticNetLogit": 1, "RF": 2, "GradientBoosting": 3}
    merged["complexity_rank"] = merged["family"].map(complexity)
    merged = merged.sort_values(
        ["worst_case_AUROC", "Brier", "slope_distance", "intercept_distance", "complexity_rank"],
        ascending=[False, True, True, True, True],
    )
    selected = merged.iloc[0].to_dict()
    return {
        "status": "CLINICAL_V2_MODEL_SELECTED",
        "best_clean_AUROC": best_auc,
        "retained_within_0_02": retained,
        "selected_family": selected["family"],
        "selection_table": merged.to_dict(orient="records"),
        "selected_model_metrics": selected,
    }


def source_hashes() -> dict[str, Any]:
    script = Path(__file__)
    git_hash = ""
    try:
        git_hash = subprocess.check_output(
            ["git", "-C", str(V1_PROJECT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_hash = "unavailable"
    package_versions = {}
    for name in ["numpy", "pandas", "sklearn", "xgboost"]:
        try:
            mod = __import__(name)
            package_versions[name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:
            package_versions[name] = f"unavailable: {exc!r}"
    return {
        "v2_source_script": str(script),
        "v2_source_script_sha256": sha256_file(script),
        "v1_project_git_head_read_only": git_hash,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions,
    }


def write_reports(
    validation: dict[str, Any],
    metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    stress_metrics: pd.DataFrame,
    ci: pd.DataFrame,
    selection: dict[str, Any],
    hashes: dict[str, Any],
) -> None:
    clean = metrics[metrics["level"].eq("mean_consensus")].sort_values("AUROC", ascending=False)
    clean_md = df_to_markdown(clean.reset_index(drop=True))
    stress_md = df_to_markdown(stress_metrics.sort_values(["family", "scenario"]).reset_index(drop=True))
    ci_md = df_to_markdown(ci.reset_index(drop=True))
    selected = selection["selected_family"]
    warning = "none"
    if float(selection["selected_model_metrics"]["AUROC"]) < 0.85:
        warning = "Best clean AUROC is below the research expectation of 0.85; this was not used as a stopping or selection rule."
    write_md(
        OUT["reports"] / "00_V2_BOOTSTRAP_REPORT.md",
        f"""
# V2 Bootstrap Report

Bootstrap unit: patient. Resamples: {N_BOOT}. Seed: {BOOTSTRAP_SEED}.

{ci_md}
""",
    )
    write_md(
        OUT["reports"] / "01_CLINICAL_DATA_AUDIT.md",
        f"""
# Clinical Data Audit

Status: PASSED

```json
{json.dumps(validation, indent=2, ensure_ascii=False)}
```

Forbidden feature audit: `audit/forbidden_feature_audit.csv`
Patient leakage audit: `audit/patient_leakage_audit.json`
""",
    )
    write_md(
        OUT["reports"] / "02_CLINICAL_NESTED_RESULTS.md",
        f"""
# Clinical Nested Results

Clean mean-consensus OOF metrics:

{clean_md}

Fold-level metrics are in `runs/clinical/clinical_seed_fold_metrics.csv`.
Patient-level private OOF predictions are in `runs/clinical/clinical_patient_oof_predictions_PRIVATE.csv`.
""",
    )
    write_md(
        OUT["reports"] / "03_CLINICAL_ROBUSTNESS_REPORT.md",
        f"""
# Clinical Robustness Report

Scenarios:

- S0: original clinical input
- S1: 10 percent label-blind random missingness on legal predictors
- S2: 20 percent label-blind random missingness on legal predictors
- S3: 30 percent label-blind random missingness on legal predictors
- S4: formal laboratory panel missingness stress; 20 percent of patients have the laboratory panel removed

{stress_md}
""",
    )
    write_md(
        OUT["reports"] / "04_CLINICAL_SELECTION_REPORT.md",
        f"""
# Clinical Selection Report

Selection status: {selection['status']}

Selected clinical model family: `{selected}`

Selection rule:

1. Find highest clean nested consensus AUROC.
2. Retain candidates within 0.02 AUROC of the top.
3. Compare worst-case AUROC across S0-S4.
4. Break close ties by lower Brier, calibration slope closer to 1, intercept closer to 0, then lower complexity.

Selection evidence:

```json
{json.dumps(selection, indent=2, ensure_ascii=False)}
```

Scientific audit warning: {warning}
""",
    )

    selected_family = selection["selected_family"]
    selected_fold_configs = fold_metrics[fold_metrics["family"].eq(selected_family)][
        ["training_seed", "outer_fold_id", "selected_missingness", "selected_params_json", "inner_cv_auroc"]
    ].to_dict(orient="records")
    mode_missingness = Counter([r["selected_missingness"] for r in selected_fold_configs]).most_common(1)[0][0]
    mode_params_json = Counter([r["selected_params_json"] for r in selected_fold_configs]).most_common(1)[0][0]
    lock = {
        "status": "CLINICAL_MODEL_LOCK_RECOMMENDATION_OOF_ONLY",
        "final_algorithm": selected_family,
        "exact_features": FEATURES,
        "preprocessing": {
            "selected_family_modal_missingness_for_future_full_development_fit": mode_missingness,
            "all_preprocessing_fit_scope": "training_fold_only",
            "per_outer_fold_selected_configs": selected_fold_configs,
        },
        "missingness_handling": "Selected inside each outer training fold from registry; future full-development fit should use modal nested-selected config unless changed by human review.",
        "hyperparameters": {
            "modal_selected_params_for_future_full_development_fit": json.loads(mode_params_json),
            "per_outer_fold_selected_configs": selected_fold_configs,
        },
        "probability_calibration": "Platt logistic calibration fitted from outer-training cross-fitted raw probabilities only.",
        "seeds": TRAINING_SEEDS,
        "split_hash": sha256_file(OUT["splits"] / "locked_nested_split_patient_manifest.csv"),
        "training_data_hash": sha256_file(
            V1_NESTED
            / "data"
            / "internal_development"
            / "clinical_features_basic_corrected_for_development.csv"
        ),
        "source_code_hash": hashes["v2_source_script_sha256"],
        "package_versions": hashes["package_versions"],
        "selection_evidence": selection,
        "full_development_refit_performed": False,
        "resubstitution_performance_reported": False,
    }
    write_json(OUT["locked"] / "CLINICAL_MODEL_LOCK.json", lock)


def write_completion(
    validation: dict[str, Any],
    metrics: pd.DataFrame,
    stress_metrics: pd.DataFrame,
    selection: dict[str, Any],
    hashes: dict[str, Any],
) -> None:
    end = datetime.now(timezone.utc)
    selected_family = selection["selected_family"]
    selected_metrics = metrics[
        metrics["family"].eq(selected_family) & metrics["level"].eq("mean_consensus")
    ].iloc[0].to_dict()
    selected_stress = stress_metrics[stress_metrics["family"].eq(selected_family)]
    completion = {
        "status": selection["status"],
        "start_time": START_TIME.isoformat(),
        "end_time": end.isoformat(),
        "patient_counts": validation["patient_counts"],
        "split_validation": validation["split_validation"],
        "candidate_models_completed": ["ElasticNetLogit", "RF", "GradientBoosting"],
        "failures": [],
        "external_label_reads": 0,
        "external_prediction_reads": 0,
        "external_metric_reads": 0,
        "leakage_checks": {
            "patient_leakage": "passed",
            "preprocessing_leakage": "passed",
            "forbidden_features": "passed",
            "probability_contract": "passed",
        },
        "selected_clinical_model": selected_family,
        "selected_model_metrics": selected_metrics,
        "robustness_summary": selected_stress.to_dict(orient="records"),
        "output_manifest": [],
        "git/source hashes": hashes,
    }
    write_json(OUT["audit"] / "RUN_COMPLETED.json", completion)
    outputs = []
    for path in ROOT.rglob("*"):
        if path.is_file() and path.name not in {"checksums.sha256", "RUN_COMPLETED.json"}:
            outputs.append({"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    completion["output_manifest"] = outputs
    write_json(OUT["audit"] / "RUN_COMPLETED.json", completion)
    outputs = []
    for path in ROOT.rglob("*"):
        if path.is_file() and path.name != "checksums.sha256":
            outputs.append({"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    checksum_lines = []
    for item in outputs:
        rel = Path(item["path"]).relative_to(ROOT)
        checksum_lines.append(f"{item['sha256']}  {rel.as_posix()}")
    (ROOT / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")


def finalize_only() -> None:
    ensure_dirs()
    case_master, original_clinical, nested, clinical, schema = read_inputs()
    validation = validate_data(case_master, original_clinical, nested, clinical)
    metrics_path = OUT["runs"] / "clinical_consensus_metrics.csv"
    fold_path = OUT["runs"] / "clinical_seed_fold_metrics.csv"
    stress_path = OUT["runs"] / "clinical_stress_metrics.csv"
    ci_path = OUT["runs"] / "clinical_bootstrap_ci.csv"
    missing = [str(p) for p in [metrics_path, fold_path, stress_path, ci_path] if not p.exists()]
    if missing:
        stop_run("finalize_only_missing_training_outputs", {"missing": missing})
    metrics = pd.read_csv(metrics_path)
    fold_metrics = pd.read_csv(fold_path)
    stress_metrics = pd.read_csv(stress_path)
    ci = pd.read_csv(ci_path)
    selection = select_model(metrics, stress_metrics)
    hashes = source_hashes()
    write_reports(validation, metrics, fold_metrics, stress_metrics, ci, selection, hashes)
    write_completion(validation, metrics, stress_metrics, selection, hashes)

    clean = metrics[metrics["level"].eq("mean_consensus")].sort_values("AUROC", ascending=False)
    selected_family = selection["selected_family"]
    selected_clean = clean[clean["family"].eq(selected_family)].iloc[0]
    selected_stress = stress_metrics[stress_metrics["family"].eq(selected_family)]
    worst = selected_stress.loc[selected_stress["AUROC"].idxmin()]
    warning = "YES" if float(selected_clean["AUROC"]) < 0.85 else "NO"
    print(f"STATUS: {selection['status']}")
    print(
        "CLEAN_CLINICAL: "
        f"{selected_family} AUROC={selected_clean['AUROC']:.4f} "
        f"AUPRC={selected_clean['AUPRC']:.4f} Brier={selected_clean['Brier']:.4f} "
        f"cal_intercept={selected_clean['calibration_intercept']:.4f} "
        f"cal_slope={selected_clean['calibration_slope']:.4f}"
    )
    print(
        "STRESS_WORST_CASE: "
        f"scenario={worst['scenario']} AUROC={worst['AUROC']:.4f} "
        f"AUPRC={worst['AUPRC']:.4f} Brier={worst['Brier']:.4f}"
    )
    print(f"CLINICAL_LOCK: {OUT['locked'] / 'CLINICAL_MODEL_LOCK.json'}")
    print(f"REPORTS: {OUT['reports']}")
    print(f"RUN_COMPLETED: {OUT['audit'] / 'RUN_COMPLETED.json'}")
    print(f"SCIENTIFIC_AUDIT_WARNINGS: {warning}")


def main() -> None:
    if "--finalize-only" in sys.argv:
        finalize_only()
        return
    ensure_dirs()
    public_release_readme()
    external_access_log()
    split_hashes = copy_locked_split_sources()
    make_governance_docs(split_hashes)
    case_master, original_clinical, nested, clinical, schema = read_inputs()
    validation = validate_data(case_master, original_clinical, nested, clinical)
    write_data_audits(case_master, clinical, nested, schema, validation)
    candidate_registry()
    pred, stress, fold_metrics = run_nested(nested, clinical)
    fold_metrics.to_csv(OUT["runs"] / "clinical_seed_fold_metrics.csv", index=False, encoding="utf-8-sig")
    metrics, consensus, stress_metrics = summarize_predictions(pred, stress)
    ci = bootstrap_reports(consensus)
    selection = select_model(metrics, stress_metrics)
    hashes = source_hashes()
    write_reports(validation, metrics, fold_metrics, stress_metrics, ci, selection, hashes)
    write_completion(validation, metrics, stress_metrics, selection, hashes)

    clean = metrics[metrics["level"].eq("mean_consensus")].sort_values("AUROC", ascending=False)
    selected_family = selection["selected_family"]
    selected_clean = clean[clean["family"].eq(selected_family)].iloc[0]
    selected_stress = stress_metrics[stress_metrics["family"].eq(selected_family)]
    worst = selected_stress.loc[selected_stress["AUROC"].idxmin()]
    warning = "YES" if float(selected_clean["AUROC"]) < 0.85 else "NO"
    print(f"STATUS: {selection['status']}")
    print(
        "CLEAN_CLINICAL: "
        f"{selected_family} AUROC={selected_clean['AUROC']:.4f} "
        f"AUPRC={selected_clean['AUPRC']:.4f} Brier={selected_clean['Brier']:.4f} "
        f"cal_intercept={selected_clean['calibration_intercept']:.4f} "
        f"cal_slope={selected_clean['calibration_slope']:.4f}"
    )
    print(
        "STRESS_WORST_CASE: "
        f"scenario={worst['scenario']} AUROC={worst['AUROC']:.4f} "
        f"AUPRC={worst['AUPRC']:.4f} Brier={worst['Brier']:.4f}"
    )
    print(f"CLINICAL_LOCK: {OUT['locked'] / 'CLINICAL_MODEL_LOCK.json'}")
    print(f"REPORTS: {OUT['reports']}")
    print(f"RUN_COMPLETED: {OUT['audit'] / 'RUN_COMPLETED.json'}")
    print(f"SCIENTIFIC_AUDIT_WARNINGS: {warning}")


if __name__ == "__main__":
    main()
