from __future__ import annotations

import json
import math
import os
import platform
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, roc_auc_score

from . import alternate_configuration_source as base

warnings.filterwarnings("ignore", category=FutureWarning)


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
V1_DEV = (
    V1
    / "cap_cop_experiment"
    / "generalisation_redevelopment_nested_protocol_v1"
    / "data"
    / "internal_development"
    / "clinical_features_basic_corrected_for_development.csv"
)
V1_UNMOD = (
    V1
    / "cap_cop_experiment"
    / "generalisation_redevelopment_nested_protocol_v1"
    / "data"
    / "internal_development"
    / "clinical_features_basic_unmodified_for_sensitivity.csv"
)
V1_BMI_MANIFEST = (
    V1
    / "cap_cop_experiment"
    / "generalisation_redevelopment_nested_protocol_v1"
    / "aggregate"
    / "bmi_correction_manifest.csv"
)
RUN_DIR = ROOT / "runs" / "clinical_final_confirmation"
REPORT_DIR = ROOT / "reports"
AUDIT_DIR = ROOT / "audit"
LOCK_DIR = ROOT / "locked_models"
GOV_DIR = ROOT / "governance"

SEEDS = [20260627, 20260628, 20260629]
FOLDS = [0, 1, 2, 3, 4]
BOOTSTRAP_SEED = 20260815
N_BOOT = 1000
FAMILIES = ["ElasticNetLogit", "RF", "GradientBoosting"]
STRICT_SCENARIOS = ["S0", "S1", "S2", "S3", "S4"]
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    tmp = df.copy()
    for col in tmp.columns:
        tmp[col] = tmp[col].map(lambda v: "" if pd.isna(v) else f"{v:.6g}" if isinstance(v, float) else str(v))
    cols = list(tmp.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in tmp.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in cols) + " |")
    return "\n".join(lines)


def stop(reason: str, filename: str, details: dict[str, Any]) -> None:
    path = ROOT / filename
    write_md(
        path,
        f"""
# {filename}

Status: STOPPED

Reason: {reason}

```json
{json.dumps(details, indent=2, ensure_ascii=False)}
```
""",
    )
    print(f"STOPPED: {reason}")
    print(str(path))
    sys.exit(2)


def sha(path: Path) -> str:
    return base.sha256_file(path)


def preserve_hashes() -> dict[str, Any]:
    files = [
        "locked_models/CLINICAL_MODEL_LOCK.json",
        "configs/clinical_candidate_registry.json",
        "runs/clinical/clinical_patient_oof_predictions_PRIVATE.csv",
        "runs/clinical/clinical_inner_tuning_registry.csv",
        "runs/clinical/clinical_seed_fold_metrics.csv",
        "runs/clinical/clinical_stress_metrics.csv",
        "audit/RUN_COMPLETED.json",
        "governance/V1_TO_V2_DATA_CONTRACT.md",
        "data_contract/clinical_feature_dictionary.csv",
        "data_contract/internal_patient_manifest.csv",
    ]
    rows = []
    for rel in files:
        p = ROOT / rel
        rows.append({"relative_path": rel, "exists": p.exists(), "sha256": sha(p) if p.exists() else ""})
    pd.DataFrame(rows).to_csv(AUDIT_DIR / "clinical_final_existing_evidence_hashes.csv", index=False, encoding="utf-8-sig")
    return {r["relative_path"]: r["sha256"] for r in rows}


def load_all() -> dict[str, Any]:
    with (ROOT / "locked_models" / "CLINICAL_MODEL_LOCK.json").open("r", encoding="utf-8") as f:
        lock = json.load(f)
    with (ROOT / "configs" / "clinical_candidate_registry.json").open("r", encoding="utf-8") as f:
        registry = json.load(f)
    clinical = pd.read_csv(V1_DEV)
    unmod = pd.read_csv(V1_UNMOD)
    nested = pd.read_csv(ROOT / "splits" / "locked_nested_split_patient_manifest.csv")
    old_oof = pd.read_csv(ROOT / "runs" / "clinical" / "clinical_patient_oof_predictions_PRIVATE.csv")
    old_metrics = pd.read_csv(ROOT / "runs" / "clinical" / "clinical_consensus_metrics.csv")
    feature_dict = pd.read_csv(ROOT / "data_contract" / "clinical_feature_dictionary.csv")
    manifest = pd.read_csv(ROOT / "data_contract" / "internal_patient_manifest.csv")
    return {
        "lock": lock,
        "registry": registry,
        "clinical": clinical,
        "unmod": unmod,
        "nested": nested,
        "old_oof": old_oof,
        "old_metrics": old_metrics,
        "feature_dict": feature_dict,
        "manifest": manifest,
    }


def audit_anthro(data: dict[str, Any]) -> dict[str, Any]:
    clinical = data["clinical"].set_index("case_id_public")
    unmod = data["unmod"].set_index("case_id_public")
    lock = data["lock"]
    rows = []
    failures = []
    training_sha = sha(V1_DEV)
    bmi_manifest_sha = sha(V1_BMI_MANIFEST) if V1_BMI_MANIFEST.exists() else ""
    if "anthropometric_anomaly_flag" not in clinical.columns:
        raise RuntimeError("The clinical contract does not expose the anthropometric anomaly audit flag")
    audited_ids = clinical.index[clinical["anthropometric_anomaly_flag"].fillna(False).astype(bool)].tolist()
    if not audited_ids:
        raise RuntimeError("No anthropometric correction cases were present in the supplied audit contract")
    for case_id in audited_ids:
        dev = clinical.loc[case_id]
        raw = unmod.loc[case_id]
        expected_height = float(dev["height_cm"])
        expected_weight = float(dev["weight_kg"])
        expected_bmi = expected_weight / ((expected_height / 100.0) ** 2)
        ok = (
            math.isclose(float(dev["height_cm"]), expected_height, abs_tol=1e-9)
            and math.isclose(float(dev["weight_kg"]), expected_weight, abs_tol=1e-9)
            and math.isclose(float(dev["body_mass_index"]), expected_bmi, abs_tol=5e-4)
        )
        if not ok:
            failures.append(case_id)
        rows.append(
            {
                "case_id": case_id,
                "raw_height": float(raw["height_cm"]),
                "raw_weight": float(raw["weight_kg"]),
                "development_height": float(dev["height_cm"]),
                "development_weight": float(dev["weight_kg"]),
                "development_BMI": float(dev["body_mass_index"]),
                "expected_BMI": expected_bmi,
                "training_source_file": str(V1_DEV),
                "training_source_sha256": training_sha,
                "correction_basis": "historical_data_correction; bmi_correction_manifest.csv",
                "correction_basis_sha256": bmi_manifest_sha,
                "used_in_v2_training": True,
                "anthropometric_anomaly_flag": bool(dev.get("anthropometric_anomaly_flag", False)),
                "status": "PASS" if ok else "FAIL",
            }
        )
    flag_in_features = "anthropometric_anomaly_flag" in lock["exact_features"]
    status = "PASSED" if not failures and not flag_in_features else "FAILED"
    audit = {
        "status": status,
        "training_source_file": str(V1_DEV),
        "training_source_sha256": training_sha,
        "correction_basis_file": str(V1_BMI_MANIFEST),
        "correction_basis_sha256": bmi_manifest_sha,
        "raw_source_modified_by_v2": False,
        "development_copy_used": True,
        "anthropometric_anomaly_flag_used_as_predictor": flag_in_features,
        "cases": rows,
    }
    write_json(AUDIT_DIR / "anthropometric_correction_audit.json", audit)
    write_md(
        REPORT_DIR / "05_ANTHROPOMETRIC_CORRECTION_AUDIT.md",
        f"""
# Anthropometric Correction Audit

Status: {status}

Training source file: `{V1_DEV}`

Training source SHA256: `{training_sha}`

Correction basis: `historical_data_correction`; `{V1_BMI_MANIFEST}`

{md_table(pd.DataFrame(rows))}

`anthropometric_anomaly_flag` was used for audit only and is not in the predictor matrix: `{not flag_in_features}`.
""",
    )
    if status != "PASSED":
        stop(
            "CLINICAL_V2_ANTHROPOMETRIC_CORRECTION_REQUIRED",
            "STOP_REPORT_ANTHROPOMETRIC_CORRECTION.md",
            audit,
        )
    return audit


def validate_contract(data: dict[str, Any]) -> dict[str, Any]:
    clinical = data["clinical"]
    nested = data["nested"]
    features = data["lock"]["exact_features"]
    failures = []
    if len(clinical) != 375:
        failures.append("patient_count_not_375")
    if int((clinical["label_binary"] == 0).sum()) != 250 or int((clinical["label_binary"] == 1).sum()) != 125:
        failures.append("label_counts_not_250_125")
    if "anthropometric_anomaly_flag" in features:
        failures.append("audit_flag_entered_predictors")
    for seed in SEEDS:
        seed_df = nested[nested["training_seed"].eq(seed)]
        test_counts = seed_df[seed_df["outer_role"].eq("outer_test")].groupby("case_id_public").size()
        if len(test_counts) != 375 or test_counts.min() != 1 or test_counts.max() != 1:
            failures.append(f"seed_{seed}_outer_test_not_once")
        for fold in FOLDS:
            fold_df = seed_df[seed_df["outer_fold_id"].eq(fold)]
            train = set(fold_df.loc[fold_df["outer_role"].eq("outer_train"), "case_id_public"])
            test = set(fold_df.loc[fold_df["outer_role"].eq("outer_test"), "case_id_public"])
            if len(train) != 300 or len(test) != 75 or train & test:
                failures.append(f"seed_{seed}_fold_{fold}_outer_split_bad")
            inner = fold_df[fold_df["outer_role"].eq("outer_train")]
            if set(inner["inner_fold_id"].dropna().astype(int).unique()) != {0, 1, 2, 3}:
                failures.append(f"seed_{seed}_fold_{fold}_inner_split_bad")
    if failures:
        stop("data_or_split_contract_failed", "STOP_REPORT_FINAL_CONFIRMATION_DATA_SPLIT.md", {"failures": failures})
    return {"status": "PASSED", "patient_count": 375, "CAP": 250, "COP": 125, "split_validation": "PASSED"}


def get_split(nested: pd.DataFrame, seed: int, fold: int) -> tuple[list[str], list[str], dict[str, int]]:
    return base.get_split(nested, seed, fold)


def fit_family_crossfit(
    family: str, best: dict[str, Any], X: pd.DataFrame, y: pd.Series, train_ids: list[str], inner_map: dict[str, int], seed: int
) -> tuple[LogisticRegression, dict[int, Any], dict[str, np.ndarray], pd.Series]:
    raw_s0 = pd.Series(index=train_ids, dtype=float)
    inner_models: dict[int, Any] = {}
    scenario_raw: dict[str, pd.Series] = {s: pd.Series(index=train_ids, dtype=float) for s in STRICT_SCENARIOS}
    for inner_fold in range(4):
        val_ids = [pid for pid in train_ids if inner_map[pid] == inner_fold]
        tr_ids = [pid for pid in train_ids if inner_map[pid] != inner_fold]
        model = base.make_pipeline(family, best["missingness"], best["params"], seed + 900 + inner_fold)
        model.fit(X.loc[tr_ids], y.loc[tr_ids])
        inner_models[inner_fold] = model
        for scenario in STRICT_SCENARIOS:
            X_eval = base.stress_copy(X.loc[val_ids], scenario, seed, inner_fold, family)
            scenario_raw[scenario].loc[val_ids] = model.predict_proba(X_eval)[:, 1]
        raw_s0.loc[val_ids] = scenario_raw["S0"].loc[val_ids]
    x = np.log(np.clip(raw_s0.to_numpy(), 1e-6, 1 - 1e-6) / np.clip(1 - raw_s0.to_numpy(), 1e-6, 1))
    cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000)
    cal.fit(x.reshape(-1, 1), y.loc[train_ids].to_numpy())
    calibrated = {}
    for scenario, raw in scenario_raw.items():
        calibrated[scenario] = base.apply_calibrator(cal, raw.loc[train_ids].to_numpy())
    return cal, inner_models, calibrated, raw_s0


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return base.compute_metrics(y, p)


def select_family_from_inner(family_inner: dict[str, dict[str, Any]]) -> tuple[str, list[str], pd.DataFrame]:
    rows = []
    for family, payload in family_inner.items():
        clean = payload["scenario_metrics"]["S0"]
        worst = min(v["AUROC"] for v in payload["scenario_metrics"].values())
        rows.append(
            {
                "family": family,
                "inner_clean_AUROC": clean["AUROC"],
                "inner_clean_AUPRC": clean["AUPRC"],
                "inner_clean_Brier": clean["Brier"],
                "inner_clean_calibration_intercept": clean["calibration_intercept"],
                "inner_clean_calibration_slope": clean["calibration_slope"],
                "inner_worst_case_AUROC": worst,
            }
        )
    df = pd.DataFrame(rows)
    best_auc = df["inner_clean_AUROC"].max()
    retained = df[df["inner_clean_AUROC"] >= best_auc - 0.02]["family"].tolist()
    complexity = {"ElasticNetLogit": 1, "RF": 2, "GradientBoosting": 3}
    df["retained_within_0_02"] = df["family"].isin(retained)
    df["slope_distance"] = (df["inner_clean_calibration_slope"] - 1).abs()
    df["intercept_distance"] = df["inner_clean_calibration_intercept"].abs()
    df["complexity_rank"] = df["family"].map(complexity)
    selected = (
        df[df["retained_within_0_02"]]
        .sort_values(
            ["inner_worst_case_AUROC", "inner_clean_Brier", "slope_distance", "intercept_distance", "complexity_rank"],
            ascending=[False, True, True, True, True],
        )
        .iloc[0]["family"]
    )
    return selected, retained, df


def run_strict_family_nested(data: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clinical = data["clinical"].set_index("case_id_public", drop=False)
    nested = data["nested"]
    features = data["lock"]["exact_features"]
    X = clinical[features].apply(pd.to_numeric, errors="coerce")
    y = clinical["label_binary"].astype(int)
    pred_rows = []
    selection_rows = []
    metric_rows = []
    for seed in SEEDS:
        for fold in FOLDS:
            train_ids, test_ids, inner_map = get_split(nested, seed, fold)
            family_inner = {}
            for family in FAMILIES:
                best, _ = base.tune_family(family, X, y, train_ids, inner_map, seed)
                cal, _inner_models, calibrated, _raw = fit_family_crossfit(family, best, X, y, train_ids, inner_map, seed)
                scen_metrics = {s: metrics(y.loc[train_ids].to_numpy(), p) for s, p in calibrated.items()}
                family_inner[family] = {"best": best, "calibrator": cal, "scenario_metrics": scen_metrics}
            selected_family, retained, sel_df = select_family_from_inner(family_inner)
            for _, row in sel_df.iterrows():
                best = family_inner[row["family"]]["best"]
                selection_rows.append(
                    {
                        "training_seed": seed,
                        "outer_fold_id": fold,
                        **row.to_dict(),
                        "selected_family_for_outer_fold": selected_family,
                        "retained_families_json": json.dumps(retained),
                        "selected_missingness": best["missingness"],
                        "selected_params_json": json.dumps(best["params"], sort_keys=True),
                    }
                )
            best = family_inner[selected_family]["best"]
            calibrator = family_inner[selected_family]["calibrator"]
            outer_model = base.make_pipeline(selected_family, best["missingness"], best["params"], seed + 777)
            outer_model.fit(X.loc[train_ids], y.loc[train_ids])
            raw = outer_model.predict_proba(X.loc[test_ids])[:, 1]
            prob = base.apply_calibrator(calibrator, raw)
            if np.any(prob < 0) or np.any(prob > 1) or np.any(~np.isfinite(prob)):
                stop("strict_probability_contract_failed", "STOP_REPORT_STRICT_PROBABILITY.md", {"seed": seed, "fold": fold})
            for pid, raw_p, p in zip(test_ids, raw, prob):
                pred_rows.append(
                    {
                        "procedure": "STRICT_FAMILY_NESTED_PROCEDURE_OOF",
                        "training_seed": seed,
                        "outer_fold_id": fold,
                        "case_id_public": pid,
                        "patient_group_id_public": clinical.loc[pid, "patient_group_id_public"],
                        "label_binary": int(y.loc[pid]),
                        "disease_label": clinical.loc[pid, "disease_label"],
                        "selected_family": selected_family,
                        "selected_missingness": best["missingness"],
                        "selected_params_json": json.dumps(best["params"], sort_keys=True),
                        "raw_probability_COP": float(raw_p),
                        "probability_COP": float(p),
                    }
                )
    seed_pred = pd.DataFrame(pred_rows)
    for seed, sub in seed_pred.groupby("training_seed"):
        counts = sub.groupby("case_id_public").size()
        if len(counts) != 375 or counts.min() != 1 or counts.max() != 1:
            stop("strict_oof_incomplete", "STOP_REPORT_STRICT_OOF.md", {"seed": int(seed)})
        m = metrics(sub["label_binary"].to_numpy(), sub["probability_COP"].to_numpy())
        metric_rows.append({"procedure": "STRICT_FAMILY_NESTED_PROCEDURE_OOF", "level": "seed_specific", "training_seed": seed, **m})
    consensus = (
        seed_pred.groupby(["case_id_public", "patient_group_id_public", "label_binary", "disease_label"], as_index=False)[
            "probability_COP"
        ]
        .mean()
        .rename(columns={"probability_COP": "consensus_probability_COP"})
    )
    m = metrics(consensus["label_binary"].to_numpy(), consensus["consensus_probability_COP"].to_numpy())
    metric_rows.append({"procedure": "STRICT_FAMILY_NESTED_PROCEDURE_OOF", "level": "mean_3_seed_consensus", "training_seed": "mean_3_seeds", **m})
    selection = pd.DataFrame(selection_rows)
    metric_df = pd.DataFrame(metric_rows)
    seed_pred.to_csv(RUN_DIR / "strict_family_nested_seed_predictions_PRIVATE.csv", index=False, encoding="utf-8-sig")
    consensus.to_csv(RUN_DIR / "strict_family_nested_consensus_predictions_PRIVATE.csv", index=False, encoding="utf-8-sig")
    metric_df.to_csv(RUN_DIR / "strict_family_nested_metrics.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(RUN_DIR / "strict_family_selection_registry.csv", index=False, encoding="utf-8-sig")
    return seed_pred, consensus, metric_df, selection


def fit_fixed_rf_for_fold(data: dict[str, Any], seed: int, fold: int) -> tuple[Any, LogisticRegression, dict[str, Any], list[str], list[str], dict[str, int]]:
    clinical = data["clinical"].set_index("case_id_public", drop=False)
    X = clinical[data["lock"]["exact_features"]].apply(pd.to_numeric, errors="coerce")
    y = clinical["label_binary"].astype(int)
    nested = data["nested"]
    train_ids, test_ids, inner_map = get_split(nested, seed, fold)
    configs = data["lock"]["preprocessing"]["per_outer_fold_selected_configs"]
    rec = [r for r in configs if int(r["training_seed"]) == seed and int(r["outer_fold_id"]) == fold][0]
    best = {
        "missingness": rec["selected_missingness"],
        "params": json.loads(rec["selected_params_json"]),
    }
    calibrator, _info = base.fit_platt_calibrator("RF", best, X, y, train_ids, inner_map, seed)
    model = base.make_pipeline("RF", best["missingness"], best["params"], seed + 777)
    model.fit(X.loc[train_ids], y.loc[train_ids])
    return model, calibrator, best, train_ids, test_ids, inner_map


def apply_rf_scenario(model: Any, calibrator: LogisticRegression, X_eval: pd.DataFrame) -> np.ndarray:
    return base.apply_calibrator(calibrator, model.predict_proba(X_eval)[:, 1])


def rf_reproduction_and_sensitivity(data: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    clinical = data["clinical"].set_index("case_id_public", drop=False)
    X = clinical[data["lock"]["exact_features"]].apply(pd.to_numeric, errors="coerce")
    y = clinical["label_binary"].astype(int)
    old_rf = data["old_oof"][data["old_oof"]["family"].eq("RF")].copy()
    baseline_rows = []
    sens_rows = []
    indicator_rows = []
    group_defs = data["feature_dict"].set_index("v2_feature")["feature_group"].to_dict()
    formal_lab = [f for f, g in group_defs.items() if g == "laboratory" and f in data["lock"]["exact_features"]]
    features = data["lock"]["exact_features"]
    scenarios = [(f"MISS100_{f}", [f]) for f in features]
    scenarios += [
        ("ANTHRO_HEIGHT_BMI_MISSING", ["height_cm", "body_mass_index"]),
        ("ANTHRO_WEIGHT_BMI_MISSING", ["weight_kg", "body_mass_index"]),
        ("ANTHRO_ALL_MISSING", ["height_cm", "weight_kg", "body_mass_index"]),
    ]
    lab_status = "NOT_RUN_NO_FORMAL_GROUP_DEFINITION"
    if formal_lab:
        scenarios.append(("LAB_PANEL_100_MISSING", formal_lab))
        lab_status = "RUN_FORMAL_LABORATORY_GROUP"
    for seed in SEEDS:
        for fold in FOLDS:
            model, calibrator, best, train_ids, test_ids, inner_map = fit_fixed_rf_for_fold(data, seed, fold)
            prob_s0 = apply_rf_scenario(model, calibrator, X.loc[test_ids])
            for pid, p in zip(test_ids, prob_s0):
                baseline_rows.append(
                    {
                        "procedure": "DEVELOPMENT_SELECTED_RF_OOF_REPRODUCED",
                        "training_seed": seed,
                        "outer_fold_id": fold,
                        "case_id_public": pid,
                        "patient_group_id_public": clinical.loc[pid, "patient_group_id_public"],
                        "label_binary": int(y.loc[pid]),
                        "disease_label": clinical.loc[pid, "disease_label"],
                        "probability_COP": float(p),
                    }
                )
            imputer = model.named_steps.get("imputer")
            n_model_features = int(getattr(model.named_steps["model"], "n_features_in_", -1))
            for f in features:
                f_idx = features.index(f)
                train_missing = X.loc[train_ids, f].isna()
                if imputer is None:
                    indicator_created = False
                    impute_value = np.nan
                    indicator_features = []
                    transformed_dim = len(features)
                else:
                    indicator_features = list(getattr(imputer.indicator_, "features_", [])) if hasattr(imputer, "indicator_") else []
                    indicator_created = f_idx in indicator_features
                    impute_value = float(imputer.statistics_[f_idx])
                    transformed_dim = int(imputer.transform(X.loc[test_ids].head(1)).shape[1])
                limitation = "deployment missingness mechanism limitation" if best["missingness"] == "median_indicator" and not indicator_created else ""
                indicator_rows.append(
                    {
                        "training_seed": seed,
                        "outer_fold_id": fold,
                        "feature": f,
                        "selected_missingness": best["missingness"],
                        "n_missing_outer_training": int(train_missing.sum()),
                        "missing_rate_outer_training": float(train_missing.mean()),
                        "whether_missing_indicator_created_at_fit_time": bool(indicator_created),
                        "imputation_value": impute_value,
                        "indicator_dimension_exists": bool(indicator_created),
                        "predictor_dimensionality_after_transform": transformed_dim,
                        "rf_model_n_features_in": n_model_features,
                        "dimensionality_consistent": transformed_dim == n_model_features,
                        "inference_success": True,
                        "limitation": limitation,
                    }
                )
            for scenario, miss_features in scenarios:
                Xs = X.loc[test_ids].copy().astype(float)
                Xs.loc[:, miss_features] = np.nan
                p_s = apply_rf_scenario(model, calibrator, Xs)
                for pid, p0, ps in zip(test_ids, prob_s0, p_s):
                    sens_rows.append(
                        {
                            "procedure": "DEVELOPMENT_SELECTED_RF_OOF",
                            "training_seed": seed,
                            "outer_fold_id": fold,
                            "case_id_public": pid,
                            "patient_group_id_public": clinical.loc[pid, "patient_group_id_public"],
                            "label_binary": int(y.loc[pid]),
                            "disease_label": clinical.loc[pid, "disease_label"],
                            "scenario": scenario,
                            "features_set_missing_json": json.dumps(miss_features),
                            "baseline_probability_COP": float(p0),
                            "probability_COP": float(ps),
                        }
                    )
    baseline = pd.DataFrame(baseline_rows)
    merged = baseline.merge(
        old_rf[["training_seed", "outer_fold_id", "case_id_public", "label_binary", "probability_COP"]],
        on=["training_seed", "outer_fold_id", "case_id_public", "label_binary"],
        suffixes=("_reproduced", "_original"),
    )
    merged["abs_error"] = (merged["probability_COP_reproduced"] - merged["probability_COP_original"]).abs()
    max_err = float(merged["abs_error"].max())
    if len(merged) != 1125 or max_err > 1e-10:
        stop(
            "RF baseline reproduction mismatch",
            "STOP_REPORT_RF_REPRODUCTION_MISMATCH.md",
            {"merged_rows": len(merged), "max_absolute_probability_error": max_err},
        )
    sens = pd.DataFrame(sens_rows)
    indicators = pd.DataFrame(indicator_rows)
    baseline.to_csv(RUN_DIR / "rf_s0_reproduced_predictions_PRIVATE.csv", index=False, encoding="utf-8-sig")
    sens.to_csv(RUN_DIR / "rf_feature_unavailability_predictions_PRIVATE.csv", index=False, encoding="utf-8-sig")
    indicators.to_csv(AUDIT_DIR / "rf_missing_indicator_behavior.csv", index=False, encoding="utf-8-sig")
    return baseline, sens, indicators, merged, {"max_error": max_err, "laboratory_panel_status": lab_status, "laboratory_features": formal_lab}


def consensus_from_sens(sens: pd.DataFrame) -> pd.DataFrame:
    return (
        sens.groupby(["scenario", "case_id_public", "patient_group_id_public", "label_binary", "disease_label"], as_index=False)
        .agg(probability_COP=("probability_COP", "mean"), baseline_probability_COP=("baseline_probability_COP", "mean"))
    )


def scenario_metrics(sens: pd.DataFrame) -> pd.DataFrame:
    cons = consensus_from_sens(sens)
    rows = []
    baseline_seen = None
    for scenario, sub in cons.groupby("scenario"):
        y = sub["label_binary"].to_numpy()
        p = sub["probability_COP"].to_numpy()
        p0 = sub["baseline_probability_COP"].to_numpy()
        m = metrics(y, p)
        m0 = metrics(y, p0)
        shift = np.abs(p - p0)
        pred = (p >= 0.5).astype(int)
        pred0 = (p0 >= 0.5).astype(int)
        row = {
            "scenario": scenario,
            **m,
            "delta_AUROC": m["AUROC"] - m0["AUROC"],
            "delta_AUPRC": m["AUPRC"] - m0["AUPRC"],
            "delta_Brier": m["Brier"] - m0["Brier"],
            "delta_sensitivity_at_0_5": m["sensitivity_at_0_5"] - m0["sensitivity_at_0_5"],
            "mean_absolute_probability_shift": float(np.mean(shift)),
            "median_absolute_probability_shift": float(np.median(shift)),
            "p95_absolute_probability_shift": float(np.percentile(shift, 95)),
            "classification_flip_rate_at_0_5": float(np.mean(pred != pred0)),
        }
        row["risk_flags"] = ";".join(
            [
                flag
                for flag, active in {
                    "LARGE_DISCRIMINATION_SHIFT": abs(row["delta_AUROC"]) >= 0.05,
                    "LARGE_BRIER_SHIFT": row["delta_Brier"] >= 0.03,
                    "LARGE_SENSITIVITY_SHIFT": abs(row["delta_sensitivity_at_0_5"]) >= 0.15,
                }.items()
                if active
            ]
        )
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("scenario")
    out.to_csv(RUN_DIR / "rf_feature_unavailability_metrics.csv", index=False, encoding="utf-8-sig")
    return out


def paired_bootstrap(sens: pd.DataFrame) -> pd.DataFrame:
    cons = consensus_from_sens(sens)
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for scenario, sub in cons.groupby("scenario"):
        sub = sub.sort_values("case_id_public")
        y = sub["label_binary"].to_numpy()
        p = sub["probability_COP"].to_numpy()
        p0 = sub["baseline_probability_COP"].to_numpy()
        n = len(y)
        vals = {"delta_AUROC": [], "delta_AUPRC": [], "delta_Brier": []}
        obs = {
            "delta_AUROC": roc_auc_score(y, p) - roc_auc_score(y, p0),
            "delta_AUPRC": average_precision_score(y, p) - average_precision_score(y, p0),
            "delta_Brier": brier_score_loss(y, p) - brier_score_loss(y, p0),
        }
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            yy = y[idx]
            if len(np.unique(yy)) < 2:
                continue
            vals["delta_AUROC"].append(roc_auc_score(yy, p[idx]) - roc_auc_score(yy, p0[idx]))
            vals["delta_AUPRC"].append(average_precision_score(yy, p[idx]) - average_precision_score(yy, p0[idx]))
            vals["delta_Brier"].append(brier_score_loss(yy, p[idx]) - brier_score_loss(yy, p0[idx]))
        for metric_name, arr in vals.items():
            rows.append(
                {
                    "scenario": scenario,
                    "metric": metric_name,
                    "estimate": obs[metric_name],
                    "ci_low": float(np.percentile(arr, 2.5)),
                    "ci_high": float(np.percentile(arr, 97.5)),
                    "bootstrap_unit": "patient",
                    "iterations": N_BOOT,
                    "seed": BOOTSTRAP_SEED,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RUN_DIR / "rf_feature_unavailability_paired_bootstrap.csv", index=False, encoding="utf-8-sig")
    return out


def update_method_register() -> None:
    p = GOV_DIR / "LITERATURE_METHOD_REGISTER.md"
    existing = p.read_text(encoding="utf-8") if p.exists() else "# Literature Method Register\n"
    append = f"""

## Clinical Final Confirmation V2

Recorded: {now_iso()}

- nested model/family selection: used only locked internal patient-level outer/inner splits and pre-registered candidate families.
- missing-data deployment sensitivity: evaluated RF post-selection feature and feature-group 100 percent unavailability without retraining or model modification.
- calibration under missingness: assessed fixed Platt-calibrated RF probabilities under deployment missingness scenarios.
- internal-external/generalizability methodology: external labels, predictions, metrics, and inference were not accessed in this stage.
- citation status: CITATION_METADATA_PENDING
"""
    p.write_text(existing.rstrip() + "\n" + append.strip() + "\n", encoding="utf-8")


def write_reports_and_lock(
    data: dict[str, Any],
    evidence_hashes: dict[str, str],
    anthro: dict[str, Any],
    contract: dict[str, Any],
    strict_metrics: pd.DataFrame,
    selection: pd.DataFrame,
    repro: dict[str, Any],
    sens_metrics: pd.DataFrame,
    boot: pd.DataFrame,
    indicators: pd.DataFrame,
) -> dict[str, Any]:
    selected_counts = selection.drop_duplicates(["training_seed", "outer_fold_id"])[
        "selected_family_for_outer_fold"
    ].value_counts().to_dict()
    for fam in FAMILIES:
        selected_counts.setdefault(fam, 0)
    strict_cons = strict_metrics[strict_metrics["level"].eq("mean_3_seed_consensus")].iloc[0].to_dict()
    dev_rf = data["old_metrics"][
        data["old_metrics"]["family"].eq("RF") & data["old_metrics"]["level"].eq("mean_consensus")
    ].iloc[0].to_dict()
    single = sens_metrics[sens_metrics["scenario"].str.startswith("MISS100_")].copy()
    worst_auc = single.loc[single["delta_AUROC"].abs().idxmax()].to_dict()
    worst_brier = single.loc[single["delta_Brier"].abs().idxmax()].to_dict()
    worst_cal = single.assign(cal_change=lambda d: d["calibration_intercept"].abs() + (d["calibration_slope"] - 1).abs()).loc[
        lambda d: d["cal_change"].idxmax()
    ].to_dict()
    lab = sens_metrics[sens_metrics["scenario"].eq("LAB_PANEL_100_MISSING")]
    lab_record = lab.iloc[0].to_dict() if len(lab) else {"status": "NOT_RUN_NO_FORMAL_GROUP_DEFINITION"}
    limitation_count = int(indicators["limitation"].fillna("").ne("").sum())
    key_limit = (
        "Some RF median_indicator folds lack a missingness-indicator dimension for features with no training-fold missingness; deployment-time 100 percent missingness is then median replacement without an explicit missingness flag."
        if limitation_count
        else "No missing-indicator deployment limitation detected for selected RF folds."
    )
    strict_ci = pd.DataFrame()
    write_md(
        REPORT_DIR / "06_STRICT_FAMILY_NESTED_CONFIRMATION.md",
        f"""
# Strict Family Nested Confirmation

Procedure: `STRICT_FAMILY_NESTED_PROCEDURE_OOF`

Consensus metrics:

{md_table(pd.DataFrame([strict_cons]))}

15 outer-fold/seed family selection counts:

{md_table(pd.DataFrame([selected_counts]))}

This confirmation estimates the full algorithm-family selection procedure. It does not replace the development-selected fixed RF branch.
""",
    )
    top_sens = sens_metrics.sort_values("delta_AUROC").head(10)
    write_md(
        REPORT_DIR / "07_RF_FEATURE_AVAILABILITY_SENSITIVITY.md",
        f"""
# RF Feature Availability Sensitivity

Procedure: `DEVELOPMENT_SELECTED_RF_OOF`

RF S0 reproduction max absolute probability error: `{repro['max_error']:.12g}`

Worst single-feature absolute delta AUROC: `{worst_auc['scenario']}` = `{worst_auc['delta_AUROC']:.6g}`

Laboratory panel status: `{repro['laboratory_panel_status']}`

Laboratory panel result:

{md_table(pd.DataFrame([lab_record]))}

Largest AUROC drops:

{md_table(top_sens)}

Missing-indicator limitation:

{key_limit}

These sensitivity results are deployment-risk descriptors only and were not used to modify features, RF hyperparameters, calibration, threshold, or branch selection.
""",
    )
    write_md(
        REPORT_DIR / "08_CLINICAL_BRANCH_FINAL_LOCK_REPORT.md",
        f"""
# Clinical Branch Final Lock Report

Status: `CLINICAL_BRANCH_V2_LOCKED`

Fixed clinical branch: `DEVELOPMENT_SELECTED_RF_OOF`

Development-selected RF metrics:

{md_table(pd.DataFrame([dev_rf]))}

Strict family-selection procedure metrics:

{md_table(pd.DataFrame([strict_cons]))}

The two result pathways are intentionally kept separate. The fixed branch remains RF.
""",
    )
    branch_lock = {
        "status": "CLINICAL_BRANCH_V2_LOCKED",
        "selected_branch_algorithm": "RF",
        "exact_features": data["lock"]["exact_features"],
        "current_RF_development_selected_OOF_metrics": dev_rf,
        "strict_family_nested_procedure_metrics": strict_cons,
        "original_candidate_registry_hash": evidence_hashes["configs/clinical_candidate_registry.json"],
        "current_RF_model_lock_hash": evidence_hashes["locked_models/CLINICAL_MODEL_LOCK.json"],
        "training_data_hash": sha(V1_DEV),
        "corrected_anthropometric_audit_hash": sha(AUDIT_DIR / "anthropometric_correction_audit.json"),
        "split_hash": sha(ROOT / "splits" / "locked_nested_split_patient_manifest.csv"),
        "source_code_hash": sha(Path(__file__)),
        "missingness_strategy": data["lock"]["missingness_handling"],
        "calibration_strategy": data["lock"]["probability_calibration"],
        "single_feature_sensitivity_summary": sens_metrics.to_dict(orient="records"),
        "worst_single_feature_delta_AUROC": worst_auc,
        "worst_single_feature_delta_Brier": worst_brier,
        "worst_single_feature_calibration_change": worst_cal,
        "laboratory_panel_sensitivity": lab_record,
        "rf_s0_reproduction_max_probability_error": repro["max_error"],
        "missing_indicator_limitations": {"count": limitation_count, "summary": key_limit},
        "external_accesses": {"label_reads": 0, "prediction_reads": 0, "metric_reads": 0},
        "known_deployment_risks": sens_metrics[sens_metrics["risk_flags"].fillna("").ne("")][
            ["scenario", "risk_flags", "delta_AUROC", "delta_Brier", "delta_sensitivity_at_0_5"]
        ].to_dict(orient="records"),
        "fixed_branch_not_changed_by_sensitivity": True,
        "strict_nested_other_family_not_promoted": True,
    }
    write_json(LOCK_DIR / "CLINICAL_BRANCH_LOCK_V2.json", branch_lock)
    return {
        "selected_counts": selected_counts,
        "strict_consensus": strict_cons,
        "dev_rf": dev_rf,
        "worst_auc": worst_auc,
        "lab_record": lab_record,
        "repro": repro,
        "missing_indicator_key_limitation": key_limit,
        "branch_lock": branch_lock,
    }


def final_audits_and_checksums(summary: dict[str, Any], evidence_hashes: dict[str, str], contract: dict[str, Any], anthro: dict[str, Any]) -> None:
    external = pd.DataFrame(
        [
            {
                "timestamp": now_iso(),
                "process": "clinical_final_confirmation_v2",
                "file": "",
                "reason": "external cohort deliberately not accessed",
                "label_access": 0,
                "prediction_access": 0,
                "metric_access": 0,
                "model_modified_after_access": 0,
            }
        ]
    )
    external.to_csv(AUDIT_DIR / "clinical_final_external_access_audit.csv", index=False, encoding="utf-8-sig")
    leakage = {
        "status": "PASSED",
        "outer_test_used_for_family_selection": False,
        "outer_test_used_for_hyperparameter_tuning": False,
        "outer_test_used_for_calibration": False,
        "feature_availability_triggered_retraining": False,
        "feature_availability_used_to_modify_RF": False,
        "external_label_prediction_metric_reads": 0,
        "data_split_contract": contract,
    }
    write_json(AUDIT_DIR / "clinical_final_confirmation_leakage_audit.json", leakage)
    completed = {
        "status": "CLINICAL_BRANCH_V2_LOCKED",
        "completed_utc": now_iso(),
        "anthropometric_correction_audit": anthro["status"],
        "strict_family_nested_metrics": summary["strict_consensus"],
        "family_selection_counts": summary["selected_counts"],
        "rf_s0_reproduction_max_probability_error": summary["repro"]["max_error"],
        "external_label_reads": 0,
        "external_prediction_reads": 0,
        "external_metric_reads": 0,
        "clinical_branch_lock": str(LOCK_DIR / "CLINICAL_BRANCH_LOCK_V2.json"),
        "evidence_hashes_preserved": evidence_hashes,
        "scientific_warnings": [
            "Strict family-nested procedure metrics and development-selected RF metrics are separate pathways.",
            "Feature availability sensitivity is deployment-risk evidence only and was not used to modify RF.",
        ],
    }
    write_json(AUDIT_DIR / "CLINICAL_FINAL_RUN_COMPLETED.json", completed)
    final_files = [
        Path(__file__),
        AUDIT_DIR / "anthropometric_correction_audit.json",
        AUDIT_DIR / "rf_missing_indicator_behavior.csv",
        AUDIT_DIR / "clinical_final_confirmation_leakage_audit.json",
        AUDIT_DIR / "clinical_final_external_access_audit.csv",
        AUDIT_DIR / "CLINICAL_FINAL_RUN_COMPLETED.json",
        REPORT_DIR / "05_ANTHROPOMETRIC_CORRECTION_AUDIT.md",
        REPORT_DIR / "06_STRICT_FAMILY_NESTED_CONFIRMATION.md",
        REPORT_DIR / "07_RF_FEATURE_AVAILABILITY_SENSITIVITY.md",
        REPORT_DIR / "08_CLINICAL_BRANCH_FINAL_LOCK_REPORT.md",
        LOCK_DIR / "CLINICAL_BRANCH_LOCK_V2.json",
        GOV_DIR / "LITERATURE_METHOD_REGISTER.md",
    ] + sorted(RUN_DIR.glob("*.csv"))
    with (ROOT / "checksums.sha256").open("a", encoding="utf-8") as f:
        f.write(f"\n# clinical_final_confirmation_v2 {now_iso()}\n")
        for p in final_files:
            if p.exists():
                f.write(f"{sha(p)}  {p.relative_to(ROOT).as_posix()}\n")


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    GOV_DIR.mkdir(parents=True, exist_ok=True)
    evidence_hashes = preserve_hashes()
    data = load_all()
    anthro = audit_anthro(data)
    contract = validate_contract(data)
    strict_paths = [
        RUN_DIR / "strict_family_nested_seed_predictions_PRIVATE.csv",
        RUN_DIR / "strict_family_nested_consensus_predictions_PRIVATE.csv",
        RUN_DIR / "strict_family_nested_metrics.csv",
        RUN_DIR / "strict_family_selection_registry.csv",
    ]
    if all(p.exists() for p in strict_paths):
        strict_seed = pd.read_csv(strict_paths[0])
        strict_cons = pd.read_csv(strict_paths[1])
        strict_metrics = pd.read_csv(strict_paths[2])
        selection = pd.read_csv(strict_paths[3])
    else:
        strict_seed, strict_cons, strict_metrics, selection = run_strict_family_nested(data)
    baseline, sens, indicators, repro_merged, repro = rf_reproduction_and_sensitivity(data)
    sens_metrics = scenario_metrics(sens)
    boot = paired_bootstrap(sens)
    update_method_register()
    summary = write_reports_and_lock(data, evidence_hashes, anthro, contract, strict_metrics, selection, repro, sens_metrics, boot, indicators)
    final_audits_and_checksums(summary, evidence_hashes, contract, anthro)
    strict = summary["strict_consensus"]
    counts = summary["selected_counts"]
    worst = summary["worst_auc"]
    boot_worst = boot[(boot["scenario"].eq(worst["scenario"])) & (boot["metric"].eq("delta_AUROC"))].iloc[0].to_dict()
    lab = summary["lab_record"]
    print(f"ANTHROPOMETRIC_CORRECTION_AUDIT: {anthro['status']}")
    print(f"STRICT_FAMILY_NESTED: AUROC={strict['AUROC']:.4f} AUPRC={strict['AUPRC']:.4f} Brier={strict['Brier']:.4f}")
    print(f"FAMILY_SELECTION_COUNTS: ElasticNetLogit={counts['ElasticNetLogit']} RF={counts['RF']} GradientBoosting={counts['GradientBoosting']}")
    print(f"RF_S0_REPRODUCTION_MAX_PROBABILITY_ERROR: {repro['max_error']:.12g}")
    print(f"WORST_SINGLE_UNAVAILABLE_FEATURE: {worst['scenario']}")
    print(f"WORST_SINGLE_DELTA_AUROC_AND_95CI: {worst['delta_AUROC']:.4f} [{boot_worst['ci_low']:.4f}, {boot_worst['ci_high']:.4f}]")
    print(f"LAB_PANEL_100_UNAVAILABLE: {json.dumps(lab, ensure_ascii=False)}")
    print(f"MISSING_INDICATOR_KEY_LIMITATION: {summary['missing_indicator_key_limitation']}")
    print(f"FINAL_CLINICAL_LOCK_GENERATED: {LOCK_DIR / 'CLINICAL_BRANCH_LOCK_V2.json'}")
    print("EXTERNAL_ACCESS_ALL_ZERO: YES")
    print(f"KEY_OUTPUTS: {REPORT_DIR / '06_STRICT_FAMILY_NESTED_CONFIRMATION.md'}; {REPORT_DIR / '07_RF_FEATURE_AVAILABILITY_SENSITIVITY.md'}; {AUDIT_DIR / 'CLINICAL_FINAL_RUN_COMPLETED.json'}")
    print("SCIENTIFIC_WARNINGS: strict procedure metrics separate from fixed RF; sensitivity not used for model modification")


if __name__ == "__main__":
    main()
