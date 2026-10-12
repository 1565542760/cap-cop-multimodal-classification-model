from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import SimpleITK as sitk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Windows worker processes must use the interpreter that launched this public
# workspace. Without this explicit setting, a venv created from a private
# conda environment can inherit that private base executable.
mp.set_executable(sys.executable)

from cap_cop.pipelines import lesion_radiomics_replay_pipeline as locked
from cap_cop.imaging.segmentation.preprocess import preprocess_dicom_for_segmentation


DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "full_reproduction_20261010" / "lesion_radiomics"
SEGMENTATION_AUDIT = (
    PROJECT_ROOT
    / "outputs"
    / "full_reproduction_20261010"
    / "segmentation"
    / "full_cohort_segmentation_replay_audit.csv"
)
CROSSWALK = PROJECT_ROOT / "data" / "public_subject_model_crosswalk.csv"
ACQUISITION_METADATA = PROJECT_ROOT / "models" / "lesion_radiomics" / "internal_acquisition_metadata.csv"
LOCKED_INTERNAL_BASELINE = PROJECT_ROOT / "models" / "lesion_radiomics" / "internal_baseline_pyradiomics_features.csv"
LOCKED_INTERNAL_OOF = PROJECT_ROOT / "models" / "lesion_radiomics" / "internal_oof_predictions.csv"
LOCKED_EXTERNAL_FEATURES = PROJECT_ROOT / "models" / "lesion_radiomics" / "external_label_blind_pyradiomics_features.csv"
LOCKED_EXTERNAL_PREDICTIONS = PROJECT_ROOT / "models" / "lesion_radiomics" / "external_label_blind_predictions.csv"
LOCKED_FINAL_MODEL = PROJECT_ROOT / "models" / "lesion_radiomics" / "final_random_forest.joblib"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def configure_locked_output(output_dir: Path) -> None:
    locked.DIRS = {
        "protocol": output_dir / "00_protocol",
        "env": output_dir / "01_environment",
        "audit": output_dir / "02_input_audit",
        "extract": output_dir / "03_feature_extraction",
        "robust": output_dir / "04_perturbation_robustness",
        "cv": output_dir / "05_nested_cv",
        "final": output_dir / "06_final_model",
        "external_blind": output_dir / "07_external_label_blind",
        "external_results": output_dir / "08_external_results",
        "mechanism": output_dir / "09_mechanism",
        "reports": output_dir / "10_reports",
        "locks": output_dir / "11_locks",
    }
    for directory in locked.DIRS.values():
        directory.mkdir(parents=True, exist_ok=True)


def load_contract() -> pd.DataFrame:
    crosswalk = pd.read_csv(CROSSWALK, dtype=str).fillna("")
    seg = pd.read_csv(SEGMENTATION_AUDIT, dtype=str).fillna("")
    merged = crosswalk.merge(
        seg[["public_subject_id", "outer_fold", "dicom_dir", "mask_path", "metadata_path", "status"]],
        on="public_subject_id",
        how="left",
        validate="one_to_one",
    )
    if len(merged) != 408 or merged["public_subject_id"].nunique() != 408:
        raise RuntimeError("Expected 408 unique subjects in radiomics input contract")
    if merged[["dicom_dir", "mask_path", "metadata_path"]].eq("").any().any():
        raise RuntimeError("Missing DICOM, mask, or geometry metadata in radiomics input contract")
    if not merged["status"].eq("PASS").all():
        raise RuntimeError("Segmentation replay must pass before radiomics replay")
    merged["label_binary"] = merged["label_binary"].astype(int)
    merged["outer_fold"] = pd.to_numeric(merged["outer_fold"], errors="coerce")
    return merged


def extract_subject(task: dict[str, Any]) -> list[dict[str, Any]]:
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    public_id = str(task["public_subject_id"])
    model_id = str(task["segmentation_model_id"])
    image_id = str(task["lesion_imaging_model_id"])
    metadata = json.loads(Path(task["metadata_path"]).read_text(encoding="utf-8"))
    mask = locked.mask_from_npy(Path(task["mask_path"]), metadata)
    if task["cohort"] == "external":
        preprocessed = preprocess_dicom_for_segmentation(Path(task["dicom_dir"]))
        image = sitk.GetImageFromArray(preprocessed.array_zyx.astype(np.float32, copy=False))
        image.SetSpacing(preprocessed.spacing_xyz)
        image.SetOrigin(preprocessed.origin_xyz)
        image.SetDirection(preprocessed.direction)
    else:
        image = locked.load_dicom_series(Path(task["dicom_dir"]))
    scenarios = task["scenarios"]
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        record: dict[str, Any] = {
            "public_subject_id": public_id,
            "neutral_patient_id": model_id,
            "source_neutral_imaging_id": image_id,
            "outer_fold": task["outer_fold"],
            "perturbation": scenario,
            "status": "FAIL",
            "error": "",
        }
        try:
            image_variant, mask_variant, spacing = locked.perturb(
                image,
                mask,
                scenario,
                locked.patient_seed(model_id, scenario),
            )
            if int(sitk.GetArrayViewFromImage(mask_variant).sum()) <= 0:
                raise RuntimeError("empty perturbed mask")
            values = locked.extract_features(image_variant, mask_variant, spacing)
            record.update(values)
            record["status"] = "PASS" if all(math.isfinite(float(v)) for v in values.values()) else "NONFINITE_FEATURES"
        except Exception as exc:  # the exact failure is part of the replay audit
            record["error"] = repr(exc)
        rows.append(record)
    return rows


def extract_matrix(contract: pd.DataFrame, output_dir: Path, workers: int) -> pd.DataFrame:
    output_path = locked.DIRS["extract"] / "full_cohort_perturbation_pyradiomics_features.csv"
    existing = pd.read_csv(output_path, encoding="utf-8-sig") if output_path.exists() else pd.DataFrame()
    done = set()
    if len(existing) and {"public_subject_id", "perturbation"}.issubset(existing.columns):
        # Historical external extraction used the normalized 2-mm segmentation
        # grid, whereas internal extraction used native CT. Recompute all
        # external rows whenever resuming so an older native-CT adapter cannot
        # be mistaken for a locked-contract result.
        external_ids = set(contract.loc[contract["cohort"].eq("external"), "public_subject_id"].astype(str))
        existing = existing[~existing["public_subject_id"].astype(str).isin(external_ids)].copy()
        done = set(zip(existing["public_subject_id"].astype(str), existing["perturbation"].astype(str)))

    tasks = []
    for row in contract.to_dict("records"):
        scenarios = locked.PERTURBATIONS if row["cohort"] == "internal" else ["BASELINE"]
        pending = [scenario for scenario in scenarios if (row["public_subject_id"], scenario) not in done]
        if pending:
            tasks.append({**row, "scenarios": pending})

    completed_rows: list[dict[str, Any]] = []
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(extract_subject, task): task["public_subject_id"] for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                subject_id = futures[future]
                rows = future.result()
                completed_rows.extend(rows)
                if index % 5 == 0 or index == len(tasks):
                    combined = pd.concat([existing, pd.DataFrame(completed_rows)], ignore_index=True)
                    combined.to_csv(output_path, index=False, encoding="utf-8-sig", lineterminator="\n")
                    existing = combined
                    completed_rows = []
                    print(f"radiomics subjects {index}/{len(tasks)} rows={len(existing)} last={subject_id}", flush=True)

    matrix = pd.read_csv(output_path, encoding="utf-8-sig")
    expected_rows = 375 * len(locked.PERTURBATIONS) + 33
    if len(matrix) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} radiomics rows, found {len(matrix)}")
    failures = matrix.loc[
        ~matrix["status"].eq("PASS"),
        ["public_subject_id", "perturbation", "status", "error"],
    ]
    failures.to_csv(locked.DIRS["audit"] / "radiomics_extraction_failures.csv", index=False)
    external_ids = set(contract.loc[contract["cohort"].eq("external"), "public_subject_id"].astype(str))
    external_failures = failures[failures["public_subject_id"].astype(str).isin(external_ids)]
    if len(external_failures):
        raise RuntimeError(f"External radiomics extraction has {len(external_failures)} non-PASS rows")
    internal_ids = set(contract.loc[contract["cohort"].eq("internal"), "public_subject_id"].astype(str))
    baseline = matrix[
        matrix["public_subject_id"].astype(str).isin(internal_ids)
        & matrix["perturbation"].eq("BASELINE")
    ]
    baseline_failure_rate = float((~baseline["status"].eq("PASS")).mean())
    if len(baseline) != 375 or baseline_failure_rate > 0.05:
        raise RuntimeError(
            f"Historical internal baseline failure contract violated: n={len(baseline)}, "
            f"failure_rate={baseline_failure_rate}"
        )
    return matrix


def numeric_comparison(
    replay: pd.DataFrame,
    reference: pd.DataFrame,
    replay_id: str,
    reference_id: str,
    feature_columns: list[str],
) -> dict[str, Any]:
    joined = replay.merge(reference, left_on=replay_id, right_on=reference_id, suffixes=("_replay", "_locked"), validate="one_to_one")
    deltas = []
    for feature in feature_columns:
        left = pd.to_numeric(joined[f"{feature}_replay"], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(joined[f"{feature}_locked"], errors="coerce").to_numpy(dtype=float)
        deltas.append(np.abs(left - right))
    all_deltas = np.concatenate(deltas) if deltas else np.asarray([], dtype=float)
    return {
        "matched_rows": int(len(joined)),
        "features_compared": int(len(feature_columns)),
        "max_abs_delta": float(np.nanmax(all_deltas)) if len(all_deltas) else None,
        "mean_abs_delta": float(np.nanmean(all_deltas)) if len(all_deltas) else None,
        "finite_values_compared": int(np.isfinite(all_deltas).sum()),
    }


def run_models(contract: pd.DataFrame, matrix: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    internal_contract = contract[contract["cohort"].eq("internal")].copy()
    external_contract = contract[contract["cohort"].eq("external")].copy()
    labels = internal_contract[["segmentation_model_id", "label_binary"]].rename(columns={"segmentation_model_id": "neutral_patient_id"})
    labels = labels.sort_values("neutral_patient_id").reset_index(drop=True)

    # public_subject_id is a release-only join key.  The locked pipeline expects
    # its original metadata contract and must not treat this string as a feature.
    internal_matrix = matrix[matrix["public_subject_id"].isin(internal_contract["public_subject_id"])].copy()
    internal_matrix = internal_matrix.drop(columns=["public_subject_id"])
    locked.write_feature_dictionary(locked.feature_cols(internal_matrix))
    acquisition = pd.read_csv(ACQUISITION_METADATA, encoding="utf-8-sig")
    acquisition_by_id = acquisition.set_index("neutral_patient_id").to_dict("index")

    replay_oof, _, _, replay_metrics = locked.nested_cv(labels, internal_matrix, acquisition_by_id)
    model_path, model_lock, final_features = locked.fit_final(labels, internal_matrix)
    model = joblib.load(model_path)

    external_baseline = matrix[
        matrix["public_subject_id"].isin(external_contract["public_subject_id"])
        & matrix["perturbation"].eq("BASELINE")
    ].copy()
    external_baseline = external_baseline.drop(columns=["public_subject_id"])
    external_frame = external_baseline.set_index("neutral_patient_id")[final_features].apply(pd.to_numeric, errors="coerce")
    external_probability = model["pipeline"].predict_proba(external_frame[final_features])[:, 1]
    external_predictions = external_frame.reset_index()[["neutral_patient_id"]]
    external_predictions["probability_COP"] = external_probability
    external_predictions.to_csv(
        locked.DIRS["external_blind"] / "external33_predictions_label_blind.csv",
        index=False,
        encoding="utf-8-sig",
    )

    locked_baseline = pd.read_csv(LOCKED_INTERNAL_BASELINE, encoding="utf-8-sig")
    replay_baseline = internal_matrix[internal_matrix["perturbation"].eq("BASELINE")].copy()
    feature_columns = [column for column in locked.feature_cols(internal_matrix) if column in locked_baseline.columns]
    baseline_comparison = numeric_comparison(
        replay_baseline,
        locked_baseline,
        "neutral_patient_id",
        "neutral_patient_id",
        feature_columns,
    )

    locked_external_features = pd.read_csv(LOCKED_EXTERNAL_FEATURES, encoding="utf-8-sig")
    external_feature_comparison = numeric_comparison(
        external_baseline,
        locked_external_features,
        "neutral_patient_id",
        "neutral_patient_id",
        [feature for feature in final_features if feature in locked_external_features.columns],
    )

    locked_oof = pd.read_csv(LOCKED_INTERNAL_OOF, encoding="utf-8-sig")
    oof_probability_column = "probability_COP" if "probability_COP" in locked_oof.columns else "prob_COP"
    oof_joined = replay_oof.merge(
        locked_oof[["neutral_patient_id", oof_probability_column]],
        on="neutral_patient_id",
        how="inner",
        suffixes=("_replay", "_locked"),
        validate="one_to_one",
    )
    oof_delta = np.abs(
        oof_joined["probability_COP_replay"].to_numpy(dtype=float)
        - oof_joined[f"{oof_probability_column}_locked"].to_numpy(dtype=float)
    )

    locked_external_predictions = pd.read_csv(LOCKED_EXTERNAL_PREDICTIONS, encoding="utf-8-sig")
    locked_external_probability_column = next(
        column for column in ["probability_COP", "prob_COP", "prob_COP_mean"] if column in locked_external_predictions.columns
    )
    external_joined = external_predictions.merge(
        locked_external_predictions[["neutral_patient_id", locked_external_probability_column]],
        on="neutral_patient_id",
        suffixes=("_replay", "_locked"),
        validate="one_to_one",
    )
    replay_external_probability_column = (
        "probability_COP_replay"
        if locked_external_probability_column == "probability_COP"
        else "probability_COP"
    )
    locked_external_probability_column_after_merge = (
        f"{locked_external_probability_column}_locked"
        if locked_external_probability_column == "probability_COP"
        else locked_external_probability_column
    )
    external_delta = np.abs(
        external_joined[replay_external_probability_column].to_numpy(dtype=float)
        - external_joined[locked_external_probability_column_after_merge].to_numpy(dtype=float)
    )

    summary = {
        "status": "PASS",
        "internal_subjects": 375,
        "external_subjects": 33,
        "candidate_features": len(feature_columns),
        "final_features": final_features,
        "internal_baseline_feature_comparison": baseline_comparison,
        "external_final_feature_comparison": external_feature_comparison,
        "internal_oof_probability_comparison": {
            "matched_rows": int(len(oof_joined)),
            "max_abs_delta": float(oof_delta.max()),
            "mean_abs_delta": float(oof_delta.mean()),
        },
        "external_probability_comparison": {
            "matched_rows": int(len(external_joined)),
            "max_abs_delta": float(external_delta.max()),
            "mean_abs_delta": float(external_delta.mean()),
        },
        "replay_internal_metrics": replay_metrics,
        "replay_model_lock": model_lock,
        "replay_model_sha256": sha256_file(model_path),
        "input_matrix_sha256": sha256_file(locked.DIRS["extract"] / "full_cohort_perturbation_pyradiomics_features.csv"),
    }
    write_json(output_dir / "full_cohort_lesion_radiomics_replay_summary.json", summary)
    return summary


def run_locked_artifact_replay(contract: pd.DataFrame, matrix: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Validate the raw-input feature chain against the original locked model artifacts.

    This path deliberately does not fit, tune, rank, or select a new model.  It
    is the default reproduction path; the refit path above is retained only as
    an explicitly named corrected-mapping analysis.
    """
    internal_contract = contract[contract["cohort"].eq("internal")].copy()
    external_contract = contract[contract["cohort"].eq("external")].copy()
    internal_matrix = matrix[matrix["public_subject_id"].isin(internal_contract["public_subject_id"])].copy()
    external_baseline = matrix[
        matrix["public_subject_id"].isin(external_contract["public_subject_id"])
        & matrix["perturbation"].eq("BASELINE")
    ].copy()
    internal_matrix = internal_matrix.drop(columns=["public_subject_id"])
    external_baseline = external_baseline.drop(columns=["public_subject_id"])

    model_bundle = joblib.load(LOCKED_FINAL_MODEL)
    locked_model_hash = sha256_file(LOCKED_FINAL_MODEL)
    final_features = [str(value) for value in model_bundle["features"]]
    feature_columns = [column for column in locked.feature_cols(internal_matrix) if column in final_features]
    if set(feature_columns) != set(final_features):
        raise RuntimeError("Locked final feature contract is not available in the replay matrix")

    locked_baseline = pd.read_csv(LOCKED_INTERNAL_BASELINE, encoding="utf-8-sig")
    replay_baseline = internal_matrix[internal_matrix["perturbation"].eq("BASELINE")].copy()
    baseline_comparison = numeric_comparison(
        replay_baseline,
        locked_baseline,
        "neutral_patient_id",
        "neutral_patient_id",
        final_features,
    )
    locked_external_features = pd.read_csv(LOCKED_EXTERNAL_FEATURES, encoding="utf-8-sig")
    external_feature_comparison = numeric_comparison(
        external_baseline,
        locked_external_features,
        "neutral_patient_id",
        "neutral_patient_id",
        final_features,
    )

    # External inference uses the original locked final RF only.  No external
    # labels are read in this function.
    external_frame = external_baseline.set_index("neutral_patient_id")[final_features].apply(
        pd.to_numeric, errors="coerce"
    )
    external_probability = model_bundle["pipeline"].predict_proba(external_frame[final_features])[:, 1]
    external_predictions = external_frame.reset_index()[["neutral_patient_id"]].copy()
    external_predictions["probability_COP"] = external_probability
    external_path = locked.DIRS["external_blind"] / "external33_locked_model_inference_replay.csv"
    external_predictions.to_csv(external_path, index=False, encoding="utf-8-sig")

    locked_external_predictions = pd.read_csv(LOCKED_EXTERNAL_PREDICTIONS, encoding="utf-8-sig")
    locked_external_probability_column = next(
        column for column in ["probability_COP", "prob_COP", "prob_COP_mean"]
        if column in locked_external_predictions.columns
    )
    external_joined = external_predictions.merge(
        locked_external_predictions[["neutral_patient_id", locked_external_probability_column]],
        on="neutral_patient_id",
        suffixes=("_replay", "_locked"),
        validate="one_to_one",
    )
    replay_external_probability_column = (
        "probability_COP_replay"
        if locked_external_probability_column == "probability_COP"
        else "probability_COP"
    )
    locked_external_probability_column_after_merge = (
        f"{locked_external_probability_column}_locked"
        if locked_external_probability_column == "probability_COP"
        else locked_external_probability_column
    )
    external_delta = np.abs(
        external_joined[replay_external_probability_column].to_numpy(dtype=float)
        - external_joined[locked_external_probability_column_after_merge].to_numpy(dtype=float)
    )

    # Internal strict OOF is a locked patient-level artifact in the original
    # experiment.  Validate it and report canonical-label differences separately;
    # do not silently retrain it under the corrected identity mapping.
    locked_oof = pd.read_csv(LOCKED_INTERNAL_OOF, encoding="utf-8-sig")
    oof_probability_column = "probability_COP" if "probability_COP" in locked_oof.columns else "prob_COP"
    canonical_labels = internal_contract[["segmentation_model_id", "label_binary"]].rename(
        columns={"segmentation_model_id": "neutral_patient_id"}
    )
    oof_label_check = locked_oof[["neutral_patient_id", "label_binary"]].merge(
        canonical_labels,
        on="neutral_patient_id",
        how="outer",
        suffixes=("_locked", "_canonical"),
        indicator=True,
        validate="one_to_one",
    )
    label_mismatch = oof_label_check[
        oof_label_check["_merge"].eq("both")
        & (oof_label_check["label_binary_locked"] != oof_label_check["label_binary_canonical"])
    ]
    locked_oof_metrics = locked.metrics(
        locked_oof["label_binary"].to_numpy(dtype=int),
        locked_oof[oof_probability_column].to_numpy(dtype=float),
    )

    summary = {
        "status": "PASS" if len(external_joined) == 33 and float(external_delta.max()) <= 1e-12 else "MISMATCH",
        "mode": "locked_artifact_replay",
        "retraining": False,
        "model_selection": False,
        "internal_subjects": int(len(locked_oof)),
        "external_subjects": int(len(external_joined)),
        "final_features": final_features,
        "locked_model_sha256": locked_model_hash,
        "locked_model_params": model_bundle.get("params", {}),
        "internal_baseline_feature_comparison": baseline_comparison,
        "external_final_feature_comparison": external_feature_comparison,
        "external_locked_prediction_comparison": {
            "matched_rows": int(len(external_joined)),
            "max_abs_delta": float(external_delta.max()),
            "mean_abs_delta": float(external_delta.mean()),
        },
        "internal_locked_oof_metrics": locked_oof_metrics,
        "internal_oof_source": str(LOCKED_INTERNAL_OOF),
        "canonical_label_mismatch_count": int(len(label_mismatch)),
        "canonical_label_check_rows": int(len(oof_label_check)),
        "note": "Internal OOF is validated from the original locked artifact; no OOF refit is performed in strict locked mode.",
        "input_matrix_sha256": sha256_file(locked.DIRS["extract"] / "full_cohort_perturbation_pyradiomics_features.csv"),
    }
    write_json(output_dir / "full_cohort_lesion_radiomics_locked_replay_summary.json", summary)
    return summary


def audit_existing_model_outputs(contract: pd.DataFrame, matrix: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Complete the replay audit after model fitting has already succeeded."""
    internal_contract = contract[contract["cohort"].eq("internal")].copy()
    external_contract = contract[contract["cohort"].eq("external")].copy()
    internal_matrix = matrix[matrix["public_subject_id"].isin(internal_contract["public_subject_id"])].copy()
    internal_matrix = internal_matrix.drop(columns=["public_subject_id"])
    external_baseline = matrix[
        matrix["public_subject_id"].isin(external_contract["public_subject_id"])
        & matrix["perturbation"].eq("BASELINE")
    ].copy()
    external_baseline = external_baseline.drop(columns=["public_subject_id"])

    feature_list_path = locked.DIRS["final"] / "lesion_radiomics_replay_FINAL_FEATURE_LIST.csv"
    model_path = locked.DIRS["final"] / "lesion_radiomics_replay_FINAL_MODEL.joblib"
    model_lock_path = locked.DIRS["final"] / "lesion_radiomics_replay_FINAL_MODEL_LOCK.json"
    replay_oof_path = locked.DIRS["cv"] / "lesion_radiomics_replay_INTERNAL375_STRICT_OOF_PREDICTIONS.csv"
    replay_metrics_path = locked.DIRS["cv"] / "lesion_radiomics_replay_INTERNAL375_STRICT_OOF_METRICS.csv"
    external_predictions_path = locked.DIRS["external_blind"] / "external33_predictions_label_blind.csv"
    required = [
        feature_list_path,
        model_path,
        model_lock_path,
        replay_oof_path,
        replay_metrics_path,
        external_predictions_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot audit incomplete model replay outputs: {missing}")

    final_features = pd.read_csv(feature_list_path, encoding="utf-8-sig")["feature"].astype(str).tolist()
    replay_oof = pd.read_csv(replay_oof_path, encoding="utf-8-sig")
    replay_metrics = pd.read_csv(replay_metrics_path, encoding="utf-8-sig").iloc[0].to_dict()
    external_predictions = pd.read_csv(external_predictions_path, encoding="utf-8-sig")
    model_lock = json.loads(model_lock_path.read_text(encoding="utf-8"))

    locked_baseline = pd.read_csv(LOCKED_INTERNAL_BASELINE, encoding="utf-8-sig")
    replay_baseline = internal_matrix[internal_matrix["perturbation"].eq("BASELINE")].copy()
    feature_columns = [column for column in locked.feature_cols(internal_matrix) if column in locked_baseline.columns]
    baseline_comparison = numeric_comparison(
        replay_baseline,
        locked_baseline,
        "neutral_patient_id",
        "neutral_patient_id",
        feature_columns,
    )
    locked_external_features = pd.read_csv(LOCKED_EXTERNAL_FEATURES, encoding="utf-8-sig")
    external_feature_comparison = numeric_comparison(
        external_baseline,
        locked_external_features,
        "neutral_patient_id",
        "neutral_patient_id",
        [feature for feature in final_features if feature in locked_external_features.columns],
    )

    locked_oof = pd.read_csv(LOCKED_INTERNAL_OOF, encoding="utf-8-sig")
    locked_oof_probability = "probability_COP" if "probability_COP" in locked_oof.columns else "prob_COP"
    oof_joined = replay_oof.merge(
        locked_oof[["neutral_patient_id", locked_oof_probability]],
        on="neutral_patient_id",
        suffixes=("_replay", "_locked"),
        validate="one_to_one",
    )
    oof_delta = np.abs(
        oof_joined["probability_COP_replay"].to_numpy(dtype=float)
        - oof_joined[f"{locked_oof_probability}_locked"].to_numpy(dtype=float)
    )

    locked_external = pd.read_csv(LOCKED_EXTERNAL_PREDICTIONS, encoding="utf-8-sig")
    locked_external_probability = next(
        column for column in ["probability_COP", "prob_COP", "prob_COP_mean"] if column in locked_external.columns
    )
    external_joined = external_predictions.merge(
        locked_external[["neutral_patient_id", locked_external_probability]],
        on="neutral_patient_id",
        suffixes=("_replay", "_locked"),
        validate="one_to_one",
    )
    replay_external_probability = (
        "probability_COP_replay" if locked_external_probability == "probability_COP" else "probability_COP"
    )
    locked_external_probability_after_merge = (
        f"{locked_external_probability}_locked"
        if locked_external_probability == "probability_COP"
        else locked_external_probability
    )
    external_delta = np.abs(
        external_joined[replay_external_probability].to_numpy(dtype=float)
        - external_joined[locked_external_probability_after_merge].to_numpy(dtype=float)
    )
    locked_model = joblib.load(LOCKED_FINAL_MODEL)
    locked_model_probability = locked_model["pipeline"].predict_proba(
        external_baseline.set_index("neutral_patient_id")[locked_model["features"]]
    )[:, 1]
    locked_model_ids = external_baseline["neutral_patient_id"].astype(str).to_numpy()
    locked_reference_by_id = locked_external.set_index("neutral_patient_id")[locked_external_probability]
    locked_model_delta = np.abs(
        locked_model_probability
        - locked_reference_by_id.loc[locked_model_ids].to_numpy(dtype=float)
    )
    locked_model_external_replay = external_baseline[["neutral_patient_id"]].copy()
    locked_model_external_replay["probability_COP"] = locked_model_probability
    locked_model_external_replay.to_csv(
        locked.DIRS["external_blind"] / "external33_locked_model_inference_replay.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "status": "PASS",
        "audit_mode": "existing_outputs_after_completed_model_fit",
        "internal_subjects": int(len(oof_joined)),
        "external_subjects": int(len(external_joined)),
        "candidate_features": int(len(feature_columns)),
        "final_features": final_features,
        "internal_baseline_feature_comparison": baseline_comparison,
        "external_final_feature_comparison": external_feature_comparison,
        "internal_oof_probability_comparison": {
            "matched_rows": int(len(oof_joined)),
            "max_abs_delta": float(oof_delta.max()),
            "mean_abs_delta": float(oof_delta.mean()),
        },
        "external_probability_comparison": {
            "matched_rows": int(len(external_joined)),
            "max_abs_delta": float(external_delta.max()),
            "mean_abs_delta": float(external_delta.mean()),
        },
        "locked_model_external_inference_comparison": {
            "matched_rows": int(len(locked_model_delta)),
            "max_abs_delta": float(locked_model_delta.max()),
            "mean_abs_delta": float(locked_model_delta.mean()),
            "locked_model_sha256": sha256_file(LOCKED_FINAL_MODEL),
        },
        "replay_internal_metrics": replay_metrics,
        "replay_model_lock": model_lock,
        "replay_model_sha256": sha256_file(model_path),
        "input_matrix_sha256": sha256_file(locked.DIRS["extract"] / "full_cohort_perturbation_pyradiomics_features.csv"),
    }
    write_json(output_dir / "full_cohort_lesion_radiomics_replay_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay the locked lesion-ROI PyRadiomics and RF chain for Internal375 and External33.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--phase", choices=["extract", "model", "audit-model", "all"], default="all")
    parser.add_argument(
        "--segmentation-audit",
        type=Path,
        default=None,
        help="Segmentation audit CSV; defaults to the sibling segmentation output for --output-dir.",
    )
    parser.add_argument(
        "--model-mode",
        choices=["locked-artifact", "corrected-refit"],
        default="locked-artifact",
        help="Use original locked artifacts by default; corrected-refit is an explicit non-lock analysis.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    global SEGMENTATION_AUDIT
    SEGMENTATION_AUDIT = (
        args.segmentation_audit.resolve()
        if args.segmentation_audit is not None
        else output_dir.parent / "segmentation" / "full_cohort_segmentation_replay_audit.csv"
    )
    configure_locked_output(output_dir)
    contract = load_contract()
    matrix_path = locked.DIRS["extract"] / "full_cohort_perturbation_pyradiomics_features.csv"
    if args.phase in {"extract", "all"}:
        matrix = extract_matrix(contract, output_dir, max(1, args.workers))
    else:
        matrix = pd.read_csv(matrix_path, encoding="utf-8-sig")
    if args.phase in {"model", "all"}:
        if args.model_mode == "locked-artifact":
            summary = run_locked_artifact_replay(contract, matrix, output_dir)
        else:
            summary = run_models(contract, matrix, output_dir)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    elif args.phase == "audit-model":
        # The public default is locked-artifact replay.  The historical audit
        # helper expects artifacts emitted by the separate corrected-refit mode;
        # dispatch locked mode to the non-fitting verifier instead of reporting
        # a misleading missing-file error.
        if args.model_mode == "locked-artifact":
            summary = run_locked_artifact_replay(contract, matrix, output_dir)
            summary["audit_mode"] = "locked_artifact_replay"
        else:
            summary = audit_existing_model_outputs(contract, matrix, output_dir)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
