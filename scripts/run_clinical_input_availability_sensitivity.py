from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cap_cop.clinical import full_structured_input_strategy_experiment as full_experiment
from cap_cop.data.release_inputs import DATA_ROOT, load_clinical_cohort


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "clinical_input_availability_scenarios.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "clinical_input_availability_sensitivity_20261013"
INTERNAL_IMAGE = (
    PROJECT_ROOT
    / "outputs"
    / "final_public_multimodal_replay_20261012"
    / "primary_lesion_radiomics_internal_canonical_replay.csv"
)
EXTERNAL_IMAGE = (
    PROJECT_ROOT
    / "outputs"
    / "final_public_multimodal_replay_20261012"
    / "primary_lesion_radiomics_external_canonical_replay.csv"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def load_cohort(cohort: str) -> pd.DataFrame:
    combined = load_clinical_cohort(cohort)
    expected = 375 if cohort == "internal" else 33
    expected_positive = 125 if cohort == "internal" else 14
    if len(combined) != expected or combined["public_subject_id"].nunique() != expected:
        raise RuntimeError(f"Unexpected {cohort} cohort size or duplicate public IDs")
    if int(combined["label_binary"].sum()) != expected_positive:
        raise RuntimeError(f"Unexpected {cohort} label distribution")
    return combined


def load_image_predictions(path: Path, expected_ids: set[str], cohort: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"public_subject_id": str})
    required = {"public_subject_id", "probability_COP_image"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{cohort} image file missing columns: {sorted(missing)}")
    ids = set(frame["public_subject_id"])
    if frame["public_subject_id"].isna().any() or frame["public_subject_id"].duplicated().any():
        raise RuntimeError(f"{cohort} image predictions contain empty or duplicate IDs")
    if ids != expected_ids:
        raise RuntimeError(f"{cohort} image prediction ID set does not match clinical cohort")
    columns = ["public_subject_id", "probability_COP_image"]
    for optional in ("clinical_weight", "image_weight"):
        if optional in frame.columns:
            columns.append(optional)
    return frame[columns].copy()


def apply_input_scenario(frame: pd.DataFrame, scenario: dict[str, Any], features: list[str], seed: int) -> pd.DataFrame:
    output = frame.copy()
    scenario_type = scenario["type"]
    if scenario_type == "clean" or scenario_type == "feature_subset":
        return output
    if scenario_type == "fixed_missing":
        for feature in scenario.get("missing_features", []):
            output[feature] = np.nan
        return output
    if scenario_type == "random_missing":
        rng = np.random.default_rng(seed)
        rate = float(scenario["rate"])
        for feature in features:
            output.loc[rng.random(len(output)) < rate, feature] = np.nan
        return output
    raise ValueError(f"Unknown scenario type: {scenario_type}")


def metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(labels)),
        "AUROC": float(roc_auc_score(labels, probability)),
        "AUPRC": float(average_precision_score(labels, probability)),
        "Brier": float(brier_score_loss(labels, probability)),
        "LogLoss": float(log_loss(labels, probability, labels=[0, 1])),
        "Sensitivity": float(recall_score(labels, prediction, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "Accuracy": float(accuracy_score(labels, prediction)),
        "F1": float(f1_score(labels, prediction, zero_division=0)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def run_internal_oof(
    frame: pd.DataFrame,
    labels: np.ndarray,
    features: list[str],
    scenario: dict[str, Any],
    seeds: list[int],
) -> np.ndarray:
    seed_predictions = []
    for seed in seeds:
        masked = apply_input_scenario(frame, scenario, features, seed)
        seed_predictions.append(
            full_experiment.cross_fitted(
                masked,
                masked,
                labels,
                features,
                "random_forest",
                "balanced_bootstrap",
                "normal",
                seed,
            )
        )
    return np.mean(np.vstack(seed_predictions), axis=0)


def run_external_inference(
    internal: pd.DataFrame,
    external: pd.DataFrame,
    labels: np.ndarray,
    features: list[str],
    scenario: dict[str, Any],
    seeds: list[int],
) -> np.ndarray:
    # Keep the historical Model A external adaptation procedure: each seed
    # fits on an internal training frame masked from the unlabeled external
    # missingness pattern, then predicts the untouched external input frame.
    # The scenario is applied before this procedure so only the clinical
    # input contract changes.
    masked_internal = apply_input_scenario(internal, scenario, features, seeds[0])
    masked_external = apply_input_scenario(external, scenario, features, seeds[0] + 100000)
    return full_experiment.external_predictions(
        masked_internal,
        masked_external,
        labels,
        features,
        "random_forest",
        "balanced_bootstrap",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run clinical-input availability sensitivity scenarios with the Model A image branch fixed.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("clinical_input_availability_sensitivity")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    full_features = list(config["clinical_features"])
    core_features = list(config["core_features"])
    feature_sets = {
        "full_clinical_input": full_features,
        "core_clinical_input": core_features,
    }
    seeds = [int(value) for value in config["seeds"]]
    internal = load_cohort("internal")
    external = load_cohort("external")
    internal_image = load_image_predictions(INTERNAL_IMAGE, set(internal["public_subject_id"]), "internal")
    external_image = load_image_predictions(EXTERNAL_IMAGE, set(external["public_subject_id"]), "external")
    internal = internal.merge(internal_image, on="public_subject_id", validate="one_to_one")
    external = external.merge(external_image, on="public_subject_id", validate="one_to_one")
    labels_internal = internal["label_binary"].to_numpy(dtype=int)
    labels_external = external["label_binary"].to_numpy(dtype=int)

    logger.info("Internal cohort: %d subjects, %d COP", len(internal), int(labels_internal.sum()))
    logger.info("External cohort: %d subjects, %d COP", len(external), int(labels_external.sum()))
    logger.info("Fixed image branch: %s", config["fusion"]["image_probability_source"])
    logger.info("Internal fusion weights: %s", config["fusion"]["internal_weight_source"])
    logger.info("External fusion weights: clinical=%.2f image=%.2f threshold=%.2f", config["fusion"]["external_clinical_weight"], config["fusion"]["external_imaging_weight"], config["fusion"]["threshold"])

    scenario_registry = []
    internal_rows = []
    external_rows = []
    internal_metrics = []
    external_metrics = []
    for scenario in config["scenarios"]:
        scenario = dict(scenario)
        feature_key = scenario["features"]
        features = feature_sets[feature_key]
        logger.info("Running scenario=%s type=%s features=%d", scenario["name"], scenario["type"], len(features))
        internal_probability = run_internal_oof(internal, labels_internal, features, scenario, seeds)
        external_probability = run_external_inference(internal, external, labels_internal, features, scenario, seeds)
        internal_clinical_weight = internal["clinical_weight"].to_numpy(float)
        internal_image_weight = internal["image_weight"].to_numpy(float)
        external_clinical_weight = float(config["fusion"]["external_clinical_weight"])
        external_image_weight = float(config["fusion"]["external_imaging_weight"])
        internal_fusion = (
            internal_clinical_weight * internal_probability
            + internal_image_weight * internal["probability_COP_image"].to_numpy(float)
        )
        external_fusion = (
            external_clinical_weight * external_probability
            + external_image_weight * external["probability_COP_image"].to_numpy(float)
        )
        internal_metrics.append({"scenario": scenario["name"], "cohort": "Internal375", **metrics(labels_internal, internal_fusion)})
        external_metrics.append({"scenario": scenario["name"], "cohort": "External33", **metrics(labels_external, external_fusion)})
        internal_rows.append(pd.DataFrame({
            "scenario": scenario["name"], "public_subject_id": internal["public_subject_id"], "label_binary": labels_internal,
            "probability_COP_clinical": internal_probability, "probability_COP_image": internal["probability_COP_image"],
            "clinical_weight": internal_clinical_weight, "image_weight": internal_image_weight,
            "probability_COP_fusion": internal_fusion, "prediction_0_5": (internal_fusion >= 0.5).astype(int),
        }))
        external_rows.append(pd.DataFrame({
            "scenario": scenario["name"], "public_subject_id": external["public_subject_id"], "label_binary": labels_external,
            "probability_COP_clinical": external_probability, "probability_COP_image": external["probability_COP_image"],
            "clinical_weight": external_clinical_weight, "image_weight": external_image_weight,
            "probability_COP_fusion": external_fusion, "prediction_0_5": (external_fusion >= 0.5).astype(int),
        }))
        scenario_registry.append({
            "scenario": scenario["name"], "type": scenario["type"], "feature_set": feature_key,
            "feature_count": len(features), "features": json.dumps(features),
            "missing_features": json.dumps(scenario.get("missing_features", [])),
            "random_missing_rate": scenario.get("rate", ""), "image_probability_reused": True,
            "external_labels_used_for_training_or_selection": False,
            "internal_fusion_weight_source": config["fusion"]["internal_weight_source"],
            "external_fusion_weights": f"{external_clinical_weight:.2f}/{external_image_weight:.2f}",
        })

    pd.DataFrame(scenario_registry).to_csv(output_dir / "scenario_registry.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(internal_metrics).to_csv(output_dir / "internal_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(external_metrics).to_csv(output_dir / "external_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(internal_rows, ignore_index=True).to_csv(output_dir / "internal_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat(external_rows, ignore_index=True).to_csv(output_dir / "external_predictions.csv", index=False, encoding="utf-8-sig")

    script_path = Path(__file__).resolve()
    metadata = {
        "analysis_name": config["analysis_name"],
        "scientific_status": config["status"],
        "fixed_image_input": "patient-level Model A lesion-ROI radiomics probabilities reused without modification",
        "clinical_input_only_change": True,
        "clinical_model_algorithm": config["clinical_model"],
        "fusion": config["fusion"],
        "internal_cohort": {"n": len(internal), "cop": int(labels_internal.sum()), "id_set_sha256": hashlib.sha256("\n".join(sorted(internal["public_subject_id"])).encode()).hexdigest()},
        "external_cohort": {"n": len(external), "cop": int(labels_external.sum()), "id_set_sha256": hashlib.sha256("\n".join(sorted(external["public_subject_id"])).encode()).hexdigest()},
        "input_files": {
            "config": {"path": str(args.config.resolve().relative_to(PROJECT_ROOT)), "sha256": sha256_file(args.config.resolve())},
            "script": {"path": str(script_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(script_path)},
            "internal_image_predictions": {"path": str(INTERNAL_IMAGE.relative_to(PROJECT_ROOT)), "sha256": sha256_file(INTERNAL_IMAGE)},
            "external_image_predictions": {"path": str(EXTERNAL_IMAGE.relative_to(PROJECT_ROOT)), "sha256": sha256_file(EXTERNAL_IMAGE)},
            "internal_clinical_cap": {"path": "data/deidentified_release_candidate/internal/CAP/clinical/clinical_features.csv", "sha256": sha256_file(DATA_ROOT / "internal/CAP/clinical/clinical_features.csv")},
            "internal_clinical_cop": {"path": "data/deidentified_release_candidate/internal/COP/clinical/clinical_features.csv", "sha256": sha256_file(DATA_ROOT / "internal/COP/clinical/clinical_features.csv")},
            "external_clinical_cap": {"path": "data/deidentified_release_candidate/external/CAP/clinical/clinical_features.csv", "sha256": sha256_file(DATA_ROOT / "external/CAP/clinical/clinical_features.csv")},
            "external_clinical_cop": {"path": "data/deidentified_release_candidate/external/COP/clinical/clinical_features.csv", "sha256": sha256_file(DATA_ROOT / "external/COP/clinical/clinical_features.csv")},
        },
        "external_labels": "read only after predictions for evaluation; not used for training, feature selection, threshold selection, or scenario selection",
        "threshold": 0.5,
        "external_inference_procedure": "historical Model A external_predictions adaptation using unlabeled external missingness patterns; no external labels used",
        "seeds": seeds,
        "outer_folds": int(config["outer_folds"]),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    write_json(output_dir / "analysis_metadata.json", metadata)
    logger.info("Completed %d scenarios", len(scenario_registry))
    logger.info("Output directory: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
