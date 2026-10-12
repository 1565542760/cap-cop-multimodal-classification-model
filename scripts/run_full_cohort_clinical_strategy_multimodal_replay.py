from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from cap_cop.private_lock_selectors import get_selector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "full_reproduction_20261010"
OUTPUT_DIR = OUTPUT_ROOT / "clinical_strategy_multimodal_comparison"
CROSSWALK = PROJECT_ROOT / "data" / "public_subject_model_crosswalk.csv"
INTERNAL_LOCK = PROJECT_ROOT / "models" / "multimodal_fusion" / "internal_oof_all_pairs_predictions.csv"
EXTERNAL_LOCK = PROJECT_ROOT / "models" / "multimodal_fusion" / "external_all_pairs_label_free_predictions.csv"
IMAGE_SELECTOR = "stable_image"

STRATEGIES = {
    "full_structured": {
        "clinical_selector": "full_structured_clinical",
        "internal": OUTPUT_ROOT / "clinical" / "full_structured_internal_oof_replay.csv",
        "external": OUTPUT_ROOT / "clinical" / "full_structured_external_replay.csv",
    },
    "structural_missingness": {
        "clinical_selector": "structural_missingness_clinical",
        "internal": OUTPUT_ROOT / "clinical" / "structural_missingness_internal_oof_replay.csv",
        "external": OUTPUT_ROOT / "clinical" / "structural_missingness_external_replay.csv",
    },
    "core_clinical": {
        "clinical_selector": "core_clinical",
        "internal": OUTPUT_ROOT / "clinical_input_comparison" / "core_internal_oof_replay_audit.csv",
        "external": OUTPUT_ROOT / "clinical_input_comparison" / "core_external_replay_audit.csv",
    },
    "availability_gated": {
        "clinical_selector": "availability_gated_clinical",
        "internal": OUTPUT_ROOT / "clinical_input_comparison" / "gated_internal_oof_replay_audit.csv",
        "external": OUTPUT_ROOT / "clinical_input_comparison" / "gated_external_replay_audit.csv",
    },
    "alternate_configuration": {
        "clinical_selector": "alternate_clinical",
        "internal": OUTPUT_ROOT / "alternate_clinical_configuration" / "alternate_internal_oof_replay_audit.csv",
        "external": OUTPUT_ROOT / "alternate_clinical_configuration" / "alternate_external_replay_audit.csv",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def score(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    predicted = (probability >= 0.5).astype(int)
    return {
        "AUROC": float(roc_auc_score(labels, probability)),
        "Accuracy": float(accuracy_score(labels, predicted)),
        "Sensitivity": float(recall_score(labels, predicted, zero_division=0)),
        "Specificity": float(recall_score(1 - labels, 1 - predicted, zero_division=0)),
        "F1": float(f1_score(labels, predicted, zero_division=0)),
    }


def image_probability(cohort: str, crosswalk: pd.DataFrame) -> pd.DataFrame:
    image_root = OUTPUT_ROOT / "whole_chest_representations"
    suffix = "internal" if cohort == "internal" else "external"
    candidates = [
        image_root / f"stable_multirepresentation_regenerated_tabular_{suffix}_replay.csv",
        image_root / f"stable_multirepresentation_locked_tabular_{suffix}_replay.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            "No stable multirepresentation replay table found. Expected one of: "
            + ", ".join(str(candidate) for candidate in candidates)
        )
    frame = pd.read_csv(path)
    frame = frame[frame["candidate"].eq("SP6_STABLE_MULTI_REPRESENTATION_RANDOM_FOREST")].copy()
    if cohort == "internal":
        frame = frame[["neutral_imaging_id", "prob_COP"]].rename(
            columns={"neutral_imaging_id": "model_id", "prob_COP": "probability_COP_image"}
        )
        mapping_column = "whole_chest_imaging_model_id"
    else:
        counts = frame.groupby("neutral_external_id")["prob_COP_mean"].nunique(dropna=False)
        if not counts.eq(1).all():
            raise RuntimeError("External SP6 fold rows disagree on patient-level mean probability")
        frame = frame.groupby("neutral_external_id", as_index=False)["prob_COP_mean"].first().rename(
            columns={"neutral_external_id": "model_id", "prob_COP_mean": "probability_COP_image"}
        )
        mapping_column = "external_tensor_id"
    mapping = crosswalk.loc[crosswalk["cohort"].eq(cohort), ["public_subject_id", mapping_column]].rename(
        columns={mapping_column: "model_id"}
    )
    result = frame.merge(mapping, on="model_id", validate="one_to_one")
    return result[["public_subject_id", "probability_COP_image"]]


def replay_strategy(name: str, config: dict[str, object], cohort: str, crosswalk: pd.DataFrame) -> dict[str, object]:
    clinical = pd.read_csv(Path(config[cohort]))[["public_subject_id", "probability_COP"]].rename(
        columns={"probability_COP": "probability_COP_clinical"}
    )
    image = image_probability(cohort, crosswalk)
    components = clinical.merge(image, on="public_subject_id", validate="one_to_one")
    labels = crosswalk.loc[crosswalk["cohort"].eq(cohort), ["public_subject_id", "label_binary", "clinical_model_id"]]
    components = components.merge(labels, on="public_subject_id", validate="one_to_one")

    clinical_selector = str(config["clinical_selector"])
    clinical_id = get_selector(clinical_selector)
    image_id = get_selector(IMAGE_SELECTOR)
    lock_path = INTERNAL_LOCK if cohort == "internal" else EXTERNAL_LOCK
    lock = pd.read_csv(lock_path)
    lock = lock[lock["clinical_id"].eq(clinical_id) & lock["image_id"].eq(image_id)].copy()
    if len(lock) != (375 if cohort == "internal" else 33):
        raise RuntimeError(f"{name}/{cohort}: expected one complete locked pair, found {len(lock)} rows")
    if cohort == "internal":
        lock = lock.rename(columns={"patient_id": "public_subject_id", "p_weighted_fusion": "probability_COP_locked"})
        joined = components.merge(
            lock[["public_subject_id", "label", "selected_outer_weight", "probability_COP_locked"]],
            on="public_subject_id",
            validate="one_to_one",
        )
        joined["clinical_weight"] = joined["selected_outer_weight"].astype(float)
        joined["image_weight"] = 1.0 - joined["clinical_weight"]
        if not joined["label_binary"].astype(int).eq(joined["label"].astype(int)).all():
            raise RuntimeError(f"{name}: internal labels disagree")
    else:
        lock = lock.merge(
            labels[["public_subject_id", "clinical_model_id"]],
            left_on="patient_id",
            right_on="clinical_model_id",
            validate="one_to_one",
        ).rename(columns={"p_weighted_fusion": "probability_COP_locked"})
        joined = components.merge(
            lock[["public_subject_id", "final_weight_clinical", "final_weight_image", "probability_COP_locked"]],
            on="public_subject_id",
            validate="one_to_one",
        )
        joined["clinical_weight"] = joined["final_weight_clinical"].astype(float)
        joined["image_weight"] = joined["final_weight_image"].astype(float)

    joined["probability_COP_replay"] = (
        joined["clinical_weight"] * joined["probability_COP_clinical"]
        + joined["image_weight"] * joined["probability_COP_image"]
    )
    joined["absolute_delta"] = np.abs(joined["probability_COP_replay"] - joined["probability_COP_locked"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{name}_{cohort}_replay_audit.csv"
    joined.to_csv(output_path, index=False, encoding="utf-8-sig")
    return {
        "subjects": int(len(joined)),
        "pair_id": str(lock["pair_id"].iloc[0]),
        "external_weight_clinical": float(lock["final_weight_clinical"].iloc[0]) if cohort == "external" else None,
        "external_weight_image": float(lock["final_weight_image"].iloc[0]) if cohort == "external" else None,
        "internal_unique_fold_weights": sorted(joined["clinical_weight"].unique().astype(float).tolist()) if cohort == "internal" else None,
        "max_abs_delta": float(joined["absolute_delta"].max()),
        "mean_abs_delta": float(joined["absolute_delta"].mean()),
        "replay_metrics": score(joined["label_binary"].to_numpy(dtype=int), joined["probability_COP_replay"].to_numpy(dtype=float)),
        "locked_metrics": score(joined["label_binary"].to_numpy(dtype=int), joined["probability_COP_locked"].to_numpy(dtype=float)),
        "output_sha256": sha256_file(output_path),
    }


def main() -> int:
    crosswalk = pd.read_csv(CROSSWALK, dtype={"label_binary": int})
    summary = {
        "status": "PASS_NUMERICAL_REPLAY_WITH_HISTORICAL_IMAGE_IDENTITY_DEFECT",
        "scope": "Five locked clinical configurations fused with the same SP6 output using each pair's stored internal fold weights and external fixed weight.",
        "warning": "The shared SP6 historical input assembly mismatches lesion-derived tabular rows for 373/375 internal subjects; this is a lock replay, not a valid single-factor causal comparison.",
        "image_selector": IMAGE_SELECTOR,
        "internal_lock_sha256": sha256_file(INTERNAL_LOCK),
        "external_lock_sha256": sha256_file(EXTERNAL_LOCK),
        "strategies": {},
    }
    for name, config in STRATEGIES.items():
        summary["strategies"][name] = {
            cohort: replay_strategy(name, config, cohort, crosswalk) for cohort in ("internal", "external")
        }
    summary_path = OUTPUT_DIR / "clinical_strategy_multimodal_replay_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
