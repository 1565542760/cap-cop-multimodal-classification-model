from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cap_cop.clinical import full_structured_input_strategy_experiment as full_experiment
from cap_cop.clinical import clinical_transportability_protocol as protocol
from cap_cop.clinical.full_structured_model import load_artifact, predict_proba
from cap_cop.clinical.structural_missingness_model import load_locked_artifact
from cap_cop.private_lock_selectors import get_selector


DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "full_reproduction_20261010" / "clinical"


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def load_cohort(cohort: str) -> pd.DataFrame:
    frames = []
    for diagnosis in ("CAP", "COP"):
        path = PROJECT_ROOT / "data" / cohort / diagnosis / "clinical" / "clinical_features.csv"
        frame = pd.read_csv(path)
        if cohort == "internal":
            frame["public_subject_id"] = frame["case_id_public"].astype(str)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    expected = 375 if cohort == "internal" else 33
    if len(combined) != expected or combined["public_subject_id"].nunique() != expected:
        raise RuntimeError(f"Expected {expected} unique {cohort} subjects")
    return combined


def locked_clinical_probabilities(scope: str, selector_name: str) -> pd.DataFrame:
    clinical_id = get_selector(selector_name)
    name = "internal_oof_all_pairs_predictions.csv" if scope == "internal" else "external_all_pairs_label_free_predictions.csv"
    frame = pd.read_csv(PROJECT_ROOT / "models" / "multimodal_fusion" / name)
    id_column = "patient_id"
    subset = frame[frame["clinical_id"].eq(clinical_id)][[id_column, "p_clinical"]].drop_duplicates()
    if subset[id_column].duplicated().any():
        counts = subset.groupby(id_column)["p_clinical"].nunique()
        if (counts > 1).any():
            raise RuntimeError(f"Locked {clinical_id} probabilities disagree across image pairs")
        subset = subset.drop_duplicates(id_column)
    return subset.rename(columns={id_column: "model_patient_id", "p_clinical": "locked_probability"})


def replay_full_structured(internal: pd.DataFrame, external: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    features = full_experiment.FEATURE_SETS["full16"]
    labels = internal["label_binary"].astype(int).to_numpy()
    internal_features = internal[features].copy()
    external_features = external[features].copy()
    seed_oof = [
        full_experiment.cross_fitted(
            internal_features,
            external_features,
            labels,
            features,
            "random_forest",
            "balanced_bootstrap",
            "normal",
            seed,
        )
        for seed in full_experiment.SEEDS
    ]
    oof = np.mean(np.vstack(seed_oof), axis=0)
    external_probability = full_experiment.external_predictions(
        internal_features,
        external_features,
        labels,
        features,
        "random_forest",
        "balanced_bootstrap",
    )

    replay_internal = pd.DataFrame(
        {
            "public_subject_id": internal["public_subject_id"],
            "label_binary": labels,
            "probability_COP": oof,
        }
    )
    replay_external = pd.DataFrame(
        {
            "public_subject_id": external["public_subject_id"],
            "label_binary": external["label_binary"].astype(int),
            "probability_COP": external_probability,
        }
    )
    replay_internal.to_csv(output_dir / "full_structured_internal_oof_replay.csv", index=False)
    replay_external.to_csv(output_dir / "full_structured_external_replay.csv", index=False)

    crosswalk = pd.read_csv(PROJECT_ROOT / "data" / "public_subject_model_crosswalk.csv", dtype=str)
    internal_map = crosswalk[crosswalk["cohort"].eq("internal")][["public_subject_id", "clinical_model_id"]]
    external_map = crosswalk[crosswalk["cohort"].eq("external")][["public_subject_id", "clinical_model_id"]]
    replay_internal = replay_internal.merge(internal_map, on="public_subject_id", validate="one_to_one")
    replay_external = replay_external.merge(external_map, on="public_subject_id", validate="one_to_one")
    locked_internal = locked_clinical_probabilities("internal", "full_structured_clinical")
    locked_external = locked_clinical_probabilities("external", "full_structured_clinical")
    internal_joined = replay_internal.merge(locked_internal, left_on="clinical_model_id", right_on="model_patient_id", validate="one_to_one")
    external_joined = replay_external.merge(locked_external, left_on="clinical_model_id", right_on="model_patient_id", validate="one_to_one")
    internal_delta = np.abs(internal_joined["probability_COP"] - internal_joined["locked_probability"])
    external_delta = np.abs(external_joined["probability_COP"] - external_joined["locked_probability"])

    artifact = load_artifact(PROJECT_ROOT / "models" / "clinical" / "full_structured" / "full_structured_clinical_ensemble.joblib")
    artifact_external = predict_proba(artifact, external)
    artifact_delta = np.abs(artifact_external - external_probability)
    return {
        "internal_subjects": len(replay_internal),
        "external_subjects": len(replay_external),
        "internal_oof_max_abs_delta": float(internal_delta.max()),
        "internal_oof_mean_abs_delta": float(internal_delta.mean()),
        "external_max_abs_delta": float(external_delta.max()),
        "external_mean_abs_delta": float(external_delta.mean()),
        "external_artifact_vs_retrained_max_abs_delta": float(artifact_delta.max()),
        "external_artifact_vs_retrained_mean_abs_delta": float(artifact_delta.mean()),
        "replay_threshold": float(round(protocol.youden_threshold(labels, oof), 6)),
    }


def replay_structural_missingness(
    internal: pd.DataFrame,
    external: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    model_root = (
        PROJECT_ROOT
        / "models"
        / "supplementary_exploratory"
        / "structural_missingness_clinical"
    )
    model_paths = sorted(
        (model_root / "ensemble_models").glob("*.joblib")
    )
    if len(model_paths) != 15:
        raise RuntimeError(f"Expected 15 structural-missingness models, found {len(model_paths)}")

    split_path = model_root / "locked_nested_split_patient_manifest.csv"
    nested = pd.read_csv(split_path)
    required_split_columns = {"training_seed", "outer_fold_id", "case_id_public", "outer_role"}
    if not required_split_columns.issubset(nested.columns):
        raise RuntimeError(f"Locked split is missing columns: {sorted(required_split_columns - set(nested.columns))}")
    internal_by_id = internal.set_index("case_id_public", drop=False)
    seed_predictions: dict[int, pd.Series] = {}
    member_rows: list[dict[str, Any]] = []
    for model_path in model_paths:
        match = re.fullmatch(r"seed_(\d+)_outer_(\d+)\.joblib", model_path.name)
        if match is None:
            raise RuntimeError(f"Cannot parse locked model member name: {model_path.name}")
        seed = int(match.group(1))
        outer_fold_id = int(match.group(2)) - 1
        fold_rows = nested[
            nested["training_seed"].eq(seed)
            & nested["outer_fold_id"].eq(outer_fold_id)
        ]
        train_ids = fold_rows.loc[fold_rows["outer_role"].eq("outer_train"), "case_id_public"].astype(str).tolist()
        test_ids = fold_rows.loc[fold_rows["outer_role"].eq("outer_test"), "case_id_public"].astype(str).tolist()
        if len(train_ids) != 300 or len(test_ids) != 75 or set(train_ids) & set(test_ids):
            raise RuntimeError(f"Invalid locked split for seed={seed}, fold={outer_fold_id}")
        if not set(test_ids).issubset(internal_by_id.index):
            raise RuntimeError(f"Locked split contains unknown internal IDs for seed={seed}, fold={outer_fold_id}")
        model = load_locked_artifact(model_path)
        probability = model.predict_proba(internal_by_id.loc[test_ids])
        series = seed_predictions.setdefault(seed, pd.Series(index=internal_by_id.index, dtype=float))
        if series.loc[test_ids].notna().any():
            raise RuntimeError(f"Duplicate held-out prediction assignment for seed={seed}")
        series.loc[test_ids] = probability
        member_rows.append(
            {
                "training_seed": seed,
                "outer_fold_id": outer_fold_id,
                "n_train": len(train_ids),
                "n_test": len(test_ids),
                "train_test_overlap": 0,
                "model_file": model_path.name,
            }
        )
    if len(seed_predictions) != 3 or any(series.isna().any() for series in seed_predictions.values()):
        raise RuntimeError("Locked structural-missingness members do not cover every internal subject once per seed")
    internal_probability = pd.concat(seed_predictions, axis=1).mean(axis=1)
    replay_internal = pd.DataFrame(
        {
            "public_subject_id": internal_by_id.loc[internal_probability.index, "public_subject_id"],
            "label_binary": internal_by_id.loc[internal_probability.index, "label_binary"].astype(int),
            "probability_COP": internal_probability.to_numpy(float),
        }
    )
    replay_internal.to_csv(output_dir / "structural_missingness_internal_oof_replay.csv", index=False)
    pd.DataFrame(member_rows).to_csv(output_dir / "structural_missingness_member_split_audit.csv", index=False)

    member_probability = np.vstack([load_locked_artifact(path).predict_proba(external) for path in model_paths])
    probability = member_probability.mean(axis=0)
    replay = pd.DataFrame(
        {
            "public_subject_id": external["public_subject_id"],
            "label_binary": external["label_binary"].astype(int),
            "probability_COP": probability,
        }
    )
    replay.to_csv(output_dir / "structural_missingness_external_replay.csv", index=False)
    crosswalk = pd.read_csv(PROJECT_ROOT / "data" / "public_subject_model_crosswalk.csv", dtype=str)
    internal_mapping = crosswalk[crosswalk["cohort"].eq("internal")][["public_subject_id", "clinical_model_id"]]
    mapping = crosswalk[crosswalk["cohort"].eq("external")][["public_subject_id", "clinical_model_id"]]
    replay_internal = replay_internal.merge(internal_mapping, on="public_subject_id", validate="one_to_one")
    replay = replay.merge(mapping, on="public_subject_id", validate="one_to_one")
    locked_internal = locked_clinical_probabilities("internal", "structural_missingness_clinical")
    locked = locked_clinical_probabilities("external", "structural_missingness_clinical")
    internal_joined = replay_internal.merge(
        locked_internal,
        left_on="clinical_model_id",
        right_on="model_patient_id",
        validate="one_to_one",
    )
    joined = replay.merge(locked, left_on="clinical_model_id", right_on="model_patient_id", validate="one_to_one")
    internal_delta = np.abs(internal_joined["probability_COP"] - internal_joined["locked_probability"])
    delta = np.abs(joined["probability_COP"] - joined["locked_probability"])
    return {
        "internal_subjects": len(replay_internal),
        "external_subjects": len(replay),
        "model_members": len(model_paths),
        "training_seeds": sorted(seed_predictions),
        "internal_oof_max_abs_delta": float(internal_delta.max()),
        "internal_oof_mean_abs_delta": float(internal_delta.mean()),
        "external_max_abs_delta": float(delta.max()),
        "external_mean_abs_delta": float(delta.mean()),
        "internal_oof_status": "REPLAYED_FROM_15_LOCKED_MEMBERS_AND_LOCKED_HELD_OUT_SPLITS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay the locked full-structured and structural-missingness clinical branches.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    internal = load_cohort("internal")
    external = load_cohort("external")
    summary = {
        "full_structured": replay_full_structured(internal, external, output_dir),
        "structural_missingness": replay_structural_missingness(internal, external, output_dir),
    }
    write_json(output_dir / "full_cohort_clinical_replay_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
