from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.run_full_cohort_clinical_replay import load_cohort
from cap_cop.clinical import core_and_availability_gated_source as historical
from cap_cop.clinical.core_and_availability_gated_model import (
    load_locked_artifact,
    predict_availability_gated,
    predict_core,
)
from cap_cop.private_lock_selectors import get_selector


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "full_reproduction_20261010"
    / "clinical_input_comparison"
)
SPLIT_MANIFEST = (
    PROJECT_ROOT
    / "models"
    / "segmentation"
    / "manifests"
    / "locked_nested_split_patient_manifest.csv"
)
MODEL_ROOT = PROJECT_ROOT / "models" / "clinical" / "core_and_availability"

MODEL_IDS = {
    "core": "core_clinical",
    "gated": "availability_gated_clinical",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def locked_probabilities(scope: str, selector_name: str) -> pd.DataFrame:
    clinical_id = get_selector(selector_name)
    filename = (
        "internal_oof_all_pairs_predictions.csv"
        if scope == "internal"
        else "external_all_pairs_label_free_predictions.csv"
    )
    frame = pd.read_csv(PROJECT_ROOT / "models" / "multimodal_fusion" / filename)
    selected = frame.loc[
        frame["clinical_id"].eq(clinical_id), ["patient_id", "p_clinical"]
    ].copy()
    disagreement = selected.groupby("patient_id")["p_clinical"].nunique()
    if (disagreement > 1).any():
        raise RuntimeError(f"Locked {clinical_id} probabilities differ across image pairs")
    return selected.drop_duplicates("patient_id").rename(
        columns={"patient_id": "clinical_model_id", "p_clinical": "locked_probability"}
    )


def split_contract() -> tuple[pd.DataFrame, dict[str, Any]]:
    split = pd.read_csv(SPLIT_MANIFEST)
    selected = split.loc[
        split["outer_seed"].eq(historical.LOCKED_SPLIT_SEED)
        & split["training_seed"].eq(historical.LOCKED_SPLIT_SEED)
    ].copy()
    if len(selected) != 1875:
        raise RuntimeError(f"Expected 1,875 selected split rows, found {len(selected)}")
    test_rows = selected.loc[selected["outer_role"].eq("outer_test")]
    fold_map = (
        test_rows.set_index("case_id_public")["outer_fold_id"].astype(int).to_dict()
    )
    if len(fold_map) != 375:
        raise RuntimeError(f"Expected 375 held-out fold assignments, found {len(fold_map)}")
    inner_maps: dict[int, dict[str, int]] = {}
    for outer_fold in range(historical.OUTER_FOLDS):
        train_rows = selected.loc[
            selected["outer_fold_id"].eq(outer_fold)
            & selected["outer_role"].eq("outer_train")
        ]
        inner_maps[outer_fold] = (
            train_rows.set_index("case_id_public")["inner_fold_id"].astype(int).to_dict()
        )
        if len(inner_maps[outer_fold]) != 300:
            raise RuntimeError(
                f"Outer fold {outer_fold} has {len(inner_maps[outer_fold])} training cases"
            )
    return selected, {"split": selected, "fold_map": fold_map, "inner_maps": inner_maps}


def metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(labels, probability)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
    }


def compare(
    replay: pd.DataFrame,
    scope: str,
    clinical_id: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    crosswalk = pd.read_csv(
        PROJECT_ROOT / "data" / "public_subject_model_crosswalk.csv", dtype=str
    )
    mapping = crosswalk.loc[
        crosswalk["cohort"].eq(scope), ["public_subject_id", "clinical_model_id"]
    ]
    joined = replay.merge(mapping, on="public_subject_id", validate="one_to_one")
    joined = joined.merge(
        locked_probabilities(scope, clinical_id),
        on="clinical_model_id",
        validate="one_to_one",
    )
    joined["absolute_delta"] = np.abs(
        joined["probability_COP"] - joined["locked_probability"]
    )
    return joined, {
        "subjects": int(len(joined)),
        "max_abs_delta": float(joined["absolute_delta"].max()),
        "mean_abs_delta": float(joined["absolute_delta"].mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay the locked core and availability-gated clinical branches."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    internal = load_cohort("internal")
    external = load_cohort("external")
    split, split_info = split_contract()
    if set(internal["case_id_public"]) != set(split_info["fold_map"]):
        raise RuntimeError("Internal clinical cohort and locked split patient sets differ")

    restored = historical.restore_clinical_oof(internal, split_info)
    base = restored["base"].rename(columns={"neutral_patient_id": "public_subject_id"})
    labels = base["label"].to_numpy(dtype=int)

    internal_replays = {
        "core": base[["public_subject_id", "outer_fold", "label", "P_core_COP"]]
        .rename(columns={"P_core_COP": "probability_COP"})
        .copy(),
        "gated": base[
            [
                "public_subject_id",
                "outer_fold",
                "label",
                "available_full_only_count",
                "clinical_route",
                "P_gated_COP",
            ]
        ]
        .rename(columns={"P_gated_COP": "probability_COP"})
        .copy(),
    }

    core_artifact_path = MODEL_ROOT / "core_clinical_ensemble.joblib"
    gated_artifact_path = MODEL_ROOT / "availability_gated_ensemble.joblib"
    core_artifact = load_locked_artifact(core_artifact_path)
    gated_artifact = load_locked_artifact(gated_artifact_path)
    core_external_probability = predict_core(core_artifact, external)
    gated_external_probability, availability, use_full = predict_availability_gated(
        gated_artifact, external
    )
    external_replays = {
        "core": pd.DataFrame(
            {
                "public_subject_id": external["public_subject_id"],
                "label": external["label_binary"].astype(int),
                "probability_COP": core_external_probability,
            }
        ),
        "gated": pd.DataFrame(
            {
                "public_subject_id": external["public_subject_id"],
                "label": external["label_binary"].astype(int),
                "available_full_only_count": availability,
                "clinical_route": np.where(use_full, "full_ensemble", "core_ensemble"),
                "probability_COP": gated_external_probability,
            }
        ),
    }

    summary: dict[str, Any] = {
        "scope": (
            "Full Internal375 strict nested OOF replay and External33 locked-artifact "
            "inference for the core7 and availability-gated clinical configurations."
        ),
        "split_manifest": {
            "path": str(SPLIT_MANIFEST.relative_to(PROJECT_ROOT)),
            "sha256": sha256(SPLIT_MANIFEST),
            "rows_all_seeds": int(len(pd.read_csv(SPLIT_MANIFEST))),
            "rows_selected_seed": int(len(split)),
        },
        "features": {
            "full16": historical.FULL_FEATURES,
            "core7": historical.CORE_FEATURES,
            "full_only9": historical.FULL_ONLY_FEATURES,
            "gate": "available_full_only_count >= 6",
        },
        "artifacts": {
            "core": {
                "path": str(core_artifact_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256(core_artifact_path),
            },
            "gated": {
                "path": str(gated_artifact_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256(gated_artifact_path),
            },
        },
        "models": {},
    }

    for name in ("core", "gated"):
        internal_joined, internal_audit = compare(
            internal_replays[name], "internal", MODEL_IDS[name]
        )
        external_joined, external_audit = compare(
            external_replays[name], "external", MODEL_IDS[name]
        )
        internal_joined.to_csv(output_dir / f"{name}_internal_oof_replay_audit.csv", index=False)
        external_joined.to_csv(output_dir / f"{name}_external_replay_audit.csv", index=False)
        probability = internal_replays[name]["probability_COP"].to_numpy(dtype=float)
        external_labels = external_replays[name]["label"].to_numpy(dtype=int)
        external_probability = external_replays[name]["probability_COP"].to_numpy(dtype=float)
        summary["models"][name] = {
            "clinical_id": MODEL_IDS[name],
            "internal": {**internal_audit, **metrics(labels, probability)},
            "external": {
                **external_audit,
                **metrics(external_labels, external_probability),
            },
        }

    restored["fold_records"].to_csv(output_dir / "outer_fold_model_selection.csv", index=False)
    restored["inner_records"].to_csv(output_dir / "inner_fold_model_selection.csv", index=False)
    write_json(output_dir / "clinical_input_comparison_replay_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
