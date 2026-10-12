from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
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
from cap_cop.clinical import alternate_configuration_confirmation_source as confirmation
from cap_cop.clinical import alternate_configuration_external_source as external_source
from cap_cop.clinical import alternate_configuration_source as historical
from cap_cop.private_lock_selectors import get_selector


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "full_reproduction_20261010"
    / "alternate_clinical_configuration"
)
MODEL_ROOT = PROJECT_ROOT / "models" / "clinical" / "alternate_configuration"
LOCK_PATH = MODEL_ROOT / "clinical_model_lock.json"
SELECTION_PATH = MODEL_ROOT / "strict_family_selection_registry.csv"
SPLIT_PATH = (
    PROJECT_ROOT
    / "models"
    / "segmentation"
    / "manifests"
    / "locked_nested_split_patient_manifest.csv"
)
CLINICAL_SELECTOR = "alternate_clinical"


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


def locked_probabilities(scope: str) -> pd.DataFrame:
    clinical_id = get_selector(CLINICAL_SELECTOR)
    filename = (
        "internal_oof_all_pairs_predictions.csv"
        if scope == "internal"
        else "external_all_pairs_label_free_predictions.csv"
    )
    frame = pd.read_csv(PROJECT_ROOT / "models" / "multimodal_fusion" / filename)
    selected = frame.loc[
        frame["clinical_id"].eq(clinical_id), ["patient_id", "p_clinical"]
    ]
    disagreement = selected.groupby("patient_id")["p_clinical"].nunique()
    if (disagreement > 1).any():
        raise RuntimeError("Locked alternate clinical probabilities differ across image pairs")
    return selected.drop_duplicates("patient_id").rename(
        columns={"patient_id": "clinical_model_id", "p_clinical": "locked_probability"}
    )


def compare(replay: pd.DataFrame, scope: str) -> tuple[pd.DataFrame, dict[str, float]]:
    crosswalk = pd.read_csv(
        PROJECT_ROOT / "data" / "public_subject_model_crosswalk.csv", dtype=str
    )
    mapping = crosswalk.loc[
        crosswalk["cohort"].eq(scope), ["public_subject_id", "clinical_model_id"]
    ]
    joined = replay.merge(mapping, on="public_subject_id", validate="one_to_one")
    joined = joined.merge(
        locked_probabilities(scope), on="clinical_model_id", validate="one_to_one"
    )
    joined["absolute_delta"] = np.abs(
        joined["probability_COP"] - joined["locked_probability"]
    )
    return joined, {
        "subjects": int(len(joined)),
        "max_abs_delta": float(joined["absolute_delta"].max()),
        "mean_abs_delta": float(joined["absolute_delta"].mean()),
    }


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    labels = frame["label_binary"].to_numpy(dtype=int)
    probability = frame["probability_COP"].to_numpy(dtype=float)
    prediction = (probability >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(labels, probability)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
    }


def replay_internal(
    internal: pd.DataFrame,
    split: pd.DataFrame,
    lock: dict[str, Any],
    selection: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    internal_by_id = internal.set_index("case_id_public", drop=False)
    features = list(lock["exact_features"])
    X = internal_by_id[features].apply(pd.to_numeric, errors="coerce")
    y = internal_by_id["label_binary"].astype(int)
    selected_rows = selection.loc[
        selection["family"].eq(selection["selected_family_for_outer_fold"])
    ].copy()
    if len(selected_rows) != 15:
        raise RuntimeError(f"Expected 15 locked strict-family selections, found {len(selected_rows)}")
    seed_predictions: dict[int, pd.Series] = {}
    audit_rows: list[dict[str, Any]] = []
    for seed in historical.TRAINING_SEEDS:
        seed_probability = pd.Series(index=internal_by_id.index, dtype=float)
        for fold in range(5):
            train_ids, test_ids, inner_map = historical.get_split(split, seed, fold)
            selected = selected_rows.loc[
                selected_rows["training_seed"].eq(seed)
                & selected_rows["outer_fold_id"].eq(fold)
            ]
            if len(selected) != 1:
                raise RuntimeError(
                    f"Expected one locked strict-family config for seed={seed}, fold={fold}"
                )
            record = selected.iloc[0]
            family = str(record["family"])
            best = {
                "missingness": record["selected_missingness"],
                "params": json.loads(record["selected_params_json"]),
            }
            calibrator, _, _, raw_calibration = confirmation.fit_family_crossfit(
                family, best, X, y, train_ids, inner_map, seed
            )
            model = historical.make_pipeline(
                family, best["missingness"], best["params"], seed + 777
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X.loc[train_ids], y.loc[train_ids])
            raw_probability = model.predict_proba(X.loc[test_ids])[:, 1]
            probability = historical.apply_calibrator(calibrator, raw_probability)
            seed_probability.loc[test_ids] = probability
            audit_rows.append(
                {
                    "training_seed": seed,
                    "outer_fold_id": fold,
                    "train_subjects": len(train_ids),
                    "test_subjects": len(test_ids),
                    "train_test_overlap": len(set(train_ids) & set(test_ids)),
                    "selected_family": family,
                    "selected_missingness": best["missingness"],
                    "selected_params_json": json.dumps(best["params"], sort_keys=True),
                    "calibration_coef": float(calibrator.coef_[0][0]),
                    "calibration_intercept": float(calibrator.intercept_[0]),
                    "calibration_subjects": int(raw_calibration.notna().sum()),
                }
            )
        if seed_probability.isna().any():
            raise RuntimeError(f"Incomplete held-out replay for seed {seed}")
        seed_predictions[seed] = seed_probability
    consensus = pd.concat(seed_predictions, axis=1).mean(axis=1)
    replay = pd.DataFrame(
        {
            "public_subject_id": consensus.index,
            "label_binary": y.loc[consensus.index].to_numpy(dtype=int),
            "probability_COP": consensus.to_numpy(dtype=float),
        }
    )
    return replay, pd.DataFrame(audit_rows)


def replay_external(
    internal: pd.DataFrame,
    external: pd.DataFrame,
    split: pd.DataFrame,
    lock: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    features = list(lock["exact_features"])
    internal_by_id = internal.set_index("case_id_public", drop=False)
    internal_x = internal_by_id[features].apply(pd.to_numeric, errors="coerce")
    external_x = external[features].apply(pd.to_numeric, errors="coerce")
    y = internal_by_id["label_binary"].astype(int)
    probabilities: dict[int, np.ndarray] = {}
    calibrator_records: list[dict[str, Any]] = []
    for seed in historical.TRAINING_SEEDS:
        raw_oof = pd.Series(index=internal_x.index, dtype=float)
        for fold in range(5):
            train_ids, test_ids, _ = historical.get_split(split, seed, fold)
            model = external_source.make_rf_pipeline(seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(internal_x.loc[train_ids], y.loc[train_ids])
            raw_oof.loc[test_ids] = model.predict_proba(internal_x.loc[test_ids])[:, 1]
        if raw_oof.isna().any():
            raise RuntimeError(f"Incomplete deployment OOF calibration probabilities for seed {seed}")
        calibrator, info = external_source.fit_platt_from_seed_oof(raw_oof, y)
        info["training_seed"] = seed
        calibrator_records.append(info)
        full_model = external_source.make_rf_pipeline(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full_model.fit(internal_x, y)
        raw_external = full_model.predict_proba(external_x)[:, 1]
        probabilities[seed] = external_source.apply_calibrator(calibrator, raw_external)
    mean_probability = np.mean(np.vstack(list(probabilities.values())), axis=0)
    replay = pd.DataFrame(
        {
            "public_subject_id": external["public_subject_id"],
            "label_binary": external["label_binary"].astype(int),
            "probability_COP": mean_probability,
        }
    )
    for seed, probability in probabilities.items():
        replay[f"probability_COP_seed_{seed}"] = probability
    return replay, calibrator_records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay the locked alternate 16-variable RF clinical configuration."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock["final_algorithm"] != "RF" or len(lock["exact_features"]) != 16:
        raise RuntimeError("Alternate clinical lock is not the expected 16-variable RF model")
    internal = load_cohort("internal")
    external = load_cohort("external")
    split = pd.read_csv(SPLIT_PATH)
    selection = pd.read_csv(SELECTION_PATH)
    if set(internal["case_id_public"]) != set(split["case_id_public"]):
        raise RuntimeError("Internal clinical cohort and locked split patient sets differ")

    internal_replay, fold_audit = replay_internal(internal, split, lock, selection)
    external_replay, calibrators = replay_external(internal, external, split, lock)
    internal_joined, internal_audit = compare(internal_replay, "internal")
    external_joined, external_audit = compare(external_replay, "external")
    internal_joined.to_csv(output_dir / "alternate_internal_oof_replay_audit.csv", index=False)
    external_joined.to_csv(output_dir / "alternate_external_replay_audit.csv", index=False)
    fold_audit.to_csv(output_dir / "alternate_internal_fold_audit.csv", index=False)

    summary = {
        "scope": (
            "Full Internal375 replay of the 15 locked strict-family configurations and "
            "External33 replay of the locked three-seed full-development RF deployment strategy. "
            "The historical family-selection decisions are read from their locked registry and are not rerun."
        ),
        "clinical_selector": CLINICAL_SELECTOR,
        "features": lock["exact_features"],
        "lock": {"path": str(LOCK_PATH.relative_to(PROJECT_ROOT)), "sha256": sha256(LOCK_PATH)},
        "selection_registry": {
            "path": str(SELECTION_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256(SELECTION_PATH),
            "selected_family_counts": {
                str(key): int(value)
                for key, value in fold_audit["selected_family"].value_counts().to_dict().items()
            },
        },
        "split": {"path": str(SPLIT_PATH.relative_to(PROJECT_ROOT)), "sha256": sha256(SPLIT_PATH)},
        "internal": {**internal_audit, **metrics(internal_replay)},
        "external": {**external_audit, **metrics(external_replay)},
        "external_calibrators": calibrators,
        "environment_contract": lock.get("package_versions", {}),
    }
    write_json(output_dir / "alternate_clinical_replay_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
