from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "paper_model_locked_pairs.json"
DEFAULT_PRIVATE_LOCK_MAP = PROJECT_ROOT / "configs" / "private_locked_pair_ids.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "full_reproduction_20261010" / "multimodal"
CROSSWALK = PROJECT_ROOT / "data" / "public_subject_model_crosswalk.csv"
INTERNAL_LOCK = PROJECT_ROOT / "models" / "multimodal_fusion" / "internal_oof_all_pairs_predictions.csv"
EXTERNAL_LOCK = PROJECT_ROOT / "models" / "multimodal_fusion" / "external_all_pairs_label_free_predictions.csv"

# The release registry and the fusion-component contract historically used two
# semantic spellings. Keep the short component names as an internal dispatch key,
# while exposing one stable paper-facing name to users and output summaries.
PUBLIC_MODEL_ALIASES = {
    "paper_model_a_primary_lesion_radiomics": "primary_lesion_radiomics",
    "paper_model_b_whole_chest_ct_comparator": "whole_chest_ct_comparator",
    "paper_model_c_stable_multirepresentation_ct": "stable_multirepresentation_comparator",
    "paper_model_d_supplementary_exploratory_comparator": "supplementary_exploratory_comparator",
    "primary_lesion_radiomics": "primary_lesion_radiomics",
    "whole_chest_ct_comparator": "whole_chest_ct_comparator",
    "stable_multirepresentation_comparator": "stable_multirepresentation_comparator",
    "supplementary_exploratory_comparator": "supplementary_exploratory_comparator",
}

PUBLIC_MODEL_ORDER = (
    "paper_model_a_primary_lesion_radiomics",
    "paper_model_b_whole_chest_ct_comparator",
    "paper_model_c_stable_multirepresentation_ct",
    "paper_model_d_supplementary_exploratory_comparator",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_probability(path: Path, id_column: str, probability_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {id_column, probability_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    result = frame[[id_column, probability_column]].copy()
    if result[id_column].isna().any() or result[id_column].astype(str).eq("").any():
        raise ValueError(f"{path} contains empty model IDs")
    if result[id_column].duplicated().any():
        raise ValueError(f"{path} contains duplicate model IDs")
    return result.rename(columns={id_column: "model_id", probability_column: "probability_COP"})


def validate_canonical_crosswalk(crosswalk: pd.DataFrame) -> None:
    required = {
        "public_subject_id",
        "cohort",
        "diagnosis_label",
        "label_binary",
        "clinical_model_id",
        "lesion_imaging_model_id",
        "whole_chest_imaging_model_id",
        "segmentation_model_id",
    }
    missing = required.difference(crosswalk.columns)
    if missing:
        raise ValueError(f"Canonical public crosswalk is missing columns: {sorted(missing)}")
    if len(crosswalk) != 408 or crosswalk["public_subject_id"].nunique() != 408:
        raise ValueError("Canonical public crosswalk must contain 408 unique public subjects")
    expected = {("internal", "CAP"): 250, ("internal", "COP"): 125, ("external", "CAP"): 19, ("external", "COP"): 14}
    observed = crosswalk.groupby(["cohort", "diagnosis_label"]).size().to_dict()
    if observed != expected:
        raise ValueError(f"Canonical cohort counts differ from the locked contract: {observed}")
    for column in ("clinical_model_id", "lesion_imaging_model_id", "whole_chest_imaging_model_id", "segmentation_model_id"):
        if crosswalk[column].eq("").any() or crosswalk[column].duplicated().any():
            raise ValueError(f"Canonical crosswalk column is empty or non-unique: {column}")


def to_public(frame: pd.DataFrame, crosswalk: pd.DataFrame, mapping_column: str) -> pd.DataFrame:
    mapping = crosswalk[["public_subject_id", mapping_column]].rename(columns={mapping_column: "model_id"})
    mapping = mapping[mapping["model_id"].notna() & mapping["model_id"].astype(str).ne("")].copy()
    if mapping["model_id"].duplicated().any():
        raise ValueError(f"Crosswalk column {mapping_column} is not unique")
    merged = frame.merge(mapping, on="model_id", how="left", validate="one_to_one")
    if merged["public_subject_id"].isna().any():
        raise ValueError(f"Unmatched IDs while mapping {mapping_column}")
    return merged[["public_subject_id", "probability_COP"]]


def find_component_file(root: Path, relative_path: str, filename: str) -> Path:
    """Resolve a component file from either a structured or flat replay root."""
    candidates = (root / relative_path / filename, root / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Component file was not found in the supplied replay root: "
        + "; ".join(str(path) for path in candidates)
    )


def metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = (probability >= threshold).astype(int)
    sensitivity = recall_score(y, predicted, zero_division=0)
    specificity = recall_score(1 - y, 1 - predicted, zero_division=0)
    return {
        "AUROC": float(roc_auc_score(y, probability)),
        "Accuracy": float(accuracy_score(y, predicted)),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "F1": float(f1_score(y, predicted, zero_division=0)),
    }


def load_components(
    model_name: str,
    cohort: str,
    crosswalk: pd.DataFrame,
    component_mode: str,
    component_root: Path,
    clinical_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = component_root
    if model_name == "primary_lesion_radiomics":
        clinical_name = "full_structured"
        lesion_subdir = "lesion_radiomics_locked_feature_replay" if component_mode == "locked" else "lesion_radiomics"
        lesion_root = root / lesion_subdir if (root / lesion_subdir).is_dir() else root
        if cohort == "internal":
            locked_path = (
                PROJECT_ROOT / "models" / "lesion_radiomics" / "internal_oof_predictions.csv"
                if component_mode == "locked"
                else lesion_root / "05_nested_cv" / "V3_5_D1_INTERNAL375_STRICT_OOF_PREDICTIONS.csv"
            )
            image = read_probability(locked_path, "neutral_patient_id", "probability_COP")
        else:
            locked_path = (
                PROJECT_ROOT / "models" / "lesion_radiomics" / "external_label_blind_predictions.csv"
                if component_mode == "locked"
                else lesion_root / "07_external_label_blind" / "external33_locked_model_inference_replay.csv"
            )
            image = read_probability(locked_path, "neutral_patient_id", "probability_COP")
        # Lesion predictions are keyed by the segmentation subject ID. This is
        # the canonical public mapping; never route them through a V2IMG table.
        image = to_public(image, crosswalk, "segmentation_model_id")
    elif model_name == "whole_chest_ct_comparator":
        clinical_name = "full_structured"
        file_name = "whole_chest_internal_oof_replay.csv" if cohort == "internal" else "whole_chest_external_replay.csv"
        id_column = "neutral_imaging_id" if cohort == "internal" else "neutral_external_id"
        probability_column = "prob_COP"
        image = read_probability(find_component_file(root, "whole_chest_representations", file_name), id_column, probability_column)
        image = to_public(image, crosswalk, "whole_chest_imaging_model_id" if cohort == "internal" else "external_tensor_id")
    elif model_name == "stable_multirepresentation_comparator":
        clinical_name = "full_structured"
        suffix = "internal" if cohort == "internal" else "external"
        input_mode = "locked_tabular" if component_mode == "locked" else "regenerated_tabular"
        try:
            replay_path = find_component_file(
                root,
                "whole_chest_representations",
                f"stable_multirepresentation_{input_mode}_{suffix}_replay.csv",
            )
        except FileNotFoundError:
            replay_path = find_component_file(root, "whole_chest_representations", f"stable_multirepresentation_{suffix}_replay.csv")
        frame = pd.read_csv(
            replay_path,
            encoding="utf-8-sig",
        )
        candidate = "SP6_STABLE_MULTI_REPRESENTATION_RANDOM_FOREST"
        frame = frame[frame["candidate"].eq(candidate)].copy()
        id_column = "neutral_imaging_id" if cohort == "internal" else "neutral_external_id"
        probability_column = "prob_COP" if cohort == "internal" else "prob_COP_mean"
        image = frame[[id_column, probability_column]].rename(
            columns={id_column: "model_id", probability_column: "probability_COP"}
        )
        if image["model_id"].duplicated().any():
            raise ValueError("SP6 patient-level replay is not unique")
        image = to_public(image, crosswalk, "whole_chest_imaging_model_id" if cohort == "internal" else "external_tensor_id")
    elif model_name == "supplementary_exploratory_comparator":
        clinical_name = "structural_missingness"
        file_name = "exploratory_densenet_internal_oof_replay.csv" if cohort == "internal" else "exploratory_densenet_external_canonical_replay.csv"
        frame = pd.read_csv(
            find_component_file(root, "supplementary_exploratory_densenet", file_name),
            encoding="utf-8-sig",
        )
        image = frame[["public_subject_id", "probability_COP"]].copy()
    else:
        raise KeyError(model_name)

    clinical_file = find_component_file(
        clinical_root,
        "clinical",
        f"{clinical_name}_{'internal_oof' if cohort == 'internal' else 'external'}_replay.csv",
    )
    clinical = pd.read_csv(clinical_file, encoding="utf-8-sig")[["public_subject_id", "probability_COP"]]
    return clinical, image


def replay_one(
    model_name: str,
    pair_id: str,
    cohort: str,
    crosswalk: pd.DataFrame,
    threshold: float,
    output_dir: Path,
    component_mode: str,
    component_root: Path,
    clinical_root: Path,
) -> dict[str, Any]:
    clinical, image = load_components(
        model_name, cohort, crosswalk, component_mode, component_root, clinical_root
    )
    components = clinical.merge(
        image,
        on="public_subject_id",
        suffixes=("_clinical", "_image"),
        validate="one_to_one",
    )
    expected = crosswalk[crosswalk["cohort"].eq(cohort)][["public_subject_id", "label_binary", "clinical_model_id"]]
    if set(components["public_subject_id"]) != set(expected["public_subject_id"]):
        raise ValueError(f"{model_name} {cohort} component ID set differs from the cohort contract")
    components = components.merge(expected, on="public_subject_id", validate="one_to_one")

    if cohort == "internal":
        locked = pd.read_csv(INTERNAL_LOCK, encoding="utf-8-sig")
        locked = locked[locked["pair_id"].eq(pair_id)].copy()
        locked = locked.rename(columns={"patient_id": "public_subject_id", "p_weighted_fusion": "probability_COP_locked"})
        joined = components.merge(
            locked[["public_subject_id", "label", "selected_outer_weight", "probability_COP_locked"]],
            on="public_subject_id",
            validate="one_to_one",
        )
        if not joined["label"].astype(int).eq(joined["label_binary"].astype(int)).all():
            raise ValueError(f"{model_name} internal labels disagree with the lock")
        joined["clinical_weight"] = joined["selected_outer_weight"].astype(float)
        joined["image_weight"] = 1.0 - joined["clinical_weight"]
    else:
        locked = pd.read_csv(EXTERNAL_LOCK, encoding="utf-8-sig")
        locked = locked[locked["pair_id"].eq(pair_id)].copy()
        locked = locked.merge(
            crosswalk[crosswalk["cohort"].eq("external")][["public_subject_id", "clinical_model_id"]],
            left_on="patient_id",
            right_on="clinical_model_id",
            validate="one_to_one",
        ).rename(columns={"p_weighted_fusion": "probability_COP_locked"})
        joined = components.merge(
            locked[["public_subject_id", "final_weight_clinical", "final_weight_image", "probability_COP_locked"]],
            on="public_subject_id",
            validate="one_to_one",
        )
        joined["clinical_weight"] = joined["final_weight_clinical"].astype(float)
        joined["image_weight"] = joined["final_weight_image"].astype(float)

    joined["probability_COP_replay"] = (
        joined["clinical_weight"] * joined["probability_COP_clinical"]
        + joined["image_weight"] * joined["probability_COP_image"]
    )
    delta = np.abs(joined["probability_COP_replay"] - joined["probability_COP_locked"])
    output_path = output_dir / f"{model_name}_{cohort}_canonical_replay.csv"
    joined[
        [
            "public_subject_id",
            "label_binary",
            "probability_COP_clinical",
            "probability_COP_image",
            "clinical_weight",
            "image_weight",
            "probability_COP_replay",
            "probability_COP_locked",
        ]
    ].to_csv(output_path, index=False, encoding="utf-8-sig")
    return {
        "subjects": int(len(joined)),
        "id_set_sha256": hashlib.sha256(
            ("\n".join(sorted(joined["public_subject_id"].astype(str))) + "\n").encode("utf-8")
        ).hexdigest().upper(),
        "probability_max_abs_delta": float(delta.max()),
        "probability_mean_abs_delta": float(delta.mean()),
        "replay_metrics": metrics(
            joined["label_binary"].to_numpy(dtype=int),
            joined["probability_COP_replay"].to_numpy(dtype=float),
            threshold,
        ),
        "locked_metrics": metrics(
            joined["label_binary"].to_numpy(dtype=int),
            joined["probability_COP_locked"].to_numpy(dtype=float),
            threshold,
        ),
        "output_sha256": sha256_file(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay the four paper multimodal models from full-cohort component outputs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--private-lock-map",
        type=Path,
        default=DEFAULT_PRIVATE_LOCK_MAP,
        help="Local governed mapping from semantic model names to historical lock selectors; never publish this file.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--component-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "full_reproduction_20261010",
        help="Directory containing upstream component replay files; defaults to this workspace.",
    )
    parser.add_argument(
        "--clinical-root",
        type=Path,
        default=None,
        help="Optional root containing the clinical replay files when the image replay root is separate.",
    )
    parser.add_argument("--lesion-root", type=Path, default=None, help="Optional direct lesion-radiomics replay directory.")
    parser.add_argument("--whole-chest-root", type=Path, default=None, help="Optional direct whole-chest replay directory.")
    parser.add_argument("--exploratory-root", type=Path, default=None, help="Optional direct exploratory replay directory.")
    parser.add_argument(
        "--component-mode",
        choices=["regenerated", "locked"],
        default="regenerated",
        help="Use regenerated upstream components or the locked intermediate components.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional semantic model names to replay; defaults to every model in the config.",
    )
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    private_lock_map_path = args.private_lock_map.resolve()
    if not private_lock_map_path.is_file():
        raise FileNotFoundError(
            "Historical lock comparison requires a governed private lock map: "
            f"{private_lock_map_path}"
        )
    private_lock_map = json.loads(private_lock_map_path.read_text(encoding="utf-8"))
    crosswalk = pd.read_csv(CROSSWALK, encoding="utf-8-sig")
    validate_canonical_crosswalk(crosswalk)
    output_dir = args.output_dir.resolve()
    component_root = args.component_root.resolve()
    clinical_root = (args.clinical_root or component_root).resolve()
    if not component_root.is_dir():
        raise FileNotFoundError(f"Component root does not exist: {component_root}")
    model_roots = {
        "primary_lesion_radiomics": (args.lesion_root or component_root).resolve(),
        "whole_chest_ct_comparator": (args.whole_chest_root or component_root).resolve(),
        "stable_multirepresentation_comparator": (args.whole_chest_root or component_root).resolve(),
        "supplementary_exploratory_comparator": (args.exploratory_root or component_root).resolve(),
    }
    for model_root in model_roots.values():
        if not model_root.is_dir():
            raise FileNotFoundError(f"Component root does not exist: {model_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold = float(config["threshold"])
    requested_models = args.models or list(PUBLIC_MODEL_ORDER)
    resolved_models = [PUBLIC_MODEL_ALIASES.get(name, name) for name in requested_models]
    unknown_models = sorted(set(resolved_models).difference(config["paper_models"]))
    if unknown_models:
        raise ValueError(f"Unknown semantic model names: {unknown_models}")
    summary: dict[str, Any] = {
        "status": "PASS",
        "threshold": threshold,
        "component_mode": args.component_mode,
        "component_root": str(component_root),
        "identity_standard": "canonical_public_subject_mapping",
        "internal_lock_sha256": sha256_file(INTERNAL_LOCK),
        "external_lock_sha256": sha256_file(EXTERNAL_LOCK),
        "models": {},
    }
    for public_name, model_name in zip(requested_models, resolved_models):
        model_config = config["paper_models"][model_name]
        if model_name not in private_lock_map:
            raise KeyError(f"No private historical lock selector for semantic model {model_name}")
        summary["models"][public_name] = {
            cohort: replay_one(
                model_name,
                str(private_lock_map[model_name]),
                cohort,
                crosswalk,
                threshold,
                output_dir,
                args.component_mode,
                model_roots[model_name],
                clinical_root,
            )
            for cohort in ("internal", "external")
        }
    deltas = [
        result[cohort]["probability_max_abs_delta"]
        for result in summary["models"].values()
        for cohort in ("internal", "external")
    ]
    if any(delta > 1e-12 for delta in deltas):
        summary["status"] = "PASS_WITH_LOCK_DIFFERENCES_REQUIRING_REVIEW"
        summary["status_note"] = (
            "The component and fusion interfaces ran, but at least one canonical replay probability differs "
            "from the historical fusion lock. This is not an exact historical-lock reproduction."
        )
    summary_path = output_dir / "full_cohort_multimodal_replay_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
