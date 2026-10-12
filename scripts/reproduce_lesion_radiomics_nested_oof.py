"""Reproduce the locked lesion-radiomics nested-CV selection from public inputs.

This command re-runs the historical internal-only feature-stability, redundancy,
inner model-selection, and strict outer OOF procedure. It deliberately does not
read external labels and does not touch the original experiment tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from cap_cop.pipelines import lesion_radiomics_replay_pipeline as locked


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = (
    PROJECT_ROOT
    / "outputs"
    / "a_raw_dicom_replay_20261011_v4"
    / "03_feature_extraction"
    / "full_cohort_perturbation_pyradiomics_features.csv"
)
DEFAULT_CROSSWALK = PROJECT_ROOT / "data" / "public_subject_model_crosswalk.csv"
DEFAULT_ACQUISITION = PROJECT_ROOT / "models" / "lesion_radiomics" / "internal_acquisition_metadata.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "a_nested_oof_reproduction_20261011"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_public_output(output_dir: Path) -> None:
    """Point the migrated historical functions at this public replay directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    locked.PROJECT_ROOT = PROJECT_ROOT
    locked.OUT = output_dir
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


def load_public_inputs(matrix_path: Path, crosswalk_path: Path, acquisition_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    matrix = pd.read_csv(matrix_path, encoding="utf-8-sig")
    crosswalk = pd.read_csv(crosswalk_path, encoding="utf-8-sig")
    acquisition = pd.read_csv(acquisition_path, encoding="utf-8-sig")

    required_matrix = {
        "public_subject_id",
        "outer_fold",
        "perturbation",
        "status",
    }
    missing = sorted(required_matrix - set(matrix.columns))
    if missing:
        raise RuntimeError(f"Replay matrix is missing required columns: {missing}")

    internal = crosswalk[crosswalk["cohort"].eq("internal")].copy()
    if len(internal) != 375 or internal["segmentation_model_id"].nunique() != 375:
        raise RuntimeError("Public crosswalk does not contain 375 unique internal model IDs")
    label_counts = internal["label_binary"].astype(int).value_counts().to_dict()
    if label_counts != {0: 250, 1: 125}:
        raise RuntimeError(f"Unexpected internal label counts: {label_counts}")

    labels = internal[["segmentation_model_id", "label_binary"]].rename(
        columns={"segmentation_model_id": "neutral_patient_id"}
    )
    labels["label_binary"] = labels["label_binary"].astype(int)
    internal_ids = set(labels["neutral_patient_id"].astype(str))
    baseline = matrix[
        matrix["public_subject_id"].astype(str).isin(set(internal["public_subject_id"].astype(str)))
        & matrix["perturbation"].eq("BASELINE")
    ].copy()
    if len(baseline) != 375 or baseline["public_subject_id"].nunique() != 375:
        raise RuntimeError("Replay matrix does not contain one baseline row for each Internal375 public ID")

    # The public release crosswalk maps public IDs to the historical neutral IDs.
    # Keep the model matrix's neutral ID contract used by the migrated pipeline.
    public_to_neutral = dict(
        zip(internal["public_subject_id"].astype(str), internal["segmentation_model_id"].astype(str))
    )
    matrix = matrix.copy()
    matrix["neutral_patient_id"] = matrix["public_subject_id"].astype(str).map(public_to_neutral)
    matrix.loc[matrix["neutral_patient_id"].isna(), "neutral_patient_id"] = matrix.loc[
        matrix["neutral_patient_id"].isna(), "public_subject_id"
    ]
    # The historical extractor emitted internal rows in neutral-ID order.  The
    # public raw replay is produced by parallel workers, so completion order is
    # nondeterministic; restore the historical order before seeded CV splitting.
    matrix = matrix.sort_values(
        ["neutral_patient_id", "perturbation"], kind="mergesort"
    ).reset_index(drop=True)
    if set(matrix.loc[matrix["neutral_patient_id"].isin(internal_ids), "neutral_patient_id"]) != internal_ids:
        raise RuntimeError("Internal model-ID bridge does not cover all 375 patients")

    acquisition_by_id = acquisition.set_index("neutral_patient_id").to_dict("index")
    missing_acquisition = sorted(internal_ids - set(acquisition_by_id))
    if missing_acquisition:
        raise RuntimeError(f"Acquisition metadata missing {len(missing_acquisition)} internal IDs")
    return labels, matrix, acquisition_by_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce Internal375 lesion-radiomics nested CV from public inputs.")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--crosswalk", type=Path, default=DEFAULT_CROSSWALK)
    parser.add_argument("--acquisition-metadata", type=Path, default=DEFAULT_ACQUISITION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    matrix_path = args.matrix.resolve()
    crosswalk_path = args.crosswalk.resolve()
    acquisition_path = args.acquisition_metadata.resolve()
    output_dir = args.output_dir.resolve()
    for path in (matrix_path, crosswalk_path, acquisition_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    configure_public_output(output_dir)
    labels, matrix, acquisition_by_id = load_public_inputs(
        matrix_path, crosswalk_path, acquisition_path
    )
    internal_matrix = matrix[matrix["neutral_patient_id"].isin(set(labels["neutral_patient_id"]))].copy()
    internal_matrix = internal_matrix.drop(columns=["public_subject_id"])

    oof, outer_metrics, selected_features, aggregate_metrics = locked.nested_cv(
        labels, internal_matrix, acquisition_by_id
    )
    summary = {
        "status": "PASS",
        "procedure": "historical_internal_only_nested_cv",
        "external_labels_read": False,
        "retraining": True,
        "model_selection": True,
        "internal_subjects": int(len(oof)),
        "label_counts": labels["label_binary"].value_counts().sort_index().astype(int).to_dict(),
        "matrix_source": str(matrix_path),
        "matrix_sha256": sha256_file(matrix_path),
        "crosswalk_source": str(crosswalk_path),
        "crosswalk_sha256": sha256_file(crosswalk_path),
        "acquisition_source": str(acquisition_path),
        "acquisition_sha256": sha256_file(acquisition_path),
        "aggregate_metrics": aggregate_metrics,
        "outer_metrics_rows": int(len(outer_metrics)),
        "selected_feature_rows": int(len(selected_features)),
        "oof_sha256": sha256_file(locked.DIRS["cv"] / "lesion_radiomics_replay_INTERNAL375_STRICT_OOF_PREDICTIONS.csv"),
        "note": "This is an independent replay of the locked internal procedure; it does not overwrite locked artifacts.",
    }
    (output_dir / "nested_oof_reproduction_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
