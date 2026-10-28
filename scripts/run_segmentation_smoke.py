from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cap_cop.imaging.segmentation.inference import all_checkpoint_paths, checkpoint_paths_for_outer, load_checkpoint_manifest
from cap_cop.imaging.segmentation.model import load_locked_checkpoint
from cap_cop.imaging.segmentation.predict_external import predict_external_ensemble
from cap_cop.imaging.segmentation.predict_internal import predict_internal_strict_oof
from cap_cop.imaging.segmentation.splits import verify_all_strict_oof_folds
from cap_cop.data.release_inputs import dicom_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one internal and one external de-identified DICOM through segmentation.")
    parser.add_argument("--internal-subject-id", required=True)
    parser.add_argument("--external-subject-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "segmentation_smoke")
    args = parser.parse_args()
    crosswalk = pd.read_csv(PROJECT_ROOT / "data" / "public_subject_model_crosswalk.csv", dtype=str).fillna("")
    selections = {}
    for cohort, subject in (("internal", args.internal_subject_id), ("external", args.external_subject_id)):
        rows = crosswalk.loc[crosswalk["cohort"].eq(cohort) & crosswalk["public_subject_id"].eq(subject)]
        if len(rows) != 1:
            raise ValueError(f"Expected one {cohort} patient in the authorised crosswalk")
        selections[cohort] = rows.iloc[0]
    internal_row, external_row = selections["internal"], selections["external"]
    splits = pd.read_csv(PROJECT_ROOT / "models" / "segmentation" / "manifests" / "locked_nested_split_patient_manifest.csv", dtype=str)
    held_out = splits.loc[splits["case_id_public"].eq(args.internal_subject_id) & splits["outer_role"].eq("outer_test")]
    folds = held_out["outer_fold_id"].unique()
    if len(folds) != 1 or not internal_row["patient_group_id_public"]:
        raise ValueError("Internal subject lacks one strict-OOF held-out fold or patient-group ID")
    outer_fold = int(folds[0])
    internal_dicom = dicom_dir("internal", internal_row["diagnosis_label"], args.internal_subject_id)
    external_dicom = dicom_dir("external", external_row["diagnosis_label"], args.external_subject_id)
    out_root = args.output_dir.resolve()

    manifest = load_checkpoint_manifest()
    first_checkpoint = all_checkpoint_paths()[0]
    model = load_locked_checkpoint(first_checkpoint, device="cpu")
    del model

    internal = predict_internal_strict_oof(
        public_subject_id=args.internal_subject_id,
        patient_group_id_public=internal_row["patient_group_id_public"],
        outer_fold=outer_fold,
        dicom_dir=internal_dicom,
        output_dir=out_root / "internal",
        device_name="cpu",
        max_models=1,
    )
    external = predict_external_ensemble(
        public_subject_id=args.external_subject_id,
        dicom_dir=external_dicom,
        output_dir=out_root / "external",
        device_name="cpu",
        max_models=1,
    )
    summary = {
        "status": "passed",
        "note": "Smoke test uses one checkpoint per case to keep runtime light; locked full internal/external ensembles are 4 and 20 checkpoints respectively.",
        "checkpoint_rows": int(len(manifest)),
        "selected_fold_checkpoint_count": len(checkpoint_paths_for_outer(outer_fold)),
        "all_checkpoint_count": len(all_checkpoint_paths()),
        "fold_check": verify_all_strict_oof_folds(),
        "internal": internal,
        "external": external,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
