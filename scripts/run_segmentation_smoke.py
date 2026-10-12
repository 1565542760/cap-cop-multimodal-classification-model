from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cap_cop.imaging.segmentation.inference import all_checkpoint_paths, checkpoint_paths_for_outer, load_checkpoint_manifest
from cap_cop.imaging.segmentation.model import load_locked_checkpoint
from cap_cop.imaging.segmentation.predict_external import predict_external_ensemble
from cap_cop.imaging.segmentation.predict_internal import predict_internal_strict_oof
from cap_cop.imaging.segmentation.splits import verify_all_strict_oof_folds


def main() -> None:
    internal_dicom = PROJECT_ROOT / "data" / "internal" / "CAP" / "imaging" / "ct" / "CAP_0001" / "dicom"
    external_dicom = PROJECT_ROOT / "data" / "external" / "CAP" / "imaging" / "ct" / "EXT_CAP_001" / "dicom"
    out_root = PROJECT_ROOT / "outputs" / "segmentation_smoke"

    manifest = load_checkpoint_manifest()
    first_checkpoint = all_checkpoint_paths()[0]
    model = load_locked_checkpoint(first_checkpoint, device="cpu")
    del model

    internal = predict_internal_strict_oof(
        public_subject_id="CAP_0001",
        patient_group_id_public="P000001",
        outer_fold=2,
        dicom_dir=internal_dicom,
        output_dir=out_root / "internal",
        device_name="cpu",
        max_models=1,
    )
    external = predict_external_ensemble(
        public_subject_id="EXT_CAP_001",
        dicom_dir=external_dicom,
        output_dir=out_root / "external",
        device_name="cpu",
        max_models=1,
    )
    summary = {
        "status": "passed",
        "note": "Smoke test uses one checkpoint per case to keep runtime light; locked full internal/external ensembles are 4 and 20 checkpoints respectively.",
        "checkpoint_rows": int(len(manifest)),
        "outer2_checkpoint_count": len(checkpoint_paths_for_outer(2)),
        "all_checkpoint_count": len(all_checkpoint_paths()),
        "fold_check": verify_all_strict_oof_folds(),
        "internal": internal,
        "external": external,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
