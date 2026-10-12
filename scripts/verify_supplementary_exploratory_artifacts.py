from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "models" / "supplementary_exploratory"


def main() -> int:
    clinical_models = sorted(
        (BASE / "structural_missingness_clinical" / "ensemble_models").glob("*.joblib")
    )
    densenet_ckpts = sorted((BASE / "full_chest_densenet" / "checkpoints").glob("*.pt"))
    internal_img = pd.read_csv(BASE / "full_chest_densenet" / "internal_oof_predictions.csv")
    external_img = pd.read_csv(BASE / "full_chest_densenet" / "external_label_blind_predictions.csv")

    checkpoint_summary = []
    for path in densenet_ckpts:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            keys = sorted(str(k) for k in obj.keys())[:20]
        else:
            keys = [type(obj).__name__]
        checkpoint_summary.append(
            {"name": path.name, "bytes": path.stat().st_size, "top_level_keys": keys}
        )

    report = {
        "clinical_structural_missingness_joblib_count": len(clinical_models),
        "densenet_checkpoint_count": len(densenet_ckpts),
        "densenet_checkpoints_readable": len(checkpoint_summary),
        "densenet_first_checkpoint": checkpoint_summary[0] if checkpoint_summary else None,
        "internal_image_prediction_rows": int(len(internal_img)),
        "external_image_prediction_rows": int(len(external_img)),
        "internal_image_columns": list(internal_img.columns),
        "external_image_columns": list(external_img.columns),
    }
    for col in ["neutral_imaging_id", "public_subject_id", "patient_id"]:
        if col in internal_img.columns:
            report["internal_unique_id_column"] = col
            report["internal_unique_ids"] = int(internal_img[col].nunique())
            break
    for col in ["neutral_external_id", "public_subject_id", "patient_id"]:
        if col in external_img.columns:
            report["external_unique_id_column"] = col
            report["external_unique_ids"] = int(external_img[col].nunique())
            break
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
