from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cap_cop.data.manifest import read_all_manifests
from cap_cop.evaluation.metrics import binary_metrics
from cap_cop.imaging.dicom import dicom_smoke_summary
from cap_cop.models.multimodal.fusion import classify, weighted_average_fusion
from cap_cop.models.registry import model_status_table
from cap_cop.paths import project_root
from cap_cop.preprocessing.clinical import full_structured_clinical_preprocessor


def _first_case(manifests: pd.DataFrame, cohort: str, label: str) -> pd.Series:
    subset = manifests[(manifests["cohort"] == cohort) & (manifests["diagnosis_label"] == label)]
    if subset.empty:
        raise AssertionError(f"No case found for {cohort}/{label}")
    return subset.sort_values("public_subject_id").iloc[0]


def _read_clinical(row: pd.Series) -> pd.DataFrame:
    clinical_path = project_root() / "data" / row["cohort"] / row["diagnosis_label"] / row["clinical_path"]
    frame = pd.read_csv(clinical_path)
    for id_column in ["public_subject_id", "case_id_public"]:
        if id_column in frame.columns:
            return frame[frame[id_column] == row["public_subject_id"]]
    return frame


def main() -> None:
    root = project_root()
    manifests = read_all_manifests(root)
    preprocessor = full_structured_clinical_preprocessor()
    dicom_results = {}
    clinical_results = {}
    for cohort, label in [("internal", "CAP"), ("internal", "COP"), ("external", "CAP"), ("external", "COP")]:
        row = _first_case(manifests, cohort, label)
        dicom_path = root / "data" / cohort / label / row["ct_path"]
        dicom_results[f"{cohort}_{label}"] = dicom_smoke_summary(dicom_path)
        transformed = preprocessor.transform(_read_clinical(row))
        clinical_frame = _read_clinical(row)
        id_column = "public_subject_id" if "public_subject_id" in clinical_frame.columns else (
            "case_id_public" if "case_id_public" in clinical_frame.columns else "none"
        )
        clinical_results[f"{cohort}_{label}"] = {
            "rows": int(transformed.shape[0]),
            "columns": int(transformed.shape[1]),
            "id_column": id_column,
            "has_label_feature": "diagnosis_label" in transformed.columns or "label" in transformed.columns,
        }

    internal_mask_files = [p for p in (root / "data" / "internal").rglob("imaging/masks/*") if p.is_file()]
    external_mask_dirs = [p for p in (root / "data" / "external").rglob("masks") if p.is_dir()]

    fused = weighted_average_fusion(np.array([0.2, 0.8]), np.array([0.4, 0.6]), 0.70, 0.30)
    metric_summary = binary_metrics([0, 1], fused, threshold=0.50)
    with tempfile.TemporaryDirectory() as tmp:
        marker = Path(tmp) / "smoke_marker.txt"
        marker.write_text("temporary output only", encoding="utf-8")

    report = {
        "status": "passed",
        "project_root": str(root),
        "dicom_smoke": dicom_results,
        "clinical_smoke": clinical_results,
        "internal_mask_file_count": len(internal_mask_files),
        "external_mask_dir_count": len(external_mask_dirs),
        "model_status": model_status_table(),
        "fusion_smoke": {"probabilities": fused.tolist(), "classes": classify(fused).tolist(), "metrics": metric_summary},
        "temporary_outputs_cleaned": True,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
