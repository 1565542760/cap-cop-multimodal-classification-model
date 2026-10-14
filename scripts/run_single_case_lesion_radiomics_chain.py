from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cap_cop.imaging.segmentation.preprocess import (
    read_primary_dicom_series,
    resample_hu_image_to_working_spacing,
)
from cap_cop.data.release_inputs import dicom_dir
def mask_image_from_npy(mask_path: Path, reference: sitk.Image) -> sitk.Image:
    mask = np.load(mask_path).astype(np.uint8)
    image_shape_zyx = tuple(sitk.GetArrayFromImage(reference).shape)
    if tuple(mask.shape) != image_shape_zyx:
        raise RuntimeError(f"Mask shape {mask.shape} does not match image shape {image_shape_zyx}")
    mask_img = sitk.GetImageFromArray(mask)
    mask_img.CopyInformation(reference)
    return mask_img


def main() -> None:
    parser = argparse.ArgumentParser(description="Run de-identified DICOM and a predicted lesion mask through PyRadiomics and the locked RF.")
    parser.add_argument("--mask-path", type=Path, default=PROJECT_ROOT / "outputs" / "segmentation_smoke" / "internal" / "CAP_0001_strict_oof_predicted_mask.npy")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "lesion_radiomics_single_case")
    args = parser.parse_args()
    case_dicom_dir = dicom_dir("internal", "CAP", "CAP_0001")
    mask_path = args.mask_path.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_artifact = joblib.load(PROJECT_ROOT / "models" / "lesion_radiomics" / "final_random_forest.joblib")
    pipeline = model_artifact["pipeline"]
    features = list(model_artifact["features"])

    image, _series_id, _files = read_primary_dicom_series(case_dicom_dir)
    image_2mm = resample_hu_image_to_working_spacing(image)
    mask_img = mask_image_from_npy(mask_path, image_2mm)

    extractor = featureextractor.RadiomicsFeatureExtractor(
        str(PROJECT_ROOT / "configs" / "lesion_roi_pyradiomics_settings.yaml")
    )
    values = extractor.execute(image_2mm, mask_img)
    row = {name: float(values[name]) for name in features}
    prob = float(pipeline.predict_proba(pd.DataFrame([row], columns=features))[:, 1][0])

    feature_path = out_dir / "CAP_0001_single_case_final_features.csv"
    pred_path = out_dir / "CAP_0001_single_case_rf_prediction.json"
    pd.DataFrame([{"public_subject_id": "CAP_0001", **row}]).to_csv(feature_path, index=False)
    pred_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "public_subject_id": "CAP_0001",
                "feature_count": len(features),
                "probability_COP": prob,
                "mask_path": str(mask_path),
                "feature_path": str(feature_path),
                "prediction_path": str(pred_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pred_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
