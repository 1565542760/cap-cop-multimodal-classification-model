from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HU_MIN = -1000.0
WORKING_SPACING_XYZ = (2.0, 2.0, 2.0)


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_primary_dicom_series(dicom_dir: Path) -> sitk.Image:
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir)) or []
    if not series_ids:
        raise RuntimeError(f"No readable DICOM series found in {dicom_dir}")
    _series_id, files = max(
        ((sid, sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), sid)) for sid in series_ids),
        key=lambda item: len(item[1]),
    )
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    return reader.Execute()


def resample_hu_to_segmentation_grid(image: sitk.Image) -> sitk.Image:
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    size = np.asarray(image.GetSize(), dtype=int)
    origin = image.GetOrigin()
    direction = image.GetDirection()
    new_size = np.maximum(np.round(size * spacing / np.asarray(WORKING_SPACING_XYZ)).astype(int), 1)

    factor_xyz = np.asarray(WORKING_SPACING_XYZ, dtype=float) / spacing
    sigma_xyz = np.maximum((factor_xyz - 1.0) / 2.0, 0.0)
    array_hu = sitk.GetArrayFromImage(image).astype(np.float32)
    sigma_zyx = (float(sigma_xyz[2]), float(sigma_xyz[1]), float(sigma_xyz[0]))
    if max(sigma_zyx) > 0:
        array_hu = ndimage.gaussian_filter(array_hu, sigma=sigma_zyx, mode="nearest")
        image = sitk.GetImageFromArray(array_hu)
        image.SetSpacing(tuple(float(v) for v in spacing))
        image.SetOrigin(origin)
        image.SetDirection(direction)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(WORKING_SPACING_XYZ)
    resampler.SetSize([int(v) for v in new_size])
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(HU_MIN)
    return resampler.Execute(image)


def mask_image_from_npy(mask_path: Path, reference: sitk.Image) -> sitk.Image:
    mask = np.load(mask_path).astype(np.uint8)
    image_shape_zyx = tuple(sitk.GetArrayFromImage(reference).shape)
    if tuple(mask.shape) != image_shape_zyx:
        raise RuntimeError(f"Mask shape {mask.shape} does not match image shape {image_shape_zyx}")
    if int(mask.sum()) == 0:
        raise RuntimeError(f"Mask is empty: {mask_path}")
    mask_img = sitk.GetImageFromArray(mask)
    mask_img.CopyInformation(reference)
    return mask_img


def run(subject_id: str, dicom_dir: Path, mask_path: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_artifact = joblib.load(PROJECT_ROOT / "models" / "lesion_radiomics" / "final_random_forest.joblib")
    pipeline = model_artifact["pipeline"]
    features = list(model_artifact["features"])

    image = read_primary_dicom_series(dicom_dir)
    image_2mm = resample_hu_to_segmentation_grid(image)
    mask_img = mask_image_from_npy(mask_path, image_2mm)

    extractor = featureextractor.RadiomicsFeatureExtractor(
        str(PROJECT_ROOT / "configs" / "lesion_roi_pyradiomics_settings.yaml")
    )
    values = extractor.execute(image_2mm, mask_img)
    row = {name: float(values[name]) for name in features}
    probability = float(pipeline.predict_proba(pd.DataFrame([row], columns=features))[:, 1][0])

    feature_path = output_dir / f"{subject_id}_model_a_subset_final_features.csv"
    prediction_path = output_dir / f"{subject_id}_model_a_subset_rf_prediction.json"
    pd.DataFrame([{"public_subject_id": subject_id, **row}]).to_csv(feature_path, index=False)
    result = {
        "status": "passed",
        "public_subject_id": subject_id,
        "feature_count": len(features),
        "features": features,
        "probability_COP": probability,
        "dicom_dir": str(dicom_dir),
        "mask_path": str(mask_path),
        "feature_path": str(feature_path),
        "prediction_path": str(prediction_path),
    }
    prediction_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Model A subset DICOM+mask -> PyRadiomics -> RF probability.")
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--dicom-dir", required=True)
    parser.add_argument("--mask-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run(args.subject_id, resolve_path(args.dicom_dir), resolve_path(args.mask_path), resolve_path(args.output_dir))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
