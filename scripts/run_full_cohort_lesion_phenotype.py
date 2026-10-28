from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cap_cop.imaging.radiomics import lesion_phenotype as exact
from cap_cop.imaging.radiomics.mask_contract import validate_locked_mask_contract, validate_mask_metadata
from cap_cop.imaging.radiomics.per_slice_dicom_sampler import (
    read_axial_series,
    sample_normalized_roi,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "full_reproduction_20261010" / "lesion_phenotype"
CROSSWALK = PROJECT_ROOT / "data" / "public_subject_model_crosswalk.csv"
DEFAULT_SEGMENTATION_AUDIT = PROJECT_ROOT / "outputs" / "canonical_a_formal_20261011" / "segmentation" / "full_cohort_segmentation_replay_audit.csv"
LOCKED_FEATURES = PROJECT_ROOT / "models" / "stable_multirepresentation_ct" / "derived_inputs" / "06_LESION_PHENOTYPE_FEATURES.csv"
LOCKED_ROI_GEOMETRY = PROJECT_ROOT / "models" / "stable_multirepresentation_ct" / "derived_inputs" / "strict_lesion_roi_geometry_manifest.csv"
ROI_MATRIX = 144
ROI_SPACING_MM = 184.0 / ROI_MATRIX
HU_MIN = -1000.0
HU_MAX = 400.0
GEOMETRY_EXCEPTIONS = PROJECT_ROOT / "data" / "lesion_phenotype_geometry_exceptions.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_image_lps(folder: Path) -> sitk.Image:
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder)) or []
    if not series_ids:
        raise RuntimeError(f"No DICOM series in {folder}")
    selected = max(series_ids, key=lambda item: len(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), item)))
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), selected))
    return sitk.DICOMOrient(reader.Execute(), "LPS")


def resample_roi_from_native(image: sitk.Image, center_xyz: tuple[float, float, float]) -> np.ndarray:
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([ROI_MATRIX, ROI_MATRIX, ROI_MATRIX])
    resampler.SetOutputSpacing([ROI_SPACING_MM] * 3)
    resampler.SetOutputDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    resampler.SetOutputOrigin([center_xyz[index] - ((ROI_MATRIX - 1) / 2.0) * ROI_SPACING_MM for index in range(3)])
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(HU_MIN)
    array_hu = sitk.GetArrayFromImage(resampler.Execute(image)).astype(np.float32)
    normalized = np.clip((np.clip(array_hu, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN), 0.0, 1.0).astype(np.float32)
    return normalized[np.newaxis, ...]


def extract_subject(task: dict[str, Any]) -> dict[str, Any]:
    use_per_slice_sampler = bool(task.get("use_per_slice_sampler", False))
    mask = np.load(task["mask_path"]).astype(np.uint8)
    metadata = json.loads(Path(task["metadata_path"]).read_text(encoding="utf-8"))
    component_labels, component_count = ndimage.label(mask > 0, structure=np.ones((3, 3, 3), dtype=np.uint8))
    validate_mask_metadata(task["public_subject_id"], mask.shape, metadata)
    components = exact.component_table(mask, metadata)
    if task["cohort"] == "internal":
        validate_locked_mask_contract(
            task["public_subject_id"], Path(task["mask_path"]),
            component_count, task.get("locked_roi_centers", []),
            mask_shape=mask.shape, metadata=metadata, components=components,
        )
    image = None if use_per_slice_sampler else read_image_lps(Path(task["dicom_dir"]))
    native_slices = read_axial_series(Path(task["dicom_dir"])) if use_per_slice_sampler else None
    if component_count != len(components):
        raise RuntimeError(f"Component count mismatch for {task['public_subject_id']}")

    locked_centers = {
        int(row["predicted_component_rank"]): (
            float(row["center_x_mm"]),
            float(row["center_y_mm"]),
            float(row["center_z_mm"]),
        )
        for row in task.get("locked_roi_centers", [])
    }
    if task["cohort"] == "internal":
        expected_ranks = set(range(1, min(3, len(components)) + 1))
        if set(locked_centers) != expected_ranks:
            raise RuntimeError(
                f"Locked ROI geometry contract mismatch for {task['public_subject_id']}: "
                f"expected={sorted(expected_ranks)} observed={sorted(locked_centers)}"
            )

    component_features = []
    for component in components[:3]:
        rank = int(component["component_rank_by_volume"])
        center = locked_centers.get(
            rank,
            (
                float(component["bbox_center_x_mm"]),
                float(component["bbox_center_y_mm"]),
                float(component["bbox_center_z_mm"]),
            ),
        )
        if native_slices is not None:
            roi = sample_normalized_roi(native_slices, center, ROI_MATRIX, ROI_SPACING_MM)
        else:
            roi = resample_roi_from_native(image, center)
        lesion_mask = exact.roi_component_mask(component_labels, int(component["label"]), center, metadata)
        if not np.any(lesion_mask):
            raise RuntimeError(f"Empty mapped ROI mask for {task['public_subject_id']}")
        features = exact.extract_component_features(roi, lesion_mask)
        component_features.append(
            {
                "component_volume_ml_from_2mm_mask": float(component["volume_ml"]),
                **features,
            }
        )

    patient: dict[str, Any] = {
        "public_subject_id": task["public_subject_id"],
        "neutral_patient_id": task["segmentation_model_id"],
        "source_neutral_imaging_id": task["lesion_imaging_model_id"],
        "neutral_external_id": task["external_tensor_id"],
        "site": "Internal" if task["cohort"] == "internal" else "External",
        "group": ("I_" if task["cohort"] == "internal" else "E_") + task["diagnosis_label"],
        "label_binary": int(task["label_binary"]),
        "radiomics_missing_indicator": 1 if not component_features else 0,
        "predicted_component_count": int(component_count),
        "radiomics_valid_component_count": len(component_features),
        "radiomics_total_component_volume_ml": float(sum(row["component_volume_ml_from_2mm_mask"] for row in component_features)),
        "radiomics_largest_component_volume_ml": float(component_features[0]["component_volume_ml_from_2mm_mask"]) if component_features else np.nan,
        "radiomics_second_component_volume_ml": float(component_features[1]["component_volume_ml_from_2mm_mask"]) if len(component_features) > 1 else np.nan,
        "radiomics_third_component_volume_ml": float(component_features[2]["component_volume_ml_from_2mm_mask"]) if len(component_features) > 2 else np.nan,
    }
    total_volume = patient["radiomics_total_component_volume_ml"]
    patient["radiomics_largest_component_fraction"] = patient["radiomics_largest_component_volume_ml"] / total_volume if total_volume else np.nan
    if component_features:
        weights = np.asarray([row["component_volume_ml_from_2mm_mask"] for row in component_features], dtype=float)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights), dtype=float) / len(weights)
        names = sorted(set().union(*(set(row) for row in component_features)) - {"component_volume_ml_from_2mm_mask"})
        for name in names:
            values = np.asarray([float(row.get(name, np.nan)) for row in component_features], dtype=float)
            patient[f"largest_{name}"] = values[0]
            valid = np.isfinite(values)
            patient[f"vwm_{name}"] = float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid])) if valid.any() else np.nan
    return patient


def load_contract(segmentation_audit: Path) -> pd.DataFrame:
    crosswalk = pd.read_csv(CROSSWALK, dtype=str).fillna("")
    if not segmentation_audit.exists():
        raise RuntimeError(f"Lesion phenotype replay requires a segmentation audit: {segmentation_audit}")
    audit = pd.read_csv(segmentation_audit, dtype=str).fillna("")
    frame = crosswalk.merge(
        audit[["public_subject_id", "dicom_dir", "mask_path", "metadata_path", "status"]],
        on="public_subject_id",
        validate="one_to_one",
    )
    if len(frame) != 408 or not frame["status"].eq("PASS").all():
        raise RuntimeError("Expected 408 passing segmentation-backed phenotype inputs")
    if not GEOMETRY_EXCEPTIONS.is_file():
        raise RuntimeError("Authorised lesion phenotype geometry-exception contract is missing from data/")
    exceptions = json.loads(GEOMETRY_EXCEPTIONS.read_text(encoding="utf-8"))
    per_slice_subjects = set(exceptions["per_slice_dicom_physical_geometry_subjects"])
    if not per_slice_subjects.issubset(set(frame["public_subject_id"])):
        raise RuntimeError("Geometry exception contract contains unknown subjects")
    frame["use_per_slice_sampler"] = frame["public_subject_id"].isin(per_slice_subjects)
    return frame


def load_locked_roi_geometry() -> dict[str, list[dict[str, str]]]:
    if not LOCKED_ROI_GEOMETRY.exists():
        raise RuntimeError(f"Internal locked ROI geometry manifest is missing: {LOCKED_ROI_GEOMETRY}")
    table = pd.read_csv(LOCKED_ROI_GEOMETRY, dtype=str).fillna("")
    if len(table) != 1073 or table["public_subject_id"].nunique() != 375:
        raise RuntimeError(
            f"Internal locked ROI geometry manifest size invalid: rows={len(table)} "
            f"subjects={table['public_subject_id'].nunique()}"
        )
    if table.duplicated(["public_subject_id", "predicted_component_rank"]).any():
        raise RuntimeError("Internal locked ROI geometry manifest has duplicate subject/rank rows")
    return {
        str(public_id): group.to_dict("records")
        for public_id, group in table.groupby("public_subject_id", sort=False)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay locked handcrafted lesion phenotype features for all 408 subjects.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--refresh-subject", action="append", default=[])
    parser.add_argument("--segmentation-audit", type=Path, default=DEFAULT_SEGMENTATION_AUDIT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "full_cohort_lesion_phenotype_features.csv"
    contract = load_contract(args.segmentation_audit.resolve())
    locked_roi_geometry = load_locked_roi_geometry()
    existing = pd.read_csv(output_path, encoding="utf-8-sig") if output_path.exists() else pd.DataFrame()
    if args.refresh_subject and not existing.empty:
        requested = set(args.refresh_subject)
        unknown = requested - set(contract["public_subject_id"])
        if unknown:
            raise RuntimeError(f"Unknown refresh subjects: {sorted(unknown)}")
        existing = existing[~existing["public_subject_id"].isin(requested)].copy()
    done = set(existing["public_subject_id"].astype(str)) if "public_subject_id" in existing else set()
    tasks = []
    for row in contract.to_dict("records"):
        if row["public_subject_id"] in done:
            continue
        row["locked_roi_centers"] = locked_roi_geometry.get(row["public_subject_id"], []) if row["cohort"] == "internal" else []
        tasks.append(row)
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(extract_subject, task): task["public_subject_id"] for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 5 == 0 or index == len(tasks):
                combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
                combined.to_csv(output_path, index=False, encoding="utf-8-sig", lineterminator="\n")
                existing = combined
                rows = []
                print(f"lesion phenotype {index}/{len(tasks)} last={futures[future]}", flush=True)

    replay = pd.read_csv(output_path, encoding="utf-8-sig")
    locked = pd.read_csv(LOCKED_FEATURES)
    compare_columns = [
        column
        for column in locked.columns
        if column not in {"neutral_patient_id", "source_neutral_imaging_id", "neutral_external_id", "site", "group", "label_binary"}
        and column in replay.columns
    ]
    internal_replay = replay[replay["site"].eq("Internal")]
    internal_locked = locked[locked["site"].eq("Internal")]
    external_replay = replay[replay["site"].eq("External")]
    external_locked = locked[locked["site"].eq("External")]
    internal = internal_replay.merge(internal_locked, on="source_neutral_imaging_id", suffixes=("_replay", "_locked"), validate="one_to_one")
    external = external_replay.merge(external_locked, left_on="neutral_patient_id", right_on="neutral_patient_id", suffixes=("_replay", "_locked"), validate="one_to_one")

    def deltas(frame: pd.DataFrame) -> tuple[float, float]:
        values = np.concatenate([
            np.abs(pd.to_numeric(frame[f"{column}_replay"], errors="coerce").to_numpy(float) - pd.to_numeric(frame[f"{column}_locked"], errors="coerce").to_numpy(float))
            for column in compare_columns
        ])
        return float(np.nanmax(values)), float(np.nanmean(values))

    internal_max, internal_mean = deltas(internal)
    external_max, external_mean = deltas(external)
    summary = {
        "status": "PASS" if internal_max <= 1e-6 and external_max <= 1e-6 else "FAIL",
        "subjects": len(replay),
        "internal_subjects": len(internal_replay),
        "external_subjects": len(external_replay),
        "features_compared": len(compare_columns),
        "internal_max_abs_delta": internal_max,
        "internal_mean_abs_delta": internal_mean,
        "external_max_abs_delta": external_max,
        "external_mean_abs_delta": external_mean,
        "output_sha256": sha256_file(output_path),
    }
    (output_dir / "full_cohort_lesion_phenotype_replay_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
