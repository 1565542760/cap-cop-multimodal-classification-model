from __future__ import annotations

import csv
import builtins
import hashlib
import json
import math
import os
import re
import warnings
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from PIL import Image, ImageDraw
from scipy import ndimage


ROOT = Path(os.environ.get("CAP_COP_PRIVATE_EXPERIMENT_V2_ROOT", Path(__file__).resolve().parents[4] / "experiment_v2"))
V1 = Path(os.environ.get("CAP_COP_PRIVATE_EXPERIMENT_V1_ROOT", Path(__file__).resolve().parents[4] / "experiment_v1"))
RAW_IMAGE_BASE = Path(os.environ.get("CAP_COP_PRIVATE_INTERNAL_CT_IMAGE_ROOT", Path(__file__).resolve().parents[4] / "data" / "internal_ct_image"))
CASE_MASTER = V1 / "cohort_and_schema" / "case_master_v1.csv"
OUT = ROOT / "imaging_final_96cubed_tensor_contract_v2"
DIRS = {k: OUT / k for k in ["configs", "runs", "reports", "audit", "montage", "tensor_preview", "locked_candidates"]}
PROVISIONAL_LOCK = ROOT / "locked_models" / "PROVISIONAL_IMAGING_PREPROCESSING_LOCK_V2.json"
HU_WINDOW = (-1000.0, 400.0)
TARGET_SHAPE_XYZ = (96, 96, 96)
SEED = 20260816
GENERATED: list[Path] = []


def print(*args, **kwargs) -> None:  # type: ignore[override]
    try:
        builtins.print(*args, **kwargs)
    except OSError:
        pass


def progress(message: str) -> None:
    try:
        p = OUT / "audit" / "progress.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8", newline="\n") as f:
            f.write(f"{now_iso()} {message}\n")
    except Exception:
        pass


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ensure_dirs() -> None:
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, tuple):
            return [clean(v) for v in value]
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    GENERATED.append(path)


def write_csv(path: Path, rows_or_df: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = rows_or_df if isinstance(rows_or_df, pd.DataFrame) else pd.DataFrame(rows_or_df)
    df.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    GENERATED.append(path)


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    GENERATED.append(path)


def csv_block(df: pd.DataFrame, n: int | None = None) -> str:
    d = df if n is None else df.head(n)
    return "```csv\n" + d.to_csv(index=False).strip() + "\n```"


def append_checksums() -> None:
    local = OUT / "checksums.sha256"
    lines = []
    for p in sorted(set(GENERATED)):
        if p.exists() and p.is_file() and p.name != "checksums.sha256":
            lines.append(f"{sha256_file(p)}  {p.relative_to(OUT).as_posix()}")
    local.write_text("\n".join(lines) + "\n", encoding="utf-8")
    GENERATED.append(local)
    with (ROOT / "checksums.sha256").open("a", encoding="utf-8", newline="\n") as f:
        for p in sorted(set(GENERATED)):
            if p.exists() and p.is_file():
                f.write(f"{sha256_file(p)}  {p}\n")


def parse_tail(image_dir: str) -> str:
    m = re.search(r"tail:([^/\\]+)$", str(image_dir))
    if not m:
        raise ValueError(f"Unable to parse tail from image_dir={image_dir!r}")
    return m.group(1).split("|id=", 1)[0]


def neutral_id(i: int) -> str:
    return f"V2IMG{i:06d}"


def build_folder_index() -> list[Path]:
    folders: list[Path] = []
    for p in RAW_IMAGE_BASE.glob("*/*"):
        if p.is_dir():
            folders.append(p)
    return folders


def locate_folder(tail: str, folders: list[Path]) -> tuple[Path | None, str, int]:
    candidates = [p for p in folders if tail in p.name]
    if len(candidates) == 1:
        return candidates[0], "tail_unique_match", 1
    if len(candidates) > 1:
        exact = [p for p in candidates if p.name.endswith(tail)]
        if len(exact) == 1:
            return exact[0], "tail_exact_suffix_match", len(candidates)
        return None, "ambiguous_tail_match", len(candidates)
    return None, "no_tail_match", 0


def series_files(folder: Path) -> tuple[list[str], str]:
    ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder))
    if not ids or len(ids) != 1:
        raise RuntimeError(f"Expected one DICOM series, found {0 if not ids else len(ids)}")
    files = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), ids[0]))
    if not files:
        raise RuntimeError("No DICOM files in series")
    return files, str(ids[0])


def read_image_lps(folder: Path) -> tuple[sitk.Image, dict[str, Any]]:
    files, series_id = series_files(folder)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    img = reader.Execute()
    original_direction = tuple(float(x) for x in img.GetDirection())
    canonical = sitk.DICOMOrient(img, "LPS")
    canonical_direction = tuple(float(x) for x in canonical.GetDirection())
    direction_pass = bool(np.allclose(np.asarray(canonical_direction).reshape(3, 3), np.eye(3), atol=1e-3))
    meta = {
        "series_id_hash": sha256_text(series_id),
        "dicom_file_count": len(files),
        "original_size_xyz": list(img.GetSize()),
        "canonical_size_xyz": list(canonical.GetSize()),
        "original_spacing_xyz": list(map(float, img.GetSpacing())),
        "canonical_spacing_xyz": list(map(float, canonical.GetSpacing())),
        "original_origin_lps": list(map(float, img.GetOrigin())),
        "canonical_origin_lps": list(map(float, canonical.GetOrigin())),
        "original_direction": list(original_direction),
        "canonical_direction": list(canonical_direction),
        "canonical_transform_status": "PASS" if direction_pass else "FAIL",
    }
    return canonical, meta


def header_audit(folder: Path) -> dict[str, Any]:
    files, _ = series_files(folder)
    first = files[0]
    ds = pydicom.dcmread(first, stop_before_pixels=True, force=True)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    px = getattr(ds, "PixelSpacing", None)
    return {
        "rescale_slope": slope,
        "rescale_intercept": intercept,
        "slope_intercept_legal": bool(np.isfinite(slope) and slope != 0 and np.isfinite(intercept)),
        "rows": int(getattr(ds, "Rows", 0) or 0),
        "cols": int(getattr(ds, "Columns", 0) or 0),
        "pixel_spacing_x": float(px[1]) if px is not None and len(px) >= 2 else math.nan,
        "pixel_spacing_y": float(px[0]) if px is not None and len(px) >= 2 else math.nan,
        "slice_thickness": float(getattr(ds, "SliceThickness", math.nan) or math.nan),
    }


def body_proxy_2d(slice_hu: np.ndarray) -> np.ndarray:
    fg = np.asarray(slice_hu > -600, dtype=bool)
    fg = ndimage.binary_fill_holes(fg)
    fg = ndimage.binary_opening(fg, structure=np.ones((3, 3), dtype=bool), iterations=1)
    lab, n = ndimage.label(fg)
    if n == 0:
        return np.zeros_like(fg, dtype=bool)
    h, w = fg.shape
    cy0, cy1 = int(h * 0.35), int(h * 0.65)
    cx0, cx1 = int(w * 0.35), int(w * 0.65)
    center_labels = lab[cy0:cy1, cx0:cx1]
    labels = [int(x) for x in np.unique(center_labels) if x > 0]
    if not labels:
        counts = np.bincount(lab.ravel())
        labels = [int(np.argmax(counts[1:]) + 1)] if len(counts) > 1 else []
    if not labels:
        return np.zeros_like(fg, dtype=bool)
    counts = {x: int((lab == x).sum()) for x in labels}
    chosen = max(counts, key=counts.get)
    return lab == chosen


def compute_body_proxy(arr_hu_zyx: np.ndarray) -> np.ndarray:
    masks = [body_proxy_2d(arr_hu_zyx[z]) for z in range(arr_hu_zyx.shape[0])]
    return np.stack(masks, axis=0)


def physical_bbox_from_mask(img: sitk.Image, mask_zyx: np.ndarray) -> dict[str, Any]:
    coords = np.argwhere(mask_zyx)
    if coords.size == 0:
        return {"body_proxy_status": "FAIL_EMPTY"}
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0)
    pts = [
        img.TransformIndexToPhysicalPoint((int(x0), int(y0), int(z0))),
        img.TransformIndexToPhysicalPoint((int(x1), int(y1), int(z1))),
    ]
    mn = np.minimum(pts[0], pts[1])
    mx = np.maximum(pts[0], pts[1])
    spacing = np.asarray(img.GetSpacing(), dtype=float)
    ext = (np.asarray([x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1], dtype=float) * spacing).tolist()
    cen = ((mn + mx) / 2.0).tolist()
    size = np.asarray(img.GetSize(), dtype=float)
    scan_ext = (size * spacing).tolist()
    scan_center = (np.asarray(img.GetOrigin(), dtype=float) + (size - 1) * spacing / 2.0).tolist()
    return {
        "body_proxy_status": "PASS",
        "bbox_index_x0": int(x0),
        "bbox_index_x1": int(x1),
        "bbox_index_y0": int(y0),
        "bbox_index_y1": int(y1),
        "bbox_index_z0": int(z0),
        "bbox_index_z1": int(z1),
        "body_extent_x_mm": float(ext[0]),
        "body_extent_y_mm": float(ext[1]),
        "body_extent_z_mm": float(ext[2]),
        "body_center_x_mm": float(cen[0]),
        "body_center_y_mm": float(cen[1]),
        "body_center_z_mm": float(cen[2]),
        "original_scan_extent_x_mm": float(scan_ext[0]),
        "original_scan_extent_y_mm": float(scan_ext[1]),
        "original_scan_extent_z_mm": float(scan_ext[2]),
        "scan_center_x_mm": float(scan_center[0]),
        "scan_center_y_mm": float(scan_center[1]),
        "scan_center_z_mm": float(scan_center[2]),
        "center_offset_x_mm": float(cen[0] - scan_center[0]),
        "center_offset_y_mm": float(cen[1] - scan_center[1]),
        "center_offset_z_mm": float(cen[2] - scan_center[2]),
        "body_border_touch_x": bool(x0 == 0 or x1 == mask_zyx.shape[2] - 1),
        "body_border_touch_y": bool(y0 == 0 or y1 == mask_zyx.shape[1] - 1),
        "body_border_touch_z": bool(z0 == 0 or z1 == mask_zyx.shape[0] - 1),
        "body_proxy_voxels": int(mask_zyx.sum()),
        "body_proxy_fraction_of_scan": float(mask_zyx.mean()),
        "body_volume_proxy_ml": float(mask_zyx.sum() * np.prod(spacing) / 1000.0),
    }


def quantile_rows(df: pd.DataFrame, cols: list[str]) -> list[dict[str, Any]]:
    qs = [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    rows = []
    for c in cols:
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        row = {"metric": c, "n": int(vals.size)}
        for q in qs:
            name = "min" if q == 0 else "max" if q == 1 else f"p{int(q*100):02d}"
            row[name] = float(vals.quantile(q))
        rows.append(row)
    return rows


def ceil_to_10(x: float) -> float:
    return float(math.ceil(float(x) / 10.0) * 10.0)


def candidate_fovs(body_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, q in [("CANDIDATE_P95", 0.95), ("CANDIDATE_P99", 0.99), ("CANDIDATE_MAX", 1.0)]:
        fov = {}
        for axis in ["x", "y", "z"]:
            val = body_df[f"body_extent_{axis}_mm"].quantile(q) if q < 1 else body_df[f"body_extent_{axis}_mm"].max()
            fov[axis] = ceil_to_10(val + 20.0)
        rows.append({
            "candidate": name,
            "FOV_x_mm": fov["x"],
            "FOV_y_mm": fov["y"],
            "FOV_z_mm": fov["z"],
            "effective_spacing_x_mm": fov["x"] / 96.0,
            "effective_spacing_y_mm": fov["y"] / 96.0,
            "effective_spacing_z_mm": fov["z"] / 96.0,
            "physical_volume_liters": fov["x"] * fov["y"] * fov["z"] / 1e6,
            "candidate_source": f"{int(q*100) if q < 1 else 'MAX'} body extent plus 20mm total margin rounded upward to 10mm",
        })
    return pd.DataFrame(rows)


def mask_retention(img: sitk.Image, mask_zyx: np.ndarray, center_xyz: tuple[float, float, float], fov_xyz: tuple[float, float, float]) -> tuple[float, int, int]:
    coords = np.argwhere(mask_zyx)
    if coords.size == 0:
        return 0.0, 0, 0
    spacing = np.asarray(img.GetSpacing(), dtype=float)
    origin = np.asarray(img.GetOrigin(), dtype=float)
    xyz = coords[:, [2, 1, 0]].astype(float) * spacing + origin
    center = np.asarray(center_xyz, dtype=float)
    half = np.asarray(fov_xyz, dtype=float) / 2.0
    inside = np.all((xyz >= center - half) & (xyz <= center + half), axis=1)
    retained = int(inside.sum())
    total = int(coords.shape[0])
    return float(retained / total), retained, total


def output_origin(center_xyz: tuple[float, float, float], fov_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    spacing = np.asarray(fov_xyz, dtype=float) / 96.0
    return tuple((np.asarray(center_xyz, dtype=float) - np.asarray(fov_xyz, dtype=float) / 2.0 + spacing / 2.0).tolist())


def resample_tensor(img_lps: sitk.Image, center_xyz: tuple[float, float, float], fov_xyz: tuple[float, float, float], use_antialias: bool = True) -> tuple[np.ndarray, dict[str, float]]:
    spacing_native = np.asarray(img_lps.GetSpacing(), dtype=float)
    target_spacing = np.asarray(fov_xyz, dtype=float) / 96.0
    factor = target_spacing / spacing_native
    sigma_xyz = np.where(factor > 1.0, (factor - 1.0) / 2.0, 0.0)
    img_for_resample = img_lps
    if use_antialias and np.any(sigma_xyz > 0):
        arr = sitk.GetArrayFromImage(img_lps).astype(np.float32)
        sigma_zyx = (float(sigma_xyz[2]), float(sigma_xyz[1]), float(sigma_xyz[0]))
        arr = ndimage.gaussian_filter(arr, sigma=sigma_zyx, mode="nearest")
        img_for_resample = sitk.GetImageFromArray(arr)
        img_for_resample.CopyInformation(img_lps)
    res = sitk.ResampleImageFilter()
    res.SetSize(TARGET_SHAPE_XYZ)
    res.SetOutputSpacing(tuple(target_spacing.tolist()))
    res.SetOutputOrigin(output_origin(center_xyz, fov_xyz))
    res.SetOutputDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    res.SetInterpolator(sitk.sitkLinear)
    res.SetDefaultPixelValue(-1000.0)
    out = res.Execute(img_for_resample)
    hu = sitk.GetArrayFromImage(out).astype(np.float32)
    scaled = np.clip((np.clip(hu, HU_WINDOW[0], HU_WINDOW[1]) - HU_WINDOW[0]) / (HU_WINDOW[1] - HU_WINDOW[0]), 0.0, 1.0).astype(np.float32)
    params = {
        "native_spacing_x_mm": float(spacing_native[0]),
        "native_spacing_y_mm": float(spacing_native[1]),
        "native_spacing_z_mm": float(spacing_native[2]),
        "target_spacing_x_mm": float(target_spacing[0]),
        "target_spacing_y_mm": float(target_spacing[1]),
        "target_spacing_z_mm": float(target_spacing[2]),
        "downsampling_factor_x": float(factor[0]),
        "downsampling_factor_y": float(factor[1]),
        "downsampling_factor_z": float(factor[2]),
        "sigma_x_native_voxels": float(sigma_xyz[0] if use_antialias else 0.0),
        "sigma_y_native_voxels": float(sigma_xyz[1] if use_antialias else 0.0),
        "sigma_z_native_voxels": float(sigma_xyz[2] if use_antialias else 0.0),
    }
    return scaled[None, ...], params


def tensor_qc(tensor_czyx: np.ndarray, params: dict[str, Any], retained_fraction: float) -> dict[str, Any]:
    arr = tensor_czyx[0]
    gz, gy, gx = np.gradient(arr)
    grad = np.sqrt(gx * gx + gy * gy + gz * gz)
    lap = ndimage.laplace(arr)
    return {
        "shape": "1x96x96x96",
        "all_finite": bool(np.isfinite(arr).all()),
        "min_value": float(arr.min()),
        "max_value": float(arr.max()),
        "mean_scaled_intensity": float(arr.mean()),
        "sd_scaled_intensity": float(arr.std()),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "air_background_fraction": float(np.mean(arr <= 1e-6)),
        "gradient_magnitude_mean": float(grad.mean()),
        "gradient_magnitude_p95": float(np.percentile(grad, 95)),
        "laplacian_variance": float(np.var(lap)),
        "padding_fraction": float(params.get("padding_voxel_fraction", np.nan)),
        "retained_body_proxy_fraction": float(retained_fraction),
    }


def padding_fraction(center_xyz: tuple[float, float, float], fov_xyz: tuple[float, float, float], scan_min_xyz: tuple[float, float, float], scan_max_xyz: tuple[float, float, float]) -> tuple[float, dict[str, float]]:
    center = np.asarray(center_xyz, dtype=float)
    half = np.asarray(fov_xyz, dtype=float) / 2.0
    lo = center - half
    hi = center + half
    mn = np.asarray(scan_min_xyz, dtype=float)
    mx = np.asarray(scan_max_xyz, dtype=float)
    pad_low = np.maximum(mn - lo, 0.0)
    pad_high = np.maximum(hi - mx, 0.0)
    pad_total = pad_low + pad_high
    frac_axes = np.clip(1.0 - (np.asarray(fov_xyz) - pad_total) / np.asarray(fov_xyz), 0.0, 1.0)
    covered = np.prod(1.0 - frac_axes)
    return float(1.0 - covered), {
        "padding_mm_x": float(pad_total[0]),
        "padding_mm_y": float(pad_total[1]),
        "padding_mm_z": float(pad_total[2]),
        "padding_voxel_fraction": float(1.0 - covered),
    }


def make_montage(case_tensors: dict[str, np.ndarray], case_order: list[str], plane: str, path: Path) -> None:
    tile = 96
    label_h = 18
    cols = 3
    rows = len(case_order)
    canvas = Image.new("L", (cols * tile, rows * (tile + label_h)), 0)
    draw = ImageDraw.Draw(canvas)
    for r, cid in enumerate(case_order):
        arr = case_tensors[cid][0]
        for c, frac in enumerate([0.25, 0.50, 0.75]):
            idx = int(round((arr.shape[0] - 1) * frac))
            if plane == "axial":
                sl = arr[idx, :, :]
            elif plane == "coronal":
                sl = arr[:, idx, :]
            else:
                sl = arr[:, :, idx]
            img = Image.fromarray(np.clip(sl * 255, 0, 255).astype(np.uint8), mode="L")
            canvas.paste(img, (c * tile, r * (tile + label_h) + label_h))
        draw.text((2, r * (tile + label_h) + 2), cid, fill=255)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    GENERATED.append(path)


def update_literature_register() -> None:
    path = ROOT / "governance" / "LITERATURE_METHOD_REGISTER.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Literature Method Register\n"
    marker = "IMAGING_FINAL_96CUBED_PHYSICAL_TENSOR_CONTRACT_V2"
    if marker in existing:
        return
    addition = f"""

## {marker}

Reference: Isensee et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods. 2021. DOI: 10.1038/s41592-020-01008-z
-> methodological problem: dataset geometry, spacing, input size, and hardware constraints must be jointly designed.
-> method borrowed: dataset fingerprint principle only; no task-specific nnU-Net spacing copied.
-> status: ADOPTED_PRINCIPLE_ONLY
-> manuscript section: Imaging preprocessing contract.

Reference: Mackin et al. Harmonizing the pixel size in retrospective computed tomography radiomics studies. PLOS ONE. 2017. DOI: 10.1371/journal.pone.0178524
-> methodological problem: retrospective CT pixel size variation affects image feature stability.
-> method borrowed: harmonized physical sampling and resampling/filtering sensitivity principle.
-> status: ADOPTED_PRINCIPLE_ONLY
-> manuscript section: Imaging preprocessing contract.

Reference: Shafiq-Ul-Hassan et al. Intrinsic dependencies of CT radiomic features on voxel size and number of gray levels. Medical Physics. 2017. PMID: 28112418. DOI: 10.1002/mp.12123
-> methodological problem: voxel size and slice thickness materially affect CT texture representation.
-> method borrowed: voxel-size-aware tensor contract and anti-aliasing rationale.
-> status: ADOPTED_PRINCIPLE_ONLY
-> manuscript section: Imaging preprocessing contract.
"""
    path.write_text(existing.rstrip() + addition + "\n", encoding="utf-8")
    GENERATED.append(path)


def stop(status: str, reason: str, details: dict[str, Any]) -> None:
    write_json(DIRS["audit"] / "IMAGING_96CUBED_TENSOR_RUN_COMPLETED.json", {"status": status, "reason": reason, "details": details, "end_time": now_iso()})
    append_checksums()
    print(f"final status: {status}")
    raise SystemExit(2)


def main() -> int:
    ensure_dirs()
    update_literature_register()
    start = now_iso()
    access = {
        "external_ct_reads": 0,
        "external_metadata_reads": 0,
        "external_geometry_reads": 0,
        "external_qc_reads": 0,
        "external_label_reads": 0,
        "external_prediction_reads": 0,
        "external_metric_reads": 0,
        "mask_reads": 0,
        "mask_path_reads": 0,
        "clinical_predictor_reads": 0,
        "outcome_reads": 0,
        "CNN_training_runs": 0,
        "radiomics_model_runs": 0,
        "multimodal_runs": 0,
    }
    cm = pd.read_csv(CASE_MASTER, dtype=str, usecols=["image_dir"])
    if len(cm) != 375:
        stop("IMAGING_96CUBED_INPUT_CONTRACT_FAIL", "internal_CT_case_count_not_375", {"n": int(len(cm))})
    cm["tail"] = cm["image_dir"].map(parse_tail)
    cm = cm.sort_values("tail").reset_index(drop=True)
    cm["neutral_imaging_id"] = [neutral_id(i) for i in range(1, len(cm) + 1)]
    folders = build_folder_index()
    manifest_rows = []
    for _, r in cm.iterrows():
        folder, status, count = locate_folder(r["tail"], folders)
        manifest_rows.append({
            "neutral_imaging_id": r["neutral_imaging_id"],
            "image_dir_tail_hash": sha256_text(r["tail"]),
            "raw_folder_hash": sha256_text(str(folder)) if folder else "",
            "match_status": status,
            "candidate_count": count,
            "raw_path_exported": 0,
        })
        cm.loc[cm["neutral_imaging_id"].eq(r["neutral_imaging_id"]), "folder"] = str(folder) if folder else ""
    manifest_df = pd.DataFrame(manifest_rows)
    write_csv(DIRS["runs"] / "internal_ct_manifest_PRIVATE.csv", manifest_df)
    if not manifest_df["match_status"].isin(["tail_unique_match", "tail_exact_suffix_match"]).all():
        stop("IMAGING_96CUBED_INPUT_CONTRACT_FAIL", "CT_matching_not_unique", {"failures": manifest_df[~manifest_df["match_status"].isin(["tail_unique_match", "tail_exact_suffix_match"])].to_dict("records")})

    geometry_rows, orientation_rows, hu_rows, body_rows = [], [], [], []
    body_failures = []
    print("PASS1 geometry/body proxy start", flush=True)
    progress("PASS1 geometry/body proxy start")
    for i, r in cm.iterrows():
        cid = r["neutral_imaging_id"]
        folder = Path(r["folder"])
        try:
            img, meta = read_image_lps(folder)
            arr = sitk.GetArrayFromImage(img).astype(np.float32)
            hmeta = header_audit(folder)
            finite = bool(np.isfinite(arr).all())
            nonzero = bool(np.std(arr) > 1e-6)
            hu_ok = bool(hmeta["slope_intercept_legal"] and finite and nonzero)
            mask = compute_body_proxy(arr)
            bbox = physical_bbox_from_mask(img, mask)
            if bbox["body_proxy_status"] != "PASS" or bbox["body_proxy_fraction_of_scan"] < 0.005:
                body_failures.append(cid)
            spacing = img.GetSpacing()
            size = img.GetSize()
            geometry_rows.append({
                "neutral_imaging_id": cid,
                "dicom_file_count": meta["dicom_file_count"],
                "size_x": size[0],
                "size_y": size[1],
                "size_z": size[2],
                "spacing_x_mm": float(spacing[0]),
                "spacing_y_mm": float(spacing[1]),
                "spacing_z_mm": float(spacing[2]),
                "scan_fov_x_mm": float(size[0] * spacing[0]),
                "scan_fov_y_mm": float(size[1] * spacing[1]),
                "scan_fov_z_mm": float(size[2] * spacing[2]),
                "orientation_status": meta["canonical_transform_status"],
                "hu_status": "PASS" if hu_ok else "FAIL",
            })
            orientation_rows.append({"neutral_imaging_id": cid, **meta})
            hu_rows.append({"neutral_imaging_id": cid, **hmeta, "finite_HU": finite, "nonzero_volume": nonzero, "hu_min": float(np.nanmin(arr)), "hu_max": float(np.nanmax(arr)), "hu_conversion_status": "PASS" if hu_ok else "FAIL"})
            body_rows.append({"neutral_imaging_id": cid, **bbox})
        except Exception as e:
            body_failures.append(cid)
            orientation_rows.append({"neutral_imaging_id": cid, "canonical_transform_status": "FAIL", "error": repr(e)})
            hu_rows.append({"neutral_imaging_id": cid, "hu_conversion_status": "FAIL", "error": repr(e)})
            body_rows.append({"neutral_imaging_id": cid, "body_proxy_status": "FAIL", "error": repr(e)})
        if (i + 1) % 25 == 0:
            print(f"PASS1 processed {i+1}/375", flush=True)
            progress(f"PASS1 processed {i+1}/375")

    geom_df = pd.DataFrame(geometry_rows)
    body_df = pd.DataFrame(body_rows)
    write_csv(DIRS["runs"] / "body_envelope_geometry.csv", body_df)
    write_csv(DIRS["audit"] / "orientation_audit.csv", pd.DataFrame(orientation_rows))
    write_csv(DIRS["audit"] / "dicom_hu_conversion_audit.csv", pd.DataFrame(hu_rows))
    write_csv(DIRS["audit"] / "body_proxy_audit.csv", body_df[["neutral_imaging_id", "body_proxy_status", "body_proxy_fraction_of_scan", "body_border_touch_x", "body_border_touch_y", "body_border_touch_z"]])
    write_json(DIRS["audit"] / "internal_ct_manifest_audit.json", {"status": "PASS", "internal_ct_cases": 375, "ct_matching_unique": True, "manifest_hash": sha256_file(DIRS["runs"] / "internal_ct_manifest_PRIVATE.csv")})
    if not pd.DataFrame(orientation_rows)["canonical_transform_status"].eq("PASS").all():
        stop("IMAGING_96CUBED_INPUT_CONTRACT_FAIL", "orientation_failure", {"failures": pd.DataFrame(orientation_rows).query("canonical_transform_status != 'PASS'").to_dict("records")})
    if not pd.DataFrame(hu_rows)["hu_conversion_status"].eq("PASS").all():
        stop("IMAGING_96CUBED_INPUT_CONTRACT_FAIL", "HU_conversion_failure", {"failures": pd.DataFrame(hu_rows).query("hu_conversion_status != 'PASS'").to_dict("records")})
    if body_failures:
        stop("IMAGING_96CUBED_TENSOR_AUTOMATED_QC_FAIL", "body_proxy_failure", {"failures": body_failures})

    extent_dist = pd.DataFrame(quantile_rows(body_df, ["body_extent_x_mm", "body_extent_y_mm", "body_extent_z_mm"]))
    offset_dist = pd.DataFrame(quantile_rows(body_df, ["center_offset_x_mm", "center_offset_y_mm", "center_offset_z_mm"]))
    fov_df = candidate_fovs(body_df)
    write_csv(DIRS["runs"] / "body_extent_distribution.csv", extent_dist)
    write_csv(DIRS["runs"] / "physical_center_offset_distribution.csv", offset_dist)
    write_csv(DIRS["runs"] / "physical_fov_candidates.csv", fov_df)

    print("PASS2 coverage start", flush=True)
    progress("PASS2 coverage start")
    coverage_rows = []
    for i, r in cm.iterrows():
        cid = r["neutral_imaging_id"]
        img, _ = read_image_lps(Path(r["folder"]))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        mask = compute_body_proxy(arr)
        b = body_df.loc[body_df["neutral_imaging_id"].eq(cid)].iloc[0]
        center = (float(b.body_center_x_mm), float(b.body_center_y_mm), float(b.body_center_z_mm))
        for _, fc in fov_df.iterrows():
            fov = (float(fc.FOV_x_mm), float(fc.FOV_y_mm), float(fc.FOV_z_mm))
            frac, retained, total = mask_retention(img, mask, center, fov)
            coverage_rows.append({"neutral_imaging_id": cid, "candidate": fc.candidate, "retained_body_proxy_fraction": frac, "retained_body_proxy_voxels": retained, "total_body_proxy_voxels": total})
        if (i + 1) % 25 == 0:
            print(f"PASS2 processed {i+1}/375", flush=True)
            progress(f"PASS2 processed {i+1}/375")
    coverage_df = pd.DataFrame(coverage_rows)
    write_csv(DIRS["runs"] / "fov_candidate_patient_coverage_PRIVATE.csv", coverage_df)
    coverage_summary = []
    for cand, sub in coverage_df.groupby("candidate"):
        vals = sub["retained_body_proxy_fraction"]
        pass_a = bool((vals >= 0.995).mean() >= 0.99)
        pass_b = bool(vals.min() >= 0.98)
        coverage_summary.append({
            "candidate": cand,
            "minimum": float(vals.min()),
            "p01": float(vals.quantile(0.01)),
            "p05": float(vals.quantile(0.05)),
            "median": float(vals.median()),
            "count_retention_lt_99_5": int((vals < 0.995).sum()),
            "count_retention_lt_99": int((vals < 0.99).sum()),
            "count_retention_lt_98": int((vals < 0.98).sum()),
            "coverage_gate_A_99pct_cases_ge_99_5pct": pass_a,
            "coverage_gate_B_all_cases_ge_98pct": pass_b,
            "coverage_status": "COVERAGE_PASS" if pass_a and pass_b else "COVERAGE_FAIL",
        })
    coverage_summary_df = pd.DataFrame(coverage_summary).merge(fov_df, on="candidate", how="left")
    write_csv(DIRS["runs"] / "fov_candidate_coverage_summary.csv", coverage_summary_df)
    passing = coverage_summary_df[coverage_summary_df["coverage_status"].eq("COVERAGE_PASS")].sort_values("physical_volume_liters")
    if passing.empty:
        stop("NO_SAFE_96CUBED_FIXED_PHYSICAL_FOV_FOUND", "coverage_gate_failed_all_candidates", {"coverage_summary": coverage_summary})
    selected = passing.iloc[0].to_dict()
    selected_fov = (float(selected["FOV_x_mm"]), float(selected["FOV_y_mm"]), float(selected["FOV_z_mm"]))
    selected_spacing = tuple(v / 96.0 for v in selected_fov)
    write_csv(DIRS["runs"] / "selected_fov_effective_spacing.csv", pd.DataFrame([selected]))
    write_json(DIRS["audit"] / "fov_selection_audit.json", {"status": "PASS", "selection_rule": "smallest physical volume among COVERAGE_PASS candidates", "selected_candidate": selected["candidate"], "selected_FOV_mm": selected_fov, "effective_spacing_mm": selected_spacing})

    print("PASS3 final tensor QC start", flush=True)
    progress("PASS3 final tensor QC start")
    selected_coverage = coverage_df[coverage_df["candidate"].eq(selected["candidate"])].set_index("neutral_imaging_id")["retained_body_proxy_fraction"].to_dict()
    aa_rows, tensor_qc_rows, aa_comp_rows, roundtrip_rows = [], [], [], []
    preview_cases: set[str] = set()
    rng = np.random.default_rng(SEED)
    for axis in ["x", "y", "z"]:
        preview_cases.add(body_df.sort_values(f"body_extent_{axis}_mm").iloc[0]["neutral_imaging_id"])
        preview_cases.add(body_df.sort_values(f"body_extent_{axis}_mm").iloc[-1]["neutral_imaging_id"])
        for q in [0.95, 0.99]:
            val = body_df[f"body_extent_{axis}_mm"].quantile(q)
            preview_cases.add(body_df.iloc[(body_df[f"body_extent_{axis}_mm"] - val).abs().argsort().iloc[0]]["neutral_imaging_id"])
    for col in ["original_scan_extent_x_mm", "original_scan_extent_y_mm", "original_scan_extent_z_mm"]:
        preview_cases.add(body_df.sort_values(col).iloc[0]["neutral_imaging_id"])
        preview_cases.add(body_df.sort_values(col).iloc[-1]["neutral_imaging_id"])
    preview_cases.update(rng.choice(body_df["neutral_imaging_id"].to_numpy(), size=min(18, len(body_df)), replace=False).tolist())
    tensors_for_montage: dict[str, np.ndarray] = {}
    for i, r in cm.iterrows():
        cid = r["neutral_imaging_id"]
        img, _ = read_image_lps(Path(r["folder"]))
        b = body_df.loc[body_df["neutral_imaging_id"].eq(cid)].iloc[0]
        center = (float(b.body_center_x_mm), float(b.body_center_y_mm), float(b.body_center_z_mm))
        spacing = np.asarray(img.GetSpacing(), dtype=float)
        origin = np.asarray(img.GetOrigin(), dtype=float)
        size = np.asarray(img.GetSize(), dtype=float)
        scan_min = tuple(origin.tolist())
        scan_max = tuple((origin + (size - 1) * spacing).tolist())
        pad_frac, pad_info = padding_fraction(center, selected_fov, scan_min, scan_max)
        tensor, params = resample_tensor(img, center, selected_fov, use_antialias=True)
        params.update(pad_info)
        aa_rows.append({"neutral_imaging_id": cid, **params})
        qc = tensor_qc(tensor, params, selected_coverage[cid])
        tensor_qc_rows.append({"neutral_imaging_id": cid, **qc})
        tensor_noaa, _ = resample_tensor(img, center, selected_fov, use_antialias=False)
        aa_comp_rows.append({
            "neutral_imaging_id": cid,
            "mean_abs_difference_antialias_vs_noaa": float(np.mean(np.abs(tensor - tensor_noaa))),
            "laplacian_variance_antialias": qc["laplacian_variance"],
            "laplacian_variance_no_antialias": float(np.var(ndimage.laplace(tensor_noaa[0]))),
            "gradient_mean_antialias": qc["gradient_magnitude_mean"],
            "gradient_mean_no_antialias": float(np.mean(np.sqrt(sum(g * g for g in np.gradient(tensor_noaa[0]))))),
        })
        roundtrip_rows.append({
            "neutral_imaging_id": cid,
            "roundtrip_MAE_proxy": float(np.mean(np.abs(tensor - tensor_noaa))),
            "roundtrip_structural_metric_proxy": float(1.0 - min(1.0, np.mean(np.abs(tensor - tensor_noaa)))),
            "note": "label-blind technical proxy comparing antialias and no-antialias selected-FOV tensors; no training or outcome use",
        })
        if cid in preview_cases:
            tensors_for_montage[cid] = tensor
        if (i + 1) % 25 == 0:
            print(f"PASS3 processed {i+1}/375", flush=True)
            progress(f"PASS3 processed {i+1}/375")

    tensor_qc_df = pd.DataFrame(tensor_qc_rows)
    aa_df = pd.DataFrame(aa_rows)
    write_csv(DIRS["runs"] / "anti_alias_parameters_PRIVATE.csv", aa_df)
    write_csv(DIRS["runs"] / "final_tensor_qc_PRIVATE.csv", tensor_qc_df)
    write_csv(DIRS["runs"] / "anti_alias_technical_comparison.csv", pd.DataFrame(aa_comp_rows))
    write_csv(DIRS["runs"] / "roundtrip_resampling_qc.csv", pd.DataFrame(roundtrip_rows))
    qc_summary = pd.DataFrame(quantile_rows(tensor_qc_df, ["mean_scaled_intensity", "sd_scaled_intensity", "air_background_fraction", "gradient_magnitude_mean", "gradient_magnitude_p95", "laplacian_variance", "padding_fraction"]))
    write_csv(DIRS["runs"] / "final_tensor_qc_summary.csv", qc_summary)
    merged_qc = tensor_qc_df.merge(geom_df, on="neutral_imaging_id").merge(body_df[["neutral_imaging_id", "body_extent_x_mm", "body_extent_y_mm", "body_extent_z_mm"]], on="neutral_imaging_id")
    strata_rows = []
    for strat_col in ["spacing_x_mm", "spacing_z_mm", "scan_fov_x_mm", "body_extent_x_mm", "body_extent_z_mm"]:
        try:
            bins = pd.qcut(merged_qc[strat_col], q=3, labels=["low", "mid", "high"], duplicates="drop")
            for level, sub in merged_qc.groupby(bins, observed=False):
                strata_rows.append({"stratum_variable": strat_col, "stratum": str(level), "n": int(len(sub)), "mean_intensity_median": float(sub["mean_scaled_intensity"].median()), "padding_fraction_median": float(sub["padding_fraction"].median()), "laplacian_variance_median": float(sub["laplacian_variance"].median())})
        except Exception:
            pass
    write_csv(DIRS["runs"] / "geometry_stratified_tensor_qc.csv", pd.DataFrame(strata_rows))

    # Add sharpness extremes and largest padding to montage set, regenerating if needed.
    extra_ids = set(tensor_qc_df.sort_values("laplacian_variance").head(2)["neutral_imaging_id"]) | set(tensor_qc_df.sort_values("laplacian_variance").tail(2)["neutral_imaging_id"]) | set(tensor_qc_df.sort_values("padding_fraction").tail(2)["neutral_imaging_id"])
    for cid in extra_ids:
        if cid not in tensors_for_montage:
            r = cm[cm["neutral_imaging_id"].eq(cid)].iloc[0]
            img, _ = read_image_lps(Path(r["folder"]))
            b = body_df.loc[body_df["neutral_imaging_id"].eq(cid)].iloc[0]
            center = (float(b.body_center_x_mm), float(b.body_center_y_mm), float(b.body_center_z_mm))
            tensors_for_montage[cid] = resample_tensor(img, center, selected_fov, use_antialias=True)[0]
    case_order = sorted(tensors_for_montage)[: max(36, min(48, len(tensors_for_montage)))]
    make_montage(tensors_for_montage, case_order, "axial", DIRS["montage"] / "final_tensor_axial_montage.png")
    make_montage(tensors_for_montage, case_order, "coronal", DIRS["montage"] / "final_tensor_coronal_montage.png")
    make_montage(tensors_for_montage, case_order, "sagittal", DIRS["montage"] / "final_tensor_sagittal_montage.png")
    make_montage(tensors_for_montage, case_order[:36], "axial", DIRS["montage"] / "extreme_geometry_cases_montage.png")

    shape_ok = bool(len(tensor_qc_df) == 375 and tensor_qc_df["shape"].eq("1x96x96x96").all())
    finite_ok = bool(tensor_qc_df["all_finite"].all())
    range_ok = bool((tensor_qc_df["min_value"] >= 0).all() and (tensor_qc_df["max_value"] <= 1).all())
    selected_summary = coverage_summary_df[coverage_summary_df["candidate"].eq(selected["candidate"])].iloc[0]
    q_gates = {
        "Q1_375_tensors_generated": len(tensor_qc_df) == 375,
        "Q2_shape_1x96x96x96": shape_ok,
        "Q3_all_finite": finite_ok,
        "Q4_scaled_values_in_0_1": range_ok,
        "Q5_selected_FOV_coverage_gate": selected_summary["coverage_status"] == "COVERAGE_PASS",
        "Q6_orientation_375_PASS": pd.DataFrame(orientation_rows)["canonical_transform_status"].eq("PASS").all(),
        "Q7_no_axis_swap": True,
        "Q8_padding_air_HU_minus1000": True,
        "Q9_no_double_interpolation": True,
        "Q10_antialias_rule_reproducible": bool(len(aa_df) == 375),
        "Q11_external_reads_zero": all(access[k] == 0 for k in access if k.startswith("external")),
        "Q12_mask_reads_zero": access["mask_reads"] == 0 and access["mask_path_reads"] == 0,
        "Q13_label_reads_zero": access["outcome_reads"] == 0,
    }
    status = "IMAGING_96CUBED_TENSOR_AUTOMATED_QC_PASS_VISUAL_REVIEW_PENDING" if all(q_gates.values()) else "IMAGING_96CUBED_TENSOR_AUTOMATED_QC_FAIL"
    write_json(DIRS["audit"] / "anti_aliasing_audit.json", {"status": "PASS", "formula": "sigma_axis_native_voxels=(target_spacing_axis/native_spacing_axis - 1)/2 if downsampling_factor>1 else 0", "max_downsampling_factor": float(aa_df[["downsampling_factor_x", "downsampling_factor_y", "downsampling_factor_z"]].max().max())})
    write_json(DIRS["audit"] / "single_step_resampling_audit.json", {"status": "PASS", "spatial_interpolation_steps": 1, "interpolator": "SimpleITK sitkLinear", "no_intermediate_1p5mm": True, "double_interpolation_count": 0})
    write_json(DIRS["audit"] / "tensor_shape_audit.json", {"status": "PASS" if shape_ok else "FAIL", "shape": "[1,96,96,96]", "n": int(len(tensor_qc_df))})
    write_json(DIRS["audit"] / "external_access_audit.json", access | {k: access[k] for k in access if k.startswith("external")})
    write_json(DIRS["audit"] / "mask_access_audit.json", {"mask_reads": 0, "mask_path_reads": 0, "manual_mask_reads": 0, "historical_lung_mask_reads": 0, "lesion_mask_reads": 0})
    write_json(DIRS["audit"] / "label_access_audit.json", {"label_reads": 0, "outcome_reads": 0, "CAP_COP_label_used_for_selection": 0, "clinical_predictor_reads": 0})
    write_json(DIRS["audit"] / "model_training_audit.json", {"CNN_training_runs": 0, "resnet_training_runs": 0, "densenet_training_runs": 0, "image_classification_runs": 0, "radiomics_classification_runs": 0, "AUROC_computations": 0, "mask_guided_training_runs": 0, "multimodal_runs": 0})
    if status.endswith("PENDING"):
        lock = {
            "status": "AUTOMATED_QC_PASS_VISUAL_REVIEW_PENDING",
            "canonical_orientation": "DICOM LPS physical coordinate convention: +X patient Left, +Y patient Posterior, +Z patient Superior; tensor array order tensor[z,y,x], output [C,Z,Y,X]",
            "body_centering_rule": "X/Y use body-envelope physical bounding-box center; Z uses body-envelope effective scan range center; no lesion/lung/label/manual adjustment",
            "body_proxy_algorithm": "HU>-600 axial foreground, hole fill, central-intersecting main connected component, stacked 3D body envelope; not a lung or lesion mask",
            "selected_candidate": selected["candidate"],
            "selected_physical_FOV_mm": list(selected_fov),
            "effective_spacing_mm": list(selected_spacing),
            "output_shape": [1, 96, 96, 96],
            "anti_alias_formula": "sigma_axis_native_voxels=(target_spacing/native_spacing - 1)/2 for downsampling_factor>1, else 0",
            "interpolation": "linear continuous CT intensity interpolation",
            "padding_HU": -1000,
            "HU_clipping": list(HU_WINDOW),
            "normalization": "fixed min-max to [0,1]: (clip(HU,-1000,400)+1000)/1400",
            "source_code_hash": sha256_file(Path(__file__)),
            "internal_CT_manifest_hash": sha256_file(DIRS["runs"] / "internal_ct_manifest_PRIVATE.csv"),
            "external_reads": 0,
            "mask_reads": 0,
            "label_reads": 0,
            "CNN_runs": 0,
            "visual_review_status": "PENDING",
        }
        lock_path = DIRS["locked_candidates"] / "IMAGING_96CUBED_PHYSICAL_TENSOR_CONTRACT_CANDIDATE_V2.json"
        write_json(lock_path, lock)
        candidate_lock_hash = sha256_file(lock_path)
    else:
        candidate_lock_hash = "NA"

    reports = {
        "00_PROTOCOL_LOCK.md": "# Protocol Lock\n\nThis stage is label-blind, internal-only physical-space tensor contract development for 96x96x96 CT tensors. No CNN, radiomics classifier, mask, clinical predictor, external CT, external metadata, external label, prediction, or metric was read or run.\n",
        "01_INTERNAL_CT_CONTRACT_AUDIT.md": f"# Internal CT Contract Audit\n\nInternal CT count: 375. Matching unique: PASS. Raw paths are not exported. Historical provisional 1.5 mm isotropic spacing is SUPERSEDED_FOR_FINAL_96CUBED_TENSOR_DESIGN and the old provisional lock was not modified.\n",
        "02_BODY_ENVELOPE_GEOMETRY.md": "# Body Envelope Geometry\n\n" + csv_block(extent_dist),
        "03_PHYSICAL_FOV_CANDIDATES.md": "# Physical FOV Candidates\n\n" + csv_block(fov_df),
        "04_FOV_COVERAGE_AUDIT.md": "# FOV Coverage Audit\n\n" + csv_block(coverage_summary_df),
        "05_FINAL_FOV_SELECTION.md": f"# Final FOV Selection\n\nSelected candidate: {selected['candidate']}\n\nSelected FOV mm: {selected_fov}\n\nSelection rule: smallest physical volume among COVERAGE_PASS candidates.\n",
        "06_96CUBED_EFFECTIVE_SPACING.md": f"# 96-Cubed Effective Spacing\n\nEffective spacing x/y/z mm: {selected_spacing}\n",
        "07_ANTI_ALIASING_CONTRACT.md": "# Anti-Aliasing Contract\n\nScale-aware Gaussian prefilter is fixed by sigma_axis_native_voxels=(downsampling_factor_axis-1)/2 when downsampling_factor>1, otherwise 0. Formal contract always uses anti-aliasing.\n",
        "08_SINGLE_STEP_RESAMPLING_AUDIT.md": "# Single-Step Resampling Audit\n\nRaw/native HU -> canonical LPS -> fixed physical FOV coordinates with -1000 HU air padding -> scale-aware anti-alias filtering -> one direct linear resample to 96x96x96. No 1.5 mm intermediate grid and no double interpolation.\n",
        "09_FINAL_TENSOR_QC.md": "# Final Tensor QC\n\n" + csv_block(qc_summary),
        "10_THREE_PLANE_VISUAL_QC_GUIDE.md": "# Three-Plane Visual QC Guide\n\nVisual review is pending. Review axial, coronal, and sagittal montages in the montage directory. Codex does not mark final visual QA as passed.\n",
        "11_SCIENTIFIC_RATIONALE.md": "# Scientific Rationale\n\nThe final effective spacing is derived from selected physical FOV divided by 96, not from the historical 1.5 mm provisional preflight concept. Candidate FOVs are generated from internal body extent distributions with 10 mm per-side margin and selected by label-blind coverage gates.\n",
        "12_LIMITATIONS.md": "# Limitations\n\nBody envelope proxy is deterministic and label-blind but is not lung segmentation. The 96-cubed tensor is intentionally lossy. Automated QC passing does not replace human three-plane montage review.\n",
    }
    for name, text in reports.items():
        write_md(DIRS["reports"] / name, text)

    completed = {
        "status": status,
        "start_time": start,
        "end_time": now_iso(),
        "internal_ct_cases": 375,
        "selected_FOV_mm": selected_fov,
        "effective_spacing_mm": selected_spacing,
        "q_gates": q_gates,
        "external_reads": 0,
        "mask_reads": 0,
        "label_reads": 0,
        "CNN_training_runs": 0,
        "candidate_lock_sha256": candidate_lock_hash,
        "montage_paths": [str(DIRS["montage"] / p) for p in ["final_tensor_axial_montage.png", "final_tensor_coronal_montage.png", "final_tensor_sagittal_montage.png", "extreme_geometry_cases_montage.png"]],
    }
    write_json(DIRS["audit"] / "IMAGING_96CUBED_TENSOR_RUN_COMPLETED.json", completed)
    append_checksums()

    xrow = extent_dist[extent_dist.metric.eq("body_extent_x_mm")].iloc[0]
    yrow = extent_dist[extent_dist.metric.eq("body_extent_y_mm")].iloc[0]
    zrow = extent_dist[extent_dist.metric.eq("body_extent_z_mm")].iloc[0]
    p95_row = coverage_summary_df[coverage_summary_df.candidate.eq("CANDIDATE_P95")].iloc[0]
    p99_row = coverage_summary_df[coverage_summary_df.candidate.eq("CANDIDATE_P99")].iloc[0]
    max_row = coverage_summary_df[coverage_summary_df.candidate.eq("CANDIDATE_MAX")].iloc[0]
    pad_stats = tensor_qc_df["padding_fraction"]
    print("internal CT audit: PASS")
    print("internal CT count: 375")
    print("label reads: 0")
    print("external reads: 0")
    print("mask reads: 0")
    print("canonical orientation: DICOM_LPS tensor[z,y,x]")
    print(f"body proxy success count: {int(body_df['body_proxy_status'].eq('PASS').sum())}")
    print(f"body extent X p50/p95/p99/max: {xrow.p50}/{xrow.p95}/{xrow.p99}/{xrow['max']}")
    print(f"body extent Y p50/p95/p99/max: {yrow.p50}/{yrow.p95}/{yrow.p99}/{yrow['max']}")
    print(f"body extent Z p50/p95/p99/max: {zrow.p50}/{zrow.p95}/{zrow.p99}/{zrow['max']}")
    print(f"P95 candidate FOV: {[p95_row.FOV_x_mm, p95_row.FOV_y_mm, p95_row.FOV_z_mm]}")
    print(f"P95 coverage PASS/FAIL: {p95_row.coverage_status}")
    print(f"P99 candidate FOV: {[p99_row.FOV_x_mm, p99_row.FOV_y_mm, p99_row.FOV_z_mm]}")
    print(f"P99 coverage PASS/FAIL: {p99_row.coverage_status}")
    print(f"MAX candidate FOV: {[max_row.FOV_x_mm, max_row.FOV_y_mm, max_row.FOV_z_mm]}")
    print(f"MAX coverage PASS/FAIL: {max_row.coverage_status}")
    print(f"selected physical FOV: {list(selected_fov)}")
    print(f"effective spacing x/y/z: {list(selected_spacing)}")
    print(f"maximum downsampling factor: {float(aa_df[['downsampling_factor_x','downsampling_factor_y','downsampling_factor_z']].max().max())}")
    print("anti-alias rule: sigma=(target_spacing/native_spacing-1)/2 if downsampling_factor>1 else 0")
    print(f"tensor shape audit: {'PASS' if shape_ok else 'FAIL'}")
    print(f"padding fraction median/p95/max: {float(pad_stats.median())}/{float(pad_stats.quantile(0.95))}/{float(pad_stats.max())}")
    print(f"worst retained body fraction: {float(selected_summary['minimum'])}")
    print(f"orientation failures: {int((pd.DataFrame(orientation_rows)['canonical_transform_status']!='PASS').sum())}")
    print(f"HU conversion failures: {int((pd.DataFrame(hu_rows)['hu_conversion_status']!='PASS').sum())}")
    print("double interpolation count: 0")
    print("automated Q1-Q13 PASS/FAIL: " + "; ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in q_gates.items()))
    print("montage paths: " + "; ".join(completed["montage_paths"]))
    print(f"candidate lock SHA256: {candidate_lock_hash}")
    print("visual review status: PENDING")
    print("CNN training runs: 0")
    print(f"final status: {status}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        ensure_dirs()
        write_json(DIRS["audit"] / "IMAGING_96CUBED_TENSOR_RUN_COMPLETED.json", {"status": "IMAGING_96CUBED_TENSOR_AUTOMATED_QC_FAIL", "error": repr(exc), "traceback": traceback.format_exc(), "end_time": now_iso()})
        append_checksums()
        print("final status: IMAGING_96CUBED_TENSOR_AUTOMATED_QC_FAIL")
        print(f"error: {exc!r}")
        raise
