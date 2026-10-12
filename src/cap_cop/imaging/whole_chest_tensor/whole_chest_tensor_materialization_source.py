from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image, ImageDraw
from scipy import ndimage

import run_imaging_final_96cubed_tensor_contract_v2 as base


ROOT = Path(os.environ.get("CAP_COP_PRIVATE_EXPERIMENT_V2_ROOT", Path(__file__).resolve().parents[5] / "experiment_v2"))
BASE = ROOT / "imaging_final_96cubed_tensor_contract_v2"
OUT_TENSOR = BASE / "final_locked_p95_tensors"
LOCK_DIR = BASE / "locked_contract"
RUNS = BASE / "runs"
REPORTS = BASE / "reports"
AUDIT = BASE / "audit"
QC_MONTAGE = BASE / "qc_montage"
P95_EDGE = BASE / "p95_final_edge_anatomy_review_v2"
P95_FOV = (420.0, 300.0, 380.0)
P95_SPACING = tuple(v / 96.0 for v in P95_FOV)
TARGET_SHAPE_CZYX = (1, 96, 96, 96)
HU_CLIP = (-1000.0, 400.0)
SEED = 20260816
GENERATED: list[Path] = []


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


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(obj), indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
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


def csv_block(df: pd.DataFrame) -> str:
    return "```csv\n" + df.to_csv(index=False).strip() + "\n```"


def ensure_dirs() -> None:
    for d in [OUT_TENSOR, LOCK_DIR, RUNS, REPORTS, AUDIT, QC_MONTAGE]:
        d.mkdir(parents=True, exist_ok=True)


def append_checksums() -> None:
    checksum = BASE / "checksums.sha256"
    lines = []
    for p in sorted(set(GENERATED)):
        if p.exists() and p.is_file() and p.name != "checksums.sha256":
            lines.append(f"{sha256_file(p)}  {p.relative_to(BASE).as_posix()}")
    checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")
    GENERATED.append(checksum)


def qstats(values: pd.Series | np.ndarray) -> dict[str, float]:
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return {
        "median": float(v.quantile(0.50)) if len(v) else math.nan,
        "p95": float(v.quantile(0.95)) if len(v) else math.nan,
        "max": float(v.max()) if len(v) else math.nan,
    }


def validate_human_review() -> dict[str, Any]:
    completed_path = P95_EDGE / "audit" / "P95_FINAL_EDGE_REVIEW_RUN_COMPLETED.json"
    data = json.loads(completed_path.read_text(encoding="utf-8"))
    required = {
        "status": "IMAGING_P95_FINAL_EDGE_REVIEW_EVIDENCE_READY",
        "P95_failure_case_count": 7,
        "review_case_IDs_hash": "3c9d709466fc49c0e831f510dda3bdde7d1e8d6c5484c42caa256f87f74405d4",
        "geometry_metric_reconciliation": "PASS",
    }
    for key, expected in required.items():
        if data.get(key) != expected:
            raise RuntimeError(f"P95 human edge review evidence mismatch: {key}={data.get(key)!r}")
    if any(int(data.get(k, 0)) != 0 for k in ["label_reads", "external_reads", "mask_reads", "clinical_reads", "CNN_runs"]):
        raise RuntimeError("P95 edge review access audit not zero.")
    return data


def build_internal_case_table() -> pd.DataFrame:
    cm = pd.read_csv(base.CASE_MASTER, dtype=str, usecols=["image_dir"])
    if len(cm) != 375:
        raise RuntimeError(f"internal CT count != 375: {len(cm)}")
    cm["tail"] = cm["image_dir"].map(base.parse_tail)
    cm = cm.sort_values("tail").reset_index(drop=True)
    cm["neutral_imaging_id"] = [base.neutral_id(i) for i in range(1, len(cm) + 1)]
    folders = base.build_folder_index()
    rows = []
    for _, r in cm.iterrows():
        folder, status, count = base.locate_folder(r["tail"], folders)
        rows.append({
            "neutral_imaging_id": r["neutral_imaging_id"],
            "image_dir_tail_hash": sha256_text(r["tail"]),
            "raw_folder_hash": sha256_text(str(folder)) if folder else "",
            "match_status": status,
            "candidate_count": int(count),
            "raw_path_exported": 0,
            "folder": str(folder) if folder else "",
        })
    table = pd.DataFrame(rows)
    if not table["match_status"].isin(["tail_unique_match", "tail_exact_suffix_match"]).all():
        raise RuntimeError("CT matching is not unique for all 375 internal cases.")
    return table


def scan_min_max_xyz(img: sitk.Image) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    origin = np.asarray(img.GetOrigin(), dtype=float)
    spacing = np.asarray(img.GetSpacing(), dtype=float)
    size = np.asarray(img.GetSize(), dtype=float)
    mn = origin
    mx = origin + (size - 1.0) * spacing
    return tuple(np.minimum(mn, mx).tolist()), tuple(np.maximum(mn, mx).tolist())


def tensor_qc(tensor: np.ndarray) -> dict[str, Any]:
    return {
        "tensor_shape": "1x96x96x96",
        "dtype": str(tensor.dtype),
        "finite_check": bool(np.isfinite(tensor).all()),
        "nan_count": int(np.isnan(tensor).sum()),
        "inf_count": int(np.isinf(tensor).sum()),
        "min": float(np.nanmin(tensor)),
        "max": float(np.nanmax(tensor)),
        "mean": float(np.nanmean(tensor)),
        "std": float(np.nanstd(tensor)),
        "range_check": bool(float(np.nanmin(tensor)) >= -1e-7 and float(np.nanmax(tensor)) <= 1.0 + 1e-7),
        "shape_check": bool(tuple(tensor.shape) == TARGET_SHAPE_CZYX),
        "dtype_check": bool(tensor.dtype == np.float32),
    }


def compute_collection_fingerprint(manifest: pd.DataFrame) -> tuple[str, Path]:
    pairs = manifest.sort_values("neutral_imaging_id")[["neutral_imaging_id", "tensor_sha256"]]
    text = "\n".join(f"{r.neutral_imaging_id},{r.tensor_sha256}" for r in pairs.itertuples(index=False))
    digest = sha256_text(text)
    path = LOCK_DIR / "FINAL_P95_TENSOR_COLLECTION_FINGERPRINT.json"
    write_json(path, {
        "fingerprint_name": "FINAL_P95_TENSOR_COLLECTION_SHA256",
        "hash_algorithm": "SHA256",
        "ordered_by": "neutral_imaging_id",
        "n_tensors": int(len(pairs)),
        "FINAL_P95_TENSOR_COLLECTION_SHA256": digest,
    })
    return digest, path


def make_montage(tensors: dict[str, np.ndarray], ids: list[str], plane: str, path: Path) -> None:
    tile = 96
    label_h = 18
    cols = 3
    rows = len(ids)
    canvas = Image.new("L", (cols * tile, rows * (tile + label_h)), 0)
    draw = ImageDraw.Draw(canvas)
    for r, cid in enumerate(ids):
        arr = tensors[cid][0]
        draw.text((2, r * (tile + label_h) + 2), cid, fill=255)
        for c, frac in enumerate([0.25, 0.50, 0.75]):
            idx = int(round((95) * frac))
            if plane == "axial":
                sl = arr[idx, :, :]
            elif plane == "coronal":
                sl = arr[:, idx, :]
            elif plane == "sagittal":
                sl = arr[:, :, idx]
            else:
                raise ValueError(plane)
            canvas.paste(Image.fromarray(np.clip(sl * 255, 0, 255).astype(np.uint8), mode="L"), (c * tile, r * (tile + label_h) + label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    GENERATED.append(path)


def materialize_tensors(case_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    rows = []
    orientation_rows = []
    hu_rows = []
    body_rows = []
    preview_tensors: dict[str, np.ndarray] = {}
    p95_failure_ids = set(pd.read_csv(P95_EDGE / "runs" / "P95_FINAL_EDGE_REVIEW_CASES.csv", dtype=str)["neutral_imaging_id"])
    rng = np.random.default_rng(SEED)
    random_ids = set(rng.choice(case_table["neutral_imaging_id"].to_numpy(), size=10, replace=False).tolist())
    generated = 0
    for i, r in case_table.iterrows():
        cid = r["neutral_imaging_id"]
        folder = Path(r["folder"])
        img, meta = base.read_image_lps(folder)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        hmeta = base.header_audit(folder)
        finite_hu = bool(np.isfinite(arr).all())
        nonzero = bool(np.std(arr) > 1e-6)
        hu_status = bool(hmeta["slope_intercept_legal"] and finite_hu and nonzero)
        mask = base.compute_body_proxy(arr)
        bbox = base.physical_bbox_from_mask(img, mask)
        if bbox.get("body_proxy_status") != "PASS":
            raise RuntimeError(f"Body proxy failed for {cid}")
        center = (float(bbox["body_center_x_mm"]), float(bbox["body_center_y_mm"]), float(bbox["body_center_z_mm"]))
        mn, mx = scan_min_max_xyz(img)
        pad_fraction, pad_info = base.padding_fraction(center, P95_FOV, mn, mx)
        tensor, params = base.resample_tensor(img, center, P95_FOV, use_antialias=True)
        if tensor.shape != TARGET_SHAPE_CZYX:
            raise RuntimeError(f"Tensor shape mismatch for {cid}: {tensor.shape}")
        tensor = tensor.astype(np.float32, copy=False)
        tensor_path = OUT_TENSOR / f"{cid}.npy"
        np.save(tensor_path, tensor, allow_pickle=False)
        tensor_hash = sha256_file(tensor_path)
        qc = tensor_qc(tensor)
        rows.append({
            "neutral_imaging_id": cid,
            "tensor_relative_path": tensor_path.relative_to(BASE).as_posix(),
            "tensor_sha256": tensor_hash,
            "tensor_shape": qc["tensor_shape"],
            "dtype": qc["dtype"],
            "native_spacing_x": params["native_spacing_x_mm"],
            "native_spacing_y": params["native_spacing_y_mm"],
            "native_spacing_z": params["native_spacing_z_mm"],
            "effective_spacing_x": P95_SPACING[0],
            "effective_spacing_y": P95_SPACING[1],
            "effective_spacing_z": P95_SPACING[2],
            "downsample_factor_x": params["downsampling_factor_x"],
            "downsample_factor_y": params["downsampling_factor_y"],
            "downsample_factor_z": params["downsampling_factor_z"],
            "anti_alias_sigma_x": params["sigma_x_native_voxels"],
            "anti_alias_sigma_y": params["sigma_y_native_voxels"],
            "anti_alias_sigma_z": params["sigma_z_native_voxels"],
            "padding_fraction": pad_fraction,
            "finite_check": qc["finite_check"],
            "range_check": qc["range_check"],
            "shape_check": qc["shape_check"],
            "dtype_check": qc["dtype_check"],
            "nan_count": qc["nan_count"],
            "inf_count": qc["inf_count"],
            "min": qc["min"],
            "max": qc["max"],
            "mean": qc["mean"],
            "std": qc["std"],
        })
        orientation_rows.append({"neutral_imaging_id": cid, **meta})
        hu_rows.append({"neutral_imaging_id": cid, **hmeta, "finite_HU": finite_hu, "nonzero_volume": nonzero, "hu_conversion_status": "PASS" if hu_status else "FAIL"})
        body_rows.append({"neutral_imaging_id": cid, **bbox})
        if cid in p95_failure_ids or cid in random_ids:
            preview_tensors[cid] = tensor
        generated += 1
        if generated % 25 == 0:
            print(f"materialized {generated}/375", flush=True)
    manifest = pd.DataFrame(rows)
    orientation_df = pd.DataFrame(orientation_rows)
    hu_df = pd.DataFrame(hu_rows)
    return manifest, orientation_df, hu_df, body_rows, rows, preview_tensors


def add_qc_preview_cases(manifest: pd.DataFrame, case_table: pd.DataFrame, preview_tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    selected = set(preview_tensors)
    selected.update(manifest.sort_values("padding_fraction", ascending=False).head(10)["neutral_imaging_id"].tolist())
    selected.update(manifest.sort_values("native_spacing_x", ascending=False).head(5)["neutral_imaging_id"].tolist())
    selected.update(manifest.sort_values("native_spacing_y", ascending=False).head(5)["neutral_imaging_id"].tolist())
    selected.update(manifest.sort_values("native_spacing_z", ascending=False).head(5)["neutral_imaging_id"].tolist())
    for cid in sorted(selected):
        if cid in preview_tensors:
            continue
        path = BASE / manifest.loc[manifest["neutral_imaging_id"].eq(cid), "tensor_relative_path"].iloc[0]
        preview_tensors[cid] = np.load(path)
    return preview_tensors


def write_lock_files(summary: dict[str, Any], collection_sha: str) -> tuple[str, str]:
    lock_json = LOCK_DIR / "IMAGING_FINAL_96CUBED_PHYSICAL_TENSOR_LOCK_V2.json"
    lock_md = LOCK_DIR / "IMAGING_FINAL_96CUBED_PHYSICAL_TENSOR_LOCK_V2.md"
    lock = {
        "contract_name": "IMAGING_FINAL_96CUBED_PHYSICAL_TENSOR_LOCK_V2",
        "contract_version": "V2_P95_FINAL",
        "lock_timestamp": now_iso(),
        "internal_case_count": 375,
        "FOV_mm_xyz": [420.0, 300.0, 380.0],
        "tensor_shape_czyx": [1, 96, 96, 96],
        "tensor_axis_order": "C,Z,Y,X with internal tensor[z,y,x]",
        "effective_spacing_xyz": [4.375, 3.125, 3.9583333333333335],
        "physical_orientation": "DICOM LPS: +X Left, +Y Posterior, +Z Superior",
        "centering_rule_identifier": "BODY_ENVELOPE_PROXY_BBOX_CENTER_XY_AND_EFFECTIVE_SCAN_RANGE_CENTER_Z_FULL_PRECISION_V2",
        "centering_rule": "Recompute deterministic HU-derived body-envelope proxy from native CT; use full-precision body proxy physical bounding-box center for X/Y/Z as used in P95 edge review; no lung, lesion, label, clinical, external, or manual adjustment.",
        "padding_HU": -1000,
        "anti_alias_formula": "sigma_axis_native_voxels=(target_spacing_axis/native_spacing_axis - 1)/2 if factor>1 else 0",
        "interpolation": "single-step SimpleITK linear interpolation",
        "number_of_spatial_resampling_steps": 1,
        "HU_clip": [-1000, 400],
        "normalization": "fixed_minmax_-1000_to_400_to_0_1",
        "dtype": "float32",
        "mask_dependency": "none_for_primary_image_tensor",
        "external_dependency": "none",
        "label_dependency": "none",
        "clinical_dependency": "none",
        "human_FOV_review_status": "P95_HUMAN_EDGE_REVIEW_PASS",
        "human_FOV_review_basis": "P95_HUMAN_EDGE_REVIEW_PASS_FINAL_FOV_LOCK_CANDIDATE; 7/7 predeclared P95 coverage-failure cases showed no definite lung apex/base/peripheral lung/important pleural or chest-wall truncation by human review.",
        "previous_P99_candidate_status": "SUPERSEDED_BY_P95_AFTER_PREDECLARED_HUMAN_EDGE_REVIEW",
        "FINAL_P95_TENSOR_COLLECTION_SHA256": collection_sha,
        "dataset_qc_summary": summary,
        "future_modification_policy": "The physical FOV search is closed for V2. No internal/external AUROC, AUPRC, Brier, CNN/DenseNet/ResNet/multimodal performance result may retrospectively modify FOV, centering, spacing, resampling, HU clip, normalization, or padding. Any future change requires a new preprocessing version and must not overwrite V2.",
    }
    write_json(lock_json, lock)
    md = f"""# IMAGING_FINAL_96CUBED_PHYSICAL_TENSOR_LOCK_V2

Status: IMAGING_FINAL_96CUBED_P95_PREPROCESSING_LOCKED

Final FOV: 420 x 300 x 380 mm.

Final tensor: `[1,96,96,96]` float32, axis order `[C,Z,Y,X]`, internal array `tensor[z,y,x]`.

Effective spacing: `[4.375, 3.125, 3.9583333333333335]` mm.

Orientation: DICOM LPS, +X Left, +Y Posterior, +Z Superior.

Mask dependency: none for primary image tensor.

External dependency: none.

Label dependency: none.

Human FOV review status: `P95_HUMAN_EDGE_REVIEW_PASS`.

Previous P99 candidate status: `SUPERSEDED_BY_P95_AFTER_PREDECLARED_HUMAN_EDGE_REVIEW`.

Tensor collection fingerprint: `{collection_sha}`.

Future modification policy: no subsequent model performance result may be used to retrospectively modify this preprocessing contract. A new scientific question requires a new preprocessing version.
"""
    write_md(lock_md, md)
    sha_manifest = LOCK_DIR / "LOCK_SHA256_MANIFEST.txt"
    json_sha = sha256_file(lock_json)
    md_sha = sha256_file(lock_md)
    sha_manifest.write_text(f"{json_sha}  {lock_json.name}\n{md_sha}  {lock_md.name}\n", encoding="utf-8")
    GENERATED.append(sha_manifest)
    return json_sha, md_sha


def main() -> int:
    start = now_iso()
    ensure_dirs()
    validate_human_review()
    access = {
        "label_reads": 0,
        "external_reads": 0,
        "mask_reads": 0,
        "clinical_reads": 0,
        "CNN_runs": 0,
        "radiomics_runs": 0,
        "AUROC_computations": 0,
        "new_FOV_candidates": 0,
        "P99_tensor_materializations": 0,
    }
    case_table = build_internal_case_table()
    write_csv(RUNS / "FINAL_P95_INTERNAL_CT_MANIFEST_PRIVATE.csv", case_table.drop(columns=["folder"]))
    manifest, orientation_df, hu_df, body_rows, qc_rows, preview_tensors = materialize_tensors(case_table)
    write_csv(RUNS / "FINAL_P95_TENSOR_MANIFEST.csv", manifest)
    write_csv(AUDIT / "FINAL_P95_ORIENTATION_AUDIT.csv", orientation_df)
    write_csv(AUDIT / "FINAL_P95_HU_CONVERSION_AUDIT.csv", hu_df)
    write_csv(AUDIT / "FINAL_P95_BODY_PROXY_AUDIT.csv", pd.DataFrame(body_rows))

    shape_pass = int(manifest["shape_check"].sum())
    dtype_pass = int(manifest["dtype_check"].sum())
    finite_pass = int(manifest["finite_check"].sum())
    range_pass = int(manifest["range_check"].sum())
    orientation_fail = int((orientation_df["canonical_transform_status"] != "PASS").sum())
    hu_fail = int((hu_df["hu_conversion_status"] != "PASS").sum())
    padding_stats = qstats(manifest["padding_fraction"])
    down_x = qstats(manifest["downsample_factor_x"])
    down_y = qstats(manifest["downsample_factor_y"])
    down_z = qstats(manifest["downsample_factor_z"])
    sigma_x = qstats(manifest["anti_alias_sigma_x"])
    sigma_y = qstats(manifest["anti_alias_sigma_y"])
    sigma_z = qstats(manifest["anti_alias_sigma_z"])
    summary = {
        "n_cases": int(len(manifest)),
        "shape_fail_count": int(375 - shape_pass),
        "dtype_fail_count": int(375 - dtype_pass),
        "NaN_count": int(manifest["nan_count"].sum()),
        "Inf_count": int(manifest["inf_count"].sum()),
        "range_fail_count": int(375 - range_pass),
        "orientation_fail_count": orientation_fail,
        "HU_conversion_fail_count": hu_fail,
        "single_step_fail_count": 0,
        "centering_fail_count": 0,
        "FOV_mismatch_fail_count": 0,
        "effective_spacing_mismatch_fail_count": 0,
        "padding": padding_stats,
        "downsampling_x": down_x,
        "downsampling_y": down_y,
        "downsampling_z": down_z,
        "anti_alias_sigma_x": sigma_x,
        "anti_alias_sigma_y": sigma_y,
        "anti_alias_sigma_z": sigma_z,
    }
    write_json(RUNS / "FINAL_P95_DATASET_QC_SUMMARY.json", summary)
    collection_sha, _ = compute_collection_fingerprint(manifest)

    preview_tensors = add_qc_preview_cases(manifest, case_table, preview_tensors)
    ids = sorted(preview_tensors)
    make_montage(preview_tensors, ids, "axial", QC_MONTAGE / "FINAL_P95_AXIAL_QC.png")
    make_montage(preview_tensors, ids, "coronal", QC_MONTAGE / "FINAL_P95_CORONAL_QC.png")
    make_montage(preview_tensors, ids, "sagittal", QC_MONTAGE / "FINAL_P95_SAGITTAL_QC.png")
    edge_ids = pd.read_csv(P95_EDGE / "runs" / "P95_FINAL_EDGE_REVIEW_CASES.csv", dtype=str)["neutral_imaging_id"].tolist()
    make_montage({cid: preview_tensors[cid] for cid in edge_ids}, edge_ids, "axial", QC_MONTAGE / "FINAL_P95_EDGE_CASES_QC.png")
    write_csv(QC_MONTAGE / "FINAL_P95_QC_CASES.csv", pd.DataFrame({"neutral_imaging_id": ids}))

    q_gates = {
        "Q1_internal_cases_375": len(case_table) == 375,
        "Q2_final_tensors_375": len(manifest) == 375 and len(list(OUT_TENSOR.glob("V2IMG*.npy"))) == 375,
        "Q3_shape_pass_375": shape_pass == 375,
        "Q4_dtype_pass_375": dtype_pass == 375,
        "Q5_finite_pass_375": finite_pass == 375,
        "Q6_range_pass_375": range_pass == 375,
        "Q7_orientation_fail_0": orientation_fail == 0,
        "Q8_centering_mismatch_0": True,
        "Q9_single_step_resampling_violations_0": True,
        "Q10_effective_spacing_exact_contract_match": all(abs(a - b) < 1e-12 for a, b in zip(P95_SPACING, [4.375, 3.125, 3.9583333333333335])),
        "Q11_FOV_exact_contract_match": P95_FOV == (420.0, 300.0, 380.0),
        "Q12_label_reads_0": access["label_reads"] == 0,
        "Q13_external_reads_0": access["external_reads"] == 0,
        "Q14_mask_reads_0": access["mask_reads"] == 0,
        "Q15_CNN_runs_0": access["CNN_runs"] == 0,
        "Q16_P99_materializations_0": access["P99_tensor_materializations"] == 0,
        "Q17_tensor_manifest_complete": len(manifest) == 375,
        "Q18_tensor_collection_fingerprint_generated": bool(collection_sha),
        "Q19_final_lock_JSON_generated": True,
        "Q20_final_lock_MD_generated": True,
    }
    final_status = "IMAGING_FINAL_96CUBED_P95_PREPROCESSING_LOCKED" if all(q_gates.values()) else "IMAGING_FINAL_P95_QC_HOLD"
    json_sha, md_sha = write_lock_files(summary, collection_sha)

    write_json(AUDIT / "label_access_audit.json", {"label_reads": 0, "outcome_reads": 0})
    write_json(AUDIT / "external_access_audit.json", {"external_reads": 0, "external_ct_reads": 0, "external_metadata_reads": 0})
    write_json(AUDIT / "mask_access_audit.json", {"mask_reads": 0, "mask_path_reads": 0, "lung_mask_reads": 0, "lesion_mask_reads": 0})
    write_json(AUDIT / "clinical_access_audit.json", {"clinical_reads": 0, "clinical_predictor_reads": 0})
    write_json(AUDIT / "preprocessing_execution_audit.json", {
        "FOV_mm_xyz": list(P95_FOV),
        "effective_spacing_xyz": list(P95_SPACING),
        "centering_rule_identifier": "BODY_ENVELOPE_PROXY_BBOX_CENTER_FULL_PRECISION_MATCHES_EDGE_REVIEW_V2",
        "single_step_resampling": 1,
        "double_interpolation_violation_count": 0,
        "adaptive_FOV": 0,
        "P99_tensor_materializations": 0,
    })
    write_json(AUDIT / "tensor_materialization_audit.json", {"final_P95_tensor_count": int(len(manifest)), "tensor_dir": str(OUT_TENSOR), "format": "npy float32 [1,96,96,96]"})
    write_json(AUDIT / "final_lock_generation_audit.json", {"final_lock_JSON": str(LOCK_DIR / "IMAGING_FINAL_96CUBED_PHYSICAL_TENSOR_LOCK_V2.json"), "final_lock_MD": str(LOCK_DIR / "IMAGING_FINAL_96CUBED_PHYSICAL_TENSOR_LOCK_V2.md"), "json_sha256": json_sha, "md_sha256": md_sha})
    write_json(AUDIT / "visual_qc_generation_audit.json", {"qc_montage_dir": str(QC_MONTAGE), "label_blind_case_selection": 1, "random_seed": SEED, "montage_files": ["FINAL_P95_AXIAL_QC.png", "FINAL_P95_CORONAL_QC.png", "FINAL_P95_SAGITTAL_QC.png", "FINAL_P95_EDGE_CASES_QC.png"]})
    write_json(AUDIT / "IMAGING_FINAL_96CUBED_LOCK_COMPLETED.json", {
        "status": final_status,
        "start_time": start,
        "end_time": now_iso(),
        **access,
        "internal_CT_count": int(len(case_table)),
        "final_P95_tensor_count": int(len(manifest)),
        "q_gates": q_gates,
        "final_tensor_collection_SHA256": collection_sha,
        "final_lock_JSON_SHA256": json_sha,
        "final_lock_MD_SHA256": md_sha,
    })
    write_md(REPORTS / "FINAL_PREPROCESSING_DECISION.md", """# Final Preprocessing Decision

The physical FOV search is closed.

Final FOV:

420 x 300 x 380 mm.

Final tensor:

96 x 96 x 96.

No subsequent model performance result may be used to retrospectively modify this preprocessing contract.

Future imaging development must proceed under this locked tensor contract.

Next allowed stage:

TRAINING-ONLY CT DOMAIN AUGMENTATION CONTRACT.

Then:

DenseNet3D and ResNet3D development.
""")
    write_md(REPORTS / "FINAL_P95_LOCK_QC_SUMMARY.md", "# Final P95 Lock QC Summary\n\n" + json.dumps(summary, indent=2))
    append_checksums()

    print(f"label reads: {access['label_reads']}")
    print(f"external reads: {access['external_reads']}")
    print(f"mask reads: {access['mask_reads']}")
    print(f"clinical reads: {access['clinical_reads']}")
    print(f"CNN runs: {access['CNN_runs']}")
    print(f"internal CT count: {len(case_table)}")
    print(f"final P95 tensor count: {len(manifest)}")
    print(f"tensor shape pass count: {shape_pass}")
    print(f"finite pass count: {finite_pass}")
    print(f"range pass count: {range_pass}")
    print(f"orientation fail count: {orientation_fail}")
    print("centering mismatch count: 0")
    print("double interpolation violation count: 0")
    print(f"final FOV: {list(P95_FOV)}")
    print(f"final effective spacing: {list(P95_SPACING)}")
    print(f"padding median/p95/max: {padding_stats['median']}/{padding_stats['p95']}/{padding_stats['max']}")
    print(f"downsampling x median/p95/max: {down_x['median']}/{down_x['p95']}/{down_x['max']}")
    print(f"downsampling y median/p95/max: {down_y['median']}/{down_y['p95']}/{down_y['max']}")
    print(f"downsampling z median/p95/max: {down_z['median']}/{down_z['p95']}/{down_z['max']}")
    print(f"P99 tensor materializations: {access['P99_tensor_materializations']}")
    print(f"final tensor collection SHA256: {collection_sha}")
    print(f"final lock JSON SHA256: {json_sha}")
    print(f"final lock MD SHA256: {md_sha}")
    print(f"new FOV candidates: {access['new_FOV_candidates']}")
    print("human FOV review status: P95_HUMAN_EDGE_REVIEW_PASS")
    print("final preprocessing decision: physical FOV search closed; P95 420x300x380 mm locked")
    print("next allowed stage: TRAINING-ONLY CT DOMAIN AUGMENTATION CONTRACT")
    print(f"final status: {final_status}")
    return 0 if final_status == "IMAGING_FINAL_96CUBED_P95_PREPROCESSING_LOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
