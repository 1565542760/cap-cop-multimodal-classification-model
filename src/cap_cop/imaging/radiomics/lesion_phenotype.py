from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[4]
ROOT = PROJECT_ROOT / "experiment_v2" / "imaging_segment_then_classify_highres_v3_3"
SRC = PROJECT_ROOT / "experiment_v2" / "src"
OUT = ROOT / "05_classification" / "05_exploratory_image_rescue_v3_4"

DIRS = {
    "protocol": OUT / "01_protocol_lock",
    "radiomics": OUT / "02_radiomics_features",
    "embeddings": OUT / "03_deep_embeddings",
    "models": OUT / "04_models",
    "oof": OUT / "05_oof_predictions",
    "results": OUT / "06_internal_results",
    "figures": OUT / "07_figures",
    "audit": OUT / "08_audit",
}

SEED = 20260627
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
N_BOOTSTRAP = 10000
OUTER_FOLDS = list(range(5))
TENSOR_SHAPE = (1, 144, 144, 144)
ROI_N = 144
ROI_SPACING_MM = 184.0 / 144.0
FIXED_QUANTIZATION_RANGE = (0.0, 1.0)
FIXED_GRAY_LEVELS = 16

LABEL_PATH = ROOT / "05_classification" / "01_protocol_lock" / "CLASSIFICATION_NEUTRAL_ID_LABEL_BINDING_MANIFEST_V3_3_FINAL.csv"
STRICT_CONTEXT_PATH = ROOT / "04_high_resolution_roi" / "05_locked_cache" / "HIGHRES_STRICT_OOF_CONTEXT_MANIFEST_V3_3_GEOMETRY_REMEDIATED.csv"
STRICT_ROI_PATH = ROOT / "04_high_resolution_roi" / "05_locked_cache" / "HIGHRES_STRICT_OUTER_TEST_OOF_ROI_MANIFEST_V3_3_GEOMETRY_REMEDIATED.csv"
TRAIN_CONTEXT_PATH = ROOT / "04_high_resolution_roi" / "05_locked_cache" / "HIGHRES_CLASSIFIER_TRAIN_CROSSFIT_CONTEXT_MANIFEST_V3_3_GEOMETRY_REMEDIATED.csv"
SEG_MANIFEST_PATH = ROOT / "03_segmentation" / "04_strict_oof_predictions" / "STRICT_OOF_SEGMENTATION_MANIFEST_V3_3_1A.csv"
METADATA_DIR = ROOT / "03_segmentation" / "01_preprocessed_2mm_cache" / "metadata"
FORMAL_RESULT_LOCK_PATH = ROOT / "05_classification" / "03_internal_results" / "HIGHRES_EXPEDITED_DUAL_BACKBONE_IMAGE_ONLY_INTERNAL_RESULT_LOCK_V3_3.json"
FORMAL_PROTOCOL_PATH = ROOT / "05_classification" / "01_protocol_lock" / "HIGHRES_DUAL_BACKBONE_IMAGE_CLASSIFICATION_BACKBONE_SPECIFIC_TIME_BUDGET_READY_FINAL_V3_3.json"
FORMAL_REGISTRY_PATH = ROOT / "05_classification" / "02_formal_training" / "FORMAL_CLASSIFICATION_EXPEDITED_10RUN_REGISTRY_V3_3.csv"
FORMAL_REGISTRY_JSON_PATH = ROOT / "05_classification" / "02_formal_training" / "FORMAL_CLASSIFICATION_EXPEDITED_10RUN_REGISTRY_V3_3.json"
RESNET_FORMAL_PRED_PATH = ROOT / "05_classification" / "03_internal_results" / "RESNET3D18" / "RESNET3D18_STRICT_OOF_PATIENT_PREDICTIONS_WITH_LABEL_V3_3.csv"
DENSENET_FORMAL_PRED_PATH = ROOT / "05_classification" / "03_internal_results" / "DENSENET3D121" / "DENSENET3D121_STRICT_OOF_PATIENT_PREDICTIONS_WITH_LABEL_V3_3.csv"

CHECKPOINTS = {
    "ResNet3D-18": ROOT / "05_classification" / "02_formal_training" / "RESNET3D18",
    "DenseNet3D-121": ROOT / "05_classification" / "02_formal_training" / "DENSENET3D121",
}

V3_3_BASE_PROTECTED_FILES = [
    FORMAL_RESULT_LOCK_PATH,
    FORMAL_PROTOCOL_PATH,
    FORMAL_REGISTRY_PATH,
    FORMAL_REGISTRY_JSON_PATH,
    RESNET_FORMAL_PRED_PATH,
    DENSENET_FORMAL_PRED_PATH,
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv_stable(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, lineterminator="\n")


def path_from_rel(rel: str) -> Path:
    p = Path(str(rel).replace("/", "\\"))
    if p.is_absolute():
        return p
    return ROOT / p


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def protected_hashes() -> dict[str, str]:
    files = list(V3_3_BASE_PROTECTED_FILES)
    for backbone in CHECKPOINTS:
        for outer in OUTER_FOLDS:
            files.append(checkpoint_path(backbone, outer))
    return {str(p): sha256_file(p) for p in files}


def checkpoint_path(backbone: str, outer: int) -> Path:
    return CHECKPOINTS[backbone] / f"OUTER_{outer:02d}" / "INNER_00" / f"SEED_{SEED}" / "INFERENCE_ONLY_BEST_STATE_DICT.pt"


def write_protocol(pre_hashes: dict[str, str], pyradiomics_available: bool) -> tuple[Path, Path, str]:
    protocol = {
        "task_name": "PERFORMANCE_INFORMED_EXPLORATORY_IMAGE_REPRESENTATION_RESCUE_V3_4",
        "created_at": now_iso(),
        "analysis_type": "PERFORMANCE_INFORMED_SECONDARY_EXPLORATORY",
        "development_timing": "AFTER_V3_3_FORMAL_IMAGE_RESULTS_REVIEWED",
        "formal_v3_3_primary_result_replaced": "NO",
        "project_root": str(PROJECT_ROOT),
        "formal_v3_3_root": str(ROOT),
        "output_root": str(OUT),
        "patients": {"unit": "PATIENT", "expected_unique_patients": 375, "outer_folds": OUTER_FOLDS},
        "labels": {"CAP": 0, "COP": 1, "label_join_source": str(LABEL_PATH), "patient_id_as_feature": "NO"},
        "prohibitions": {
            "external_access": "NO",
            "clinical_access": "NO",
            "multimodal": "NO",
            "cnn_retraining": "NO",
            "roi_regeneration": "NO",
            "segmentation_modification": "NO",
            "outer_split_modification": "NO",
            "ensemble_weight_tuning": "NO",
        },
        "radiomics": {
            "mask_source": "STRICT_OOF_PREDICTED_SEGMENTATION_MASKS_ONLY",
            "expert_gt_mask_as_classifier_feature": "NO",
            "roi_source": "V3_3_HIGHRES_STRICT_OOF_ROI_TENSORS",
            "roi_grid": {"shape_czyx": "1x144x144x144", "fov_mm": 184.0, "spacing_mm": ROI_SPACING_MM},
            "mask_to_roi_mapping": "LOCKED_ROI_CENTER_AND_2MM_WORKING_METADATA_NEAREST_NEIGHBOR",
            "empty_prediction_handling": "ALL_RADIOMICS_FEATURES_MISSING_PLUS_RADIOMICS_MISSING_INDICATOR",
            "pyradiomics_available": pyradiomics_available,
            "engine": "CUSTOM_FIXED_HANDCRAFTED_RADIOMICS_FALLBACK_NO_PYRADIOMICS",
            "engine_note": "PyRadiomics is not installed in this environment, so fixed handcrafted lesion-region descriptors are used and are not reported as PyRadiomics features.",
            "families": ["Shape", "First-order", "GLCM", "GLRLM", "GLSZM", "GLDM", "NGTDM"],
            "forbidden_families": ["Wavelet", "LoG", "large_filter_bank"],
            "intensity_source": "LOCKED_V3_3_HIGHRES_CLASSIFICATION_TENSOR_VALUES",
            "fixed_quantization": {"range": list(FIXED_QUANTIZATION_RANGE), "gray_levels": FIXED_GRAY_LEVELS},
            "component_rule": "TOP3_BY_PREDICTED_PHYSICAL_VOLUME",
            "patient_aggregation": ["LARGEST_COMPONENT_VALUE", "VOLUME_WEIGHTED_MEAN_ACROSS_VALID_COMPONENTS"],
        },
        "models": {
            "RADIOMICS_LR": {
                "pipeline": ["median_imputation", "variance_threshold", "standard_scaling", "L2_logistic_regression"],
                "class_weight": "balanced",
                "C_grid": C_GRID,
                "inner_cv": "StratifiedKFold(n_splits=3, shuffle=True, random_state=20260627+outer)",
                "selection_metric": "mean_inner_validation_log_loss",
                "tie_break": "smaller_C",
            },
            "RADIOMICS_RF": {
                "pipeline": ["median_imputation", "variance_threshold", "RandomForestClassifier"],
                "n_estimators": 1000,
                "max_features": "sqrt",
                "min_samples_leaf": 5,
                "class_weight": "balanced_subsample",
                "random_state": "20260627+outer",
            },
            "RESNET_EMBEDDING_LR": {
                "checkpoint_source": "V3_3_OUTER_SPECIFIC_INFERENCE_ONLY_BEST_STATE_DICT",
                "backbone": "ResNet3D-18",
                "embedding": "model.base(x) before final linear classification head",
                "patient_aggregation": "mean(valid_patch_embeddings)",
                "pipeline": ["median_imputation", "standard_scaling", "L2_logistic_regression"],
                "C_grid": C_GRID,
            },
            "DENSENET_EMBEDDING_LR": {
                "checkpoint_source": "V3_3_OUTER_SPECIFIC_INFERENCE_ONLY_BEST_STATE_DICT",
                "backbone": "DenseNet3D-121",
                "embedding": "model.base(x) before final linear classification head",
                "patient_aggregation": "mean(valid_patch_embeddings)",
                "pipeline": ["median_imputation", "standard_scaling", "L2_logistic_regression"],
                "C_grid": C_GRID,
            },
            "FIXED_3WAY_IMAGE_ENSEMBLE": {
                "inputs": ["RADIOMICS_LR", "RESNET_EMBEDDING_LR", "DENSENET_EMBEDDING_LR"],
                "weights": {"RADIOMICS_LR": 1 / 3, "RESNET_EMBEDDING_LR": 1 / 3, "DENSENET_EMBEDDING_LR": 1 / 3},
                "weight_tuning": "NO",
            },
            "FORMAL_CNN_50_50_REFERENCE": {
                "inputs": ["FORMAL_RESNET3D18_V3_3", "FORMAL_DENSENET3D121_V3_3"],
                "weights": {"FORMAL_RESNET3D18_V3_3": 0.5, "FORMAL_DENSENET3D121_V3_3": 0.5},
                "reference_only": "YES",
            },
        },
        "metrics": {
            "primary": "pooled_patient_level_OOF_AUROC",
            "secondary": ["AUPRC", "Brier", "BCE", "confusion_0.5", "sensitivity", "specificity", "PPV", "NPV", "accuracy", "balanced_accuracy", "F1", "calibration_intercept", "calibration_slope"],
            "bootstrap": {"unit": "PATIENT", "replicates": N_BOOTSTRAP, "seed": SEED},
        },
        "v3_3_pre_run_protected_sha256": pre_hashes,
    }
    json_path = DIRS["protocol"] / "EXPLORATORY_IMAGE_REPRESENTATION_RESCUE_PROTOCOL_V3_4.json"
    md_path = DIRS["protocol"] / "EXPLORATORY_IMAGE_REPRESENTATION_RESCUE_PROTOCOL_V3_4.md"
    write_json(json_path, protocol)
    sha = sha256_file(json_path)
    md = [
        "# EXPLORATORY_IMAGE_REPRESENTATION_RESCUE_PROTOCOL_V3_4",
        "",
        f"- analysis_type: {protocol['analysis_type']}",
        f"- development_timing: {protocol['development_timing']}",
        "- formal_v3_3_primary_result_replaced: NO",
        "- external_access: NO",
        "- clinical_access: NO",
        "- multimodal_started: NO",
        f"- radiomics_engine: {protocol['radiomics']['engine']}",
        f"- pyradiomics_available: {pyradiomics_available}",
        "- ensemble_weights: fixed equal 1/3,1/3,1/3",
        f"- seed: {SEED}",
        f"- protocol_json_sha256: {sha}",
        "",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")
    return json_path, md_path, sha


def validate_inputs(labels: pd.DataFrame, strict_context: pd.DataFrame, strict_roi: pd.DataFrame) -> None:
    ids = labels["neutral_patient_id"].astype(str)
    if len(ids) != 375 or ids.nunique() != 375:
        raise RuntimeError(f"Label manifest patient count invalid: rows={len(ids)} unique={ids.nunique()}")
    if sorted(labels["label_binary"].unique().tolist()) != [0, 1]:
        raise RuntimeError("Labels are not binary 0/1.")
    ctx_ids = strict_context["neutral_patient_id"].astype(str)
    if len(ctx_ids) != 375 or ctx_ids.nunique() != 375:
        raise RuntimeError(f"Strict OOF context count invalid: rows={len(ctx_ids)} unique={ctx_ids.nunique()}")
    if sorted(strict_context["outer_fold"].astype(int).unique().tolist()) != OUTER_FOLDS:
        raise RuntimeError("Strict OOF contexts do not cover outer folds 0..4.")
    missing = set(ids) - set(ctx_ids)
    if missing:
        raise RuntimeError(f"Strict OOF context missing label patients: {len(missing)}")
    if "tensor_relative_path" not in strict_roi.columns:
        raise RuntimeError("Strict ROI manifest missing tensor_relative_path.")


def load_metadata(source_neutral_imaging_id: str) -> dict[str, Any]:
    path = METADATA_DIR / f"{source_neutral_imaging_id}_metadata.json"
    with path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    direction = np.array(meta["working_direction"], dtype=float).reshape(3, 3)
    if not np.allclose(direction, np.eye(3), atol=1e-4):
        raise RuntimeError(f"Non-identity working direction not supported for strict mapping: {path}")
    return meta


def component_table(mask: np.ndarray, meta: dict[str, Any]) -> list[dict[str, Any]]:
    labels, n = ndimage.label(mask > 0, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n == 0:
        return []
    origin = np.array(meta["working_origin_xyz"], dtype=float)
    spacing = np.array(meta["working_spacing_xyz"], dtype=float)
    rows: list[dict[str, Any]] = []
    slices = ndimage.find_objects(labels)
    counts = np.bincount(labels.ravel())
    for lab in range(1, n + 1):
        slc = slices[lab - 1]
        if slc is None:
            continue
        z0, z1 = slc[0].start, slc[0].stop - 1
        y0, y1 = slc[1].start, slc[1].stop - 1
        x0, x1 = slc[2].start, slc[2].stop - 1
        center_idx_xyz = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0])
        center_phys = origin + center_idx_xyz * spacing
        voxels = int(counts[lab])
        rows.append(
            {
                "label": lab,
                "voxels": voxels,
                "volume_ml": voxels * float(np.prod(spacing)) / 1000.0,
                "bbox_center_x_mm": float(center_phys[0]),
                "bbox_center_y_mm": float(center_phys[1]),
                "bbox_center_z_mm": float(center_phys[2]),
                "bbox_extent_x_mm": float((x1 - x0 + 1) * spacing[0]),
                "bbox_extent_y_mm": float((y1 - y0 + 1) * spacing[1]),
                "bbox_extent_z_mm": float((z1 - z0 + 1) * spacing[2]),
            }
        )
    # Match the V3.3 bridge/ROI geometry generator exactly: np.argsort(sizes)[::-1].
    rows_by_label = {int(row["label"]): row for row in rows}
    sorted_labels = (np.argsort(counts[1:])[::-1] + 1).tolist()
    rows = [rows_by_label[int(label)] for label in sorted_labels if int(label) in rows_by_label]
    for idx, row in enumerate(rows, start=1):
        row["component_rank_by_volume"] = idx
    return rows


def roi_component_mask(
    comp_labels: np.ndarray,
    comp_label: int,
    center_xyz: tuple[float, float, float],
    meta: dict[str, Any],
) -> np.ndarray:
    origin = np.array(meta["working_origin_xyz"], dtype=float)
    spacing = np.array(meta["working_spacing_xyz"], dtype=float)
    cx, cy, cz = center_xyz
    offset = (np.arange(ROI_N, dtype=np.float64) - (ROI_N - 1) / 2.0) * ROI_SPACING_MM
    xs = cx + offset
    ys = cy + offset
    zs = cz + offset
    xi = np.rint((xs - origin[0]) / spacing[0]).astype(np.int64)
    yi = np.rint((ys - origin[1]) / spacing[1]).astype(np.int64)
    zi = np.rint((zs - origin[2]) / spacing[2]).astype(np.int64)
    valid_x = (xi >= 0) & (xi < comp_labels.shape[2])
    valid_y = (yi >= 0) & (yi < comp_labels.shape[1])
    valid_z = (zi >= 0) & (zi < comp_labels.shape[0])
    safe_x = np.clip(xi, 0, comp_labels.shape[2] - 1)
    safe_y = np.clip(yi, 0, comp_labels.shape[1] - 1)
    safe_z = np.clip(zi, 0, comp_labels.shape[0] - 1)
    sampled = comp_labels[safe_z[:, None, None], safe_y[None, :, None], safe_x[None, None, :]]
    valid = valid_z[:, None, None] & valid_y[None, :, None] & valid_x[None, None, :]
    return (sampled == comp_label) & valid


def quantize(values: np.ndarray) -> np.ndarray:
    lo, hi = FIXED_QUANTIZATION_RANGE
    clipped = np.clip(values, lo, hi)
    q = np.floor((clipped - lo) / max(hi - lo, 1e-12) * FIXED_GRAY_LEVELS).astype(np.int16) + 1
    return np.clip(q, 1, FIXED_GRAY_LEVELS)


def safe_float(v: Any) -> float:
    try:
        out = float(v)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def first_order_features(vals: np.ndarray) -> dict[str, float]:
    vals = vals.astype(np.float64, copy=False)
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    centered = vals - mean
    p = np.percentile(vals, [10, 25, 50, 75, 90])
    hist, _ = np.histogram(vals, bins=FIXED_GRAY_LEVELS, range=FIXED_QUANTIZATION_RANGE)
    prob = hist.astype(np.float64) / max(hist.sum(), 1)
    prob = prob[prob > 0]
    return {
        "first_mean": mean,
        "first_std": std,
        "first_min": float(np.min(vals)),
        "first_max": float(np.max(vals)),
        "first_median": float(p[2]),
        "first_p10": float(p[0]),
        "first_p25": float(p[1]),
        "first_p75": float(p[3]),
        "first_p90": float(p[4]),
        "first_iqr": float(p[3] - p[1]),
        "first_mad": float(np.mean(np.abs(centered))),
        "first_energy": float(np.sum(vals * vals)),
        "first_entropy": float(-np.sum(prob * np.log2(prob))),
        "first_skew": float(np.mean(centered**3) / (std**3 + 1e-12)),
        "first_kurtosis": float(np.mean(centered**4) / (std**4 + 1e-12)),
    }


def shape_features(mask: np.ndarray, spacing_mm: float) -> dict[str, float]:
    coords = np.argwhere(mask)
    count = int(coords.shape[0])
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    ext = (maxs - mins + 1).astype(np.float64) * spacing_mm
    volume_ml = count * (spacing_mm**3) / 1000.0
    bbox_volume_ml = float(np.prod(ext) / 1000.0)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    surface_voxels = int(np.count_nonzero(mask & ~eroded))
    centroid = coords.mean(axis=0)
    patch_center = np.array([(ROI_N - 1) / 2.0] * 3)
    centroid_offset = (centroid - patch_center) * spacing_mm
    return {
        "shape_voxel_count": float(count),
        "shape_volume_ml": float(volume_ml),
        "shape_bbox_extent_x_mm": float(ext[2]),
        "shape_bbox_extent_y_mm": float(ext[1]),
        "shape_bbox_extent_z_mm": float(ext[0]),
        "shape_bbox_volume_ml": bbox_volume_ml,
        "shape_extent_fraction": float(volume_ml / (bbox_volume_ml + 1e-12)),
        "shape_surface_voxel_count": float(surface_voxels),
        "shape_surface_to_volume": float(surface_voxels / max(count, 1)),
        "shape_centroid_offset_x_mm": float(centroid_offset[2]),
        "shape_centroid_offset_y_mm": float(centroid_offset[1]),
        "shape_centroid_offset_z_mm": float(centroid_offset[0]),
    }


def crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask)
    mins = np.maximum(coords.min(axis=0) - 1, 0)
    maxs = np.minimum(coords.max(axis=0) + 2, np.array(mask.shape))
    slc = tuple(slice(int(mins[i]), int(maxs[i])) for i in range(3))
    return arr[slc], mask[slc]


def glcm_features(q: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    mat = np.zeros((FIXED_GRAY_LEVELS, FIXED_GRAY_LEVELS), dtype=np.float64)
    for axis in range(3):
        sl1 = [slice(None)] * 3
        sl2 = [slice(None)] * 3
        sl1[axis] = slice(0, -1)
        sl2[axis] = slice(1, None)
        valid = mask[tuple(sl1)] & mask[tuple(sl2)]
        a = q[tuple(sl1)][valid] - 1
        b = q[tuple(sl2)][valid] - 1
        if a.size:
            np.add.at(mat, (a, b), 1)
            np.add.at(mat, (b, a), 1)
    if mat.sum() == 0:
        return {f"glcm_{k}": 0.0 for k in ["contrast", "dissimilarity", "homogeneity", "asm", "energy", "entropy", "correlation"]}
    p = mat / mat.sum()
    i, j = np.indices(p.shape)
    mu_i = float((p * i).sum())
    mu_j = float((p * j).sum())
    sd_i = math.sqrt(float((p * (i - mu_i) ** 2).sum()))
    sd_j = math.sqrt(float((p * (j - mu_j) ** 2).sum()))
    nz = p[p > 0]
    return {
        "glcm_contrast": float((p * (i - j) ** 2).sum()),
        "glcm_dissimilarity": float((p * np.abs(i - j)).sum()),
        "glcm_homogeneity": float((p / (1.0 + (i - j) ** 2)).sum()),
        "glcm_asm": float((p * p).sum()),
        "glcm_energy": float(math.sqrt((p * p).sum())),
        "glcm_entropy": float(-np.sum(nz * np.log2(nz))),
        "glcm_correlation": float(((p * (i - mu_i) * (j - mu_j)).sum()) / (sd_i * sd_j + 1e-12)),
    }


def glrlm_features(q: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    run_lengths: list[int] = []
    run_grays: list[int] = []
    for axis in range(3):
        moved_q = np.moveaxis(q, axis, -1)
        moved_m = np.moveaxis(mask, axis, -1)
        for idx in np.ndindex(moved_m.shape[:-1]):
            mline = moved_m[idx]
            if not mline.any():
                continue
            qline = moved_q[idx]
            start = None
            last_g = None
            for pos, in_mask in enumerate(mline):
                g = int(qline[pos]) if in_mask else None
                if in_mask and start is None:
                    start = pos
                    last_g = g
                elif in_mask and g != last_g:
                    run_lengths.append(pos - start)
                    run_grays.append(last_g if last_g is not None else 0)
                    start = pos
                    last_g = g
                elif (not in_mask) and start is not None:
                    run_lengths.append(pos - start)
                    run_grays.append(last_g if last_g is not None else 0)
                    start = None
                    last_g = None
            if start is not None:
                run_lengths.append(len(mline) - start)
                run_grays.append(last_g if last_g is not None else 0)
    if not run_lengths:
        return {k: 0.0 for k in ["glrlm_run_count", "glrlm_sre", "glrlm_lre", "glrlm_gln", "glrlm_rln", "glrlm_run_percentage"]}
    lengths = np.asarray(run_lengths, dtype=np.float64)
    grays = np.asarray(run_grays, dtype=np.int16)
    n_runs = float(len(lengths))
    gl_counts = np.array([np.sum(grays == g) for g in range(1, FIXED_GRAY_LEVELS + 1)], dtype=np.float64)
    rl_counts = np.array(list(Counter(lengths.astype(int)).values()), dtype=np.float64)
    return {
        "glrlm_run_count": n_runs,
        "glrlm_sre": float(np.mean(1.0 / (lengths**2))),
        "glrlm_lre": float(np.mean(lengths**2)),
        "glrlm_gln": float(np.sum(gl_counts**2) / n_runs),
        "glrlm_rln": float(np.sum(rl_counts**2) / n_runs),
        "glrlm_run_percentage": float(n_runs / max(np.count_nonzero(mask), 1)),
    }


def glszm_features(q: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    sizes: list[int] = []
    grays: list[int] = []
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    for g in range(1, FIXED_GRAY_LEVELS + 1):
        lab, n = ndimage.label(mask & (q == g), structure=structure)
        if n == 0:
            continue
        counts = np.bincount(lab.ravel())[1:]
        sizes.extend([int(v) for v in counts if v > 0])
        grays.extend([g] * int(np.count_nonzero(counts > 0)))
    if not sizes:
        return {k: 0.0 for k in ["glszm_zone_count", "glszm_sze", "glszm_lze", "glszm_gln", "glszm_zsn", "glszm_zone_percentage"]}
    sizes_arr = np.asarray(sizes, dtype=np.float64)
    grays_arr = np.asarray(grays, dtype=np.int16)
    n_zones = float(len(sizes_arr))
    gl_counts = np.array([np.sum(grays_arr == g) for g in range(1, FIXED_GRAY_LEVELS + 1)], dtype=np.float64)
    size_counts = np.array(list(Counter(sizes_arr.astype(int)).values()), dtype=np.float64)
    return {
        "glszm_zone_count": n_zones,
        "glszm_sze": float(np.mean(1.0 / (sizes_arr**2))),
        "glszm_lze": float(np.mean(sizes_arr**2)),
        "glszm_gln": float(np.sum(gl_counts**2) / n_zones),
        "glszm_zsn": float(np.sum(size_counts**2) / n_zones),
        "glszm_zone_percentage": float(n_zones / max(np.count_nonzero(mask), 1)),
    }


def gldm_ngtdm_features(q: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    kernel = np.ones((3, 3, 3), dtype=np.float64)
    kernel[1, 1, 1] = 0.0
    mask_f = mask.astype(np.float64)
    neighbor_count = ndimage.convolve(mask_f, kernel, mode="constant", cval=0.0)
    q_f = q.astype(np.float64)
    neighbor_sum = ndimage.convolve(q_f * mask_f, kernel, mode="constant", cval=0.0)
    deps = neighbor_count[mask].astype(np.float64)
    if deps.size == 0:
        return {k: 0.0 for k in ["gldm_dependence_mean", "gldm_dependence_std", "gldm_sde", "gldm_lde", "gldm_dn", "ngtdm_coarseness", "ngtdm_contrast", "ngtdm_busyness"]}
    valid_ng = mask & (neighbor_count > 0)
    diffs = np.abs(q_f[valid_ng] - neighbor_sum[valid_ng] / np.maximum(neighbor_count[valid_ng], 1))
    dep_counts = np.array(list(Counter(deps.astype(int)).values()), dtype=np.float64)
    q_vals = q_f[mask]
    sum_diff = float(np.sum(diffs))
    return {
        "gldm_dependence_mean": float(np.mean(deps)),
        "gldm_dependence_std": float(np.std(deps)),
        "gldm_sde": float(np.mean(1.0 / ((deps + 1.0) ** 2))),
        "gldm_lde": float(np.mean((deps + 1.0) ** 2)),
        "gldm_dn": float(np.sum(dep_counts**2) / max(len(deps), 1)),
        "ngtdm_coarseness": float(1.0 / (sum_diff + 1e-12)),
        "ngtdm_contrast": float(np.var(q_vals) * sum_diff / max(len(q_vals), 1)),
        "ngtdm_busyness": float(sum_diff / (np.sum(np.abs(q_vals - np.mean(q_vals))) + 1e-12)),
    }


def extract_component_features(roi: np.ndarray, lesion_mask: np.ndarray) -> dict[str, float]:
    if tuple(roi.shape) != TENSOR_SHAPE:
        raise RuntimeError(f"Bad ROI tensor shape: {roi.shape}")
    arr = roi[0].astype(np.float32, copy=False)
    if lesion_mask.shape != arr.shape:
        raise RuntimeError(f"Mask/ROI shape mismatch: {lesion_mask.shape} vs {arr.shape}")
    if not np.isfinite(arr).all():
        raise RuntimeError("Non-finite values in ROI tensor.")
    if lesion_mask.dtype != bool:
        lesion_mask = lesion_mask.astype(bool)
    if np.count_nonzero(lesion_mask) <= 0:
        raise RuntimeError("Mapped lesion mask has zero voxels.")
    cropped_arr, cropped_mask = crop_to_mask(arr, lesion_mask)
    vals = cropped_arr[cropped_mask]
    q = np.zeros(cropped_arr.shape, dtype=np.int16)
    q[cropped_mask] = quantize(vals)
    feats: dict[str, float] = {}
    feats.update(shape_features(cropped_mask, ROI_SPACING_MM))
    feats.update(first_order_features(vals))
    feats.update(glcm_features(q, cropped_mask))
    feats.update(glrlm_features(q, cropped_mask))
    feats.update(glszm_features(q, cropped_mask))
    feats.update(gldm_ngtdm_features(q, cropped_mask))
    return feats


def extract_radiomics(labels: pd.DataFrame, strict_roi: pd.DataFrame, seg_manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_path = DIRS["radiomics"] / "STRICT_OOF_PREDICTED_LESION_RADIOMICS_PATIENT_FEATURES_V3_4.csv"
    component_path = DIRS["radiomics"] / "STRICT_OOF_PREDICTED_LESION_RADIOMICS_COMPONENT_FEATURES_V3_4.csv"
    qc_path = DIRS["audit"] / "STRICT_OOF_PREDICTED_LESION_RADIOMICS_ALIGNMENT_QC_V3_4.csv"
    if feature_path.exists() and component_path.exists() and qc_path.exists():
        return pd.read_csv(feature_path), pd.read_csv(component_path), pd.read_csv(qc_path)

    id_to_source = dict(zip(labels["neutral_patient_id"], labels["source_neutral_imaging_id"]))
    seg_by_id = seg_manifest.set_index("neutral_patient_id")
    roi_by_id = {pid: g.sort_values("ROI_rank") for pid, g in strict_roi.groupby("neutral_patient_id", sort=False)}
    patient_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    all_feature_names: set[str] = set()

    for idx, pid in enumerate(labels["neutral_patient_id"].astype(str).tolist(), start=1):
        if idx % 25 == 0:
            print(f"[radiomics] {idx}/375", flush=True)
        seg = seg_by_id.loc[pid]
        empty = parse_bool(seg["empty_prediction"])
        patient_components: list[dict[str, Any]] = []
        if not empty:
            source_id = id_to_source[pid]
            meta = load_metadata(source_id)
            mask_path = path_from_rel(seg["prediction_mask_path"])
            mask = np.load(mask_path)
            expected_shape = tuple(int(x) for x in str(seg["shape_zyx"]).split("x"))
            if tuple(mask.shape) != expected_shape or tuple(mask.shape) != tuple(meta["working_shape_zyx"]):
                raise RuntimeError(f"Mask shape/metadata mismatch for {pid}: {mask.shape}, {expected_shape}, {meta['working_shape_zyx']}")
            labels_img, n_comp = ndimage.label(mask > 0, structure=np.ones((3, 3, 3), dtype=np.uint8))
            comps = component_table(mask, meta)
            declared = int(seg["predicted_component_count"])
            if declared != len(comps):
                raise RuntimeError(f"Component count mismatch for {pid}: manifest={declared} computed={len(comps)}")
            rows = roi_by_id.get(pid)
            if rows is None or rows.empty:
                raise RuntimeError(f"No strict ROI rows for non-empty predicted mask patient {pid}")
            for _, roi_row in rows.sort_values("ROI_rank").head(3).iterrows():
                rank = int(roi_row["predicted_component_rank"])
                if rank > len(comps):
                    continue
                comp = comps[rank - 1]
                center_xyz = (float(roi_row["center_X_mm"]), float(roi_row["center_Y_mm"]), float(roi_row["center_Z_mm"]))
                comp_center = np.array([comp["bbox_center_x_mm"], comp["bbox_center_y_mm"], comp["bbox_center_z_mm"]], dtype=float)
                center_dist = float(np.linalg.norm(np.array(center_xyz, dtype=float) - comp_center))
                if center_dist > 4.0:
                    raise RuntimeError(f"ROI/component center alignment failed for {pid} rank={rank}: distance_mm={center_dist:.3f}")
                lesion_mask = roi_component_mask(labels_img, int(comp["label"]), center_xyz, meta)
                vox = int(np.count_nonzero(lesion_mask))
                if vox <= 0:
                    raise RuntimeError(f"Mapped ROI lesion mask empty for {pid} rank={rank}")
                roi_path = path_from_rel(roi_row["tensor_relative_path"])
                roi = np.load(roi_path).astype(np.float32, copy=False)
                feats = extract_component_features(roi, lesion_mask)
                all_feature_names.update(feats.keys())
                component_record = {
                    "neutral_patient_id": pid,
                    "outer_fold": int(roi_row["outer_fold"]),
                    "component_rank": rank,
                    "roi_rank": int(roi_row["ROI_rank"]),
                    "component_volume_ml_from_2mm_mask": comp["volume_ml"],
                    "mapped_roi_mask_voxels": vox,
                    "mapped_roi_mask_volume_ml": vox * (ROI_SPACING_MM**3) / 1000.0,
                    "roi_component_center_distance_mm": center_dist,
                }
                component_record.update(feats)
                component_rows.append(component_record)
                patient_components.append(component_record)
                qc_rows.append(
                    {
                        "neutral_patient_id": pid,
                        "component_rank": rank,
                        "roi_tensor_shape_pass": tuple(roi.shape) == TENSOR_SHAPE,
                        "mapped_mask_shape": "144x144x144",
                        "mapped_mask_voxels": vox,
                        "mapped_mask_volume_ml": vox * (ROI_SPACING_MM**3) / 1000.0,
                        "center_distance_mm": center_dist,
                        "alignment_pass": True,
                    }
                )
        patient_record: dict[str, Any] = {
            "neutral_patient_id": pid,
            "radiomics_missing_indicator": 1 if len(patient_components) == 0 else 0,
            "predicted_component_count": int(seg["predicted_component_count"]),
            "empty_prediction": empty,
            "radiomics_valid_component_count": len(patient_components),
            "radiomics_total_component_volume_ml": float(sum(c["component_volume_ml_from_2mm_mask"] for c in patient_components)),
            "radiomics_largest_component_volume_ml": float(patient_components[0]["component_volume_ml_from_2mm_mask"]) if patient_components else np.nan,
            "radiomics_second_component_volume_ml": float(patient_components[1]["component_volume_ml_from_2mm_mask"]) if len(patient_components) > 1 else np.nan,
            "radiomics_third_component_volume_ml": float(patient_components[2]["component_volume_ml_from_2mm_mask"]) if len(patient_components) > 2 else np.nan,
        }
        total_vol = patient_record["radiomics_total_component_volume_ml"]
        patient_record["radiomics_largest_component_fraction"] = (
            patient_record["radiomics_largest_component_volume_ml"] / total_vol if total_vol and math.isfinite(total_vol) else np.nan
        )
        if patient_components:
            weights = np.array([c["component_volume_ml_from_2mm_mask"] for c in patient_components], dtype=np.float64)
            if weights.sum() <= 0:
                weights = np.ones(len(patient_components), dtype=np.float64)
            weights = weights / weights.sum()
            for feat in sorted(all_feature_names):
                vals = np.array([safe_float(c.get(feat, np.nan)) for c in patient_components], dtype=np.float64)
                patient_record[f"largest_{feat}"] = vals[0] if vals.size else np.nan
                valid = np.isfinite(vals)
                patient_record[f"vwm_{feat}"] = float(np.sum(vals[valid] * weights[valid]) / np.sum(weights[valid])) if valid.any() else np.nan
        patient_rows.append(patient_record)

    component_df = pd.DataFrame(component_rows)
    feature_cols = sorted([c for c in component_df.columns if c not in {"neutral_patient_id", "outer_fold", "component_rank", "roi_rank", "component_volume_ml_from_2mm_mask", "mapped_roi_mask_voxels", "mapped_roi_mask_volume_ml", "roi_component_center_distance_mm"}])
    final_rows: list[dict[str, Any]] = []
    for row in patient_rows:
        complete = dict(row)
        for feat in feature_cols:
            complete.setdefault(f"largest_{feat}", np.nan)
            complete.setdefault(f"vwm_{feat}", np.nan)
        final_rows.append(complete)
    feature_df = pd.DataFrame(final_rows)
    feature_df = feature_df.sort_values("neutral_patient_id").reset_index(drop=True)
    component_df = component_df.sort_values(["neutral_patient_id", "component_rank"]).reset_index(drop=True)
    qc_df = pd.DataFrame(qc_rows).sort_values(["neutral_patient_id", "component_rank"]).reset_index(drop=True)
    write_csv_stable(feature_df, feature_path)
    write_csv_stable(component_df, component_path)
    write_csv_stable(qc_df, qc_path)
    return feature_df, component_df, qc_df


def make_lr_pipeline(C: float, use_variance: bool = True) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if use_variance:
        steps.append(("variance", VarianceThreshold()))
    steps.extend(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=C,
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=20000,
                    random_state=SEED,
                ),
            ),
        ]
    )
    return Pipeline(steps)


def choose_c_and_fit(X: pd.DataFrame, y: np.ndarray, outer: int, use_variance: bool, model_name: str) -> tuple[Pipeline, float, pd.DataFrame]:
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + outer)
    records: list[dict[str, Any]] = []
    best_c = C_GRID[0]
    best_loss = float("inf")
    for C in C_GRID:
        losses = []
        for tr, va in splitter.split(X, y):
            pipe = make_lr_pipeline(C, use_variance=use_variance)
            pipe.fit(X.iloc[tr], y[tr])
            pred = np.clip(pipe.predict_proba(X.iloc[va])[:, 1], 1e-6, 1 - 1e-6)
            losses.append(log_loss(y[va], pred, labels=[0, 1]))
        mean_loss = float(np.mean(losses))
        records.append({"model": model_name, "outer_fold": outer, "C": C, "mean_inner_log_loss": mean_loss})
        if mean_loss < best_loss - 1e-12:
            best_loss = mean_loss
            best_c = C
    final = make_lr_pipeline(best_c, use_variance=use_variance)
    final.fit(X, y)
    return final, best_c, pd.DataFrame(records)


def model_radiomics(labels: pd.DataFrame, folds: pd.Series, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_name = "RADIOMICS_LR"
    rf_name = "RADIOMICS_RF"
    non_feature_cols = {"neutral_patient_id", "empty_prediction"}
    X_all = features.drop(columns=[c for c in non_feature_cols if c in features.columns]).copy()
    X_all = X_all.apply(pd.to_numeric, errors="coerce")
    y_all = labels.set_index("neutral_patient_id").loc[features["neutral_patient_id"], "label_binary"].to_numpy(dtype=int)
    ids = features["neutral_patient_id"].astype(str).to_numpy()
    oof_lr: list[dict[str, Any]] = []
    oof_rf: list[dict[str, Any]] = []
    cv_records = []
    coef_rows = []
    for outer in OUTER_FOLDS:
        test = folds.loc[ids].to_numpy(dtype=int) == outer
        train = ~test
        pipe, best_c, cv_df = choose_c_and_fit(X_all.loc[train], y_all[train], outer, use_variance=True, model_name=model_name)
        joblib.dump(pipe, DIRS["models"] / f"RADIOMICS_LR_OUTER_{outer:02d}_PIPELINE_V3_4.pkl")
        cv_records.append(cv_df.assign(selected_C=best_c))
        pred = pipe.predict_proba(X_all.loc[test])[:, 1]
        for pid, p in zip(ids[test], pred):
            oof_lr.append({"neutral_patient_id": pid, "outer_fold": outer, "model": model_name, "probability": float(p)})
        feat_names = np.asarray(X_all.columns)
        selected = pipe.named_steps["variance"].get_support()
        sel_names = feat_names[selected]
        coefs = pipe.named_steps["lr"].coef_.reshape(-1)
        for name, coef in zip(sel_names, coefs):
            coef_rows.append({"model": model_name, "outer_fold": outer, "feature": name, "standardized_coefficient": float(coef), "selected_C": best_c})

        rf = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("variance", VarianceThreshold()),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=1000,
                        max_features="sqrt",
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        random_state=SEED + outer,
                        n_jobs=max(1, min(8, os.cpu_count() or 1)),
                    ),
                ),
            ]
        )
        rf.fit(X_all.loc[train], y_all[train])
        joblib.dump(rf, DIRS["models"] / f"RADIOMICS_RF_OUTER_{outer:02d}_PIPELINE_V3_4.pkl")
        pred_rf = rf.predict_proba(X_all.loc[test])[:, 1]
        for pid, p in zip(ids[test], pred_rf):
            oof_rf.append({"neutral_patient_id": pid, "outer_fold": outer, "model": rf_name, "probability": float(p)})
    cv_all = pd.concat(cv_records, ignore_index=True)
    write_csv_stable(cv_all, DIRS["audit"] / "RADIOMICS_LR_INNER_CV_LOG_LOSS_V3_4.csv")
    coef_df = pd.DataFrame(coef_rows)
    write_csv_stable(coef_df, DIRS["results"] / "RADIOMICS_LR_OUTER_STANDARDIZED_COEFFICIENTS_V3_4.csv")
    return pd.DataFrame(oof_lr + oof_rf), coef_df


def import_formal_model() -> Any:
    sys.path.insert(0, str(SRC))
    from run_formal_highres_dual_backbone_classification_internal_training_v3_3 import V2InheritedHighresClassifier

    return V2InheritedHighresClassifier


def extract_embeddings_for_fold(
    backbone: str,
    outer: int,
    labels: pd.DataFrame,
    folds: pd.Series,
    train_context: pd.DataFrame,
    strict_context: pd.DataFrame,
) -> pd.DataFrame:
    import torch

    short = "RESNET" if backbone.startswith("ResNet") else "DENSENET"
    csv_path = DIRS["embeddings"] / f"{short}_OUTER{outer:02d}_PATIENT_EMBEDDINGS.csv"
    npy_path = DIRS["embeddings"] / f"{short}_OUTER{outer:02d}_PATIENT_EMBEDDINGS.npy"
    if csv_path.exists() and npy_path.exists():
        return pd.read_csv(csv_path)

    Model = import_formal_model()
    ckpt_path = checkpoint_path(backbone, outer)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = Model(backbone)
    model.load_state_dict(ckpt["model_state"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    ids = labels["neutral_patient_id"].astype(str).tolist()
    test_ids = set(folds[folds == outer].index.astype(str).tolist())
    train_ctx_outer = train_context[train_context["outer_fold"].astype(int) == outer].set_index("neutral_patient_id")
    strict_ctx_outer = strict_context[strict_context["outer_fold"].astype(int) == outer].set_index("neutral_patient_id")
    rows: list[dict[str, Any]] = []
    embeds: list[np.ndarray] = []
    with torch.inference_mode():
        for idx, pid in enumerate(ids, start=1):
            if idx % 50 == 0:
                print(f"[embedding] {backbone} outer={outer} {idx}/375", flush=True)
            ctx = strict_ctx_outer.loc[pid] if pid in test_ids else train_ctx_outer.loc[pid]
            paths = [path_from_rel(part) for part in str(ctx["ROI_paths"]).split("|") if part]
            patch_embs: list[np.ndarray] = []
            for path in paths:
                arr = np.load(path).astype(np.float32, copy=False)
                if tuple(arr.shape) != TENSOR_SHAPE:
                    raise RuntimeError(f"Bad tensor shape {arr.shape} for {path}")
                x = torch.from_numpy(arr[None, ...]).to(device=device, dtype=torch.float32)
                emb = model.base(x).detach().cpu().numpy().reshape(-1).astype(np.float32)
                patch_embs.append(emb)
            patient_emb = np.mean(np.stack(patch_embs, axis=0), axis=0).astype(np.float32)
            embeds.append(patient_emb)
            row = {"neutral_patient_id": pid, "outer_fold": outer, "embedding_role": "outer_test" if pid in test_ids else "outer_development"}
            row.update({f"emb_{i:04d}": float(v) for i, v in enumerate(patient_emb)})
            rows.append(row)
    arr = np.stack(embeds, axis=0)
    np.save(npy_path, arr)
    df = pd.DataFrame(rows)
    write_csv_stable(df, csv_path)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return df


def model_embeddings(labels: pd.DataFrame, folds: pd.Series, train_context: pd.DataFrame, strict_context: pd.DataFrame, backbone: str, model_name: str) -> pd.DataFrame:
    y_by_id = labels.set_index("neutral_patient_id")["label_binary"]
    oof_rows: list[dict[str, Any]] = []
    cv_records = []
    for outer in OUTER_FOLDS:
        emb_df = extract_embeddings_for_fold(backbone, outer, labels, folds, train_context, strict_context)
        emb_df = emb_df.set_index("neutral_patient_id")
        feature_cols = [c for c in emb_df.columns if c.startswith("emb_")]
        X = emb_df[feature_cols].apply(pd.to_numeric, errors="coerce")
        y = y_by_id.loc[X.index].to_numpy(dtype=int)
        test = folds.loc[X.index].to_numpy(dtype=int) == outer
        train = ~test
        pipe, best_c, cv_df = choose_c_and_fit(X.loc[train], y[train], outer, use_variance=False, model_name=model_name)
        joblib.dump(pipe, DIRS["models"] / f"{model_name}_OUTER_{outer:02d}_PIPELINE_V3_4.pkl")
        cv_records.append(cv_df.assign(selected_C=best_c))
        pred = pipe.predict_proba(X.loc[test])[:, 1]
        for pid, p in zip(X.index[test], pred):
            oof_rows.append({"neutral_patient_id": pid, "outer_fold": outer, "model": model_name, "probability": float(p)})
    write_csv_stable(pd.concat(cv_records, ignore_index=True), DIRS["audit"] / f"{model_name}_INNER_CV_LOG_LOSS_V3_4.csv")
    return pd.DataFrame(oof_rows)


def write_label_blind_oof(oof_long: pd.DataFrame) -> dict[str, str]:
    shas = {}
    for model, g in oof_long.groupby("model", sort=True):
        blind = g[["neutral_patient_id", "outer_fold", "probability"]].sort_values("neutral_patient_id").reset_index(drop=True)
        if len(blind) != 375 or blind["neutral_patient_id"].nunique() != 375 or blind["probability"].isna().any():
            raise RuntimeError(f"Invalid label-blind OOF for {model}")
        path = DIRS["oof"] / f"{model}_LABEL_BLIND_OOF_PREDICTIONS_V3_4.csv"
        write_csv_stable(blind, path)
        shas[model] = sha256_file(path)
    return shas


def metric_record(y: np.ndarray, p: np.ndarray, model: str) -> tuple[dict[str, Any], list[list[int]]]:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=int)
    cm = confusion_matrix(y, p >= 0.5, labels=[0, 1]).tolist()
    tn, fp = cm[0]
    fn, tp = cm[1]
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    acc = (tp + tn) / max(len(y), 1)
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=20000)
        cal.fit(logits, y)
        intercept = float(cal.intercept_[0])
        slope = float(cal.coef_[0, 0])
    except Exception:
        intercept = np.nan
        slope = np.nan
    return (
        {
            "model": model,
            "n": int(len(y)),
            "CAP": int(np.sum(y == 0)),
            "COP": int(np.sum(y == 1)),
            "AUROC": float(roc_auc_score(y, p)),
            "AUPRC": float(average_precision_score(y, p)),
            "Brier": float(brier_score_loss(y, p)),
            "BCE": float(log_loss(y, p, labels=[0, 1])),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "PPV": float(ppv),
            "NPV": float(npv),
            "accuracy": float(acc),
            "balanced_accuracy": float(np.nanmean([sens, spec])),
            "F1": float(f1),
            "calibration_intercept": intercept,
            "calibration_slope": slope,
        },
        cm,
    )


def evaluate(oof_wide: pd.DataFrame, labels: pd.DataFrame, folds: pd.Series) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = labels.set_index("neutral_patient_id").loc[oof_wide["neutral_patient_id"], "label_binary"].to_numpy(dtype=int)
    metrics = []
    conf = {}
    roc_rows = []
    pr_rows = []
    cal_rows = []
    fold_rows = []
    model_cols = [c for c in oof_wide.columns if c not in {"neutral_patient_id", "label_binary", "disease_label", "outer_fold"}]
    for model in model_cols:
        p = oof_wide[model].to_numpy(dtype=float)
        rec, cm = metric_record(y, p, model)
        metrics.append(rec)
        conf[model] = cm
        fpr, tpr, thr = roc_curve(y, p)
        roc_rows.extend({"model": model, "fpr": float(a), "tpr": float(b), "threshold": float(c)} for a, b, c in zip(fpr, tpr, thr))
        precision, recall, thresholds = precision_recall_curve(y, p)
        thr2 = list(thresholds) + [np.nan]
        pr_rows.extend({"model": model, "precision": float(a), "recall": float(b), "threshold": safe_float(c)} for a, b, c in zip(precision, recall, thr2))
        bins = pd.qcut(pd.Series(p), q=10, duplicates="drop")
        tmp = pd.DataFrame({"p": p, "y": y, "bin": bins})
        for b, g in tmp.groupby("bin", observed=True):
            cal_rows.append({"model": model, "calibration_type": "bin", "bin": str(b), "n": len(g), "mean_predicted": float(g["p"].mean()), "observed_fraction": float(g["y"].mean())})
        for pid, yy, pp in zip(oof_wide["neutral_patient_id"], y, p):
            cal_rows.append({"model": model, "calibration_type": "raw", "neutral_patient_id": pid, "label_binary": int(yy), "probability": float(pp)})
        for outer in OUTER_FOLDS:
            idx = folds.loc[oof_wide["neutral_patient_id"]].to_numpy(dtype=int) == outer
            yy = y[idx]
            pp = p[idx]
            fold_rows.append(
                {
                    "model": model,
                    "outer_fold": outer,
                    "n": int(idx.sum()),
                    "CAP": int(np.sum(yy == 0)),
                    "COP": int(np.sum(yy == 1)),
                    "AUROC": float(roc_auc_score(yy, pp)) if len(np.unique(yy)) == 2 else np.nan,
                    "AUPRC": float(average_precision_score(yy, pp)) if len(np.unique(yy)) == 2 else np.nan,
                    "Brier": float(brier_score_loss(yy, pp)),
                }
            )
    return pd.DataFrame(metrics), conf, pd.DataFrame(roc_rows), pd.DataFrame(pr_rows), pd.DataFrame(cal_rows), pd.DataFrame(fold_rows)


def paired_bootstrap(oof_wide: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    y = labels.set_index("neutral_patient_id").loc[oof_wide["neutral_patient_id"], "label_binary"].to_numpy(dtype=int)
    comparisons = [
        ("RADIOMICS_LR", "FORMAL_RESNET3D18_V3_3"),
        ("RADIOMICS_LR", "FORMAL_DENSENET3D121_V3_3"),
        ("RESNET_EMBEDDING_LR", "FORMAL_RESNET3D18_V3_3"),
        ("DENSENET_EMBEDDING_LR", "FORMAL_DENSENET3D121_V3_3"),
        ("FIXED_3WAY_IMAGE_ENSEMBLE", "FORMAL_RESNET3D18_V3_3"),
        ("FIXED_3WAY_IMAGE_ENSEMBLE", "FORMAL_DENSENET3D121_V3_3"),
    ]
    rng = np.random.default_rng(SEED)
    rows = []
    n = len(y)
    for a, b in comparisons:
        pa = oof_wide[a].to_numpy(dtype=float)
        pb = oof_wide[b].to_numpy(dtype=float)
        observed = float(roc_auc_score(y, pa) - roc_auc_score(y, pb))
        deltas = []
        skipped = 0
        for _ in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(y[idx])) < 2:
                skipped += 1
                continue
            deltas.append(float(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx])))
        arr = np.asarray(deltas, dtype=np.float64)
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "metric": "AUROC",
                "observed_delta_a_minus_b": observed,
                "bootstrap_replicates_requested": N_BOOTSTRAP,
                "bootstrap_replicates_used": int(arr.size),
                "bootstrap_skipped_single_class": int(skipped),
                "delta_mean": float(np.mean(arr)),
                "delta_ci025": float(np.quantile(arr, 0.025)),
                "delta_ci975": float(np.quantile(arr, 0.975)),
                "p_two_sided_sign": float(2 * min(np.mean(arr <= 0), np.mean(arr >= 0))),
            }
        )
    return pd.DataFrame(rows)


def coefficient_stability(coef_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat, g in coef_df.groupby("feature", sort=True):
        coefs = g["standardized_coefficient"].to_numpy(dtype=float)
        rows.append(
            {
                "feature": feat,
                "n_outer_selected": int(len(coefs)),
                "median_abs_standardized_coefficient": float(np.median(np.abs(coefs))),
                "mean_standardized_coefficient": float(np.mean(coefs)),
                "positive_fraction": float(np.mean(coefs > 0)),
                "negative_fraction": float(np.mean(coefs < 0)),
                "sign_consistency": float(max(np.mean(coefs > 0), np.mean(coefs < 0))),
            }
        )
    return pd.DataFrame(rows).sort_values(["sign_consistency", "median_abs_standardized_coefficient"], ascending=[False, False])


def make_figures(metrics: pd.DataFrame, roc_df: pd.DataFrame, pr_df: pd.DataFrame, fold_df: pd.DataFrame, coef_stability: pd.DataFrame, oof: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 6))
    for model, g in roc_df.groupby("model"):
        auc = metrics.set_index("model").loc[model, "AUROC"]
        plt.plot(g["fpr"], g["tpr"], label=f"{model} ({auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(DIRS["figures"] / "ALL_MODEL_ROC_V3_4.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 6))
    for model, g in pr_df.groupby("model"):
        ap = metrics.set_index("model").loc[model, "AUPRC"]
        plt.plot(g["recall"], g["precision"], label=f"{model} ({ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(DIRS["figures"] / "ALL_MODEL_PR_V3_4.png", dpi=180)
    plt.close()

    summary = metrics[["model", "AUROC", "AUPRC", "Brier"]].melt("model", var_name="metric", value_name="value")
    write_csv_stable(summary, DIRS["figures"] / "AUROC_AUPRC_BRIER_SUMMARY_SOURCE_V3_4.csv")
    pivot = metrics.set_index("model")[["AUROC", "AUPRC", "Brier"]]
    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_ylim(0, max(1.0, float(pivot[["AUROC", "AUPRC"]].max().max()) + 0.05))
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(DIRS["figures"] / "AUROC_AUPRC_BRIER_SUMMARY_V3_4.png", dpi=180)
    plt.close()

    top = coef_stability.head(20).copy()
    write_csv_stable(top, DIRS["figures"] / "RADIOMICS_TOP_STABLE_COEFFICIENTS_SOURCE_V3_4.csv")
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"][::-1], top["median_abs_standardized_coefficient"][::-1])
    plt.xlabel("Median |standardized coefficient|")
    plt.tight_layout()
    plt.savefig(DIRS["figures"] / "RADIOMICS_TOP_STABLE_COEFFICIENTS_V3_4.png", dpi=180)
    plt.close()

    write_csv_stable(fold_df, DIRS["figures"] / "FIVE_FOLD_AUROC_VARIATION_SOURCE_V3_4.csv")
    plt.figure(figsize=(9, 5))
    for model, g in fold_df.groupby("model"):
        plt.plot(g["outer_fold"], g["AUROC"], marker="o", label=model)
    plt.xlabel("Outer fold")
    plt.ylabel("AUROC")
    plt.ylim(0, 1)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(DIRS["figures"] / "FIVE_FOLD_AUROC_VARIATION_V3_4.png", dpi=180)
    plt.close()

    paired = oof[["neutral_patient_id", "label_binary", "FORMAL_RESNET3D18_V3_3", "RESNET_EMBEDDING_LR", "FORMAL_DENSENET3D121_V3_3", "DENSENET_EMBEDDING_LR"]].copy()
    write_csv_stable(paired, DIRS["figures"] / "FORMAL_CNN_VS_EMBEDDING_LR_PAIRED_PROBABILITY_SOURCE_V3_4.csv")
    plt.figure(figsize=(6, 6))
    plt.scatter(paired["FORMAL_RESNET3D18_V3_3"], paired["RESNET_EMBEDDING_LR"], s=14, alpha=0.65, label="ResNet")
    plt.scatter(paired["FORMAL_DENSENET3D121_V3_3"], paired["DENSENET_EMBEDDING_LR"], s=14, alpha=0.65, label="DenseNet")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Formal CNN probability")
    plt.ylabel("Embedding LR probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DIRS["figures"] / "FORMAL_CNN_VS_EMBEDDING_LR_PAIRED_PROBABILITY_V3_4.png", dpi=180)
    plt.close()


def interpretation(metrics: pd.DataFrame) -> dict[str, str]:
    m = metrics.set_index("model")["AUROC"].to_dict()
    q1 = "YES" if m["RADIOMICS_LR"] > max(m["FORMAL_RESNET3D18_V3_3"], m["FORMAL_DENSENET3D121_V3_3"]) + 0.02 else ("MIXED" if m["RADIOMICS_LR"] > min(m["FORMAL_RESNET3D18_V3_3"], m["FORMAL_DENSENET3D121_V3_3"]) + 0.02 else "NO")
    res_better = m["RESNET_EMBEDDING_LR"] > m["FORMAL_RESNET3D18_V3_3"] + 0.02
    den_better = m["DENSENET_EMBEDDING_LR"] > m["FORMAL_DENSENET3D121_V3_3"] + 0.02
    q2 = "YES" if res_better and den_better else ("MIXED" if res_better or den_better else "NO")
    component_best = max(m["RADIOMICS_LR"], m["RESNET_EMBEDDING_LR"], m["DENSENET_EMBEDDING_LR"])
    q3 = "YES" if m["FIXED_3WAY_IMAGE_ENSEMBLE"] > component_best + 0.02 else ("MIXED" if m["FIXED_3WAY_IMAGE_ENSEMBLE"] >= component_best - 0.01 else "NO")
    if all(m[k] <= 0.60 for k in ["RADIOMICS_LR", "RESNET_EMBEDDING_LR", "DENSENET_EMBEDDING_LR", "FIXED_3WAY_IMAGE_ENSEMBLE"]):
        overall = "CURRENT_CT_SIGNAL_LIMITED"
    elif q1 == "YES":
        overall = "LOW_CAPACITY_HANDCRAFTED_FEATURES_MORE_STABLE_IN_SMALL_SAMPLE"
    elif q2 == "YES":
        overall = "CNN_REPRESENTATION_USEFUL_BUT_HEAD_UNSTABLE"
    elif q3 == "YES":
        overall = "COMPLEMENTARY_IMAGE_REPRESENTATIONS"
    else:
        overall = "MIXED"
    return {"QUESTION_1": q1, "QUESTION_2": q2, "QUESTION_3": q3, "OVERALL_INTERPRETATION": overall}


def write_result_lock(
    metrics: pd.DataFrame,
    conf: dict[str, Any],
    comparisons: pd.DataFrame,
    protocol_sha: str,
    pre_hashes: dict[str, str],
    post_hashes: dict[str, str],
    label_blind_shas: dict[str, str],
    interp: dict[str, str],
) -> tuple[Path, Path, str]:
    best = metrics[metrics["model"].isin(["RADIOMICS_LR", "RADIOMICS_RF", "RESNET_EMBEDDING_LR", "DENSENET_EMBEDDING_LR", "FIXED_3WAY_IMAGE_ENSEMBLE"])].sort_values("AUROC", ascending=False).iloc[0]
    modified = {k: pre_hashes[k] != post_hashes[k] for k in pre_hashes}
    payload = {
        "task_name": "PERFORMANCE_INFORMED_EXPLORATORY_IMAGE_REPRESENTATION_RESCUE_V3_4",
        "completed_at": now_iso(),
        "analysis_type": "PERFORMANCE_INFORMED_SECONDARY_EXPLORATORY",
        "development_timing": "AFTER_V3_3_FORMAL_IMAGE_RESULTS_REVIEWED",
        "formal_v3_3_primary_result_replaced": "NO",
        "patients": 375,
        "outer_folds": 5,
        "metrics": metrics.to_dict(orient="records"),
        "confusion_matrices": conf,
        "paired_comparisons": comparisons.to_dict(orient="records"),
        "best_exploratory_model": str(best["model"]),
        "best_exploratory_AUROC": float(best["AUROC"]),
        "best_model_selected_post_hoc_for_description_only": "YES",
        "formal_primary_model_changed": "NO",
        "interpretation": interp,
        "external_accessed": "NO",
        "clinical_accessed": "NO",
        "multimodal_started": "NO",
        "expert_gt_mask_used_as_classifier_feature": "NO",
        "strict_oof_predicted_mask_used": "YES",
        "train_fold_only_preprocessing": "YES",
        "outer_test_leakage_detected": "NO",
        "label_blind_oof_sha256": label_blind_shas,
        "v3_4_protocol_sha256": protocol_sha,
        "v3_3_pre_run_protected_sha256": pre_hashes,
        "v3_3_post_run_protected_sha256": post_hashes,
        "v3_3_formal_result_lock_modified": "YES" if modified[str(FORMAL_RESULT_LOCK_PATH)] else "NO",
        "v3_3_oof_predictions_modified": "YES" if modified[str(RESNET_FORMAL_PRED_PATH)] or modified[str(DENSENET_FORMAL_PRED_PATH)] else "NO",
        "v3_3_registry_modified": "YES" if modified[str(FORMAL_REGISTRY_PATH)] or modified[str(FORMAL_REGISTRY_JSON_PATH)] else "NO",
        "v3_3_checkpoint_modified": "YES" if any(
            pre_hashes[k] != post_hashes[k]
            for k in pre_hashes
            if k.endswith("INFERENCE_ONLY_BEST_STATE_DICT.pt")
        ) else "NO",
        "v3_3_protocol_modified": "YES" if modified[str(FORMAL_PROTOCOL_PATH)] else "NO",
        "final_status": "EXPLORATORY_IMAGE_REPRESENTATION_RESCUE_V3_4_COMPLETE",
        "next_action": "HUMAN_REVIEW_EXPLORATORY_IMAGE_RESCUE_RESULTS",
    }
    json_path = DIRS["results"] / "EXPLORATORY_IMAGE_REPRESENTATION_RESCUE_RESULT_LOCK_V3_4.json"
    md_path = DIRS["results"] / "EXPLORATORY_IMAGE_REPRESENTATION_RESCUE_RESULT_LOCK_V3_4.md"
    write_json(json_path, payload)
    result_sha = sha256_file(json_path)
    lines = [
        "# EXPLORATORY_IMAGE_REPRESENTATION_RESCUE_RESULT_LOCK_V3_4",
        "",
        f"- analysis_type: {payload['analysis_type']}",
        "- formal_v3_3_primary_result_replaced: NO",
        f"- best_exploratory_model: {payload['best_exploratory_model']}",
        f"- best_exploratory_AUROC: {payload['best_exploratory_AUROC']:.6f}",
        f"- QUESTION_1: {interp['QUESTION_1']}",
        f"- QUESTION_2: {interp['QUESTION_2']}",
        f"- QUESTION_3: {interp['QUESTION_3']}",
        f"- OVERALL_INTERPRETATION: {interp['OVERALL_INTERPRETATION']}",
        f"- V3.4 protocol SHA256: {protocol_sha}",
        f"- V3.4 result lock SHA256: {result_sha}",
        "- final_status: EXPLORATORY_IMAGE_REPRESENTATION_RESCUE_V3_4_COMPLETE",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path, result_sha


def final_console(metrics: pd.DataFrame, conf: dict[str, Any], interp: dict[str, str], protocol_sha: str, result_sha: str) -> str:
    m = metrics.set_index("model")
    best = metrics[metrics["model"].isin(["RADIOMICS_LR", "RADIOMICS_RF", "RESNET_EMBEDDING_LR", "DENSENET_EMBEDDING_LR", "FIXED_3WAY_IMAGE_ENSEMBLE"])].sort_values("AUROC", ascending=False).iloc[0]

    def block(model: str, title: str, include_conf: bool = False) -> list[str]:
        rows = [
            "",
            f"{title}:",
            "",
            "OOF patients:",
            str(int(m.loc[model, "n"])),
            "",
            "AUROC:",
            f"{float(m.loc[model, 'AUROC']):.6f}",
            "",
            "AUPRC:",
            f"{float(m.loc[model, 'AUPRC']):.6f}",
            "",
            "Brier:",
            f"{float(m.loc[model, 'Brier']):.6f}",
        ]
        if include_conf:
            rows += ["", "confusion:", json.dumps(conf[model], separators=(",", ":"))]
        return rows

    lines = [
        "================ EXPLORATORY IMAGE RESCUE V3.4 ================",
        "",
        "analysis type:",
        "PERFORMANCE_INFORMED_SECONDARY_EXPLORATORY",
        "",
        "formal V3.3 primary result replaced:",
        "NO",
        "",
        "patients:",
        "375",
        "",
        "outer folds:",
        "5",
        "",
        "FORMAL CNN REFERENCE:",
        "",
        "ResNet AUROC:",
        f"{float(m.loc['FORMAL_RESNET3D18_V3_3', 'AUROC']):.6f}",
        "",
        "DenseNet AUROC:",
        f"{float(m.loc['FORMAL_DENSENET3D121_V3_3', 'AUROC']):.6f}",
    ]
    lines += block("RADIOMICS_LR", "RADIOMICS LR", include_conf=True)
    lines += block("RADIOMICS_RF", "RADIOMICS RF")
    lines += block("RESNET_EMBEDDING_LR", "RESNET EMBEDDING LR")
    lines += block("DENSENET_EMBEDDING_LR", "DENSENET EMBEDDING LR")
    lines += block("FIXED_3WAY_IMAGE_ENSEMBLE", "FIXED 3-WAY IMAGE ENSEMBLE")
    lines += [
        "",
        "FORMAL CNN 50/50 REFERENCE:",
        "",
        "AUROC:",
        f"{float(m.loc['FORMAL_CNN_50_50_REFERENCE', 'AUROC']):.6f}",
        "",
        "AUPRC:",
        f"{float(m.loc['FORMAL_CNN_50_50_REFERENCE', 'AUPRC']):.6f}",
        "",
        "Brier:",
        f"{float(m.loc['FORMAL_CNN_50_50_REFERENCE', 'Brier']):.6f}",
        "",
        "BEST EXPLORATORY MODEL:",
        str(best["model"]),
        "",
        "BEST EXPLORATORY AUROC:",
        f"{float(best['AUROC']):.6f}",
        "",
        "BUT:",
        "",
        "best model selected post hoc for description only:",
        "YES",
        "",
        "formal primary model changed:",
        "NO",
        "",
        "QUESTION 1:",
        f"RADIOMICS BETTER THAN END-TO-END CNN? {interp['QUESTION_1']}",
        "",
        "QUESTION 2:",
        f"EMBEDDING LR BETTER THAN ORIGINAL CNN HEAD? {interp['QUESTION_2']}",
        "",
        "QUESTION 3:",
        f"IMAGE REPRESENTATIONS COMPLEMENTARY? {interp['QUESTION_3']}",
        "",
        "OVERALL INTERPRETATION:",
        interp["OVERALL_INTERPRETATION"],
        "",
        "external accessed:",
        "NO",
        "",
        "clinical accessed:",
        "NO",
        "",
        "multimodal started:",
        "NO",
        "",
        "expert GT mask used as classifier feature:",
        "NO",
        "",
        "strict OOF predicted mask used:",
        "YES",
        "",
        "train-fold-only preprocessing:",
        "YES",
        "",
        "outer-test leakage detected:",
        "NO",
        "",
        "V3.4 protocol SHA256:",
        protocol_sha,
        "",
        "V3.4 result lock SHA256:",
        result_sha,
        "",
        "final status:",
        "EXPLORATORY_IMAGE_REPRESENTATION_RESCUE_V3_4_COMPLETE",
        "",
        "next action:",
        "HUMAN_REVIEW_EXPLORATORY_IMAGE_RESCUE_RESULTS",
        "",
        "===============================================================",
    ]
    return "\n".join(lines)


def main() -> int:
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    pre_hashes = protected_hashes()
    try:
        import radiomics  # noqa: F401

        pyradiomics_available = True
    except Exception:
        pyradiomics_available = False
    _, _, protocol_sha = write_protocol(pre_hashes, pyradiomics_available)
    print(f"[protocol] sha256={protocol_sha}", flush=True)

    labels = pd.read_csv(LABEL_PATH)
    strict_context = pd.read_csv(STRICT_CONTEXT_PATH)
    strict_roi = pd.read_csv(STRICT_ROI_PATH)
    train_context = pd.read_csv(TRAIN_CONTEXT_PATH)
    seg_manifest = pd.read_csv(SEG_MANIFEST_PATH)
    validate_inputs(labels, strict_context, strict_roi)
    folds = strict_context.set_index("neutral_patient_id")["outer_fold"].astype(int)

    features, component_features, qc = extract_radiomics(labels, strict_roi, seg_manifest)
    if len(features) != 375 or features["neutral_patient_id"].nunique() != 375:
        raise RuntimeError("Radiomics feature table does not contain 375 unique patients.")
    print(f"[radiomics] features={features.shape[1]-1} components={len(component_features)} qc_rows={len(qc)}", flush=True)

    radiomics_oof, coef_df = model_radiomics(labels, folds, features)
    resnet_oof = model_embeddings(labels, folds, train_context, strict_context, "ResNet3D-18", "RESNET_EMBEDDING_LR")
    densenet_oof = model_embeddings(labels, folds, train_context, strict_context, "DenseNet3D-121", "DENSENET_EMBEDDING_LR")

    res_formal = pd.read_csv(RESNET_FORMAL_PRED_PATH)[["neutral_patient_id", "outer_fold", "patient_probability"]].rename(columns={"patient_probability": "FORMAL_RESNET3D18_V3_3"})
    den_formal = pd.read_csv(DENSENET_FORMAL_PRED_PATH)[["neutral_patient_id", "patient_probability"]].rename(columns={"patient_probability": "FORMAL_DENSENET3D121_V3_3"})

    oof_long = pd.concat([radiomics_oof, resnet_oof, densenet_oof], ignore_index=True)
    wide = oof_long.pivot(index="neutral_patient_id", columns="model", values="probability").reset_index()
    wide = wide.merge(res_formal, on="neutral_patient_id", how="inner").merge(den_formal, on="neutral_patient_id", how="inner")
    wide["FIXED_3WAY_IMAGE_ENSEMBLE"] = (wide["RADIOMICS_LR"] + wide["RESNET_EMBEDDING_LR"] + wide["DENSENET_EMBEDDING_LR"]) / 3.0
    wide["FORMAL_CNN_50_50_REFERENCE"] = (wide["FORMAL_RESNET3D18_V3_3"] + wide["FORMAL_DENSENET3D121_V3_3"]) / 2.0
    wide = wide.merge(labels[["neutral_patient_id", "disease_label", "label_binary"]], on="neutral_patient_id", how="inner")
    wide = wide.sort_values("neutral_patient_id").reset_index(drop=True)
    if len(wide) != 375 or wide["neutral_patient_id"].nunique() != 375:
        raise RuntimeError("Final OOF table does not contain 375 unique patients.")

    label_blind_long = pd.concat(
        [
            oof_long,
            wide[["neutral_patient_id", "outer_fold", "FIXED_3WAY_IMAGE_ENSEMBLE"]].rename(columns={"FIXED_3WAY_IMAGE_ENSEMBLE": "probability"}).assign(model="FIXED_3WAY_IMAGE_ENSEMBLE"),
            wide[["neutral_patient_id", "outer_fold", "FORMAL_CNN_50_50_REFERENCE"]].rename(columns={"FORMAL_CNN_50_50_REFERENCE": "probability"}).assign(model="FORMAL_CNN_50_50_REFERENCE"),
            wide[["neutral_patient_id", "outer_fold", "FORMAL_RESNET3D18_V3_3"]].rename(columns={"FORMAL_RESNET3D18_V3_3": "probability"}).assign(model="FORMAL_RESNET3D18_V3_3"),
            wide[["neutral_patient_id", "outer_fold", "FORMAL_DENSENET3D121_V3_3"]].rename(columns={"FORMAL_DENSENET3D121_V3_3": "probability"}).assign(model="FORMAL_DENSENET3D121_V3_3"),
        ],
        ignore_index=True,
    )
    label_blind_shas = write_label_blind_oof(label_blind_long)

    write_csv_stable(wide, DIRS["results"] / "EXPLORATORY_IMAGE_OOF_PATIENT_PREDICTIONS_V3_4.csv")
    metrics, conf, roc_df, pr_df, cal_df, fold_df = evaluate(wide, labels, folds)
    metrics = metrics.sort_values("AUROC", ascending=False).reset_index(drop=True)
    write_csv_stable(metrics, DIRS["results"] / "EXPLORATORY_IMAGE_MODEL_METRICS_V3_4.csv")
    write_json(DIRS["results"] / "EXPLORATORY_IMAGE_MODEL_METRICS_V3_4.json", {"metrics": metrics.to_dict(orient="records")})
    write_json(DIRS["results"] / "EXPLORATORY_IMAGE_CONFUSION_MATRICES_V3_4.json", conf)
    write_csv_stable(roc_df, DIRS["results"] / "EXPLORATORY_IMAGE_ROC_RAW_V3_4.csv")
    write_csv_stable(pr_df, DIRS["results"] / "EXPLORATORY_IMAGE_PR_RAW_V3_4.csv")
    write_csv_stable(cal_df, DIRS["results"] / "EXPLORATORY_IMAGE_CALIBRATION_RAW_V3_4.csv")
    write_csv_stable(fold_df, DIRS["results"] / "EXPLORATORY_IMAGE_OUTER_FOLD_METRICS_V3_4.csv")
    comparisons = paired_bootstrap(wide, labels)
    write_csv_stable(comparisons, DIRS["results"] / "EXPLORATORY_IMAGE_PAIRED_COMPARISONS_V3_4.csv")
    coef_stab = coefficient_stability(coef_df)
    write_csv_stable(coef_stab, DIRS["results"] / "RADIOMICS_FEATURE_COEFFICIENT_STABILITY_V3_4.csv")
    make_figures(metrics, roc_df, pr_df, fold_df, coef_stab, wide)

    post_hashes = protected_hashes()
    interp = interpretation(metrics)
    _, _, result_sha = write_result_lock(metrics, conf, comparisons, protocol_sha, pre_hashes, post_hashes, label_blind_shas, interp)
    console = final_console(metrics, conf, interp, protocol_sha, result_sha)
    (DIRS["results"] / "EXPLORATORY_IMAGE_RESCUE_V3_4_FINAL_CONSOLE.txt").write_text(console + "\n", encoding="utf-8")
    print(console, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
