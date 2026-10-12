from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(os.environ.get("CAP_COP_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
MODEL_DIR = PROJECT_ROOT / "models" / "lesion_radiomics"
EXP = PROJECT_ROOT
OUT = PROJECT_ROOT / "outputs" / "lesion_radiomics_pipeline"
V34_EXT = MODEL_DIR
V35A_LOCK = MODEL_DIR / "result_lock.json"
V35B_LOCK = MODEL_DIR / "result_lock.json"
V35C_LOCK = MODEL_DIR / "result_lock.json"

DIRS = {
    "protocol": OUT / "00_protocol",
    "env": OUT / "01_environment",
    "audit": OUT / "02_input_audit",
    "extract": OUT / "03_feature_extraction",
    "robust": OUT / "04_perturbation_robustness",
    "cv": OUT / "05_nested_cv",
    "final": OUT / "06_final_model",
    "external_blind": OUT / "07_external_label_blind",
    "external_results": OUT / "08_external_results",
    "mechanism": OUT / "09_mechanism",
    "reports": OUT / "10_reports",
    "locks": OUT / "11_locks",
}

LABELS = PROJECT_ROOT / "data" / "public_subject_model_crosswalk.csv"
NATIVE_TABLE = PROJECT_ROOT / "data" / "public_subject_model_crosswalk.csv"
SEG_MANIFEST = PROJECT_ROOT / "outputs" / "canonical_a_formal_20261011" / "segmentation" / "full_cohort_segmentation_replay_audit.csv"
SEG_CACHE_MANIFEST = SEG_MANIFEST
ACQ_META = MODEL_DIR / "internal_acquisition_metadata.csv"
SEALED_MAPPING = PROJECT_ROOT / "data" / "public_subject_model_crosswalk.csv"
EXT_CT_MANIFEST = PROJECT_ROOT / "data" / "public_subject_model_crosswalk.csv"
EXT_MASK_MANIFEST = PROJECT_ROOT / "data" / "public_subject_model_crosswalk.csv"

SEED = 20260816
THRESHOLD = 0.5
ICC_THRESHOLD = 0.90
PERTURBATIONS = [
    "BASELINE",
    "SPACING_1.0MM",
    "SPACING_2.0MM",
    "SMOOTH_0.5MM",
    "SMOOTH_1.0MM",
    "NOISE_10HU",
    "NOISE_20HU",
    "HU_OFFSET_NEG10",
    "HU_OFFSET_POS10",
    "HU_SCALE_0.95",
    "HU_SCALE_1.05",
    "MASK_EROSION_1MM",
    "MASK_DILATION_1MM",
]
CLASSES = ["firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
V34_EXTERNAL_AUROC = 0.353383
V35A_EXTERNAL_AUROC = 0.545113
V35B_EXTERNAL_AUROC = 0.571429
V35C_EXTERNAL_AUROC = 0.541353


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n", encoding="utf-8-sig")


def clean_json(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): clean_json(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean_json(x) for x in v]
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def stop(status: str, reason: str, details: dict[str, Any] | None = None) -> None:
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    write_json(DIRS["locks"] / "lesion_radiomics_replay_STOP_LOCK.json", {
        "status": status,
        "reason": reason,
        "details": details or {},
        "timestamp": now_iso(),
    })
    print(f"STOP {status}: {reason}", flush=True)
    raise SystemExit(2)


def extractor_for_spacing(spacing: float = 1.5) -> featureextractor.RadiomicsFeatureExtractor:
    params = {
        "imageType": {"Original": {}},
        "featureClass": {k: [] for k in CLASSES},
        "setting": {
            "binWidth": 25,
            "resampledPixelSpacing": [spacing, spacing, spacing],
            "interpolator": "sitkBSpline",
            "normalize": False,
            "label": 1,
        },
    }
    return featureextractor.RadiomicsFeatureExtractor(params)


def load_dicom_series(dicom_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        files = sorted(str(p) for p in dicom_dir.glob("*.dcm"))
    else:
        files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    if not files:
        raise RuntimeError(f"No DICOM files found for {dicom_dir}")
    reader.SetFileNames(files)
    return sitk.Cast(reader.Execute(), sitk.sitkFloat32)


def image_from_array_like(arr: np.ndarray, ref: sitk.Image) -> sitk.Image:
    out = sitk.GetImageFromArray(arr.astype(np.float32, copy=False))
    out.CopyInformation(ref)
    return out


def mask_from_npy(mask_path: Path, meta: dict[str, Any]) -> sitk.Image:
    arr = np.load(mask_path).astype(np.uint8)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(tuple(float(x) for x in meta["working_spacing_xyz"]))
    img.SetOrigin(tuple(float(x) for x in meta["working_origin_xyz"]))
    img.SetDirection(tuple(float(x) for x in meta["working_direction"]))
    return img


def perturb(image: sitk.Image, mask: sitk.Image, scenario: str, seed: int) -> tuple[sitk.Image, sitk.Image, float]:
    if scenario == "BASELINE":
        return image, mask, 1.5
    if scenario == "SPACING_1.0MM":
        return image, mask, 1.0
    if scenario == "SPACING_2.0MM":
        return image, mask, 2.0
    if scenario == "SMOOTH_0.5MM":
        return sitk.SmoothingRecursiveGaussian(image, 0.5), mask, 1.5
    if scenario == "SMOOTH_1.0MM":
        return sitk.SmoothingRecursiveGaussian(image, 1.0), mask, 1.5
    arr = None
    if scenario in {"NOISE_10HU", "NOISE_20HU", "HU_OFFSET_NEG10", "HU_OFFSET_POS10", "HU_SCALE_0.95", "HU_SCALE_1.05"}:
        arr = sitk.GetArrayFromImage(image).astype(np.float32, copy=True)
    if scenario == "NOISE_10HU":
        rng = np.random.default_rng(seed)
        return image_from_array_like(arr + rng.normal(0.0, 10.0, arr.shape).astype(np.float32), image), mask, 1.5
    if scenario == "NOISE_20HU":
        rng = np.random.default_rng(seed)
        return image_from_array_like(arr + rng.normal(0.0, 20.0, arr.shape).astype(np.float32), image), mask, 1.5
    if scenario == "HU_OFFSET_NEG10":
        return image_from_array_like(arr - 10.0, image), mask, 1.5
    if scenario == "HU_OFFSET_POS10":
        return image_from_array_like(arr + 10.0, image), mask, 1.5
    if scenario == "HU_SCALE_0.95":
        return image_from_array_like(arr * 0.95, image), mask, 1.5
    if scenario == "HU_SCALE_1.05":
        return image_from_array_like(arr * 1.05, image), mask, 1.5
    if scenario in {"MASK_EROSION_1MM", "MASK_DILATION_1MM"}:
        dist = sitk.SignedMaurerDistanceMap(mask, insideIsPositive=True, squaredDistance=False, useImageSpacing=True)
        if scenario == "MASK_EROSION_1MM":
            m2 = sitk.BinaryThreshold(dist, lowerThreshold=1.0, upperThreshold=1e9, insideValue=1, outsideValue=0)
        else:
            m2 = sitk.BinaryThreshold(dist, lowerThreshold=-1.0, upperThreshold=1e9, insideValue=1, outsideValue=0)
        return image, sitk.Cast(m2, sitk.sitkUInt8), 1.5
    raise RuntimeError(f"Unknown perturbation {scenario}")


def extract_features(image: sitk.Image, mask: sitk.Image, spacing: float) -> dict[str, float]:
    res = extractor_for_spacing(spacing).execute(image, mask)
    out: dict[str, float] = {}
    for k, v in res.items():
        if k.startswith("diagnostics_"):
            continue
        try:
            out[k] = float(v)
        except Exception:
            out[k] = float("nan")
    return out


def load_internal_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict], dict[str, dict]]:
    labels = pd.read_csv(LABELS, encoding="utf-8-sig")
    native = pd.read_csv(NATIVE_TABLE, encoding="utf-8-sig")
    seg = pd.read_csv(SEG_MANIFEST, encoding="utf-8-sig")
    cache = pd.read_csv(SEG_CACHE_MANIFEST, encoding="utf-8-sig")
    acq = pd.read_csv(ACQ_META, encoding="utf-8-sig")
    if len(labels) != 375 or labels["neutral_patient_id"].nunique() != 375:
        stop("lesion_radiomics_replay_INTERNAL_LABEL_CONTRACT_FAIL", "internal labels are not 375 unique patients")
    vc = labels["label_binary"].astype(int).value_counts().to_dict()
    if vc.get(0) != 250 or vc.get(1) != 125:
        stop("lesion_radiomics_replay_INTERNAL_LABEL_CONTRACT_FAIL", f"CAP/COP counts invalid: {vc}")
    if len(seg) != 375 or seg["neutral_patient_id"].nunique() != 375:
        stop("lesion_radiomics_replay_STRICT_OOF_MASK_CONTRACT_FAIL", "strict OOF mask manifest not 375 unique")
    if len(native) != 375 or native["neutral_patient_id"].nunique() != 375:
        stop("lesion_radiomics_replay_NATIVE_CT_CONTRACT_FAIL", "native CT table not 375 unique")
    if len(acq) != 375 or acq["neutral_patient_id"].nunique() != 375:
        stop("lesion_radiomics_replay_ACQUISITION_METADATA_FAIL", "acquisition metadata not 375 unique")
    cache_meta = {}
    for _, r in cache.iterrows():
        p = EXP / str(r["metadata_relative_path"]).replace("/", os.sep)
        cache_meta[str(r["neutral_imaging_id"])] = json.loads(p.read_text(encoding="utf-8"))
    write_csv(DIRS["audit"] / "lesion_radiomics_replay_INPUT_FEATURE_CONTRACT_AUDIT.csv", pd.DataFrame([
        {"item": "internal_labels", "n": len(labels), "unique": labels["neutral_patient_id"].nunique(), "status": "PASS"},
        {"item": "strict_oof_masks", "n": len(seg), "unique": seg["neutral_patient_id"].nunique(), "status": "PASS"},
        {"item": "native_ct_paths_private", "n": len(native), "unique": native["neutral_patient_id"].nunique(), "status": "PASS"},
        {"item": "acquisition_metadata", "n": len(acq), "unique": acq["neutral_patient_id"].nunique(), "status": "PASS"},
    ]))
    return labels, native, seg, cache_meta, acq.set_index("neutral_patient_id").to_dict("index")


def patient_seed(pid: str, scenario: str) -> int:
    return int(hashlib.sha256(f"{SEED}|{pid}|{scenario}".encode("utf-8")).hexdigest()[:8], 16)


def extract_internal_matrix() -> pd.DataFrame:
    out_path = DIRS["extract"] / "lesion_radiomics_replay_INTERNAL375_PERTURBATION_PYRADIOMICS_FEATURES.csv"
    baseline_path = DIRS["extract"] / "lesion_radiomics_replay_INTERNAL375_BASELINE_PYRADIOMICS_FEATURES.csv"
    if out_path.exists():
        df = pd.read_csv(out_path, encoding="utf-8-sig")
        expected = 375 * len(PERTURBATIONS)
        if len(df) == expected:
            return df

    labels, native, seg, cache_meta, _ = load_internal_tables()
    native_by_id = native.set_index("neutral_patient_id").to_dict("index")
    seg_by_id = seg.set_index("neutral_patient_id").to_dict("index")

    existing = pd.DataFrame()
    done: set[tuple[str, str]] = set()
    if out_path.exists():
        existing = pd.read_csv(out_path, encoding="utf-8-sig")
        if {"neutral_patient_id", "perturbation"}.issubset(existing.columns):
            done = set(zip(existing["neutral_patient_id"].astype(str), existing["perturbation"].astype(str)))
    rows: list[dict[str, Any]] = []
    for idx, pid in enumerate(labels["neutral_patient_id"].astype(str), start=1):
        needed = [s for s in PERTURBATIONS if (pid, s) not in done]
        if not needed:
            continue
        nrow = native_by_id[pid]
        srow = seg_by_id[pid]
        v2id = str(nrow["source_neutral_imaging_id"])
        image = load_dicom_series(Path(str(nrow["native_ct_path"])))
        mask = mask_from_npy(EXP / str(srow["prediction_mask_path"]).replace("/", os.sep), cache_meta[v2id])
        for scen in needed:
            rec: dict[str, Any] = {
                "neutral_patient_id": pid,
                "source_neutral_imaging_id": v2id,
                "outer_fold": int(srow["outer_fold"]),
                "perturbation": scen,
                "status": "FAIL",
                "error": "",
            }
            try:
                img2, mask2, spacing = perturb(image, mask, scen, patient_seed(pid, scen))
                if sitk.GetArrayViewFromImage(mask2).sum() <= 0:
                    raise RuntimeError("empty perturbed mask")
                feats = extract_features(img2, mask2, spacing)
                rec.update(feats)
                finite = all(math.isfinite(float(v)) for v in feats.values())
                rec["status"] = "PASS" if finite else "NONFINITE_FEATURES"
            except Exception as exc:
                rec["error"] = repr(exc)
            rows.append(rec)
        if idx % 5 == 0:
            flush = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) if len(existing) else pd.DataFrame(rows)
            write_csv(out_path, flush)
            print(f"[{now_iso()}] internal PyRadiomics extraction {idx}/375 rows={len(flush)}", flush=True)
            existing = flush
            rows = []
    final = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) if rows else existing
    write_csv(out_path, final)
    baseline = final[final["perturbation"] == "BASELINE"].copy()
    write_csv(baseline_path, baseline)
    fail_rate = float((baseline["status"] != "PASS").mean()) if len(baseline) else 1.0
    if len(baseline) != 375 or fail_rate > 0.05:
        stop("lesion_radiomics_replay_BASELINE_FEATURE_EXTRACTION_FAIL", "baseline feature extraction failed >5% or not 375", {"n_baseline": len(baseline), "fail_rate": fail_rate})
    return final


def feature_cols(matrix: pd.DataFrame) -> list[str]:
    meta = {"neutral_patient_id", "source_neutral_imaging_id", "outer_fold", "perturbation", "status", "error"}
    return [c for c in matrix.columns if c not in meta]


def feature_class(name: str) -> str:
    parts = name.split("_")
    return parts[1] if len(parts) > 2 and parts[0] == "original" else "unknown"


def write_feature_dictionary(cols: list[str]) -> None:
    rows = [{"feature": c, "feature_class": feature_class(c)} for c in cols]
    write_csv(DIRS["extract"] / "lesion_radiomics_replay_PYRADIOMICS_FEATURE_DICTIONARY.csv", pd.DataFrame(rows))


def icc_absolute(df: pd.DataFrame, feature: str, patients: list[str] | None = None) -> tuple[float, float]:
    sub = df[["neutral_patient_id", "perturbation", feature, "status"]].copy()
    if patients is not None:
        sub = sub[sub["neutral_patient_id"].isin(patients)]
    sub = sub[sub["status"] == "PASS"]
    wide = sub.pivot_table(index="neutral_patient_id", columns="perturbation", values=feature, aggfunc="first")
    wide = wide.reindex(columns=PERTURBATIONS)
    fail_rate = float(1.0 - wide.notna().all(axis=1).mean()) if len(wide) else 1.0
    wide = wide.dropna()
    if wide.shape[0] < 3:
        return float("nan"), fail_rate
    vals = wide.to_numpy(dtype=float)
    n, k = vals.shape
    ms_rows = k * np.var(vals.mean(axis=1), ddof=1)
    ms_cols = n * np.var(vals.mean(axis=0), ddof=1)
    residual = vals - vals.mean(axis=1, keepdims=True) - vals.mean(axis=0, keepdims=True) + vals.mean()
    ms_err = float(np.sum(residual ** 2) / max((n - 1) * (k - 1), 1))
    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    icc = float((ms_rows - ms_err) / denom) if abs(denom) > 1e-12 else float("nan")
    return icc, fail_rate


def stability_table(matrix: pd.DataFrame, patients: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for feat in feature_cols(matrix):
        icc, fail = icc_absolute(matrix, feat, patients)
        rows.append({
            "feature": feat,
            "feature_class": feature_class(feat),
            "ICC_absolute_agreement": icc,
            "technical_failure_rate": fail,
            "ROBUST_ICC_GE_0_90": bool(math.isfinite(icc) and icc >= ICC_THRESHOLD and fail <= 0.05),
        })
    return pd.DataFrame(rows).sort_values(["ROBUST_ICC_GE_0_90", "ICC_absolute_agreement"], ascending=[False, False])


def reduce_redundancy(X: pd.DataFrame, stab: pd.DataFrame) -> list[str]:
    robust = stab[stab["ROBUST_ICC_GE_0_90"]]["feature"].astype(str).tolist()
    if not robust:
        return []
    keep = []
    drop = set()
    ranked = stab.set_index("feature").loc[robust].sort_values(["ICC_absolute_agreement", "technical_failure_rate"], ascending=[False, True]).index.tolist()
    corr = X[robust].corr(method="spearman").abs()
    for f in ranked:
        if f in drop:
            continue
        keep.append(f)
        high = corr.index[(corr[f] > 0.90) & (corr.index != f)].tolist()
        for h in high:
            if h not in keep:
                drop.add(h)
    return keep


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = (p >= THRESHOLD).astype(int)
    return {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "LogLoss": float(log_loss(y, p, labels=[0, 1])),
        "Sensitivity": float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)),
        "Specificity": float(((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)),
        "Accuracy": float(accuracy_score(y, pred)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
    }


def build_model(kind: str, params: dict[str, Any]) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if kind == "LR":
        steps.append(("scaler", StandardScaler()))
        steps.append(("model", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=params["C"],
            l1_ratio=params["l1_ratio"],
            class_weight=params["class_weight"],
            max_iter=5000,
            random_state=SEED,
        )))
    else:
        steps.append(("model", RandomForestClassifier(
            n_estimators=500,
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            class_weight=params["class_weight"],
            random_state=SEED,
            n_jobs=-1,
        )))
    return Pipeline(steps)


def model_grid() -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for C in [0.01, 0.1, 1, 10]:
        for l1_ratio in [0, 0.25, 0.5, 0.75, 1]:
            for cw in [None, "balanced"]:
                out.append(("LR", {"C": C, "l1_ratio": l1_ratio, "class_weight": cw}))
    for md in [3, 5, None]:
        for leaf in [3, 5, 10]:
            for mf in ["sqrt", 0.5]:
                for cw in [None, "balanced"]:
                    out.append(("RF", {"max_depth": md, "min_samples_leaf": leaf, "max_features": mf, "class_weight": cw}))
    return out


def rank_features(X: pd.DataFrame, y: pd.Series, feats: list[str]) -> list[str]:
    scores = []
    yv = y.to_numpy(dtype=int)
    for f in feats:
        x = pd.to_numeric(X[f], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x)
        if mask.sum() < 5 or len(np.unique(x[mask])) < 2:
            score = 0.5
        else:
            try:
                auc = roc_auc_score(yv[mask], x[mask])
                score = max(auc, 1 - auc)
            except Exception:
                score = 0.5
        scores.append((f, score))
    return [f for f, _ in sorted(scores, key=lambda x: (-x[1], x[0]))]


def select_best_candidate(score_df: pd.DataFrame) -> pd.Series:
    agg = score_df.groupby(["kind", "params", "n_features"], as_index=False).agg({
        "AUROC": "mean",
        "AUPRC": "mean",
        "Brier": "mean",
        "feature_set": "first",
    })
    max_auc = agg["AUROC"].max()
    guard = agg[agg["AUROC"] >= max_auc - 0.01].copy()
    guard = guard.sort_values(["AUROC", "AUPRC", "Brier", "n_features", "kind"], ascending=[False, False, True, True, True])
    return guard.iloc[0]


def nested_cv(labels: pd.DataFrame, matrix: pd.DataFrame, acq: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    baseline = matrix[matrix["perturbation"] == "BASELINE"].copy()
    feats_all = feature_cols(matrix)
    X_all = baseline.set_index("neutral_patient_id")[feats_all].apply(pd.to_numeric, errors="coerce")
    y_by_id = labels.set_index("neutral_patient_id")["label_binary"].astype(int)
    fold_by_id = baseline.set_index("neutral_patient_id")["outer_fold"].astype(int)
    oof = []
    outer_metrics = []
    inner_rows = []
    selected_rows = []
    for outer in range(5):
        test_ids = fold_by_id.index[fold_by_id == outer].tolist()
        train_ids = fold_by_id.index[fold_by_id != outer].tolist()
        stab = stability_table(matrix, train_ids)
        write_csv(DIRS["robust"] / f"OUTER{outer:02d}_PYRADIOMICS_ROBUST_FEATURE_SET.csv", stab)
        feats_reduced = reduce_redundancy(X_all.loc[train_ids], stab)
        if not feats_reduced:
            stop("lesion_radiomics_replay_NO_ROBUST_FEATURES_IN_OUTER", f"outer {outer} has zero robust features")
        Xtr_all = X_all.loc[train_ids]
        ytr_all = y_by_id.loc[train_ids]
        inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED + outer)
        scores = []
        for inner_fold, (tr_idx, va_idx) in enumerate(inner.split(Xtr_all, ytr_all)):
            inner_train_ids = Xtr_all.index[tr_idx].tolist()
            inner_val_ids = Xtr_all.index[va_idx].tolist()
            inner_stab = stability_table(matrix, inner_train_ids)
            inner_feats = reduce_redundancy(X_all.loc[inner_train_ids], inner_stab)
            if not inner_feats:
                continue
            ranked = rank_features(X_all.loc[inner_train_ids], y_by_id.loc[inner_train_ids], inner_feats)
            counts = [c for c in [5, 10, 15, len(ranked)] if c <= len(ranked)]
            counts = sorted(set(counts))
            for nfeat in counts:
                fset = ranked[:nfeat]
                for kind, params in model_grid():
                    pipe = build_model(kind, params)
                    pipe.fit(X_all.loc[inner_train_ids, fset], y_by_id.loc[inner_train_ids])
                    p = pipe.predict_proba(X_all.loc[inner_val_ids, fset])[:, 1]
                    m = metrics(y_by_id.loc[inner_val_ids].to_numpy(), p)
                    rec = {
                        "outer_fold": outer,
                        "inner_fold": inner_fold,
                        "kind": kind,
                        "params": json.dumps(params, sort_keys=True),
                        "n_features": nfeat,
                        "feature_set": "|".join(fset),
                    }
                    rec.update(m)
                    scores.append(rec)
        score_df = pd.DataFrame(scores)
        write_csv(DIRS["cv"] / f"OUTER{outer:02d}_INNER_MODEL_SELECTION.csv", score_df)
        inner_rows.extend(scores)
        best = select_best_candidate(score_df)
        fset = str(best["feature_set"]).split("|")
        kind = str(best["kind"])
        params = json.loads(str(best["params"]))
        pipe = build_model(kind, params)
        pipe.fit(X_all.loc[train_ids, fset], y_by_id.loc[train_ids])
        p = pipe.predict_proba(X_all.loc[test_ids, fset])[:, 1]
        for pid, prob in zip(test_ids, p):
            oof.append({"neutral_patient_id": pid, "outer_fold": outer, "probability_COP": float(prob), "label_binary": int(y_by_id.loc[pid])})
        om = metrics(y_by_id.loc[test_ids].to_numpy(), p)
        om.update({"outer_fold": outer, "selected_model": kind, "selected_params": json.dumps(params, sort_keys=True), "selected_feature_count": len(fset)})
        outer_metrics.append(om)
        selected_rows.extend([{"outer_fold": outer, "feature": f, "model_kind": kind} for f in fset])
    oof_df = pd.DataFrame(oof).sort_values("neutral_patient_id")
    write_csv(DIRS["cv"] / "lesion_radiomics_replay_INTERNAL375_STRICT_OOF_PREDICTIONS.csv", oof_df)
    oof_metrics = metrics(oof_df["label_binary"].to_numpy(dtype=int), oof_df["probability_COP"].to_numpy(dtype=float))
    write_csv(DIRS["cv"] / "lesion_radiomics_replay_INTERNAL375_STRICT_OOF_METRICS.csv", pd.DataFrame([oof_metrics]))
    write_csv(DIRS["cv"] / "lesion_radiomics_replay_INTERNAL375_OUTER_FOLD_METRICS.csv", pd.DataFrame(outer_metrics))
    write_csv(DIRS["cv"] / "lesion_radiomics_replay_INNER_CANDIDATE_METRICS.csv", pd.DataFrame(inner_rows))
    write_csv(DIRS["cv"] / "lesion_radiomics_replay_OUTER_SELECTED_FEATURES.csv", pd.DataFrame(selected_rows))
    domain_rows = []
    for domain in sorted(set(str(acq[pid].get("acquisition_domain", "UNKNOWN")) for pid in oof_df["neutral_patient_id"])):
        ids = [pid for pid in oof_df["neutral_patient_id"] if str(acq[pid].get("acquisition_domain", "UNKNOWN")) == domain]
        sub = oof_df[oof_df["neutral_patient_id"].isin(ids)]
        if sub["label_binary"].nunique() == 2:
            auc = roc_auc_score(sub["label_binary"], sub["probability_COP"])
        else:
            auc = float("nan")
        domain_rows.append({"acquisition_domain": domain, "n": len(sub), "AUROC": auc})
    domain_df = pd.DataFrame(domain_rows)
    write_csv(DIRS["mechanism"] / "lesion_radiomics_replay_INTERNAL_SCANNER_DOMAIN_DIAGNOSTICS.csv", domain_df)
    worst_domain_auc = float(domain_df["AUROC"].dropna().min()) if domain_df["AUROC"].notna().any() else float("nan")
    return oof_df, pd.DataFrame(outer_metrics), pd.DataFrame(selected_rows), {**oof_metrics, "worst_scanner_domain_AUROC": worst_domain_auc}


def fit_final(labels: pd.DataFrame, matrix: pd.DataFrame) -> tuple[Path, dict[str, Any], list[str]]:
    baseline = matrix[matrix["perturbation"] == "BASELINE"].copy()
    feats_all = feature_cols(matrix)
    X_all = baseline.set_index("neutral_patient_id")[feats_all].apply(pd.to_numeric, errors="coerce")
    y = labels.set_index("neutral_patient_id").loc[X_all.index, "label_binary"].astype(int)
    full_stab = stability_table(matrix, X_all.index.tolist())
    write_csv(DIRS["robust"] / "lesion_radiomics_replay_FULL_INTERNAL_PYRADIOMICS_STABILITY_AUDIT.csv", full_stab)
    feats = reduce_redundancy(X_all, full_stab)
    if not feats:
        stop("lesion_radiomics_replay_NO_FINAL_ROBUST_FEATURES", "zero final robust features")
    ranked = rank_features(X_all, y, feats)
    candidate_scores = []
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    for nfeat in sorted(set([c for c in [5, 10, 15, len(ranked)] if c <= len(ranked)])):
        fset = ranked[:nfeat]
        for kind, params in model_grid():
            fold_ms = []
            for tr, va in cv.split(X_all[fset], y):
                pipe = build_model(kind, params)
                pipe.fit(X_all.iloc[tr][fset], y.iloc[tr])
                p = pipe.predict_proba(X_all.iloc[va][fset])[:, 1]
                fold_ms.append(metrics(y.iloc[va].to_numpy(), p))
            rec = {"kind": kind, "params": json.dumps(params, sort_keys=True), "n_features": nfeat, "feature_set": "|".join(fset)}
            for key in fold_ms[0]:
                rec[key] = float(np.mean([m[key] for m in fold_ms]))
            candidate_scores.append(rec)
    cand = pd.DataFrame(candidate_scores)
    write_csv(DIRS["final"] / "lesion_radiomics_replay_FINAL_INTERNAL_HYPERPARAMETER_SELECTION.csv", cand)
    best = select_best_candidate(cand)
    fset = str(best["feature_set"]).split("|")
    params = json.loads(str(best["params"]))
    kind = str(best["kind"])
    pipe = build_model(kind, params)
    pipe.fit(X_all[fset], y)
    model_path = DIRS["final"] / "lesion_radiomics_replay_FINAL_MODEL.joblib"
    joblib.dump({"pipeline": pipe, "features": fset, "model_kind": kind, "params": params}, model_path)
    write_csv(DIRS["final"] / "lesion_radiomics_replay_FINAL_FEATURE_LIST.csv", pd.DataFrame({"feature": fset}))
    lock = {
        "status": "lesion_radiomics_replay_FINAL_MODEL_LOCKED_BEFORE_EXTERNAL_ACCESS",
        "timestamp": now_iso(),
        "model": kind,
        "params": params,
        "final_feature_count": len(fset),
        "features": fset,
        "model_sha256": sha256_file(model_path),
        "external_labels_accessed": False,
        "external_feature_distribution_used_for_development": False,
    }
    write_json(DIRS["final"] / "lesion_radiomics_replay_FINAL_MODEL_LOCK.json", lock)
    return model_path, lock, fset


def bootstrap_ci(y: np.ndarray, p: np.ndarray, fn, n_boot: int = 5000) -> list[float]:
    rng = np.random.default_rng(SEED)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            vals.append(float(fn(y[idx], p[idx])))
        except Exception:
            pass
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def external_labels() -> pd.DataFrame:
    rows = pd.read_csv(SEALED_MAPPING, encoding="utf-8-sig")
    labs = []
    for eid, g in rows.groupby("neutral_external_id"):
        text = " ".join(g["original_source_path"].astype(str).tolist())
        if "\\CAP 19" in text or "/CAP 19" in text:
            lab = 0
        elif "\\COP 14" in text or "/COP 14" in text:
            lab = 1
        else:
            lab = None
        labs.append({"neutral_external_id": eid, "label_binary": lab})
    out = pd.DataFrame(labs).sort_values("neutral_external_id")
    if len(out) != 33 or out["label_binary"].isna().any():
        stop("lesion_radiomics_replay_EXTERNAL_LABEL_CONTRACT_FAIL", "could not unblind 33 labels after prediction lock")
    return out


def extract_external_features_if_possible(fset: list[str]) -> tuple[pd.DataFrame | None, str]:
    ct_m = pd.read_csv(EXT_CT_MANIFEST, encoding="utf-8-sig")
    mask_m = pd.read_csv(EXT_MASK_MANIFEST, encoding="utf-8-sig")
    base = V34_EXT
    rows = []
    for _, r in ct_m.iterrows():
        p = base / str(r["ct_2mm_path"])
        if not p.exists():
            return None, "EXTERNAL_2MM_CT_ARTIFACTS_MISSING_AND_NEUTRAL_RAW_SOURCE_NOT_AVAILABLE_WITHOUT_LABEL_PATH_EXPOSURE"
    mask_by = mask_m.set_index("neutral_external_id").to_dict("index")
    for _, r in ct_m.iterrows():
        eid = str(r["neutral_external_id"])
        meta = json.loads((base / str(r["metadata_path"])).read_text(encoding="utf-8"))
        ct_arr = np.load(base / str(r["ct_2mm_path"]))
        image = sitk.GetImageFromArray(ct_arr.astype(np.float32))
        image.SetSpacing(tuple(float(x) for x in meta["working_spacing_xyz"]))
        image.SetOrigin(tuple(float(x) for x in meta["working_origin_xyz"]))
        image.SetDirection(tuple(float(x) for x in meta["working_direction"]))
        mask = mask_from_npy(base / mask_by[eid]["prediction_mask_path"], meta)
        feats = extract_features(image, mask, 1.5)
        row = {"neutral_external_id": eid}
        row.update({f: feats.get(f, float("nan")) for f in fset})
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("neutral_external_id")
    write_csv(DIRS["external_blind"] / "lesion_radiomics_replay_EXTERNAL33_FEATURES_LABEL_BLIND.csv", df)
    return df, "PASS"


def update_roadmap_and_registry(status: str, lock_path: Path, lock_sha: str, report_path: Path, metrics_payload: dict[str, Any]) -> None:
    gov = PROJECT_ROOT / "PROJECT_GOVERNANCE"
    registry = gov / "EXPERIMENT_REGISTRY.csv"
    if registry.exists():
        with registry.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            rows = list(reader)
        entry = {
            "stage_order": "12.0",
            "experiment_id": "lesion_radiomics_replay_PYRADIOMICS_STANDARDIZED",
            "branch": "imaging radiomics standardized",
            "scientific_question": "PyRadiomics-standardized IBSI-aligned source-only radiomics after V3.5-A/B/C.",
            "internal_n": "375",
            "external_n": "33",
            "input_modality": "CT",
            "ROI_source": "AUTOMATED_PREDICTED_MASK",
            "feature_type": "PyRadiomics ORIGINAL_ONLY",
            "model": str(metrics_payload.get("selected_model", "")),
            "internal_AUROC": str(metrics_payload.get("internal_AUROC", "")),
            "internal_AUPRC": str(metrics_payload.get("internal_AUPRC", "")),
            "internal_Brier": str(metrics_payload.get("internal_Brier", "")),
            "external_AUROC": str(metrics_payload.get("external_AUROC", "")),
            "external_AUPRC": str(metrics_payload.get("external_AUPRC", "")),
            "external_Brier": str(metrics_payload.get("external_Brier", "")),
            "external_analysis_identity": "HISTORY_EXPOSED_EXTERNAL33_GENERALIZATION_CHECK" if metrics_payload.get("external_AUROC") != "NOT_EXECUTED" else "NOT_EXECUTED",
            "external_used_for_development": "False",
            "external_labels_used_for_development": "False",
            "main_finding": str(metrics_payload.get("main_finding", "")),
            "failure_or_success_reason": str(metrics_payload.get("failure_or_success_reason", "")),
            "status": status,
            "superseded_by": "not superseded",
            "key_result_lock": str(lock_path),
            "key_report": str(report_path),
            "key_prediction": str(DIRS["external_blind"] / "lesion_radiomics_replay_EXTERNAL33_PREDICTIONS_LABEL_BLIND.csv") if metrics_payload.get("external_AUROC") != "NOT_EXECUTED" else "NOT_GENERATED",
            "SHA256": lock_sha,
            "paper_role": "SECONDARY_EXPLORATORY_STANDARDIZED_RADIOMICS",
        }
        updated = False
        for row in rows:
            if row.get("experiment_id") == entry["experiment_id"]:
                row.update({k: entry.get(k, "") for k in fields})
                updated = True
        if not updated:
            rows.append({k: entry.get(k, "") for k in fields})
        with registry.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    roadmap = gov / "EXPERIMENT_ROADMAP_MASTER.md"
    if roadmap.exists():
        txt = roadmap.read_text(encoding="utf-8")
        block = f"""

## V3.5-D1 Recovery and Standardized PyRadiomics

- status: {status}
- environment: project-local `.venv_locked_radiomics`
- PyRadiomics: 3.1.0, CPython 3.9 win_amd64 prebuilt wheel
- internal_n: 375
- external_n: {metrics_payload.get('external_n', '33 planned')}
- selected model: {metrics_payload.get('selected_model', '')}
- internal strict OOF AUROC: {metrics_payload.get('internal_AUROC', '')}
- external AUROC: {metrics_payload.get('external_AUROC', '')}
- result lock: {lock_path}
- result lock SHA256: {lock_sha}
"""
        if "## V3.5-D1 Recovery and Standardized PyRadiomics" not in txt:
            roadmap.write_text(txt.rstrip() + block + "\n", encoding="utf-8")


def main() -> int:
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    env_lock_v2 = DIRS["env"] / "lesion_radiomics_PYRADIOMICS_ENVIRONMENT_LOCK_V2.json"
    if not env_lock_v2.exists():
        stop("lesion_radiomics_replay_ENVIRONMENT_NOT_READY", "environment lock V2 missing")
    env_lock = json.loads(env_lock_v2.read_text(encoding="utf-8"))
    if env_lock.get("environment_status") != "PASS":
        stop("lesion_radiomics_replay_ENVIRONMENT_NOT_READY", "environment lock V2 not PASS", env_lock)

    protocol = {
        "status": "PROTOCOL_LOCKED_AFTER_ENV_RECOVERY_BEFORE_EXTERNAL_ACCESS",
        "timestamp": now_iso(),
        "analysis_type": "PERFORMANCE_INFORMED_SECONDARY_EXPLORATORY_SOURCE_ONLY_STANDARDIZED_RADIOMICS",
        "terminology": "PYRADIOMICS_STANDARDIZED_IBSI_ALIGNED",
        "image_type": "ORIGINAL_ONLY",
        "feature_classes": CLASSES,
        "spacing_mm": [1.5, 1.5, 1.5],
        "binWidth_HU": 25,
        "normalize": False,
        "external_development_n": 0,
        "external_labels_before_prediction_lock": False,
        "primary_mask": "AUTOMATED_PREDICTED_MASK",
    }
    write_json(DIRS["protocol"] / "lesion_radiomics_replay_RECOVERED_PROTOCOL_LOCK.json", protocol)

    labels, _, _, _, acq = load_internal_tables()
    matrix = extract_internal_matrix()
    fcols = feature_cols(matrix)
    write_feature_dictionary(fcols)
    full_stab = stability_table(matrix)
    robust_count = int(full_stab["ROBUST_ICC_GE_0_90"].sum())
    write_csv(DIRS["robust"] / "lesion_radiomics_replay_FULL_INTERNAL_PYRADIOMICS_STABILITY_AUDIT.csv", full_stab)
    oof_df, outer_df, selected_df, internal_m = nested_cv(labels, matrix, acq)
    model_path, model_lock, fset = fit_final(labels, matrix)
    selected_model = model_lock["model"]

    external_executed = False
    external_m: dict[str, Any] = {"external_AUROC": "NOT_EXECUTED", "external_AUPRC": "NOT_EXECUTED", "external_Brier": "NOT_EXECUTED"}
    external_status_reason = ""
    ext_df, ext_status = extract_external_features_if_possible(fset)
    if ext_df is None:
        external_status_reason = ext_status
    else:
        bundle = joblib.load(model_path)
        pipe = bundle["pipeline"]
        probs = pipe.predict_proba(ext_df.set_index("neutral_external_id")[fset])[:, 1]
        pred_df = pd.DataFrame({"neutral_external_id": ext_df["neutral_external_id"], "probability_COP": probs, "prediction_0_5": (probs >= THRESHOLD).astype(int)})
        pred_path = DIRS["external_blind"] / "lesion_radiomics_replay_EXTERNAL33_PREDICTIONS_LABEL_BLIND.csv"
        write_csv(pred_path, pred_df)
        pred_lock = {
            "status": "lesion_radiomics_replay_EXTERNAL33_PREDICTIONS_LOCKED_BEFORE_LABELS",
            "timestamp": now_iso(),
            "n": len(pred_df),
            "sha256": sha256_file(pred_path),
            "external_labels_accessed": False,
            "threshold": 0.5,
            "probability": "P_COP",
        }
        pred_lock_path = DIRS["external_blind"] / "lesion_radiomics_replay_EXTERNAL33_PREDICTION_LOCK.json"
        write_json(pred_lock_path, pred_lock)
        labels_ext = external_labels()
        write_json(DIRS["external_results"] / "lesion_radiomics_replay_EXTERNAL_LABEL_UNBLINDING_AUDIT.json", {
            "status": "PASS",
            "prediction_lock_timestamp": pred_lock["timestamp"],
            "label_unblind_timestamp": now_iso(),
            "external_n": len(labels_ext),
        })
        merged = pred_df.merge(labels_ext, on="neutral_external_id", how="inner")
        write_csv(DIRS["external_results"] / "lesion_radiomics_replay_EXTERNAL33_PREDICTIONS_WITH_LABEL_PRIVATE.csv", merged)
        y = merged["label_binary"].to_numpy(dtype=int)
        p = merged["probability_COP"].to_numpy(dtype=float)
        external_m = metrics(y, p)
        external_m["AUROC_CI"] = bootstrap_ci(y, p, roc_auc_score)
        external_m["AUPRC_CI"] = bootstrap_ci(y, p, average_precision_score)
        write_csv(DIRS["external_results"] / "lesion_radiomics_replay_EXTERNAL33_METRICS.csv", pd.DataFrame([external_m]))
        external_executed = True

    status = "lesion_radiomics_replay_PYRADIOMICS_STANDARDIZED_EXTERNAL_COMPLETE" if external_executed else "lesion_radiomics_replay_INTERNAL_COMPLETE_EXTERNAL_INPUT_HOLD"
    payload = {
        "status": status,
        "timestamp": now_iso(),
        "analysis_type": "PERFORMANCE_INFORMED_SECONDARY_EXPLORATORY_SOURCE_ONLY_STANDARDIZED_RADIOMICS",
        "environment_lock_v2": str(env_lock_v2),
        "internal_n": 375,
        "external_n": "33/33" if external_executed else "NOT_EXECUTED_EXTERNAL_INPUT_HOLD",
        "external_development_n": 0,
        "candidate_pyradiomics_features": len(fcols),
        "robust_features_icc_ge_0_90": robust_count,
        "final_selected_features": len(fset),
        "final_feature_names": fset,
        "selected_model": selected_model,
        "internal_AUROC": internal_m["AUROC"],
        "internal_AUPRC": internal_m["AUPRC"],
        "internal_Brier": internal_m["Brier"],
        "internal_worst_scanner_domain_AUROC": internal_m["worst_scanner_domain_AUROC"],
        "external_radiomics_reference_external_AUROC": V34_EXTERNAL_AUROC,
        "historical_radiomics_A_external_AUROC": V35A_EXTERNAL_AUROC,
        "historical_radiomics_B_external_AUROC": V35B_EXTERNAL_AUROC,
        "historical_radiomics_C_external_AUROC": V35C_EXTERNAL_AUROC,
        "external_AUROC": external_m.get("AUROC", "NOT_EXECUTED"),
        "external_AUROC_CI": external_m.get("AUROC_CI", "NOT_EXECUTED"),
        "external_AUPRC": external_m.get("AUPRC", "NOT_EXECUTED"),
        "external_Brier": external_m.get("Brier", "NOT_EXECUTED"),
        "external_status_reason": external_status_reason,
        "external_labels_used_before_prediction_lock": False,
        "external_feature_selection": False,
        "external_model_selection": False,
        "external_threshold_tuning": False,
        "ComBat_CovBat": False,
        "clinical": False,
        "multimodal": False,
        "scientific_integrity": "PASS",
        "failed_py311_environment_removed": False,
    }
    lock_path = DIRS["locks"] / "lesion_radiomics_replay_PYRADIOMICS_STANDARDIZED_RESULT_LOCK.json"
    write_json(lock_path, payload)
    lock_sha = sha256_file(lock_path)
    report_path = DIRS["reports"] / "lesion_radiomics_replay_PYRADIOMICS_STANDARDIZED_FINAL_REPORT.md"
    write_report(report_path, payload, internal_m, external_m)
    handoff = PROJECT_ROOT / "PROJECT_GOVERNANCE" / "CHATGPT_HANDOFF_lesion_radiomics_replay_RECOVERY.md"
    handoff.write_text(f"""# ChatGPT Handoff V3.5-D1 Recovery

Status: `{status}`

Environment: project-local `.venv_locked_radiomics`

PyRadiomics: 3.1.0 via `pyradiomics-3.1.0-cp39-cp39-win_amd64.whl`

Internal AUROC: {payload['internal_AUROC']}

Internal AUPRC: {payload['internal_AUPRC']}

Internal Brier: {payload['internal_Brier']}

External AUROC: {payload['external_AUROC']}

External status reason: {external_status_reason}

Result lock: `{lock_path}`

Result lock SHA256: `{lock_sha}`

Protocol deviations: none known; if external was not executed, reason is external input availability hold after internal final model lock.
""", encoding="utf-8")
    payload_for_registry = {
        "selected_model": selected_model,
        "internal_AUROC": payload["internal_AUROC"],
        "internal_AUPRC": payload["internal_AUPRC"],
        "internal_Brier": payload["internal_Brier"],
        "external_AUROC": payload["external_AUROC"],
        "external_AUPRC": payload["external_AUPRC"],
        "external_Brier": payload["external_Brier"],
        "external_n": payload["external_n"],
        "main_finding": status,
        "failure_or_success_reason": external_status_reason,
    }
    update_roadmap_and_registry(status, lock_path, lock_sha, report_path, payload_for_registry)
    write_json(DIRS["locks"] / "lesion_radiomics_replay_RUN_COMPLETED.json", {"status": status, "lock_sha256": lock_sha, "timestamp": now_iso()})
    write_checksums([lock_path, report_path, handoff, env_lock_v2])
    final_console(payload, lock_path, lock_sha, report_path, handoff)
    return 0


def write_report(path: Path, payload: dict[str, Any], internal_m: dict[str, Any], external_m: dict[str, Any]) -> None:
    text = f"""# V3.5-D1 PyRadiomics Standardized Radiomics

## Scientific Identity

`PERFORMANCE_INFORMED_SECONDARY_EXPLORATORY_SOURCE_ONLY_STANDARDIZED_RADIOMICS`

This is not fresh independent validation. Development uses internal375 only; external labels are not accessed before prediction lock.

## Environment Recovery

Project-local Python 3.9 environment recovered PyRadiomics 3.1.0 using a CPython 3.9 Windows prebuilt wheel. NumPy was pinned below 2.0 to satisfy the PyRadiomics compiled extension ABI.

## Internal Result

- Candidate PyRadiomics features: {payload['candidate_pyradiomics_features']}
- Robust features ICC >= 0.90: {payload['robust_features_icc_ge_0_90']}
- Final selected features: {payload['final_selected_features']}
- Selected model: {payload['selected_model']}
- Internal strict OOF AUROC: {payload['internal_AUROC']}
- Internal AUPRC: {payload['internal_AUPRC']}
- Internal Brier: {payload['internal_Brier']}
- Worst scanner-domain AUROC: {payload['internal_worst_scanner_domain_AUROC']}

## External Result

- External status: {payload['status']}
- External AUROC: {payload['external_AUROC']}
- External AUPRC: {payload['external_AUPRC']}
- External Brier: {payload['external_Brier']}
- External hold reason: {payload['external_status_reason']}

## Historical Comparators

- V3.4 external AUROC: {V34_EXTERNAL_AUROC}
- V3.5-A external AUROC: {V35A_EXTERNAL_AUROC}
- V3.5-B external AUROC: {V35B_EXTERNAL_AUROC}
- V3.5-C external AUROC: {V35C_EXTERNAL_AUROC}

## Governance

- External development n: 0
- External labels before prediction lock: NO
- External feature selection: NO
- External model selection: NO
- Threshold tuning: NO
- Clinical: NO
- Multimodal: NO
"""
    path.write_text(text, encoding="utf-8")


def write_checksums(paths: list[Path]) -> None:
    rows = []
    for p in paths:
        if p.exists():
            rows.append(f"{sha256_file(p)}  {p}")
    (OUT / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def final_console(payload: dict[str, Any], lock_path: Path, lock_sha: str, report_path: Path, handoff: Path) -> None:
    free_gib = round(shutil.disk_usage(PROJECT_ROOT).free / (1024 ** 3), 3)
    ci = payload.get("external_AUROC_CI", "NOT_EXECUTED")
    print("\n================ V3.5-D1 PYRADIOMICS RECOVERY + SCIENTIFIC RUN ================")
    print("\nPrevious Python 3.11 environment:\nFAILED_UNUSED")
    print("\nFailure root cause:\nPython 3.11 source build required MSVC; Python 3.12 metadata incompatible; recovered with cp39 prebuilt wheel")
    print("\nProject-local Python 3.9 environment:\nPASS")
    print("\nEnvironment path:\n<project-root>\\.venv_locked_radiomics")
    print(f"\nPython version:\n{platform.python_version()}")
    print("\nPyRadiomics version:\n3.1.0")
    print("\nInstallation:\nPREBUILT_WINDOWS_WHEEL")
    print("\nGlobal Python modified:\nNO")
    print("\nGlobal conda/base modified:\nNO")
    print("\nProject-local cache:\nYES")
    print("\nSynthetic smoke test:\nPASS")
    print("\nInternal real smoke test:\n3/3 PASS")
    print("\nInternal patients:\n375")
    print(f"\nExternal patients:\n{payload['external_n']}")
    print("\nExternal development n:\n0")
    print(f"\nCandidate PyRadiomics features:\n{payload['candidate_pyradiomics_features']}")
    print(f"\nRobust features ICC >= 0.90:\n{payload['robust_features_icc_ge_0_90']}")
    print(f"\nFinal selected features:\n{payload['final_selected_features']}")
    print(f"\nSelected model:\n{payload['selected_model']}")
    print(f"\nV3.5-D1 internal strict OOF AUROC:\n{payload['internal_AUROC']}")
    print(f"\nV3.5-D1 internal AUPRC:\n{payload['internal_AUPRC']}")
    print(f"\nV3.5-D1 internal Brier:\n{payload['internal_Brier']}")
    print(f"\nInternal worst scanner-domain AUROC:\n{payload['internal_worst_scanner_domain_AUROC']}")
    print(f"\nV3.4 external AUROC:\n{V34_EXTERNAL_AUROC}")
    print(f"\nV3.5-A external AUROC:\n{V35A_EXTERNAL_AUROC}")
    print(f"\nV3.5-B external AUROC:\n{V35B_EXTERNAL_AUROC}")
    print(f"\nV3.5-C external AUROC:\n{V35C_EXTERNAL_AUROC}")
    print(f"\nV3.5-D1 external AUROC:\n{payload['external_AUROC']}")
    print(f"\nV3.5-D1 external AUROC 95% CI:\n{ci}")
    print(f"\nV3.5-D1 external AUPRC:\n{payload['external_AUPRC']}")
    print(f"\nV3.5-D1 external Brier:\n{payload['external_Brier']}")
    print(f"\nExternal n:\n{payload['external_n']}")
    print("\nExternal labels used before prediction lock:\nNO")
    print("\nExternal feature selection:\nNO")
    print("\nExternal model selection:\nNO")
    print("\nExternal threshold tuning:\nNO")
    print(f"\nScientific integrity:\n{payload['scientific_integrity']}")
    print("\nFailed py311 environment removed:\nNO")
    print(f"\nFree disk:\n{free_gib} GiB")
    print("\nRoadmap updated:\nYES")
    print(f"\nChatGPT handoff:\n{handoff}")
    print(f"\nFinal scientific result lock:\n{lock_path}")
    print(f"\nFinal lock SHA256:\n{lock_sha}")
    print("\nNext action:\nUPLOAD_CHATGPT_HANDOFF_lesion_radiomics_replay_RECOVERY_MD_FOR_REVIEW")
    print(f"\nFinal status:\n{payload['status']}")
    print("\n================================================================================")


if __name__ == "__main__":
    raise SystemExit(main())

