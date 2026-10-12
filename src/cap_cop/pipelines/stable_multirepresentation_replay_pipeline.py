from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)


PROJECT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT / "models" / "stable_multirepresentation_ct"
WHOLE_CHEST_MODEL_DIR = PROJECT / "models" / "whole_chest_ct"
ROOT = PROJECT / "outputs" / "stable_multirepresentation_pipeline"
CODE = PROJECT / "src" / "cap_cop"
SEED = 20260909
BOOTSTRAP_REPS = 10000
INNER_SPLITS = 3
EPS = 1e-8

PATHS = {
    "internal_oof_ref": MODEL_DIR / "derived_inputs" / "09_V3_4_STANDARD_INTERNAL_OOF_PREDICTIONS.csv",
    "lodo_ref": MODEL_DIR / "derived_inputs" / "09_V3_4_STANDARD_INTERNAL_OOF_PREDICTIONS.csv",
    "domain_lock": MODEL_DIR / "derived_inputs" / "03_PRIMARY_DG_DOMAIN_LOCK.json",
    "v34_internal_metrics": MODEL_DIR / "internal_oof_predictions.csv",
    "v34_lodo_metrics": MODEL_DIR / "internal_oof_predictions.csv",
    "v34_external_metrics": MODEL_DIR / "external_predictions.csv",
    "v35_internal_metrics": WHOLE_CHEST_MODEL_DIR / "internal_oof_predictions.csv",
    "v35_lodo_metrics": WHOLE_CHEST_MODEL_DIR / "internal_oof_predictions.csv",
    "v35_external_metrics": WHOLE_CHEST_MODEL_DIR / "external_predictions.csv",
    "internal_crosswalk": PROJECT / "data" / "public_subject_model_crosswalk.csv",
    "external_crosswalk": MODEL_DIR / "derived_inputs" / "EXTERNAL_EXTV2_TO_EXTERNAL_PATIENT_CROSSWALK.csv",
    "external_labels": MODEL_DIR / "derived_inputs" / "EXTERNAL_LOCKED_PREDICTIONS_WITH_LABELS_V2.csv",
    "lesion_features": MODEL_DIR / "derived_inputs" / "06_LESION_PHENOTYPE_FEATURES.csv",
    "v37_feature_registry": MODEL_DIR / "derived_inputs" / "06_LESION_PHENOTYPE_FEATURES.csv",
    "d1_internal_features": MODEL_DIR / "derived_inputs" / "V3_5_D1_INTERNAL375_BASELINE_PYRADIOMICS_FEATURES.csv",
    "d1_external_features": MODEL_DIR / "derived_inputs" / "V3_5_D1_EXTERNAL33_FEATURES_LABEL_BLIND.csv",
    "d1_feature_list": MODEL_DIR / "derived_inputs" / "V3_5_D1_FINAL_FEATURE_LIST.csv",
    "d1_internal_metrics": MODEL_DIR / "internal_oof_predictions.csv",
    "d1_external_metrics": MODEL_DIR / "external_predictions.csv",
    "tangerine_internal_embeddings": WHOLE_CHEST_MODEL_DIR / "embeddings" / "internal_embeddings.npy",
    "tangerine_internal_manifest": WHOLE_CHEST_MODEL_DIR / "internal_embedding_manifest.csv",
    "tangerine_external_embeddings": WHOLE_CHEST_MODEL_DIR / "embeddings" / "external_embeddings.npy",
    "tangerine_external_manifest": WHOLE_CHEST_MODEL_DIR / "external_embedding_manifest.csv",
    "genesis_checkpoint": MODEL_DIR / "foundation_encoders" / "genesis_chest_ct.pt",
    "genesis_code": PROJECT / "src" / "cap_cop" / "vendor" / "models_genesis" / "unet3d.py",
    "internal_tensor_dir": PROJECT / "outputs" / "runtime_intermediate" / "internal_p95_tensors",
    "external_tensor_dir": PROJECT / "outputs" / "runtime_intermediate" / "external_p95_tensors",
}

DIRS = {
    "protocol": ROOT / "00_protocol",
    "literature": ROOT / "01_literature",
    "audit": ROOT / "02_feature_audit",
    "handcrafted": ROOT / "03_handcrafted",
    "genesis": ROOT / "04_genesis_embedding",
    "tangerine": ROOT / "05_tangerine_embedding",
    "selection": ROOT / "06_stability_selection",
    "oof": ROOT / "07_internal_oof",
    "lodo": ROOT / "08_lodo",
    "models": ROOT / "09_models",
    "external": ROOT / "10_external_predictions",
    "metrics": ROOT / "11_metrics",
    "reports": ROOT / "12_reports",
    "locks": ROOT / "13_locks",
    "logs": ROOT / "14_logs",
    "code": CODE,
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def mkdirs() -> None:
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sha(path: Path) -> None:
    write_text(path.with_name(path.name + "_SHA256.txt"), f"{sha256_file(path)}  {path.name}\n")


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2:
        return {"AUROC": np.nan, "AUPRC": np.nan, "Brier": np.nan, "LogLoss": np.nan}
    return {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "LogLoss": float(log_loss(y, p, labels=[0, 1])),
    }


def strat_bootstrap(y: np.ndarray, p: np.ndarray, reps: int = BOOTSTRAP_REPS) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rows: list[dict[str, float]] = []
    for _ in range(reps):
        sample = np.r_[rng.choice(idx0, size=len(idx0), replace=True), rng.choice(idx1, size=len(idx1), replace=True)]
        rows.append(metrics(y[sample], p[sample]))
    out: dict[str, float] = {}
    for key in ["AUROC", "AUPRC", "Brier"]:
        vals = np.array([r[key] for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        out[f"{key}_ci_low"] = float(np.quantile(vals, 0.025)) if len(vals) else np.nan
        out[f"{key}_ci_high"] = float(np.quantile(vals, 0.975)) if len(vals) else np.nan
    return out


def hedges_g(cap: np.ndarray, cop: np.ndarray) -> tuple[float, float]:
    cap = np.asarray(cap, dtype=float)
    cop = np.asarray(cop, dtype=float)
    cap = cap[np.isfinite(cap)]
    cop = cop[np.isfinite(cop)]
    n0, n1 = len(cap), len(cop)
    if n0 < 2 or n1 < 2:
        return np.nan, np.nan
    s0, s1 = np.var(cap, ddof=1), np.var(cop, ddof=1)
    sp = math.sqrt(((n0 - 1) * s0 + (n1 - 1) * s1) / max(n0 + n1 - 2, 1))
    if sp <= EPS:
        return np.nan, np.nan
    d = (np.mean(cop) - np.mean(cap)) / sp
    j = 1 - 3 / max(4 * (n0 + n1) - 9, 1)
    g = float(j * d)
    var_g = float((n0 + n1) / (n0 * n1) + (g * g) / (2 * max(n0 + n1 - 2, 1)))
    return g, var_g


def random_effects(rows: list[dict[str, Any]]) -> dict[str, Any]:
    yi = np.array([r["domain_g"] for r in rows], dtype=float)
    vi = np.array([r["domain_var"] for r in rows], dtype=float)
    ok = np.isfinite(yi) & np.isfinite(vi) & (vi > 0)
    yi, vi = yi[ok], vi[ok]
    k = len(yi)
    if k == 0:
        return {"pooled_g": np.nan, "ci_low": np.nan, "ci_high": np.nan, "tau2": np.nan, "I2": np.nan}
    if k == 1:
        se = math.sqrt(float(vi[0]))
        return {"pooled_g": float(yi[0]), "ci_low": float(yi[0] - 1.96 * se), "ci_high": float(yi[0] + 1.96 * se), "tau2": 0.0, "I2": 0.0}
    w = 1 / vi
    fixed = float(np.sum(w * yi) / np.sum(w))
    q = float(np.sum(w * (yi - fixed) ** 2))
    df = k - 1
    c = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    wr = 1 / (vi + tau2)
    pooled = float(np.sum(wr * yi) / np.sum(wr))
    se = math.sqrt(float(1 / np.sum(wr)))
    i2 = max(0.0, (q - df) / q) * 100 if q > 0 else 0.0
    return {"pooled_g": pooled, "ci_low": pooled - 1.96 * se, "ci_high": pooled + 1.96 * se, "tau2": tau2, "I2": float(i2)}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_base() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    oof = read_csv(PATHS["internal_oof_ref"])
    base = (
        oof[oof["candidate"].eq("R0_GENESIS_B2_BASELINE")]
        [["neutral_imaging_id", "label", "fold_or_domain", "primary_domain"]]
        .drop_duplicates("neutral_imaging_id")
        .rename(columns={"fold_or_domain": "outer_fold"})
    )
    base["outer_fold"] = base["outer_fold"].astype(int)
    base["label"] = base["label"].astype(int)
    with PATHS["domain_lock"].open("r", encoding="utf-8") as f:
        lock = json.load(f)
    domains = list(lock["lodo_evaluable_domains"])
    ext_xw = read_csv(PATHS["external_crosswalk"])
    return base, ext_xw, domains


def load_external_labels(ext_xw: pd.DataFrame) -> pd.DataFrame:
    labels = read_csv(PATHS["external_labels"])[["neutral_external_id", "true_label"]].copy()
    labels["label"] = labels["true_label"].astype(int)
    out = ext_xw.merge(labels, on="neutral_external_id", how="left")
    return out[["neutral_external_id", "neutral_patient_id", "label"]]


def numeric_cols(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            cols.append(c)
            df[c] = s
    return cols


def load_tabular_features(base: pd.DataFrame, ext_xw: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, list[str]], list[dict[str, Any]]]:
    audit: list[dict[str, Any]] = []
    internal: dict[str, pd.DataFrame] = {}
    external: dict[str, pd.DataFrame] = {}
    feats: dict[str, list[str]] = {}

    lesion = read_csv(PATHS["lesion_features"])
    a_feats = numeric_cols(lesion, {"neutral_patient_id", "source_neutral_imaging_id", "neutral_external_id", "site", "group", "label_binary"})
    li = lesion[lesion["site"].eq("Internal")].copy()
    li = base.merge(li, left_on="neutral_imaging_id", right_on="source_neutral_imaging_id", how="left")
    le = lesion[lesion["site"].eq("External")].copy()
    if "neutral_external_id" in le.columns:
        le = le.drop(columns=["neutral_external_id"])
    le = ext_xw.merge(le, on="neutral_patient_id", how="left")
    internal["FAMILY_A_HANDCRAFTED"] = li[["neutral_imaging_id", "label", "outer_fold", "primary_domain"] + a_feats]
    external["FAMILY_A_HANDCRAFTED"] = le[["neutral_external_id", "neutral_patient_id"] + a_feats]
    feats["FAMILY_A_HANDCRAFTED"] = a_feats
    audit.append({"resource": "FAMILY_A_HANDCRAFTED", "path": str(PATHS["lesion_features"]), "available": True, "internal_rows": len(li), "external_rows": len(le), "feature_count": len(a_feats), "status": "EXECUTABLE"})

    d1i = read_csv(PATHS["d1_internal_features"])
    d1e = read_csv(PATHS["d1_external_features"])
    d1_list = read_csv(PATHS["d1_feature_list"])["feature"].astype(str).tolist()
    d1_feats = [f for f in d1_list if f in d1i.columns and f in d1e.columns]
    d1i = d1i.drop(columns=[c for c in ["outer_fold", "label", "label_binary", "primary_domain"] if c in d1i.columns])
    d1i = base.merge(d1i, left_on="neutral_imaging_id", right_on="source_neutral_imaging_id", how="left")
    d1e = ext_xw.merge(d1e, on="neutral_patient_id", how="left")
    for f in d1_feats:
        d1i[f] = pd.to_numeric(d1i[f], errors="coerce")
        d1e[f] = pd.to_numeric(d1e[f], errors="coerce")
    internal["FAMILY_B_RAD_D1_FINAL13"] = d1i[["neutral_imaging_id", "label", "outer_fold", "primary_domain"] + d1_feats]
    external["FAMILY_B_RAD_D1_FINAL13"] = d1e[["neutral_external_id", "neutral_patient_id"] + d1_feats]
    feats["FAMILY_B_RAD_D1_FINAL13"] = d1_feats
    audit.append({"resource": "FAMILY_B_RAD_D1_FINAL13", "path": str(PATHS["d1_internal_features"]), "available": True, "internal_rows": len(d1i), "external_rows": len(d1e), "feature_count": len(d1_feats), "status": "EXECUTABLE"})
    return internal, external, feats, audit


def import_genesis():
    spec = importlib.util.spec_from_file_location("v38_models_genesis_unet3d", PATHS["genesis_code"])
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def extract_genesis_embeddings(ids: list[str], tensor_dir: Path, out_npy: Path, out_manifest: Path) -> bool:
    if out_npy.exists() and out_manifest.exists():
        return True
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        write_text(DIRS["genesis"] / "GENESIS_EXTRACTION_ERROR.txt", f"torch import failed: {exc}\n")
        return False
    try:
        mod = import_genesis()
        raw = torch.load(PATHS["genesis_checkpoint"], map_location="cpu", weights_only=False)
        state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
        state = {str(k).replace("module.", "").replace("model.", "").replace("encoder.", ""): v for k, v in state.items() if torch.is_tensor(v)}
        model = mod.UNet3D(n_class=1, act="relu")
        keep = {k: v for k, v in state.items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
        model.load_state_dict(keep, strict=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        pool = nn.AdaptiveAvgPool3d(1)
        rows, manifest = [], []
        with torch.no_grad():
            for i, nid in enumerate(ids):
                arr = np.load(tensor_dir / f"{nid}.npy").astype(np.float32, copy=False)
                if arr.ndim == 3:
                    arr = arr[None, None]
                elif arr.ndim == 4:
                    arr = arr[None]
                x = torch.from_numpy(arr).to(device)
                out64, _ = model.down_tr64(x)
                out128, _ = model.down_tr128(out64)
                out256, _ = model.down_tr256(out128)
                out512, _ = model.down_tr512(out256)
                emb = pool(out512).flatten(1).detach().cpu().numpy()[0].astype(np.float32)
                rows.append(emb)
                manifest.append({"id": nid, "embedding_index": i, "embedding_dim": int(emb.shape[0]), "source_tensor": str(tensor_dir / f"{nid}.npy")})
        np.save(out_npy, np.vstack(rows).astype(np.float32))
        pd.DataFrame(manifest).to_csv(out_manifest, index=False)
        return True
    except Exception as exc:
        write_text(DIRS["genesis"] / "GENESIS_EXTRACTION_ERROR.txt", f"{type(exc).__name__}: {exc}\n")
        return False


def load_embedding_family(base: pd.DataFrame, ext_xw: pd.DataFrame, ext_labels: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]]]:
    internal_meta: dict[str, pd.DataFrame] = {}
    external_meta: dict[str, pd.DataFrame] = {}
    internal_arr: dict[str, np.ndarray] = {}
    external_arr: dict[str, np.ndarray] = {}
    audit: list[dict[str, Any]] = []

    gi_npy = DIRS["genesis"] / "stable_multirepresentation_GENESIS_FROZEN_INTERNAL375_EMBEDDINGS.npy"
    gi_man = DIRS["genesis"] / "stable_multirepresentation_GENESIS_FROZEN_INTERNAL375_EMBEDDING_MANIFEST.csv"
    ge_npy = DIRS["genesis"] / "stable_multirepresentation_GENESIS_FROZEN_EXTERNAL33_EMBEDDINGS.npy"
    ge_man = DIRS["genesis"] / "stable_multirepresentation_GENESIS_FROZEN_EXTERNAL33_EMBEDDING_MANIFEST.csv"
    genesis_ok = all(p.exists() for p in [PATHS["genesis_checkpoint"], PATHS["genesis_code"], PATHS["internal_tensor_dir"], PATHS["external_tensor_dir"]])
    if genesis_ok:
        genesis_ok = extract_genesis_embeddings(base["neutral_imaging_id"].tolist(), PATHS["internal_tensor_dir"], gi_npy, gi_man)
        genesis_ok = genesis_ok and extract_genesis_embeddings(ext_xw["neutral_external_id"].tolist(), PATHS["external_tensor_dir"], ge_npy, ge_man)
    if genesis_ok:
        internal_arr["FAMILY_C_GENESIS_FROZEN"] = np.load(gi_npy)
        external_arr["FAMILY_C_GENESIS_FROZEN"] = np.load(ge_npy)
        internal_meta["FAMILY_C_GENESIS_FROZEN"] = base.copy()
        external_meta["FAMILY_C_GENESIS_FROZEN"] = ext_xw.copy()
        audit.append({"resource": "FAMILY_C_GENESIS_FROZEN", "path": str(gi_npy), "available": True, "internal_rows": len(base), "external_rows": len(ext_xw), "feature_count": internal_arr["FAMILY_C_GENESIS_FROZEN"].shape[1], "status": "EXECUTABLE_FROZEN_EXTRACTED_OR_CACHED"})
    else:
        audit.append({"resource": "FAMILY_C_GENESIS_FROZEN", "path": str(PATHS["genesis_checkpoint"]), "available": False, "internal_rows": len(base), "external_rows": len(ext_xw), "feature_count": 0, "status": "NOT_EXECUTABLE"})

    if PATHS["tangerine_internal_embeddings"].exists() and PATHS["tangerine_external_embeddings"].exists():
        ti = pd.read_csv(PATHS["tangerine_internal_manifest"]).merge(base, on="neutral_imaging_id", how="left")
        te = pd.read_csv(PATHS["tangerine_external_manifest"]).merge(ext_xw, on="neutral_external_id", how="left")
        ti = ti.sort_values("embedding_index")
        te = te.sort_values("embedding_index")
        internal_arr["FAMILY_D_TANGERINE_FROZEN"] = np.load(PATHS["tangerine_internal_embeddings"])
        external_arr["FAMILY_D_TANGERINE_FROZEN"] = np.load(PATHS["tangerine_external_embeddings"])
        internal_meta["FAMILY_D_TANGERINE_FROZEN"] = ti[["neutral_imaging_id", "label", "outer_fold", "primary_domain"]].copy()
        external_meta["FAMILY_D_TANGERINE_FROZEN"] = te[["neutral_external_id", "neutral_patient_id"]].copy()
        audit.append({"resource": "FAMILY_D_TANGERINE_FROZEN", "path": str(PATHS["tangerine_internal_embeddings"]), "available": True, "internal_rows": len(ti), "external_rows": len(te), "feature_count": internal_arr["FAMILY_D_TANGERINE_FROZEN"].shape[1], "status": "EXECUTABLE_EXISTING_FROZEN"})
    else:
        audit.append({"resource": "FAMILY_D_TANGERINE_FROZEN", "path": str(PATHS["tangerine_internal_embeddings"]), "available": False, "internal_rows": 0, "external_rows": 0, "feature_count": 0, "status": "NOT_EXECUTABLE"})
    return internal_meta, external_meta, internal_arr, external_arr, audit


def meta_select(df: pd.DataFrame, features: list[str], train_mask: np.ndarray, domains: list[str], max_features: int, prefix: str = "") -> tuple[list[str], pd.DataFrame]:
    train = df.loc[train_mask].copy()
    rows: list[dict[str, Any]] = []
    for f in features:
        vals = pd.to_numeric(train[f], errors="coerce")
        if vals.isna().mean() > 0.20:
            rows.append({"feature": f, "selection_tier": "EXCLUDED_MISSING_GT20", "selected": False})
            continue
        dom_rows = []
        for d in domains:
            sub = train[train["primary_domain"].eq(d)]
            cap = pd.to_numeric(sub.loc[sub["label"].eq(0), f], errors="coerce").dropna().to_numpy(float)
            cop = pd.to_numeric(sub.loc[sub["label"].eq(1), f], errors="coerce").dropna().to_numpy(float)
            if len(cap) < 5 or len(cop) < 5:
                continue
            g, var = hedges_g(cap, cop)
            if np.isfinite(g) and np.isfinite(var):
                dom_rows.append({"domain": d, "domain_g": g, "domain_var": var, "n_CAP": len(cap), "n_COP": len(cop)})
        meta = random_effects(dom_rows)
        k = len(dom_rows)
        pooled = meta["pooled_g"]
        direction = np.sign(pooled) if np.isfinite(pooled) and pooled != 0 else 0
        dir_cons = float(np.mean([np.sign(r["domain_g"]) == direction for r in dom_rows])) if k and direction else 0.0
        primary = k >= 3 and abs(pooled) >= 0.30 and dir_cons >= 0.75 and meta["I2"] <= 60
        fallback = k >= 3 and abs(pooled) >= 0.20 and dir_cons >= 0.67 and meta["I2"] <= 75
        rows.append({
            "feature": prefix + f,
            "raw_feature": f,
            "evaluable_domains": k,
            "pooled_Hedges_g": pooled,
            "ci_low": meta["ci_low"],
            "ci_high": meta["ci_high"],
            "tau2": meta["tau2"],
            "I2": meta["I2"],
            "direction_consistency": dir_cons,
            "selection_tier": "PRIMARY" if primary else ("FALLBACK_ELIGIBLE" if fallback else "NOT_STABLE"),
            "selected": primary,
        })
    res = pd.DataFrame(rows)
    selected = res.loc[res["selected"].eq(True), "raw_feature"].astype(str).tolist() if not res.empty else []
    tier = "PRIMARY"
    if len(selected) < 3:
        res["selected"] = res["selection_tier"].isin(["PRIMARY", "FALLBACK_ELIGIBLE"])
        selected = res.loc[res["selected"].eq(True), "raw_feature"].astype(str).tolist()
        tier = "FALLBACK"
    if len(selected) < 3:
        res["selected"] = False
        res["selection_tier_final"] = "INSUFFICIENT_STABLE_FEATURES"
        return [], res
    res["selection_tier_final"] = tier
    selected = prune_features(train, selected, res, max_features)
    res["selected_after_pruning"] = res["raw_feature"].isin(selected)
    return selected, res


def prune_features(train: pd.DataFrame, selected: list[str], meta: pd.DataFrame, max_features: int) -> list[str]:
    selected = list(dict.fromkeys(selected))
    score = meta.set_index("raw_feature")
    keep: list[str] = []
    for f in sorted(selected, key=lambda x: (-abs(float(score.loc[x, "pooled_Hedges_g"])), float(score.loc[x, "I2"]), x)):
        redundant = False
        for k in keep:
            a = pd.to_numeric(train[f], errors="coerce")
            b = pd.to_numeric(train[k], errors="coerce")
            ok = a.notna() & b.notna()
            rho = spearmanr(a[ok], b[ok]).correlation if ok.sum() > 3 else 0
            if np.isfinite(rho) and abs(rho) > 0.85:
                redundant = True
                break
        if not redundant:
            keep.append(f)
        if len(keep) >= max_features:
            break
    return keep


@dataclass
class FitResult:
    candidate: str
    fold_name: str
    valid: bool
    pred_ids: list[str]
    pred_y: list[int]
    pred_p: list[float]
    ext_long: list[dict[str, Any]]
    selection: pd.DataFrame
    dimension: int
    params: dict[str, Any]
    status: str


def select_hyperparams(x: np.ndarray, y: np.ndarray, classifier: str) -> tuple[dict[str, Any], Any]:
    skf = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)
    candidates = []
    if classifier == "elasticnet":
        for c in [0.1, 1.0, 10.0]:
            for l1 in [0.0, 0.5, 1.0]:
                candidates.append({"C": c, "l1_ratio": l1})
    else:
        for md in [3, 5, None]:
            for leaf in [3, 5, 10]:
                for mf in ["sqrt", 0.5]:
                    candidates.append({"max_depth": md, "min_samples_leaf": leaf, "max_features": mf, "n_estimators": 500})
    scored = []
    for params in candidates:
        fold_scores = []
        for tr, va in skf.split(x, y):
            model = make_model(classifier, params)
            model.fit(x[tr], y[tr])
            p = model.predict_proba(x[va])[:, 1]
            m = metrics(y[va], p)
            fold_scores.append(m)
        scored.append({
            "params": params,
            "AUROC": np.nanmean([s["AUROC"] for s in fold_scores]),
            "AUPRC": np.nanmean([s["AUPRC"] for s in fold_scores]),
            "Brier": np.nanmean([s["Brier"] for s in fold_scores]),
        })
    if classifier == "elasticnet":
        scored.sort(key=lambda r: (-r["AUROC"], -r["AUPRC"], r["Brier"], r["params"]["C"]))
    else:
        scored.sort(key=lambda r: (-r["AUROC"], -r["AUPRC"], r["Brier"], 999 if r["params"]["max_depth"] is None else r["params"]["max_depth"], r["params"]["min_samples_leaf"], str(r["params"]["max_features"])))
    best = scored[0]["params"]
    model = make_model(classifier, best)
    model.fit(x, y)
    return best, model


def make_model(classifier: str, params: dict[str, Any]):
    if classifier == "elasticnet":
        return LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=float(params["C"]),
            l1_ratio=float(params["l1_ratio"]),
            class_weight=None,
            max_iter=10000,
            random_state=SEED,
        )
    return RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=params["max_depth"],
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params["max_features"],
        class_weight=None,
        random_state=SEED,
        n_jobs=-1,
    )


def fit_tabular_candidate(candidate: str, classifier: str, df: pd.DataFrame, ext: pd.DataFrame, features: list[str], train_mask: np.ndarray, test_mask: np.ndarray, domains: list[str], max_features: int, fold_name: str) -> FitResult:
    selected, sel = meta_select(df, features, train_mask, domains, max_features)
    sel["candidate"] = candidate
    sel["fold_model"] = fold_name
    if len(selected) < 3:
        return FitResult(candidate, fold_name, False, [], [], [], [], sel, 0, {}, "INSUFFICIENT_STABLE_FEATURES")
    train = df.loc[train_mask]
    test = df.loc[test_mask]
    imp = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    xtr = scaler.fit_transform(imp.fit_transform(train[selected]))
    ytr = train["label"].astype(int).to_numpy()
    xte = scaler.transform(imp.transform(test[selected]))
    xext = scaler.transform(imp.transform(ext[selected]))
    params, model = select_hyperparams(xtr, ytr, classifier)
    pte = model.predict_proba(xte)[:, 1]
    pext = model.predict_proba(xext)[:, 1]
    ext_long = [{"neutral_external_id": r["neutral_external_id"], "neutral_patient_id": r["neutral_patient_id"], "candidate": candidate, "fold_model": fold_name, "prob_COP_fold": float(p)} for (_, r), p in zip(ext.iterrows(), pext)]
    return FitResult(candidate, fold_name, True, test["neutral_imaging_id"].astype(str).tolist(), test["label"].astype(int).tolist(), pte.astype(float).tolist(), ext_long, sel, len(selected), params, "PASS")


def embedding_pc_frame(arr: np.ndarray, meta: pd.DataFrame, train_mask: np.ndarray, apply_mask: np.ndarray | None, ext_arr: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str], dict[str, Any]]:
    xtr_raw = arr[train_mask]
    std = StandardScaler()
    xtr_std = std.fit_transform(xtr_raw)
    pca_full = PCA(random_state=SEED)
    pca_full.fit(xtr_std)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    k = min(int(np.searchsorted(cum, 0.90) + 1), 20, arr.shape[1], arr.shape[0] - 1)
    pca = PCA(n_components=k, random_state=SEED)
    pca.fit(xtr_std)
    all_pc = pca.transform(std.transform(arr))
    pc_cols = [f"PC{i+1:02d}" for i in range(k)]
    frame = meta[["neutral_imaging_id", "label", "outer_fold", "primary_domain"]].copy()
    for i, c in enumerate(pc_cols):
        frame[c] = all_pc[:, i]
    ext_frame = None
    if ext_arr is not None:
        ext_pc = pca.transform(std.transform(ext_arr))
        ext_frame = pd.DataFrame({"neutral_external_id": apply_mask["neutral_external_id"].astype(str).tolist(), "neutral_patient_id": apply_mask["neutral_patient_id"].astype(str).tolist()})
        for i, c in enumerate(pc_cols):
            ext_frame[c] = ext_pc[:, i]
    info = {"n_components": k, "variance_explained": float(np.sum(pca.explained_variance_ratio_))}
    return frame, ext_frame, pc_cols, info


def fit_embedding_candidate(candidate: str, family: str, classifier: str, meta: pd.DataFrame, arr: np.ndarray, ext_meta: pd.DataFrame, ext_arr: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray, domains: list[str], max_features: int, fold_name: str) -> FitResult:
    pc_df, pc_ext, pc_cols, pca_info = embedding_pc_frame(arr, meta, train_mask, ext_meta, ext_arr)
    selected, sel = meta_select(pc_df, pc_cols, train_mask, domains, max_features, prefix=f"{family}_")
    sel["candidate"] = candidate
    sel["fold_model"] = fold_name
    sel["pca_n_components"] = pca_info["n_components"]
    sel["pca_variance_explained"] = pca_info["variance_explained"]
    if len(selected) < 3:
        return FitResult(candidate, fold_name, False, [], [], [], [], sel, 0, {}, "INSUFFICIENT_STABLE_PCS")
    train = pc_df.loc[train_mask]
    test = pc_df.loc[test_mask]
    imp = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    xtr = scaler.fit_transform(imp.fit_transform(train[selected]))
    ytr = train["label"].astype(int).to_numpy()
    xte = scaler.transform(imp.transform(test[selected]))
    xext = scaler.transform(imp.transform(pc_ext[selected]))
    params, model = select_hyperparams(xtr, ytr, classifier)
    pte = model.predict_proba(xte)[:, 1]
    pext = model.predict_proba(xext)[:, 1]
    ext_long = [{"neutral_external_id": r["neutral_external_id"], "neutral_patient_id": r["neutral_patient_id"], "candidate": candidate, "fold_model": fold_name, "prob_COP_fold": float(p)} for (_, r), p in zip(pc_ext.iterrows(), pext)]
    return FitResult(candidate, fold_name, True, test["neutral_imaging_id"].astype(str).tolist(), test["label"].astype(int).tolist(), pte.astype(float).tolist(), ext_long, sel, len(selected), params | pca_info, "PASS")


def combine_tabular(a: pd.DataFrame, b: pd.DataFrame, fa: list[str], fb: list[str]) -> tuple[pd.DataFrame, list[str]]:
    cols = ["neutral_imaging_id", "label", "outer_fold", "primary_domain"]
    out = a[cols + fa].merge(b[["neutral_imaging_id"] + fb], on="neutral_imaging_id", how="left")
    return out, fa + fb


def run_standard(base: pd.DataFrame, domains: list[str], tab_i: dict[str, pd.DataFrame], tab_e: dict[str, pd.DataFrame], tab_f: dict[str, list[str]], emb_mi: dict[str, pd.DataFrame], emb_me: dict[str, pd.DataFrame], emb_i: dict[str, np.ndarray], emb_e: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[FitResult]]]:
    results: dict[str, list[FitResult]] = {}
    all_sel, all_oof, all_ext = [], [], []
    hr_i, hr_feats = combine_tabular(tab_i["FAMILY_A_HANDCRAFTED"], tab_i["FAMILY_B_RAD_D1_FINAL13"], tab_f["FAMILY_A_HANDCRAFTED"], tab_f["FAMILY_B_RAD_D1_FINAL13"])
    hr_e = tab_e["FAMILY_A_HANDCRAFTED"][["neutral_external_id", "neutral_patient_id"] + tab_f["FAMILY_A_HANDCRAFTED"]].merge(tab_e["FAMILY_B_RAD_D1_FINAL13"][["neutral_patient_id"] + tab_f["FAMILY_B_RAD_D1_FINAL13"]], on="neutral_patient_id", how="left")
    for fold in sorted(base["outer_fold"].unique()):
        train_mask = base["outer_fold"].ne(fold).to_numpy()
        test_mask = base["outer_fold"].eq(fold).to_numpy()
        fold_name = f"fold_{fold}"
        fold_results = [
            fit_tabular_candidate("SP1_STABLE_HANDCRAFTED_ELASTICNET", "elasticnet", hr_i, hr_e, hr_feats, train_mask, test_mask, domains, 12, fold_name),
            fit_tabular_candidate("SP2_STABLE_HANDCRAFTED_RANDOM_FOREST", "rf", hr_i, hr_e, hr_feats, train_mask, test_mask, domains, 12, fold_name),
        ]
        if "FAMILY_C_GENESIS_FROZEN" in emb_i:
            fold_results.append(fit_embedding_candidate("SP3_STABLE_GENESIS_SUBSPACE_ELASTICNET", "GENESIS", "elasticnet", emb_mi["FAMILY_C_GENESIS_FROZEN"], emb_i["FAMILY_C_GENESIS_FROZEN"], emb_me["FAMILY_C_GENESIS_FROZEN"], emb_e["FAMILY_C_GENESIS_FROZEN"], train_mask, test_mask, domains, 8, fold_name))
        if "FAMILY_D_TANGERINE_FROZEN" in emb_i:
            fold_results.append(fit_embedding_candidate("SP4_STABLE_TANGERINE_SUBSPACE_ELASTICNET", "TANGERINE", "elasticnet", emb_mi["FAMILY_D_TANGERINE_FROZEN"], emb_i["FAMILY_D_TANGERINE_FROZEN"], emb_me["FAMILY_D_TANGERINE_FROZEN"], emb_e["FAMILY_D_TANGERINE_FROZEN"], train_mask, test_mask, domains, 8, fold_name))
        for fr in fold_results:
            results.setdefault(fr.candidate, []).append(fr)
            all_sel.append(fr.selection)
            for nid, y, p in zip(fr.pred_ids, fr.pred_y, fr.pred_p):
                all_oof.append({"neutral_imaging_id": nid, "label": y, "candidate": fr.candidate, "fold_model": fold_name, "prob_COP": p, "dimension": fr.dimension, "technical_status": fr.status})
            all_ext.extend(fr.ext_long)
        # Multi-representation uses only current fold selected source-only transformed features.
        ok_sp1 = next((r for r in fold_results if r.candidate.startswith("SP1") and r.valid), None)
        ok_sp3 = next((r for r in fold_results if r.candidate.startswith("SP3") and r.valid), None)
        ok_sp4 = next((r for r in fold_results if r.candidate.startswith("SP4") and r.valid), None)
        multi_parts_i, multi_parts_e, multi_feats = [], [], []
        if ok_sp1:
            sel = ok_sp1.selection.loc[ok_sp1.selection["selected_after_pruning"].eq(True), "raw_feature"].astype(str).tolist()
            multi_parts_i.append(hr_i[["neutral_imaging_id", "label", "outer_fold", "primary_domain"] + sel].copy())
            multi_parts_e.append(hr_e[["neutral_external_id", "neutral_patient_id"] + sel].copy())
            multi_feats += sel
        for ok, fam, label in [(ok_sp3, "FAMILY_C_GENESIS_FROZEN", "GENESIS"), (ok_sp4, "FAMILY_D_TANGERINE_FROZEN", "TANGERINE")]:
            if ok:
                meta = emb_mi[fam]
                pc_df, pc_ext, _, _ = embedding_pc_frame(emb_i[fam], meta, train_mask, emb_me[fam], emb_e[fam])
                sel = ok.selection.loc[ok.selection["selected_after_pruning"].eq(True), "raw_feature"].astype(str).tolist()
                renamed = {s: f"{label}_{s}" for s in sel}
                part_i = pc_df[["neutral_imaging_id", "label", "outer_fold", "primary_domain"] + sel].rename(columns=renamed)
                part_e = pc_ext[["neutral_external_id", "neutral_patient_id"] + sel].rename(columns=renamed)
                multi_parts_i.append(part_i)
                multi_parts_e.append(part_e)
                multi_feats += list(renamed.values())
        if multi_feats:
            mi = multi_parts_i[0]
            for p in multi_parts_i[1:]:
                mi = mi.merge(p.drop(columns=["label", "outer_fold", "primary_domain"]), on="neutral_imaging_id", how="left")
            me = multi_parts_e[0]
            for p in multi_parts_e[1:]:
                me = me.merge(p.drop(columns=["neutral_external_id"]), on="neutral_patient_id", how="left")
            for cand, clf in [("SP5_STABLE_MULTI_REPRESENTATION_ELASTICNET", "elasticnet"), ("SP6_STABLE_MULTI_REPRESENTATION_RANDOM_FOREST", "rf")]:
                fr = fit_tabular_candidate(cand, clf, mi, me, multi_feats, train_mask, test_mask, domains, 20, fold_name)
                results.setdefault(fr.candidate, []).append(fr)
                all_sel.append(fr.selection)
                for nid, y, p in zip(fr.pred_ids, fr.pred_y, fr.pred_p):
                    all_oof.append({"neutral_imaging_id": nid, "label": y, "candidate": fr.candidate, "fold_model": fold_name, "prob_COP": p, "dimension": fr.dimension, "technical_status": fr.status})
                all_ext.extend(fr.ext_long)
    return pd.DataFrame(all_oof), pd.DataFrame(all_ext), pd.concat(all_sel, ignore_index=True) if all_sel else pd.DataFrame(), results


def add_equal_ensemble(oof: pd.DataFrame, ext: pd.DataFrame, label: str = "SP7_STABLE_REPRESENTATION_EQUAL_ENSEMBLE") -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    members = [m for m in ["SP1_STABLE_HANDCRAFTED_ELASTICNET", "SP3_STABLE_GENESIS_SUBSPACE_ELASTICNET", "SP4_STABLE_TANGERINE_SUBSPACE_ELASTICNET"] if m in set(oof["candidate"])]
    valid_members = []
    for m in members:
        if oof[oof["candidate"].eq(m)]["neutral_imaging_id"].nunique() == 375:
            valid_members.append(m)
    if not valid_members:
        return oof, ext, []
    piv = oof[oof["candidate"].isin(valid_members)].pivot_table(index=["neutral_imaging_id", "label"], columns="candidate", values="prob_COP")
    piv[label] = piv[valid_members].mean(axis=1)
    add = piv.reset_index()[["neutral_imaging_id", "label", label]].rename(columns={label: "prob_COP"})
    add["candidate"] = label
    add["fold_model"] = "equal_average_standard_oof_members"
    add["dimension"] = len(valid_members)
    add["technical_status"] = "PASS"
    oof = pd.concat([oof, add[oof.columns]], ignore_index=True)
    epiv = ext[ext["candidate"].isin(valid_members)].groupby(["neutral_external_id", "neutral_patient_id", "candidate"], as_index=False)["prob_COP_fold"].mean()
    epiv = epiv.pivot_table(index=["neutral_external_id", "neutral_patient_id"], columns="candidate", values="prob_COP_fold")
    epiv[label] = epiv[valid_members].mean(axis=1)
    eadd = epiv.reset_index()[["neutral_external_id", "neutral_patient_id", label]].rename(columns={label: "prob_COP_fold"})
    eadd["candidate"] = label
    eadd["fold_model"] = "equal_average_of_member_means"
    ext = pd.concat([ext, eadd[ext.columns]], ignore_index=True)
    return oof, ext, valid_members


def summarize_predictions(pred: pd.DataFrame, cohort: str) -> pd.DataFrame:
    rows = []
    for cand, sub in pred.groupby("candidate"):
        p = sub["prob_COP"].to_numpy(float)
        y = sub["label"].to_numpy(int)
        m = metrics(y, p)
        ci = strat_bootstrap(y, p)
        rows.append({"candidate": cand, "cohort": cohort, "n": len(sub), **m, **ci})
    return pd.DataFrame(rows)


def external_metrics(ext_long: pd.DataFrame, ext_labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean = ext_long.groupby(["neutral_external_id", "neutral_patient_id", "candidate"], as_index=False)["prob_COP_fold"].mean().rename(columns={"prob_COP_fold": "prob_COP_mean"})
    merged = mean.merge(ext_labels[["neutral_external_id", "label"]], on="neutral_external_id", how="left")
    rows, boot = [], []
    for cand, sub in merged.groupby("candidate"):
        y = sub["label"].astype(int).to_numpy()
        p = sub["prob_COP_mean"].to_numpy(float)
        m = metrics(y, p)
        ci = strat_bootstrap(y, p)
        cap_mean = float(np.mean(p[y == 0])) if np.any(y == 0) else np.nan
        cop_mean = float(np.mean(p[y == 1])) if np.any(y == 1) else np.nan
        rows.append({"candidate": cand, "cohort": "External33", "n": len(sub), **m, "AUROC_CI": f"{ci['AUROC_ci_low']:.6f}-{ci['AUROC_ci_high']:.6f}", "CAP_mean_prob_COP": cap_mean, "COP_mean_prob_COP": cop_mean, "class_separation": cop_mean - cap_mean})
        boot.append({"candidate": cand, **ci})
    return pd.DataFrame(rows), pd.DataFrame(boot)


def lodo(tab_i: dict[str, pd.DataFrame], tab_e: dict[str, pd.DataFrame], tab_f: dict[str, list[str]], emb_mi: dict[str, pd.DataFrame], emb_me: dict[str, pd.DataFrame], emb_i: dict[str, np.ndarray], emb_e: dict[str, np.ndarray], domains: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hr_i, hr_feats = combine_tabular(tab_i["FAMILY_A_HANDCRAFTED"], tab_i["FAMILY_B_RAD_D1_FINAL13"], tab_f["FAMILY_A_HANDCRAFTED"], tab_f["FAMILY_B_RAD_D1_FINAL13"])
    hr_e = tab_e["FAMILY_A_HANDCRAFTED"][["neutral_external_id", "neutral_patient_id"] + tab_f["FAMILY_A_HANDCRAFTED"]].merge(tab_e["FAMILY_B_RAD_D1_FINAL13"][["neutral_patient_id"] + tab_f["FAMILY_B_RAD_D1_FINAL13"]], on="neutral_patient_id", how="left")
    preds, sels = [], []
    for d in domains:
        train_mask = hr_i["primary_domain"].ne(d).to_numpy()
        test_mask = hr_i["primary_domain"].eq(d).to_numpy()
        fold_name = f"heldout_{d}"
        frs = [
            fit_tabular_candidate("SP1_STABLE_HANDCRAFTED_ELASTICNET", "elasticnet", hr_i, hr_e, hr_feats, train_mask, test_mask, [x for x in domains if x != d], 12, fold_name),
            fit_tabular_candidate("SP2_STABLE_HANDCRAFTED_RANDOM_FOREST", "rf", hr_i, hr_e, hr_feats, train_mask, test_mask, [x for x in domains if x != d], 12, fold_name),
        ]
        if "FAMILY_C_GENESIS_FROZEN" in emb_i:
            frs.append(fit_embedding_candidate("SP3_STABLE_GENESIS_SUBSPACE_ELASTICNET", "GENESIS", "elasticnet", emb_mi["FAMILY_C_GENESIS_FROZEN"], emb_i["FAMILY_C_GENESIS_FROZEN"], emb_me["FAMILY_C_GENESIS_FROZEN"], emb_e["FAMILY_C_GENESIS_FROZEN"], emb_mi["FAMILY_C_GENESIS_FROZEN"]["primary_domain"].ne(d).to_numpy(), emb_mi["FAMILY_C_GENESIS_FROZEN"]["primary_domain"].eq(d).to_numpy(), [x for x in domains if x != d], 8, fold_name))
        if "FAMILY_D_TANGERINE_FROZEN" in emb_i:
            frs.append(fit_embedding_candidate("SP4_STABLE_TANGERINE_SUBSPACE_ELASTICNET", "TANGERINE", "elasticnet", emb_mi["FAMILY_D_TANGERINE_FROZEN"], emb_i["FAMILY_D_TANGERINE_FROZEN"], emb_me["FAMILY_D_TANGERINE_FROZEN"], emb_e["FAMILY_D_TANGERINE_FROZEN"], emb_mi["FAMILY_D_TANGERINE_FROZEN"]["primary_domain"].ne(d).to_numpy(), emb_mi["FAMILY_D_TANGERINE_FROZEN"]["primary_domain"].eq(d).to_numpy(), [x for x in domains if x != d], 8, fold_name))
        ok_sp1 = next((r for r in frs if r.candidate.startswith("SP1") and r.valid), None)
        ok_sp3 = next((r for r in frs if r.candidate.startswith("SP3") and r.valid), None)
        ok_sp4 = next((r for r in frs if r.candidate.startswith("SP4") and r.valid), None)
        multi_parts_i, multi_parts_e, multi_feats = [], [], []
        if ok_sp1:
            sel = ok_sp1.selection.loc[ok_sp1.selection["selected_after_pruning"].eq(True), "raw_feature"].astype(str).tolist()
            multi_parts_i.append(hr_i[["neutral_imaging_id", "label", "outer_fold", "primary_domain"] + sel].copy())
            multi_parts_e.append(hr_e[["neutral_external_id", "neutral_patient_id"] + sel].copy())
            multi_feats += sel
        for ok, fam, label in [(ok_sp3, "FAMILY_C_GENESIS_FROZEN", "GENESIS"), (ok_sp4, "FAMILY_D_TANGERINE_FROZEN", "TANGERINE")]:
            if ok:
                meta = emb_mi[fam]
                em_train_mask = meta["primary_domain"].ne(d).to_numpy()
                pc_df, pc_ext, _, _ = embedding_pc_frame(emb_i[fam], meta, em_train_mask, emb_me[fam], emb_e[fam])
                sel = ok.selection.loc[ok.selection["selected_after_pruning"].eq(True), "raw_feature"].astype(str).tolist()
                renamed = {s: f"{label}_{s}" for s in sel}
                multi_parts_i.append(pc_df[["neutral_imaging_id", "label", "outer_fold", "primary_domain"] + sel].rename(columns=renamed))
                multi_parts_e.append(pc_ext[["neutral_external_id", "neutral_patient_id"] + sel].rename(columns=renamed))
                multi_feats += list(renamed.values())
        if multi_feats:
            mi = multi_parts_i[0]
            for part in multi_parts_i[1:]:
                mi = mi.merge(part.drop(columns=["label", "outer_fold", "primary_domain"]), on="neutral_imaging_id", how="left")
            me = multi_parts_e[0]
            for part in multi_parts_e[1:]:
                me = me.merge(part.drop(columns=["neutral_external_id"]), on="neutral_patient_id", how="left")
            mi_train_mask = mi["primary_domain"].ne(d).to_numpy()
            mi_test_mask = mi["primary_domain"].eq(d).to_numpy()
            frs.extend([
                fit_tabular_candidate("SP5_STABLE_MULTI_REPRESENTATION_ELASTICNET", "elasticnet", mi, me, multi_feats, mi_train_mask, mi_test_mask, [x for x in domains if x != d], 20, fold_name),
                fit_tabular_candidate("SP6_STABLE_MULTI_REPRESENTATION_RANDOM_FOREST", "rf", mi, me, multi_feats, mi_train_mask, mi_test_mask, [x for x in domains if x != d], 20, fold_name),
            ])
        for fr in frs:
            sels.append(fr.selection)
            for nid, y, p in zip(fr.pred_ids, fr.pred_y, fr.pred_p):
                preds.append({"neutral_imaging_id": nid, "label": y, "primary_domain": d, "candidate": fr.candidate, "fold_model": fold_name, "prob_COP": p, "dimension": fr.dimension, "technical_status": fr.status})
    p = pd.DataFrame(preds)
    if not p.empty:
        possible_members = [
            "SP1_STABLE_HANDCRAFTED_ELASTICNET",
            "SP3_STABLE_GENESIS_SUBSPACE_ELASTICNET",
            "SP4_STABLE_TANGERINE_SUBSPACE_ELASTICNET",
        ]
        valid_members = [m for m in possible_members if m in set(p["candidate"])]
        if valid_members:
            piv = p[p["candidate"].isin(valid_members)].pivot_table(
                index=["neutral_imaging_id", "label", "primary_domain"],
                columns="candidate",
                values="prob_COP",
            )
            piv["SP7_STABLE_REPRESENTATION_EQUAL_ENSEMBLE"] = piv[valid_members].mean(axis=1)
            add = (
                piv.reset_index()[["neutral_imaging_id", "label", "primary_domain", "SP7_STABLE_REPRESENTATION_EQUAL_ENSEMBLE"]]
                .rename(columns={"SP7_STABLE_REPRESENTATION_EQUAL_ENSEMBLE": "prob_COP"})
            )
            add["candidate"] = "SP7_STABLE_REPRESENTATION_EQUAL_ENSEMBLE"
            add["fold_model"] = "equal_average_lodo_members"
            add["dimension"] = len(valid_members)
            add["technical_status"] = "PASS"
            p = pd.concat([p, add[p.columns]], ignore_index=True)
    per, agg = [], []
    for (cand, dom), sub in p.groupby(["candidate", "primary_domain"]):
        per.append({"candidate": cand, "heldout_domain": dom, "n": len(sub), **metrics(sub["label"].to_numpy(int), sub["prob_COP"].to_numpy(float))})
    per_df = pd.DataFrame(per)
    for cand, sub in p.groupby("candidate"):
        m = metrics(sub["label"].to_numpy(int), sub["prob_COP"].to_numpy(float))
        dom_aucs = per_df.loc[per_df["candidate"].eq(cand), "AUROC"].dropna().to_numpy(float)
        agg.append({"candidate": cand, "cohort": "Internal375_LODO", "n": len(sub), **m, "worst_domain_AUROC": float(np.min(dom_aucs)) if len(dom_aucs) else np.nan, "median_domain_AUROC": float(np.median(dom_aucs)) if len(dom_aucs) else np.nan, "domain_AUROC_range": float(np.max(dom_aucs) - np.min(dom_aucs)) if len(dom_aucs) else np.nan})
    return p, per_df, pd.DataFrame(agg)


def historical_reference() -> pd.DataFrame:
    rows = []
    def add(path: Path, label_map: dict[str, str], cohort_hint: str | None = None):
        if not path.exists():
            return
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            cand = str(r.get("candidate", r.get("model", "")))
            if cand in label_map:
                rows.append({"candidate": label_map[cand], "source_candidate": cand, "cohort": cohort_hint or r.get("cohort", ""), "AUROC": r.get("AUROC", np.nan), "AUPRC": r.get("AUPRC", np.nan), "Brier": r.get("Brier", np.nan), "source_file": str(path)})
    add(PATHS["d1_internal_metrics"], {"lesion_radiomics_reference_PYRADIOMICS_STANDARDIZED_RF": "H0_D1_STANDARDIZED_RADIOMICS"}, "Internal375_STANDARD_OOF")
    add(PATHS["d1_external_metrics"], {"lesion_radiomics_reference_PYRADIOMICS_STANDARDIZED_RF": "H0_D1_STANDARDIZED_RADIOMICS"}, "External33")
    add(PATHS["v34_internal_metrics"], {"R0_GENESIS_B2_BASELINE": "H1_GENESIS_B2", "R1_GENESIS_B2_DEEP_CORAL": "H2_C3_GRL", "R2_GENESIS_B2_FISH": "H_UNSTABLE_R2_FISH"}, None)
    add(PATHS["v34_lodo_metrics"], {"R0_GENESIS_B2_BASELINE": "H1_GENESIS_B2", "R1_GENESIS_B2_DEEP_CORAL": "H2_C3_GRL", "R2_GENESIS_B2_FISH": "H_UNSTABLE_R2_FISH"}, None)
    add(PATHS["v34_external_metrics"], {"R0_GENESIS_B2_BASELINE": "H1_GENESIS_B2", "R1_GENESIS_B2_DEEP_CORAL": "H2_C3_GRL", "R2_GENESIS_B2_FISH": "H_UNSTABLE_R2_FISH"}, None)
    add(PATHS["v35_internal_metrics"], {"T1_TANGERINE_FROZEN_LINEAR_PROBE": "H3_TANGERINE_FROZEN"}, None)
    add(PATHS["v35_lodo_metrics"], {"T1_TANGERINE_FROZEN_LINEAR_PROBE": "H3_TANGERINE_FROZEN"}, None)
    add(PATHS["v35_external_metrics"], {"T1_TANGERINE_FROZEN_LINEAR_PROBE": "H3_TANGERINE_FROZEN"}, None)
    return pd.DataFrame(rows)


def write_protocol_files(audit: list[dict[str, Any]], domains: list[str], ext_labels_loaded: bool) -> None:
    write_text(ROOT / "00_stable_multirepresentation_README.md", "# V3.8 Stable Phenotype Representation\n\nSource-only low-dimensional stable phenotype image-only experiment. V4 was not entered.\n")
    literature = [
        ("Stable/reproducible radiomics", "DIRECTLY_SUPPORTED", "IBSI standardization: Zwanenburg et al., Radiology 2020, DOI 10.1148/radiol.2020191145."),
        ("Cross-domain feature stability", "ADAPTED_FROM_LITERATURE", "Dataset shift and shortcut-learning literature motivate source-domain stability checks."),
        ("Scanner-sensitive imaging features", "DIRECTLY_SUPPORTED", "CT radiomics reproducibility studies and IBSI describe scanner/protocol sensitivity."),
        ("Multi-domain disease-effect consistency", "ADAPTED_FROM_LITERATURE", "Random-effects synthesis is adapted as an internal source-domain stability screen."),
        ("Feature dimensionality reduction", "DIRECTLY_SUPPORTED", "Low-dimensional penalized models reduce variance in small-sample medical imaging."),
        ("Stability selection", "DIRECTLY_SUPPORTED", "Meinshausen and Buhlmann stability-selection concept adapted to foldwise feature frequency."),
        ("Random-effects heterogeneity", "ADAPTED_FROM_LITERATURE", "Hedges g, tau2 and I2 are used to screen cross-domain consistency, not as a formal meta-analysis claim."),
        ("Frozen embedding downstream classification", "ADAPTED_FROM_LITERATURE", "Frozen pretrained representations with internal-only PCA support low-dimensional downstream probes."),
        ("Domain-generalization model selection", "PROJECT_SPECIFIC", "LODO is project-locked from V3.4 primary internal DG domains."),
    ]
    md = ["# Literature Justification\n"]
    for topic, status, note in literature:
        md.append(f"- {topic}: `{status}`. {note}")
    md.append("\nWording guard: this is not claimed as META_ANALYSIS_PROVEN.")
    write_text(ROOT / "01_LITERATURE_JUSTIFICATION.md", "\n".join(md) + "\n")
    pd.DataFrame(literature, columns=["topic", "support_level", "note"]).to_csv(DIRS["literature"] / "01_LITERATURE_JUSTIFICATION.csv", index=False)
    pd.DataFrame(audit).to_csv(ROOT / "02_stable_multirepresentation_FEATURE_RESOURCE_AUDIT.csv", index=False)
    candidates = [
        ("SP1_STABLE_HANDCRAFTED_ELASTICNET", "FAMILY_A+FAMILY_B", "ElasticNet Logistic Regression", "EXECUTE_IF_FEATURES_AVAILABLE"),
        ("SP2_STABLE_HANDCRAFTED_RANDOM_FOREST", "FAMILY_A+FAMILY_B", "Random Forest", "EXECUTE_IF_FEATURES_AVAILABLE"),
        ("SP3_STABLE_GENESIS_SUBSPACE_ELASTICNET", "FAMILY_C", "ElasticNet Logistic Regression", "EXECUTE_IF_GENESIS_FROZEN_EMBEDDING_AVAILABLE"),
        ("SP4_STABLE_TANGERINE_SUBSPACE_ELASTICNET", "FAMILY_D", "ElasticNet Logistic Regression", "EXECUTE_IF_TANGERINE_FROZEN_EMBEDDING_AVAILABLE"),
        ("SP5_STABLE_MULTI_REPRESENTATION_ELASTICNET", "FAMILY_E", "ElasticNet Logistic Regression", "EXECUTE_IF_MEMBER_FEATURES_AVAILABLE"),
        ("SP6_STABLE_MULTI_REPRESENTATION_RANDOM_FOREST", "FAMILY_E", "Random Forest", "EXECUTE_IF_MEMBER_FEATURES_AVAILABLE"),
        ("SP7_STABLE_REPRESENTATION_EQUAL_ENSEMBLE", "SP1+SP3+SP4", "Equal probability average", "EXECUTE_WITH_TECHNICALLY_EXECUTABLE_MEMBERS"),
    ]
    pd.DataFrame(candidates, columns=["candidate", "representation_family", "classifier", "prelocked_status"]).to_csv(ROOT / "03_stable_multirepresentation_CANDIDATE_REGISTRY.csv", index=False)
    lock = {
        "created_at": now(),
        "candidate_set_prelocked_before_disease_training": True,
        "external_labels_loaded_before_candidate_lock": ext_labels_loaded,
        "external_labels_allowed_only_after_prediction_lock": True,
        "primary_internal_dg_domains": domains,
        "stable_rule": {"min_evaluable_domains": 3, "abs_pooled_Hedges_g": 0.30, "direction_consistency": 0.75, "I2_max_percent": 60},
        "fallback_rule": {"min_evaluable_domains": 3, "abs_pooled_Hedges_g": 0.20, "direction_consistency": 0.67, "I2_max_percent": 75},
        "max_dimensions": {"SP1_SP2": 12, "SP3_SP4": 8, "SP5_SP6": 20},
        "elasticnet_grid": {"C": [0.1, 1.0, 10.0], "l1_ratio": [0.0, 0.5, 1.0], "class_weight": None},
        "random_forest_grid": {"n_estimators": 500, "max_depth": [3, 5, None], "min_samples_leaf": [3, 5, 10], "max_features": ["sqrt", 0.5], "class_weight": None},
    }
    lock_path = ROOT / "04_stable_multirepresentation_CANDIDATE_SET_LOCK.json"
    write_json(lock_path, lock)
    write_text(ROOT / "04_stable_multirepresentation_CANDIDATE_SET_LOCK_SHA256.txt", f"{sha256_file(lock_path)}  04_stable_multirepresentation_CANDIDATE_SET_LOCK.json\n")


def write_outputs(oof: pd.DataFrame, ext_long: pd.DataFrame, sel: pd.DataFrame, lodo_pred: pd.DataFrame, lodo_per: pd.DataFrame, lodo_agg: pd.DataFrame, ext_labels: pd.DataFrame, sp7_members: list[str], hist: pd.DataFrame, domains: list[str]) -> dict[str, Any]:
    sel.to_csv(ROOT / "05_stable_multirepresentation_FOLDWISE_STABLE_FEATURE_SELECTION.csv", index=False)
    freq = sel[sel.get("selected_after_pruning", False).eq(True)] if "selected_after_pruning" in sel.columns else pd.DataFrame()
    if not freq.empty:
        freq_out = freq.groupby(["candidate", "feature"], as_index=False).agg(selected_fold_count=("fold_model", "nunique"), median_pooled_Hedges_g=("pooled_Hedges_g", "median"), median_I2=("I2", "median"), median_direction_consistency=("direction_consistency", "median"))
        freq_out["selection_frequency"] = freq_out["selected_fold_count"] / 5
        freq_out["CORE_STABLE_FEATURE"] = np.where(freq_out["selected_fold_count"] >= 4, "YES", "NO")
    else:
        freq_out = pd.DataFrame(columns=["candidate", "feature", "selected_fold_count", "selection_frequency"])
    freq_out.to_csv(ROOT / "06_stable_multirepresentation_FEATURE_SELECTION_FREQUENCY.csv", index=False)
    oof.to_csv(ROOT / "07_stable_multirepresentation_INTERNAL_OOF_PREDICTIONS.csv", index=False)
    internal_m = summarize_predictions(oof, "Internal375_STANDARD_OOF")
    internal_m.to_csv(ROOT / "08_stable_multirepresentation_INTERNAL_METRICS.csv", index=False)
    iboot = []
    for cand, sub in oof.groupby("candidate"):
        iboot.append({"candidate": cand, **strat_bootstrap(sub["label"].to_numpy(int), sub["prob_COP"].to_numpy(float))})
    pd.DataFrame(iboot).to_csv(ROOT / "09_stable_multirepresentation_INTERNAL_BOOTSTRAP.csv", index=False)
    lodo_pred.to_csv(ROOT / "10_stable_multirepresentation_LODO_PREDICTIONS.csv", index=False)
    lodo_per.to_csv(ROOT / "11_stable_multirepresentation_LODO_PER_DOMAIN_METRICS.csv", index=False)
    lodo_agg.to_csv(ROOT / "12_stable_multirepresentation_LODO_AGGREGATE_METRICS.csv", index=False)
    sel.to_csv(ROOT / "13_stable_multirepresentation_STABILITY_META_ANALYSIS.csv", index=False)
    hetero = sel.groupby("feature", as_index=False).agg(median_I2=("I2", "median"), max_I2=("I2", "max"), evaluable_folds=("fold_model", "nunique")) if not sel.empty else pd.DataFrame()
    hetero.to_csv(ROOT / "14_stable_multirepresentation_FEATURE_HETEROGENEITY.csv", index=False)
    pareto = internal_m.merge(lodo_agg[["candidate", "AUROC", "worst_domain_AUROC"]].rename(columns={"AUROC": "LODO_AUROC"}), on="candidate", how="left")
    write_text(ROOT / "15_stable_multirepresentation_INTERNAL_PARETO_ANALYSIS.md", "# Internal Stable Representation Pareto\n\n" + pareto.sort_values(["AUROC", "LODO_AUROC"], ascending=False).to_markdown(index=False) + "\n")
    ilock = {"created_at": now(), "status": "PASS", "candidates": sorted(oof["candidate"].unique()), "internal_predictions": str(ROOT / "07_stable_multirepresentation_INTERNAL_OOF_PREDICTIONS.csv"), "external_labels_used": False}
    write_json(ROOT / "16_stable_multirepresentation_INTERNAL_MODEL_LOCK.json", ilock)
    write_text(ROOT / "16_stable_multirepresentation_INTERNAL_MODEL_LOCK_SHA256.txt", f"{sha256_file(ROOT / '16_stable_multirepresentation_INTERNAL_MODEL_LOCK.json')}  16_stable_multirepresentation_INTERNAL_MODEL_LOCK.json\n")
    ext_mean = ext_long.groupby(["neutral_external_id", "neutral_patient_id", "candidate"], as_index=False)["prob_COP_fold"].mean().rename(columns={"prob_COP_fold": "prob_COP_mean"})
    ext_long = ext_long.merge(ext_mean, on=["neutral_external_id", "neutral_patient_id", "candidate"], how="left")
    ext_long.to_csv(ROOT / "17_stable_multirepresentation_EXTERNAL33_PREDICTIONS.csv", index=False)
    pred_lock = {"created_at": now(), "external_label_blind_prediction_complete": True, "unique_external_patients": int(ext_mean["neutral_external_id"].nunique()), "row_count": int(len(ext_mean)), "probability_identity": "P(COP)", "no_probability_flip": True, "external_labels_in_prediction_file": False, "sp7_members": sp7_members}
    write_json(ROOT / "18_stable_multirepresentation_EXTERNAL33_PREDICTION_LOCK.json", pred_lock)
    write_text(ROOT / "18_stable_multirepresentation_EXTERNAL33_PREDICTION_LOCK_SHA256.txt", f"{sha256_file(ROOT / '18_stable_multirepresentation_EXTERNAL33_PREDICTION_LOCK.json')}  18_stable_multirepresentation_EXTERNAL33_PREDICTION_LOCK.json\n")
    ext_m, eboot = external_metrics(ext_long, ext_labels)
    ext_m.to_csv(ROOT / "19_stable_multirepresentation_EXTERNAL33_METRICS.csv", index=False)
    eboot.to_csv(ROOT / "20_stable_multirepresentation_EXTERNAL33_BOOTSTRAP.csv", index=False)
    gap = internal_m[["candidate", "AUROC"]].rename(columns={"AUROC": "Internal_AUROC"}).merge(lodo_agg[["candidate", "AUROC"]].rename(columns={"AUROC": "LODO_AUROC"}), on="candidate", how="left").merge(ext_m[["candidate", "AUROC"]].rename(columns={"AUROC": "External_AUROC"}), on="candidate", how="left")
    gap["Internal_External_gap"] = gap["Internal_AUROC"] - gap["External_AUROC"]
    gap["LODO_External_gap"] = gap["LODO_AUROC"] - gap["External_AUROC"]
    gap.to_csv(ROOT / "21_stable_multirepresentation_GENERALIZATION_GAP.csv", index=False)
    target = gap.copy()
    target["INTERNAL_GE080"] = np.where(target["Internal_AUROC"] >= 0.80, "YES", "NO")
    target["EXTERNAL_GE070"] = np.where(target["External_AUROC"] >= 0.70, "YES", "NO")
    target["BOTH_TARGETS_REACHED"] = np.where((target["Internal_AUROC"] >= 0.80) & (target["External_AUROC"] >= 0.70), "YES", "NO")
    target["INTERNAL_GE082"] = np.where(target["Internal_AUROC"] >= 0.82, "YES", "NO")
    target["EXTERNAL_GE072"] = np.where(target["External_AUROC"] >= 0.72, "YES", "NO")
    target["EXTERNAL_GE075"] = np.where(target["External_AUROC"] >= 0.75, "YES", "NO")
    target.to_csv(ROOT / "22_stable_multirepresentation_TARGET_ACHIEVEMENT_TABLE.csv", index=False)
    write_text(ROOT / "23_stable_multirepresentation_STABLE_FEATURE_INTERPRETATION.md", "# Stable Feature Interpretation\n\n" + freq_out.sort_values(["selected_fold_count", "median_pooled_Hedges_g"], ascending=[False, False]).head(30).to_markdown(index=False) + "\n\nSingle features are interpreted only as stable predictive phenotype candidates.\n")
    best_ext = ext_m.sort_values("AUROC", ascending=False).iloc[0] if not ext_m.empty else None
    best_int = internal_m.sort_values("AUROC", ascending=False).iloc[0] if not internal_m.empty else None
    best_lodo = lodo_agg.sort_values("AUROC", ascending=False).iloc[0] if not lodo_agg.empty else None
    balanced = gap.dropna(subset=["Internal_AUROC", "LODO_AUROC", "External_AUROC"]).copy()
    balanced = balanced[(balanced["Internal_AUROC"] >= 0.70) & (balanced["LODO_AUROC"] >= 0.60)].sort_values(["External_AUROC", "LODO_AUROC", "Internal_AUROC"], ascending=False)
    best_bal = balanced.iloc[0]["candidate"] if not balanced.empty else (best_ext["candidate"] if best_ext is not None else "NONE")
    write_text(ROOT / "24_stable_multirepresentation_IMAGE_MODEL_RANKING.md", f"# V3.8 Image Model Ranking\n\nBest internal candidate: `{best_int['candidate'] if best_int is not None else 'NONE'}`.\n\nBest LODO candidate: `{best_lodo['candidate'] if best_lodo is not None else 'NONE'}`.\n\nBest observed External candidate: `{best_ext['candidate'] if best_ext is not None else 'NONE'}`.\n\nBEST_RELIABLE_BALANCED_IMAGE_MODEL: `{best_bal}`.\n\nHistorical references are read-only; R2 FISH remains UNSTABLE_REFERENCE if listed.\n")
    fusion = "YES" if (target["BOTH_TARGETS_REACHED"].eq("YES").any()) else "NO"
    write_text(ROOT / "25_stable_multirepresentation_FUSION_READINESS_ANALYSIS.md", f"# Fusion Readiness\n\nFUSION_READINESS: `{fusion}`.\n\nNo multimodal model was created in V3.8.\n")
    write_text(ROOT / "26_stable_multirepresentation_ARTICLE_IMPLICATIONS.md", "# Article Implications\n\nV3.8 tests whether source-only domain-stable low-dimensional image phenotypes improve External33 generalization. External status remains PERFORMANCE_INFORMED_EXPLORATORY_EXTERNAL_GENERALIZATION.\n")
    main = internal_m[["candidate", "AUROC", "AUPRC", "Brier"]].rename(columns={"AUROC": "Internal AUROC", "AUPRC": "Internal AUPRC", "Brier": "Internal Brier"})
    main = main.merge(lodo_agg[["candidate", "AUROC", "worst_domain_AUROC"]].rename(columns={"AUROC": "LODO AUROC", "worst_domain_AUROC": "Worst-domain AUROC"}), on="candidate", how="left")
    main = main.merge(ext_m[["candidate", "AUROC", "AUROC_CI", "AUPRC", "Brier", "class_separation"]].rename(columns={"AUROC": "External AUROC", "AUROC_CI": "External AUROC CI", "AUPRC": "External AUPRC", "Brier": "External Brier", "class_separation": "External class separation"}), on="candidate", how="left")
    dim = oof.groupby("candidate", as_index=False)["dimension"].median().rename(columns={"dimension": "Final dimension"})
    main = main.merge(dim, on="candidate", how="left")
    main["Representation family"] = main["candidate"].map(lambda x: "Stable phenotype representation")
    main["Classifier"] = main["candidate"].map(lambda x: "Random Forest" if "RANDOM_FOREST" in x else ("Equal ensemble" if "ENSEMBLE" in x else "ElasticNet Logistic Regression"))
    main["Internal-External gap"] = main["Internal AUROC"] - main["External AUROC"]
    main["LODO-External gap"] = main["LODO AUROC"] - main["External AUROC"]
    main["Internal_GE080"] = np.where(main["Internal AUROC"] >= 0.80, "YES", "NO")
    main["External_GE070"] = np.where(main["External AUROC"] >= 0.70, "YES", "NO")
    main["Both_targets_reached"] = np.where((main["Internal AUROC"] >= 0.80) & (main["External AUROC"] >= 0.70), "YES", "NO")
    main["Feature-selection stability"] = "see 06_stable_multirepresentation_FEATURE_SELECTION_FREQUENCY.csv"
    main["Technical status"] = "PASS"
    main = main.rename(columns={"candidate": "Candidate"})
    main.to_csv(ROOT / "stable_multirepresentation_FINAL_MAIN_RESULT_TABLE.csv", index=False)
    hist.to_csv(ROOT / "stable_multirepresentation_HISTORICAL_IMAGE_REFERENCE_METRICS.csv", index=False)
    status = "PASS" if target["BOTH_TARGETS_REACHED"].eq("YES").any() else "PASS_WITH_LIMITATIONS"
    lock = {
        "created_at": now(),
        "status": status,
        "v4_entered": False,
        "external_status": "PERFORMANCE_INFORMED_EXPLORATORY_EXTERNAL_GENERALIZATION",
        "external_labels_used_in_training_feature_selection_tuning": False,
        "external_prediction_lock_sha256": sha256_file(ROOT / "18_stable_multirepresentation_EXTERNAL33_PREDICTION_LOCK.json"),
        "internal_n": 375,
        "external_n": 33,
        "sp7_members": sp7_members,
        "best_internal_candidate": str(best_int["candidate"] if best_int is not None else "NONE"),
        "best_lodo_candidate": str(best_lodo["candidate"] if best_lodo is not None else "NONE"),
        "best_observed_external_candidate": str(best_ext["candidate"] if best_ext is not None else "NONE"),
        "BEST_RELIABLE_BALANCED_IMAGE_MODEL": str(best_bal),
        "stable_multirepresentation_PRIMARY_ENGINEERING_TARGET_REACHED": bool(target["BOTH_TARGETS_REACHED"].eq("YES").any()),
        "FUSION_READINESS": fusion,
        "outputs": [str(ROOT / "stable_multirepresentation_FINAL_MAIN_RESULT_TABLE.csv"), str(ROOT / "27_stable_multirepresentation_FINAL_RESULT_LOCK.json")],
    }
    write_json(ROOT / "27_stable_multirepresentation_FINAL_RESULT_LOCK.json", lock)
    write_text(ROOT / "27_stable_multirepresentation_FINAL_RESULT_LOCK_SHA256.txt", f"{sha256_file(ROOT / '27_stable_multirepresentation_FINAL_RESULT_LOCK.json')}  27_stable_multirepresentation_FINAL_RESULT_LOCK.json\n")
    write_text(ROOT / "CHATGPT_HANDOFF_stable_multirepresentation_STABLE_PHENOTYPE_REPRESENTATION.md", f"# ChatGPT Handoff V3.8 Stable Phenotype Representation\n\nFinal status: {status}\n\nExternal status: PERFORMANCE_INFORMED_EXPLORATORY_EXTERNAL_GENERALIZATION\n\nV4 entered: NO\n\nExternal labels used in training/feature selection/tuning: NO\n\nSP7 members: {', '.join(sp7_members)}\n\nBest observed External candidate: {lock['best_observed_external_candidate']}\n\nBEST_RELIABLE_BALANCED_IMAGE_MODEL: {lock['BEST_RELIABLE_BALANCED_IMAGE_MODEL']}\n\nPrimary target reached: {'YES' if lock['stable_multirepresentation_PRIMARY_ENGINEERING_TARGET_REACHED'] else 'NO'}\n\nFusion readiness: {fusion}\n\nFinal table: {ROOT / 'stable_multirepresentation_FINAL_MAIN_RESULT_TABLE.csv'}\n\nFinal lock: {ROOT / '27_stable_multirepresentation_FINAL_RESULT_LOCK.json'}\n\nFinal lock SHA256: {sha256_file(ROOT / '27_stable_multirepresentation_FINAL_RESULT_LOCK.json')}\n\nNext: STOP_AND_WAIT_FOR_CHATGPT_REVIEW\n")
    return lock


def main() -> None:
    mkdirs()
    print(f"{now()} V3.8 started", flush=True)
    base, ext_xw, domains = load_base()
    tab_i, tab_e, tab_f, audit = load_tabular_features(base, ext_xw)
    ext_labels_prelock_loaded = False
    ext_labels_placeholder = pd.DataFrame()
    emb_mi, emb_me, emb_i, emb_e, eaudit = load_embedding_family(base, ext_xw, ext_labels_placeholder)
    audit += eaudit
    write_protocol_files(audit, domains, ext_labels_prelock_loaded)
    oof, ext_long, sel, _ = run_standard(base, domains, tab_i, tab_e, tab_f, emb_mi, emb_me, emb_i, emb_e)
    oof, ext_long, sp7_members = add_equal_ensemble(oof, ext_long)
    lodo_pred, lodo_per, lodo_agg = lodo(tab_i, tab_e, tab_f, emb_mi, emb_me, emb_i, emb_e, domains)
    ext_labels = load_external_labels(ext_xw)
    hist = historical_reference()
    lock = write_outputs(oof, ext_long, sel, lodo_pred, lodo_per, lodo_agg, ext_labels, sp7_members, hist, domains)
    main_table = pd.read_csv(ROOT / "stable_multirepresentation_FINAL_MAIN_RESULT_TABLE.csv")
    print("=" * 60)
    print("V3.8 STABLE PHENOTYPE REPRESENTATION")
    print("=" * 60)
    print("Primary Internal DG domain:")
    print(",".join(domains))
    print("")
    print("Feature families available:")
    for family in ["FAMILY_A_HANDCRAFTED", "FAMILY_B_RAD_D1_FINAL13", "FAMILY_C_GENESIS_FROZEN", "FAMILY_D_TANGERINE_FROZEN"]:
        row = next((r for r in audit if r["resource"] == family), None)
        print(f"{family}: {'YES' if row and row['available'] else 'NO'}")
    print("--------------------------------------------")
    for cand in [
        "SP1_STABLE_HANDCRAFTED_ELASTICNET",
        "SP2_STABLE_HANDCRAFTED_RANDOM_FOREST",
        "SP3_STABLE_GENESIS_SUBSPACE_ELASTICNET",
        "SP4_STABLE_TANGERINE_SUBSPACE_ELASTICNET",
        "SP5_STABLE_MULTI_REPRESENTATION_ELASTICNET",
        "SP6_STABLE_MULTI_REPRESENTATION_RANDOM_FOREST",
        "SP7_STABLE_REPRESENTATION_EQUAL_ENSEMBLE",
    ]:
        sub = main_table[main_table["Candidate"].eq(cand)]
        if sub.empty:
            print(cand)
            print("Technical status: NOT_EXECUTED")
        else:
            r = sub.iloc[0]
            print(cand)
            print(f"Dimension: {r['Final dimension']}")
            print(f"Internal AUROC: {r['Internal AUROC']}")
            print(f"LODO AUROC: {r['LODO AUROC']}")
            print(f"External AUROC: {r['External AUROC']}")
            print(f"External AUPRC: {r['External AUPRC']}")
            print(f"External Brier: {r['External Brier']}")
            print(f"INTERNAL_GE080: {r['Internal_GE080']}")
            print(f"EXTERNAL_GE070: {r['External_GE070']}")
            print(f"BOTH: {r['Both_targets_reached']}")
        print("--------------------------------------------")
    print(f"Best Internal candidate: {lock['best_internal_candidate']}")
    print(f"Best LODO candidate: {lock['best_lodo_candidate']}")
    print(f"Best observed External candidate: {lock['best_observed_external_candidate']}")
    print(f"BEST_RELIABLE_BALANCED_IMAGE_MODEL: {lock['BEST_RELIABLE_BALANCED_IMAGE_MODEL']}")
    print(f"BOTH TARGETS REACHED BY SAME MODEL: {'YES' if lock['stable_multirepresentation_PRIMARY_ENGINEERING_TARGET_REACHED'] else 'NO'}")
    print(f"FUSION_READINESS: {lock['FUSION_READINESS']}")
    print("External labels used in training/feature selection/tuning: NO")
    print("External status: PERFORMANCE_INFORMED_EXPLORATORY_EXTERNAL_GENERALIZATION")
    print("V4 entered: NO")
    print(f"Final lock: {lock['status']}")
    print(f"Handoff: {ROOT / 'CHATGPT_HANDOFF_stable_multirepresentation_STABLE_PHENOTYPE_REPRESENTATION.md'}")
    print("Next: STOP_AND_WAIT_FOR_CHATGPT_REVIEW")
    print("=" * 60)


if __name__ == "__main__":
    main()

