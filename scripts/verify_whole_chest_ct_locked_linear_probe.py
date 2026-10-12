from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "whole_chest_ct"
LOCKED_SKLEARN_VERSION = "1.9.0"


def fixed_linear_probe():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=5000,
            class_weight="balanced",
            random_state=20260905,
        ),
    )


def max_abs_delta(left: pd.Series, right: pd.Series) -> float:
    return float(np.max(np.abs(left.to_numpy(float) - right.to_numpy(float))))


def main() -> int:
    internal_manifest = pd.read_csv(MODEL_DIR / "internal_embedding_manifest.csv")
    external_manifest = pd.read_csv(MODEL_DIR / "external_embedding_manifest.csv")
    locked_internal = pd.read_csv(MODEL_DIR / "internal_oof_predictions.csv")
    locked_external = pd.read_csv(MODEL_DIR / "external_predictions.csv")
    internal_embeddings = np.load(MODEL_DIR / "embeddings" / "internal_embeddings.npy")
    external_embeddings = np.load(MODEL_DIR / "embeddings" / "external_embeddings.npy")

    labels_folds = locked_internal[
        ["neutral_imaging_id", "label", "fold_id", "primary_domain"]
    ].drop_duplicates()
    df = internal_manifest.merge(labels_folds, on="neutral_imaging_id", how="left")
    if df[["label", "fold_id"]].isna().any().any():
        raise RuntimeError("Internal embedding manifest could not be fully matched to locked OOF labels/folds")
    df = df.sort_values("embedding_index").reset_index(drop=True)
    if list(df["embedding_index"]) != list(range(len(df))):
        raise RuntimeError("Internal embedding_index is not contiguous and sorted")

    y = df["label"].to_numpy(dtype=int)
    folds = df["fold_id"].to_numpy(dtype=int)
    oof_prob = np.zeros(len(df), dtype=float)
    models = []
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        clf = fixed_linear_probe()
        clf.fit(internal_embeddings[train], y[train])
        oof_prob[test] = clf.predict_proba(internal_embeddings[test])[:, 1]
        models.append(clf)

    rebuilt_internal = pd.DataFrame(
        {"neutral_imaging_id": df["neutral_imaging_id"], "rebuilt_prob_COP": oof_prob}
    )
    internal_compare = locked_internal.merge(rebuilt_internal, on="neutral_imaging_id", how="inner")
    if len(internal_compare) != 375:
        raise RuntimeError(f"Internal comparison has {len(internal_compare)} rows, expected 375")

    ext_df = external_manifest.sort_values("embedding_index").reset_index(drop=True)
    if list(ext_df["embedding_index"]) != list(range(len(ext_df))):
        raise RuntimeError("External embedding_index is not contiguous and sorted")
    fold_probs = np.vstack([model.predict_proba(external_embeddings)[:, 1] for model in models])
    rebuilt_external = pd.DataFrame(
        {
            "neutral_external_id": ext_df["neutral_external_id"],
            "rebuilt_prob_COP": fold_probs.mean(axis=0),
            "rebuilt_prob_COP_std": fold_probs.std(axis=0),
        }
    )
    external_compare = locked_external.merge(rebuilt_external, on="neutral_external_id", how="inner")
    if len(external_compare) != 33:
        raise RuntimeError(f"External comparison has {len(external_compare)} rows, expected 33")

    report = {
        "model": "whole_chest_ct_foundation_transfer",
        "sklearn_version": sklearn.__version__,
        "locked_sklearn_version": LOCKED_SKLEARN_VERSION,
        "environment_matches_lock": bool(sklearn.__version__ == LOCKED_SKLEARN_VERSION),
        "internal_rows": int(len(internal_compare)),
        "external_rows": int(len(external_compare)),
        "internal_max_abs_prob_delta": max_abs_delta(internal_compare["prob_COP"], internal_compare["rebuilt_prob_COP"]),
        "external_max_abs_prob_delta": max_abs_delta(external_compare["prob_COP"], external_compare["rebuilt_prob_COP"]),
        "external_std_max_abs_delta": max_abs_delta(external_compare["prob_COP_std"], external_compare["rebuilt_prob_COP_std"]),
        "internal_allclose_1e_12": bool(np.allclose(internal_compare["prob_COP"], internal_compare["rebuilt_prob_COP"], atol=1e-12, rtol=0.0)),
        "external_allclose_1e_12": bool(np.allclose(external_compare["prob_COP"], external_compare["rebuilt_prob_COP"], atol=1e-12, rtol=0.0)),
        "external_allclose_csv_serialization_1e_7": bool(np.allclose(external_compare["prob_COP"], external_compare["rebuilt_prob_COP"], atol=1e-7, rtol=0.0)),
    }
    report["status"] = (
        "PASS"
        if report["environment_matches_lock"]
        and report["internal_allclose_1e_12"]
        and report["external_allclose_csv_serialization_1e_7"]
        else "FAIL_REPRODUCTION_ENVIRONMENT_OR_NUMERICAL_CONTRACT"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
