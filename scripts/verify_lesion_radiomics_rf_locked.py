from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "lesion_radiomics"


def main() -> None:
    artifact = joblib.load(MODEL_DIR / "final_random_forest.joblib")
    pipeline = artifact["pipeline"]
    features = list(artifact["features"])

    feature_list = pd.read_csv(MODEL_DIR / "final_feature_list.csv")["feature"].tolist()
    if features != feature_list:
        raise RuntimeError("RF artifact feature order differs from final_feature_list.csv")

    external_features = pd.read_csv(MODEL_DIR / "external_label_blind_pyradiomics_features.csv")
    locked_external = pd.read_csv(MODEL_DIR / "external_label_blind_predictions.csv")

    merged = external_features.merge(
        locked_external,
        on="neutral_patient_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_feature", "_locked"),
    )
    if len(merged) != len(external_features) or len(merged) != len(locked_external):
        raise RuntimeError("External feature/prediction patient matching is not one-to-one")

    predicted = pipeline.predict_proba(merged[features])[:, 1]
    locked = merged["probability_COP"].to_numpy(dtype=float)
    abs_diff = np.abs(predicted - locked)

    payload = {
        "status": "passed" if float(abs_diff.max()) <= 1e-12 else "failed",
        "n_external": int(len(merged)),
        "feature_count": int(len(features)),
        "features": features,
        "max_abs_probability_diff": float(abs_diff.max()),
        "mean_abs_probability_diff": float(abs_diff.mean()),
        "allclose_at_1e_12": bool(np.allclose(predicted, locked, atol=1e-12, rtol=0.0)),
    }
    print(json.dumps(payload, indent=2))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
