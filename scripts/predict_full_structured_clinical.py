from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cap_cop.clinical.full_structured_model import load_artifact, predict_proba


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    artifact = load_artifact(args.model)
    probability = predict_proba(artifact, frame)
    threshold = float(artifact["threshold"])
    if "patient_id" in frame.columns:
        patient_id = frame["patient_id"].astype(str)
    elif "case_id_public" in frame.columns:
        patient_id = frame["case_id_public"].astype(str)
    else:
        patient_id = pd.Series(np.arange(len(frame)), name="row_index").astype(str)
    result = pd.DataFrame(
        {
            "patient_id": patient_id,
            "clinical_probability": probability,
            "clinical_logit_margin": np.log(
                np.clip(probability, 1e-7, 1 - 1e-7)
                / np.clip(1 - probability, 1e-7, 1 - 1e-7)
            ),
            "clinical_model": artifact["model_name"],
            "clinical_threshold": threshold,
            "clinical_prediction": (probability >= threshold).astype(int),
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"WROTE={output}")


if __name__ == "__main__":
    main()
