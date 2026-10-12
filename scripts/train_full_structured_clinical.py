from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cap_cop.clinical.full_structured_model import (
    DEFAULT_SEEDS,
    FEATURES,
    fit_ensemble,
    save_artifact,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    internal_path = Path(args.internal)
    output_dir = Path(args.output_dir)
    internal = pd.read_csv(internal_path)
    required = set(FEATURES) | {"label_binary"}
    missing = sorted(required - set(internal.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    labels = internal["label_binary"].astype(int).to_numpy()
    if len(internal) != 375 or int(labels.sum()) != 125:
        raise ValueError("Expected internal cohort of 375 rows with 125 COP labels.")

    artifact = fit_ensemble(internal, labels, DEFAULT_SEEDS)
    artifact_path = output_dir / "full_structured_clinical_ensemble.joblib"
    save_artifact(artifact, artifact_path)
    manifest = {
        "model_name": artifact["model_name"],
        "features": FEATURES,
        "positive_class": "COP",
        "negative_class": "CAP",
        "locked_threshold": artifact["threshold"],
        "training_rows": int(len(internal)),
        "training_CAP": int((labels == 0).sum()),
        "training_COP": int((labels == 1).sum()),
        "seeds": DEFAULT_SEEDS,
        "source_internal_sha256": sha256_file(internal_path),
        "artifact_file": artifact_path.name,
        "artifact_type": "full-internal deployment fit; no external labels used",
        "research_evaluation_note": (
            "The frozen v34 external evaluation used predefined unlabeled "
            "external-missingness stress masks during training. This artifact "
            "is the deployable full-internal fit and is not a replacement for "
            "the locked research evaluation outputs."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
