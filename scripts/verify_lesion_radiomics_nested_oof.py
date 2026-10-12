"""Verify a public nested-OOF replay against the locked public reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY = (
    PROJECT_ROOT
    / "outputs"
    / "a_nested_oof_reproduction_20261011_v2"
    / "05_nested_cv"
    / "lesion_radiomics_replay_INTERNAL375_STRICT_OOF_PREDICTIONS.csv"
)
DEFAULT_REFERENCE = PROJECT_ROOT / "models" / "lesion_radiomics" / "internal_oof_predictions.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "a_nested_oof_reproduction_20261011_v2" / "nested_oof_verification.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify replayed Internal375 strict OOF predictions.")
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args()

    replay_path = args.replay.resolve()
    reference_path = args.reference.resolve()
    replay = pd.read_csv(replay_path, encoding="utf-8-sig")
    reference = pd.read_csv(reference_path, encoding="utf-8-sig")
    required = {"neutral_patient_id", "outer_fold", "probability_COP", "label_binary"}
    if not required.issubset(replay.columns) or not required.issubset(reference.columns):
        raise RuntimeError("Replay and reference must contain the locked OOF columns")

    joined = replay.merge(
        reference[list(required)],
        on="neutral_patient_id",
        suffixes=("_replay", "_reference"),
        validate="one_to_one",
        how="outer",
        indicator=True,
    )
    if len(joined) != 375 or not joined["_merge"].eq("both").all():
        raise RuntimeError("Replay/reference ID sets are not exactly equal at 375 rows")
    if not np.array_equal(joined["label_binary_replay"], joined["label_binary_reference"]):
        raise RuntimeError("Replay/reference labels differ")
    if not np.array_equal(joined["outer_fold_replay"], joined["outer_fold_reference"]):
        raise RuntimeError("Replay/reference outer-fold assignments differ")

    delta = np.abs(joined["probability_COP_replay"] - joined["probability_COP_reference"])
    replay_auc = float(roc_auc_score(replay["label_binary"], replay["probability_COP"]))
    reference_auc = float(roc_auc_score(reference["label_binary"], reference["probability_COP"]))
    summary = {
        "status": "PASS" if float(delta.max()) <= args.atol else "FAIL",
        "rows": int(len(joined)),
        "id_set_equal": True,
        "labels_equal": True,
        "outer_folds_equal": True,
        "max_abs_probability_delta": float(delta.max()),
        "mean_abs_probability_delta": float(delta.mean()),
        "atol": args.atol,
        "replay_auroc": replay_auc,
        "reference_auroc": reference_auc,
        "replay_sha256": sha256_file(replay_path),
        "reference_sha256": sha256_file(reference_path),
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
