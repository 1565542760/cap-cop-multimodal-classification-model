from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "stable_multirepresentation_ct"


def main() -> int:
    internal = pd.read_csv(MODEL_DIR / "internal_oof_predictions.csv")
    external = pd.read_csv(MODEL_DIR / "external_predictions.csv")
    lock = json.loads((MODEL_DIR / "result_lock.json").read_text(encoding="utf-8"))

    sp6_internal = internal[internal["candidate"] == "SP6_STABLE_MULTI_REPRESENTATION_RANDOM_FOREST"].copy()
    sp6_external = external[external["candidate"] == "SP6_STABLE_MULTI_REPRESENTATION_RANDOM_FOREST"].copy()
    sp7_internal = internal[internal["candidate"] == "SP7_STABLE_REPRESENTATION_EQUAL_ENSEMBLE"].copy()
    sp7_external = external[external["candidate"] == "SP7_STABLE_REPRESENTATION_EQUAL_ENSEMBLE"].copy()

    if len(sp6_internal) != 375:
        raise RuntimeError(f"SP6 internal rows {len(sp6_internal)} != 375")
    if len(sp7_internal) != 375:
        raise RuntimeError(f"SP7 internal rows {len(sp7_internal)} != 375")
    if sp6_internal["neutral_imaging_id"].nunique() != 375:
        raise RuntimeError("SP6 internal IDs are not one row per patient")
    if sp7_internal["neutral_imaging_id"].nunique() != 375:
        raise RuntimeError("SP7 internal IDs are not one row per patient")

    ext_summary = {}
    for name, frame in [("SP6", sp6_external), ("SP7", sp7_external)]:
        counts = frame.groupby("neutral_external_id").size()
        ext_summary[name] = {
            "rows": int(len(frame)),
            "unique_ids": int(frame["neutral_external_id"].nunique()),
            "min_rows_per_id": int(counts.min()) if len(counts) else 0,
            "max_rows_per_id": int(counts.max()) if len(counts) else 0,
            "mean_probability_unique_per_id": bool(
                frame.groupby("neutral_external_id")["prob_COP_mean"].nunique().max() == 1
            ),
        }

    report = {
        "model": "stable_multirepresentation_ct",
        "lock_best_observed_external_candidate": lock.get("best_observed_external_candidate"),
        "lock_sp7_members": lock.get("sp7_members"),
        "all_candidates": sorted(internal["candidate"].unique().tolist()),
        "internal_rows_total": int(len(internal)),
        "external_rows_total": int(len(external)),
        "sp6_internal_rows": int(len(sp6_internal)),
        "sp7_internal_rows": int(len(sp7_internal)),
        "sp6_internal_unique_ids": int(sp6_internal["neutral_imaging_id"].nunique()),
        "sp7_internal_unique_ids": int(sp7_internal["neutral_imaging_id"].nunique()),
        "external_candidate_summary": ext_summary,
        "sp6_internal_prob_range": [
            float(np.min(sp6_internal["prob_COP"])),
            float(np.max(sp6_internal["prob_COP"])),
        ],
        "sp7_internal_prob_range": [
            float(np.min(sp7_internal["prob_COP"])),
            float(np.max(sp7_internal["prob_COP"])),
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
