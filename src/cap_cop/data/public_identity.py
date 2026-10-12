from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


REQUIRED_CROSSWALK_COLUMNS = {
    "public_subject_id",
    "cohort",
    "diagnosis_label",
    "label_binary",
    "clinical_model_id",
    "lesion_imaging_model_id",
    "whole_chest_imaging_model_id",
}


def canonical_id_set_sha256(values: pd.Series | list[str]) -> str:
    payload = "\n".join(sorted({str(value).strip() for value in values})).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def read_public_crosswalk(
    path: str | Path,
    expected_counts: dict[tuple[str, str], int] | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    missing = REQUIRED_CROSSWALK_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Public crosswalk is missing columns: {sorted(missing)}")
    if frame["public_subject_id"].eq("").any():
        raise ValueError("Public crosswalk contains an empty public_subject_id.")
    if frame["public_subject_id"].duplicated().any():
        duplicate = sorted(frame.loc[frame["public_subject_id"].duplicated(), "public_subject_id"].unique())
        raise ValueError(f"Public crosswalk contains duplicate IDs: {duplicate}")
    expected = expected_counts or {
        ("internal", "CAP"): 250,
        ("internal", "COP"): 125,
        ("external", "CAP"): 19,
        ("external", "COP"): 14,
    }
    observed = frame.groupby(["cohort", "diagnosis_label"]).size().to_dict()
    if observed != expected:
        raise ValueError(f"Unexpected cohort counts: {observed}; expected {expected}")
    return frame
