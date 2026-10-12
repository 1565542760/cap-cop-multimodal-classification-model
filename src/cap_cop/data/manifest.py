from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"public_subject_id", "cohort", "diagnosis_label"}


def read_manifest(path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    if manifest["public_subject_id"].isna().any():
        raise ValueError("Manifest contains empty public_subject_id values.")
    if manifest["public_subject_id"].duplicated().any():
        raise ValueError("Manifest contains duplicated public_subject_id values.")
    return manifest


def id_set_digest(ids: list[str]) -> str:
    payload = "\n".join(sorted(set(map(str, ids)))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def cohort_manifest_paths(root: Path) -> list[Path]:
    return [
        root / "data" / "internal" / "CAP" / "manifest.csv",
        root / "data" / "internal" / "COP" / "manifest.csv",
        root / "data" / "external" / "CAP" / "manifest.csv",
        root / "data" / "external" / "COP" / "manifest.csv",
    ]


def read_all_manifests(root: Path) -> pd.DataFrame:
    frames = []
    for path in cohort_manifest_paths(root):
        frame = read_manifest(path)
        frame["manifest_relative_path"] = str(path.relative_to(root)).replace("\\", "/")
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    if merged["public_subject_id"].duplicated().any():
        duplicated = sorted(merged.loc[merged["public_subject_id"].duplicated(), "public_subject_id"].unique())
        raise ValueError(f"Duplicated public_subject_id across manifests: {duplicated}")
    return merged
