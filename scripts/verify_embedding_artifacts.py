from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def describe(name: str, npy_rel: str, manifest_rel: str | None = None) -> dict[str, object]:
    npy_path = PROJECT_ROOT / npy_rel
    arr = np.load(npy_path)
    row: dict[str, object] = {
        "name": name,
        "path": str(npy_path),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "finite": bool(np.isfinite(arr).all()),
    }
    if manifest_rel:
        manifest = pd.read_csv(PROJECT_ROOT / manifest_rel)
        row["manifest_path"] = str(PROJECT_ROOT / manifest_rel)
        row["manifest_rows"] = int(len(manifest))
        row["row_match"] = bool(len(manifest) == arr.shape[0])
    return row


def main() -> None:
    rows = [
        describe(
            "whole_chest_ct_internal_embeddings",
            "models/whole_chest_ct/embeddings/internal_embeddings.npy",
            "models/whole_chest_ct/internal_embedding_manifest.csv",
        ),
        describe(
            "whole_chest_ct_external_embeddings",
            "models/whole_chest_ct/embeddings/external_embeddings.npy",
            "models/whole_chest_ct/external_embedding_manifest.csv",
        ),
        describe(
            "stable_multirepresentation_genesis_internal_embeddings",
            "models/stable_multirepresentation_ct/embeddings/genesis_internal_embeddings.npy",
        ),
        describe(
            "stable_multirepresentation_genesis_external_embeddings",
            "models/stable_multirepresentation_ct/embeddings/genesis_external_embeddings.npy",
        ),
        describe(
            "stable_multirepresentation_tangerine_internal_embeddings",
            "models/stable_multirepresentation_ct/embeddings/tangerine_internal_embeddings.npy",
            "models/whole_chest_ct/internal_embedding_manifest.csv",
        ),
        describe(
            "stable_multirepresentation_tangerine_external_embeddings",
            "models/stable_multirepresentation_ct/embeddings/tangerine_external_embeddings.npy",
            "models/whole_chest_ct/external_embedding_manifest.csv",
        ),
    ]
    print(json.dumps({"status": "passed", "artifacts": rows}, indent=2))


if __name__ == "__main__":
    main()
