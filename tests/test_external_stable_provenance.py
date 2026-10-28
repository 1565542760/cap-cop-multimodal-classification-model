"""Synthetic tests for the external imaging provenance audit."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_external_stable_provenance import collection_sha256, require_same, require_unique, sha256_file


def test_collection_fingerprint_is_content_bound(tmp_path):
    source = tmp_path / "slice.dcm"
    source.write_bytes(b"synthetic dicom bytes")
    first = collection_sha256([source], tmp_path)
    assert len(first) == 64
    assert sha256_file(source) != first
    source.write_bytes(b"changed synthetic bytes")
    assert collection_sha256([source], tmp_path) != first


def test_identity_audit_rejects_duplicate_and_swapped_sources():
    with pytest.raises(ValueError, match="duplicate ID"):
        require_unique(pd.DataFrame({"public_subject_id": ["SYN_1", "SYN_1"]}),
                       "public_subject_id", 2, "synthetic")
    with pytest.raises(ValueError, match="provenance mismatch"):
        require_same(pd.Series(["SYN_A", "SYN_B"]), pd.Series(["SYN_B", "SYN_A"]),
                     "synthetic source")
