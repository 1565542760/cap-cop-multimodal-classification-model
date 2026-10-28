"""Source provenance stays verifiable after .git is omitted from a ZIP."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "source_archive_provenance.py"
SPEC = importlib.util.spec_from_file_location("source_archive_provenance", MODULE_PATH)
source = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(source)


def test_source_entries_hash_only_named_files(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "module.py").write_bytes(b"safe source")
    actual = source.source_entries(tmp_path, ["src/module.py"])
    assert actual == [{"path": "src/module.py", "sha256": hashlib.sha256(b"safe source").hexdigest()}]


@pytest.mark.parametrize("name", ["../private.csv", "C:/private.csv", "src\\module.py", "", "data/patient.csv", "models/weights.pt"])
def test_manifest_rejects_unsafe_paths(name):
    with pytest.raises(ValueError):
        source.safe_relative(name)


def test_exported_archive_verifies_without_git(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    code = tmp_path / "src" / "module.py"
    code.write_bytes(b"verified source")
    monkeypatch.setattr(source, "ROOT", tmp_path)
    manifest_path = tmp_path / "SOURCE_MANIFEST.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "git_commit_sha": "a" * 40,
        "clean_worktree": True,
        "files": source.source_entries(tmp_path, ["src/module.py"]),
    }), encoding="utf-8")
    assert source.verify_manifest(manifest_path)["status"] == "PASS_LISTED_FILES"
    code.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="differs"):
        source.verify_manifest(manifest_path)


def test_committed_archive_manifest_verifies_without_git(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_bytes(b"committed source")
    monkeypatch.setattr(source, "ROOT", tmp_path)
    manifest_path = tmp_path / "SOURCE_MANIFEST.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 3,
        "git_commit_sha": "b" * 40,
        "release_status": "Committed source snapshot",
        "files": source.source_entries(tmp_path, ["src/module.py"]),
    }), encoding="utf-8")
    assert source.verify_manifest(manifest_path)["status"] == "PASS_LISTED_FILES"
