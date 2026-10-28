"""Record a clean source revision or verify an exported code-only archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], stderr=subprocess.PIPE)


def safe_relative(name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not name or "\\" in name:
        raise ValueError("Source manifest contains an unsafe path")
    if name.startswith("data/") or name.startswith("outputs/") and name != "outputs/.gitkeep":
        raise ValueError("Patient-level data or runtime output cannot enter a source manifest")
    if name.startswith("models/") and name != "models/README.md":
        raise ValueError("Model artifacts cannot enter a source manifest")
    return relative


def source_entries(root: Path, names: list[str]) -> list[dict[str, str]]:
    entries = []
    for name in sorted(names):
        relative = safe_relative(name)
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Tracked source is missing or linked: {name}")
        entries.append({"path": name, "sha256": sha256(path)})
    return entries


def write_manifest(output: Path) -> dict:
    if git("status", "--porcelain=v1", "--untracked-files=normal").strip():
        raise RuntimeError("Source worktree is not clean; commit reviewed code before recording a release SHA")
    names = [item.decode("utf-8") for item in git("ls-files", "-z").split(b"\0") if item]
    for name in names:
        safe_relative(name)
    manifest = {
        "schema_version": 1,
        "git_commit_sha": git("rev-parse", "HEAD").decode("ascii").strip(),
        "clean_worktree": True,
        "files": source_entries(ROOT, names),
    }
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def verify_manifest(input_path: Path) -> dict:
    manifest = json.loads(input_path.read_text(encoding="utf-8"))
    entries = manifest.get("files")
    schema = manifest.get("schema_version")
    commit = manifest.get("base_git_commit_sha") if schema == 2 else manifest.get("git_commit_sha")
    if schema not in (1, 2, 3) or not isinstance(entries, list) or not re.fullmatch(
        r"[0-9a-fA-F]{40}", str(commit or "")
    ):
        raise ValueError("Invalid source manifest")
    if schema == 2 and manifest.get("release_status") != "Uncommitted release candidate":
        raise ValueError("Candidate manifest must disclose its uncommitted status")
    if schema == 3 and manifest.get("release_status") != "Committed source snapshot":
        raise ValueError("Committed archive must disclose its source status")
    names = [entry["path"] for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate source paths")
    actual = source_entries(ROOT, names)
    if actual != sorted(entries, key=lambda item: item["path"]):
        raise ValueError("Code archive differs from the recorded source manifest")
    return {"base_git_commit_sha": commit, "files_verified": len(actual),
            "release_status": manifest.get("release_status", "Clean committed source"),
            "status": "PASS_LISTED_FILES"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", type=Path)
    action.add_argument("--verify", type=Path)
    args = parser.parse_args()
    try:
        result = write_manifest(args.write) if args.write else verify_manifest(args.verify)
    except (FileExistsError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"Source provenance failed: {error}\n")
    print(json.dumps(result if args.verify else {
        "git_commit_sha": result["git_commit_sha"], "files_recorded": len(result["files"]), "status": "PASS"
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
