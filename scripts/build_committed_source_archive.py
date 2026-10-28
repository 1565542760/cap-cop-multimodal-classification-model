"""Build a code-only ZIP from committed Git blobs, never the working tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from build_code_release_candidate import check_source


ROOT = Path(__file__).resolve().parents[1]


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], stderr=subprocess.PIPE)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(output_dir: Path, version: str) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if output_dir.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Archive output must be outside the source worktree")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("Expected a semantic version such as 0.2.0")
    if git("status", "--porcelain=v1", "--untracked-files=normal").strip():
        raise RuntimeError("Commit and clean the release worktree before building")
    commit = git("rev-parse", "HEAD").decode("ascii").strip()
    names = sorted(item.decode("utf-8") for item in git("ls-files", "-z").split(b"\0") if item)
    contents: dict[str, bytes] = {}
    for name in names:
        content = git("show", f"{commit}:{name}")
        check_source(name, content)
        contents[name] = content
    manifest = {
        "schema_version": 3,
        "release_status": "Committed source snapshot",
        "git_commit_sha": commit,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder_python": sys.version.split()[0],
        "files": [{"path": name, "sha256": digest(contents[name])} for name in names],
    }
    contents["SOURCE_MANIFEST.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output_dir.mkdir(parents=True)
    archive = output_dir / f"CAP_COP_SOURCE_v{version}.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name, content in sorted(contents.items()):
            bundle.writestr(name, content)
    result = {
        "git_commit_sha": commit,
        "archive": str(archive),
        "archive_sha256": digest(archive.read_bytes()),
        "source_files": len(names),
        "manifest_sha256": digest(contents["SOURCE_MANIFEST.json"]),
    }
    (output_dir / "ZIP_SHA256.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir, args.version), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
