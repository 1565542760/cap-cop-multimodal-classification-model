"""Build an explicit code-only candidate from Git HEAD plus reviewed overlays."""

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


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_BASELINE = {"configs/lesion_phenotype_geometry_exceptions.json"}
OVERLAYS = {
    ".gitignore", "README.md", "README_FOR_COLLABORATOR.md", "pyproject.toml",
    "requirements-radiomics-replay-tested.txt", "configs/artifact_selectors.json",
    "scripts/check_authorized_artifact_bundle.py", "scripts/check_github_release.py",
    "scripts/run_full_cohort_lesion_phenotype.py", "scripts/run_full_cohort_multimodal_replay.py",
    "scripts/run_full_cohort_whole_chest_representations.py",
    "scripts/run_stable_multirepresentation_subset_replay.py",
    "scripts/run_segmentation_smoke.py", "scripts/run_single_case_lesion_radiomics_chain.py",
    "scripts/run_minimal_reproduction.ps1", "scripts/source_archive_provenance.py",
    "scripts/build_code_release_candidate.py", "scripts/check_runtime_environment.py",
    "scripts/audit_external_stable_provenance.py", "scripts/audit_stable_representation_inputs.py",
    "scripts/export_stable_cohort_metadata.py", "scripts/prepare_stable_multirepresentation_run.py",
    "scripts/run_full_cohort_stable_multirepresentation.py",
    "scripts/run_supplementary_exploratory_subset_replay.py",
    "src/cap_cop/data/model_contracts.py", "src/cap_cop/imaging/radiomics/mask_contract.py",
    "src/cap_cop/imaging/radiomics/lesion_phenotype.py",
    "src/cap_cop/pipelines/lesion_radiomics_replay_pipeline.py",
    "src/cap_cop/pipelines/stable_multirepresentation_replay_pipeline.py",
    "src/cap_cop/private_lock_selectors.py",
    "tests/test_external_stable_provenance.py", "tests/test_lesion_phenotype_mask_contract.py",
    "tests/test_model_c_identity.py", "tests/test_release_contracts.py",
    "tests/test_source_archive_provenance.py",
}
TEXT_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".toml", ".md", ".ps1", ".txt", ".csv"}
PRIVATE_PATTERNS = (
    re.compile(r"(?<![A-Z])(?:CAP|COP)_\d{4}(?!\d)"),
    re.compile(r"[A-Za-z]:[\\/](?:Users|wechatdownload|3D_CT)(?:[\\/]|$)", re.I),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
)


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], stderr=subprocess.PIPE)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_source(name: str, content: bytes) -> None:
    if name.startswith(("data/", "outputs/")) and name != "outputs/.gitkeep":
        raise ValueError(f"Restricted input/output in source candidate: {name}")
    if name.startswith("models/") and name != "models/README.md":
        raise ValueError(f"Model artefact in source candidate: {name}")
    if Path(name).suffix.lower() in TEXT_SUFFIXES:
        text = content.decode("utf-8-sig")
        for pattern in PRIVATE_PATTERNS:
            if name == "tests/test_smoke.py" and "3D_CT" in pattern.pattern:
                continue  # Negative path-isolation assertion, not a runtime path.
            if pattern.search(text):
                raise ValueError(f"Privacy/path pattern found in candidate file: {name}")


def build(output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Candidate output already exists: {output_dir}")
    if output_dir.is_relative_to(ROOT):
        raise ValueError("Candidate output must be outside the development worktree")
    tracked = {item.decode("utf-8") for item in git("ls-files", "-z").split(b"\0") if item}
    names = (tracked - EXCLUDED_BASELINE) | OVERLAYS
    if not OVERLAYS.issubset(names):
        raise AssertionError("Overlay closure is incomplete")
    contents: dict[str, bytes] = {}
    overlays = []
    for name in sorted(names):
        if name in OVERLAYS:
            source = ROOT / name
            if not source.is_file() or source.is_symlink():
                raise FileNotFoundError(source)
            content = source.read_bytes()
            baseline = git("show", f"HEAD:{name}") if name in tracked else None
            if baseline != content:
                overlays.append({"path": name, "base_sha256": digest(baseline) if baseline is not None else None,
                                 "candidate_sha256": digest(content)})
        else:
            content = git("show", f"HEAD:{name}")
        check_source(name, content)
        contents[name] = content

    output_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 2,
        "release_status": "Uncommitted release candidate",
        "base_git_commit_sha": git("rev-parse", "HEAD").decode("ascii").strip(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_builder": sys.version.split()[0],
        "reviewed_overlays": overlays,
        "excluded_baseline_paths": sorted(EXCLUDED_BASELINE),
        "files": [{"path": name, "sha256": digest(content)} for name, content in contents.items()],
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    contents["SOURCE_MANIFEST.json"] = manifest_bytes
    archive = output_dir / "CAP_COP_REPRODUCIBILITY_RC_20261027.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name, content in sorted(contents.items()):
            bundle.writestr(name, content)
    result = {"zip": str(archive), "zip_sha256": digest(archive.read_bytes()),
              "base_git_commit_sha": manifest["base_git_commit_sha"],
              "release_status": manifest["release_status"], "files": len(contents),
              "reviewed_overlays": len(overlays)}
    (output_dir / "ZIP_SHA256.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
