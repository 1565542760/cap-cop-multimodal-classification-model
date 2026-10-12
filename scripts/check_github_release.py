"""Run the read-only GitHub code-package release gate.

This gate checks package hygiene and executable public interfaces. It does not
claim that restricted data are publishable or that every historical paper lock
is scientifically accepted.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cap_cop.config import load_model_configs
from cap_cop.models.unimodal.clinical.alternate_clinical_configuration import (
    FEATURES as ALTERNATE_FEATURES,
)
from cap_cop.models.unimodal.clinical.availability_gated_clinical import (
    AVAILABILITY_THRESHOLD,
    FULL_ONLY_FEATURES,
)
from cap_cop.models.unimodal.clinical.core_clinical import CORE_FEATURES
from cap_cop.models.unimodal.clinical.structural_missingness_clinical import (
    ENSEMBLE_MEMBER_COUNT,
    FEATURES as STRUCTURAL_FEATURES,
    INDICATOR_FEATURE_COUNT,
)


RUNTIME_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".toml"}
RUNTIME_ROOTS = (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts", PROJECT_ROOT / "configs")
PUBLIC_TEXT_ROOTS = (
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "docs",
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "configs",
)
_drive = "D:"
_separator = "\\\\"
_experiment_root = "3D_CT"
FORBIDDEN_RUNTIME_PATH = re.compile(
    re.escape(_drive + _separator + _experiment_root)
    + "|"
    + re.escape(_drive + "/" + _experiment_root),
    re.IGNORECASE,
)
FORBIDDEN_PUBLIC_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/]3D_CT(?:[\\/]|$)|"
    r"[A-Za-z]:[\\/]CAP_COP(?:[\\/]|$)|"
    r"[A-Za-z]:[\\/]wechatdownload(?:[\\/]|$)|"
    r"[A-Za-z]:[\\/]Users[\\/])",
    re.IGNORECASE,
)
REQUIRED_DOCS = (
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "docs" / "github_release_gate_20261011.md",
    PROJECT_ROOT / "docs" / "replay_acceptance_20261011.md",
)
INTERNAL_IDENTIFIER_IN_NAME = re.compile(
    r"(?i)(?:^|[/\\])(?:C0[1-5]|I(?:08|14|21|23)|V3[._-](?:5|6|8|13)|SP[67])(?:[/\\]|[._-]|$)"
)
MODEL_ROOT = PROJECT_ROOT / "models"
MINIMAL_ENTRYPOINT = PROJECT_ROOT / "scripts" / "run_minimal_reproduction.ps1"
ARTIFACT_PREFLIGHT = PROJECT_ROOT / "scripts" / "check_authorized_artifact_bundle.py"


def check_runtime_paths() -> list[str]:
    failures: list[str] = []
    for root in RUNTIME_ROOTS:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in RUNTIME_SUFFIXES:
                text = path.read_text(encoding="utf-8", errors="ignore")
                if FORBIDDEN_RUNTIME_PATH.search(text):
                    failures.append(f"runtime path literal: {path.relative_to(PROJECT_ROOT)}")
    return failures


def check_public_path_hygiene() -> list[str]:
    """Reject host-specific paths in files that can be committed to GitHub."""
    failures: list[str] = []
    for root in PUBLIC_TEXT_ROOTS:
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in RUNTIME_SUFFIXES | {".md", ".csv"}:
                continue
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            if git_ignored(relative):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if FORBIDDEN_PUBLIC_PATH.search(text):
                failures.append(f"public path literal: {relative}")
    return failures


def git_ignored(relative_path: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "check-ignore", "-q", "--", relative_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def check_model_boundary() -> list[str]:
    """Ensure governed model artifacts cannot enter the public candidate set."""
    failures: list[str] = []
    readme = MODEL_ROOT / "README.md"
    if not readme.is_file():
        failures.append("missing models/README.md governed-artifact boundary")
    for path in MODEL_ROOT.rglob("*"):
        if not path.is_file() or path == readme:
            continue
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if not git_ignored(relative):
            failures.append(f"model artifact is not Git-ignored: {relative}")
    return failures


def check_public_semantic_names() -> list[str]:
    """Reject internal experiment identifiers in public file and model names."""
    failures: list[str] = []
    roots = (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts", PROJECT_ROOT / "configs")
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            if INTERNAL_IDENTIFIER_IN_NAME.search(relative):
                failures.append(f"internal identifier in public file name: {relative}")
    model_config = PROJECT_ROOT / "configs" / "models.json"
    if model_config.is_file():
        payload = json.loads(model_config.read_text(encoding="utf-8"))
        for item in payload.get("models", []):
            name = str(item.get("name", ""))
            if re.search(r"(?i)(?:C0[1-5]|I(?:08|14|21|23)|V3[._-]|SP[67])", name):
                failures.append(f"internal identifier in model config name: {name}")
    return failures


def main() -> int:
    failures: list[str] = []
    configs = load_model_configs()
    if len(configs) != 9:
        failures.append(f"expected 9 semantic model configs, found {len(configs)}")
    if any("blocked" in item.status.lower() for item in configs):
        failures.append("a model config is marked blocked")
    failures.extend(check_runtime_paths())
    failures.extend(check_public_path_hygiene())
    failures.extend(check_model_boundary())
    failures.extend(check_public_semantic_names())
    failures.extend(
        f"missing required document: {path.relative_to(PROJECT_ROOT)}"
        for path in REQUIRED_DOCS
        if not path.is_file()
    )

    if not MINIMAL_ENTRYPOINT.is_file():
        failures.append("missing minimal reproduction entrypoint")
    else:
        minimal_text = MINIMAL_ENTRYPOINT.read_text(encoding="utf-8", errors="ignore")
        if ".venv_py39_radiomics" in minimal_text or ".venv\\Scripts\\python.exe" in minimal_text:
            failures.append("minimal reproduction entrypoint references an obsolete environment")
        if "Invoke-CheckedPython" not in minimal_text:
            failures.append("minimal reproduction entrypoint does not check native Python exit codes")

    if not ARTIFACT_PREFLIGHT.is_file():
        failures.append("missing authorized artifact bundle preflight")

    fusion_text = (PROJECT_ROOT / "scripts" / "run_full_cohort_multimodal_replay.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    for paper_name in (
        "paper_model_a_primary_lesion_radiomics",
        "paper_model_b_whole_chest_ct_comparator",
        "paper_model_c_stable_multirepresentation_ct",
        "paper_model_d_supplementary_exploratory_comparator",
    ):
        if paper_name not in fusion_text:
            failures.append(f"missing paper-facing fusion alias: {paper_name}")

    external_masks = [
        path for path in (PROJECT_ROOT / "data" / "external").rglob("masks") if path.is_dir()
    ]
    if external_masks:
        failures.extend(f"external mask directory present: {path.relative_to(PROJECT_ROOT)}" for path in external_masks)

    if len(CORE_FEATURES) != 7:
        failures.append("core clinical interface does not expose 7 features")
    if len(FULL_ONLY_FEATURES) != 9 or AVAILABILITY_THRESHOLD != 6:
        failures.append("availability-gated contract mismatch")
    if len(STRUCTURAL_FEATURES) != 16 or INDICATOR_FEATURE_COUNT != 32 or ENSEMBLE_MEMBER_COUNT != 15:
        failures.append("structural-missingness exploratory contract mismatch")
    if len(ALTERNATE_FEATURES) != 16:
        failures.append("alternate clinical lock does not expose 16 features")

    # Check representative files inside ignored directories. Git does not
    # necessarily report a directory itself as ignored when an allowed
    # README or placeholder is intentionally kept inside it.
    ignore_probes = (
        ("data/", "data/__gitignore_probe__.dcm"),
        ("outputs/", "outputs/__gitignore_probe__.csv"),
        (".venv/", ".venv/__gitignore_probe__.py"),
        (
            "models/segmentation/locked_checkpoints/",
            "models/segmentation/locked_checkpoints/__gitignore_probe__.pth",
        ),
    )
    for label, path in ignore_probes:
        if not git_ignored(path):
            failures.append(f"Git ignore missing for {label}")

    result = {
        "code_package_gate": "PASS" if not failures else "FAIL",
        "scientific_result_gate": "REVIEW_REQUIRED",
        "data_publication_gate": "GOVERNED_DATA_AND_MODEL_AUTHORIZATION_REQUIRED",
        "project_root": str(PROJECT_ROOT),
        "model_config_count": len(configs),
        "external_mask_directory_count": len(external_masks),
        "failures": failures,
        "note": "A/C corrected result locks and restricted patient-level inputs remain separate scientific/publication decisions.",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
