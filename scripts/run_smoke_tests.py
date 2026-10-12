from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd

from cap_cop.config import load_model_configs
from cap_cop.data.manifest import id_set_digest, read_manifest
from cap_cop.data.public_identity import read_public_crosswalk
from cap_cop.evaluation.metrics import binary_metrics
from cap_cop.models.multimodal.fusion import classify, weighted_average_fusion
from cap_cop.models.unimodal.clinical.availability_gated_clinical import (
    AVAILABILITY_THRESHOLD,
    FULL_ONLY_FEATURES,
)
from cap_cop.models.unimodal.clinical.core_clinical import CORE_FEATURES
from cap_cop.models.unimodal.clinical.alternate_clinical_configuration import (
    FEATURES as ALTERNATE_FEATURES,
    STATUS as ALTERNATE_STATUS,
)
from cap_cop.models.unimodal.clinical.structural_missingness_clinical import (
    ENSEMBLE_MEMBER_COUNT,
    FEATURES as STRUCTURAL_FEATURES,
    INDICATOR_FEATURE_COUNT,
    STATUS as STRUCTURAL_STATUS,
)
from cap_cop.paths import project_root
from cap_cop.preprocessing.clinical import full_structured_clinical_preprocessor


def check(condition: bool, name: str) -> None:
    if not condition:
        raise AssertionError(name)


def main() -> None:
    root = project_root()
    checks: list[str] = []

    check((root / "configs" / "models.json").is_file(), "project root")
    checks.append("project root")

    configs = load_model_configs()
    check(len(configs) == 9, "model config count")
    check(any("canonical_identity" in item.status for item in configs), "canonical identity reporting")
    checks.append("model config loading")

    fusion_entrypoint = Path(root, "scripts", "run_full_cohort_multimodal_replay.py").read_text(
        encoding="utf-8"
    )
    for paper_name in (
        "paper_model_a_primary_lesion_radiomics",
        "paper_model_b_whole_chest_ct_comparator",
        "paper_model_c_stable_multirepresentation_ct",
        "paper_model_d_supplementary_exploratory_comparator",
    ):
        check(paper_name in fusion_entrypoint, f"paper-facing fusion alias {paper_name}")
    checks.append("paper-facing fusion aliases")

    crosswalk_path = Path(root, "data", "public_subject_model_crosswalk.csv")
    if crosswalk_path.is_file():
        crosswalk = read_public_crosswalk(crosswalk_path)
        expected_unique_ids = 408
    else:
        crosswalk = read_public_crosswalk(
            Path(root, "tests", "fixtures", "synthetic_crosswalk.csv"),
            expected_counts={
                ("internal", "CAP"): 2,
                ("internal", "COP"): 2,
                ("external", "CAP"): 2,
                ("external", "COP"): 2,
            },
        )
        expected_unique_ids = 8
    check(crosswalk["segmentation_model_id"].nunique() == expected_unique_ids, "segmentation IDs unique")
    check(crosswalk["whole_chest_imaging_model_id"].nunique() == expected_unique_ids, "whole-chest IDs unique")
    checks.append("canonical crosswalk")

    config_text = Path(root, "configs", "models.json").read_text(encoding="utf-8")
    serialised = json.dumps(json.loads(config_text))
    forbidden_prefixes = ("C", "I")
    forbidden_suffixes = {"C": ("01", "02", "03", "04", "05"), "I": ("14", "21", "23")}
    for prefix in forbidden_prefixes:
        for suffix in forbidden_suffixes[prefix]:
            token = prefix + suffix
            check(token not in serialised, f"forbidden public experiment token {token}")
    checks.append("public config naming")

    frame = pd.DataFrame({name: [1] for name in full_structured_clinical_preprocessor().feature_names})
    frame["label"] = [1]
    transformed = full_structured_clinical_preprocessor().transform(frame)
    check(transformed.shape == (1, 16), "clinical feature count")
    check("label" not in transformed.columns, "label excluded from clinical features")
    checks.append("clinical preprocessing")

    check(len(CORE_FEATURES) == 7, "core clinical feature contract")
    check(len(FULL_ONLY_FEATURES) == 9, "availability-gated full-only feature contract")
    check(AVAILABILITY_THRESHOLD == 6, "availability-gated threshold")
    check(len(STRUCTURAL_FEATURES) == 16, "structural-missingness feature contract")
    check(INDICATOR_FEATURE_COUNT == 32, "structural-missingness indicator contract")
    check(ENSEMBLE_MEMBER_COUNT == 15, "structural-missingness ensemble size")
    check(STRUCTURAL_STATUS.startswith("exploratory_locked"), "structural status boundary")
    check(len(ALTERNATE_FEATURES) == 16, "alternate clinical feature contract")
    check(ALTERNATE_STATUS == "locked_strict_family_replay_available", "alternate status")
    checks.append("semantic clinical branch interfaces")

    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "manifest.csv"
        manifest.write_text(
            "public_subject_id,cohort,diagnosis_label\nP002,internal,COP\nP001,internal,CAP\n",
            encoding="utf-8",
        )
        mf = read_manifest(manifest)
        check(mf.shape == (2, 3), "manifest shape")
        check(id_set_digest(mf["public_subject_id"].tolist()) == id_set_digest(["P001", "P002"]), "manifest digest")
    checks.append("manifest validation")

    fused = weighted_average_fusion(np.array([0.1, 0.8]), np.array([0.2, 0.6]), 0.70, 0.30)
    check(np.allclose(fused, np.array([0.13, 0.74])), "fusion result")
    check(classify(fused, 0.50).tolist() == [0, 1], "classification")
    metrics = binary_metrics([0, 1, 1, 0], [0.1, 0.9, 0.6, 0.3])
    check(metrics["auroc"] == 1.0, "metrics")
    checks.append("fusion and metrics")

    print(json.dumps({"status": "passed", "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
