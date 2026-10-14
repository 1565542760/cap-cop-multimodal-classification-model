from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cap_cop.config import load_model_configs
from cap_cop.data.manifest import cohort_manifest_paths, id_set_digest, read_manifest
from cap_cop.data import release_inputs
from cap_cop.data.public_identity import read_public_crosswalk
from cap_cop.evaluation.metrics import binary_metrics
from cap_cop.models.multimodal.fusion import classify, weighted_average_fusion
from cap_cop.models.unimodal.clinical.availability_gated_clinical import (
    AVAILABILITY_THRESHOLD,
    FULL_ONLY_FEATURES,
)
from cap_cop.models.unimodal.clinical.alternate_clinical_configuration import (
    FEATURES as ALTERNATE_FEATURES,
)
from cap_cop.models.unimodal.clinical.core_clinical import CORE_FEATURES
from cap_cop.models.unimodal.clinical.structural_missingness_clinical import (
    ENSEMBLE_MEMBER_COUNT,
    FEATURES as STRUCTURAL_FEATURES,
    INDICATOR_FEATURE_COUNT,
)
from cap_cop.paths import project_root
from cap_cop.preprocessing.clinical import full_structured_clinical_preprocessor


def test_project_root_is_cap_cop():
    assert Path(project_root(), "configs", "models.json").is_file()


def test_model_configs_load():
    configs = load_model_configs()
    assert len(configs) == 9
    assert configs[0].clinical_weight == 0.70
    assert configs[0].imaging_weight == 0.30
    assert any("canonical_identity" in item.status for item in configs)


def test_paper_facing_fusion_aliases_are_present():
    source = Path(project_root(), "scripts", "run_full_cohort_multimodal_replay.py").read_text(
        encoding="utf-8"
    )
    for name in (
        "paper_model_a_primary_lesion_radiomics",
        "paper_model_b_whole_chest_ct_comparator",
        "paper_model_c_stable_multirepresentation_ct",
        "paper_model_d_supplementary_exploratory_comparator",
    ):
        assert name in source


def test_public_crosswalk_is_canonical_and_branch_specific():
    data_path = Path(project_root(), "data", "public_subject_model_crosswalk.csv")
    if data_path.is_file():
        crosswalk = read_public_crosswalk(data_path)
        expected_unique_ids = 408
    else:
        crosswalk = read_public_crosswalk(
            Path(project_root(), "tests", "fixtures", "synthetic_crosswalk.csv"),
            expected_counts={
                ("internal", "CAP"): 2,
                ("internal", "COP"): 2,
                ("external", "CAP"): 2,
                ("external", "COP"): 2,
            },
        )
        expected_unique_ids = 8
    assert crosswalk["segmentation_model_id"].nunique() == expected_unique_ids
    assert crosswalk["clinical_model_id"].nunique() == expected_unique_ids
    assert crosswalk["whole_chest_imaging_model_id"].nunique() == expected_unique_ids
    internal = crosswalk[crosswalk["cohort"].eq("internal")]
    assert internal["segmentation_model_id"].notna().all()
    assert internal["segmentation_model_id"].ne("").all()
    assert internal["lesion_imaging_model_id"].ne(internal["whole_chest_imaging_model_id"]).any()


def test_multimodal_replay_has_no_historical_identity_mode():
    source = Path(project_root(), "scripts", "run_full_cohort_multimodal_replay.py").read_text(encoding="utf-8")
    assert "historical_lock" not in source
    assert "HISTORICAL_LESION_CROSSWALK" not in source
    assert "segmentation_model_id" in source


def test_runtime_config_paths_are_project_relative():
    config = json.loads(
        Path(project_root(), "configs", "clinical", "availability_gated_clinical.json").read_text(
            encoding="utf-8"
        )
    )
    assert not Path(config["source_code"]).is_absolute()
    assert not Path(config["outer_split_contract"]).is_absolute()


def test_no_experiment_ids_in_public_config_keys():
    text = Path(project_root(), "configs", "models.json").read_text(encoding="utf-8")
    payload = json.loads(text)
    forbidden = [prefix + suffix for prefix, suffixes in {"C": ("01", "02", "03", "04", "05"), "I": ("14", "21", "23")}.items() for suffix in suffixes]
    serialised = json.dumps(payload)
    for token in forbidden:
        assert token not in serialised


def test_clinical_preprocessor_uses_sixteen_features():
    frame = pd.DataFrame(
        {
            "age": [60],
            "height_cm": [170],
            "weight_kg": [70],
            "body_mass_index": [24.2],
            "fever": [1],
            "cough": [1],
            "sputum_category": [0],
            "chest_tightness": [0],
            "dyspnoea": [0],
            "haemoptysis": [0],
            "weight_loss": [0],
            "lymphocyte_count": [1.2],
            "erythrocyte_sedimentation_rate": [30],
            "c_reactive_protein": [20],
            "high_sensitivity_c_reactive_protein": [18],
            "procalcitonin": [0.05],
            "label": [1],
        }
    )
    transformed = full_structured_clinical_preprocessor().transform(frame)
    assert transformed.shape == (1, 16)
    assert "label" not in transformed.columns


def test_semantic_clinical_branch_interfaces_are_not_placeholders():
    assert len(CORE_FEATURES) == 7
    assert len(FULL_ONLY_FEATURES) == 9
    assert AVAILABILITY_THRESHOLD == 6
    assert len(STRUCTURAL_FEATURES) == 16
    assert INDICATOR_FEATURE_COUNT == 32
    assert ENSEMBLE_MEMBER_COUNT == 15
    assert len(ALTERNATE_FEATURES) == 16


def test_manifest_validation_and_digest(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "public_subject_id,cohort,diagnosis_label\n"
        "P002,internal,COP\n"
        "P001,internal,CAP\n",
        encoding="utf-8",
    )
    frame = read_manifest(manifest)
    assert frame.shape == (2, 3)
    assert id_set_digest(frame["public_subject_id"].tolist()) == id_set_digest(["P001", "P002"])


def test_runtime_manifest_paths_use_only_deidentified_data(tmp_path):
    paths = cohort_manifest_paths(tmp_path)
    assert len(paths) == 4
    assert all("deidentified_release_candidate" in path.parts for path in paths)


def test_runtime_entrypoints_do_not_reference_source_cohort_trees():
    root = project_root()
    entrypoints = [
        "scripts/run_full_cohort_clinical_replay.py",
        "scripts/run_clinical_input_availability_sensitivity.py",
        "scripts/run_reproducibility_smoke.py",
        "scripts/run_segmentation_smoke.py",
        "scripts/run_single_case_lesion_radiomics_chain.py",
        "scripts/run_supplementary_exploratory_subset_replay.py",
        "src/cap_cop/imaging/whole_chest_tensor/generate_whole_chest_tensor.py",
    ]
    for name in entrypoints:
        source = (root / name).read_text(encoding="utf-8").replace("\\", "/")
        assert '"data" / "internal"' not in source, name
        assert '"data" / "external"' not in source, name
        assert "D:/3D_CT" not in source, name


def test_release_clinical_contract_and_legacy_alias(tmp_path, monkeypatch):
    monkeypatch.setattr(release_inputs, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(
        release_inputs,
        "EXPECTED_COUNTS",
        {(cohort, diagnosis): 2 for cohort in ("internal", "external") for diagnosis in ("CAP", "COP")},
    )
    for cohort in ("internal", "external"):
        for diagnosis, label in (("CAP", 0), ("COP", 1)):
            root = tmp_path / cohort / diagnosis
            (root / "clinical").mkdir(parents=True)
            ids = [f"{cohort}_{diagnosis}_{index}" for index in (1, 2)]
            frame = pd.DataFrame({name: [1, 2] for name in release_inputs.clinical_features()})
            frame.insert(0, "public_subject_id", ids)
            frame["label_binary"] = label
            frame.to_csv(root / "clinical" / "clinical_features.csv", index=False)
            frame[["public_subject_id", "label_binary"]].to_csv(root / "manifest.csv", index=False)
    internal = release_inputs.load_clinical_cohort("internal", legacy_id_alias=True)
    external = release_inputs.load_clinical_cohort("external")
    assert len(internal) == len(external) == 4
    assert internal["case_id_public"].equals(internal["public_subject_id"])
    assert "case_id_public" not in external.columns
    assert internal["label_binary"].tolist() == [0, 0, 1, 1]
    assert release_inputs.dicom_dir("internal", "CAP", "internal_CAP_1").is_relative_to(tmp_path)


def test_fusion_and_metrics_interfaces():
    fused = weighted_average_fusion(np.array([0.1, 0.8]), np.array([0.2, 0.6]), 0.70, 0.30)
    assert np.allclose(fused, np.array([0.13, 0.74]))
    assert classify(fused, 0.50).tolist() == [0, 1]
    scores = binary_metrics([0, 1, 1, 0], [0.1, 0.9, 0.6, 0.3])
    assert scores["auroc"] == 1.0
