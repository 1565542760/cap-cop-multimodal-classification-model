import json
import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

from cap_cop.data.model_contracts import (
    align_radiomics_training_inputs,
    read_stable_cohort_metadata,
    require_binary_classes,
)
from cap_cop.private_lock_selectors import get_selector, load_selectors


ROOT = Path(__file__).resolve().parents[1]


def test_public_selectors_cover_models_and_clinical_branches():
    selectors = load_selectors()
    config = json.loads((ROOT / "configs/paper_model_locked_pairs.json").read_text())
    assert set(config["paper_models"]) <= set(selectors)
    assert len(selectors) == 11
    for name in selectors:
        assert get_selector(name) == selectors[name]
    assert not any("patient" in name or "subject" in name for name in selectors)


def test_referenced_drivers_are_in_the_package():
    for name in [
        "run_supplementary_exploratory_subset_replay.py",
        "run_full_cohort_exploratory_densenet.py",
        "run_full_cohort_whole_chest_representations.py",
        "run_full_cohort_stable_multirepresentation.py",
        "export_stable_cohort_metadata.py",
    ]:
        assert (ROOT / "scripts" / name).is_file()


def training_tables():
    rows = [
        {"neutral_patient_id": f"sample_{fold}_{label}", "label_binary": label,
         "outer_fold": fold, "feature": float(fold + label)}
        for fold in range(5) for label in range(2)
    ]
    baseline = pd.DataFrame(rows)
    labels = baseline[["neutral_patient_id", "label_binary"]].sample(frac=1, random_state=17)
    return labels, baseline


def test_shuffled_label_rows_keep_patient_alignment():
    labels, baseline = training_tables()
    x, y, folds = align_radiomics_training_inputs(labels, baseline, ["feature"])
    assert x.index.equals(y.index) and x.index.equals(folds.index)
    assert y.tolist() == baseline["label_binary"].tolist()
    pd.testing.assert_series_equal(
        y, labels.set_index("neutral_patient_id")["label_binary"].loc[x.index],
        check_dtype=False,
    )


@pytest.mark.parametrize("problem", ["duplicate_labels", "duplicate_baseline", "missing", "conflict", "single_class"])
def test_invalid_training_identity_fails_before_fitting(problem):
    labels, baseline = training_tables()
    if problem == "duplicate_labels":
        labels = pd.concat([labels, labels.iloc[[0]]])
    elif problem == "duplicate_baseline":
        baseline = pd.concat([baseline, baseline.iloc[[0]]])
    elif problem == "missing":
        labels = labels.iloc[1:]
    elif problem == "conflict":
        baseline.loc[0, "label_binary"] = 1
    else:
        labels["label_binary"] = 0
    with pytest.raises(ValueError):
        align_radiomics_training_inputs(labels, baseline, ["feature"])


def test_one_class_error_reports_counts_not_patient_ids():
    with pytest.raises(ValueError, match="observed counts") as error:
        require_binary_classes(pd.Series([0, 0], index=["private_a", "private_b"]), "Inner training")
    assert "private" not in str(error.value)


def test_radiomics_feature_columns_never_include_patient_ids():
    from cap_cop.pipelines.lesion_radiomics_replay_pipeline import feature_cols

    table = pd.DataFrame({"public_subject_id": ["synthetic"],
                          "neutral_patient_id": ["neutral"],
                          "source_neutral_imaging_id": ["source"],
                          "outer_fold": [0], "perturbation": ["BASELINE"],
                          "status": ["PASS"], "error": [""],
                          "original_shape_Area": [1.0]})
    assert feature_cols(table) == ["original_shape_Area"]


def stable_frame():
    _, baseline = training_tables()
    return baseline.rename(columns={"neutral_patient_id": "neutral_imaging_id", "label_binary": "label"}).assign(primary_domain="scanner_group")


def test_stable_metadata_requires_no_predictions(tmp_path):
    frame = stable_frame()
    path = tmp_path / "metadata.csv"
    frame.drop(columns="feature").to_csv(path, index=False)
    actual = read_stable_cohort_metadata(path)
    assert actual.columns.tolist() == ["neutral_imaging_id", "label", "outer_fold", "primary_domain"]
    assert len(actual) == 10


def test_legacy_metadata_ignores_probability_values(tmp_path):
    legacy = stable_frame().rename(columns={"outer_fold": "fold_or_domain"})
    legacy["candidate"] = "R0_GENESIS_B2_BASELINE"
    legacy["prob_COP"] = "not_a_probability"
    path = tmp_path / "legacy.csv"
    legacy.to_csv(path, index=False)
    first = read_stable_cohort_metadata(path, legacy=True)
    legacy["prob_COP"] = -9000
    legacy.to_csv(path, index=False)
    second = read_stable_cohort_metadata(path, legacy=True)
    pd.testing.assert_frame_equal(first, second)


def test_conflicting_legacy_metadata_is_not_silently_deduplicated(tmp_path):
    frame = stable_frame().rename(columns={"outer_fold": "fold_or_domain"})
    frame["candidate"] = "R0_GENESIS_B2_BASELINE"
    extra = frame.iloc[[0]].copy()
    extra["label"] = 1
    path = tmp_path / "conflict.csv"
    pd.concat([frame, extra]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        read_stable_cohort_metadata(path, legacy=True)


def test_invalid_selector_schema(tmp_path):
    path = tmp_path / "selectors.json"
    path.write_text('{"model": null}')
    with pytest.raises(ValueError):
        load_selectors(path)


def test_matrix_modelling_import_does_not_require_radiomics(monkeypatch):
    monkeypatch.setitem(sys.modules, "radiomics", None)
    pipeline = importlib.import_module("cap_cop.pipelines.lesion_radiomics_replay_pipeline")
    assert callable(pipeline.nested_cv)
    # Extraction still requires the real package; never substitute a stub.
    with pytest.raises(ModuleNotFoundError):
        pipeline.extractor_for_spacing()
