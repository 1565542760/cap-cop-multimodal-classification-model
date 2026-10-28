"""Synthetic regression cases for stable-representation patient alignment."""

import pandas as pd
import numpy as np
import pytest

from cap_cop.data.model_contracts import align_stable_patient_features, stable_patient_contract
from cap_cop.pipelines import stable_multirepresentation_replay_pipeline as stable
from cap_cop.pipelines.stable_multirepresentation_replay_pipeline import combine_tabular


def test_optional_fit_artifact_writer_keeps_inputs_and_refuses_overwrite(tmp_path):
    train = pd.DataFrame({"neutral_imaging_id": ["SYN_A", "SYN_B"],
                          "label": [0, 1], "outer_fold": [0, 1], "feature": [1.0, 2.0]})
    held_out = train.iloc[:1].copy()
    external = pd.DataFrame({"neutral_external_id": ["SYN_EXT"],
                             "neutral_patient_id": ["SYN_SEG"], "feature": [3.0]})
    target = tmp_path / "fit_artifacts"
    stable.record_candidate_inputs(target, train, held_out, external,
                                   np.ones((2, 1)), np.ones((1, 1)), np.ones((1, 1)))
    assert (target / "train_model_matrix.npy").is_file()
    assert not (target / "model.joblib").exists()
    stable.record_fitted_candidate(target, train, held_out, external, ["feature"],
                                   np.ones((2, 1)), np.ones((1, 1)), np.ones((1, 1)),
                                   {"imputer": "synthetic"}, {"scaler": "synthetic"},
                                   {"model": "synthetic"}, {"max_depth": 3})
    assert (target / "artifact_manifest.json").is_file()
    assert (target / "train_model_matrix.npy").is_file()
    with pytest.raises(FileExistsError):
        stable.record_fitted_candidate(target, train, held_out, external, ["feature"],
                                       np.ones((2, 1)), np.ones((1, 1)), np.ones((1, 1)),
                                       {}, {}, {}, {})


def inputs():
    crosswalk = pd.DataFrame([
        {"public_subject_id": f"SUBJ_{n}", "cohort": "internal", "label_binary": n % 2,
         "segmentation_model_id": f"SEG_{n}", "lesion_imaging_model_id": f"SAME_{(n + 1) % 5}",
         "whole_chest_imaging_model_id": f"SAME_{n}", "external_tensor_id": ""}
        for n in range(5)
    ] + [
        {"public_subject_id": f"EXT_{n}", "cohort": "external", "label_binary": n % 2,
         "segmentation_model_id": f"ESEG_{n}", "lesion_imaging_model_id": f"ELES_{n}",
         "whole_chest_imaging_model_id": f"EWHOLE_{n}", "external_tensor_id": f"TENSOR_{n}"}
        for n in range(2)
    ])
    base = pd.DataFrame({"neutral_imaging_id": [f"SAME_{n}" for n in range(5)],
                         "label": [n % 2 for n in range(5)], "outer_fold": list(range(5)),
                         "primary_domain": ["site"] * 5})
    external = pd.DataFrame({"neutral_external_id": ["TENSOR_0", "TENSOR_1"],
                             "neutral_patient_id": ["ESEG_0", "ESEG_1"]})
    source = pd.DataFrame({"neutral_patient_id": [f"SEG_{n}" for n in range(5)],
                           "source_neutral_imaging_id": [f"SAME_{(n + 1) % 5}" for n in range(5)],
                           "label_binary": [n % 2 for n in range(5)],
                           "outer_fold": list(range(5)), "feature": list(range(5))})
    return crosswalk, base, external, source


def aligned():
    crosswalk, base, external, source = inputs()
    internal_contract, external_contract = stable_patient_contract(base, external, crosswalk)
    joined = align_stable_patient_features(internal_contract, source, ["feature"],
                                            cohort="internal", check_lesion_id=True)
    return joined, external_contract


def test_cross_namespace_same_strings_do_not_match_wrong_patient():
    joined, _ = aligned()
    assert joined["feature"].tolist() == list(range(5))
    assert joined["source_neutral_imaging_id"].tolist() != joined["neutral_imaging_id"].tolist()


def test_source_row_shuffle_is_invariant():
    crosswalk, base, external, source = inputs()
    internal, _ = stable_patient_contract(base, external, crosswalk)
    shuffled = source.sample(frac=1, random_state=17).reset_index(drop=True)
    joined = align_stable_patient_features(internal, shuffled, ["feature"],
                                            cohort="internal", check_lesion_id=True)
    assert joined["feature"].tolist() == list(range(5))


@pytest.mark.parametrize("field", ["source_neutral_imaging_id", "label_binary", "outer_fold"])
def test_swapped_source_identity_or_metadata_fails(field):
    crosswalk, base, external, source = inputs()
    internal, _ = stable_patient_contract(base, external, crosswalk)
    source.loc[[0, 1], field] = source.loc[[1, 0], field].to_numpy()
    with pytest.raises(ValueError):
        align_stable_patient_features(internal, source, ["feature"], cohort="internal",
                                      check_lesion_id=True)


def test_duplicate_or_missing_source_id_fails():
    crosswalk, base, external, source = inputs()
    internal, _ = stable_patient_contract(base, external, crosswalk)
    for changed in (source.assign(neutral_patient_id=["SEG_0"] * 5), source.iloc[:-1]):
        with pytest.raises(ValueError):
            align_stable_patient_features(internal, changed, ["feature"],
                                          cohort="internal", check_lesion_id=True)


def test_external_pair_must_be_verified():
    crosswalk, base, external, _ = inputs()
    _, contract = stable_patient_contract(base, external, crosswalk)
    source = pd.DataFrame({"neutral_patient_id": ["ESEG_0", "ESEG_1"],
                           "neutral_external_id": ["TENSOR_0", "TENSOR_1"],
                           "feature": [1.0, 2.0]})
    aligned_source = align_stable_patient_features(contract, source, ["feature"],
                                                    cohort="external", check_external_tensor=True)
    assert aligned_source["feature"].tolist() == [1.0, 2.0]
    source.loc[[0, 1], "neutral_external_id"] = source.loc[[1, 0], "neutral_external_id"].to_numpy()
    with pytest.raises(ValueError, match="external tensor ID"):
        align_stable_patient_features(contract, source, ["feature"],
                                      cohort="external", check_external_tensor=True)


def test_external_source_imaging_id_is_checked_when_supplied():
    crosswalk, base, external, _ = inputs()
    _, contract = stable_patient_contract(base, external, crosswalk)
    source = pd.DataFrame({"neutral_patient_id": ["ESEG_0", "ESEG_1"],
                           "source_neutral_imaging_id": ["ELES_0", "ELES_1"],
                           "feature": [1.0, 2.0]})
    assert len(align_stable_patient_features(contract, source, ["feature"],
                                              cohort="external", check_lesion_id=True)) == 2
    source.loc[0, "source_neutral_imaging_id"] = "ELES_1"
    with pytest.raises(ValueError, match="lesion imaging ID"):
        align_stable_patient_features(contract, source, ["feature"],
                                      cohort="external", check_lesion_id=True)


def test_feature_families_must_have_same_patient_rows():
    joined, _ = aligned()
    family = joined[["public_subject_id", "neutral_imaging_id", "label", "outer_fold", "primary_domain", "feature"]]
    other = family.rename(columns={"feature": "other"}).copy()
    combined, names = combine_tabular(family, other, ["feature"], ["other"])
    assert len(combined) == 5 and names == ["feature", "other"]
    with pytest.raises(ValueError, match="patient rows"):
        combine_tabular(family, other.iloc[::-1].reset_index(drop=True), ["feature"], ["other"])


def test_prefit_rejects_row_order_label_fold_and_embedding_drift(tmp_path, monkeypatch):
    crosswalk, base, external_index, source = inputs()
    crosswalk_path = tmp_path / "synthetic_crosswalk.csv"
    crosswalk.to_csv(crosswalk_path, index=False)
    monkeypatch.setitem(stable.PATHS, "internal_crosswalk", crosswalk_path)
    internal_contract, external_contract = stable_patient_contract(base, external_index, crosswalk)
    internal = align_stable_patient_features(internal_contract, source, ["feature"],
                                             cohort="internal", check_lesion_id=True)
    i = internal[["public_subject_id", "neutral_imaging_id", "label", "outer_fold", "primary_domain", "feature"]]
    e = external_contract[["public_subject_id", "neutral_external_id", "neutral_patient_id"]]
    ti, te = {"phenotype": i}, {"phenotype": e}
    mi, me = {"encoder": base.copy()}, {"encoder": external_index.copy()}
    ai, ae = {"encoder": np.zeros((5, 2))}, {"encoder": np.zeros((2, 2))}
    assert stable.validate_stable_prefit(base, ti, te, mi, me, ai, ae)["internal_patients"] == 5
    for column in ("public_subject_id", "label", "outer_fold"):
        bad = i.copy()
        bad.loc[[0, 1], column] = bad.loc[[1, 0], column].to_numpy()
        with pytest.raises(ValueError):
            stable.validate_stable_prefit(base, {"phenotype": bad}, te, mi, me, ai, ae)
    with pytest.raises(ValueError, match="embedding manifest"):
        stable.validate_stable_prefit(base, ti, te, {"encoder": base.iloc[::-1]}, me, ai, ae)


def test_standard_entry_stops_before_any_fit_on_bad_identity(tmp_path, monkeypatch):
    crosswalk, base, external_index, source = inputs()
    crosswalk_path = tmp_path / "synthetic_crosswalk.csv"
    crosswalk.to_csv(crosswalk_path, index=False)
    monkeypatch.setitem(stable.PATHS, "internal_crosswalk", crosswalk_path)
    internal_contract, external_contract = stable_patient_contract(base, external_index, crosswalk)
    joined = align_stable_patient_features(internal_contract, source, ["feature"],
                                            cohort="internal", check_lesion_id=True)
    internal = joined[["public_subject_id", "neutral_imaging_id", "label", "outer_fold", "primary_domain", "feature"]].copy()
    internal.loc[0, "public_subject_id"] = "SUBJ_1"
    external = external_contract[["public_subject_id", "neutral_external_id", "neutral_patient_id"]]
    monkeypatch.setattr(stable, "fit_tabular_candidate", lambda *args, **kwargs: pytest.fail("fit was reached"))
    with pytest.raises(ValueError, match="duplicate patient identifiers"):
        stable.run_standard(base, [], {"phenotype": internal}, {"phenotype": external},
                            {}, {}, {}, {}, {})
