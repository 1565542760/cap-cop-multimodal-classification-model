"""Audit stable-representation training inputs without fitting or writing predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import numpy as np
import pandas as pd

from cap_cop.data.public_identity import canonical_id_set_sha256
from cap_cop.pipelines import stable_multirepresentation_replay_pipeline as stable


def set_input_root(input_root: Path) -> None:
    """Read authorised inputs outside the source ZIP without changing model rules."""
    global PROJECT
    source_root = stable.PROJECT.resolve()
    target = input_root.resolve()
    if not target.is_dir():
        raise FileNotFoundError(f"Input root is missing: {target}")
    stable.PATHS = {
        key: target / path.resolve().relative_to(source_root)
        if path.resolve().is_relative_to(source_root) else path
        for key, path in stable.PATHS.items()
    }
    stable.PROJECT = target
    stable.MODEL_DIR = target / "models" / "stable_multirepresentation_ct"
    PROJECT = target


def legacy_mismatch_count(base: pd.DataFrame, source: pd.DataFrame, crosswalk: pd.DataFrame) -> int:
    source = source.loc[source["site"].eq("Internal")] if "site" in source else source
    wrong = base[["neutral_imaging_id"]].merge(
        source[["neutral_patient_id", "source_neutral_imaging_id"]],
        left_on="neutral_imaging_id", right_on="source_neutral_imaging_id",
        how="left", validate="one_to_one")
    internal = crosswalk.loc[crosswalk["cohort"].eq("internal")]
    targets = base[["neutral_imaging_id"]].merge(
        internal[["public_subject_id", "whole_chest_imaging_model_id"]],
        left_on="neutral_imaging_id", right_on="whole_chest_imaging_model_id",
        validate="one_to_one")
    sources = wrong.merge(internal[["public_subject_id", "segmentation_model_id"]],
                          left_on="neutral_patient_id", right_on="segmentation_model_id",
                          how="left", validate="many_to_one")
    return int((targets["public_subject_id"].to_numpy() != sources["public_subject_id"].to_numpy()).sum())


def matrix_value_mismatch_count(frame: pd.DataFrame, source: pd.DataFrame,
                                crosswalk: pd.DataFrame, features: list[str], cohort: str) -> int:
    patients = crosswalk.loc[crosswalk["cohort"].eq(cohort),
                             ["public_subject_id", "segmentation_model_id"]]
    if "site" in source:
        source = source.loc[source["site"].eq(cohort.title())]
    expected = patients.merge(source[["neutral_patient_id"] + features],
                              left_on="segmentation_model_id", right_on="neutral_patient_id",
                              how="left", validate="one_to_one")
    expected = frame[["public_subject_id"]].merge(expected, on="public_subject_id",
                                                   how="left", validate="one_to_one")
    left = frame[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    right = expected[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return int((~np.isclose(left, right, equal_nan=True, rtol=0, atol=0)).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path,
                        help="Root of an authorised local input bundle; source code remains in this ZIP")
    parser.add_argument("--regenerated-input-dir", type=Path,
                        help="Directory containing the locked-contract phenotype and internal/external radiomics CSVs")
    args = parser.parse_args()
    if args.input_root:
        set_input_root(args.input_root)
    if args.regenerated_input_dir:
        root = args.regenerated_input_dir.resolve()
        stable.PATHS["lesion_features"] = root / "locked_contract_lesion_phenotype.csv"
        stable.PATHS["d1_internal_features"] = root / "internal_lesion_radiomics.csv"
        stable.PATHS["d1_external_features"] = root / "external_lesion_radiomics.csv"
    base, ext, _ = stable.load_base()
    crosswalk = stable.read_csv(stable.PATHS["internal_crosswalk"])
    old_phenotype = legacy_mismatch_count(base, stable.read_csv(stable.PATHS["lesion_features"]), crosswalk)
    old_radiomics = legacy_mismatch_count(base, stable.read_csv(stable.PATHS["d1_internal_features"]), crosswalk)
    internal, external, features, _ = stable.load_tabular_features(base, ext)
    matrix_mismatches = {}
    for family, source_path in (
        ("FAMILY_A_HANDCRAFTED", stable.PATHS["lesion_features"]),
        ("FAMILY_B_RAD_D1_FINAL13", stable.PATHS["d1_internal_features"]),
    ):
        matrix_mismatches[f"{family}_internal"] = matrix_value_mismatch_count(
            internal[family], stable.read_csv(source_path), crosswalk, features[family], "internal")
    matrix_mismatches["FAMILY_A_HANDCRAFTED_external"] = matrix_value_mismatch_count(
        external["FAMILY_A_HANDCRAFTED"], stable.read_csv(stable.PATHS["lesion_features"]),
        crosswalk, features["FAMILY_A_HANDCRAFTED"], "external")
    matrix_mismatches["FAMILY_B_RAD_D1_FINAL13_external"] = matrix_value_mismatch_count(
        external["FAMILY_B_RAD_D1_FINAL13"], stable.read_csv(stable.PATHS["d1_external_features"]),
        crosswalk, features["FAMILY_B_RAD_D1_FINAL13"], "external")
    model_dir = PROJECT / "models" / "stable_multirepresentation_ct" / "embeddings"
    whole_dir = PROJECT / "models" / "whole_chest_ct"
    tan_i = pd.read_csv(whole_dir / "internal_embedding_manifest.csv")
    tan_e = pd.read_csv(whole_dir / "external_embedding_manifest.csv")
    emb_mi = {
        "FAMILY_C_GENESIS_FROZEN": base.copy(),
        "FAMILY_D_TANGERINE_FROZEN": tan_i.merge(base, on="neutral_imaging_id", how="left", validate="one_to_one")
            .sort_values("embedding_index")[["neutral_imaging_id", "label", "outer_fold", "primary_domain"]],
    }
    emb_me = {
        "FAMILY_C_GENESIS_FROZEN": ext.copy(),
        "FAMILY_D_TANGERINE_FROZEN": tan_e.merge(ext, on="neutral_external_id", how="left", validate="one_to_one")
            .sort_values("embedding_index")[["neutral_external_id", "neutral_patient_id"]],
    }
    emb_i = {
        "FAMILY_C_GENESIS_FROZEN": np.load(model_dir / "genesis_internal_embeddings.npy", mmap_mode="r"),
        "FAMILY_D_TANGERINE_FROZEN": np.load(model_dir / "tangerine_internal_embeddings.npy", mmap_mode="r"),
    }
    emb_e = {
        "FAMILY_C_GENESIS_FROZEN": np.load(model_dir / "genesis_external_embeddings.npy", mmap_mode="r"),
        "FAMILY_D_TANGERINE_FROZEN": np.load(model_dir / "tangerine_external_embeddings.npy", mmap_mode="r"),
    }
    outcome = stable.validate_stable_prefit(base, internal, external, emb_mi, emb_me, emb_i, emb_e)
    external_phenotype = stable.read_csv(stable.PATHS["lesion_features"])
    empty_source_count = int(external_phenotype.loc[external_phenotype["site"].eq("External"),
                                                "source_neutral_imaging_id"]
                             .fillna("").astype(str).str.strip().eq("").sum())
    outcome.update({
        "legacy_direct_join_phenotype_mismatches": old_phenotype,
        "legacy_direct_join_radiomics_mismatches": old_radiomics,
        "internal_id_set_sha256": canonical_id_set_sha256(internal["FAMILY_A_HANDCRAFTED"]["public_subject_id"]),
        "external_id_set_sha256": canonical_id_set_sha256(external["FAMILY_A_HANDCRAFTED"]["public_subject_id"]),
        "feature_counts": {family: len(names) for family, names in features.items()},
        "matrix_numeric_value_mismatches": matrix_mismatches,
        "external_phenotype_source_id_empty_rows": empty_source_count,
        "external_phenotype_source_id_crosswalk_status": (
            "source field empty; original CT provenance not certified" if empty_source_count
            else "source imaging ID matches public crosswalk; raw CT provenance is separate"),
    })
    print(json.dumps(outcome, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
