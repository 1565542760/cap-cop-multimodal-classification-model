"""Freeze corrected patient-matched model inputs without fitting a model."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_external_stable_provenance import audit as audit_external
from cap_cop.data.model_contracts import read_stable_cohort_metadata
from cap_cop.pipelines import stable_multirepresentation_replay_pipeline as stable


def prepare(output_dir: Path) -> dict[str, object]:
    root = ROOT.resolve()
    target = output_dir.resolve()
    if not target.is_relative_to((root / "outputs").resolve()):
        raise ValueError("Prepared run must be under the project outputs directory")
    if target.exists():
        raise FileExistsError("Prepared run directory already exists; historical outputs cannot be overwritten")

    metadata_source = stable.PATHS["internal_oof_ref"]
    phenotype_path = root / "outputs" / "c_raw_dicom_replay_20261011_v2" / "full_cohort_lesion_phenotype_features.csv"
    radiomics_path = root / "outputs" / "a_raw_dicom_replay_20261011_v4" / "03_feature_extraction" / "full_cohort_perturbation_pyradiomics_features.csv"
    locked_phenotype = stable.PATHS["lesion_features"]
    for path in (metadata_source, phenotype_path, radiomics_path, locked_phenotype):
        if not path.is_file():
            raise FileNotFoundError("A required cohort, phenotype or radiomics source is absent")
    metadata = read_stable_cohort_metadata(metadata_source, legacy=True)
    crosswalk = pd.read_csv(stable.PATHS["internal_crosswalk"], dtype=str)
    phenotype = pd.read_csv(phenotype_path)
    locked_columns = pd.read_csv(locked_phenotype, nrows=0).columns.tolist()
    if set(locked_columns).difference(phenotype.columns):
        raise ValueError("Regenerated phenotype does not meet the locked column contract")
    radiomics = pd.read_csv(radiomics_path)
    radiomics = radiomics.loc[radiomics["perturbation"].eq("BASELINE")].copy()
    if len(radiomics) != 408 or radiomics["public_subject_id"].nunique() != 408:
        raise ValueError("Full radiomics source lacks a unique 408-subject baseline")
    target.mkdir(parents=True, exist_ok=False)
    candidate_inputs = target / "validated_inputs"
    candidate_inputs.mkdir()
    phenotype_contract = candidate_inputs / "locked_contract_lesion_phenotype.csv"
    phenotype[locked_columns].to_csv(phenotype_contract, index=False)
    internal_ids = set(crosswalk.loc[crosswalk["cohort"].eq("internal"), "public_subject_id"])
    external_ids = set(crosswalk.loc[crosswalk["cohort"].eq("external"), "public_subject_id"])
    internal_radiomics = candidate_inputs / "internal_lesion_radiomics.csv"
    external_radiomics = candidate_inputs / "external_lesion_radiomics.csv"
    radiomics.loc[radiomics["public_subject_id"].isin(internal_ids)].to_csv(internal_radiomics, index=False)
    radiomics.loc[radiomics["public_subject_id"].isin(external_ids)].to_csv(external_radiomics, index=False)
    metadata_path = target / "locked_cohort_metadata.csv"
    with metadata_path.open("x", encoding="utf-8", newline="") as stream:
        metadata.to_csv(stream, index=False)
    stable.PATHS.update({"cohort_metadata": metadata_path, "lesion_features": phenotype_contract,
                         "d1_internal_features": internal_radiomics, "d1_external_features": external_radiomics})
    base, external_crosswalk, domains = stable.load_base()
    tab_i, tab_e, _, _ = stable.load_tabular_features(base, external_crosswalk)
    whole = root / "models" / "whole_chest_ct"
    embedding_root = root / "models" / "stable_multirepresentation_ct" / "embeddings"
    tan_i = pd.read_csv(whole / "internal_embedding_manifest.csv")
    tan_e = pd.read_csv(whole / "external_embedding_manifest.csv")
    emb_mi = {
        "FAMILY_C_GENESIS_FROZEN": base.copy(),
        "FAMILY_D_TANGERINE_FROZEN": tan_i.merge(base, on="neutral_imaging_id", how="left", validate="one_to_one")
            .sort_values("embedding_index")[["neutral_imaging_id", "label", "outer_fold", "primary_domain"]],
    }
    emb_me = {
        "FAMILY_C_GENESIS_FROZEN": external_crosswalk.copy(),
        "FAMILY_D_TANGERINE_FROZEN": tan_e.merge(external_crosswalk, on="neutral_external_id", how="left", validate="one_to_one")
            .sort_values("embedding_index")[["neutral_external_id", "neutral_patient_id"]],
    }
    emb_i = {family: np.load(embedding_root / filename, mmap_mode="r") for family, filename in (
        ("FAMILY_C_GENESIS_FROZEN", "genesis_internal_embeddings.npy"),
        ("FAMILY_D_TANGERINE_FROZEN", "tangerine_internal_embeddings.npy"))}
    emb_e = {family: np.load(embedding_root / filename, mmap_mode="r") for family, filename in (
        ("FAMILY_C_GENESIS_FROZEN", "genesis_external_embeddings.npy"),
        ("FAMILY_D_TANGERINE_FROZEN", "tangerine_external_embeddings.npy"))}
    identity = stable.validate_stable_prefit(base, tab_i, tab_e, emb_mi, emb_me, emb_i, emb_e)
    source = audit_external(root)
    try:
        torch_check = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.__version__, torch.cuda.is_available())"],
            capture_output=True, text=True, timeout=45, check=False)
        torch_returncode = torch_check.returncode
        torch_stdout = torch_check.stdout.strip()
        torch_error_type = "OSError" if "OSError" in torch_check.stderr else (
            "other" if torch_returncode else "none")
    except subprocess.TimeoutExpired:
        torch_returncode = -1
        torch_stdout = ""
        torch_error_type = "TimeoutExpired"
    free = shutil.disk_usage(target.parent).free
    input_paths = {
        "cohort_source": metadata_source,
        "crosswalk": stable.PATHS["internal_crosswalk"],
        "lesion_phenotype": phenotype_path,
        "lesion_radiomics": radiomics_path,
        "segmentation_audit": root / "outputs" / "canonical_a_formal_20261011" / "segmentation" / "full_cohort_segmentation_replay_audit.csv",
        "tangerine_encoder": whole / "foundation_encoder" / "tangerine_foundation_encoder.pth",
        "genesis_encoder": root / "models" / "stable_multirepresentation_ct" / "foundation_encoders" / "genesis_chest_ct.pt",
        "source_pipeline": root / "src" / "cap_cop" / "pipelines" / "stable_multirepresentation_replay_pipeline.py",
        "runner": root / "scripts" / "run_full_cohort_whole_chest_representations.py",
    }
    if any(not path.is_file() for path in input_paths.values()):
        raise FileNotFoundError("A required model C input, encoder or source file is missing")

    representation = [sys.executable, str(root / "scripts" / "run_full_cohort_stable_multirepresentation.py"),
                      "--phase", "representations", "--output-dir", str(target), "--device", "cuda"]
    fit = [sys.executable, str(root / "scripts" / "run_full_cohort_stable_multirepresentation.py"),
           "--phase", "stable_model", "--output-dir", str(target),
           "--cohort-metadata", str(metadata_path), "--capture-fit-artifacts",
           "--phenotype-features", str(phenotype_path),
           "--radiomics-features", str(radiomics_path)]
    manifest = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PREPARED_NOT_TRAINED" if torch_returncode == 0 else "BLOCKED_TORCH_IMPORT_NOT_TRAINED",
        "source_code_is_uncommitted": True,
        "seed": stable.SEED,
        "outer_folds": sorted(int(value) for value in base["outer_fold"].unique()),
        "inner_folds": stable.INNER_SPLITS,
        "identity_audit": identity,
        "radiomics_baseline_status_counts": {str(k): int(v) for k, v in radiomics["status"].value_counts().items()},
        "validated_input_sha256": {p.name: stable.sha256_file(p) for p in candidate_inputs.iterdir()},
        "external_source_audit": {key: source[key] for key in ("status", "external_subjects", "public_id_set_sha256", "source_id_mismatches", "file_hash_mismatches")},
        "input_sha256": {name: stable.sha256_file(path) for name, path in input_paths.items()},
        "cohort_metadata_sha256": stable.sha256_file(metadata_path),
        "free_bytes_before_fit": free,
        "torch_import_exit_code": torch_returncode,
        "torch_stdout": torch_stdout,
        "torch_error_type": torch_error_type,
        "run_commands_not_executed": [representation, fit],
    }
    (target / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.output_dir)
    print(json.dumps({key: result[key] for key in ("status", "identity_audit", "external_source_audit",
                                                   "cohort_metadata_sha256", "free_bytes_before_fit",
                                                   "torch_import_exit_code", "run_commands_not_executed")},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
