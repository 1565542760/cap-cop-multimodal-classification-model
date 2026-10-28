"""Check the local governed artifact bundle without reading patient-level values.

The public repository can run its code smoke tests without this bundle. Full
Internal375/External33 replay requires the authorized files checked here.
Only relative paths, counts, sizes, and SHA-256 values are reported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = {
    "cohort_crosswalk": "data/public_subject_model_crosswalk.csv",
    "model_config": "configs/models.json",
    "artifact_selectors": "configs/artifact_selectors.json",
    "segmentation_checkpoint_manifest": "models/segmentation/manifests/locked_checkpoint_manifest.csv",
    "segmentation_split_manifest": "models/segmentation/manifests/locked_nested_split_patient_manifest.csv",
    "segmentation_protocol_lock": "models/segmentation/manifests/lesion_segmentation_protocol_lock.json",
    "lesion_rf": "models/lesion_radiomics/final_random_forest.joblib",
    "lesion_feature_list": "models/lesion_radiomics/final_feature_list.csv",
    "whole_chest_encoder": "models/whole_chest_ct/foundation_encoder/tangerine_foundation_encoder.pth",
    "whole_chest_result_lock": "models/whole_chest_ct/result_lock.json",
    "stable_result_lock": "models/stable_multirepresentation_ct/result_lock.json",
    "stable_encoder": "models/stable_multirepresentation_ct/foundation_encoders/genesis_chest_ct.pt",
    "stable_cohort_metadata_source": "models/stable_multirepresentation_ct/derived_inputs/09_V3_4_STANDARD_INTERNAL_OOF_PREDICTIONS.csv",
    "stable_domain_contract": "models/stable_multirepresentation_ct/derived_inputs/03_PRIMARY_DG_DOMAIN_LOCK.json",
    "exploratory_external_lock": "models/supplementary_exploratory/full_chest_densenet/external_prediction_lock.json",
    "fusion_weight_lock": "models/multimodal_fusion/final_weight_lock.csv",
    "fusion_result_lock": "models/multimodal_fusion/final_result_lock.json",
}

COHORT_CONTRACTS = {
    "internal_cap": {
        "manifest": "data/deidentified_release_candidate/internal/CAP/manifest.csv",
        "clinical": "data/deidentified_release_candidate/internal/CAP/clinical/clinical_features.csv",
        "dicom": "data/deidentified_release_candidate/internal/CAP/imaging/ct",
        "rows": 250,
        "diagnosis": "CAP",
        "label": "0",
    },
    "internal_cop": {
        "manifest": "data/deidentified_release_candidate/internal/COP/manifest.csv",
        "clinical": "data/deidentified_release_candidate/internal/COP/clinical/clinical_features.csv",
        "dicom": "data/deidentified_release_candidate/internal/COP/imaging/ct",
        "rows": 125,
        "diagnosis": "COP",
        "label": "1",
    },
    "external_cap": {
        "manifest": "data/deidentified_release_candidate/external/CAP/manifest.csv",
        "clinical": "data/deidentified_release_candidate/external/CAP/clinical/clinical_features.csv",
        "dicom": "data/deidentified_release_candidate/external/CAP/imaging/ct",
        "rows": 19,
        "diagnosis": "CAP",
        "label": "0",
    },
    "external_cop": {
        "manifest": "data/deidentified_release_candidate/external/COP/manifest.csv",
        "clinical": "data/deidentified_release_candidate/external/COP/clinical/clinical_features.csv",
        "dicom": "data/deidentified_release_candidate/external/COP/imaging/ct",
        "rows": 14,
        "diagnosis": "COP",
        "label": "1",
    },
}

CLINICAL_FEATURES = (
    "age",
    "height_cm",
    "weight_kg",
    "body_mass_index",
    "fever",
    "cough",
    "sputum_category",
    "chest_tightness",
    "dyspnoea",
    "haemoptysis",
    "weight_loss",
    "lymphocyte_count",
    "erythrocyte_sedimentation_rate",
    "c_reactive_protein",
    "high_sensitivity_c_reactive_protein",
    "procalcitonin",
)

REQUIRED_DIRECTORIES = {
    "internal_cap_clinical": "data/deidentified_release_candidate/internal/CAP/clinical",
    "internal_cap_dicom": "data/deidentified_release_candidate/internal/CAP/imaging/ct",
    "internal_cop_clinical": "data/deidentified_release_candidate/internal/COP/clinical",
    "internal_cop_dicom": "data/deidentified_release_candidate/internal/COP/imaging/ct",
    "external_cap_clinical": "data/deidentified_release_candidate/external/CAP/clinical",
    "external_cap_dicom": "data/deidentified_release_candidate/external/CAP/imaging/ct",
    "external_cop_clinical": "data/deidentified_release_candidate/external/COP/clinical",
    "external_cop_dicom": "data/deidentified_release_candidate/external/COP/imaging/ct",
    "segmentation_checkpoints": "models/segmentation/locked_checkpoints",
    "clinical_structural_members": "models/supplementary_exploratory/structural_missingness_clinical/ensemble_models",
    "exploratory_checkpoints": "models/supplementary_exploratory/full_chest_densenet/checkpoints",
}

EXPECTED_COUNTS = {
    "segmentation_checkpoints": ("*.pt", 20),
    "clinical_structural_members": ("*.joblib", 15),
    "exploratory_checkpoints": ("*.pt", 15),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": relative(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256(path) if path.is_file() else None,
    }


def directory_record(path: Path) -> dict[str, Any]:
    files = [item for item in path.rglob("*") if item.is_file()] if path.is_dir() else []
    subject_directories = [item for item in path.iterdir() if item.is_dir()] if path.is_dir() else []
    return {
        "path": relative(path),
        "exists": path.is_dir(),
        "file_count": len(files),
        "total_bytes": sum(item.stat().st_size for item in files),
        "direct_subject_directory_count": len(subject_directories),
    }


def csv_contract(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": relative(path), "exists": False}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    ids = [str(row.get("public_subject_id", "")).strip() for row in rows]
    return {
        "path": relative(path),
        "exists": True,
        "rows": len(rows),
        "columns": list(reader.fieldnames or []),
        "unique_public_subject_ids": len(set(ids)),
        "empty_public_subject_id_count": sum(not value for value in ids),
        "diagnosis_values": sorted(
            {str(row.get("diagnosis_label", "")).strip() for row in rows if str(row.get("diagnosis_label", "")).strip()}
        ),
        "label_values": sorted(
            {str(row.get("label_binary", "")).strip() for row in rows if str(row.get("label_binary", "")).strip()}
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "project_root": ".",
        "status": "PASS",
        "required_files": {},
        "required_directories": {},
        "count_checks": {},
        "external_mask_directory_count": 0,
        "missing": [],
    }

    for name, relative_path in REQUIRED_FILES.items():
        record = file_record(PROJECT_ROOT / relative_path)
        result["required_files"][name] = record
        if not record["exists"]:
            result["missing"].append(relative_path)

    for name, relative_path in REQUIRED_DIRECTORIES.items():
        record = directory_record(PROJECT_ROOT / relative_path)
        result["required_directories"][name] = record
        if not record["exists"]:
            result["missing"].append(relative_path)

    for name, (pattern, expected) in EXPECTED_COUNTS.items():
        directory = PROJECT_ROOT / REQUIRED_DIRECTORIES[name]
        actual = len(list(directory.glob(pattern))) if directory.is_dir() else 0
        result["count_checks"][name] = {"pattern": pattern, "expected": expected, "actual": actual}
        if actual != expected:
            result["missing"].append(f"{REQUIRED_DIRECTORIES[name]}/{pattern} (expected {expected}, found {actual})")

    result["cohort_contracts"] = {}
    for name, contract in COHORT_CONTRACTS.items():
        manifest = PROJECT_ROOT / contract["manifest"]
        clinical = PROJECT_ROOT / contract["clinical"]
        dicom = PROJECT_ROOT / contract["dicom"]
        manifest_record = csv_contract(manifest)
        clinical_record = csv_contract(clinical)
        dicom_record = directory_record(dicom)
        result["cohort_contracts"][name] = {
            "manifest": manifest_record,
            "clinical": clinical_record,
            "dicom": dicom_record,
        }
        if not manifest_record.get("exists") or not clinical_record.get("exists"):
            result["missing"].extend([contract["manifest"], contract["clinical"]])
            continue
        if manifest_record["rows"] != contract["rows"]:
            result["missing"].append(f"{contract['manifest']} row count")
        if clinical_record["rows"] != contract["rows"]:
            result["missing"].append(f"{contract['clinical']} row count")
        if manifest_record["unique_public_subject_ids"] != contract["rows"]:
            result["missing"].append(f"{contract['manifest']} unique ID count")
        if clinical_record["unique_public_subject_ids"] != contract["rows"]:
            result["missing"].append(f"{contract['clinical']} unique ID count")
        if set(manifest_record["columns"]) < {"public_subject_id", "diagnosis_label", "label_binary"}:
            result["missing"].append(f"{contract['manifest']} label columns")
        if set(clinical_record["columns"]) != {"public_subject_id", "label_binary", *CLINICAL_FEATURES}:
            result["missing"].append(f"{contract['clinical']} 16-feature contract")
        if manifest_record["diagnosis_values"] != [contract["diagnosis"]] or manifest_record["label_values"] != [contract["label"]]:
            result["missing"].append(f"{contract['manifest']} diagnosis/label contract")
        if clinical_record["diagnosis_values"] != [] or clinical_record["label_values"] != [contract["label"]]:
            result["missing"].append(f"{contract['clinical']} label contract")
        if dicom_record["direct_subject_directory_count"] != contract["rows"]:
            result["missing"].append(f"{contract['dicom']} subject directory count")

    external_root = PROJECT_ROOT / "data" / "deidentified_release_candidate" / "external"
    external_masks = [path for path in external_root.rglob("masks") if path.is_dir()] if external_root.is_dir() else []
    result["external_mask_directory_count"] = len(external_masks)
    if external_masks:
        result["missing"].append("external masks must remain absent from the public data tree")

    if result["missing"]:
        result["status"] = "MISSING_AUTHORIZED_ARTIFACTS"

    text = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    print(text, end="")
    if args.json_out:
        output = args.json_out.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
