"""Audit the external imaging source chain without exposing subject rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def collection_sha256(files: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n".encode("utf-8"))
    return digest.hexdigest().upper()


def require_unique(frame: pd.DataFrame, key: str, expected: int, stage: str) -> None:
    values = frame[key].fillna("").astype(str).str.strip()
    if len(frame) != expected or values.eq("").any() or values.duplicated().any():
        raise ValueError(f"{stage}: count, empty ID or duplicate ID failed")


def require_same(left: pd.Series, right: pd.Series, stage: str) -> None:
    if not left.fillna("").astype(str).reset_index(drop=True).equals(
        right.fillna("").astype(str).reset_index(drop=True)
    ):
        raise ValueError(f"{stage}: patient provenance mismatch")


def id_set_sha256(values: pd.Series) -> str:
    payload = "\n".join(sorted(values.astype(str).unique())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def audit(root: Path, *, rehash_dicom: bool = False) -> dict[str, object]:
    release = root / "data" / "deidentified_release_candidate" / "external"
    manifests = [release / diagnosis / "manifest.csv" for diagnosis in ("CAP", "COP")]
    crosswalk_path = root / "data" / "public_subject_model_crosswalk.csv"
    segmentation_path = root / "outputs" / "canonical_a_formal_20261011" / "segmentation" / "full_cohort_segmentation_replay_audit.csv"
    representation_path = root / "outputs" / "b_c_raw_dicom_replay_20261011" / "whole_chest_representation_replay_manifest.csv"
    phenotype_path = root / "outputs" / "c_raw_dicom_replay_20261011_v2" / "full_cohort_lesion_phenotype_features.csv"
    radiomics_path = root / "outputs" / "b_c_raw_dicom_replay_20261011" / "stable_runtime_regenerated_tabular" / "regenerated_tabular_inputs" / "external_lesion_radiomics.csv"
    source_paths = [*manifests, crosswalk_path, segmentation_path, representation_path, phenotype_path, radiomics_path]
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required provenance stage is absent: {path.relative_to(root)}")

    release_rows = pd.concat([pd.read_csv(path, dtype=str).assign(_manifest_dir=str(path.parent)) for path in manifests], ignore_index=True).fillna("")
    crosswalk = pd.read_csv(crosswalk_path, dtype=str).fillna("")
    crosswalk = crosswalk.loc[crosswalk["cohort"].eq("external")].copy()
    segmentation = pd.read_csv(segmentation_path, dtype=str).fillna("")
    segmentation = segmentation.loc[segmentation["cohort"].eq("external")].copy()
    representations = pd.read_csv(representation_path, dtype=str).fillna("")
    representations = representations.loc[representations["cohort"].eq("external")].copy()
    phenotype = pd.read_csv(phenotype_path, dtype=str).fillna("")
    phenotype = phenotype.loc[phenotype["site"].eq("External")].copy()
    radiomics = pd.read_csv(radiomics_path, dtype=str).fillna("")
    radiomics = radiomics.loc[radiomics["perturbation"].eq("BASELINE")].copy()
    stages = {"DICOM": release_rows, "crosswalk": crosswalk, "segmentation": segmentation,
              "tensor_and_embeddings": representations, "phenotype": phenotype, "radiomics": radiomics}
    for name, frame in stages.items():
        require_unique(frame, "public_subject_id", 33, name)
        if set(frame["public_subject_id"]) != set(crosswalk["public_subject_id"]):
            raise ValueError(f"{name}: public patient set mismatch")

    joined = crosswalk.set_index("public_subject_id")
    for name, frame in stages.items():
        if name == "crosswalk":
            continue
        aligned = frame.set_index("public_subject_id").loc[joined.index]
        if "label_binary" in aligned:
            require_same(aligned["label_binary"], joined["label_binary"], f"{name} labels")
        if name == "segmentation" and not aligned["status"].eq("PASS").all():
            raise ValueError("External segmentation audit is not PASS for all patients")
        if name == "phenotype":
            require_same(aligned["neutral_patient_id"], joined["segmentation_model_id"], "phenotype segmentation ID")
            require_same(aligned["source_neutral_imaging_id"], joined["lesion_imaging_model_id"], "phenotype lesion ID")
            require_same(aligned["neutral_external_id"], joined["external_tensor_id"], "phenotype tensor ID")
            require_same(aligned["label_binary"], joined["label_binary"], "phenotype labels")
        if name == "radiomics":
            require_same(aligned["neutral_patient_id"], joined["segmentation_model_id"], "radiomics segmentation ID")
            require_same(aligned["source_neutral_imaging_id"], joined["lesion_imaging_model_id"], "radiomics lesion ID")
            if not aligned["status"].eq("PASS").all():
                raise ValueError("External baseline radiomics extraction is not PASS for all patients")
        if name == "tensor_and_embeddings":
            require_same(aligned["external_tensor_id"], joined["external_tensor_id"], "representation tensor ID")

    rel = release_rows.set_index("public_subject_id").loc[joined.index]
    seg = segmentation.set_index("public_subject_id").loc[joined.index]
    rep = representations.set_index("public_subject_id").loc[joined.index]
    dicom_files = 0
    dicom_bytes = 0
    for public_id in joined.index:
        row = rel.loc[public_id]
        dicom_dir = Path(row["_manifest_dir"]) / row["ct_path"]
        if not dicom_dir.is_dir() or Path(seg.loc[public_id, "dicom_dir"]).resolve() != dicom_dir.resolve():
            raise ValueError("DICOM directory does not match the segmentation source")
        files = sorted(dicom_dir.glob("*.dcm"))
        size = sum(path.stat().st_size for path in files)
        if len(files) != int(row["dicom_file_count"]) or size != int(row["dicom_size_bytes"]):
            raise ValueError("DICOM count or size differs from release manifest")
        if rehash_dicom and collection_sha256(files, dicom_dir) != row["dicom_collection_sha256"].upper():
            raise ValueError("DICOM collection SHA-256 differs from release manifest")
        dicom_files += len(files)
        dicom_bytes += size
        mask = Path(seg.loc[public_id, "mask_path"])
        if not mask.is_file() or sha256_file(mask) != seg.loc[public_id, "mask_sha256"].upper():
            raise ValueError("Segmentation mask hash differs from audit")
        runtime = root / "outputs" / "b_c_raw_dicom_replay_20261011"
        qc_path = runtime / "runtime_tensors" / f"{public_id}_p95_96cubed_tensor_qc.json"
        if not qc_path.is_file():
            raise ValueError("Tensor QC file is missing")
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        tensor = runtime / "runtime_tensors" / f"{public_id}_p95_96cubed_tensor.npy"
        if Path(qc["dicom_dir"]).resolve() != dicom_dir.resolve() or qc["public_subject_id"] != public_id:
            raise ValueError("Tensor QC points to another DICOM source")
        if not tensor.is_file():
            raise ValueError("Tensor file is missing")
        tensor_hash = sha256_file(tensor)
        if tensor_hash != qc["tensor_sha256"].upper() or tensor_hash != rep.loc[public_id, "tensor_sha256"].upper():
            raise ValueError("Tensor hash differs from QC or representation manifest")
        for family in ("tangerine", "genesis"):
            embedding = runtime / "runtime_embeddings" / f"{public_id}_{family}_embedding.npy"
            if not embedding.is_file():
                raise ValueError(f"{family} embedding file is missing")
            if sha256_file(embedding) != rep.loc[public_id, f"{family}_sha256"].upper():
                raise ValueError(f"{family} embedding hash differs from representation manifest")

    return {
        "status": "PASS_FOR_EXISTING_REGENERATED_CHAIN_NOT_HISTORICAL_V2_FIT_CERTIFICATION",
        "external_subjects": len(joined),
        "public_id_set_sha256": id_set_sha256(joined.index.to_series()),
        "dicom_files": dicom_files,
        "dicom_bytes": dicom_bytes,
        "dicom_collection_hashes_recomputed": rehash_dicom,
        "source_file_sha256": {str(path.relative_to(root)): sha256_file(path) for path in source_paths},
        "source_id_mismatches": 0,
        "label_mismatches": 0,
        "file_hash_mismatches": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--rehash-dicom", action="store_true", help="Rehash every DICOM in the external cohort")
    args = parser.parse_args()
    print(json.dumps(audit(args.project_root.resolve(), rehash_dicom=args.rehash_dicom), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
