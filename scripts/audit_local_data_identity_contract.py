from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
CANONICAL = DATA_ROOT / "public_subject_model_crosswalk.csv"
RELEASE_ROOT = DATA_ROOT / "deidentified_release_candidate"
OUTPUT = PROJECT_ROOT / "docs" / "full_reproduction_evidence_20261011" / "local_data_identity_contract.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def set_sha256(values: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest().upper()


def read_clinical_ids(path: Path) -> tuple[pd.DataFrame, str]:
    frame = pd.read_csv(path, dtype=str)
    for column in ("public_subject_id", "case_id_public"):
        if column in frame.columns:
            return frame, column
    raise ValueError(f"No public ID column in {path}")


def audit_tree(root: Path, canonical: pd.DataFrame, release: bool) -> dict[str, object]:
    groups: dict[str, object] = {}
    all_manifest_ids: set[str] = set()
    all_clinical_ids: set[str] = set()
    all_ct_ids: set[str] = set()
    for cohort, diagnosis in (("internal", "CAP"), ("internal", "COP"), ("external", "CAP"), ("external", "COP")):
        expected = canonical.loc[
            canonical["cohort"].eq(cohort) & canonical["diagnosis_label"].eq(diagnosis)
        ].copy()
        expected_ids = set(expected["public_subject_id"])
        group_root = root / cohort / diagnosis
        manifest_path = group_root / "manifest.csv"
        clinical_path = group_root / "clinical" / "clinical_features.csv"
        ct_root = group_root / "imaging" / "ct"
        manifest = pd.read_csv(manifest_path, dtype=str)
        clinical, clinical_id_column = read_clinical_ids(clinical_path)
        manifest_ids = set(manifest["public_subject_id"])
        clinical_ids = set(clinical[clinical_id_column])
        ct_ids = {path.name for path in ct_root.iterdir() if path.is_dir()}
        duplicate_manifest_ids = int(manifest["public_subject_id"].duplicated().sum())
        duplicate_clinical_ids = int(clinical[clinical_id_column].duplicated().sum())
        empty_manifest_ids = int(manifest["public_subject_id"].isna().sum() + manifest["public_subject_id"].eq("").sum())
        empty_clinical_ids = int(clinical[clinical_id_column].isna().sum() + clinical[clinical_id_column].eq("").sum())
        manifest_label_matches = True
        if "label_binary" in manifest.columns:
            label_map = expected.set_index("public_subject_id")["label_binary"].astype(str)
            manifest_label_matches = manifest.set_index("public_subject_id")["label_binary"].astype(str).eq(label_map).all()
        clinical_label_matches = True
        if "label_binary" in clinical.columns:
            label_map = expected.set_index("public_subject_id")["label_binary"].astype(str)
            clinical_labels = clinical.set_index(clinical_id_column)["label_binary"].astype(str)
            clinical_label_matches = clinical_labels.eq(label_map).all()
        groups[f"{cohort}_{diagnosis}"] = {
            "expected_subjects": len(expected_ids),
            "manifest_rows": len(manifest),
            "clinical_rows": len(clinical),
            "ct_subject_directories": len(ct_ids),
            "manifest_id_set_matches": manifest_ids == expected_ids,
            "clinical_id_set_matches": clinical_ids == expected_ids,
            "ct_id_set_matches": ct_ids == expected_ids,
            "duplicate_manifest_ids": duplicate_manifest_ids,
            "duplicate_clinical_ids": duplicate_clinical_ids,
            "empty_manifest_ids": empty_manifest_ids,
            "empty_clinical_ids": empty_clinical_ids,
            "manifest_labels_match": bool(manifest_label_matches),
            "clinical_labels_match": bool(clinical_label_matches),
            "id_set_sha256": set_sha256(expected_ids),
            "manifest_sha256": sha256_file(manifest_path),
            "clinical_sha256": sha256_file(clinical_path),
        }
        all_manifest_ids.update(manifest_ids)
        all_clinical_ids.update(clinical_ids)
        all_ct_ids.update(ct_ids)
    expected_all = set(canonical["public_subject_id"])
    cohort_hashes = {
        cohort: set_sha256(set(canonical.loc[canonical["cohort"].eq(cohort), "public_subject_id"]))
        for cohort in ("internal", "external")
    }
    external_masks = [str(path.relative_to(root)) for path in (root / "external").rglob("masks") if path.is_dir()]
    return {
        "root_role": "technical_deidentified_release_candidate" if release else "local_reproduction_input",
        "groups": groups,
        "all_manifest_ids_match": all_manifest_ids == expected_all,
        "all_clinical_ids_match": all_clinical_ids == expected_all,
        "all_ct_ids_match": all_ct_ids == expected_all,
        "all_id_set_sha256": set_sha256(expected_all),
        "cohort_id_set_sha256": cohort_hashes,
        "external_masks_directories": external_masks,
    }


def main() -> int:
    canonical = pd.read_csv(CANONICAL, dtype=str)
    if len(canonical) != 408 or canonical["public_subject_id"].nunique() != 408:
        raise RuntimeError("Canonical crosswalk is not 408 unique subjects")
    local = audit_tree(DATA_ROOT, canonical, release=False)
    release = audit_tree(RELEASE_ROOT, canonical, release=True)
    release_summary_path = RELEASE_ROOT / "audit" / "release_manifest_summary.json"
    technical_summary_path = RELEASE_ROOT / "audit" / "deidentification_summary.json"
    release_summary = json.loads(release_summary_path.read_text(encoding="utf-8"))
    technical_summary = json.loads(technical_summary_path.read_text(encoding="utf-8"))
    all_group_checks = [
        all(
            bool(group[key])
            for key in (
                "manifest_id_set_matches",
                "clinical_id_set_matches",
                "ct_id_set_matches",
                "manifest_labels_match",
                "clinical_labels_match",
            )
        )
        and group["duplicate_manifest_ids"] == 0
        and group["duplicate_clinical_ids"] == 0
        and group["empty_manifest_ids"] == 0
        and group["empty_clinical_ids"] == 0
        for tree in (local, release)
        for group in tree["groups"].values()
    ]
    status = "PASS_IDENTITY_CONTRACT_RELEASE_GATES_PENDING" if all(all_group_checks) else "FAIL"
    result = {
        "status": status,
        "canonical_crosswalk_sha256": sha256_file(CANONICAL),
        "canonical_subjects": len(canonical),
        "local_reproduction_input": local,
        "technical_deidentified_release_candidate": release,
        "technical_deidentification": {
            "subjects": technical_summary["subjects"],
            "files": technical_summary["files"],
            "bytes": technical_summary["bytes"],
            "failure_count": technical_summary["failure_count"],
            "summary_sha256": sha256_file(technical_summary_path),
        },
        "release_status": release_summary["status"],
        "remaining_release_gates": release_summary["remaining_release_gates"],
        "release_manifest_summary_sha256": sha256_file(release_summary_path),
        "interpretation": "Public-subject ID sets and labels are unambiguous across local clinical, manifest and CT directories. This technical check does not authorise publication.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if status.startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
