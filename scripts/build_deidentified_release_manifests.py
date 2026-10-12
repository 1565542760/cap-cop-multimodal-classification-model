from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def collection_sha256(files: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(f"{sha256_file(path)}  {relative}\n".encode("utf-8"))
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build public-ID-only manifests for the technically de-identified local release candidate."
    )
    parser.add_argument(
        "--release-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "deidentified_release_candidate",
    )
    parser.add_argument("--hash-dicom-files", action="store_true")
    args = parser.parse_args()

    release_root = args.release_root.resolve()
    crosswalk = pd.read_csv(PROJECT_ROOT / "data" / "public_subject_model_crosswalk.csv", dtype=str)
    subject_audit = pd.read_csv(release_root / "audit" / "subject_deidentification_audit.csv", dtype=str)
    crosswalk_public = crosswalk[
        ["public_subject_id", "cohort", "diagnosis_label", "label_binary"]
    ].copy()
    merged = crosswalk_public.merge(
        subject_audit[
            [
                "public_subject_id",
                "output_file_count",
                "excluded_non_image_file_count",
                "pixel_data_all_match",
                "remaining_private_tag_count",
                "remaining_direct_identifier_keywords",
                "technical_deidentification_pass",
            ]
        ],
        on="public_subject_id",
        how="left",
        validate="one_to_one",
    )
    if len(merged) != 408 or merged["public_subject_id"].nunique() != 408:
        raise RuntimeError("Release manifest input does not contain 408 unique public subjects.")
    if merged["technical_deidentification_pass"].str.lower().ne("true").any():
        raise RuntimeError("At least one subject failed technical DICOM de-identification.")

    all_rows: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        class_root = release_root / row.cohort / row.diagnosis_label
        dicom_dir = class_root / "imaging" / "ct" / row.public_subject_id / "dicom"
        clinical_path = class_root / "clinical" / "clinical_features.csv"
        files = sorted(dicom_dir.glob("*.dcm"))
        if len(files) != int(row.output_file_count):
            raise RuntimeError(
                f"DICOM count mismatch for {row.public_subject_id}: {len(files)} != {row.output_file_count}"
            )
        all_rows.append(
            {
                "public_subject_id": row.public_subject_id,
                "cohort": row.cohort,
                "diagnosis_label": row.diagnosis_label,
                "label_binary": int(row.label_binary),
                "clinical_path": clinical_path.relative_to(class_root).as_posix(),
                "ct_path": dicom_dir.relative_to(class_root).as_posix(),
                "dicom_file_count": len(files),
                "dicom_size_bytes": sum(path.stat().st_size for path in files),
                "dicom_collection_sha256": (
                    collection_sha256(files, dicom_dir) if args.hash_dicom_files else "NOT_COMPUTED"
                ),
                "excluded_non_image_file_count": int(row.excluded_non_image_file_count),
                "pixel_data_all_match_source": str(row.pixel_data_all_match).lower(),
                "remaining_private_tag_count": int(row.remaining_private_tag_count),
                "remaining_direct_identifier_keywords": str(row.remaining_direct_identifier_keywords),
                "technical_deidentification_pass": str(row.technical_deidentification_pass).lower(),
                "burned_in_annotation_review": "PENDING_INDEPENDENT_PIXEL_REVIEW",
                "data_sharing_authorisation": "PENDING_HOSPITAL_ETHICS_CONFIRMATION",
            }
        )

    all_frame = pd.DataFrame(all_rows).sort_values(["cohort", "diagnosis_label", "public_subject_id"])
    for (cohort, diagnosis), frame in all_frame.groupby(["cohort", "diagnosis_label"], sort=True):
        target = release_root / cohort / diagnosis / "manifest.csv"
        frame.to_csv(target, index=False, encoding="utf-8", lineterminator="\n")

    audit_path = release_root / "audit" / "release_manifest_all_subjects.csv"
    all_frame.to_csv(audit_path, index=False, encoding="utf-8", lineterminator="\n")
    summary = {
        "status": "TECHNICAL_RELEASE_CANDIDATE_NOT_AUTHORISED_FOR_PUBLICATION",
        "subjects": len(all_frame),
        "internal_CAP": int(((all_frame.cohort == "internal") & (all_frame.diagnosis_label == "CAP")).sum()),
        "internal_COP": int(((all_frame.cohort == "internal") & (all_frame.diagnosis_label == "COP")).sum()),
        "external_CAP": int(((all_frame.cohort == "external") & (all_frame.diagnosis_label == "CAP")).sum()),
        "external_COP": int(((all_frame.cohort == "external") & (all_frame.diagnosis_label == "COP")).sum()),
        "dicom_files": int(all_frame.dicom_file_count.sum()),
        "dicom_bytes": int(all_frame.dicom_size_bytes.sum()),
        "collection_hashes_computed": bool(args.hash_dicom_files),
        "manifest_sha256": sha256_file(audit_path),
        "identity_contract": "One stable public_subject_id is used across manifest, clinical table, CT directory, labels, and replay outputs. Private source-ID mappings are excluded.",
        "remaining_release_gates": [
            "independent pixel burned-in annotation review",
            "hospital and ethics data-sharing authorisation",
            "independent privacy review of the final publication payload",
        ],
    }
    summary_path = release_root / "audit" / "release_manifest_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
