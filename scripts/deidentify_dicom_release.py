from __future__ import annotations

import argparse
import csv
import hashlib
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pydicom
from pydicom.dataset import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIRECT_IDENTIFIER_KEYWORDS = {
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "OtherPatientIDs",
    "OtherPatientNames",
    "AccessionNumber",
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "PatientComments",
    "ImageComments",
    "DeviceSerialNumber",
    "StationName",
    "ProtocolName",
    "SeriesDescription",
}
UID_KEEP_KEYWORDS = {
    "SOPClassUID",
    "MediaStorageSOPClassUID",
    "TransferSyntaxUID",
    "ImplementationClassUID",
    "ReferencedSOPClassUID",
    "CodingSchemeUID",
    "ContextGroupExtensionCreatorUID",
    "ContextUID",
    "MappingResourceUID",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def deterministic_uid(salt: str, public_subject_id: str, original_uid: str) -> str:
    payload = f"{salt}|{public_subject_id}|{original_uid}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")
    return f"2.25.{integer}"


def is_overlay_or_curve_group(group: int) -> bool:
    return (0x5000 <= group <= 0x50FF) or (0x6000 <= group <= 0x60FF)


def clean_dataset(dataset: Dataset, public_subject_id: str, salt: str, uid_map: dict[str, str]) -> None:
    dataset.remove_private_tags()
    for element in list(dataset):
        if is_overlay_or_curve_group(element.tag.group):
            del dataset[element.tag]
            continue
        if element.VR == "SQ":
            for item in element.value:
                clean_dataset(item, public_subject_id, salt, uid_map)
            continue
        keyword = element.keyword
        if keyword in DIRECT_IDENTIFIER_KEYWORDS or element.VR in {"DA", "DT", "TM"}:
            element.value = ""
            continue
        if element.VR == "UI" and keyword not in UID_KEEP_KEYWORDS and element.value:
            original = str(element.value)
            uid_map.setdefault(original, deterministic_uid(salt, public_subject_id, original))
            element.value = uid_map[original]

    dataset.PatientName = public_subject_id
    dataset.PatientID = public_subject_id
    dataset.PatientIdentityRemoved = "YES"
    dataset.DeidentificationMethod = "Basic profile; public ID; private tags removed; dates blanked"
    dataset.StudyID = ("ST" + public_subject_id.replace("_", ""))[:16]
    if "SOPInstanceUID" in dataset and getattr(dataset, "file_meta", None) is not None:
        dataset.file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID


def audit_dataset(dataset: Dataset, expected_public_id: str) -> dict[str, Any]:
    nonempty_direct: list[str] = []
    private_count = 0
    date_time_count = 0
    overlay_curve_count = 0
    uid_values: list[str] = []
    for element in dataset.iterall():
        if element.tag.is_private:
            private_count += 1
        if is_overlay_or_curve_group(element.tag.group):
            overlay_curve_count += 1
        if element.VR in {"DA", "DT", "TM"} and str(element.value).strip():
            date_time_count += 1
        if element.keyword in DIRECT_IDENTIFIER_KEYWORDS and str(element.value).strip():
            nonempty_direct.append(element.keyword)
        if element.VR == "UI" and element.keyword not in UID_KEEP_KEYWORDS and element.value:
            uid_values.append(str(element.value))
    patient_name_ok = str(getattr(dataset, "PatientName", "")) == expected_public_id
    patient_id_ok = str(getattr(dataset, "PatientID", "")) == expected_public_id
    identity_removed_ok = str(getattr(dataset, "PatientIdentityRemoved", "")) == "YES"
    return {
        "patient_name_ok": patient_name_ok,
        "patient_id_ok": patient_id_ok,
        "identity_removed_ok": identity_removed_ok,
        "private_tag_count": private_count,
        "nonempty_direct_identifier_keywords": sorted(set(nonempty_direct)),
        "nonempty_date_time_count": date_time_count,
        "overlay_curve_count": overlay_curve_count,
        "remapped_uid_count": len(uid_values),
        "pass": patient_name_ok
        and patient_id_ok
        and identity_removed_ok
        and private_count == 0
        and not nonempty_direct
        and date_time_count == 0
        and overlay_curve_count == 0,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def subject_source_dir(project_root: Path, row: pd.Series) -> Path:
    return project_root / "data" / row["cohort"] / row["diagnosis_label"] / "imaging" / "ct" / row["public_subject_id"] / "dicom"


def subject_output_dir(output_root: Path, row: pd.Series) -> Path:
    return output_root / row["cohort"] / row["diagnosis_label"] / "imaging" / "ct" / row["public_subject_id"] / "dicom"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create and audit a technically de-identified DICOM release candidate.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--crosswalk", type=Path, default=PROJECT_ROOT / "data" / "public_subject_model_crosswalk.csv")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "deidentified_release_candidate")
    parser.add_argument("--salt-file", type=Path, required=True, help="Private salt file kept outside the public project.")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    salt = args.salt_file.read_text(encoding="utf-8").strip()
    if len(salt) < 32:
        raise ValueError("Private de-identification salt must contain at least 32 characters.")
    crosswalk = pd.read_csv(args.crosswalk, dtype=str).fillna("")
    if args.limit is not None:
        crosswalk = crosswalk.head(args.limit)

    warnings.filterwarnings("ignore", module="pydicom")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    exclusions: list[dict[str, str]] = []
    total_files = 0
    total_bytes = 0
    for subject_index, row in crosswalk.iterrows():
        public_id = row["public_subject_id"]
        source_dir = subject_source_dir(args.project_root, row)
        output_dir = subject_output_dir(args.output_root, row)
        files = sorted(path for path in source_dir.iterdir() if path.is_file())
        uid_map: dict[str, str] = {}
        subject_pass = True
        subject_pixel_match = True
        subject_private_tags = 0
        subject_direct_fields: Counter[str] = Counter()
        output_dir.mkdir(parents=True, exist_ok=True)
        for file_index, source in enumerate(files, start=1):
            destination = output_dir / f"IM{file_index:06d}.dcm"
            try:
                source_dataset = pydicom.dcmread(source, force=True)
                if "PixelData" not in source_dataset or not getattr(source_dataset, "SOPClassUID", "") or not getattr(source_dataset, "SOPInstanceUID", ""):
                    exclusions.append(
                        {
                            "public_subject_id": public_id,
                            "file_index": str(file_index),
                            "source_name_sha256": hashlib.sha256(source.name.encode("utf-8")).hexdigest().upper(),
                            "size_bytes": str(source.stat().st_size),
                            "reason": "non_image_or_missing_required_sop_metadata",
                        }
                    )
                    continue
                source_pixel_sha = hashlib.sha256(bytes(source_dataset.PixelData)).hexdigest().upper() if "PixelData" in source_dataset else ""
                if not args.audit_only and not (args.resume and destination.exists()):
                    clean_dataset(source_dataset, public_id, salt, uid_map)
                    source_dataset.save_as(destination, enforce_file_format=True)
                target = destination if destination.exists() else source
                target_dataset = pydicom.dcmread(target, force=True)
                audit = audit_dataset(target_dataset, public_id) if destination.exists() else {"pass": False}
                target_pixel_sha = hashlib.sha256(bytes(target_dataset.PixelData)).hexdigest().upper() if "PixelData" in target_dataset else ""
                pixel_match = source_pixel_sha == target_pixel_sha
                subject_pass = subject_pass and bool(audit.get("pass")) and pixel_match
                subject_pixel_match = subject_pixel_match and pixel_match
                subject_private_tags += int(audit.get("private_tag_count", 0))
                subject_direct_fields.update(audit.get("nonempty_direct_identifier_keywords", []))
                total_files += 1
                total_bytes += destination.stat().st_size if destination.exists() else source.stat().st_size
            except Exception as exc:
                subject_pass = False
                failures.append({"public_subject_id": public_id, "file_index": str(file_index), "error": f"{type(exc).__name__}: {exc}"})
        rows.append(
            {
                "public_subject_id": public_id,
                "cohort": row["cohort"],
                "diagnosis_label": row["diagnosis_label"],
                "source_file_count": len(files),
                "output_file_count": len(list(output_dir.glob("*.dcm"))),
                "excluded_non_image_file_count": sum(1 for item in exclusions if item["public_subject_id"] == public_id),
                "pixel_data_all_match": subject_pixel_match,
                "remaining_private_tag_count": subject_private_tags,
                "remaining_direct_identifier_keywords": ";".join(sorted(subject_direct_fields)),
                "technical_deidentification_pass": subject_pass,
            }
        )
        print(f"[{subject_index + 1}/{len(crosswalk)}] {public_id}: files={len(files)} pass={subject_pass}", flush=True)

    audit_dir = args.output_root / "audit"
    write_rows(audit_dir / "subject_deidentification_audit.csv", rows)
    write_rows(audit_dir / "file_failures.csv", failures)
    write_rows(audit_dir / "non_image_exclusions.csv", exclusions)
    summary = {
        "status": "passed" if rows and all(row["technical_deidentification_pass"] for row in rows) and not failures else "failed",
        "subjects": len(rows),
        "files": total_files,
        "bytes": total_bytes,
        "subject_pass_count": sum(bool(row["technical_deidentification_pass"]) for row in rows),
        "failure_count": len(failures),
        "non_image_exclusion_count": len(exclusions),
        "crosswalk_sha256": sha256_file(args.crosswalk),
        "policy_sha256": sha256_file(PROJECT_ROOT / "configs" / "deidentification_policy.json"),
        "release_boundary": "Technical DICOM de-identification only. Pixel burned-in annotation review and hospital/ethics data-sharing approval remain separate release gates.",
    }
    (audit_dir / "deidentification_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
