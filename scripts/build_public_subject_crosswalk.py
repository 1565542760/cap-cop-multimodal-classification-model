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


def canonical_set_hash(values: pd.Series) -> str:
    payload = "\n".join(sorted(values.astype(str).unique())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def load_clinical(cohort: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for diagnosis in ("CAP", "COP"):
        path = PROJECT_ROOT / "data" / cohort / diagnosis / "clinical" / "clinical_features.csv"
        frame = pd.read_csv(path)
        public_column = "case_id_public" if cohort == "internal" else "public_subject_id"
        frame = frame.rename(columns={public_column: "public_subject_id"})
        frame["cohort"] = cohort
        frame["diagnosis_label"] = diagnosis
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    internal_private = pd.read_csv(args.internal_private_manifest, dtype=str)
    external_private = pd.read_csv(args.external_private_manifest, dtype=str)
    internal_patient = pd.read_csv(args.internal_patient_manifest, dtype=str)
    whole_chest = pd.read_csv(args.whole_chest_tensor_manifest, dtype=str)
    internal_clinical = load_clinical("internal")
    external_clinical = load_clinical("external")

    whole_chest = whole_chest[["neutral_imaging_id", "case_uid"]].merge(
        internal_patient[["case_id_public", "v1_case_uid_private"]],
        left_on="case_uid",
        right_on="v1_case_uid_private",
        how="left",
        validate="one_to_one",
    )
    if len(whole_chest) != 375 or whole_chest["case_id_public"].isna().any():
        raise RuntimeError("Whole-chest tensor namespace did not map one-to-one to Internal375.")
    whole_chest = whole_chest[["case_id_public", "neutral_imaging_id"]].rename(
        columns={
            "case_id_public": "public_subject_id",
            "neutral_imaging_id": "whole_chest_imaging_model_id",
        }
    )

    internal = internal_private[
        ["public_subject_id", "patient_group_id_public", "neutral_patient_id", "source_neutral_imaging_id"]
    ].copy()
    internal = internal.merge(
        internal_clinical[["public_subject_id", "patient_group_id_public", "diagnosis_label", "label_binary"]],
        on=["public_subject_id", "patient_group_id_public"],
        how="left",
        validate="one_to_one",
    )
    internal["cohort"] = "internal"
    internal["clinical_model_id"] = internal["public_subject_id"]
    internal["lesion_imaging_model_id"] = internal["source_neutral_imaging_id"]
    internal = internal.merge(whole_chest, on="public_subject_id", how="left", validate="one_to_one")
    # Kept only for backward compatibility with pre-audit local artifacts.
    internal["imaging_model_id"] = internal["lesion_imaging_model_id"]
    internal["segmentation_model_id"] = internal["neutral_patient_id"]
    internal["external_tensor_id"] = ""

    external = external_private[["public_subject_id", "neutral_patient_id"]].copy()
    external = external.merge(
        external_clinical[["public_subject_id", "diagnosis_label", "label_binary"]],
        on="public_subject_id",
        how="left",
        validate="one_to_one",
    )
    external_crosswalk = pd.read_csv(args.external_model_crosswalk, dtype=str)
    external = external.merge(external_crosswalk, on="neutral_patient_id", how="left", validate="one_to_one")
    external["cohort"] = "external"
    external["clinical_model_id"] = external["neutral_patient_id"]
    external["lesion_imaging_model_id"] = external["neutral_external_id"]
    external["whole_chest_imaging_model_id"] = external["neutral_external_id"]
    external["imaging_model_id"] = external["lesion_imaging_model_id"]
    external["segmentation_model_id"] = external["neutral_patient_id"]
    external["external_tensor_id"] = external["neutral_external_id"]
    external["patient_group_id_public"] = ""

    columns = [
        "public_subject_id",
        "cohort",
        "diagnosis_label",
        "label_binary",
        "patient_group_id_public",
        "clinical_model_id",
        "lesion_imaging_model_id",
        "whole_chest_imaging_model_id",
        "imaging_model_id",
        "segmentation_model_id",
        "external_tensor_id",
    ]
    result = pd.concat([internal[columns], external[columns]], ignore_index=True)
    result["label_binary"] = result["label_binary"].astype(int)
    result = result.sort_values(["cohort", "diagnosis_label", "public_subject_id"]).reset_index(drop=True)

    if len(result) != 408 or result["public_subject_id"].nunique() != 408:
        raise RuntimeError("Expected 408 unique public subjects.")
    expected = {("internal", "CAP"): 250, ("internal", "COP"): 125, ("external", "CAP"): 19, ("external", "COP"): 14}
    observed = result.groupby(["cohort", "diagnosis_label"]).size().to_dict()
    if observed != expected:
        raise RuntimeError(f"Unexpected cohort composition: {observed}")
    if result[["clinical_model_id", "lesion_imaging_model_id", "whole_chest_imaging_model_id", "segmentation_model_id"]].eq("").any().any():
        raise RuntimeError("A model-namespace ID is missing from the public crosswalk.")

    audit = {
        "status": "passed",
        "rows": len(result),
        "unique_public_subject_ids": result["public_subject_id"].nunique(),
        "cohort_counts": {f"{key[0]}_{key[1]}": value for key, value in observed.items()},
        "all_public_id_set_sha256": canonical_set_hash(result["public_subject_id"]),
        "internal_public_id_set_sha256": canonical_set_hash(result.loc[result["cohort"].eq("internal"), "public_subject_id"]),
        "external_public_id_set_sha256": canonical_set_hash(result.loc[result["cohort"].eq("external"), "public_subject_id"]),
        "source_hashes": {
            "internal_private_manifest": sha256_file(Path(args.internal_private_manifest)),
            "external_private_manifest": sha256_file(Path(args.external_private_manifest)),
            "external_model_crosswalk": sha256_file(Path(args.external_model_crosswalk)),
            "internal_patient_manifest": sha256_file(Path(args.internal_patient_manifest)),
            "whole_chest_tensor_manifest": sha256_file(Path(args.whole_chest_tensor_manifest)),
        },
        "namespace_rule": "The internal migration manifest binds segmentation and lesion IDs to the clinical public subject; the whole-chest namespace is joined independently. imaging_model_id is a deprecated lesion-ID alias.",
        "privacy_boundary": "No source path or original hospital identifier is written to the public crosswalk.",
    }
    return result, audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an unambiguous de-identified subject/model namespace crosswalk.")
    parser.add_argument("--internal-private-manifest", required=True, type=Path)
    parser.add_argument("--external-private-manifest", required=True, type=Path)
    parser.add_argument("--external-model-crosswalk", required=True, type=Path)
    parser.add_argument("--internal-patient-manifest", required=True, type=Path)
    parser.add_argument("--whole-chest-tensor-manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "public_subject_model_crosswalk.csv")
    parser.add_argument("--audit-output", type=Path, default=PROJECT_ROOT / "outputs" / "identity_audit" / "public_subject_crosswalk_audit.json")
    args = parser.parse_args()
    frame, audit = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    audit["output_sha256"] = sha256_file(args.output)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
