from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FEATURES = [
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
]
DIRECT_IDENTIFIER_TOKENS = {
    "name",
    "姓名",
    "patient_id",
    "住院",
    "medical_record",
    "mrn",
    "phone",
    "电话",
    "address",
    "地址",
    "birth",
    "dob",
    "身份证",
    "id_card",
    "date",
    "日期",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def source_table(project_root: Path, cohort: str, diagnosis: str) -> Path:
    return project_root / "data" / cohort / diagnosis / "clinical" / "clinical_features.csv"


def output_table(output_root: Path, cohort: str, diagnosis: str) -> Path:
    return output_root / cohort / diagnosis / "clinical" / "clinical_features.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the minimum de-identified clinical release tables used by the models.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--crosswalk", type=Path, default=PROJECT_ROOT / "data" / "public_subject_model_crosswalk.csv")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "deidentified_release_candidate")
    args = parser.parse_args()

    crosswalk = pd.read_csv(args.crosswalk, dtype={"public_subject_id": str})
    rows: list[dict[str, object]] = []
    for cohort in ("internal", "external"):
        for diagnosis in ("CAP", "COP"):
            source = source_table(args.project_root, cohort, diagnosis)
            frame = pd.read_csv(source)
            id_column = "case_id_public" if cohort == "internal" else "public_subject_id"
            required = {id_column, "label_binary", *MODEL_FEATURES}
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(f"{source} is missing columns: {sorted(missing)}")
            release = frame[[id_column, *MODEL_FEATURES, "label_binary"]].rename(columns={id_column: "public_subject_id"})
            expected_ids = set(
                crosswalk.loc[
                    crosswalk["cohort"].eq(cohort) & crosswalk["diagnosis_label"].eq(diagnosis),
                    "public_subject_id",
                ].astype(str)
            )
            if set(release["public_subject_id"].astype(str)) != expected_ids:
                raise RuntimeError(f"Clinical/public crosswalk mismatch for {cohort}/{diagnosis}.")
            suspicious = [
                column
                for column in release.columns
                if column != "public_subject_id" and any(token in column.lower() for token in DIRECT_IDENTIFIER_TOKENS)
            ]
            if suspicious:
                raise RuntimeError(f"Potential identifier columns remain in {cohort}/{diagnosis}: {suspicious}")
            destination = output_table(args.output_root, cohort, diagnosis)
            destination.parent.mkdir(parents=True, exist_ok=True)
            release.to_csv(destination, index=False)
            rows.append(
                {
                    "cohort": cohort,
                    "diagnosis_label": diagnosis,
                    "rows": len(release),
                    "columns": len(release.columns),
                    "public_subject_id_unique": release["public_subject_id"].nunique(),
                    "source_sha256": sha256_file(source),
                    "output_sha256": sha256_file(destination),
                    "output_relative_path": destination.relative_to(args.output_root).as_posix(),
                }
            )

    audit_dir = args.output_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(audit_dir / "clinical_release_audit.csv", index=False)
    summary = {
        "status": "passed",
        "tables": len(rows),
        "rows": sum(int(row["rows"]) for row in rows),
        "feature_count": len(MODEL_FEATURES),
        "released_columns": ["public_subject_id", *MODEL_FEATURES, "label_binary"],
        "excluded_fields": [
            "patient_group_id_public",
            "disease_label",
            "anthropometric_anomaly_flag",
            "all source paths and original hospital identifiers",
        ],
        "crosswalk_sha256": sha256_file(args.crosswalk),
    }
    (audit_dir / "clinical_release_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
