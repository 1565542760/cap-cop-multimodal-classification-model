from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FULL_STRUCTURED_CLINICAL_VARIABLES = [
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


@dataclass
class ClinicalPreprocessor:
    feature_names: list[str]

    numeric_features = [
        "age",
        "height_cm",
        "weight_kg",
        "body_mass_index",
        "lymphocyte_count",
        "erythrocyte_sedimentation_rate",
        "c_reactive_protein",
        "high_sensitivity_c_reactive_protein",
        "procalcitonin",
    ]
    categorical_features = [
        "fever",
        "cough",
        "sputum_category",
        "chest_tightness",
        "dyspnoea",
        "haemoptysis",
        "weight_loss",
    ]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in self.feature_names if col not in frame.columns]
        if missing:
            raise ValueError(f"Clinical input missing columns: {missing}")
        transformed = frame.loc[:, self.feature_names].copy()
        for column in self.numeric_features:
            transformed[column] = pd.to_numeric(transformed[column], errors="coerce")
        for column in ["height_cm", "weight_kg", "body_mass_index"]:
            transformed.loc[transformed[column] == 0, column] = pd.NA
        recompute = (
            transformed["body_mass_index"].isna()
            & transformed["height_cm"].notna()
            & transformed["weight_kg"].notna()
            & (transformed["height_cm"] > 0)
            & (transformed["weight_kg"] > 0)
        )
        transformed.loc[recompute, "body_mass_index"] = (
            transformed.loc[recompute, "weight_kg"] / (transformed.loc[recompute, "height_cm"] / 100.0) ** 2
        )
        for column in self.categorical_features:
            transformed[column] = pd.to_numeric(transformed[column], errors="coerce")
        return transformed

    def validate_no_identifier_columns(self) -> None:
        forbidden = {
            "case_id_public",
            "patient_group_id_public",
            "patient_id",
            "label",
            "disease_label",
            "diagnosis_label",
            "anthropometric_anomaly_flag",
        }
        leaked = sorted(forbidden.intersection(self.feature_names))
        if leaked:
            raise ValueError(f"Identifier or label columns are configured as features: {leaked}")


def full_structured_clinical_preprocessor() -> ClinicalPreprocessor:
    preprocessor = ClinicalPreprocessor(FULL_STRUCTURED_CLINICAL_VARIABLES)
    preprocessor.validate_no_identifier_columns()
    return preprocessor
