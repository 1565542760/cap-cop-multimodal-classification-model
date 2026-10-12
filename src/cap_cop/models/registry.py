from __future__ import annotations

from dataclasses import dataclass

from cap_cop.config import ModelConfig, load_model_configs


@dataclass(frozen=True)
class BranchImplementationStatus:
    name: str
    status: str
    notes: str


CLINICAL_BRANCH_STATUS = {
    "full_structured_clinical": BranchImplementationStatus(
        "full_structured_clinical",
        "full_cohort_replay_exact_to_tolerance",
        "The 16-variable fold-fitted clinical workflow and copied locked artifact reproduce Internal375 and External33 probabilities to floating-point tolerance.",
    ),
    "structural_missingness_clinical": BranchImplementationStatus(
        "structural_missingness_clinical",
        "full_cohort_replay_exact_to_tolerance",
        "The artifact-backed structural-missingness branch and external inference workflow were replayed for 375/33 subjects.",
    ),
    "core_clinical": BranchImplementationStatus(
        "core_clinical",
        "full_cohort_replay_exact_to_tolerance",
        "The seven-variable core contract and nested replay were confirmed against locked probabilities.",
    ),
    "availability_gated_clinical": BranchImplementationStatus(
        "availability_gated_clinical",
        "full_cohort_replay_exact_to_tolerance",
        "The full16/core7 availability gate and threshold of six available full-only variables were replayed against the lock.",
    ),
    "alternate_clinical_configuration": BranchImplementationStatus(
        "alternate_clinical_configuration",
        "full_cohort_replay_exact_to_tolerance_with_registry_name_caveat",
        "Internal strict-family replay uses 12 RF, two elastic-net, and one gradient-boosting outer models; external deployment is a fixed RF ensemble.",
    ),
}

IMAGING_BRANCH_STATUS = {
    "lesion_roi_radiomics": BranchImplementationStatus(
        "lesion_roi_radiomics",
        "raw_chain_uses_canonical_identity; corrected_result_lock_pending",
        "The public lesion chain uses the canonical segmentation-subject mapping. Historical fusion outputs are retained only as audit evidence until the corrected patient-level replay is locked.",
    ),
    "whole_chest_ct": BranchImplementationStatus(
        "whole_chest_ct",
        "full_cohort_raw_dicom_replay_exact_to_tolerance",
        "The P95 tensor, frozen TANGERINE embedding, linear-probe and fusion chain ran for 375/33 subjects.",
    ),
    "stable_multirepresentation_ct": BranchImplementationStatus(
        "stable_multirepresentation_ct",
        "canonical_identity_replay_completed_result_lock_pending",
        "The public SP6 workflow uses branch-specific canonical IDs. The raw lesion-phenotype replay is complete; historical internal joins are not used as the public result standard.",
    ),
    "exploratory_comparator_imaging": BranchImplementationStatus(
        "exploratory_comparator_imaging",
        "full_cohort_replay_completed_exploratory",
        "The exploratory DenseNet branch replayed for 375/33 subjects; retain its supplementary exploratory status.",
    ),
}


def model_status_table() -> list[dict[str, str]]:
    rows = []
    for config in load_model_configs():
        clinical = CLINICAL_BRANCH_STATUS[config.clinical_strategy]
        imaging = IMAGING_BRANCH_STATUS[config.imaging_strategy]
        replay_pending = "replay_required" in clinical.status or "replay_pending" in imaging.status
        lock_pending = "lock_pending" in clinical.status or "lock_pending" in imaging.status
        rows.append(
            {
                "model": config.name,
                "clinical_strategy": config.clinical_strategy,
                "clinical_status": clinical.status,
                "imaging_strategy": config.imaging_strategy,
                "imaging_status": imaging.status,
                "multimodal_status": (
                    "canonical_replay_pending"
                    if replay_pending
                    else "full_cohort_replay_available_result_lock_pending"
                    if lock_pending
                    else "full_cohort_replay_available"
                ),
                "weights": f"{config.clinical_weight:.2f}/{config.imaging_weight:.2f}",
                "threshold": f"{config.threshold:.2f}",
            }
        )
    return rows


def require_model_config(name: str) -> ModelConfig:
    for config in load_model_configs():
        if config.name == name:
            return config
    raise KeyError(f"Unknown model config: {name}")
