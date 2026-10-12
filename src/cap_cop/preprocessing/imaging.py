from __future__ import annotations

from pathlib import Path


def require_existing_inputs(paths: list[str | Path]) -> list[Path]:
    resolved = [Path(p) for p in paths]
    missing = [str(p) for p in resolved if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing imaging inputs: {missing}")
    return resolved


def describe_imaging_strategy(name: str) -> str:
    descriptions = {
        "lesion_roi_radiomics": "Lesion-ROI PyRadiomics features extracted with fixed settings.",
        "whole_chest_ct": "Full-field chest CT comparator branch.",
        "stable_multirepresentation_ct": "Stable multi-representation CT branch.",
        "exploratory_comparator_imaging": "Supplementary exploratory imaging branch.",
    }
    return descriptions.get(name, "Unconfirmed imaging strategy.")


def require_mask_for_internal(mask_path: str | Path | None, *, strategy: str) -> Path:
    if not mask_path:
        raise FileNotFoundError(f"{strategy} requires an internal lesion mask/ROI, but mask_path is empty.")
    path = Path(mask_path)
    if not path.exists():
        raise FileNotFoundError(f"{strategy} requires an internal lesion mask/ROI, but this path is missing: {path}")
    return path


def external_mask_policy(strategy: str) -> str:
    if strategy == "lesion_roi_radiomics":
        return "blocked_if_locked_flow_requires_external_mask"
    return "not_required_for_current_local_smoke_test"
