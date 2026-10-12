from __future__ import annotations

from pathlib import Path


def require_pyradiomics() -> None:
    try:
        import radiomics  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyRadiomics is required for lesion-ROI feature extraction.") from exc


def describe_lesion_roi_input(image_path: str | Path, mask_path: str | Path) -> dict[str, str]:
    require_pyradiomics()
    return {
        "image_path": str(Path(image_path)),
        "mask_path": str(Path(mask_path)),
        "status": "ready_for_pyradiomics_extraction",
        "note": "This module is the handoff point from U-Net predicted lesion masks to lesion-ROI PyRadiomics features.",
    }
