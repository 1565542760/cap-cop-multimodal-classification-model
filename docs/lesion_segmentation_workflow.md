# Lesion Segmentation Workflow

This workflow prepares lesion regions for the Full Structured
Clinical--Lesion-ROI Radiomics pathway.

## Local Modules

- `src/cap_cop/imaging/segmentation/model.py`: 3D U-Net architecture and checkpoint loading.
- `src/cap_cop/imaging/segmentation/preprocess.py`: DICOM series loading, HU clipping, normalisation and 2 mm isotropic resampling.
- `src/cap_cop/imaging/segmentation/splits.py`: held-out split validation.
- `src/cap_cop/imaging/segmentation/inference.py`: sliding-window inference, probability averaging and thresholding.
- `src/cap_cop/imaging/segmentation/predict_internal.py`: case inference for internal data.
- `src/cap_cop/imaging/segmentation/predict_external.py`: ensemble inference for external data.
- `src/cap_cop/imaging/segmentation/train.py`: training and split-contract utilities.
- `src/cap_cop/imaging/radiomics/extract_lesion_roi_features.py`: handoff from predicted masks to PyRadiomics extraction.

## Model Files

Checkpoint files are expected in `models/segmentation/locked_checkpoints/`.
They are runtime artifacts and are ignored by Git.

Split and mask manifests under `models/segmentation/manifests/` are ignored when
they contain patient-level rows.

## Inference Rules

- Internal held-out inference uses the checkpoints assigned to the case's split.
- External inference uses the configured ensemble of checkpoints.
- Probability aggregation: arithmetic mean.
- Mask threshold: `0.5`.
- Outputs: write masks and probability maps to `outputs/`, which is ignored by Git.

