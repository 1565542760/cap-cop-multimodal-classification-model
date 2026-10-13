# Lesion Segmentation Workflow

This local workspace includes the U-Net segmentation link required by the Full
Structured Clinical--Lesion-ROI Radiomics pathway.

## Local Modules

- `src/cap_cop/imaging/segmentation/model.py`: 3D U-Net architecture and checkpoint loading.
- `src/cap_cop/imaging/segmentation/preprocess.py`: DICOM series loading, HU clipping, normalisation and 2 mm isotropic resampling.
- `src/cap_cop/imaging/segmentation/splits.py`: locked strict-OOF split validation.
- `src/cap_cop/imaging/segmentation/inference.py`: sliding-window inference, probability averaging and thresholding.
- `src/cap_cop/imaging/segmentation/predict_internal.py`: internal strict-OOF case inference.
- `src/cap_cop/imaging/segmentation/predict_external.py`: external 20-checkpoint ensemble inference.
- `src/cap_cop/imaging/segmentation/train.py`: fold-contract check; full retraining is disabled unless explicitly authorised.
- `src/cap_cop/imaging/radiomics/extract_lesion_roi_features.py`: handoff from predicted masks to PyRadiomics extraction.

## Local-Only Model Files

The 20 locked checkpoint files are expected in `models/segmentation/locked_checkpoints/`.
They are necessary for inference but are not public source code and are ignored by Git.

Patient-level split and mask manifests under `models/segmentation/manifests/` are
local-only evidence and are ignored where they contain patient-level rows.

## Inference Rules

- Internal strict OOF: use the four checkpoints from the patient's outer fold, after
  verifying the patient is in that outer test fold and absent from the corresponding
  train/validation sets.
- External: use all 20 locked internal-trained checkpoints.
- Probability aggregation: arithmetic mean.
- Mask threshold: `0.5`.
- Outputs: write masks and probability maps to `outputs/`, which is ignored by Git.

## Boundary

The single-case smoke test proves module import, config/checkpoint loading, DICOM
preprocessing and U-Net forward inference. It is not a full 375/33-case workflow
and does not regenerate paper metrics.
