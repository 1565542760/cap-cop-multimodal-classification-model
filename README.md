# Multimodal Clinical CT Classification of CAP and COP

This repository contains the code associated with the study of patient-level
classification between community-acquired pneumonia (CAP) and cryptogenic
organising pneumonia (COP) using pretreatment structured clinical information
and chest CT.

The software is provided for research and method-reproduction purposes. It is
not a clinical diagnostic device and must not be used for patient-care decisions.

## Study Workflows

The code follows the workflows reported in the manuscript and supplementary
materials:

- a structured clinical-information branch using 16 pretreatment variables;
- a lesion-region CT branch using predicted lesion masks and PyRadiomics;
- a full-chest, full-field-of-view three-dimensional CT branch;
- a stable multi-representation chest CT branch;
- clinical-only, imaging-only, and clinical-imaging multimodal models;
- patient-level probability fusion and evaluation in internal and external
  cohorts.

The lesion-region workflow is:

```text
DICOM CT -> three-dimensional lesion segmentation -> predicted lesion mask
         -> lesion-region PyRadiomics -> imaging probability
         -> clinical-imaging probability fusion
```

Internal performance is evaluated with patient-level out-of-fold predictions.
External performance is evaluated after model freezing. Classification measures
use the study's fixed threshold, and discrimination is evaluated from continuous
patient-level probabilities.

## Repository Layout

```text
configs/                 Model and preprocessing configuration
data/                    Local input data, not versioned
docs/                    Method and usage documentation
models/                  Local model artifacts, not versioned
outputs/                 Local runtime outputs, not versioned
scripts/                 Workflow entry points
src/cap_cop/             Python package
tests/                   Lightweight code tests
```

## Installation

Create a Python environment and install the package:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

The imaging workflows require the packages specified by the supplied
requirements files, including the relevant versions of PyTorch, SimpleITK,
PyRadiomics, NumPy, pandas, and scikit-learn.

## Code Check

The repository includes a lightweight check for imports, configuration loading,
preprocessing interfaces, probability fusion, and metric calculation. It does
not train models or require patient-level data.

```powershell
$ProjectRoot = (Get-Location).Path
$env:PYTHONPATH = "$ProjectRoot\src"
python .\scripts\run_smoke_tests.py
python -m pytest
```

Use the `--help` option on the workflow scripts to view their input and output
arguments.

## Data and Model Artifacts

Patient-level clinical tables, labels, CT images, lesion masks, trained model
weights, and generated predictions are not included in this repository. They
must be accessed only under the applicable institutional and data-sharing
requirements.

The repository can be inspected and its lightweight checks can be run without
patient-level data. Full patient-level evaluation requires the corresponding
authorised input data and model artifacts described by the study materials.

Do not place names, hospital numbers, original identifiers, private identifier
mappings, or unrestricted clinical records in this repository.

## Output and Reporting

Runtime outputs should be written under `outputs/`, which is excluded from
version control. When reproducing study results, record the software environment,
input data version, model-artifact checksums, cohort definition, and command
used.

## Citation

When using this code, cite the associated CAP/COP study and identify the version
of this repository used for the analysis.

## License

Licensing terms for this repository and third-party components should be
confirmed by the authors before redistribution.
