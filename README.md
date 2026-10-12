# Multimodal Clinical-CT Classification of CAP and COP

This repository contains research code for multimodal classification of
community-acquired pneumonia (CAP) and cryptogenic organising pneumonia (COP)
using structured clinical information and CT-derived representations.

The software is intended for research use and method reproduction. It is not a
clinical diagnostic device and must not be used to make patient-care decisions.

## Scope

The repository provides semantic, modular implementations for:

- structured clinical-data preprocessing and classification;
- lesion-region CT radiomics;
- whole-chest CT representations;
- stable multi-representation CT features;
- lesion segmentation used by the lesion-region workflow;
- unimodal evaluation and clinical-imaging fusion;
- configuration, validation, and lightweight tests.

The public modules use descriptive names rather than internal experiment labels.

## Repository Layout

```text
configs/                 Model and preprocessing configuration
clinical/                Clinical-data documentation
cohorts/                 Cohort and input-contract documentation
data/                    Local, restricted input data (not versioned)
docs/                    Method and usage documentation
imaging/                 Imaging documentation
models/                  Local model artifacts (not versioned)
outputs/                 Local runtime outputs (not versioned)
scripts/                 Command-line entry points
src/cap_cop/             Python package
tests/                   Lightweight automated tests
```

## Installation

Create an environment with Python 3.10 or later, then install the package and
the dependencies required by the workflow you intend to run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Some imaging workflows require additional packages such as PyTorch, SimpleITK,
PyRadiomics, NumPy, pandas, and scikit-learn. Use the supplied requirements
files or the environment specification provided with the authorised data and
model artifacts.

## Quick Verification

The lightweight test suite checks package imports, configuration parsing,
identity handling, preprocessing interfaces, fusion, and evaluation utilities.
It does not train a model or require patient-level data.

```powershell
$ProjectRoot = (Get-Location).Path
$env:PYTHONPATH = "$ProjectRoot\src"
python .\scripts\run_smoke_tests.py
python -m pytest
```

## Research Workflows

The main command-line entry points are grouped by workflow:

```text
scripts/run_full_cohort_clinical_replay.py
scripts/run_full_cohort_lesion_segmentation.py
scripts/run_full_cohort_lesion_radiomics.py
scripts/run_full_cohort_whole_chest_representations.py
scripts/run_full_cohort_multimodal_replay.py
```

The corresponding Python modules are under `src/cap_cop/clinical`,
`src/cap_cop/imaging`, `src/cap_cop/models`, and `src/cap_cop/pipelines`.
Use the `--help` option on each script to view the required input and output
arguments.

The lesion-region workflow follows:

```text
DICOM CT -> lesion segmentation -> lesion region -> PyRadiomics features
         -> imaging classification -> clinical-imaging fusion
```

Whole-chest workflows read CT series, apply the configured spatial and intensity
transformations, generate the required representation, and evaluate the
corresponding imaging or multimodal classifier.

## Data and Model Artifacts

Patient-level clinical data, CT images, segmentation masks, labels, trained
weights, prediction files, and generated outputs are intentionally excluded
from version control. They must be obtained and used only under the applicable
institutional, ethical, and data-sharing approvals.

Place authorised local artifacts under the relative paths documented by the
relevant configuration and manifest files. Do not place names, hospital numbers,
original identifiers, private identifier mappings, or unrestricted clinical
records in this repository.

The repository can be used to inspect the code and run lightweight tests without
restricted artifacts. Full patient-level replay requires the corresponding
authorised input data and model artifacts.

## Reproducibility and Outputs

All runtime paths are resolved from the project root or from explicit command-line
arguments. Generated masks, features, embeddings, predictions, logs, and other
temporary files belong under `outputs/`, which is excluded from version control.

When reporting results, record the software environment, input manifest version,
model-artifact checksums, cohort definition, and command used. Do not overwrite
reference artifacts with newly generated outputs.

## Data Availability

The source code is provided for research transparency. Patient-level data and
model artifacts are not distributed in this repository because their use and
sharing are governed by institutional approval, privacy protection, and data-use
conditions. Qualified users should follow the procedures of the responsible
institution to request access.

## Citation

When using this code, cite the associated CAP/COP study and identify the version
of this repository used for the analysis.

## License

The licensing terms for this repository and any third-party components should be
confirmed by the authors before public redistribution.
