# CAP/COP Multimodal Classification

This repository provides the model implementations and reproduction workflows
for patient-level classification between community-acquired pneumonia (CAP)
and cryptogenic organising pneumonia (COP) using structured clinical
information and chest CT.

## Models

The repository includes the following model workflows:

- **Full Structured Clinical--Lesion-ROI Radiomics**: structured clinical
  information fused with lesion-ROI PyRadiomics features.
- **Full Structured Clinical--Whole-Chest CT**: structured clinical
  information fused with a whole-chest, full-field-of-view three-dimensional
  CT representation.
- **Full Structured Clinical--Stable Multi-Representation CT**: structured
  clinical information fused with a stable multi-representation CT branch.
- **Supplementary Exploratory Multimodal Comparator**: an exploratory
  clinical--imaging comparator reported separately from the main workflows.
- **Clinical Input Availability Sensitivity**: fixed lesion-ROI radiomics
  probabilities with predefined clinical-input availability scenarios,
  including full clinical input, core clinical input, feature-specific
  unavailability, inflammatory laboratory-panel unavailability, and
  external-pattern missingness.

The lesion-ROI workflow is:

```text
DICOM CT
  -> lesion segmentation
  -> predicted lesion mask
  -> lesion-ROI PyRadiomics features
  -> imaging classifier
  -> clinical--imaging probability fusion
```

The whole-chest workflow is:

```text
DICOM CT
  -> whole-chest preprocessing and representation generation
  -> imaging classifier
  -> clinical--imaging probability fusion
```

## Reproduction Workflow

### 1. Install

From the repository root, create an environment for the package and tests:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

For the lesion-ROI PyRadiomics workflow, use an environment with:

```powershell
python -m pip install -r requirements-radiomics-replay-tested.txt
```

For the whole-chest CT and representation workflows, use an environment with:

```powershell
python -m pip install -r requirements-replay-tested.txt
```

The two replay requirements files describe separate replay environments and
should not be installed into the same environment when their pinned package
versions conflict.

### Required Python environments

The workflows require the following environments:

| Workflow | Python | Dependency specification |
|---|---:|---|
| Clinical models, multimodal fusion, and clinical-input availability sensitivity | 3.12.14 | `pyproject.toml` and the test environment |
| Lesion segmentation, whole-chest CT, and B/C representation replay | 3.11.16 | `requirements-replay-tested.txt` |
| Lesion-ROI PyRadiomics and A replay | 3.9.23 | `requirements-radiomics-replay-tested.txt` |

The clinical-input availability sensitivity workflow requires NumPy 2.5.3,
pandas 3.0.6, and scikit-learn 1.9.1. The separate replay environments use
the package combinations specified for the imaging workflows.

The workflows are kept in separate environments because the tested imaging
package combinations include compiled dependencies and serialized model
artifacts. A single compatible environment may be possible, but the workflows
are specified with the separate environments above.

### 2. Configure the project root

Run commands from the repository root and expose the local package:

```powershell
$ProjectRoot = (Get-Location).Path
$env:PYTHONPATH = "$ProjectRoot\src"
```

All workflow scripts resolve paths relative to the repository root or accept
paths through command-line arguments.

### 3. Check the implementation

Run the lightweight implementation checks:

```powershell
python .\scripts\run_smoke_tests.py
python -m pytest
```

These checks cover package imports, configuration loading, preprocessing
interfaces, probability fusion, and metric calculation.

### 4. Run the model workflows

Use the workflow entry points in the following order:

```text
clinical table and cohort manifest
  -> clinical preprocessing and clinical model

DICOM series
  -> segmentation or CT representation preprocessing
  -> imaging model

clinical probability + imaging probability
  -> fixed-weight multimodal fusion
  -> patient-level evaluation
```

The main entry points are:

```text
scripts/run_full_cohort_lesion_segmentation.py
scripts/run_full_cohort_lesion_radiomics.py
scripts/run_full_cohort_whole_chest_representations.py
scripts/run_full_cohort_multimodal_replay.py
scripts/run_clinical_input_availability_sensitivity.py
```

Inspect the available arguments before running a workflow:

```powershell
python .\scripts\run_full_cohort_lesion_segmentation.py --help
python .\scripts\run_full_cohort_lesion_radiomics.py --help
python .\scripts\run_full_cohort_whole_chest_representations.py --help
python .\scripts\run_full_cohort_multimodal_replay.py --help
python .\scripts\run_clinical_input_availability_sensitivity.py --help
```

### 5. Clinical-input availability scenarios

The clinical-input sensitivity workflow is configured in:

```text
configs/clinical_input_availability_scenarios.json
```

Run it with:

```powershell
python .\scripts\run_clinical_input_availability_sensitivity.py `
  --config .\configs\clinical_input_availability_scenarios.json `
  --output-dir .\outputs\clinical_input_availability_sensitivity
```

The workflow keeps the lesion-ROI imaging probabilities fixed and evaluates
the configured clinical-input scenarios using the same preprocessing,
cross-validation, fusion, and threshold settings defined in the configuration.

### 6. Reproduction outputs

Workflow outputs are written to the selected output directory. The evaluation
outputs contain patient-level probabilities, classification metrics, scenario
registries, and run metadata for the corresponding workflow.
