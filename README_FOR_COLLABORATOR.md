# CAP/COP code package: collaborator setup

This is a code-only package. Patient-level clinical records, DICOM, masks,
crosswalks, model weights, and predictions are not included. Obtain only the
authorised input bundle for the workflow you intend to run. Never substitute
row order, diagnosis labels, or matching-looking IDs for a patient crosswalk.

## Windows PowerShell

Extract the ZIP to a new directory, then run the following **from that
directory**. Python 3.12 handles tabular tools, Python 3.9 handles
PyRadiomics and lesion phenotypes, and Python 3.11 handles Torch imaging.
Use the Python executable installed
on your own computer in place of `python` if necessary.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[test,phenotype]"
.\.venv\Scripts\python.exe scripts\source_archive_provenance.py --verify SOURCE_MANIFEST.json
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_runtime_environment.py --profile phenotype --import-check
```

The base package does not need Torch. The `phenotype` extra provides DICOM
and ROI input support without Torch or Matplotlib. For PyRadiomics, create a
**separate** Python 3.9 environment and install the radiomics extra:

Install `.[clinical]` separately before running the clinical XGBoost branch,
then check it with `scripts\check_runtime_environment.py --profile clinical --import-check`.

```powershell
py -3.9 -m venv .venv-radiomics
.\.venv-radiomics\Scripts\python.exe -m pip install "pip==23.0.1"
.\.venv-radiomics\Scripts\python.exe -m pip install -e ".[radiomics]"
.\.venv-radiomics\Scripts\python.exe scripts\check_runtime_environment.py --profile radiomics --import-check
```

The PyRadiomics 3.1 source distribution has inconsistent version metadata
under newer pip releases; the Python 3.9 / pip 23.0.1 combination above is
the one independently installed and exercised for this package. The
`radiomics` extra has not passed installation on Python 3.12. Use the separate
Python 3.9 environment for that stage; do not silently substitute another
PyRadiomics version. Load the supplied locked Model A RF joblib in that same
Python 3.9 / scikit-learn 1.6.1 environment. A Python 3.12 / scikit-learn
1.9.1 attempt raised `SimpleImputer._fill_dtype` while loading its pipeline;
the 3.12 tools environment is not a substitute for that locked artefact.

For segmentation and whole-chest representation generation, create a
separate Python 3.11 environment. Install the imaging extra and check the
actual Torch/CUDA build before using authorised checkpoints:

```powershell
py -3.11 -m venv .venv-imaging
.\.venv-imaging\Scripts\python.exe -m pip install -e ".[imaging]"
.\.venv-imaging\Scripts\python.exe scripts\check_runtime_environment.py --profile representations --import-check
.\.venv-imaging\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

`requirements-replay-tested.txt` records the tested Python 3.11 package
versions but does not include PyRadiomics, which belongs in Python 3.9.
The Torch wheel must match the collaborator's CPU/CUDA host; import success
alone does not establish encoder inference. The imaging extra pins
Torch 2.11.0, torchvision 0.26.0, timm 1.0.30 and MONAI 1.6.0;
do not silently upgrade these during a locked replay. Torch is unnecessary for
phenotype-only and tabular tests.

With all three environments and authorised inputs prepared, the single-case
cross-environment check is:

```powershell
$InternalId = Read-Host "Authorised internal public_subject_id"
$ExternalId = Read-Host "Authorised external public_subject_id"
.\scripts\run_minimal_reproduction.ps1 -InternalSubject $InternalId -ExternalSubject $ExternalId
```

Its defaults use `.venv`, `.venv-imaging` and `.venv-radiomics` under
the extracted project. Override with `-ToolPython`, `-ImagingPython` and
`-RadiomicsPython` as needed. This is a one-checkpoint segmentation smoke
and locked-artifact check, not a full-cohort replay or a certified Model C
result. It requires the matching authorised model and data bundle.

## Patient alignment and processing

Synthetic contract tests are included; they do not prove the status of a real
input bundle. With authorised local inputs in the documented project layout:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_release_contracts.py tests\test_model_c_identity.py tests\test_lesion_phenotype_mask_contract.py 2>&1 | Tee-Object .\contract_tests.log
.\.venv\Scripts\python.exe scripts\reproduce_lesion_radiomics_nested_oof.py --help
$InputRoot = Read-Host 'Enter the path to the authorised local input bundle'
.\.venv\Scripts\python.exe scripts\audit_stable_representation_inputs.py --input-root $InputRoot
.\.venv\Scripts\python.exe scripts\run_full_cohort_lesion_phenotype.py --help
```

Run the replay commands only after placing the matching authorised clinical
table, DICOM, segmentation outputs, ROI geometry manifest, and model artefacts
in their expected local locations. The phenotype workflow also requires the
authorised `data/lesion_phenotype_geometry_exceptions.json` case contract;
it is not distributed in the code-only ZIP. Use `--output-dir` to keep new results
separate from reference artefacts. Full patient-level replay is **not** proven
by the synthetic tests. The Model C historical numerical version remains
unverified; its explicit crosswalk check is a pre-fit safety gate.

If Model A reports `expected both binary classes`, first check the crosswalk
ID set and CAP/COP counts, unique baseline feature IDs, outer/inner fold
membership, and the actual label values presented to fit. Do not duplicate
patients or change folds to make the error disappear. If phenotype extraction
fails, compare the mask SHA-256 and component geometry to the ROI manifest,
then check mask shape, origin, spacing, direction, and DICOM series version.
Preserve the full traceback and package manifest. A previous package can be
distinguished by its ZIP SHA-256 and `SOURCE_MANIFEST.json`; a Git commit
alone does not identify an uncommitted release candidate.
