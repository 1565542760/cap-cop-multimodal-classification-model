# CAP/COP Local Reproducibility Workspace

## Scientific Status

Full-cohort replay entry points have been executed for the clinical, segmentation, radiomics, whole-chest, stable-representation, exploratory imaging, and fusion chains. This does **not** mean every historical paper result passed patient-level validation.

The public workflow uses the canonical subject mapping: lesion predictions are joined by
the segmentation subject ID, while whole-chest predictions use their own whole-chest
model ID. The historical V3.11/V3.13 crosswalk conflict is retained only as private audit
evidence; it is not a runtime option and must not be used to justify paper results.
The canonical replay separates two claims: external lock validation and corrected internal
identity replay. The historical V3.13 fusion lock used a whole-chest-to-case bridge for
lesion predictions; the public workspace keeps the native lesion/CT bridge. Therefore an
internal probability difference from the historical lock is expected and must be reported,
not silently treated as an exact historical reproduction.

This directory is a local, host-only reproducibility workspace for the CAP/COP study.
Public code and configs use paths relative to this directory. Patient-level data, CT,
masks, labels, generated outputs, weights, caches and logs are ignored by Git.

## Directory Layout

- `configs/`: semantic model configuration files.
- `data/`: local patient-level data for host-only testing. This directory is ignored.
- `src/cap_cop/`: importable package.
- `tests/`: smoke tests for imports, configs, manifests, preprocessing, fusion and metrics.
- `docs/`: public documentation and semantic model mapping.
- `outputs/`: local test outputs, ignored by Git.

## Running Smoke Tests

From any current directory, set the clone location once in the PowerShell session:

```powershell
$ProjectRoot = "C:\path\to\CAP_COP"
$env:PYTHONPATH = "$ProjectRoot\src"
& "$ProjectRoot\.venv_locked_torch\python.exe" "$ProjectRoot\scripts\run_smoke_tests.py"
```

This is the tested smoke-test entry point and does not require pytest. These smoke tests
do not retrain models, tune parameters, regenerate locked predictions or reproduce the
full paper metrics. They verify that code structure, configuration parsing, data-protection
boundaries and lightweight interfaces are runnable. An optional pytest runner is declared
in the project test extra; it is not required for the smoke command above.

When the governed `data/` and `models/` trees are absent, the same smoke command uses the
non-patient synthetic crosswalk at `tests/fixtures/synthetic_crosswalk.csv`. This verifies
the public code boundary without implying that restricted cohort replay is possible. Full
Internal375/External33 replay still requires the separately authorized artifact bundle.

Full-cohort replay entry points are separate scripts under `scripts/run_full_cohort_*.py`. They use the independent local locked environments at `.venv_locked_torch` and `.venv_locked_radiomics`; generated patient-level outputs remain Git-ignored. The public replay entry points use only the canonical public-subject mapping. Historical locks are audit evidence, not a public runtime mode. The older `.venv_py39_radiomics` is not an independent release environment and must not be used for replay.

The tested numerical contract for A is pinned in `requirements-radiomics-replay-tested.txt`.
The tested numerical contract for B/C is pinned in `requirements-replay-tested.txt`.
In particular, the locked linear-probe replay requires scikit-learn 1.9.0; using a
different version can change probabilities even when embeddings and folds are identical.
See `docs/replay_acceptance_20261011.md` for the current raw-DICOM and locked-input
acceptance boundaries for A-D.
The release decision and remaining GitHub-publication gates are in
`docs/github_release_gate_20261011.md`.

Before a full replay, validate an authorized local artifact bundle without printing
patient-level values:

```powershell
$env:PYTHONPATH = "$ProjectRoot\src"
$ProjectRoot\.venv_locked_torch\python.exe $ProjectRoot\scripts\check_authorized_artifact_bundle.py
```

The command checks relative data/model paths, checkpoint counts, file sizes and SHA-256
values, and verifies the cohort contract (Internal CAP250/COP125, External CAP19/COP14,
matching clinical/DICOM IDs, 16 clinical features, and CAP=0/COP=1 labels). It returns
`PASS` only when the governed bundle is complete; a public clone
without restricted artifacts returns `MISSING_AUTHORIZED_ARTIFACTS` and must not be
described as a full patient-level reproduction.

The minimal integration entry point accepts explicit interpreter paths when the
locked environments are stored outside the project clone:

```powershell
& "$ProjectRoot\scripts\run_minimal_reproduction.ps1" `
  -ProjectRoot $ProjectRoot `
  -MainPython "C:\path\to\torch\python.exe" `
  -RadiomicsPython "C:\path\to\radiomics\python.exe"
```

The supplied interpreters must match the versions documented in the replay-tested
requirements files; changing the interpreter path does not change model parameters,
weights, folds, thresholds, or locked results.

For the fixed-image clinical-input comparison, run the locked replay stages in this
order from an authorized local artifact bundle:

```powershell
$env:PYTHONPATH = "$ProjectRoot\src"
$ProjectRoot\.venv_locked_torch\python.exe $ProjectRoot\scripts\run_full_cohort_clinical_replay.py
$ProjectRoot\.venv_locked_torch\python.exe $ProjectRoot\scripts\run_full_cohort_clinical_input_comparison_replay.py
$ProjectRoot\.venv_locked_torch\python.exe $ProjectRoot\scripts\run_full_cohort_alternate_clinical_replay.py
$ProjectRoot\.venv_locked_torch\python.exe $ProjectRoot\scripts\run_full_cohort_whole_chest_representations.py `
  --phase stable_model --device cuda
$ProjectRoot\.venv_locked_torch\python.exe $ProjectRoot\scripts\run_full_cohort_clinical_strategy_multimodal_replay.py
```

These commands use the locked scikit-learn 1.9.0 environment and write patient-level
replay outputs only under the Git-ignored `outputs/` tree. The five strategies share the
SP6 image output but retain their stored clinical/image weights; this is a locked
configuration comparison, not a claim that clinical input is the only changed causal
factor. The final command checks replay probabilities against the historical fusion
locks and reports the documented internal identity-bridge warning.

The read-only code-package gate can be run from any current directory:

```powershell
$env:PYTHONPATH = "$ProjectRoot\src"
$ProjectRoot\.venv_locked_torch\python.exe $ProjectRoot\scripts\check_github_release.py
```

`code_package_gate=PASS` means the semantic code/configuration package is clean and
the required local interfaces are present. The separate scientific and data gates
remain explicit: a code gate pass does not authorize patient-level publication or
silently resolve the A/C historical identity issue.

### C lesion-phenotype geometry contract

The Internal375 raw-DICOM lesion-phenotype replay uses the deidentified locked
ROI-centre manifest at
`models/stable_multirepresentation_ct/derived_inputs/strict_lesion_roi_geometry_manifest.csv`.
This restores the historical ROI sampling geometry without changing the model
or its parameters. The feature replay matches the locked tabular features to
floating-point precision. When regenerating the upstream phenotype table, the public
replay filters it to the fixed V3.8 phenotype contract from
`models/stable_multirepresentation_ct/derived_inputs/06_LESION_PHENOTYPE_FEATURES.csv`;
additional diagnostic descriptors are not allowed to become candidate inputs. With that
contract, regenerated SP6 and C fusion replay match the locked probabilities to machine
precision. The independent historical branch-ID audit (373/375 internal subjects) is
still retained as a provenance issue for manuscript review and does not get silently
rewritten by the public runner.

### Strict locked replay modes

The lesion-radiomics entry point defaults to artifact replay:

```powershell
$env:PYTHONPATH = "$ProjectRoot\src"
$ProjectRoot\.venv_locked_radiomics\python.exe $ProjectRoot\scripts\run_full_cohort_lesion_radiomics.py `
  --phase model --model-mode locked-artifact `
  --output-dir $ProjectRoot\outputs\lesion_radiomics_locked_replay
```

`locked-artifact` loads the locked RF and locked feature/prediction artifacts without
fitting, tuning, or selecting a model. `--model-mode corrected-refit` is an explicitly
separate analysis mode and must not be described as reproducing the paper lock.

The raw A replay was run on all 408 subjects. The historical feature contract contains
six internal baseline rows marked `FAIL`; the locked RF's median imputer handles those
rows. The replay therefore records 77 failed perturbation records (the six cases across
perturbation scenarios), but no missing subject-level baseline row. These failures must
remain visible in the audit and must not be silently replaced with a new feature value.

To start the complete A chain from the local DICOM release candidate, run the following
steps in order. The commands write only to the ignored `outputs/` tree and require the
locally governed segmentation checkpoints, masks/labels contract, and PyRadiomics
environment; they do not modify the original experiment tree.

```powershell
$env:PYTHONPATH = "$ProjectRoot\src"
$ProjectRoot\.venv_locked_torch\python.exe $ProjectRoot\scripts\run_full_cohort_lesion_segmentation.py `
  --data-root $ProjectRoot\data\deidentified_release_candidate `
  --output-root $ProjectRoot\outputs\a_replay\segmentation `
  --cohort all --device cuda

$ProjectRoot\.venv_locked_radiomics\python.exe $ProjectRoot\scripts\run_full_cohort_lesion_radiomics.py `
  --output-dir $ProjectRoot\outputs\a_replay\lesion_radiomics `
  --segmentation-audit $ProjectRoot\outputs\a_replay\segmentation\full_cohort_segmentation_replay_audit.csv `
  --phase all --model-mode locked-artifact

$ProjectRoot\.venv_locked_radiomics\python.exe $ProjectRoot\scripts\reproduce_lesion_radiomics_nested_oof.py `
  --matrix $ProjectRoot\outputs\a_replay\lesion_radiomics\03_feature_extraction\full_cohort_perturbation_pyradiomics_features.csv `
  --output-dir $ProjectRoot\outputs\a_replay\nested_oof
$ProjectRoot\.venv_locked_radiomics\python.exe $ProjectRoot\scripts\verify_lesion_radiomics_nested_oof.py `
  --replay $ProjectRoot\outputs\a_replay\nested_oof\05_nested_cv\V3_5_D1_INTERNAL375_STRICT_OOF_PREDICTIONS.csv
```

The final command is a verifier, not a training shortcut: it checks the generated
375-row strict OOF file against the local locked reference. The package does not publish
patient-level DICOM, masks, labels, or checkpoints in Git; those files must be obtained
under the applicable data and model-sharing controls.

The entire local `models/` tree is governed and Git-ignored. It contains the checkpoints,
patient-level prediction/ID manifests, and historical lock artifacts used by the local
replay. A GitHub checkout therefore needs an authorized artifact bundle mounted at the
same relative `models/` paths before the full replay commands can run; the repository
does not silently substitute missing locks or claim exact historical comparison without
those artifacts.

For the primary multimodal model, use the locked component mode and select the semantic
model explicitly:

```powershell
$env:PYTHONPATH = "$ProjectRoot\src"
$ProjectRoot\.venv_locked_torch\Scripts\python.exe $ProjectRoot\scripts\run_full_cohort_multimodal_replay.py `
  --component-root $ProjectRoot\outputs\canonical_a_formal_20261011\fusion_components `
  --component-mode locked --models primary_lesion_radiomics `
  --output-dir $ProjectRoot\outputs\multimodal_primary_locked_replay
```

The command reports `PASS` only when replay probabilities match the comparison lock within
floating-point tolerance. If the canonical identity correction changes internal results,
it reports `PASS_WITH_LOCK_DIFFERENCES_REQUIRING_REVIEW`; this means the pipeline ran, not
that the historical, incorrectly bridged lock was reproduced.

Historical pair selectors are deliberately kept out of the public configuration. The
local governed file `configs/private_locked_pair_ids.json` is Git-ignored and is used
only when comparing against private historical fusion locks. A public clone must supply
its own authorized lock-selector mapping (or run the component/fusion path without a
historical-lock comparison); the semantic model configuration does not expose internal
experiment IDs.

The four paper-facing model names are accepted by the same entry point. When component
replay outputs are stored in separate local directories, run them independently with
the corresponding root (all output remains under the ignored `outputs/` tree):

```powershell
# A: lesion-ROI radiomics
$ProjectRoot\.venv_locked_torch\python.exe $ProjectRoot\scripts\run_full_cohort_multimodal_replay.py `
  --component-root $ProjectRoot\outputs\canonical_a_formal_20261011\fusion_components `
  --component-mode locked --models paper_model_a_primary_lesion_radiomics

# B: whole-chest CT
$ProjectRoot\.venv_locked_torch\python.exe $ProjectRoot\scripts\run_full_cohort_multimodal_replay.py `
  --component-root $ProjectRoot\outputs\b_multimodal_components_20261011 `
  --component-mode regenerated --models paper_model_b_whole_chest_ct_comparator

# C: stable multi-representation CT
$ProjectRoot\.venv_locked_torch\python.exe $ProjectRoot\scripts\run_full_cohort_multimodal_replay.py `
  --component-root $ProjectRoot\outputs\c_locked_multimodal_components_20261011 `
  --component-mode locked --models paper_model_c_stable_multirepresentation_ct

# D: supplementary exploratory comparator; its clinical replay is a flat root
$ProjectRoot\.venv_locked_torch\python.exe $ProjectRoot\scripts\run_full_cohort_multimodal_replay.py `
  --component-root $ProjectRoot\outputs\d_exploratory_full_replay_20261011 `
  --clinical-root $ProjectRoot\outputs\validation_full_clinical_replay_20261011 `
  --component-mode regenerated --models paper_model_d_supplementary_exploratory_comparator
```

The replay entry point also accepts the shorter legacy semantic aliases, but writes the
paper-facing names in its summary. It resolves both structured (`clinical/`) and flat
component directories; this is only a path-layout convenience and does not alter model
weights, thresholds, splits, or prediction values.

## Lesion Segmentation In The Model A Imaging Flow

Model A uses a lesion-ROI imaging chain:

```text
DICOM -> lesion segmentation U-Net -> predicted mask -> lesion-ROI PyRadiomics -> RF probability -> clinical/imaging fusion
```

The U-Net code lives under `src/cap_cop/imaging/segmentation/`.
The locked 20 segmentation checkpoints are local-only files under
`models/segmentation/locked_checkpoints/` and are ignored by Git. Patient-level
segmentation manifests are also local-only and ignored. Do not commit or publish them.

Internal inference uses the strict out-of-fold rule: for each held-out internal patient,
the code verifies that the patient is in the relevant outer test fold and not in the
training or validation patients for the four inner segmentation models used to generate
the mask. External inference uses the 20 locked internal-trained checkpoints, averages
probabilities with equal weight, and thresholds at `0.5`.

Lightweight segmentation smoke test:

```powershell
$env:PYTHONPATH = "$ProjectRoot\src"
$ProjectRoot\.venv_locked_torch\python.exe $ProjectRoot\scripts\run_segmentation_smoke.py
```

That smoke test intentionally uses one checkpoint per internal/external case so it does
not claim to reproduce the full locked 4-model internal ensemble or 20-model external
ensemble. Omit the `--max-models` argument in the module CLIs only when intentionally
running the full locked ensemble.

The complete A-branch strict internal OOF training/selection replay is launched from
the public PyRadiomics matrix with:

```powershell
$env:PYTHONPATH = "$ProjectRoot\src"
$ProjectRoot\.venv_locked_radiomics\python.exe $ProjectRoot\scripts\reproduce_lesion_radiomics_nested_oof.py
$ProjectRoot\.venv_locked_radiomics\python.exe $ProjectRoot\scripts\verify_lesion_radiomics_nested_oof.py
```

The first command reruns the locked 5-outer/4-inner internal-only procedure; it does
not read External33 labels. The second command checks the generated 375-row OOF
probabilities against the local locked reference at floating-point tolerance. Both
commands write only under `outputs/`, which is ignored by Git.

## Whole-Chest Tensor Input For Models B/C

The whole-chest CT branches use the locked P95 96-cubed tensor contract:

```text
DICOM -> LPS orientation -> HU body-envelope centre -> 420 x 300 x 380 mm FOV
      -> single-step SimpleITK linear resampling -> HU [-1000,400] fixed min-max
      -> float32 [1,96,96,96] tensor
```

The recovered source snapshots live under
`src/cap_cop/imaging/whole_chest_tensor/`. The runnable local wrapper is:

```powershell
$env:PYTHONPATH = "$ProjectRoot\src"
$ProjectRoot\.venv_locked_torch\python.exe $ProjectRoot\scripts\run_p95_tensor_single_case.py --project-root $ProjectRoot --subject-id EXT_CAP_001
```

Generated tensors are runtime intermediates and must be returned to the private runtime
archive after validation. They are not retained in `$ProjectRoot` as public inputs.

## Data Protection

The current CT data are not approved for public release. Keep `data/` local. Do not
commit it, zip it, upload it or publish it. The local
`data/deidentified_release_candidate` tree contains the candidate Internal375 and
External33 original DICOM inputs used for host-side replay; its technical manifest still
requires independent burned-in-pixel review and hospital/ethics authorization. External
`.npy` files, when present, are derived intermediate arrays for checksum comparison, not
original CT input. The external source contained 66 readable series; the local candidate
manifest retains one canonical series per External33 case and does not use the duplicate
source-version series. External cohort directories intentionally do not include masks.
If a future locked external imaging branch is proven to require masks, that branch must
be marked blocked until the input contract is resolved.


