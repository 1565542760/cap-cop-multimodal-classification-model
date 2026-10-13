# Strict Workflow run Status - 2026-10-11

This record documents the current public-workspace workflow run boundary. It is a local
workflow execution record, not a claim that the historical paper lock is scientifically correct.

## Changes in the public workspace

- `scripts/run_full_cohort_lesion_radiomics.py` now defaults to
  `--model-mode locked-artifact`. This mode loads the locked RF and locked feature/prediction
  artifacts and does not fit, tune, or select a model. `corrected-refit` remains an explicit,
  separate analysis mode.
- `scripts/run_full_cohort_multimodal_replay.py` supports `--models` so one semantic model can
  be audited without silently treating missing B-D component outputs as a pass.
- Locked lesion imaging components are read from `models/lesion_radiomics/` in locked mode,
  rather than from an older regenerated workflow run directory.
- The multimodal command reports
  `PASS_WITH_LOCK_DIFFERENCES_REQUIRING_REVIEW` when canonical workflow run probabilities differ from
  the historical fusion lock by more than `1e-12`.

## Identity evidence

The public canonical crosswalk is:

`<project-root>\data\public_subject_model_crosswalk.csv`

SHA-256: `167742F880DA7DF67E4CD7AEC159655672A4493910A74262106F75876D2AC2ED`

The historical identity audit is:

`<project-root>\outputs\identity_audit\historical_internal_imaging_identity_audit.json`

The audit reports 375/375 agreement between the migration manifest and native-geometry
governance, but only 2/375 historical subject matches. It reports 373/375 historical
subject mismatches and 40/375 diagnosis mismatches. Its interpretation is that the historical
V3.11/V3.13 fusion join reused a whole-chest V2IMG-to-case crosswalk for lesion predictions.

The public I14 OOF reference used for verification is:

`models/lesion_radiomics/internal_oof_predictions.csv`

SHA-256: `0CD32B974B274D70D1CD17FBE8B06549620B4DE19A09EA16942151B357693D82`

When joined with the current canonical crosswalk, its patient-level probabilities do not
match the historical V3.13 fusion lock. The maximum absolute difference is
`0.6994143492428594`; with the historical bridge the maximum difference is below
`3.4e-16`. This is evidence of an identity-join defect in the historical fusion result, not
evidence that the public RF was refit.

## A primary-lesion locked component workflow run

Command, run from outside the project directory:

```powershell
$env:PYTHONPATH = "<project-root>\src"
<project-root>\.venv_py39_radiomics\Scripts\python.exe <project-root>\scripts\run_full_cohort_multimodal_replay.py `
  --component-root <project-root>\outputs\canonical_a_formal_20261011\fusion_components `
  --component-mode locked --models primary_lesion_radiomics `
  --output-dir <project-root>\outputs\canonical_a_formal_20261011\multimodal_a_locked_canonical_v2
```

Exit code: `0`.

Summary:

`<project-root>\outputs\canonical_a_formal_20261011\multimodal_a_locked_canonical_v2\full_cohort_multimodal_replay_summary.json`

- Status: `PASS_WITH_LOCK_DIFFERENCES_REQUIRING_REVIEW`.
- Internal: 375 subjects; canonical ID-set SHA-256
  `2826CAADA919764FEB31A86C9A964B396A5AD370FC0FD82261E530FDFADF94BE`.
- Internal corrected workflow run metrics: AUROC `0.859872`, accuracy `0.792`, sensitivity `0.680`,
  specificity `0.848`, F1 `0.6854838709677419`.
- Historical internal lock metrics: AUROC `0.871296`, accuracy `0.8053333333333333`,
  sensitivity `0.640`, specificity `0.888`, F1 `0.6866952789699571`.
- Internal workflow run versus historical lock: maximum probability difference
  `0.20982430477285802`, mean difference `0.06167089485625127`.
- External: 33 subjects; ID-set SHA-256
  `FC90C915BA793FF256DA8ECBBA03DE9161CF97262DAC989A7AAB1F0A0A96C0EF`.
- External workflow run versus historical lock: maximum difference
  `1.1102230246251565e-16`, mean difference `3.1960965860421176e-17`.
- External AUROC `0.7330827067669173`, accuracy `0.696969696969697`, sensitivity `0.5714285714285714`,
  specificity `0.7894736842105263`, F1 `0.6153846153846154`.

The internal difference is an expected consequence of using the corrected canonical lesion
identity. The public code must not silently report the historical internal lock as matched.

## A strict OOF regeneration

The public command `scripts/reproduce_lesion_radiomics_nested_oof.py` reruns the
historical 5-outer/4-inner Internal375 feature-stability, redundancy, model-selection,
and outer-prediction procedure from the public raw feature matrix, crosswalk, and
acquisition metadata. External labels are not read. The companion
`scripts/verify_lesion_radiomics_nested_oof.py` checks the generated 375-row OOF file
against the public reference above. The full workflow run passed with maximum probability
delta `3.3306690738754696e-16`, mean delta `4.0264088359739013e-17`, and identical
AUROC `0.6987039999999999`.

## Lesion RF locked-artifact workflow run

The strict artifact workflow run summary is:

`<project-root>\outputs\canonical_a_formal_20261011\lesion_radiomics\full_cohort_lesion_radiomics_locked_replay_summary.json`

Status: `PASS` for feature and external prediction artifact comparison. The locked RF SHA-256
is `5CF7E8B242949E93BE3A252C84A1156E984C797622CDD080D0D17A17E654F3D0`. The comparison used
the original locked RF parameters (`max_features=sqrt`, `max_depth=5`, `min_samples_leaf=10`,
`n_estimators=500`, `class_weight=balanced`, `random_state=20260816`) and did not refit it.

## Lightweight tests

- Manual execution of all nine functions in `tests/test_smoke.py`: all passed.
- `audit_local_data_identity_contract.py`: exit code `0`; internal CAP/COP `250/125`, external
  CAP/COP `19/14`, duplicate and empty IDs `0`, all local manifest/clinical/CT ID-set checks true.
- Runtime source/config scan for `D:\\3D_CT` in Python, JSON, YAML and PowerShell under
  `src/`, `scripts/` and `configs/`: no matches.
- `pytest` was not installed in the checked environments, so the pytest runner itself was not
  executed. The test functions were called directly without changing project files.

## Workflow execution boundary

The public workspace now runs the raw A chain (DICOM to predicted lesion mask to PyRadiomics),
the locked lesion RF, and the strict nested Internal375 OOF procedure. The strict OOF file is
numerically identical to the public reference at floating-point tolerance. The canonical A
multimodal fusion can also be rerun, but it does not generate the historical Internal375 V3.13
fusion numbers because the historical lock contains an identity-join defect. The corrected
canonical result is therefore available for scientific review, but no replacement result lock
has been created without an author decision.

B has a full raw-DICOM workflow run through the P95 96^3 foundation representation and its locked
fusion. C has a full raw-DICOM workflow run through lesion-phenotype extraction and its downstream
representation; its regenerated canonical multimodal result likewise differs from the historical
lock because of the historical identity mismatch. D has a full exploratory workflow run, with a small
aggregate difference from its historical summary, and remains a supplementary exploratory
comparator. These statuses must not be collapsed into a claim that every historical paper number
is matched exactly.

