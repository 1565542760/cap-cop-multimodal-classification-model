# Public Candidate Audit - 2026-10-12

## Scope

This audit was run against a temporary clean clone assembled from the current
Git candidate boundary (`git ls-files --others --exclude-standard`). The clone
contained 132 candidate files and did not contain `data/`, `models/`, or
patient-level outputs.

## Commands

```text
python -m compileall -q src scripts
python scripts/run_smoke_tests.py
python -m pytest -q tests
python scripts/check_github_release.py
```

All four commands exited with code 0; the pytest suite reported `11 passed`.
The smoke checks covered project-relative
configuration loading, paper-facing model aliases, synthetic crosswalk validation,
clinical preprocessing, semantic clinical contracts, manifest validation, fusion,
and metrics. The release gate reported:

The same sequence was additionally run from a newly initialized temporary Git
repository after `pip install --no-deps -e .`, with the clone root as the working
directory. Editable installation, compilation, pytest, smoke checks, and the release
gate all exited with code 0. The temporary editable package and clone were removed
afterward.

```text
code_package_gate=PASS
scientific_result_gate=REVIEW_REQUIRED
data_publication_gate=GOVERNED_DATA_AND_MODEL_AUTHORIZATION_REQUIRED
```

The release gate also checks that public source/configuration file names and model
registry keys use semantic names rather than internal experiment identifiers. That
check passed. Historical identifiers remain only in audit/provenance text or legacy
artifact selectors and are not exposed as public model/configuration names.

After adding this naming check, the current 132-file candidate was rebuilt in a
fresh temporary Git repository and rerun with editable installation. Compilation,
pytest (`11 passed`), smoke checks, and the release gate all exited with code 0;
the temporary repository and test dependencies were removed afterward.

## Boundary

The clean candidate is a valid public code/configuration package, not a patient-level
reproduction bundle. Exact Internal375/External33 replay requires the separately
authorized `data/` and `models/` trees. The command
`scripts/check_authorized_artifact_bundle.py` checks that bundle when it is available.
It does not publish or embed restricted files.

No commit, push, upload, ZIP, or modification of the original experiment tree was
performed during this audit.

## Local governed-artifact validation

The authorized local artifact preflight was run with the locked torch environment
and exited with code 0. It verified the 250/125/19/14 cohort contract, 20
segmentation checkpoints, 15 structural-missingness members, 15 exploratory
checkpoints, the locked A/B/C/D artifacts, and zero external mask directories.

The repository entry point `scripts/run_minimal_reproduction.ps1` was then run
with the two locked local environments and exited with code 0. It executed:

- one internal and one external DICOM through the segmentation preprocessing and
  forward-inference path, with non-empty masks and the locked threshold `0.5`;
- one internal DICOM plus predicted mask through PyRadiomics extraction of the
  13-feature RF contract;
- one external DICOM through the P95 `96^3` tensor path;
- locked A, B, and C component checks and D artifact checks;
- one external DICOM through the supplementary exploratory DICOM-to-model-to-fusion
  path.

This is an actual local integration run, but it remains a single-case/minimal
validation entry point. It is not evidence that all historical paper metrics are
reproduced, and it does not resolve the documented A/C historical identity-bridge
issue or the governed data-publication decision.

The same minimal entry point was rerun with `-MainPython` and `-RadiomicsPython`
explicitly pointing to the two locked environments rather than using the default
project-relative locations; it also exited with code 0. This verifies that the
public entry point can use externally located compatible environments without
changing the replay contract.
