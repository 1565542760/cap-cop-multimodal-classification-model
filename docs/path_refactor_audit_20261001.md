# Path Refactor Audit - 2026-10-01

This audit separates runnable, locally adapted entrypoints from copied historical source snapshots that still preserve original experiment-tree paths for provenance.

## Runnable Local Entrypoints Checked

- `scripts/run_p95_tensor_single_case.py`
  - Uses `--project-root` and defaults to the local project root.
  - Reads DICOM from `data/*/*/manifest.csv`.
  - Writes temporary tensors under `outputs/p95_tensor_replay`.
  - Tested for `CAP_0001` and `EXT_CAP_001`.

- `src/cap_cop/imaging/whole_chest_tensor/generate_whole_chest_tensor.py`
  - Uses local manifests and local DICOM paths.
  - Calls recovered tensor-contract functions but does not read from `<private-experiment-root>`.

- `scripts/verify_lesion_radiomics_rf_locked.py`
  - Reads copied local locked feature and model artifacts under `models/lesion_radiomics`.
  - Does not read live files from `<private-experiment-root>`.

- `scripts/verify_whole_chest_ct_locked_linear_probe.py`
  - Reads copied local embedding and prediction artifacts under `models/whole_chest_ct`.
  - Does not read live files from `<private-experiment-root>`.

- `scripts/verify_stable_multirepresentation_locked_tables.py`
  - Reads copied local stable-multirepresentation lock tables under `models/stable_multirepresentation_ct`.
  - Does not read live files from `<private-experiment-root>`.

## Historical Source Snapshots With Original Paths

The following copied files still contain `<private-experiment-root>` constants. They are retained as source-provenance snapshots or compatibility sources, not as standalone runnable public entrypoints unless separately wrapped or refactored.

- `src/cap_cop/imaging/whole_chest_tensor/whole_chest_tensor_contract_source.py`
- `src/cap_cop/imaging/whole_chest_tensor/whole_chest_tensor_materialization_source.py`
- `src/cap_cop/pipelines/lesion_roi_pyradiomics_pipeline.py`
- `src/cap_cop/pipelines/whole_chest_ct_foundation_transfer_pipeline.py`
- `src/cap_cop/pipelines/stable_multirepresentation_ct_pipeline.py`
- `src/cap_cop/pipelines/locked_weighted_multimodal_fusion_pipeline.py`
- `src/cap_cop/pipelines/exploratory_full_chest_densenet_pipeline.py`
- `src/cap_cop/pipelines/structural_missingness_clinical_artifact_recovery.py`
- `src/cap_cop/pipelines/structural_missingness_clinical_external_generalization.py`
- `src/cap_cop/pipelines/structural_missingness_lesion_radiomics_exploratory_fusion.py`

These files should not be used to claim standalone execution from `<project-root>` until a wrapper or refactored copy has been tested. Keeping the original path literals in these provenance snapshots helps preserve exact historical context and prevents silent algorithm drift.

## Current Boundary

The tested local commands no longer require live runtime reads from
`<private-experiment-root>`. Full A/B/C/D workflow run entry points have now been run
from the public workspace inputs and local governed model artifacts. The original
experiment tree remains necessary only as the authoritative archive for provenance,
original run evidence, private ID mappings, and any runtime intermediates that are
not part of the public workspace; it is not a live runtime dependency of the tested
public entry points.
# 2026-10-11 Final Runtime Path Pass

The public/local runtime tree was rescanned after full-cohort workflow run. All `<private-experiment-root>` literals were removed from `scripts`, `src`, and `configs`. The migrated semantic pipeline copies now default to the current project root and accept only project-root configuration; no runtime entry point requires a live read from the original experiment tree. These edits change only path/configuration text and environment reporting; no algorithm, feature, split, seed, model parameter, weight, threshold, or metric calculation was changed.

Current SHA-256 values after path-only adaptation:

- `exploratory_full_chest_densenet_pipeline.py`: `140369E3F996FA1082BD0B4BCC894554AF800837BD0E8C4F46774B5A81158F4F`
- `lesion_roi_pyradiomics_pipeline.py`: `B687F9C41C021AE5F7FC6048CD06432FDA350CEFC1E0E5B56E28CE47D2C059DD`
- `locked_weighted_multimodal_fusion_pipeline.py`: `C92EAD8970F0DB126E073BAD400E3AC834995EEA5A1F20671E29AF1DBDFF8582`
- `stable_multirepresentation_ct_pipeline.py`: `73B6962A55D289D5814750EA5F1B41F19A0741F82A42282C1BCA90D4635B0A26`
- `structural_missingness_lesion_radiomics_exploratory_fusion.py`: `EE3B8D191CE09CC1FDD3464764792F1DAB2B0A6E8325CF394C1662E8A2B5A430`
- `whole_chest_ct_foundation_transfer_pipeline.py`: `47C2ECB50867ED91BED163C26EE6135115DA85F5CA4CD277E2D1E4E0CABABAC9`
- `audit_locked_delong.py`: `7B9AFB6774CE3E533846BFC079E69A5F8ABDFCBE9E3DC35852D8FCEDCE32999C`

Final static scan result: zero `3D_CT` references in runtime Python, PowerShell, JSON, YAML, or YML files.

