# Public Release Safety Audit - 2026-10-11

This note separates a locally reproducible workspace from a GitHub release.

## Runtime path isolation

The static scan over `src/`, `scripts/`, and `configs/` found no runtime literal
for `<private-experiment-root>`. The tested commands were launched from outside `<project-root>` and
used project-relative defaults or explicit `<project-root>` paths. Historical lock
files and audit notes may still record the original experiment location; those
records are provenance, not runtime inputs.

## Local-only records

The following classes are intentionally ignored by Git because they can contain
private source paths, patient-level IDs, acquisition metadata, or migration
history:

- `docs/ci_audit/`
- `docs/migration_manifests/`
- `docs/full_reproduction_evidence_20261011/` and the local Supplementary Exploratory Multimodal Comparator provenance note,
  because these aggregate records retain internal pair selectors
- historical migration/workflow execution acceptance notes listed in `.gitignore`
- case-level model manifests and acquisition manifests under `models/`
- `data/`, `outputs/`, checkpoints, weights, predictions, and caches

These files remain available for local scientific audit. Their omission from a
GitHub commit does not mean the associated locked experiment evidence is absent;
it means that evidence must be shared separately under the applicable data and
privacy controls.

## Release boundary

Before publishing, commit only semantic source code, non-sensitive configuration,
public documentation, and explicitly approved aggregate/reference artifacts.
Do not force-add ignored patient-level data, DICOM, masks, labels, checkpoints,
or private provenance records.

The following local scripts are also outside the GitHub candidate set because
they directly consume governed patient-level lock tables or post-hoc audit
inputs: `audit_historical_imaging_identity_mapping.py`,
`audit_locked_delong.py`, `compute_locked_prediction_ci.py`, and
`run_supplementary_exploratory_subset_replay.py`. The public semantic workflow run
entry points remain separate from these controlled audits.

