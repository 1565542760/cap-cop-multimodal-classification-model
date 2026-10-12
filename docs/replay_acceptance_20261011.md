# Replay Acceptance - 2026-10-11

This record is the current evidence boundary for the public workspace. It does not
replace the original locks and does not claim that a historical patient-identity join
was scientifically valid.

## C raw-DICOM replay follow-up (2026-10-11)

The C lesion-phenotype raw-DICOM replay was rerun after restoring the locked
internal lesion-ROI geometry contract. The public replay now uses the
deidentified `strict_lesion_roi_geometry_manifest.csv` for Internal375
component centres; it does not alter the segmentation mask, model, features,
weights, threshold, folds, or labels.

- Geometry manifest: `models/stable_multirepresentation_ct/derived_inputs/strict_lesion_roi_geometry_manifest.csv`
  - 1,073 rows covering 375 internal subjects; SHA-256
    `57FB1D6B96F820F7831FCEF1212BF7B005623768DBF1D9F26684F689D3B55B6C`.
- Replay summary: `outputs/c_raw_dicom_replay_20261011_v2/full_cohort_lesion_phenotype_replay_summary.json`
  - 408 subjects (Internal375/External33), status `PASS`.
  - Internal maximum absolute feature delta: `5.684341886080802e-14`.
  - External maximum absolute feature delta: `2.842170943040401e-14`.
  - Summary SHA-256: `463B6BF151380540CEEFDC760FE24193E656ACDF0955042E6012EAF9C48112D5`.

The first regenerated SP6/C replay passed the wrong tabular contract and produced
the historical-join discrepancy recorded below. The public entry point was then
corrected to filter regenerated phenotype inputs to the locked V3.8 contract
(19 phenotype columns plus metadata). After that correction, the fresh
regenerated SP6 component and C fusion matched their locked references at
machine precision; the separate 373/375 historical branch-ID audit remains a
provenance issue and must not be silently relabelled as the corrected canonical
result.

The original strict ROI tensor files referenced by the historical geometry
manifest are not present at their recorded paths; the public replay therefore
reconstructs them from DICOM using the locked centres. The geometry manifest is
retained as an auditable input contract rather than claiming that the historical
tensor cache was copied.

## Verified environment

The exact B/C downstream checks used:

`<project-root>\.venv_locked_torch\python.exe`

Observed versions: Python 3.11.16, torch 2.11.0+cu128, scikit-learn 1.9.0,
numpy 2.4.6, pandas 3.0.3, scipy 1.17.1, SimpleITK 2.5.5, matplotlib 3.11.0.
The public lock file is `requirements-replay-tested.txt`; the torch wheel itself
is host/GPU dependent and must not be copied into a public package.

## A - lesion ROI radiomics

The raw input chain was also executed from the local DICOM and predicted-mask inputs
using the independent environment `<project-root>\.venv_locked_radiomics` (Python 3.9.23,
PyRadiomics v3.1.0, SimpleITK 2.5.6). The command completed with exit code 0 for all
408 subjects and produced 4,908 rows: 13 perturbation rows for each Internal375 case
and one baseline row for each External33 case. The generated matrix is
`outputs\a_raw_dicom_replay_20261011_v4\03_feature_extraction\full_cohort_perturbation_pyradiomics_features.csv`.

The source and replay audit both contain six internal baseline cases with `status=FAIL`
and 77 failed perturbation records in total. The failures are geometry/ROI-bound
errors for the same six cases, not missing public subjects; their feature columns are
NaN and the locked RF's `SimpleImputer(strategy="median")` is the historical handling
in the locked model. The replay matrix matched the locked 13-feature values at
machine precision for all 4,797 finite internal baseline comparisons and all 429
external comparisons (`max_abs_delta=0.0`). This is an exact feature-contract replay
including the historical failure/imputation behavior, not an assertion that every
perturbation is numerically available for every patient.

The locked-artifact model phase completed with exit code 0. It used the locked RF
SHA-256 `5CF7E8B242949E93BE3A252C84A1156E984C797622CDD080D0D17A17E654F3D0` without
refitting. The generated External33 probabilities matched the lock with maximum
absolute difference `2.220446049250313e-16`.

The public nested-CV entry point is
`scripts/reproduce_lesion_radiomics_nested_oof.py`. It reruns the historical
Internal375-only 5-outer/4-inner procedure from the public raw feature matrix,
crosswalk, and acquisition metadata, with external labels unread. Its full run
completed with exit code 0 and produced 375 OOF rows. The replayed OOF matched the
public locked reference `models/lesion_radiomics/internal_oof_predictions.csv`:
ID set, labels, and outer-fold assignments all matched; maximum probability delta
`3.3306690738754696e-16`; mean delta `4.0264088359739013e-17`; replay and reference
AUROC both `0.6987039999999999`. The companion verifier is
`scripts/verify_lesion_radiomics_nested_oof.py` and returned `PASS` at `1e-12`
tolerance. This is a code-generated strict OOF reproduction; the public package
does not need to retain five separate fold RF joblib files to reproduce the OOF.

The locked RF artifact replay passed without fitting or model selection:

`outputs\canonical_a_formal_20261011\lesion_radiomics\full_cohort_lesion_radiomics_locked_replay_summary.json`

It compared 375 internal and 33 external locked feature rows. The RF artifact
SHA-256 is `5CF7E8B242949E93BE3A252C84A1156E984C797622CDD080D0D17A17E654F3D0`.
The canonical multimodal replay intentionally differs from the historical internal
fusion lock because the historical lesion/whole-chest ID bridge mismatched 373/375
subjects. That difference is reported, not hidden.

The generated External33 A component was also passed through the public fusion
interface together with the locked Internal375 OOF component. External fusion matched
the historical lock to `1.1102230246251565e-16` maximum probability difference and
reproduced AUROC `0.7330827067669173`, Accuracy `0.696969696969697`, Sensitivity
`0.5714285714285714`, Specificity `0.7894736842105263`, and F1 `0.6153846153846154`.
Internal fusion used the canonical public mapping and therefore differed from the
historical identity-defective lock: maximum probability difference
`0.20982430477285802`, AUROC `0.859872` versus locked `0.871296`. It is a completed
canonical replay with a required scientific review, not an exact historical internal
result reproduction.

## Clinical full16 and controlled-input replay

The public clinical replay was rerun in the isolated output directory
`outputs/validation_full_clinical_replay_20261011` and completed with exit code 0.
It processed Internal375 and External33 and reproduced the locked full16 clinical
probabilities at maximum absolute differences of `9.71445146547012e-17` internally
and `8.326672684688674e-17` externally. The structural-missingness branch replayed
all 15 locked members with maximum differences of `2.220446049250313e-16` internally
and `1.1102230246251565e-16` externally.

The same run reports a separate comparison between a normally refitted full16
ensemble artifact and the locked external clinical probabilities: maximum difference
`0.3024520200131003`, mean difference `0.1392590543927006`. This is not a numerical
failure of the locked-probability replay. The public full16 locked prediction path
uses the historical external-mask training scenarios (`external_mask_1` through
`external_mask_3`) defined in `src/cap_cop/clinical/full_structured_input_strategy_experiment.py`,
whereas `models/clinical/full_structured/full_structured_clinical_ensemble.joblib`
is a separately serialized normal-training artifact. The package therefore reports
these as two distinct artifacts and does not call the normal-training joblib an exact
replacement for the locked external prediction path.

The controlled clinical-input comparison replay also completed for the core7 and
availability-gated branches. Core7 maximum probability differences were
`9.71445146547012e-17` internally and `8.326672684688674e-17` externally; the
availability-gated differences were `1.1102230246251565e-16` internally and
`8.326672684688674e-17` externally. The gate is `available_full_only_count >= 6`.
These comparisons use the historical locked SP6 pairing and remain subject to the
separate 373/375 internal identity-join boundary documented above.

## B - whole-chest foundation transfer

Raw-DICOM representation generation completed for 408 subjects:

```powershell
<project-root>\.venv_locked_torch\python.exe <project-root>\scripts\run_full_cohort_whole_chest_representations.py --phase representations --device cuda --segmentation-audit <project-root>\outputs\canonical_a_formal_20261011\segmentation\full_cohort_segmentation_replay_audit.csv --output-dir <project-root>\outputs\b_c_raw_dicom_replay_20261011
<project-root>\.venv_locked_torch\python.exe <project-root>\scripts\run_full_cohort_whole_chest_representations.py --phase whole_chest_model --device cuda --segmentation-audit <project-root>\outputs\canonical_a_formal_20261011\segmentation\full_cohort_segmentation_replay_audit.csv --output-dir <project-root>\outputs\b_c_raw_dicom_replay_20261011
```

The resulting summary is `outputs\b_c_raw_dicom_replay_20261011\full_cohort_whole_chest_replay_summary.json`.
Internal embedding and probability differences were at machine precision; external
probability difference was at most `2.7729797347220142e-08`, consistent with the
historical External33 CSV being written at finite precision. Internal AUROC was
`0.87008`; no parameters, weights, folds, or thresholds were changed.

Using the raw B component with the locked full-structured clinical component, the
multimodal replay also passed: Internal AUROC `0.929152`, External AUROC
`0.612781954887218`, with maximum fusion probability differences
`1.1102230246251565e-16` and `8.326672684688674e-17`, respectively. The component
assembly and summary are under `outputs\b_multimodal_replay_20261011`.

The independent locked-input verifier is
`scripts\verify_whole_chest_ct_locked_linear_probe.py`. It returns `PASS` only with
scikit-learn 1.9.0, exact Internal375 agreement, and external CSV-serialization
tolerance.

## C - stable multi-representation CT

The locked intermediate-input replay passed for SP6 on 375 internal and 33 external
subjects. Summary:

`outputs\b_c_locked_input_replay_20261011\full_cohort_whole_chest_replay_summary.json`

Maximum SP6 probability differences were `2.7755575615628914e-16` internally and
`1.1102230246251565e-16` externally. This is a locked embedding/tabular-input replay,
not proof that every historical raw phenotype and radiomics generation step is exact.
The corresponding locked-input multimodal fusion replay passed with maximum
probability differences `1.6653345369377348e-16` internally and
`8.326672684688674e-17` externally; its metrics were Internal AUROC `0.865024` and
External AUROC `0.7631578947368421`.

A raw-DICOM/de novo tabular replay was subsequently rerun with the locked ROI-centre
manifest. The first replay accidentally passed all 123 regenerated phenotype columns to
the V3.8 selector; the historical contract contains only the fixed columns in
`06_LESION_PHENOTYPE_FEATURES.csv` (19 phenotype features plus metadata). The public
entry point now filters regenerated phenotype input to that locked column contract and
fails if any locked column is missing.

After this correction, the fresh regenerated SP6 component replay matched the locked
component for all 375 Internal and 33 External subjects: maximum probability deltas were
`2.7755575615628914e-16` internally and `1.1102230246251565e-16` externally. The
canonical public fusion replay also passed, with maximum deltas of
`1.3877787807814457e-16` internally and `8.326672684688674e-17` externally; replay and
locked AUROC were `0.865024` internally and `0.7631578947368421` externally.

The separate historical identity audit still reports that the original lesion/whole-chest
branch-ID assembly agrees with the canonical public mapping for only 2/375 internal
subjects and differs for 373/375. That audit finding remains a scientific provenance
decision for the manuscript; it is not a reason to mislabel the corrected public replay
or to silently change the historical lock.

The corrected C component files from the rerun are:

- `outputs\\b_c_raw_dicom_replay_20261011\\stable_multirepresentation_regenerated_tabular_internal_replay.csv`
  SHA-256 `4114C1CA4F90961FF52073E14DC6E31AAAEE1563EA69B956D3FD9777A8057039`;
- `outputs\\b_c_raw_dicom_replay_20261011\\stable_multirepresentation_regenerated_tabular_external_replay.csv`
  SHA-256 `437DB48B039BD24D51BD5B7A215EC633D4105452FED52C71245A3D097171B951`;
- `outputs\\final_public_multimodal_replay_20261012_c_canonical_v2\\full_cohort_multimodal_replay_summary.json`
  SHA-256 `A390EE9FF0617765BD544D10FBAB8DCCAA896ADC70484E6153E150C660B67F5F`.

The corrected replay used the existing locked segmentation audit and existing public
runtime embeddings; it did not train, retune, or regenerate a paper lock.

## D - supplementary exploratory comparator

The full DICOM-to-DenseNet replay completed using the historical validation batch
contract (`batch_size=2`, `shuffle=False`):

`outputs\d_exploratory_full_replay_20261012_end_to_end_batch2\full_cohort_exploratory_densenet_replay_summary.json`

It processed 408 subjects and all 15 locked checkpoints, producing 375 internal OOF
subjects, 33 external subjects, 1125 internal member predictions, and 495 external
member predictions. Internal probability difference was at most
`1.6653345369377348e-16` (mean `4.469804156433573e-17`), and external probability
difference was at most `9.71445146547012e-17` (mean `3.301231342161924e-17`). The
earlier `3.4386912981659856e-05` discrepancy was caused by the public replay using
batch size 1 while the historical validation code used batch size 2; the checkpoint,
preprocessed tensors, and model definition were unchanged. The summary SHA-256 is
`B5F23AD184FDE177632C02A3EE15D2D5ABC44F8963061BB71AEE40204E75237E`.

The same run's DICOM preprocessing was independently validated against the locked
processed-input cache: 408 subjects were processed, all 375 Internal375 cases had an
exact cache match, and `internal_cache_differences=0`. The validation summary is
`outputs\d_exploratory_cache_validation_20261012\full_cohort_exploratory_densenet_replay_summary.json`.

D remains a supplementary exploratory comparator. The external reference boundary is
unchanged: the replay uses one canonical series per External33 subject and 15
checkpoints, while the historical exploratory lock averaged two source-version rows
and 30 member predictions per subject.

## Current release status

- B: raw-DICOM full-cohort replay accepted to numerical tolerance.
- A: locked RF/artifact validation accepted; corrected canonical fusion remains distinct
  from the historical identity-defective internal lock.
- C: raw-DICOM lesion-phenotype and regenerated SP6/C fusion replay accepted to the
  locked numerical tolerance after enforcing the fixed V3.8 phenotype-column contract;
  the separate 373/375 historical branch-ID audit remains documented for scientific
  review.
- D: full-cohort exploratory replay accepted to numerical tolerance after restoring the
  historical batch-size-2 validation behavior; its supplementary exploratory status is
  unchanged.
- `pytest` remains unavailable in the checked environments; smoke test functions were
  executed directly. The full-cohort commands above were actually run and their exit
  codes were recorded in the session log.
- Patient-level data and generated replay outputs remain local-only and Git-ignored.

## 2026-10-12 semantic entry-point verification

The public fusion entry point was updated to resolve the paper-facing semantic names
(`paper_model_a_*` through `paper_model_d_*`) to the component dispatch names without
changing any model parameter, weight, threshold, split, or prediction calculation. It
also accepts either structured component roots with a `clinical/` subdirectory or flat
replay roots, and supports a separate `--clinical-root` for Model D's exploratory
clinical artifact.

The four model-specific commands were executed from outside the project directory:

| Model | Internal subjects | External subjects | Max probability delta | Result |
|---|---:|---:|---:|---|
| A | 375 | 33 | `0.20982430477285802` internal; `1.11e-16` external | canonical replay; historical internal lock review required |
| B | 375 | 33 | `1.11e-16` internal; `8.33e-17` external | PASS |
| C | 375 | 33 | `1.39e-16` internal; `8.33e-17` external | PASS for regenerated component and fusion replay |
| D | 375 | 33 | `1.67e-16` internal; `9.71e-17` external | PASS to numerical tolerance; supplementary exploratory comparator |

All four commands exited with code 0. A's internal difference remains the documented
canonical-versus-historical identity issue; D now matches its historical prediction
lock to numerical tolerance after restoring the historical batch shape. These checks
establish runnable semantic entrypoints, not a scientific decision to replace the
historical paper locks.

## 2026-10-12 minimal entry-point replay

The public `scripts/run_minimal_reproduction.ps1` entry point was rerun from outside
the project directory with the locked Torch and PyRadiomics environments. All seven
steps exited with code 0. The internal single-case chain now uses the same LPS-oriented
2-mm resampling helper as U-Net segmentation, preventing a prior mask/image grid
mismatch (`169x175x175` versus `168x174x174`). The corrected chain produced 13
PyRadiomics features and a Model A RF probability of `0.3002668918308495` for the
local smoke case. The run also verified the external `96x96x96` P95 tensor contract,
the Model A external RF lock, the Model B linear-probe lock, the Model C/SP6 tables,
and the Model D artifact/single-case fusion path.

The PowerShell entry point now checks every native Python exit code and stops on the
first failed step; it no longer reports completion after a swallowed Python error.
This remains a single-case/locked-artifact acceptance run and is not a substitute for
the full Internal375/External33 replay records above.

## 2026-10-12 stable-representation semantic pipeline replay

The semantic stable-representation pipeline was rerun from the existing public
runtime tensor/embedding directory and regenerated the fixed phenotype and
PyRadiomics tabular inputs before entering the locked downstream model. The replay
completed for Internal375 and External33 with exit code 0:

- Internal probability maximum absolute difference: `3.3306690738754696e-16`;
- Internal probability mean absolute difference: `6.004456191514389e-17`;
- External probability maximum absolute difference: `1.1102230246251565e-16`;
- External probability mean absolute difference: `3.700743415417188e-17`.

The output summary is
`outputs\\b_c_raw_dicom_replay_20261011\\full_cohort_whole_chest_replay_summary.json`.
Its SHA-256 is `8AAF17E414918C344AC4B7D7F2A639F9BD1DFF6AE3CD75937479207242A4DFAE`.
The semantic module now defaults to public `models/` and `outputs/` roots instead of
an old experiment-tree root; the runtime result is unchanged. The historical
cross-branch identity audit remains explicit: only 2 of 375 internal branch IDs map
to the same public subject under the legacy bridge, while 373 do not.

## 2026-10-12 clinical entry-point and environment verification

The semantic lock-selector boundary was completed for the clinical replay entry points;
the governed historical pair IDs remain in the Git-ignored local selector file and are
not embedded in public configuration names. A first run with the radiomics environment
failed before prediction because the clinical joblib artifacts were serialized under
scikit-learn `1.9.0`, while that environment had `1.6.1` and the deserialized
`LogisticRegression` objects lacked `multi_class`. This was an environment/artifact
compatibility mismatch, not a model-data failure. Re-running with the existing locked
Torch environment (`scikit-learn 1.9.0`, `numpy 2.4.6`, `pandas 3.0.3`, `joblib 1.6.0`)
completed successfully without changing any artifact.

The full clinical-input comparison replay exited 0 for Internal375 and External33:

- core clinical: internal AUROC `0.773824`, external AUROC `0.7406015037593985`;
  maximum lock probability delta `9.71445146547012e-17` internal and
  `8.326672684688674e-17` external;
- availability-gated clinical: internal AUROC `0.8370879999999999`, external AUROC
  `0.6804511278195489`; maximum lock probability delta `1.1102230246251565e-16`
  internal and `8.326672684688674e-17` external.

The alternate clinical replay also exited 0: internal AUROC `0.8291839999999999`,
external AUROC `0.6654135338345866`, with maximum lock probability deltas
`2.220446049250313e-16` and `9.71445146547012e-17`, respectively.

The five-strategy fixed-SP6 fusion replay then exited 0 for all 375 internal and 33
external subjects in each strategy. Maximum replay-vs-lock deltas were:

| strategy | internal | external |
|---|---:|---:|
| full structured | `1.3877787807814457e-16` | `8.326672684688674e-17` |
| structural missingness | `3.3306690738754696e-16` | `2.220446049250313e-16` |
| core clinical | `2.220446049250313e-16` | `1.1102230246251565e-16` |
| availability gated | `1.6653345369377348e-16` | `1.1102230246251565e-16` |
| alternate configuration | `2.220446049250313e-16` | `1.249000902703301e-16` |

The public strategy entry point now accepts the actual regenerated stable-representation
table names as well as the locked-table fallback. Its scientific warning is unchanged:
the historical SP6 internal branch-ID bridge mismatches lesion-derived rows for 373/375
subjects, so these are numerical lock replays and not evidence for a single-factor
causal claim.

