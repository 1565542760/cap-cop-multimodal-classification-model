# Full-Cohort Reproduction Acceptance - 2026-10-11

## Decision

The local workspace now contains runnable full-cohort chains for the clinical branches, segmentation, lesion radiomics, whole-chest representation, exploratory DenseNet, fusion, and the five clinical-strategy comparisons. It is **not scientifically valid to describe the complete paper model set as fully reproduced without qualification**.

Two separate outcomes must be retained:

1. Historical numerical replay: most locked probabilities can be reproduced to floating-point tolerance.
2. Patient-level validity: the historical internal V3.11/V3.13 lesion/whole-chest namespace join is defective for Models A and C and for the SP6 clinical-strategy comparisons.

## Cohort Contract

- Internal: 375 unique public subjects, CAP 250 and COP 125.
- External: 33 unique public subjects, CAP 19 and COP 14.
- Canonical internal ID-set SHA-256 (sorted IDs joined by newline, no terminal newline): `A5F527FD2AAE767C0A40DF1A177C690037498295D829587E65C8F2A7EF447CF5`.
- Canonical external ID-set SHA-256 under the same convention: `492FF445637D2F7C7A1469A2B9058E60DEE5AA4ABD3BF60F99983E10DDBA2CAC`.
- Replay CSV summaries use a terminal-newline convention and therefore report `2826CAADA919764FEB31A86C9A964B396A5AD370FC0FD82261E530FDFADF94BE` / `FC90C915BA793FF256DA8ECBBA03DE9161CF97262DAC989A7AAB1F0A0A96C0EF`; the underlying ID sets are identical.
- Canonical crosswalk SHA-256: `167742F880DA7DF67E4CD7AEC159655672A4493910A74262106F75876D2AC2ED`.
- Rebuilding the crosswalk from the private migration manifests and independent whole-chest manifest produced that exact SHA-256.

## Identity Audit

The canonical DICOM migration manifest and the native-geometry governance table agree for all 375 internal subjects. The historical V3.11/V3.13 code instead mapped lesion `SEG` predictions through a reused whole-chest `V2IMG -> case` crosswalk.

- Same canonical and historical subject: 2/375.
- Different subject: 373/375.
- Different CAP/COP diagnosis: 40/375.
- Model A image probability versus lock under canonical mapping: max absolute delta `0.6994143492428594`, mean `0.20614242222603502`.
- Model A image probability versus lock under historical mapping: max `3.3306690738754696e-16`, mean `5.121828886937389e-17`.

Primary evidence:

- `outputs/identity_audit/historical_internal_imaging_identity_audit.json`, SHA-256 `426CA9BE265FA67117E2BECFFCE0CA7C5F618FA337B5CCD41F5CE176E8B6B9ED`.
- Historical V3.11 source SHA-256 `651A5598916B217A738CF0B06136D0EAADEFF73F9002F43A8C28477C25805BD2`.
- Private migration manifest SHA-256 `01BE038D44F11E03A670AE17E12AB3207DD73F8F33648894A08A589DCB7FFF96`.
- Native-geometry governance SHA-256 `1FD072316F1A3E028B86062EE32354743992D2B7789F2BC03394D0149CADE92C`.
- Historical whole-chest crosswalk SHA-256 `6EBFC6713B719E5E2540663A9FF6F2156D49C24D8E4FF4BC7FB70B1A3176869E`.

The rejected temporary crosswalk under `outputs/identity_audit/corrected_public_subject_model_crosswalk.csv` is not canonical and must not be used. The canonical file in `data/public_subject_model_crosswalk.csv` was not overwritten.

## Model Status

| Model or branch | Full-cohort run | Numerical result | Scientific boundary |
|---|---|---|---|
| Segmentation | Internal375 strict OOF and External33 20-checkpoint ensemble | 408/408 mask hashes match locked outputs | Valid local replay |
| Full structured clinical | Internal375 OOF and External33 | Locked-probability replay exact to floating-point tolerance; separately serialized normal-training artifact differs from the locked external-mask path by max `0.3024520200131003` | Valid replay with two explicitly separated artifact paths |
| Structural-missingness clinical | Internal375 OOF and External33 | Exact to floating-point tolerance | Valid artifact-backed replay |
| Core7 clinical | Internal375 OOF and External33 | Max delta below `1e-16` | Valid replay |
| Availability-gated clinical | Internal375 OOF and External33 | Max delta below `2e-16` | Valid replay |
| Alternate clinical | Internal375 OOF and External33 | Max delta below `3e-16` | Internal registry label says RF although selected outer families are 12 RF, 2 elastic-net, 1 gradient boosting |
| A lesion radiomics | 408 raw DICOM cases, raw feature extraction, and public nested-OOF replay | 375-row code-generated strict OOF matches the locked reference to max probability delta `3.3306690738754696e-16`; raw feature values match the locked 13-feature contract | Internal paper fusion still uses the defective historical identity mapping; canonical internal AUROC is `0.859872`, historical lock is `0.871296` |
| B whole-chest CT | 408 DICOM-to-P95-to-TANGERINE representations and linear probes | Embeddings exact; internal probabilities at machine precision, external max delta `2.7729797347220142e-08` | Valid replay to stated tolerance |
| C stable multirepresentation | 408 DICOM deep representations; raw lesion-phenotype replay with locked ROI geometry | Raw lesion-phenotype features match to floating-point tolerance; historical SP6 lock remains numerically replayable | Historical internal tabular join uses the wrong lesion-derived row for 373/375 subjects; canonical regenerated fusion is reported separately |
| D exploratory comparator | Internal375 and External33 | Internal image max delta about `3.44e-05`; external at machine precision | Supplementary exploratory only |

## Multimodal Replay

Historical-lock mode reproduces the stored A-D fusion probabilities. Canonical-identity mode
reassigns lesion-derived and SP6-linked internal rows using the branch-native public mapping;
therefore Model A, Model C, and the fixed-SP6 clinical-input comparisons must be reported
separately from their historical identity-defective locks.

- A canonical internal AUROC: `0.859872`; historical locked AUROC: `0.871296`.
- A external AUROC: `0.7330827067669173`, exact under both modes because the external mapping is unaffected.
- B internal/external AUROC: `0.929152` / `0.612781954887218`, exact to tolerance.
- C internal/external AUROC: `0.865024` / `0.7631578947368421`, numerically exact historical replay, with the internal input-identity defect above.
- D internal/external AUROC: `0.896096` / `0.5413533834586466`, exploratory.

Canonical summary SHA-256: `024112938A7C8E5811C43CC119F58A6E37DF7A9D6CA8FD98213D3587BC6FBB92`.
Historical summary SHA-256: `4762D9A5FA26B491E06F65A0F5E11C8B27B70A8E47C59DC9B69EDBE8E77A3D9B`.

## Fixed-SP6 Clinical Strategy Comparison

The five stored multimodal pairs were replayed from the re-executed clinical probabilities, the same locked SP6 output, and each pair's stored fold-specific internal weights/fixed external weight. All fusion probabilities match the lock within `2.220446049250313e-16`.

External AUROC values are `0.7631578947368421`, `0.6954887218045113`, `0.7255639097744362`, `0.6954887218045113`, and `0.7105263157894737` for full structured, structural missingness, core, availability-gated, and alternate configurations, respectively.

This is a numerical lock replay only. Because the shared SP6 historical input assembly has the 373/375 internal identity defect and because weights differ by clinical configuration, it cannot support a causal or strict single-factor claim.

## Commands Executed

The commands used the independent `<project-root>/.venv_locked_torch/python.exe` or `<project-root>/.venv_locked_radiomics/python.exe`. Key entry points were:

- `scripts/run_full_cohort_lesion_segmentation.py`
- `scripts/run_full_cohort_clinical_replay.py`
- `scripts/run_full_cohort_clinical_input_comparison_replay.py`
- `scripts/run_full_cohort_alternate_clinical_replay.py`
- `scripts/run_full_cohort_lesion_radiomics.py`
- `scripts/run_locked_lesion_radiomics_model_replay.py`
- `scripts/run_full_cohort_whole_chest_representations.py`
- `scripts/run_full_cohort_exploratory_densenet.py`
- `scripts/audit_historical_imaging_identity_mapping.py`
- `scripts/run_full_cohort_multimodal_replay.py`
- `scripts/run_full_cohort_clinical_strategy_multimodal_replay.py`

No external label was used to retrain, tune, or choose a weight or threshold in these replay commands.

## Data Release Boundary

The public-subject mapping is now unambiguous in the canonical local crosswalk. This does not itself authorize data publication. CT remains local and Git-ignored. DICOM de-identification had no technical failures in the 408-case candidate run, but burned-in annotation review and hospital/ethics/data-sharing approval remain required before release.

The final local identity-contract audit passed for both the reproduction inputs and the technical de-identification candidate: manifest, clinical table, and CT-directory sets all match the same 408 canonical IDs; duplicates and empty IDs are zero; external mask directories are absent. Evidence is `docs/full_reproduction_evidence_20261011/local_data_identity_contract.json` (SHA-256 recorded in the adjacent aggregate manifest). The technical candidate contains 110,286 DICOM files and 58,176,859,522 bytes with zero recorded technical failures, but its status remains `TECHNICAL_RELEASE_CANDIDATE_NOT_AUTHORISED_FOR_PUBLICATION`.

## Runtime Intermediate Archival

After validation, 3,480 runtime files totalling 5,724,377,762 bytes were moved from `<project-root>/outputs` to the governed runtime archive. Every target file was rehashed after the move; verification failures were zero. The private move manifest SHA-256 is `0B425AAF6A034F84ACCF973B2D5ED6BE21B17B283FC97C73140C682789620709`. `<project-root>/outputs` now contains only `.gitkeep`. Aggregate, non-patient summaries remain under `docs/full_reproduction_evidence_20261011`; their manifest SHA-256 is `2E4C7A7F4CE39BE0E57E245D0D6CBA3F77A60115C13CEA19E59F4FE07197808A`.

## Final Acceptance

- Code execution coverage: full-cohort replay entry points exist and ran for all listed branches.
- Historical result reproducibility: largely yes, with the stated raw-radiomics and D numerical tolerances.
- Scientifically valid end-to-end reproduction of every published internal multimodal result: **no**, because A and C/SP6 inherit a confirmed historical patient-identity join defect.
- Corrective re-training or replacement paper results were not performed in this audit.

