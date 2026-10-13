# Paper Coverage Audit

> Current status notice (2026-10-11): the earlier matrix below records the state before the full-cohort workflow run. The authoritative current assessment is `docs/full_reproduction_acceptance_20261011.md`. Full-cohort code coverage is now present, but a historical internal identity defect was confirmed for Full Structured Clinical--Lesion-ROI Radiomics model and the lesion-derived inputs in Full Structured Clinical--Stable Multi-Representation CT model/SP6 (373/375 subjects mismatched). Do not cite the older "partial" rows as current, and do not describe the paper model set as fully validated.

> A-specific update (2026-10-11): the public entry points
> `scripts/reproduce_lesion_radiomics_nested_oof.py` and
> `scripts/verify_lesion_radiomics_nested_oof.py` now rerun and verify the
> Internal375 5-outer/4-inner strict OOF procedure from the public raw feature
> matrix. The regenerated 375-row OOF matches the locked reference to a maximum
> probability delta of `3.3306690738754696e-16`; the older A partial row below is
> historical audit context only.

This audit uses the unmodified manuscript as the scope baseline:

- Manuscript: `<manuscript-source>\xwechat_files\wxid_zvsn2s1ehkpq22_4ede\msg\file\2026-09\CAP_COP_审阅稿_余乐.docx`
- Modified time: 2026-09-28 16:32:24
- SHA-256: `D3C9DC2621171B063986FC0F4A19A71935116E0BA2DA9A8B1BBD3A0A586B253C`

The Codex draft is treated only as a comparison file, not as the scope authority:

- Draft copy: `<manuscript-source>\xwechat_files\wxid_zvsn2s1ehkpq22_4ede\msg\file\2026-09\CAP_COP_审阅稿_余乐_Codex初改.docx`
- Modified time: 2026-09-28 21:08:15
- SHA-256: `B5AA6C56B0504FBE73496E7305F6F347CBC5698A7BD89187EC95A27FC0029521`

## Coverage Status Summary

| Manuscript item | Claim or method | Workspace evidence | Current status | Gap or boundary |
|---|---|---|---|---|
| Cohort construction | Internal candidate 382, final Internal375, External33 | Supplementary Table S1 and Figure S1 source data in `<project-root>_SUPPLEMENTARY_MATERIALS`; cohort manifests in `<project-root>\data` are local-only | Locked artifact verification | Exclusion reason for the 7 labelled candidate cases still needs author confirmation before final wording |
| Clinical branch full structured input | Sixteen pretreatment structured clinical variables | `<project-root>\src\cap_cop\preprocessing\clinical.py`; `scripts\run_full_cohort_clinical_replay.py`; Supplementary Tables S2-S4 | Full-cohort workflow run and artifact verification | Historical internal fusion validity remains subject to the documented identity-join correction |
| Structural-missingness clinical branch | Used by supplementary exploratory comparator | `<project-root>\src\cap_cop\clinical\structural_missingness_model.py`; `scripts\run_full_cohort_clinical_replay.py`; Supplementary Table S14 source | Full locked-artifact workflow run available, exploratory only | The deployment lock explicitly says `CLINICAL_V2B_STRICT_PROCEDURE_NOT_QUALIFIED`; do not present it as a newly qualified primary contract |
| Core, availability-gated, alternate clinical branches | Fixed-image clinical input comparison | `<project-root>\src\cap_cop\clinical\core_and_availability_gated_model.py`; semantic interfaces under `src\cap_cop\models\unimodal\clinical\`; `scripts\run_full_cohort_alternate_clinical_replay.py` | Full-cohort workflow run available; corrected comparison lock pending | Core/gate contracts are verified; C05 has a locked 16-feature contract but retains 12 RF/2 ElasticNet/1 GradientBoosting outer-family selection |
| DICOM reading and local image input | CT input from DICOM for local runnability | `<project-root>\src\cap_cop\imaging\dicom.py`; local `data/` ignored by Git | Partial implementation | DICOM is not de-identified for public release; it is local-only |
| U-Net lesion segmentation | DICOM to predicted lesion mask for Full Structured Clinical--Lesion-ROI Radiomics model | `<project-root>\src\cap_cop\imaging\segmentation\`; `<project-root>\models\segmentation\locked_checkpoints\` | Full-cohort workflow run and hash verification | Patient-level checkpoints and masks remain local-only and are excluded from Git |
| Internal strict OOF segmentation | Held-out internal cases use non-self-trained segmentation models | `<project-root>\src\cap_cop\imaging\segmentation\splits.py`; segmentation manifests | Full-cohort workflow run | Local locked split/checkpoint manifests are required and ignored by Git |
| External segmentation | External33 uses 20 locked internal-trained checkpoints | `<project-root>\src\cap_cop\imaging\segmentation\predict_external.py`; checkpoint manifest | Runnable configuration | External masks are generated to ignored outputs if run; no external mask is a fixed public input |
| PyRadiomics extraction | Lesion-ROI PyRadiomics with locked settings | `src\cap_cop\imaging\radiomics\extract_lesion_roi_features.py`; workflow run acceptance evidence; Supplementary Tables S6-S8 | Full raw-DICOM extraction workflow run | Six baseline extraction failures remain explicit in the audit; the locked RF imputer handles them |
| Radiomics 107 -> 16 -> 13 | Candidate, robust, final features | `src\cap_cop\pipelines\lesion_roi_pyradiomics_pipeline.py`; `scripts\reproduce_lesion_radiomics_nested_oof.py`; Supplementary Table S7 | Full strict nested-OOF procedure completed | Exact generated OOF matches the public reference; external labels are not read |
| Radiomics RF | Final lesion-ROI RF imaging branch | `scripts\run_full_cohort_lesion_radiomics.py`; `scripts\verify_lesion_radiomics_rf_locked.py`; Supplementary Table S8 | Full workflow run and locked artifact verification | Historical internal fusion still has the documented identity boundary |
| Full Structured Clinical--Lesion-ROI Radiomics model | Full clinical plus lesion-ROI radiomics, weights 0.70/0.30 | `configs/models.json`; `model_mapping_public.md`; A raw workflow run, strict OOF and fusion evidence; Supplementary Table S9 | Full raw workflow run; corrected result lock pending | Historical internal fusion differs under the canonical patient mapping; no replacement lock created without authorization |
| Full Structured Clinical--Whole-Chest CT model | Full clinical plus full-chest/full-field-of-view 3D CT, weights 0.60/0.40 | `configs/models.json`; whole-chest tensor and foundation workflow run scripts; Supplementary Table S9 | Full raw-DICOM workflow run to tolerance | Locked weights/embeddings remain local-only |
| Full Structured Clinical--Stable Multi-Representation CT model | Full clinical plus stable multi-representation CT, weights 0.75/0.25 | `configs/models.json`; stable representation workflow run scripts; Supplementary Table S9 | Full raw workflow run; corrected result lock pending | Historical internal tabular join has the documented identity mismatch |
| Supplementary Exploratory Multimodal Comparator | Structural-missingness clinical plus exploratory imaging comparator, weights 0.50/0.50 | `configs/models.json`; exploratory workflow run scripts; structural-missingness deployment lock; historical Table S14 source; Supplementary Table S9 | Full exploratory workflow run | C02 is explicitly not a qualified strict primary branch and I23 provenance remains an exploratory boundary |
| Fusion | Patient-level weighted average with fixed weights and threshold 0.50 | `<project-root>\src\cap_cop\models\multimodal\fusion.py`; `configs/models.json` | Runnable formula | Weight-selection chronology remains a manuscript interpretation boundary |
| Metrics | AUROC, accuracy, sensitivity, specificity, F1 and related tables | `metrics.py`; full-cohort workflow run scripts; Supplementary Tables S10-S12 | Runnable full-cohort metric calculation and lock comparison | Historical-versus-canonical identity differences must remain visible; CI analyses are separate post-hoc evidence |
| Table 1 | Baseline characteristics | Manuscript table; current workspace has no full baseline table-generation script | Missing or locked-manuscript only | Needs source table/generation script if it must be rebuilt from local data |
| Tables 2-3 | Model performance and multimodal comparison | Supplementary Tables S10-S12 and figure data | Locked artifact verification | Full metric regeneration from predictions is not yet implemented |
| Figures 1-3 | Main workflow and performance figures | Supplementary figure data; no main-figure regeneration script identified in current workspace | Partial support | Main figures are not fully regenerated from code here |
| Supplementary Figures S1-S8 | Rendered supplementary figures | PNGs copied into `<project-root>_SUPPLEMENTARY_MATERIALS\figures` with provenance hashes | Complete copied materials | Figure S3 remains a PHI-reviewed rendered panel, not raw imaging data |
| Bootstrap CI, DeLong, calibration, DCA | Raised in Word comments | No locked implementation found in current workspace | Not adopted or unresolved comment | Do not write as performed unless a post-hoc analysis is explicitly accepted and documented |

## Comment Boundary

The manuscript comments are in scope for tracking, but not all comments are accepted analyses. Current classification:

- Accepted or already reflected in body/table wording: the corrected AUROC difference wording around `0.0151` appears in the current manuscript text.
- Unresolved suggestions or questions: confidence intervals, DeLong/bootstrap tests, threshold provenance, calibration, Brier score, DCA, segmentation Dice/manual correction, ROI QC, external probability drift, missingness assumptions, and full feature-selection leakage checks.
- Rejected or not applicable: no explicit rejection record was found.
- Author, mentor, hospital, or ethics confirmation required: external cohort period and source, external ethics approval, de-identification and data-sharing permission, clinical data supplementation provenance, and final wording for the seven internal labelled candidates not included in Internal375.

## D 3D CT Dependency Classification

Current static scan found no `<private-experiment-root>` runtime literal in `src/`, `scripts/`,
`configs/`, or the current README. The following historical references remain
provenance-only and are not read by the current public entry points:

- `<project-root>\models\segmentation\manifests\lesion_segmentation_protocol_lock.json`:
  historical experiment root under `<private-experiment-root>`.
- `<project-root>_SUPPLEMENTARY_MATERIALS\source_manifest\source_manifest.csv`: contains the phrase `reproducible_without_3D_CT` as a column name only.

No evidence was found that current smoke tests or workflow run entry points need to read
data, code, or model files from `<private-experiment-root>`. Historical source snapshots and lock
manifests may still contain `<private-experiment-root>` provenance paths and are not standalone
runtime entrypoints unless wrapped.

## 2026-10-01 Targeted Workflow run Addendum

- B/C upstream input chain: complete internal P95 DICOM-to-96-cubed tensor production scripts were found at `<private-experiment-root>\experiment_v2\src\run_imaging_final_96cubed_tensor_contract_v2.py` and `<private-experiment-root>\experiment_v2\src\run_imaging_final_96cubed_physical_tensor_lock_v2.py`; exact copies were placed under `<project-root>\src\cap_cop\imaging\whole_chest_tensor\`.
- B/C P95 single-case workflow run: `EXT_CAP_001` generated tensor SHA-256 exactly matched locked `EXTV2_001` final tensor SHA-256. This confirms the recovered source and local wrapper can generate at least one external locked tensor from local DICOM.
- B/C full workflow run level: B completed the full 408-case DICOM-to-P95 tensor-to-frozen-representation workflow run. C completed the full 408-case raw-DICOM representation workflow run plus regenerated lesion phenotype/PyRadiomics tabular inputs and downstream SP6 comparison. The C workflow run is numerically exact to the locked downstream probabilities; the historical cross-branch identity discrepancy remains a scientific result-lock issue.
- A/I14 workflow run level: the full 408-case DICOM-to-predicted-mask-to-PyRadiomics extraction completed, followed by full RF inference and strict Internal375 nested-OOF workflow run. The locked 13-feature RF and OOF probabilities match at floating-point tolerance; the canonical internal multimodal fusion differs from the historical lock because of the documented patient-identity bridge defect.

## 2026-10-11 Full-Cohort Workflow run Superseding Addendum

- Internal375 and External33 segmentation, clinical, radiomics, whole-chest, stable-representation, exploratory imaging, fusion, and five clinical-strategy comparison entry points were executed.
- B is matched from DICOM through its frozen representation and downstream model to numerical tolerance.
- D is matched as an exploratory comparator; after restoring the historical internal inference batch-size contract, its internal and external imaging probability differences are `1.6653345369377348e-16` and `9.71445146547012e-17`, respectively.
- A raw radiomics features and the strict nested-OOF procedure are regenerated from the public workflow run inputs. The 13-feature RF lock and OOF probabilities generate at floating-point tolerance; the 77 logged perturbation failures remain explicit and are not silently repaired.
- The canonical migration and native-geometry evidence agree for all Internal375 subjects. Historical V3.11/V3.13 mapped lesion IDs through the whole-chest crosswalk: 373/375 different subjects and 40/375 different diagnoses. Historical mapping generates the A lock; canonical mapping does not.
- C/SP6 locked-input workflow run generates its historical lock exactly. The raw lesion-phenotype
  feature chain now also matches the locked table to floating-point tolerance after the
  locked ROI-centre manifest was restored; the same historical namespace error still causes
  373/375 internal lesion-derived tabular rows to be joined to another whole-chest patient.
- Therefore numerical workflow run is complete for the stored locks, but the internal A/C results are not accepted as scientifically valid patient-matched end-to-end workflow executions.


