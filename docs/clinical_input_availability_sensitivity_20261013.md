# Clinical Input Availability Sensitivity

Status: post-hoc exploratory analysis. This analysis does not replace or modify the locked paper results.

## Controlled design

The lesion-ROI radiomics image probabilities were held fixed for every scenario. The clinical branch was the only deliberately changed component. Internal evaluation used repeated five-fold patient-level OOF predictions with the same three seeds, random-forest family, balanced-bootstrap augmentation, fold-fitted median imputation, 1%/99% clipping, categorical encoding, and missingness indicators used by the public full-structured implementation. External inference reused the Full Structured Clinical--Lesion-ROI Radiomics external adaptation procedure; external labels were read only after prediction for evaluation.

Fusion used the locked Full Structured Clinical--Lesion-ROI Radiomics weights: the Internal375 workflow used the per-fold weights carried in the canonical component table, and External33 used clinical/image weights of 0.70/0.30. The classification threshold was 0.50 for every scenario.

## Scenarios

The run contains 21 clinical-input scenarios. `full_clinical_input` is the
baseline clinical branch of the Full Structured Clinical--Lesion-ROI Radiomics model, not a second implementation:

- `full_clinical_input`;
- random missingness at 20% and 30% across the full clinical input;
- `inflammatory_laboratory_panel_unavailable`;
- `core_clinical_input`;
- one-at-a-time unavailability of each of the 16 predefined clinical variables.

Random masks are deterministic and label-blind. The one-at-a-time scenarios set the named input unavailable in both cohorts before fitting and evaluation. These scenarios are not claims that the hospital data were actually missing in this exact pattern.

## Selected point estimates

| Scenario | Internal375 AUROC | External33 AUROC |
|---|---:|---:|
| Full clinical input | 0.859872 | 0.7330827068 |
| Random missing 20% | 0.855936 | 0.6541353383 |
| Random missing 30% | 0.852160 | 0.6203007519 |
| Inflammatory laboratory panel unavailable | 0.832192 | 0.7067669173 |
| Core clinical input | 0.792832 | 0.7406015038 |
| Missing sputum category | 0.855264 | 0.6766917293 |
| Missing C-reactive protein | 0.843584 | 0.7330827068 |

The complete metric tables and patient-level prediction files are in the output directory. Results are descriptive comparisons; no scenario was selected using External33 labels, and no causal claim is implied.

## Workflow files

- Configuration: `configs/clinical_input_availability_scenarios.json`
- Code: `scripts/run_clinical_input_availability_sensitivity.py`
- Output directory: `outputs/clinical_input_availability_sensitivity_20261013/`
- Scenario definitions: `scenario_registry.csv`
- Internal metrics: `internal_metrics.csv`
- External metrics: `external_metrics.csv`
- Internal predictions: `internal_predictions.csv`
- External predictions: `external_predictions.csv`
- Execution log: `run.log`
- Provenance and input hashes: `analysis_metadata.json`

## Verification

- 21 scenarios completed with process exit code 0.
- Every scenario contains 375 Internal375 rows and 33 External33 rows.
- Image probabilities are identical across all scenarios for each patient.
- Full16 clinical probabilities match the existing canonical component table with maximum absolute difference 0.
- Full16 fusion probabilities match the existing canonical component table with maximum absolute difference `1.1102230246251565e-16`.
- No original experiment files or locked result files were modified.

This analysis may support a supplementary description of robustness to predefined clinical-input unavailability. It should not be described as a prespecified primary analysis or as proof that clinical missingness causes a performance change.
