# Canonical Identity Correction

## Decision

The public reproduction workspace uses one patient identity standard. The public
multimodal replay does not expose or accept the historical lesion identity join.

Lesion-derived predictions are matched as:

```text
segmentation_model_id -> public_subject_id -> clinical row and label
```

Whole-chest predictions use `whole_chest_imaging_model_id`, and clinical predictions
use `clinical_model_id`. No branch is matched by row order, filename order, or by
reusing an imaging ID from another branch.

## Evidence checked

`data/public_subject_model_crosswalk.csv` contains 408 unique public subjects with the
locked cohort counts: Internal CAP 250, Internal COP 125, External CAP 19, and External
COP 14. The crosswalk has unique clinical, lesion-imaging, whole-chest-imaging, and
segmentation identifiers. The local identity-contract audit reported zero duplicate or
empty public IDs and exact clinical/CT ID-set agreement.

The original private governance records were compared outside this package. The
canonical public crosswalk agrees with the earliest direct lesion imaging-to-patient
mapping for all 375 internal subjects. The later historical V2IMG-to-case crosswalk
does not: 373/375 internal subjects differ and 40/375 diagnoses cross. That historical
table is retained only as audit evidence outside the public runtime standard.

## Runtime consequence

`scripts/run_full_cohort_multimodal_replay.py` uses the canonical mapping only. The
former historical identity mode was removed. Existing historical paper locks are not
silently relabelled or treated as corrected results. A and SP6-linked internal result
locks must be regenerated or independently re-locked under the canonical mapping before
their corrected metrics are used in the manuscript.

This correction changes identity handling only. It does not change model architecture,
feature definitions, weights, thresholds, splits, seeds, or the historical lock files.

## Validation replay

On 2026-10-11, the public multimodal replay was run with `--component-mode locked` and
the canonical mapping. The component cache was the previously archived locked replay
cache; no model was trained and no weight or threshold was selected.

- Internal ID set: 375 subjects; SHA-256 `2826CAADA919764FEB31A86C9A964B396A5AD370FC0FD82261E530FDFADF94BE`.
- External ID set: 33 subjects; SHA-256 `FC90C915BA793FF256DA8ECBBA03DE9161CF97262DAC989A7AAB1F0A0A96C0EF`.
- Model A canonical internal AUROC: `0.859872`; historical lock comparison: `0.871296`.
- Model A canonical external AUROC: `0.7330827067669173`, equal to the locked external metric.
- Model B internal/external AUROC: `0.929152` / `0.612781954887218`.
- Model C internal/external AUROC: `0.865024` / `0.7631578947368421`.
- Model D internal/external AUROC: `0.896096` / `0.5413533834586466`; D remains exploratory.

The temporary validation summary had SHA-256
`94C3F571D66951F1ACC5D039A425C8D9A44802A6740032946E100BF0C014B5BF` and was removed
from the public workspace after verification. These values are a canonical replay
check, not a newly locked paper result.
