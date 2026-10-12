# Public Model Mapping

This document uses semantic public model names. The private lineage table that maps
these names to locked experimental identifiers is stored outside this workspace and is
not part of the public directory.

| Public name | Paper role | Clinical strategy | Imaging strategy | Clinical/Image weight | Status |
|---|---|---|---|---|---|
| paper_model_a_primary_lesion_radiomics | Model A, main model | Full structured clinical input | Lesion-ROI PyRadiomics | 0.70/0.30 | Full raw replay and strict OOF verified; corrected result lock pending |
| paper_model_b_whole_chest_ct_comparator | Model B comparator | Full structured clinical input | Full-field chest CT | 0.60/0.40 | Full raw-DICOM replay verified to tolerance |
| paper_model_c_stable_multirepresentation_ct | Model C comparator | Full structured clinical input | Stable multi-representation CT | 0.75/0.25 | Raw lesion-phenotype replay verified; corrected result lock pending |
| paper_model_d_supplementary_exploratory_comparator | Supplementary exploratory comparator | Structural missingness clinical branch | Exploratory imaging branch | 0.50/0.50 | Full-cohort exploratory replay; not a primary model |

The clinical-input comparison group is described as a fixed stable multi-representation
CT comparison across locked clinical branch configurations. It must not be described as
a single-factor causal experiment because several locked configurations use their own
weights and contract details.
