# Public Model Names

This document uses semantic public model names. The private lineage table that maps
these names to locked experimental identifiers is stored outside this workspace and is
not part of the public directory.

| Public model name | Role | Clinical strategy | Imaging strategy | Clinical/Image weight | Status |
|---|---|---|---|---|---|
| Full Structured Clinical--Lesion-ROI Radiomics | Main multimodal model | Full structured clinical input | Lesion-ROI PyRadiomics | 0.70/0.30 | Full raw workflow and strict OOF verified; corrected result lock pending |
| Full Structured Clinical--Whole-Chest CT | CT comparator | Full structured clinical input | Full-field chest CT | 0.60/0.40 | Full raw-DICOM workflow verified to tolerance |
| Full Structured Clinical--Stable Multi-Representation CT | CT comparator | Full structured clinical input | Stable multi-representation CT | 0.75/0.25 | Raw lesion-phenotype workflow verified; corrected result lock pending |
| Supplementary Exploratory Multimodal Comparator | Supplementary exploratory comparator | Structural missingness clinical branch | Exploratory imaging branch | 0.50/0.50 | Full-cohort exploratory workflow; not a primary model |

The clinical-input comparison group is described as a fixed stable multi-representation
CT comparison across locked clinical branch configurations. It must not be described as
a single-factor causal experiment because several locked configurations use their own
weights and contract details.
