# Model Names

The public workflows use descriptive names for the model components and their
input types.

| Model name | Clinical input | Imaging input | Purpose |
|---|---|---|---|
| Full Structured Clinical--Lesion-ROI Radiomics | Structured clinical information | Lesion-ROI PyRadiomics | Multimodal classification |
| Full Structured Clinical--Whole-Chest CT | Structured clinical information | Whole-chest three-dimensional CT | Multimodal classification |
| Full Structured Clinical--Stable Multi-Representation CT | Structured clinical information | Stable multi-representation CT | Multimodal classification |
| Supplementary Exploratory Multimodal Comparator | Structured clinical information | Exploratory CT representation | Additional workflow |

The corresponding entry points and configuration files use the same descriptive
names. Input preparation, model inference and probability fusion are kept as
separate stages so that each component can be run independently.
