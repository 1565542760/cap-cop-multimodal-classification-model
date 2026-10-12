# Governed model artifacts

The local `models/` tree contains controlled replay artifacts, checkpoints,
patient-level prediction tables, ID manifests, and lock evidence used for local
validation. It is intentionally excluded from the public Git repository.

The public code expects authorized users to provide the required model artifacts
under this relative directory, or to use an approved artifact registry. Do not
commit checkpoints, patient-level predictions, clinical labels, ID crosswalks,
absolute source paths, or private experiment manifests. The semantic contracts
and model roles are documented in `configs/models.json` and `docs/`; the private
historical pair selectors are kept separately in the ignored local configuration.

The public repository does not claim that a clone without this governed bundle can
reproduce patient-level paper numbers. A clean clone can run the dependency and semantic
smoke tests; exact replay requires the authorized bundle at these relative roots:
`models/clinical/`, `models/imaging/`, `models/segmentation/`,
`models/multimodal_fusion/`, and the corresponding ignored `data/` manifests and inputs.
