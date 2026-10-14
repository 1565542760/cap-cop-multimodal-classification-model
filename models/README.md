# Model Artifacts

Runtime model artifacts are supplied separately from the source code and are
ignored by Git. Place authorised artifacts under these relative directories:

```text
models/clinical/
models/imaging/
models/segmentation/
models/multimodal_fusion/
```

Do not commit checkpoints, patient-level predictions, clinical labels, identity
crosswalks or local data. The configuration files describe the expected model
interfaces and input contracts.
