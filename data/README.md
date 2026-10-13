# Local Data Directory

This directory is for local, patient-level CAP/COP data used on the host machine only.

It is intentionally ignored by Git. Do not commit, package, or upload contents of this
directory. The `deidentified_release_candidate` tree contains the locally assembled
Internal375 and External33 DICOM candidate inputs and their clinical manifests. Its
technical manifest records `technical_deidentification_pass=true`, but independent
burned-in-pixel review and hospital/ethics authorization remain pending; therefore the
candidate is not a public data release.

External cohort directories intentionally do not contain `masks` directories because the
locked external workflows use no migrated external masks. External `.npy` files, when
present elsewhere in the local workspace, are derived intermediate arrays for checksum
comparison only; they are not a replacement for original DICOM input. Local workflows may
use the candidate DICOM tree, but publication requires the manifest's pixel-review and
data-sharing fields to be resolved first.
