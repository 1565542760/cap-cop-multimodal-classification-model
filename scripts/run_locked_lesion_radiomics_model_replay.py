from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_full_cohort_lesion_radiomics import (
    audit_existing_model_outputs,
    configure_locked_output,
    load_contract,
    run_models,
)


DEFAULT_INPUT = PROJECT_ROOT / "models" / "lesion_radiomics" / "locked_internal_perturbation_features.csv"
DEFAULT_EXTERNAL_INPUT = (
    PROJECT_ROOT
    / "outputs"
    / "full_reproduction_20261010"
    / "lesion_radiomics"
    / "03_feature_extraction"
    / "full_cohort_perturbation_pyradiomics_features.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "full_reproduction_20261010" / "lesion_radiomics_locked_feature_replay"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay the locked lesion-radiomics nested CV and final RF from the historical derived feature matrix."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--external-input", type=Path, default=DEFAULT_EXTERNAL_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    contract = load_contract()
    internal = contract[contract["cohort"].eq("internal")][
        ["public_subject_id", "segmentation_model_id"]
    ].rename(columns={"segmentation_model_id": "neutral_patient_id"})
    internal_matrix = pd.read_csv(args.input.resolve(), encoding="utf-8-sig")
    internal_matrix = internal_matrix.merge(internal, on="neutral_patient_id", how="left", validate="many_to_one")
    if internal_matrix["public_subject_id"].isna().any():
        raise RuntimeError("Locked radiomics matrix contains IDs outside the Internal375 contract")
    external_ids = set(contract.loc[contract["cohort"].eq("external"), "public_subject_id"].astype(str))
    external_matrix = pd.read_csv(args.external_input.resolve(), encoding="utf-8-sig")
    external_matrix = external_matrix[
        external_matrix["public_subject_id"].astype(str).isin(external_ids)
        & external_matrix["perturbation"].eq("BASELINE")
    ].copy()
    matrix = pd.concat([internal_matrix, external_matrix], ignore_index=True, sort=False)

    output_dir = args.output_dir.resolve()
    configure_locked_output(output_dir)
    replay_matrix_path = output_dir / "03_feature_extraction" / "full_cohort_perturbation_pyradiomics_features.csv"
    replay_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(replay_matrix_path, index=False, encoding="utf-8-sig")
    summary = (
        audit_existing_model_outputs(contract, matrix, output_dir)
        if args.audit_only
        else run_models(contract, matrix, output_dir)
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
