from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cap_cop.imaging.whole_chest_tensor.generate_whole_chest_tensor import materialize_dicom_dir
from cap_cop.pipelines import stable_multirepresentation_replay_pipeline as stable
from cap_cop.vendor.models_genesis import unet3d as genesis_unet3d
from run_stable_multirepresentation_subset_replay import extract_state_dict, patch_stable_paths
from run_whole_chest_foundation_subset_replay import (
    extract_foundation_embedding,
    fixed_linear_probe,
    load_foundation_encoder,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "full_reproduction_20261010" / "whole_chest_representations"
CROSSWALK = PROJECT_ROOT / "data" / "public_subject_model_crosswalk.csv"
DEFAULT_SEGMENTATION_AUDIT = (
    PROJECT_ROOT
    / "outputs"
    / "canonical_a_formal_20261011"
    / "segmentation"
    / "full_cohort_segmentation_replay_audit.csv"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def input_contract(segmentation_audit: Path | None, *, require_segmentation_audit: bool = True) -> pd.DataFrame:
    crosswalk = pd.read_csv(CROSSWALK, dtype=str).fillna("")
    if require_segmentation_audit:
        if segmentation_audit is None or not segmentation_audit.exists():
            raise RuntimeError(
                f"Raw-DICOM representation replay requires a segmentation audit: {segmentation_audit}"
            )
        audit = pd.read_csv(segmentation_audit, dtype=str).fillna("")
        contract = crosswalk.merge(
            audit[["public_subject_id", "outer_fold", "dicom_dir", "status"]],
            on="public_subject_id",
            how="left",
            validate="one_to_one",
        )
    else:
        contract = crosswalk.copy()
        contract["outer_fold"] = ""
        contract["dicom_dir"] = ""
        contract["status"] = "LOCKED_INPUT_ONLY"
    if len(contract) != 408 or contract["public_subject_id"].nunique() != 408:
        raise RuntimeError("Expected 408 unique subjects")
    if require_segmentation_audit and not contract["status"].eq("PASS").all():
        raise RuntimeError("DICOM contract is not backed by a passing segmentation audit")
    contract["label_binary"] = contract["label_binary"].astype(int)
    return contract


def load_encoders(device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module, Path, Path]:
    tangerine_checkpoint = PROJECT_ROOT / "models" / "whole_chest_ct" / "foundation_encoder" / "tangerine_foundation_encoder.pth"
    genesis_checkpoint = PROJECT_ROOT / "models" / "stable_multirepresentation_ct" / "foundation_encoders" / "genesis_chest_ct.pt"
    tangerine_model = load_foundation_encoder(tangerine_checkpoint, device)
    raw = torch.load(genesis_checkpoint, map_location="cpu", weights_only=False)
    state = extract_state_dict(raw)
    genesis_model = genesis_unet3d.UNet3D(n_class=1, act="relu")
    model_state = genesis_model.state_dict()
    genesis_model.load_state_dict(
        {key: value for key, value in state.items() if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)},
        strict=False,
    )
    genesis_model.eval().to(device)
    return tangerine_model, genesis_model, tangerine_checkpoint, genesis_checkpoint


@torch.no_grad()
def extract_genesis_embedding_loaded(model: torch.nn.Module, tensor_path: Path, device: torch.device) -> np.ndarray:
    array = np.load(tensor_path).astype(np.float32, copy=False)
    if array.ndim == 3:
        array = array[None, None]
    elif array.ndim == 4:
        array = array[None]
    elif array.ndim != 5:
        raise ValueError(f"Unexpected tensor shape for Genesis: {array.shape}")
    tensor = torch.from_numpy(array).to(device)
    out64, _ = model.down_tr64(tensor)
    out128, _ = model.down_tr128(out64)
    out256, _ = model.down_tr256(out128)
    out512, _ = model.down_tr512(out256)
    return nn.AdaptiveAvgPool3d(1)(out512).flatten(1).detach().cpu().numpy().astype(np.float32, copy=False)


def replay_representations(contract: pd.DataFrame, output_dir: Path, device: torch.device) -> None:
    tensor_dir = output_dir / "runtime_tensors"
    embedding_dir = output_dir / "runtime_embeddings"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    tangerine_model, genesis_model, tangerine_checkpoint, genesis_checkpoint = load_encoders(device)

    rows = []
    for index, row in enumerate(contract.to_dict("records"), start=1):
        subject_id = str(row["public_subject_id"])
        tensor_path = tensor_dir / f"{subject_id}_p95_96cubed_tensor.npy"
        tensor_qc_path = tensor_dir / f"{subject_id}_p95_96cubed_tensor_qc.json"
        tangerine_path = embedding_dir / f"{subject_id}_tangerine_embedding.npy"
        genesis_path = embedding_dir / f"{subject_id}_genesis_embedding.npy"

        if not tensor_path.exists() or not tensor_qc_path.exists():
            materialize_dicom_dir(subject_id, Path(row["dicom_dir"]), tensor_dir)
        if not tangerine_path.exists():
            embedding = extract_foundation_embedding(tangerine_model, tensor_path, device)
            np.save(tangerine_path, embedding, allow_pickle=False)
        if not genesis_path.exists():
            embedding = extract_genesis_embedding_loaded(genesis_model, tensor_path, device)
            np.save(genesis_path, embedding, allow_pickle=False)

        tangerine = np.load(tangerine_path)
        genesis = np.load(genesis_path)
        rows.append(
            {
                "public_subject_id": subject_id,
                "cohort": row["cohort"],
                "whole_chest_imaging_model_id": row["whole_chest_imaging_model_id"],
                "external_tensor_id": row["external_tensor_id"],
                "tensor_sha256": sha256_file(tensor_path),
                "tangerine_sha256": sha256_file(tangerine_path),
                "genesis_sha256": sha256_file(genesis_path),
                "tensor_shape": "x".join(map(str, np.load(tensor_path, mmap_mode="r").shape)),
                "tangerine_shape": "x".join(map(str, tangerine.shape)),
                "genesis_shape": "x".join(map(str, genesis.shape)),
                "finite": bool(np.isfinite(tangerine).all() and np.isfinite(genesis).all()),
            }
        )
        if index % 5 == 0 or index == len(contract):
            pd.DataFrame(rows).to_csv(output_dir / "whole_chest_representation_replay_manifest.csv", index=False, encoding="utf-8-sig")
            print(f"whole-chest representations {index}/{len(contract)} last={subject_id}", flush=True)

    write_json(
        output_dir / "encoder_provenance.json",
        {
            "device": str(device),
            "tangerine_checkpoint": str(tangerine_checkpoint),
            "tangerine_checkpoint_sha256": sha256_file(tangerine_checkpoint),
            "genesis_checkpoint": str(genesis_checkpoint),
            "genesis_checkpoint_sha256": sha256_file(genesis_checkpoint),
            "subjects": len(contract),
        },
    )


def stack_by_ids(contract: pd.DataFrame, ids: list[str], id_column: str, family: str, output_dir: Path) -> np.ndarray:
    lookup = contract.set_index(id_column)["public_subject_id"].to_dict()
    rows = []
    for model_id in ids:
        if model_id not in lookup:
            raise RuntimeError(f"No public subject mapping for {id_column}={model_id}")
        path = output_dir / "runtime_embeddings" / f"{lookup[model_id]}_{family}_embedding.npy"
        rows.append(np.load(path).reshape(-1))
    return np.vstack(rows).astype(np.float32, copy=False)


def probability_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    positives = labels == 1
    negatives = labels == 0
    return {
        "AUROC": float(roc_auc_score(labels, probability)),
        "Accuracy": float(accuracy_score(labels, prediction)),
        "Sensitivity": float(((prediction == 1) & positives).sum() / positives.sum()),
        "Specificity": float(((prediction == 0) & negatives).sum() / negatives.sum()),
        "F1": float(f1_score(labels, prediction)),
    }


def replay_whole_chest_probe(contract: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    model_dir = PROJECT_ROOT / "models" / "whole_chest_ct"
    internal_manifest = pd.read_csv(model_dir / "internal_embedding_manifest.csv")
    external_manifest = pd.read_csv(model_dir / "external_embedding_manifest.csv")
    locked_internal = pd.read_csv(model_dir / "internal_oof_predictions.csv")
    locked_external = pd.read_csv(model_dir / "external_predictions.csv")
    labels_folds = locked_internal[["neutral_imaging_id", "label", "fold_id", "primary_domain"]].drop_duplicates()
    internal = internal_manifest.merge(labels_folds, on="neutral_imaging_id", validate="one_to_one").sort_values("embedding_index")
    external = external_manifest.sort_values("embedding_index")
    internal_embeddings = stack_by_ids(contract, internal["neutral_imaging_id"].astype(str).tolist(), "whole_chest_imaging_model_id", "tangerine", output_dir)
    external_embeddings = stack_by_ids(contract, external["neutral_external_id"].astype(str).tolist(), "external_tensor_id", "tangerine", output_dir)

    labels = internal["label"].to_numpy(dtype=int)
    folds = internal["fold_id"].to_numpy(dtype=int)
    oof = np.full(len(internal), np.nan, dtype=float)
    external_fold_probabilities = []
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        model = fixed_linear_probe()
        model.fit(internal_embeddings[train], labels[train])
        oof[test] = model.predict_proba(internal_embeddings[test])[:, 1]
        external_fold_probabilities.append(model.predict_proba(external_embeddings)[:, 1])
    external_matrix = np.vstack(external_fold_probabilities)
    external_probability = external_matrix.mean(axis=0)
    external_std = external_matrix.std(axis=0)

    replay_internal = internal[["neutral_imaging_id", "label", "fold_id", "primary_domain"]].copy()
    replay_internal["prob_COP"] = oof
    replay_external = external[["neutral_external_id"]].copy()
    replay_external["prob_COP"] = external_probability
    replay_external["prob_COP_std"] = external_std
    replay_internal.to_csv(output_dir / "whole_chest_internal_oof_replay.csv", index=False)
    replay_external.to_csv(output_dir / "whole_chest_external_replay.csv", index=False)

    internal_joined = replay_internal.merge(locked_internal[["neutral_imaging_id", "prob_COP"]], on="neutral_imaging_id", suffixes=("_replay", "_locked"), validate="one_to_one")
    external_joined = replay_external.merge(locked_external[["neutral_external_id", "prob_COP", "prob_COP_std"]], on="neutral_external_id", suffixes=("_replay", "_locked"), validate="one_to_one")
    internal_delta = np.abs(internal_joined["prob_COP_replay"] - internal_joined["prob_COP_locked"])
    external_delta = np.abs(external_joined["prob_COP_replay"] - external_joined["prob_COP_locked"])

    locked_internal_embeddings = np.load(model_dir / "embeddings" / "internal_embeddings.npy")
    locked_external_embeddings = np.load(model_dir / "embeddings" / "external_embeddings.npy")
    return {
        "internal_subjects": len(internal),
        "external_subjects": len(external),
        "internal_embedding_max_abs_delta": float(np.max(np.abs(internal_embeddings - locked_internal_embeddings))),
        "internal_embedding_mean_abs_delta": float(np.mean(np.abs(internal_embeddings - locked_internal_embeddings))),
        "external_embedding_max_abs_delta": float(np.max(np.abs(external_embeddings - locked_external_embeddings))),
        "external_embedding_mean_abs_delta": float(np.mean(np.abs(external_embeddings - locked_external_embeddings))),
        "internal_probability_max_abs_delta": float(internal_delta.max()),
        "internal_probability_mean_abs_delta": float(internal_delta.mean()),
        "external_probability_max_abs_delta": float(external_delta.max()),
        "external_probability_mean_abs_delta": float(external_delta.mean()),
        "internal_metrics": probability_metrics(labels, oof),
    }


def replay_stable_multirepresentation(
    contract: pd.DataFrame,
    output_dir: Path,
    *,
    use_regenerated_tabular_inputs: bool,
    phenotype_path: Path | None = None,
    radiomics_path: Path | None = None,
) -> dict[str, Any]:
    mode = "regenerated_tabular" if use_regenerated_tabular_inputs else "locked_tabular"
    stable_output = output_dir / f"stable_runtime_{mode}"
    patch_stable_paths(PROJECT_ROOT, stable_output)
    model_dir = PROJECT_ROOT / "models" / "stable_multirepresentation_ct"
    phenotype_path = phenotype_path or (PROJECT_ROOT / "outputs" / "full_reproduction_20261010" / "lesion_phenotype" / "full_cohort_lesion_phenotype_features.csv")
    radiomics_path = radiomics_path or (PROJECT_ROOT / "outputs" / "full_reproduction_20261010" / "lesion_radiomics" / "03_feature_extraction" / "full_cohort_perturbation_pyradiomics_features.csv")
    if use_regenerated_tabular_inputs:
        if not phenotype_path.exists() or not radiomics_path.exists():
            raise RuntimeError("C raw/regenerated replay requires full-cohort lesion phenotype and PyRadiomics inputs")
        radiomics = pd.read_csv(radiomics_path, encoding="utf-8-sig")
        radiomics = radiomics[radiomics["perturbation"].eq("BASELINE")].copy()
        internal_radiomics = radiomics[radiomics["public_subject_id"].isin(contract.loc[contract["cohort"].eq("internal"), "public_subject_id"])]
        external_radiomics = radiomics[radiomics["public_subject_id"].isin(contract.loc[contract["cohort"].eq("external"), "public_subject_id"])]
        # The locked V3.8 representation used the fixed 19-column phenotype
        # contract. Raw replay may contain additional diagnostic descriptors;
        # they must not silently become candidate inputs.
        locked_phenotype = model_dir / "derived_inputs" / "06_LESION_PHENOTYPE_FEATURES.csv"
        locked_phenotype_columns = pd.read_csv(locked_phenotype, nrows=0).columns.tolist()
        regenerated_phenotype = pd.read_csv(phenotype_path, encoding="utf-8-sig")
        missing_phenotype_columns = [c for c in locked_phenotype_columns if c not in regenerated_phenotype.columns]
        if missing_phenotype_columns:
            raise RuntimeError(
                "Regenerated phenotype input is missing locked V3.8 columns: "
                + ", ".join(missing_phenotype_columns)
            )
        regenerated_phenotype = regenerated_phenotype[locked_phenotype_columns].copy()
        regenerated_inputs = stable_output / "regenerated_tabular_inputs"
        regenerated_inputs.mkdir(parents=True, exist_ok=True)
        filtered_phenotype_path = regenerated_inputs / "locked_contract_lesion_phenotype.csv"
        regenerated_phenotype.to_csv(filtered_phenotype_path, index=False)
        internal_radiomics_path = regenerated_inputs / "internal_lesion_radiomics.csv"
        external_radiomics_path = regenerated_inputs / "external_lesion_radiomics.csv"
        internal_radiomics.to_csv(internal_radiomics_path, index=False)
        external_radiomics.to_csv(external_radiomics_path, index=False)
        stable.PATHS["lesion_features"] = filtered_phenotype_path
        stable.PATHS["d1_internal_features"] = internal_radiomics_path
        stable.PATHS["d1_external_features"] = external_radiomics_path
    else:
        locked_required = [
            model_dir / "embeddings" / "genesis_internal_embeddings.npy",
            model_dir / "embeddings" / "genesis_external_embeddings.npy",
            model_dir / "embeddings" / "tangerine_internal_embeddings.npy",
            model_dir / "embeddings" / "tangerine_external_embeddings.npy",
            model_dir / "derived_inputs" / "06_LESION_PHENOTYPE_FEATURES.csv",
            model_dir / "derived_inputs" / "V3_5_D1_INTERNAL375_BASELINE_PYRADIOMICS_FEATURES.csv",
            model_dir / "derived_inputs" / "V3_5_D1_EXTERNAL33_FEATURES_LABEL_BLIND.csv",
        ]
        missing = [str(path) for path in locked_required if not path.exists()]
        if missing:
            raise RuntimeError(f"C locked-input replay is missing public locked inputs: {missing}")
    base, external_crosswalk, domains = stable.load_base()
    stable.load_external_labels(external_crosswalk)
    tabular_internal, tabular_external, tabular_features, _ = stable.load_tabular_features(base, external_crosswalk)

    if use_regenerated_tabular_inputs:
        genesis_internal = stack_by_ids(contract, base["neutral_imaging_id"].astype(str).tolist(), "whole_chest_imaging_model_id", "genesis", output_dir)
        genesis_external = stack_by_ids(contract, external_crosswalk["neutral_external_id"].astype(str).tolist(), "external_tensor_id", "genesis", output_dir)
    else:
        genesis_internal = np.load(model_dir / "embeddings" / "genesis_internal_embeddings.npy")
        genesis_external = np.load(model_dir / "embeddings" / "genesis_external_embeddings.npy")
    tangerine_manifest_internal = pd.read_csv(PROJECT_ROOT / "models" / "whole_chest_ct" / "internal_embedding_manifest.csv")
    tangerine_manifest_external = pd.read_csv(PROJECT_ROOT / "models" / "whole_chest_ct" / "external_embedding_manifest.csv")
    if use_regenerated_tabular_inputs:
        tangerine_internal = stack_by_ids(contract, tangerine_manifest_internal.sort_values("embedding_index")["neutral_imaging_id"].astype(str).tolist(), "whole_chest_imaging_model_id", "tangerine", output_dir)
        tangerine_external = stack_by_ids(contract, tangerine_manifest_external.sort_values("embedding_index")["neutral_external_id"].astype(str).tolist(), "external_tensor_id", "tangerine", output_dir)
    else:
        tangerine_internal = np.load(model_dir / "embeddings" / "tangerine_internal_embeddings.npy")
        tangerine_external = np.load(model_dir / "embeddings" / "tangerine_external_embeddings.npy")

    embedding_manifest_internal = {
        "FAMILY_C_GENESIS_FROZEN": base.copy(),
        "FAMILY_D_TANGERINE_FROZEN": tangerine_manifest_internal.merge(base, on="neutral_imaging_id", how="left").sort_values("embedding_index")[["neutral_imaging_id", "label", "outer_fold", "primary_domain"]],
    }
    embedding_manifest_external = {
        "FAMILY_C_GENESIS_FROZEN": external_crosswalk.copy(),
        "FAMILY_D_TANGERINE_FROZEN": tangerine_manifest_external.merge(external_crosswalk, on="neutral_external_id", how="left").sort_values("embedding_index")[["neutral_external_id", "neutral_patient_id"]],
    }
    embedding_internal = {
        "FAMILY_C_GENESIS_FROZEN": genesis_internal,
        "FAMILY_D_TANGERINE_FROZEN": tangerine_internal,
    }
    embedding_external = {
        "FAMILY_C_GENESIS_FROZEN": genesis_external,
        "FAMILY_D_TANGERINE_FROZEN": tangerine_external,
    }
    oof, external_long, _, _ = stable.run_standard(
        base,
        domains,
        tabular_internal,
        tabular_external,
        tabular_features,
        embedding_manifest_internal,
        embedding_manifest_external,
        embedding_internal,
        embedding_external,
    )
    oof, external_long, _ = stable.add_equal_ensemble(oof, external_long)
    external_mean = external_long.groupby(["neutral_external_id", "neutral_patient_id", "candidate"], as_index=False)["prob_COP_fold"].mean().rename(columns={"prob_COP_fold": "prob_COP_mean"})
    oof.to_csv(output_dir / f"stable_multirepresentation_{mode}_internal_replay.csv", index=False)
    external_mean.to_csv(output_dir / f"stable_multirepresentation_{mode}_external_replay.csv", index=False)

    candidate = "SP6_STABLE_MULTI_REPRESENTATION_RANDOM_FOREST"
    replay_oof = oof[oof["candidate"].eq(candidate)]
    replay_external = external_mean[external_mean["candidate"].eq(candidate)]
    locked_oof = pd.read_csv(model_dir / "internal_oof_predictions.csv")
    locked_external = pd.read_csv(model_dir / "external_predictions.csv")
    locked_oof = locked_oof[locked_oof["candidate"].eq(candidate)]
    locked_external = locked_external[locked_external["candidate"].eq(candidate)]
    locked_external_probability_counts = locked_external.groupby("neutral_external_id")["prob_COP_mean"].nunique(dropna=False)
    if not locked_external_probability_counts.eq(1).all():
        raise RuntimeError("Locked SP6 external fold rows disagree on their patient-level mean probability")
    locked_external = locked_external.groupby("neutral_external_id", as_index=False)["prob_COP_mean"].first()
    internal_joined = replay_oof.merge(locked_oof[["neutral_imaging_id", "prob_COP"]], on="neutral_imaging_id", suffixes=("_replay", "_locked"), validate="one_to_one")
    external_joined = replay_external.merge(locked_external[["neutral_external_id", "prob_COP_mean"]], on="neutral_external_id", suffixes=("_replay", "_locked"), validate="one_to_one")
    internal_delta = np.abs(internal_joined["prob_COP_replay"] - internal_joined["prob_COP_locked"])
    external_delta = np.abs(external_joined["prob_COP_mean_replay"] - external_joined["prob_COP_mean_locked"])
    internal_contract = contract[contract["cohort"].eq("internal")]
    whole_namespace = internal_contract[["public_subject_id", "whole_chest_imaging_model_id"]].rename(
        columns={"public_subject_id": "whole_public_subject_id", "whole_chest_imaging_model_id": "historical_join_id"}
    )
    lesion_namespace = internal_contract[["public_subject_id", "lesion_imaging_model_id"]].rename(
        columns={"public_subject_id": "lesion_public_subject_id", "lesion_imaging_model_id": "historical_join_id"}
    )
    namespace_join = whole_namespace.merge(lesion_namespace, on="historical_join_id", how="left", validate="one_to_one")
    same_subject = namespace_join["whole_public_subject_id"].eq(namespace_join["lesion_public_subject_id"])
    return {
        "input_mode": mode,
        "scope": (
            "raw-DICOM deep representations plus regenerated lesion phenotype and PyRadiomics tabular families"
            if use_regenerated_tabular_inputs
            else "locked deep-representation and tabular intermediate inputs"
        ),
        "internal_subjects": len(replay_oof),
        "external_subjects": len(replay_external),
        "internal_probability_max_abs_delta": float(internal_delta.max()),
        "internal_probability_mean_abs_delta": float(internal_delta.mean()),
        "external_probability_max_abs_delta": float(external_delta.max()),
        "external_probability_mean_abs_delta": float(external_delta.mean()),
        "historical_internal_tabular_join_same_subjects": int(same_subject.sum()),
        "historical_internal_tabular_join_mismatched_subjects": int((~same_subject).sum()),
        "historical_join_warning": "The historical pipeline joined branch-specific reused imaging IDs as one namespace; preserve only for locked-result audit, not as a patient-matched design.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay P95 tensors, frozen representations, and locked B/C downstream models for all 408 subjects.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--segmentation-audit", type=Path, default=DEFAULT_SEGMENTATION_AUDIT)
    parser.add_argument("--phenotype-features", type=Path)
    parser.add_argument("--radiomics-features", type=Path)
    parser.add_argument(
        "--phase",
        choices=["representations", "whole_chest_model", "stable_model", "stable_locked_inputs", "models", "all"],
        default="all",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = input_contract(
        args.segmentation_audit.resolve(),
        require_segmentation_audit=args.phase != "stable_locked_inputs",
    )
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if args.phase in {"representations", "all"}:
        replay_representations(contract, output_dir, device)
    if args.phase in {"whole_chest_model", "stable_model", "stable_locked_inputs", "models", "all"}:
        summary_path = output_dir / "full_cohort_whole_chest_replay_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        if args.phase in {"whole_chest_model", "models", "all"}:
            summary["whole_chest_foundation_transfer"] = replay_whole_chest_probe(contract, output_dir)
        if args.phase in {"stable_model", "models", "all"}:
            summary["stable_multirepresentation"] = replay_stable_multirepresentation(
                contract,
                output_dir,
                use_regenerated_tabular_inputs=True,
                phenotype_path=args.phenotype_features.resolve() if args.phenotype_features else None,
                radiomics_path=args.radiomics_features.resolve() if args.radiomics_features else None,
            )
        if args.phase == "stable_locked_inputs":
            summary["stable_multirepresentation"] = replay_stable_multirepresentation(
                contract,
                output_dir,
                use_regenerated_tabular_inputs=False,
            )
        write_json(summary_path, summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
