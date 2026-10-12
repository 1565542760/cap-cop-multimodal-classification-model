from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cap_cop.imaging.full_chest_densenet.model import load_checkpoint, probability
from cap_cop.imaging.full_chest_densenet.preprocess import preprocess_dicom, sha256_file


DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "full_reproduction_20261010" / "supplementary_exploratory_densenet"
CROSSWALK = PROJECT_ROOT / "data" / "public_subject_model_crosswalk.csv"
DEFAULT_SEGMENTATION_AUDIT = PROJECT_ROOT / "outputs" / "canonical_a_formal_20261011" / "segmentation" / "full_cohort_segmentation_replay_audit.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "supplementary_exploratory" / "full_chest_densenet"
INTERNAL_INFERENCE_BATCH_SIZE = 2
LOCKED_CACHE_MANIFEST = MODEL_DIR / "locked_processed_ct_manifest.csv"


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def contract(segmentation_audit: Path) -> pd.DataFrame:
    crosswalk = pd.read_csv(CROSSWALK, dtype=str).fillna("")
    if not segmentation_audit.exists():
        raise RuntimeError(f"DICOM replay requires a segmentation audit: {segmentation_audit}")
    audit = pd.read_csv(segmentation_audit, dtype=str).fillna("")
    frame = crosswalk.merge(audit[["public_subject_id", "dicom_dir", "status"]], on="public_subject_id", validate="one_to_one")
    if len(frame) != 408 or not frame["status"].eq("PASS").all():
        raise RuntimeError("Expected 408 DICOM contracts backed by passing audit rows")
    frame["label_binary"] = frame["label_binary"].astype(int)
    return frame


def preprocess_all(frame: pd.DataFrame, output_dir: Path, validate_locked_cache: bool) -> dict[str, Any]:
    tensor_dir = output_dir / "runtime_tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    cache_by_public: dict[str, dict[str, str]] = {}
    if validate_locked_cache:
        locked = pd.read_csv(LOCKED_CACHE_MANIFEST, dtype=str).fillna("")
        cache_by_public = locked.set_index("case_id_public").to_dict("index")
    rows = []
    for index, row in enumerate(frame.to_dict("records"), start=1):
        public_id = str(row["public_subject_id"])
        tensor_path = tensor_dir / f"{public_id}_network_tensor.npy"
        if tensor_path.exists():
            tensor_array = np.load(tensor_path)
            normalized = None
            metadata = json.loads((tensor_dir / f"{public_id}_preprocessing.json").read_text(encoding="utf-8"))
        else:
            normalized, tensor, metadata = preprocess_dicom(Path(row["dicom_dir"]))
            tensor_array = tensor.squeeze(0).squeeze(0).numpy().astype(np.float32, copy=False)
            np.save(tensor_path, tensor_array, allow_pickle=False)
            write_json(tensor_dir / f"{public_id}_preprocessing.json", metadata)

        cache_status = "NOT_REQUESTED"
        cache_max_delta = None
        cache_mean_delta = None
        neutral_image_id = ""
        if validate_locked_cache and row["cohort"] == "internal":
            if public_id not in cache_by_public:
                raise RuntimeError(f"No locked processed-CT cache row for {public_id}")
            cache_row = cache_by_public[public_id]
            neutral_image_id = cache_row["neutral_image_id"]
            cache_path = Path(cache_row["processed_ct_path"])
            if normalized is None:
                normalized, _, _ = preprocess_dicom(Path(row["dicom_dir"]))
            locked_array = np.load(cache_path)["array"].astype(np.float32, copy=False)
            if normalized.shape != locked_array.shape:
                raise RuntimeError(f"Locked cache shape mismatch for {public_id}: {normalized.shape} != {locked_array.shape}")
            delta = np.abs(normalized - locked_array)
            cache_max_delta = float(delta.max())
            cache_mean_delta = float(delta.mean())
            cache_status = "EXACT" if np.array_equal(normalized, locked_array) else "DIFF"
        rows.append(
            {
                "public_subject_id": public_id,
                "cohort": row["cohort"],
                "neutral_image_id": neutral_image_id,
                "tensor_path": str(tensor_path),
                "tensor_sha256": sha256_file(tensor_path),
                "tensor_shape": "x".join(map(str, tensor_array.shape)),
                "locked_cache_validation": cache_status,
                "locked_cache_max_abs_delta": cache_max_delta,
                "locked_cache_mean_abs_delta": cache_mean_delta,
            }
        )
        if index % 5 == 0 or index == len(frame):
            pd.DataFrame(rows).to_csv(output_dir / "exploratory_densenet_preprocessing_audit.csv", index=False, encoding="utf-8-sig")
            print(f"exploratory DenseNet preprocessing {index}/{len(frame)} last={public_id}", flush=True)
    audit = pd.DataFrame(rows)
    return {
        "subjects": len(audit),
        "internal_exact_cache_matches": int(audit["locked_cache_validation"].eq("EXACT").sum()),
        "internal_cache_differences": int(audit["locked_cache_validation"].eq("DIFF").sum()),
        "validation_requested": validate_locked_cache,
    }


def replay_models(frame: pd.DataFrame, output_dir: Path, device: torch.device) -> dict[str, Any]:
    tensors = output_dir / "runtime_tensors"
    locked_cache = pd.read_csv(LOCKED_CACHE_MANIFEST, dtype=str).fillna("")
    public_by_neutral = locked_cache.set_index("neutral_image_id")["case_id_public"].to_dict()
    membership = pd.read_csv(MODEL_DIR / "internal_oof_membership.csv", dtype={"seed": str, "fold": int, "neutral_image_id": str})
    checkpoint_paths = sorted((MODEL_DIR / "checkpoints").glob("seed_*_fold_*_best.pt"))
    if len(checkpoint_paths) != 15:
        raise RuntimeError(f"Expected 15 locked checkpoints, found {len(checkpoint_paths)}")

    internal_rows = []
    external_rows = []
    external_contract = frame[frame["cohort"].eq("external")].sort_values("clinical_model_id")
    for index, checkpoint_path in enumerate(checkpoint_paths, start=1):
        match = re.fullmatch(r"seed_(\d+)_fold_(\d+)_best\.pt", checkpoint_path.name)
        if match is None:
            raise RuntimeError(f"Cannot parse checkpoint identity: {checkpoint_path.name}")
        seed, fold = match.group(1), int(match.group(2))
        model = load_checkpoint(checkpoint_path, device)
        held_out = membership[(membership["seed"].eq(seed)) & (membership["fold"].eq(fold))]
        if len(held_out) != 75:
            raise RuntimeError(f"Expected 75 held-out rows for seed={seed} fold={fold}, found {len(held_out)}")
        held_out = held_out.sort_values("neutral_image_id", kind="mergesort").reset_index(drop=True)
        # The locked Stage 4 validation path used DataLoader(batch_size=2,
        # shuffle=False). Preserve that batch shape because CUDA convolution
        # kernels can produce small, deterministic differences for batch=1.
        for start in range(0, len(held_out), INTERNAL_INFERENCE_BATCH_SIZE):
            batch_members = held_out.iloc[start : start + INTERNAL_INFERENCE_BATCH_SIZE]
            public_ids = [public_by_neutral[str(value)] for value in batch_members["neutral_image_id"]]
            batch_array = np.stack(
                [np.load(tensors / f"{public_id}_network_tensor.npy") for public_id in public_ids],
                axis=0,
            )
            batch_tensor = torch.from_numpy(batch_array)[:, None]
            with torch.no_grad():
                probabilities = torch.softmax(model(batch_tensor.to(device=device, dtype=torch.float32)), dim=1)[:, 1]
            for member, public_id, prob in zip(
                batch_members.itertuples(index=False),
                public_ids,
                probabilities.detach().cpu().tolist(),
            ):
                internal_rows.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "neutral_image_id": member.neutral_image_id,
                        "public_subject_id": public_id,
                        "label": int(member.label),
                        "probability_COP": float(prob),
                    }
                )
        for external in external_contract.itertuples(index=False):
            tensor = torch.from_numpy(np.load(tensors / f"{external.public_subject_id}_network_tensor.npy"))[None, None]
            external_rows.append(
                {
                    "checkpoint": checkpoint_path.name,
                    "public_subject_id": external.public_subject_id,
                    "neutral_patient_id": external.clinical_model_id,
                    "probability_COP": probability(model, tensor, device),
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"exploratory DenseNet checkpoints {index}/15 {checkpoint_path.name}", flush=True)

    internal_long = pd.DataFrame(internal_rows)
    external_long = pd.DataFrame(external_rows)
    internal_mean = internal_long.groupby(["neutral_image_id", "public_subject_id", "label"], as_index=False)["probability_COP"].mean()
    external_mean = external_long.groupby(["public_subject_id", "neutral_patient_id"], as_index=False)["probability_COP"].mean()
    internal_long.to_csv(output_dir / "exploratory_densenet_internal_member_predictions.csv", index=False)
    external_long.to_csv(output_dir / "exploratory_densenet_external_member_predictions.csv", index=False)
    internal_mean.to_csv(output_dir / "exploratory_densenet_internal_oof_replay.csv", index=False)
    external_mean.to_csv(output_dir / "exploratory_densenet_external_canonical_replay.csv", index=False)

    locked_internal = pd.read_csv(MODEL_DIR / "internal_oof_predictions.csv")
    locked_external = pd.read_csv(MODEL_DIR / "external_label_blind_predictions.csv")
    internal_joined = internal_mean.merge(locked_internal, on="neutral_image_id", suffixes=("_replay", "_locked"), validate="one_to_one")
    external_joined = external_mean.merge(locked_external, on="neutral_patient_id", suffixes=("_replay", "_locked"), validate="one_to_one")
    internal_delta = np.abs(internal_joined["probability_COP"] - internal_joined["prob_COP_mean_across_3_seed_oof"])
    external_delta = np.abs(external_joined["probability_COP_replay"] - external_joined["probability_COP_locked"])
    return {
        "internal_subjects": len(internal_mean),
        "external_subjects": len(external_mean),
        "internal_member_predictions": len(internal_long),
        "external_member_predictions": len(external_long),
        "internal_probability_max_abs_delta": float(internal_delta.max()),
        "internal_probability_mean_abs_delta": float(internal_delta.mean()),
        "external_probability_max_abs_delta": float(external_delta.max()),
        "external_probability_mean_abs_delta": float(external_delta.mean()),
        "external_reference_boundary": "Replay uses one canonical series per External33 subject and 15 checkpoints; historical exploratory lock averaged two source versions and 30 predictions per subject.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay the supplementary exploratory whole-chest DenseNet from DICOM.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", choices=["preprocess", "models", "all"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--segmentation-audit", type=Path, default=DEFAULT_SEGMENTATION_AUDIT)
    parser.add_argument("--validate-locked-cache", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = contract(args.segmentation_audit.resolve())
    summary: dict[str, Any] = {}
    if args.phase in {"preprocess", "all"}:
        summary["preprocessing"] = preprocess_all(frame, output_dir, args.validate_locked_cache)
    if args.phase in {"models", "all"}:
        device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
        summary["model_replay"] = replay_models(frame, output_dir, device)
    write_json(output_dir / "full_cohort_exploratory_densenet_replay_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
