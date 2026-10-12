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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
TANGERINE_VENDOR_ROOT = SRC_ROOT / "cap_cop" / "vendor" / "tangerine_3d_mae"
if str(TANGERINE_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(TANGERINE_VENDOR_ROOT))

from cap_cop.pipelines import stable_multirepresentation_replay_pipeline as stable

from run_whole_chest_foundation_subset_replay import (
    extract_foundation_embedding,
    load_foundation_encoder,
)
from cap_cop.imaging.whole_chest_tensor.generate_whole_chest_tensor import materialize_subject
from cap_cop.vendor.models_genesis import unet3d as genesis_unet3d


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        state = checkpoint.get("state_dict", checkpoint)
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint type: {type(checkpoint)}")
    return {
        str(k).replace("module.", "").replace("model.", "").replace("encoder.", ""): v
        for k, v in state.items()
        if torch.is_tensor(v)
    }


def extract_genesis_embedding(tensor_path: Path, checkpoint_path: Path, device: torch.device) -> np.ndarray:
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = extract_state_dict(raw)
    model = genesis_unet3d.UNet3D(n_class=1, act="relu")
    model_state = model.state_dict()
    keep = {k: v for k, v in state.items() if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)}
    model.load_state_dict(keep, strict=False)
    model.to(device)
    model.eval()
    pool = nn.AdaptiveAvgPool3d(1)

    arr = np.load(tensor_path).astype(np.float32, copy=False)
    if arr.ndim == 3:
        arr = arr[None, None]
    elif arr.ndim == 4:
        arr = arr[None]
    elif arr.ndim != 5:
        raise ValueError(f"Unexpected tensor shape for Genesis: {arr.shape}")
    x = torch.from_numpy(arr).to(device)
    with torch.no_grad():
        out64, _ = model.down_tr64(x)
        out128, _ = model.down_tr128(out64)
        out256, _ = model.down_tr256(out128)
        out512, _ = model.down_tr512(out256)
        emb = pool(out512).flatten(1).detach().cpu().numpy()
    if emb.shape[0] != 1:
        raise RuntimeError(f"Unexpected Genesis embedding shape: {emb.shape}")
    return emb.astype(np.float32, copy=False)


def patch_stable_paths(project_root: Path, output_dir: Path) -> None:
    model_dir = project_root / "models" / "stable_multirepresentation_ct"
    derived = model_dir / "derived_inputs"
    stable.PATHS.update(
        {
            "internal_oof_ref": derived / "09_V3_4_STANDARD_INTERNAL_OOF_PREDICTIONS.csv",
            "domain_lock": derived / "03_PRIMARY_DG_DOMAIN_LOCK.json",
            "external_crosswalk": derived / "EXTERNAL_EXTV2_TO_EXTERNAL_PATIENT_CROSSWALK.csv",
            "external_labels": derived / "EXTERNAL_LOCKED_PREDICTIONS_WITH_LABELS_V2.csv",
            "lesion_features": derived / "06_LESION_PHENOTYPE_FEATURES.csv",
            "d1_internal_features": derived / "V3_5_D1_INTERNAL375_BASELINE_PYRADIOMICS_FEATURES.csv",
            "d1_external_features": derived / "V3_5_D1_EXTERNAL33_FEATURES_LABEL_BLIND.csv",
            "d1_feature_list": derived / "V3_5_D1_FINAL_FEATURE_LIST.csv",
            "tangerine_internal_embeddings": model_dir / "embeddings" / "tangerine_internal_embeddings.npy",
            "tangerine_internal_manifest": project_root / "models" / "whole_chest_ct" / "internal_embedding_manifest.csv",
            "tangerine_external_embeddings": model_dir / "embeddings" / "tangerine_external_embeddings.npy",
            "tangerine_external_manifest": project_root / "models" / "whole_chest_ct" / "external_embedding_manifest.csv",
            "genesis_checkpoint": model_dir / "foundation_encoders" / "genesis_chest_ct.pt",
            "genesis_code": SRC_ROOT / "cap_cop" / "vendor" / "models_genesis" / "unet3d.py",
            "internal_tensor_dir": output_dir / "not_used_internal_tensors",
            "external_tensor_dir": output_dir / "not_used_external_tensors",
        }
    )
    for key in stable.DIRS:
        stable.DIRS[key] = output_dir / "stable_runtime" / key


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay one stable multi-representation CT external case with regenerated Genesis and TANGERINE embeddings."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--locked-external-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_stable_paths(project_root, output_dir)

    tensor_qc = materialize_subject(project_root, args.subject_id, output_dir / "p95_tensor")
    tensor_path = Path(tensor_qc["tensor_path"])

    requested = torch.device(args.device)
    device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")

    tangerine_checkpoint = project_root / "models" / "whole_chest_ct" / "foundation_encoder" / "tangerine_foundation_encoder.pth"
    tangerine_model = load_foundation_encoder(tangerine_checkpoint, device)
    tangerine_embedding = extract_foundation_embedding(tangerine_model, tensor_path, device)

    genesis_checkpoint = project_root / "models" / "stable_multirepresentation_ct" / "foundation_encoders" / "genesis_chest_ct.pt"
    genesis_embedding = extract_genesis_embedding(tensor_path, genesis_checkpoint, device)

    model_dir = project_root / "models" / "stable_multirepresentation_ct"
    ext_xw = pd.read_csv(model_dir / "derived_inputs" / "EXTERNAL_EXTV2_TO_EXTERNAL_PATIENT_CROSSWALK.csv")
    ext_order = ext_xw["neutral_external_id"].astype(str).tolist()
    ext_index = ext_order.index(args.locked_external_id)

    genesis_ext = np.load(model_dir / "embeddings" / "genesis_external_embeddings.npy").copy()
    tangerine_ext = np.load(model_dir / "embeddings" / "tangerine_external_embeddings.npy").copy()
    genesis_locked_row = genesis_ext[ext_index : ext_index + 1].copy()
    tangerine_locked_row = tangerine_ext[ext_index : ext_index + 1].copy()
    genesis_ext[ext_index : ext_index + 1] = genesis_embedding
    tangerine_ext[ext_index : ext_index + 1] = tangerine_embedding

    base, ext_xw2, domains = stable.load_base()
    ext_labels = stable.load_external_labels(ext_xw2)
    tab_i, tab_e, tab_f, _ = stable.load_tabular_features(base, ext_xw2)
    emb_mi = {
        "FAMILY_C_GENESIS_FROZEN": base.copy(),
        "FAMILY_D_TANGERINE_FROZEN": pd.read_csv(project_root / "models" / "whole_chest_ct" / "internal_embedding_manifest.csv")
        .merge(base, on="neutral_imaging_id", how="left")
        .sort_values("embedding_index")[["neutral_imaging_id", "label", "outer_fold", "primary_domain"]],
    }
    emb_me = {
        "FAMILY_C_GENESIS_FROZEN": ext_xw2.copy(),
        "FAMILY_D_TANGERINE_FROZEN": pd.read_csv(project_root / "models" / "whole_chest_ct" / "external_embedding_manifest.csv")
        .merge(ext_xw2, on="neutral_external_id", how="left")
        .sort_values("embedding_index")[["neutral_external_id", "neutral_patient_id"]],
    }
    emb_i = {
        "FAMILY_C_GENESIS_FROZEN": np.load(model_dir / "embeddings" / "genesis_internal_embeddings.npy"),
        "FAMILY_D_TANGERINE_FROZEN": np.load(model_dir / "embeddings" / "tangerine_internal_embeddings.npy"),
    }
    emb_e = {
        "FAMILY_C_GENESIS_FROZEN": genesis_ext,
        "FAMILY_D_TANGERINE_FROZEN": tangerine_ext,
    }
    oof, ext_long, _, _ = stable.run_standard(base, domains, tab_i, tab_e, tab_f, emb_mi, emb_me, emb_i, emb_e)
    oof, ext_long, sp7_members = stable.add_equal_ensemble(oof, ext_long)
    ext_mean = (
        ext_long.groupby(["neutral_external_id", "neutral_patient_id", "candidate"], as_index=False)["prob_COP_fold"]
        .mean()
        .rename(columns={"prob_COP_fold": "prob_COP_mean"})
    )

    locked_ext = pd.read_csv(model_dir / "external_predictions.csv")
    locked_sp6 = locked_ext[
        locked_ext["neutral_external_id"].astype(str).eq(args.locked_external_id)
        & locked_ext["candidate"].eq("SP6_STABLE_MULTI_REPRESENTATION_RANDOM_FOREST")
    ]["prob_COP_mean"].iloc[0]
    locked_sp7 = locked_ext[
        locked_ext["neutral_external_id"].astype(str).eq(args.locked_external_id)
        & locked_ext["candidate"].eq("SP7_STABLE_REPRESENTATION_EQUAL_ENSEMBLE")
    ]["prob_COP_mean"].iloc[0]
    replay_sp6 = ext_mean[
        ext_mean["neutral_external_id"].astype(str).eq(args.locked_external_id)
        & ext_mean["candidate"].eq("SP6_STABLE_MULTI_REPRESENTATION_RANDOM_FOREST")
    ]["prob_COP_mean"].iloc[0]
    replay_sp7 = ext_mean[
        ext_mean["neutral_external_id"].astype(str).eq(args.locked_external_id)
        & ext_mean["candidate"].eq("SP7_STABLE_REPRESENTATION_EQUAL_ENSEMBLE")
    ]["prob_COP_mean"].iloc[0]

    np.save(output_dir / f"{args.subject_id}_genesis_embedding.npy", genesis_embedding, allow_pickle=False)
    np.save(output_dir / f"{args.subject_id}_tangerine_embedding.npy", tangerine_embedding, allow_pickle=False)
    ext_mean.to_csv(output_dir / "stable_multirepresentation_external_replay_mean_predictions.csv", index=False)

    report = {
        "public_subject_id": args.subject_id,
        "locked_external_id": args.locked_external_id,
        "device": str(device),
        "tensor_sha256": sha256_file(tensor_path),
        "genesis_embedding_shape": list(genesis_embedding.shape),
        "genesis_embedding_sha256": sha256_file(output_dir / f"{args.subject_id}_genesis_embedding.npy"),
        "genesis_embedding_max_abs_delta_vs_locked_row": float(np.max(np.abs(genesis_embedding - genesis_locked_row))),
        "genesis_embedding_mean_abs_delta_vs_locked_row": float(np.mean(np.abs(genesis_embedding - genesis_locked_row))),
        "tangerine_embedding_shape": list(tangerine_embedding.shape),
        "tangerine_embedding_sha256": sha256_file(output_dir / f"{args.subject_id}_tangerine_embedding.npy"),
        "tangerine_embedding_max_abs_delta_vs_locked_row": float(np.max(np.abs(tangerine_embedding - tangerine_locked_row))),
        "tangerine_embedding_mean_abs_delta_vs_locked_row": float(np.mean(np.abs(tangerine_embedding - tangerine_locked_row))),
        "sp6_replay_prob_COP_mean": float(replay_sp6),
        "sp6_locked_prob_COP_mean": float(locked_sp6),
        "sp6_abs_delta_vs_locked": float(abs(replay_sp6 - locked_sp6)),
        "sp7_replay_prob_COP_mean": float(replay_sp7),
        "sp7_locked_prob_COP_mean": float(locked_sp7),
        "sp7_abs_delta_vs_locked": float(abs(replay_sp7 - locked_sp7)),
        "sp7_members": sp7_members,
        "boundary": "Deep embeddings for the target external row were regenerated from local DICOM/P95. Handcrafted lesion phenotype and D1 radiomics tables were copied locked derived inputs, not regenerated for all 33 external cases in this script.",
    }
    report_path = output_dir / f"{args.subject_id}_stable_multirepresentation_subset_replay.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
