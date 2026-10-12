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
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
TANGERINE_VENDOR_ROOT = SRC_ROOT / "cap_cop" / "vendor" / "tangerine_3d_mae"
if str(TANGERINE_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(TANGERINE_VENDOR_ROOT))

from cap_cop.imaging.whole_chest_tensor.generate_whole_chest_tensor import materialize_subject
from cap_cop.vendor.tangerine_3d_mae import models_vit


SEED = 20260909


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict) and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise RuntimeError("No tensor state_dict found in TANGERINE checkpoint")


def load_foundation_encoder(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = extract_state_dict(checkpoint)
    model = models_vit.vit_large_patch16_yo(num_classes=1, global_pool=True, img_size=256)
    model_state = model.state_dict()
    loadable = {
        key: value
        for key, value in state.items()
        if key in model_state
        and torch.is_tensor(value)
        and tuple(value.shape) == tuple(model_state[key].shape)
    }
    model.load_state_dict(loadable, strict=False)
    model.eval().to(device)
    return model


def prepare_tensor_for_encoder(tensor_path: Path, device: torch.device) -> torch.Tensor:
    arr = np.load(tensor_path).astype(np.float32, copy=False)
    if arr.shape == (1, 96, 96, 96):
        arr = arr[None, ...]
    elif arr.shape == (96, 96, 96):
        arr = arr[None, None, ...]
    elif arr.ndim == 5:
        pass
    else:
        raise ValueError(f"Unexpected tensor shape {arr.shape} for {tensor_path}")
    x = torch.from_numpy(arr)
    x = F.interpolate(x, size=(256, 256, 256), mode="trilinear", align_corners=False)
    return x.to(device=device, non_blocking=False)


@torch.no_grad()
def extract_foundation_embedding(model: torch.nn.Module, tensor_path: Path, device: torch.device) -> np.ndarray:
    x = prepare_tensor_for_encoder(tensor_path, device)
    if x.is_cuda:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            emb = model.forward_features(x)
    else:
        emb = model.forward_features(x)
    emb_np = emb.detach().float().cpu().numpy()
    if emb_np.shape != (1, 2048):
        raise RuntimeError(f"Unexpected embedding shape {emb_np.shape}; expected (1, 2048)")
    return emb_np.astype(np.float32, copy=False)


def fixed_linear_probe() -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=5000,
            class_weight="balanced",
            random_state=SEED,
        ),
    )


def external_probability_from_locked_internal_models(model_dir: Path, embedding: np.ndarray) -> tuple[float, float, list[float]]:
    internal_manifest = pd.read_csv(model_dir / "internal_embedding_manifest.csv")
    locked_internal = pd.read_csv(model_dir / "internal_oof_predictions.csv")
    internal_embeddings = np.load(model_dir / "embeddings" / "internal_embeddings.npy")

    labels_folds = locked_internal[
        ["neutral_imaging_id", "label", "fold_id", "primary_domain"]
    ].drop_duplicates()
    df = internal_manifest.merge(labels_folds, on="neutral_imaging_id", how="left")
    if df[["label", "fold_id"]].isna().any().any():
        raise RuntimeError("Internal embedding manifest could not be fully matched to locked OOF labels/folds")
    df = df.sort_values("embedding_index").reset_index(drop=True)
    if list(df["embedding_index"]) != list(range(len(df))):
        raise RuntimeError("Internal embedding_index is not contiguous and sorted")

    y = df["label"].to_numpy(dtype=int)
    folds = df["fold_id"].to_numpy(dtype=int)
    fold_probs: list[float] = []
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        clf = fixed_linear_probe()
        clf.fit(internal_embeddings[train], y[train])
        fold_probs.append(float(clf.predict_proba(embedding)[:, 1][0]))
    return float(np.mean(fold_probs)), float(np.std(fold_probs)), fold_probs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay one whole-chest foundation-transfer case from DICOM to P95 tensor, frozen embedding, and locked-probe probability."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--locked-external-id", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tensor_qc = materialize_subject(project_root, args.subject_id, output_dir / "p95_tensor")
    tensor_path = Path(tensor_qc["tensor_path"])

    requested = torch.device(args.device)
    device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
    checkpoint_path = project_root / "models" / "whole_chest_ct" / "foundation_encoder" / "tangerine_foundation_encoder.pth"
    model = load_foundation_encoder(checkpoint_path, device)
    embedding = extract_foundation_embedding(model, tensor_path, device)

    embedding_path = output_dir / f"{args.subject_id}_foundation_embedding.npy"
    np.save(embedding_path, embedding, allow_pickle=False)

    model_dir = project_root / "models" / "whole_chest_ct"
    prob_mean, prob_std, fold_probs = external_probability_from_locked_internal_models(model_dir, embedding)
    result = {
        "public_subject_id": args.subject_id,
        "locked_external_id": args.locked_external_id,
        "device": str(device),
        "tensor_path": str(tensor_path),
        "tensor_sha256": sha256_file(tensor_path),
        "embedding_path": str(embedding_path),
        "embedding_sha256": sha256_file(embedding_path),
        "embedding_shape": list(embedding.shape),
        "embedding_dtype": str(embedding.dtype),
        "foundation_checkpoint_path": str(checkpoint_path),
        "foundation_checkpoint_sha256": sha256_file(checkpoint_path),
        "prob_COP": prob_mean,
        "prob_COP_std": prob_std,
        "fold_prob_COP": fold_probs,
        "linear_probe_training_source": "locked internal whole-chest embeddings and locked OOF folds under models/whole_chest_ct",
        "replay_scope": "single external-subset DICOM-to-P95-to-frozen-embedding-to-locked-probe probability",
    }
    if args.locked_external_id:
        locked = pd.read_csv(model_dir / "external_predictions.csv")
        hit = locked[locked["neutral_external_id"].eq(args.locked_external_id)]
        if len(hit) == 1:
            result["locked_prob_COP"] = float(hit.iloc[0]["prob_COP"])
            result["locked_prob_COP_std"] = float(hit.iloc[0]["prob_COP_std"])
            result["prob_abs_delta_vs_locked"] = abs(prob_mean - result["locked_prob_COP"])
            result["prob_std_abs_delta_vs_locked"] = abs(prob_std - result["locked_prob_COP_std"])
        else:
            result["locked_comparison_status"] = f"locked external id matched {len(hit)} rows"

    result_path = output_dir / f"{args.subject_id}_whole_chest_foundation_subset_replay.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
