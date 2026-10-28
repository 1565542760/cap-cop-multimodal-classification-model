from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from monai.networks.nets import DenseNet121

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cap_cop.clinical.structural_missingness_model import load_locked_artifact
from cap_cop.data.release_inputs import dicom_dir, load_clinical_cohort
from cap_cop.private_lock_selectors import get_selector

TARGET_SHAPE = (96, 96, 96)
SPACING = (1.5, 1.5, 1.5)
HU_MIN = -1000.0
HU_MAX = 400.0
FUSION_WEIGHT_CLINICAL = 0.50
FUSION_WEIGHT_IMAGE = 0.50


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_dicom_series(dicom_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if series_ids:
        best_files: list[str] = []
        for series_id in series_ids:
            files = list(reader.GetGDCMSeriesFileNames(str(dicom_dir), series_id))
            if len(files) > len(best_files):
                best_files = files
        reader.SetFileNames(best_files)
        return reader.Execute()
    files = [str(p) for p in sorted(dicom_dir.iterdir()) if p.is_file()]
    if not files:
        raise RuntimeError(f"No DICOM files found in {dicom_dir}")
    reader.SetFileNames(files)
    return reader.Execute()


def make_reference_image(img: sitk.Image) -> sitk.Image:
    orig_size = np.array(list(img.GetSize()), dtype=np.float64)
    orig_spacing = np.array(list(img.GetSpacing()), dtype=np.float64)
    target_spacing = np.array(SPACING, dtype=np.float64)
    new_size = np.maximum(np.round(orig_size * (orig_spacing / target_spacing)).astype(np.int64), 1)
    ref = sitk.Image([int(x) for x in new_size.tolist()], sitk.sitkFloat32)
    ref.SetOrigin(img.GetOrigin())
    ref.SetDirection(img.GetDirection())
    ref.SetSpacing(SPACING)
    return ref


def dicom_to_network_tensor(dicom_dir: Path, output_dir: Path, subject_id: str) -> tuple[torch.Tensor, dict[str, Any]]:
    img = read_dicom_series(dicom_dir)
    ref = make_reference_image(img)
    resampled = sitk.Resample(img, ref, sitk.Transform(), sitk.sitkLinear, HU_MIN, sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(resampled).astype(np.float32)
    norm = (np.clip(arr, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)
    tensor = torch.from_numpy(norm)[None, None]
    if tuple(norm.shape) != TARGET_SHAPE:
        tensor = F.interpolate(tensor, size=TARGET_SHAPE, mode="trilinear", align_corners=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_np = tensor.squeeze(0).squeeze(0).numpy().astype(np.float32)
    tensor_path = output_dir / f"{subject_id}_supplementary_exploratory_96cubed_tensor.npy"
    np.save(tensor_path, tensor_np)
    metadata = {
        "dicom_dir": str(dicom_dir),
        "dicom_file_count": len([p for p in dicom_dir.iterdir() if p.is_file()]),
        "original_size_xyz": list(img.GetSize()),
        "original_spacing_xyz": list(img.GetSpacing()),
        "resampled_shape_zyx": list(norm.shape),
        "network_shape_zyx": list(tensor_np.shape),
        "spacing_xyz": list(SPACING),
        "hu_window": [HU_MIN, HU_MAX],
        "normalization": "[0,1]",
        "tensor_path": str(tensor_path),
        "tensor_sha256": sha256_file(tensor_path),
    }
    return tensor, metadata


def make_densenet() -> torch.nn.Module:
    return DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)


def load_external_tables() -> pd.DataFrame:
    return load_clinical_cohort("external")


def public_to_neutral_patient(public_subject_id: str) -> str:
    crosswalk = pd.read_csv(PROJECT_ROOT / "data" / "public_subject_model_crosswalk.csv", dtype=str)
    hit = crosswalk[
        crosswalk["public_subject_id"].eq(public_subject_id)
        & crosswalk["cohort"].eq("external")
    ]
    if len(hit) != 1:
        raise RuntimeError(f"Expected one explicit external crosswalk row for {public_subject_id}; found {len(hit)}")
    return str(hit.iloc[0]["clinical_model_id"])


def dicom_dir_for_public_subject(public_subject_id: str) -> Path:
    row = load_clinical_row(public_subject_id)
    label = int(row.iloc[0]["label_binary"])
    if label not in (0, 1):
        raise ValueError("Expected a binary diagnosis label")
    diagnosis = {0: "CAP", 1: "COP"}[label]
    return dicom_dir("external", diagnosis, public_subject_id)


def load_clinical_row(public_subject_id: str) -> pd.DataFrame:
    clinical = load_external_tables()
    row = clinical[clinical["public_subject_id"].eq(public_subject_id)].copy()
    if len(row) != 1:
        raise RuntimeError(f"Expected exactly one clinical row for {public_subject_id}, found {len(row)}")
    return row


def predict_structural_missingness_clinical(public_subject_id: str) -> dict[str, Any]:
    row = load_clinical_row(public_subject_id)
    model_dir = PROJECT_ROOT / "models" / "supplementary_exploratory" / "structural_missingness_clinical" / "ensemble_models"
    model_paths = sorted(model_dir.glob("*.joblib"))
    if len(model_paths) != 15:
        raise RuntimeError(f"Expected 15 structural-missingness clinical models, found {len(model_paths)}")
    probs = []
    hashes = []
    for path in model_paths:
        model = load_locked_artifact(path)
        probs.append(float(model.predict_proba(row)[0]))
        hashes.append(sha256_file(path))
    return {
        "clinical_probability": float(np.mean(probs)),
        "clinical_member_probabilities": probs,
        "clinical_model_count": len(model_paths),
        "clinical_model_sha256": ";".join(hashes),
    }


def predict_full_chest_densenet(tensor: torch.Tensor) -> dict[str, Any]:
    ckpt_dir = PROJECT_ROOT / "models" / "supplementary_exploratory" / "full_chest_densenet" / "checkpoints"
    ckpt_paths = sorted(ckpt_dir.glob("*.pt"))
    if len(ckpt_paths) != 15:
        raise RuntimeError(f"Expected 15 full-chest DenseNet checkpoints, found {len(ckpt_paths)}")
    device = torch.device("cpu")
    tensor = tensor.to(device=device, dtype=torch.float32)
    probs = []
    hashes = []
    with torch.no_grad():
        for path in ckpt_paths:
            model = make_densenet().to(device)
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            logits = model(tensor)
            probs.append(float(torch.softmax(logits, dim=1)[0, 1].detach().cpu()))
            hashes.append(sha256_file(path))
            del model
    return {
        "image_probability": float(np.mean(probs)),
        "image_member_probabilities": probs,
        "image_checkpoint_count": len(ckpt_paths),
        "image_checkpoint_sha256": ";".join(hashes),
    }


def load_locked_references(neutral_patient_id: str) -> dict[str, Any]:
    image_path = PROJECT_ROOT / "models" / "supplementary_exploratory" / "full_chest_densenet" / "external_label_blind_predictions.csv"
    image = pd.read_csv(image_path)
    image_row = image[image["neutral_patient_id"].eq(neutral_patient_id)]
    if len(image_row) != 1:
        raise RuntimeError(f"Expected one locked image row for {neutral_patient_id}, found {len(image_row)}")
    fusion_path = PROJECT_ROOT / "models" / "multimodal_fusion" / "external_all_pairs_label_free_predictions.csv"
    fusion = pd.read_csv(fusion_path)
    pair = get_selector("supplementary_exploratory_comparator")
    fusion_row = fusion[
        fusion["pair_id"].eq(pair)
        & fusion["patient_id"].eq(neutral_patient_id)
    ]
    if len(fusion_row) != 1:
        raise RuntimeError(f"Expected one locked fusion row for {neutral_patient_id}, found {len(fusion_row)}")
    return {
        "locked_image_probability": float(image_row.iloc[0]["probability_COP"]),
        "locked_multimodal_clinical_probability": float(fusion_row.iloc[0]["p_clinical"]),
        "locked_multimodal_image_probability": float(fusion_row.iloc[0]["p_image"]),
        "locked_multimodal_probability": float(fusion_row.iloc[0]["p_weighted_fusion"]),
        "locked_clinical_weight": float(fusion_row.iloc[0]["final_weight_clinical"]),
        "locked_image_weight": float(fusion_row.iloc[0]["final_weight_image"]),
        "locked_image_source": str(image_path),
        "locked_fusion_source": str(fusion_path),
    }


def run(public_subject_id: str, output_dir: Path) -> dict[str, Any]:
    neutral_patient_id = public_to_neutral_patient(public_subject_id)
    dicom_dir = dicom_dir_for_public_subject(public_subject_id)
    tensor, tensor_meta = dicom_to_network_tensor(dicom_dir, output_dir, public_subject_id)
    clinical = predict_structural_missingness_clinical(public_subject_id)
    image = predict_full_chest_densenet(tensor)
    fused = FUSION_WEIGHT_CLINICAL * clinical["clinical_probability"] + FUSION_WEIGHT_IMAGE * image["image_probability"]
    locked = load_locked_references(neutral_patient_id)
    row = {
        "public_subject_id": public_subject_id,
        "neutral_patient_id": neutral_patient_id,
        **tensor_meta,
        **clinical,
        **image,
        "clinical_weight": FUSION_WEIGHT_CLINICAL,
        "image_weight": FUSION_WEIGHT_IMAGE,
        "replay_multimodal_probability": float(fused),
        **locked,
    }
    row["clinical_probability_delta_vs_locked_fusion_table"] = (
        row["clinical_probability"] - row["locked_multimodal_clinical_probability"]
    )
    row["image_probability_delta_vs_locked_image_table"] = row["image_probability"] - row["locked_image_probability"]
    row["multimodal_probability_delta_vs_locked_fusion_table"] = (
        row["replay_multimodal_probability"] - row["locked_multimodal_probability"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "supplementary_exploratory_subset_replay.csv", [row])
    (output_dir / "supplementary_exploratory_subset_replay.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the structural-missingness clinical and full-chest DenseNet comparator for one local case.")
    parser.add_argument("--public-subject-id", default="EXT_CAP_001")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "supplementary_exploratory_subset_replay",
    )
    args = parser.parse_args()
    row = run(args.public_subject_id, args.output_dir)
    print(json.dumps({k: v for k, v in row.items() if "probabilities" not in k and "sha256" not in k}, indent=2))


if __name__ == "__main__":
    main()
