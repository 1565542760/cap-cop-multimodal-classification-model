from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as functional


TARGET_SPACING_XYZ = (1.5, 1.5, 1.5)
NETWORK_SHAPE_ZYX = (96, 96, 96)
HU_MIN = -1000.0
HU_MAX = 400.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_largest_dicom_series(dicom_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir)) or []
    if series_ids:
        files = max(
            (list(reader.GetGDCMSeriesFileNames(str(dicom_dir), series_id)) for series_id in series_ids),
            key=len,
        )
    else:
        files = [str(path) for path in sorted(dicom_dir.iterdir()) if path.is_file()]
    if not files:
        raise RuntimeError(f"No readable DICOM files in {dicom_dir}")
    reader.SetFileNames(files)
    return reader.Execute()


def resample_and_normalize(image: sitk.Image) -> np.ndarray:
    size = np.asarray(image.GetSize(), dtype=np.float64)
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    target_spacing = np.asarray(TARGET_SPACING_XYZ, dtype=np.float64)
    target_size = np.maximum(np.round(size * spacing / target_spacing).astype(np.int64), 1)
    reference = sitk.Image([int(value) for value in target_size], sitk.sitkFloat32)
    reference.SetOrigin(image.GetOrigin())
    reference.SetDirection(image.GetDirection())
    reference.SetSpacing(TARGET_SPACING_XYZ)
    resampled = sitk.Resample(
        image,
        reference,
        sitk.Transform(),
        sitk.sitkLinear,
        HU_MIN,
        sitk.sitkFloat32,
    )
    array = sitk.GetArrayFromImage(resampled).astype(np.float32)
    return ((np.clip(array, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32, copy=False)


def network_tensor(array: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(array.astype(np.float32, copy=False))[None, None]
    if tuple(array.shape) != NETWORK_SHAPE_ZYX:
        tensor = functional.interpolate(tensor, size=NETWORK_SHAPE_ZYX, mode="trilinear", align_corners=False)
    return tensor


def preprocess_dicom(dicom_dir: Path) -> tuple[np.ndarray, torch.Tensor, dict[str, Any]]:
    image = read_largest_dicom_series(dicom_dir)
    normalized = resample_and_normalize(image)
    tensor = network_tensor(normalized)
    metadata = {
        "dicom_dir": str(dicom_dir),
        "original_size_xyz": list(image.GetSize()),
        "original_spacing_xyz": list(image.GetSpacing()),
        "original_origin_xyz": list(image.GetOrigin()),
        "original_direction": list(image.GetDirection()),
        "resampled_shape_zyx": list(normalized.shape),
        "network_shape_zyx": list(tensor.shape[-3:]),
        "target_spacing_xyz": list(TARGET_SPACING_XYZ),
        "hu_window": [HU_MIN, HU_MAX],
        "normalization": "fixed_minmax_after_clipping_to_0_1",
        "network_resize": "trilinear_align_corners_false",
    }
    return normalized, tensor, metadata
