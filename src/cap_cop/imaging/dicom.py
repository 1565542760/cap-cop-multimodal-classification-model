from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import SimpleITK as sitk
except Exception:  # pragma: no cover
    sitk = None


@dataclass(frozen=True)
class DicomSeriesInfo:
    path: Path
    series_id: str
    file_count: int
    size_xyz: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]
    direction: tuple[float, ...]


def discover_series(dicom_dir: str | Path) -> list[tuple[str, list[str]]]:
    if sitk is None:
        raise RuntimeError("SimpleITK is required for DICOM series reading.")
    path = Path(dicom_dir)
    if not path.exists():
        raise FileNotFoundError(f"DICOM directory not found: {path}")
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(path)) or []
    series = []
    for series_id in series_ids:
        files = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(path), series_id))
        if files:
            series.append((series_id, files))
    if not series:
        raise ValueError(f"No readable DICOM series found in {path}")
    return series


def read_primary_series(dicom_dir: str | Path) -> tuple[object, DicomSeriesInfo]:
    series_id, files = max(discover_series(dicom_dir), key=lambda item: len(item[1]))
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    image = reader.Execute()
    return image, DicomSeriesInfo(
        path=Path(dicom_dir),
        series_id=series_id,
        file_count=len(files),
        size_xyz=tuple(int(v) for v in image.GetSize()),
        spacing_xyz=tuple(float(v) for v in image.GetSpacing()),
        origin_xyz=tuple(float(v) for v in image.GetOrigin()),
        direction=tuple(float(v) for v in image.GetDirection()),
    )


def dicom_smoke_summary(dicom_dir: str | Path) -> dict[str, object]:
    _image, info = read_primary_series(dicom_dir)
    return {
        "file_count": info.file_count,
        "size_xyz": info.size_xyz,
        "spacing_xyz": info.spacing_xyz,
        "origin_xyz": info.origin_xyz,
        "direction_len": len(info.direction),
    }
