from __future__ import annotations

__all__ = ["LesionSegmentationUNet3D"]


def __getattr__(name: str):
    if name == "LesionSegmentationUNet3D":
        from cap_cop.imaging.segmentation.model import LesionSegmentationUNet3D

        return LesionSegmentationUNet3D
    raise AttributeError(name)
