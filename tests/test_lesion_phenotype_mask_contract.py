"""A locked ROI centre is valid only for the mask that produced it."""

import hashlib
import pytest

from cap_cop.imaging.radiomics.mask_contract import validate_locked_mask_contract, validate_mask_metadata



def test_locked_geometry_requires_matching_mask_and_component_count(tmp_path):
    mask = tmp_path / "synthetic_mask.npy"
    mask.write_bytes(b"synthetic mask content")
    digest = hashlib.sha256(mask.read_bytes()).hexdigest()
    rows = [{"prediction_mask_sha256": digest, "predicted_component_count": "2"}]

    validate_locked_mask_contract("SYNTHETIC_001", mask, 2, rows)
    with pytest.raises(RuntimeError, match="Mask version differs"):
        validate_locked_mask_contract(
            "SYNTHETIC_001", mask, 2,
            [{"prediction_mask_sha256": "0" * 64, "predicted_component_count": "2"}],
        )
    with pytest.raises(RuntimeError, match="Component count differs"):
        validate_locked_mask_contract("SYNTHETIC_001", mask, 3, rows)
    with pytest.raises(RuntimeError, match="Missing locked ROI geometry"):
        validate_locked_mask_contract("SYNTHETIC_001", mask, 2, [])


def synthetic_geometry():
    metadata = {
        "public_subject_id": "SYNTHETIC_001", "shape_zyx": [4, 5, 6],
        "working_spacing_xyz": [2.0, 2.0, 2.0],
        "working_origin_xyz": [10.0, 20.0, 30.0],
        "working_direction": [1, 0, 0, 0, 1, 0, 0, 0, 1], "threshold": 0.5,
    }
    components = [{"component_rank_by_volume": 1, "bbox_center_x_mm": 13.0,
                   "bbox_center_y_mm": 24.0, "bbox_center_z_mm": 33.0}]
    return metadata, components


def test_locked_centres_and_metadata_are_checked(tmp_path):
    mask = tmp_path / "synthetic_mask.npy"
    mask.write_bytes(b"stable synthetic input")
    metadata, components = synthetic_geometry()
    rows = [{"prediction_mask_sha256": hashlib.sha256(mask.read_bytes()).hexdigest(),
             "predicted_component_count": "1", "predicted_component_rank": "1",
             "center_x_mm": "13", "center_y_mm": "24", "center_z_mm": "33"}]
    validate_locked_mask_contract("SYNTHETIC_001", mask, 1, rows,
                                  mask_shape=(4, 5, 6), metadata=metadata, components=components)
    wrong = [dict(rows[0], center_x_mm="14")]
    with pytest.raises(RuntimeError, match="centre"):
        validate_locked_mask_contract("SYNTHETIC_001", mask, 1, wrong,
                                      mask_shape=(4, 5, 6), metadata=metadata, components=components)
    for field, replacement, message in (
        ("public_subject_id", "OTHER", "patient ID"),
        ("shape_zyx", [4, 5, 7], "shape"),
        ("working_spacing_xyz", [2.1, 2.0, 2.0], None),
        ("working_direction", [0, 1, 0, 1, 0, 0, 0, 0, 1], "direction"),
    ):
        changed = dict(metadata, **{field: replacement})
        if message is None:
            validate_mask_metadata("SYNTHETIC_001", (4, 5, 6), changed)
        else:
            with pytest.raises(RuntimeError, match=message):
                validate_mask_metadata("SYNTHETIC_001", (4, 5, 6), changed)


def test_incomplete_geometry_and_wrong_threshold_fail():
    metadata, _ = synthetic_geometry()
    for change, message in (
        ({"working_origin_xyz": None}, "working_origin_xyz"),
        ({"working_spacing_xyz": [2.0, -1.0, 2.0]}, "spacing"),
        ({"threshold": 0.6}, "threshold"),
    ):
        with pytest.raises(RuntimeError, match=message):
            validate_mask_metadata("SYNTHETIC_001", (4, 5, 6), dict(metadata, **change))


def test_nearly_axis_aligned_direction_uses_physical_roi_centre(tmp_path):
    mask = tmp_path / "synthetic_mask.npy"
    mask.write_bytes(b"synthetic tilted geometry")
    metadata, components = synthetic_geometry()
    tilt = 1.6e-7
    metadata["working_direction"] = [1, 0, 0, 0, 1, tilt, 0, -tilt, 1]
    rows = [{"prediction_mask_sha256": hashlib.sha256(mask.read_bytes()).hexdigest(),
             "predicted_component_count": "1", "predicted_component_rank": "1",
             "center_x_mm": "13", "center_y_mm": str(24 + tilt * 3),
             "center_z_mm": str(33 - tilt * 4)}]
    validate_locked_mask_contract("SYNTHETIC_001", mask, 1, rows,
                                  mask_shape=(4, 5, 6), metadata=metadata, components=components)
    with pytest.raises(RuntimeError, match="centre"):
        validate_locked_mask_contract("SYNTHETIC_001", mask, 1,
                                      [dict(rows[0], center_y_mm="24.01")],
                                      mask_shape=(4, 5, 6), metadata=metadata, components=components)
