from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cap_cop.imaging.segmentation.inference import (  # noqa: E402
    THRESHOLD,
    all_checkpoint_paths,
    checkpoint_paths_for_outer,
    ensemble_probability_from_models,
)
from cap_cop.imaging.segmentation.model import load_locked_checkpoint  # noqa: E402
from cap_cop.imaging.segmentation.preprocess import preprocess_dicom_for_segmentation  # noqa: E402
from cap_cop.imaging.segmentation.splits import (  # noqa: E402
    SPLIT_SEED,
    assert_strict_oof_patient_not_in_training,
    load_locked_split_manifest,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def locked_checkpoint_hashes() -> dict[str, str]:
    manifest = pd.read_csv(
        PROJECT_ROOT / "models" / "segmentation" / "manifests" / "locked_checkpoint_manifest.csv",
        dtype=str,
    )
    return {
        f"{row.run_id}_best_checkpoint.pt": str(row.checkpoint_sha256).upper()
        for row in manifest.itertuples(index=False)
    }


def verify_checkpoint_paths(paths: list[Path]) -> list[str]:
    expected = locked_checkpoint_hashes()
    hashes: list[str] = []
    for path in paths:
        actual = sha256_file(path)
        wanted = expected.get(path.name)
        if wanted is None or actual != wanted:
            raise RuntimeError(f"Locked checkpoint hash mismatch for {path}: {actual} != {wanted}")
        hashes.append(actual)
    return hashes


def load_models(paths: list[Path], device: torch.device) -> list[torch.nn.Module]:
    verify_checkpoint_paths(paths)
    return [load_locked_checkpoint(path, device=device) for path in paths]


def release_dicom_dir(data_root: Path, row: pd.Series) -> Path:
    return (
        data_root
        / str(row["cohort"])
        / str(row["diagnosis_label"])
        / "imaging"
        / "ct"
        / str(row["public_subject_id"])
        / "dicom"
    )


def internal_outer_folds(crosswalk: pd.DataFrame) -> dict[str, int]:
    splits = load_locked_split_manifest()
    test_rows = splits[
        splits["training_seed"].eq(SPLIT_SEED)
        & splits["outer_role"].eq("outer_test")
    ][["patient_group_id_public", "outer_fold_id"]].drop_duplicates()
    counts = test_rows.groupby("patient_group_id_public")["outer_fold_id"].nunique()
    if len(counts) != 375 or not counts.eq(1).all():
        raise RuntimeError("Locked segmentation split does not assign each Internal375 patient to one outer test fold.")
    mapping = dict(
        zip(test_rows["patient_group_id_public"].astype(str), test_rows["outer_fold_id"].astype(int))
    )
    internal = crosswalk[crosswalk["cohort"].eq("internal")]
    if set(internal["patient_group_id_public"].astype(str)) != set(mapping):
        raise RuntimeError("Internal public crosswalk and locked segmentation split patient sets differ.")
    return mapping


def locked_mask_references() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    internal = pd.read_csv(
        PROJECT_ROOT / "models" / "segmentation" / "manifests" / "internal_strict_oof_mask_manifest.csv",
        dtype=str,
    ).set_index("neutral_patient_id")
    external = pd.read_csv(
        PROJECT_ROOT / "models" / "segmentation" / "manifests" / "external_segmentation_mask_manifest_reference.csv",
        dtype=str,
    ).set_index("neutral_external_id")
    return internal.to_dict("index"), external.to_dict("index")


def write_audit(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def existing_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def run_subject(
    row: pd.Series,
    dicom_dir: Path,
    output_root: Path,
    models: list[torch.nn.Module],
    checkpoint_hashes: list[str],
    device: torch.device,
    locked_reference: dict[str, str],
    outer_fold: int | None,
    save_probability: bool,
) -> dict[str, object]:
    public_id = str(row["public_subject_id"])
    cohort = str(row["cohort"])
    if not dicom_dir.exists():
        raise FileNotFoundError(dicom_dir)

    preprocessed = preprocess_dicom_for_segmentation(dicom_dir)
    probability = ensemble_probability_from_models(preprocessed.array_zyx, models, device)
    mask = (probability >= THRESHOLD).astype(np.uint8)

    subject_dir = output_root / cohort / public_id
    subject_dir.mkdir(parents=True, exist_ok=True)
    mask_path = subject_dir / f"{public_id}_predicted_mask.npy"
    np.save(mask_path, mask, allow_pickle=False)
    probability_path = subject_dir / f"{public_id}_probability.npy"
    if save_probability:
        np.save(probability_path, probability.astype(np.float32, copy=False), allow_pickle=False)

    component_count = int(ndimage.label(mask)[1])
    voxel_ml = float(np.prod(preprocessed.spacing_xyz) / 1000.0)
    metadata = {
        "public_subject_id": public_id,
        "cohort": cohort,
        "outer_fold": outer_fold,
        "working_spacing_xyz": list(preprocessed.spacing_xyz),
        "working_origin_xyz": list(preprocessed.origin_xyz),
        "working_direction": list(preprocessed.direction),
        "shape_zyx": list(mask.shape),
        "threshold": THRESHOLD,
        "models_used": len(models),
        "checkpoint_sha256": checkpoint_hashes,
        "mask_voxels": int(mask.sum()),
        "predicted_volume_ml": float(mask.sum() * voxel_ml),
        "predicted_component_count": component_count,
        "global_max_probability": float(probability.max()),
    }
    metadata_path = subject_dir / f"{public_id}_segmentation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    mask_sha = sha256_file(mask_path)
    expected_mask_sha = str(locked_reference["prediction_mask_sha256"]).upper()
    shape_text = "x".join(str(value) for value in mask.shape)
    expected_shape = str(locked_reference["shape_zyx"])
    probability_sha = sha256_file(probability_path) if save_probability else ""
    expected_probability_sha = str(locked_reference.get("probability_map_sha256", "")).upper()
    return {
        "public_subject_id": public_id,
        "cohort": cohort,
        "diagnosis_label": str(row["diagnosis_label"]),
        "label_binary": int(row["label_binary"]),
        "outer_fold": outer_fold,
        "dicom_dir": str(dicom_dir),
        "models_used": len(models),
        "threshold": THRESHOLD,
        "shape_zyx": shape_text,
        "expected_shape_zyx": expected_shape,
        "shape_match": shape_text == expected_shape,
        "mask_sha256": mask_sha,
        "expected_mask_sha256": expected_mask_sha,
        "mask_sha256_match": mask_sha == expected_mask_sha,
        "probability_sha256": probability_sha,
        "expected_probability_sha256": expected_probability_sha,
        "probability_sha256_match": (
            probability_sha == expected_probability_sha if expected_probability_sha else None
        ),
        "mask_nonempty": bool(mask.sum() > 0),
        "mask_voxels": int(mask.sum()),
        "predicted_volume_ml": float(mask.sum() * voxel_ml),
        "predicted_component_count": component_count,
        "locked_predicted_volume_ml": float(locked_reference["predicted_volume_ml"]),
        "locked_predicted_component_count": int(locked_reference["predicted_component_count"]),
        "global_max_probability": float(probability.max()),
        "locked_global_max_probability": float(locked_reference["global_max_probability"]),
        "mask_path": str(mask_path),
        "probability_path": str(probability_path) if save_probability else "",
        "metadata_path": str(metadata_path),
        "status": "PASS" if mask_sha == expected_mask_sha and shape_text == expected_shape else "MISMATCH",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay locked Internal375 strict-OOF and External33 lesion segmentation from DICOM."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "deidentified_release_candidate",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "full_reproduction_20261010" / "segmentation",
    )
    parser.add_argument("--cohort", choices=("all", "internal", "external"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-internal-probability", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    if project_root != PROJECT_ROOT:
        raise RuntimeError(f"This installed replay is rooted at {PROJECT_ROOT}, not {project_root}")
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    requested = torch.device(args.device)
    device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False

    crosswalk = pd.read_csv(project_root / "data" / "public_subject_model_crosswalk.csv", dtype=str)
    crosswalk["label_binary"] = crosswalk["label_binary"].astype(int)
    if len(crosswalk) != 408 or crosswalk["public_subject_id"].nunique() != 408:
        raise RuntimeError("Expected a unique 408-subject public crosswalk.")
    fold_map = internal_outer_folds(crosswalk)
    internal_refs, external_refs = locked_mask_references()

    audit_path = output_root / "full_cohort_segmentation_replay_audit.csv"
    audit_rows = existing_rows(audit_path)
    completed = {
        str(row["public_subject_id"])
        for row in audit_rows
        if str(row.get("status", "")) == "PASS"
    }

    groups: list[tuple[str, int | None, pd.DataFrame, list[Path], bool]] = []
    if args.cohort in {"all", "internal"}:
        internal = crosswalk[crosswalk["cohort"].eq("internal")].copy()
        internal["outer_fold"] = internal["patient_group_id_public"].map(fold_map).astype(int)
        for outer_fold in range(5):
            members = internal[internal["outer_fold"].eq(outer_fold)].sort_values("public_subject_id")
            groups.append(("internal", outer_fold, members, checkpoint_paths_for_outer(outer_fold), args.save_internal_probability))
    if args.cohort in {"all", "external"}:
        external = crosswalk[crosswalk["cohort"].eq("external")].sort_values("public_subject_id")
        groups.append(("external", None, external, all_checkpoint_paths(), True))

    processed_this_run = 0
    for cohort, outer_fold, members, checkpoint_paths, save_probability in groups:
        pending = members[~members["public_subject_id"].isin(completed)]
        if args.limit is not None:
            remaining = max(args.limit - processed_this_run, 0)
            pending = pending.head(remaining)
        if pending.empty:
            continue
        if args.limit is not None and processed_this_run >= args.limit:
            break

        print(
            f"Loading {len(checkpoint_paths)} locked models for cohort={cohort} outer_fold={outer_fold}",
            flush=True,
        )
        checkpoint_hashes = verify_checkpoint_paths(checkpoint_paths)
        models = load_models(checkpoint_paths, device)
        for row in pending.itertuples(index=False):
            series = pd.Series(row._asdict())
            public_id = str(series["public_subject_id"])
            if cohort == "internal":
                assert_strict_oof_patient_not_in_training(
                    str(series["patient_group_id_public"]), int(outer_fold)
                )
                reference = internal_refs[str(series["segmentation_model_id"])]
            else:
                reference = external_refs[str(series["segmentation_model_id"])]
            dicom_dir = release_dicom_dir(data_root, series)
            result = run_subject(
                series,
                dicom_dir,
                output_root,
                models,
                checkpoint_hashes,
                device,
                reference,
                outer_fold,
                save_probability,
            )
            audit_rows = [
                existing for existing in audit_rows if str(existing["public_subject_id"]) != public_id
            ]
            audit_rows.append(result)
            write_audit(audit_path, audit_rows)
            processed_this_run += 1
            print(
                f"[{len(audit_rows)}/408] {public_id}: {result['status']} "
                f"mask_sha_match={result['mask_sha256_match']} shape={result['shape_zyx']}",
                flush=True,
            )
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    final = pd.DataFrame(audit_rows)
    summary = {
        "status": "PASS" if len(final) == 408 and final["status"].eq("PASS").all() else "INCOMPLETE_OR_MISMATCH",
        "rows": int(len(final)),
        "unique_public_subject_ids": int(final["public_subject_id"].nunique()) if len(final) else 0,
        "mask_sha256_match_count": int(final["mask_sha256_match"].fillna(False).sum()) if len(final) else 0,
        "shape_match_count": int(final["shape_match"].fillna(False).sum()) if len(final) else 0,
        "internal_rows": int(final["cohort"].eq("internal").sum()) if len(final) else 0,
        "external_rows": int(final["cohort"].eq("external").sum()) if len(final) else 0,
        "device": str(device),
        "input_data_root": str(data_root),
        "audit_csv": str(audit_path),
    }
    summary_path = output_root / "full_cohort_segmentation_replay_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
