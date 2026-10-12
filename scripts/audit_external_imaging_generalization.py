from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit
from scipy.stats import ks_2samp
from sklearn.metrics import brier_score_loss


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_existing(columns: list[str], candidates: list[str]) -> str | None:
    return next((name for name in candidates if name in columns), None)


def load_external_labels(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cohort, label in (("CAP", 0), ("COP", 1)):
        manifest = pd.read_csv(root / "data" / "external" / cohort / "manifest.csv")
        for public_subject_id in manifest["public_subject_id"].astype(str):
            rows.append(
                {
                    "public_subject_id": public_subject_id,
                    "cohort": cohort,
                    "label": label,
                }
            )
    labels = pd.DataFrame(rows)
    labels["prediction_id"] = [f"external_patient_{i:04d}" for i in range(1, len(labels) + 1)]
    return labels


def load_predictions(path: Path, labels: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path)
    id_column = first_existing(list(frame.columns), ["neutral_patient_id", "patient_id", "public_subject_id"])
    probability_column = first_existing(
        list(frame.columns), ["probability_COP", "p_image", "probability", "prediction_probability"]
    )
    if id_column is None or probability_column is None:
        raise ValueError(f"Could not identify ID/probability columns in {path}")
    if len(frame) != len(labels):
        raise ValueError(f"Unexpected row count in {path}: {len(frame)}")
    frame = frame.copy()
    frame["prediction_id"] = frame[id_column].astype(str)
    frame["probability"] = frame[probability_column].astype(float)
    frame = frame.merge(labels, on="prediction_id", how="left", validate="one_to_one")
    if frame["label"].isna().any():
        raise ValueError(f"Unable to map labels for {path}")
    frame["label"] = frame["label"].astype(int)
    frame["source_file"] = str(path)
    return frame


def calibration_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    p = np.clip(probability.astype(float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    y_mean = float(np.mean(y))
    if y_mean <= 0 or y_mean >= 1:
        intercept = float("nan")
    else:
        intercept = float(brentq(lambda value: np.mean(expit(logit + value)) - y_mean, -50, 50))

    def objective(params: np.ndarray) -> float:
        linear = params[0] + params[1] * logit
        return float(np.sum(np.logaddexp(0, linear) - y * linear))

    fit = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    return {
        "n": int(len(y)),
        "prevalence": y_mean,
        "brier_score": float(brier_score_loss(y, p)),
        "calibration_intercept": intercept,
        "calibration_slope": float(fit.x[1]) if fit.success else float("nan"),
        "calibration_fit_success": bool(fit.success),
    }


def calibration_bins(cohort: str, frame: pd.DataFrame) -> pd.DataFrame:
    bins = pd.cut(
        frame["probability"],
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        include_lowest=True,
        right=True,
    )
    rows = []
    for interval, group in frame.groupby(bins, observed=False):
        if len(group) == 0:
            continue
        rows.append(
            {
                "cohort": cohort,
                "probability_bin": str(interval),
                "n": int(len(group)),
                "mean_predicted_probability": float(group["probability"].mean()),
                "observed_COP_fraction": float(group["label"].mean()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_calibration_ci(
    y: np.ndarray,
    probability: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 20261004,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = {"brier_score": [], "calibration_intercept": [], "calibration_slope": []}
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y), size=len(y))
        sample_y = y[indices]
        if np.unique(sample_y).size < 2:
            continue
        sample = calibration_metrics(sample_y, probability[indices])
        for name in values:
            value = sample[name]
            if np.isfinite(value):
                values[name].append(float(value))
    result: dict[str, float] = {}
    for name, samples in values.items():
        result[f"{name}_ci_low"] = float(np.percentile(samples, 2.5)) if samples else float("nan")
        result[f"{name}_ci_high"] = float(np.percentile(samples, 97.5)) if samples else float("nan")
    return result


def probability_summary(cohort: str, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in frame.groupby("label"):
        values = group["probability"]
        rows.append(
            {
                "cohort": cohort,
                "label": int(label),
                "label_name": "COP" if label else "CAP",
                "n": int(len(group)),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(group) > 1 else float("nan"),
                "q25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "q75": float(values.quantile(0.75)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def read_dicom_metadata(case_dir: Path) -> dict[str, object]:
    try:
        import SimpleITK as sitk
    except Exception as exc:
        return {"status": "SimpleITK_unavailable", "error": str(exc)}

    files = [path for path in case_dir.rglob("*") if path.is_file() and path.name != "VERSION"]
    if not files:
        return {"status": "no_files"}
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(files[0]))
    try:
        reader.ReadImageInformation()
        tags = {
            "manufacturer": "0008|0070",
            "slice_thickness": "0018|0050",
            "spacing_between_slices": "0018|0088",
            "pixel_spacing": "0028|0030",
            "kvp": "0018|0060",
            "tube_current": "0018|1151",
            "exposure_time": "0018|1150",
            "reconstruction_kernel": "0018|1210",
            "series_description": "0008|103e",
        }
        result = {"status": "ok", "file_count": len(files)}
        for name, tag in tags.items():
            result[name] = reader.GetMetaData(tag) if reader.HasMetaDataKey(tag) else None
        return result
    except Exception as exc:
        return {"status": "metadata_read_error", "file_count": len(files), "error": str(exc)}


def dicom_summary(root: Path, labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, item in labels.iterrows():
        cohort = str(item["cohort"])
        subject = str(item["public_subject_id"])
        path = root / "data" / "external" / cohort / "imaging" / "ct" / subject / "dicom"
        row = {"cohort": cohort, "public_subject_id": subject, "ct_path": str(path)}
        row.update(read_dicom_metadata(path))
        rows.append(row)
    return pd.DataFrame(rows)


def segmentation_qc(root: Path, labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, item in labels.iterrows():
        cohort = str(item["cohort"])
        subject = str(item["public_subject_id"])
        manifest = pd.read_csv(root / "data" / "external" / cohort / "manifest.csv")
        record = manifest.loc[manifest["public_subject_id"].astype(str) == subject].iloc[0]
        mask_path = record.get("mask_path", "")
        mask_present = pd.notna(mask_path) and str(mask_path).strip().lower() not in {"", "nan", "none"}
        rows.append(
            {
                "cohort": cohort,
                "public_subject_id": subject,
                "mask_path_present_in_manifest": bool(mask_present),
                "dice_calculable_from_public_workspace": False,
                "reason": "external ground-truth mask not present in local manifest/input",
            }
        )
    return pd.DataFrame(rows)


def radiomics_shift(root: Path, labels: pd.DataFrame) -> pd.DataFrame:
    internal_path = root / "models" / "lesion_radiomics" / "internal_baseline_pyradiomics_features.csv"
    external_path = root / "models" / "lesion_radiomics" / "external_label_blind_pyradiomics_features.csv"
    if not internal_path.exists() or not external_path.exists():
        return pd.DataFrame([{"status": "feature_files_missing"}])
    internal = pd.read_csv(internal_path)
    external = pd.read_csv(external_path)
    internal_label = first_existing(list(internal.columns), ["label", "label_binary", "y"])
    if internal_label is None:
        oof_path = root / "models" / "lesion_radiomics" / "internal_oof_predictions.csv"
        if oof_path.exists() and "neutral_patient_id" in internal.columns:
            oof = pd.read_csv(oof_path)
            oof_id = first_existing(list(oof.columns), ["neutral_patient_id", "patient_id", "public_subject_id"])
            oof_label = first_existing(list(oof.columns), ["label", "label_binary", "y", "target"])
            if oof_id and oof_label:
                linked = oof[[oof_id, oof_label]].rename(
                    columns={oof_id: "neutral_patient_id", oof_label: "_linked_label"}
                )
                internal = internal.merge(linked, on="neutral_patient_id", how="left", validate="one_to_one")
                internal_label = "_linked_label"
    if internal_label is None:
        return pd.DataFrame([{"status": "internal_feature_labels_missing"}])
    external = external.copy()
    external["label"] = labels["label"].to_numpy()
    excluded = {"label", "label_binary", "y", "patient_id", "neutral_patient_id", "public_subject_id"}
    features = [
        name
        for name in internal.columns
        if name in external.columns
        and name not in excluded
        and pd.api.types.is_numeric_dtype(internal[name])
        and pd.api.types.is_numeric_dtype(external[name])
    ]
    rows = []
    for feature in features:
        for label, label_name in ((0, "CAP"), (1, "COP")):
            a = internal.loc[internal[internal_label].astype(int) == label, feature].dropna().to_numpy()
            b = external.loc[external["label"] == label, feature].dropna().to_numpy()
            if len(a) < 2 or len(b) < 2:
                continue
            pooled = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
            rows.append(
                {
                    "feature": feature,
                    "label": label,
                    "label_name": label_name,
                    "internal_n": len(a),
                    "external_n": len(b),
                    "internal_median": float(np.median(a)),
                    "external_median": float(np.median(b)),
                    "standardized_mean_difference": float((np.mean(b) - np.mean(a)) / pooled)
                    if pooled > 0
                    else float("nan"),
                    "ks_statistic": float(ks_2samp(a, b).statistic),
                    "ks_p_value": float(ks_2samp(a, b).pvalue),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result.insert(0, "interpretation_scope", "unadjusted descriptive comparison; not a causal analysis")
    return result


def write_plots(output: Path, cohort_frames: dict[str, pd.DataFrame]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for cohort, frame in cohort_frames.items():
        grouped = calibration_bins(cohort, frame)
        axes[0].plot(
            grouped["mean_predicted_probability"],
            grouped["observed_COP_fraction"],
            marker="o",
            label=cohort,
        )
        for label, group in frame.groupby("label"):
            axes[1].hist(
                group["probability"],
                bins=np.linspace(0, 1, 11),
                alpha=0.35,
                label=f"{cohort} {'COP' if label else 'CAP'}",
            )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    axes[0].set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Observed COP fraction")
    axes[0].legend(frameon=False)
    axes[0].set_title("Calibration summary")
    axes[1].set_xlabel("Predicted COP probability")
    axes[1].set_ylabel("Count")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("Probability distributions")
    fig.savefig(output / "external_imaging_probability_calibration.png", dpi=200)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "docs" / "ci_audit" / "external_imaging_generalization_audit"
    output.mkdir(parents=True, exist_ok=True)
    labels = load_external_labels(root)
    external = load_predictions(root / "models" / "lesion_radiomics" / "external_label_blind_predictions.csv", labels)
    internal_path = root / "models" / "lesion_radiomics" / "internal_oof_predictions.csv"

    cohort_frames = {"external_frozen": external}
    if internal_path.exists():
        internal = pd.read_csv(internal_path)
        probability_column = first_existing(list(internal.columns), ["probability_COP", "p_image", "probability", "prediction_probability"])
        label_column = first_existing(list(internal.columns), ["label", "label_binary", "y"])
        if probability_column and label_column:
            internal = internal[[probability_column, label_column]].rename(
                columns={probability_column: "probability", label_column: "label"}
            )
            internal["label"] = internal["label"].astype(int)
            cohort_frames["internal_oof"] = internal

    metric_rows = []
    bin_frames = []
    distribution_frames = []
    for cohort, frame in cohort_frames.items():
        y = frame["label"].to_numpy()
        probability = frame["probability"].to_numpy()
        metric_rows.append(
            {
                "cohort": cohort,
                "bootstrap_n": 2000,
                **calibration_metrics(y, probability),
                **bootstrap_calibration_ci(y, probability),
            }
        )
        bin_frames.append(calibration_bins(cohort, frame))
        distribution_frames.append(probability_summary(cohort, frame))
    pd.DataFrame(metric_rows).to_csv(output / "calibration_metrics.csv", index=False)
    pd.concat(bin_frames, ignore_index=True).to_csv(output / "calibration_bins.csv", index=False)
    pd.concat(distribution_frames, ignore_index=True).to_csv(output / "probability_distribution_summary.csv", index=False)
    dicom_frame = dicom_summary(root, labels)
    dicom_frame.to_csv(output / "external_dicom_acquisition_summary.csv", index=False)
    group_columns = ["cohort", "manufacturer", "slice_thickness", "reconstruction_kernel", "kvp"]
    available_groups = [name for name in group_columns if name in dicom_frame.columns]
    (
        dicom_frame.groupby(available_groups, dropna=False)
        .size()
        .reset_index(name="n_cases")
        .to_csv(output / "external_dicom_acquisition_distribution.csv", index=False)
    )
    segmentation_qc(root, labels).to_csv(output / "external_segmentation_qc_scope.csv", index=False)
    radiomics_frame = radiomics_shift(root, labels)
    radiomics_frame.to_csv(output / "lesion_radiomics_distribution_shift.csv", index=False)
    write_plots(output, cohort_frames)

    manifest = {
        "analysis": "locked external imaging generalization audit",
        "model": "lesion-ROI standardised radiomics unimodal model",
        "no_training_or_refitting": True,
        "external_n": int(len(external)),
        "external_label_counts": external["label"].value_counts().sort_index().to_dict(),
        "inputs": {
            "external_predictions": str(root / "models" / "lesion_radiomics" / "external_label_blind_predictions.csv"),
            "internal_oof_predictions": str(internal_path),
        },
        "input_sha256": {
            "external_predictions": sha256_file(root / "models" / "lesion_radiomics" / "external_label_blind_predictions.csv"),
            "internal_oof_predictions": sha256_file(internal_path) if internal_path.exists() else None,
        },
        "segmentation_dice": "not calculable because external ground-truth masks are absent from the local manifest/input",
        "radiomics_distribution_scope": "descriptive only; internal/external feature scale and preprocessing equivalence must be confirmed before interpretation",
        "interpretation": "descriptive audit; does not change locked predictions, threshold, or primary results",
    }
    (output / "audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report_lines = [
        "# External Imaging Generalization Audit",
        "",
        "This audit reads locked patient-level predictions and existing DICOM/feature artifacts. It does not retrain, refit, recalibrate, change the threshold, or overwrite primary results.",
        "",
        "## Calibration",
        "",
        pd.DataFrame(metric_rows).to_string(index=False),
        "",
        "## Scope limitations",
        "",
        "- External segmentation Dice is not calculable because external ground-truth masks are absent from the local manifest/input.",
        "- DICOM acquisition summaries are descriptive and limited to tags present in the files.",
        "- Radiomics distribution comparisons are unadjusted descriptive audits; feature-scale and preprocessing equivalence must be confirmed before attributing any shift to lesion phenotype.",
        "- The audit does not establish a causal explanation for external performance differences.",
    ]
    (output / "AUDIT_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(pd.DataFrame(metric_rows).to_string(index=False))


if __name__ == "__main__":
    main()
