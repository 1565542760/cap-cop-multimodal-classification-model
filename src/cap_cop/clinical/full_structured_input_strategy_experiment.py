from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from cap_cop.clinical import clinical_missingness_simulation as v30
from cap_cop.clinical import clinical_transportability_protocol as v3


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "transportability_v34_train_fold_augmentation_20260909"
INTERNAL_PATH = v3.INTERNAL_PATH
EXTERNAL_PATH = v3.EXTERNAL_PATH

SEEDS = [20260908, 20260909, 20260910]
N_SPLITS = 5
SCENARIOS = ["normal", "external_mask_1", "external_mask_2", "external_mask_3"]

FEATURE_SETS = {
    "full16": v3.ALL_FEATURES,
    "transportable15_no_CRP": [
        feature for feature in v3.ALL_FEATURES if feature != "c_reactive_protein"
    ],
    "transportable11": v3.TRANSPORTABLE11,
    "core7_hscrp_pct": v3.CORE7_HSCRP_PCT,
    "core7": v3.CORE7,
}

MODEL_SPECS = ["logistic", "svm", "random_forest"]
AUGMENTATION_MODES = ["none", "balanced_bootstrap", "balanced_jitter"]
JITTER_SCALE = 0.05


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scenario_frame(
    frame: pd.DataFrame,
    external: pd.DataFrame,
    features: list[str],
    scenario: str,
    seed: int,
) -> pd.DataFrame:
    if scenario == "normal":
        return frame.copy()
    return v30.apply_external_mask(frame, external, features, scenario, seed)


def augment_training_design(
    design: pd.DataFrame,
    y: np.ndarray,
    features: list[str],
    mode: str,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Augment only an already-imputed training fold."""
    if mode == "none":
        return design.copy(), np.asarray(y, dtype=int).copy()

    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    class_indices = {
        cls: np.flatnonzero(y == cls)
        for cls in np.unique(y)
    }
    target_n = max(len(indices) for indices in class_indices.values())

    categorical = [
        feature for feature in features if feature in v30.CATEGORICAL
    ]
    continuous = [
        column for column in design.columns
        if column not in categorical
        and not column.startswith("__missing_")
    ]

    rows = [design.copy()]
    labels = [y.copy()]
    for cls, indices in sorted(class_indices.items()):
        if len(indices) >= target_n:
            continue
        sampled = rng.choice(indices, size=target_n - len(indices), replace=True)
        synthetic = design.iloc[sampled].copy().reset_index(drop=True)

        if mode == "balanced_jitter" and continuous:
            for column in continuous:
                values = pd.to_numeric(design.iloc[indices][column], errors="coerce")
                scale = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                if not np.isfinite(scale) or scale <= 0:
                    continue
                noise = rng.normal(
                    loc=0.0,
                    scale=JITTER_SCALE * scale,
                    size=len(synthetic),
                )
                lower = float(values.min())
                upper = float(values.max())
                synthetic[column] = (
                    pd.to_numeric(synthetic[column], errors="coerce")
                    + noise
                ).clip(lower=lower, upper=upper)

        rows.append(synthetic)
        labels.append(np.full(len(synthetic), cls, dtype=int))

    augmented = pd.concat(rows, ignore_index=True)
    augmented_y = np.concatenate(labels)
    order = rng.permutation(len(augmented))
    return augmented.iloc[order].reset_index(drop=True), augmented_y[order]


def fit_predict_one(
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    y_train: np.ndarray,
    features: list[str],
    model_spec: str,
    augmentation: str,
    seed: int,
) -> np.ndarray:
    train_clean = v30.clean(train_frame, features)
    eval_clean = v30.clean(eval_frame, features)
    imputer = v30.FoldImputer(features, "median", seed)
    imputer.fit(train_clean)
    train_imputed = imputer.transform(train_clean)
    eval_imputed = imputer.transform(eval_clean)
    train_design = v30.make_design(train_imputed, train_clean, features)
    eval_design = v30.make_design(eval_imputed, eval_clean, features)
    augmented_design, augmented_y = augment_training_design(
        train_design,
        y_train,
        features,
        augmentation,
        seed + 7919,
    )
    if model_spec == "svm":
        categorical = [
            feature for feature in features if feature in v30.CATEGORICAL
        ]
        numeric = [
            column
            for column in augmented_design.columns
            if column not in categorical
        ]
        encoded_train = pd.get_dummies(
            augmented_design,
            columns=categorical,
            drop_first=False,
            dtype=float,
        )
        encoded_eval = pd.get_dummies(
            eval_design,
            columns=categorical,
            drop_first=False,
            dtype=float,
        ).reindex(columns=encoded_train.columns, fill_value=0.0)
        scaler = RobustScaler()
        matrix_train = scaler.fit_transform(encoded_train)
        matrix_eval = scaler.transform(encoded_eval)
        model = SVC(
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=seed,
        )
        model.fit(matrix_train, augmented_y)
        return model.predict_proba(matrix_eval)[:, 1]

    fitted = v30.fit_one_model(
        augmented_design,
        augmented_y,
        features,
        model_spec,
        seed,
    )
    return v30.predict_one_model(fitted, eval_design, features)


def cross_fitted(
    X: pd.DataFrame,
    external: pd.DataFrame,
    y: np.ndarray,
    features: list[str],
    model_spec: str,
    augmentation: str,
    scenario: str,
    seed: int,
) -> np.ndarray:
    from sklearn.model_selection import StratifiedKFold

    masked = scenario_frame(X, external, features, scenario, seed)
    predictions = np.full(len(masked), np.nan, dtype=float)
    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed,
    )
    for fold, (train_idx, val_idx) in enumerate(splitter.split(masked, y)):
        predictions[val_idx] = fit_predict_one(
            masked.iloc[train_idx],
            masked.iloc[val_idx],
            y[train_idx],
            features,
            model_spec,
            augmentation,
            seed * 100 + fold,
        )
    return predictions


def external_predictions(
    X: pd.DataFrame,
    external: pd.DataFrame,
    y: np.ndarray,
    features: list[str],
    model_spec: str,
    augmentation: str,
) -> np.ndarray:
    predictions = []
    for index, seed in enumerate(SEEDS):
        train_frame = scenario_frame(
            X,
            external,
            features,
            f"external_mask_{index + 1}",
            seed,
        )
        predictions.append(
            fit_predict_one(
                train_frame,
                external,
                y,
                features,
                model_spec,
                augmentation,
                seed,
            )
        )
    return np.mean(np.vstack(predictions), axis=0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    internal = pd.read_csv(INTERNAL_PATH)
    external = pd.read_csv(EXTERNAL_PATH)
    X = internal[v3.ALL_FEATURES].copy()
    y = internal["label_binary"].astype(int).to_numpy()
    y_external = external["label"].astype(int).to_numpy()

    if len(internal) != 375 or int(y.sum()) != 125:
        raise ValueError("Unexpected internal cohort.")
    if len(external) != 33 or int(y_external.sum()) != 14:
        raise ValueError("Unexpected external cohort.")

    rows = []
    oof_store: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    for feature_name, features in FEATURE_SETS.items():
        for augmentation in AUGMENTATION_MODES:
            for model_spec in MODEL_SPECS:
                scenario_predictions = {}
                scenario_auroc = {}
                scenario_brier = {}
                for scenario in SCENARIOS:
                    seed_predictions = [
                        cross_fitted(
                            X,
                            external,
                            y,
                            features,
                            model_spec,
                            augmentation,
                            scenario,
                            seed,
                        )
                        for seed in SEEDS
                    ]
                    pooled = np.mean(np.vstack(seed_predictions), axis=0)
                    scenario_predictions[scenario] = pooled
                    scenario_auroc[scenario] = v3.roc_auc_score(y, pooled)
                    scenario_brier[scenario] = v3.brier_score_loss(y, pooled)

                key = (feature_name, augmentation, model_spec)
                oof_store[key] = scenario_predictions
                stress_values = [
                    scenario_auroc[scenario]
                    for scenario in SCENARIOS
                    if scenario != "normal"
                ]
                rows.append(
                    {
                        "feature_set": feature_name,
                        "augmentation": augmentation,
                        "model_spec": model_spec,
                        "normal_AUROC": scenario_auroc["normal"],
                        "mean_stress_AUROC": float(np.mean(stress_values)),
                        "worst_stress_AUROC": float(np.min(stress_values)),
                        "composite_AUROC": float(
                            0.5 * scenario_auroc["normal"]
                            + 0.5 * np.mean(stress_values)
                        ),
                        "mean_scenario_Brier": float(
                            np.mean(list(scenario_brier.values()))
                        ),
                    }
                )

    comparison = pd.DataFrame(rows).sort_values(
        [
            "composite_AUROC",
            "worst_stress_AUROC",
            "normal_AUROC",
            "mean_scenario_Brier",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    selected = comparison.iloc[0]
    selected_key = (
        str(selected["feature_set"]),
        str(selected["augmentation"]),
        str(selected["model_spec"]),
    )
    selected_oof = oof_store[selected_key]["normal"]
    locked_threshold = round(v3.youden_threshold(y, selected_oof), 6)
    selected_external = external_predictions(
        X,
        external,
        y,
        FEATURE_SETS[selected_key[0]],
        selected_key[2],
        selected_key[1],
    )
    performance = pd.DataFrame(
        [
            {
                "cohort": "internal_OOF_normal",
                **v3.metric_point(y, selected_oof, locked_threshold),
                **v3.bootstrap_ci(y, selected_oof, locked_threshold, 20260911),
            },
            {
                "cohort": "external_frozen_all",
                **v3.metric_point(
                    y_external,
                    selected_external,
                    locked_threshold,
                ),
                **v3.bootstrap_ci(
                    y_external,
                    selected_external,
                    locked_threshold,
                    20260912,
                ),
            },
        ]
    )

    comparison.to_csv(
        OUT / "candidate_comparison_internal_only.csv",
        index=False,
        encoding="utf-8-sig",
    )
    performance.to_csv(
        OUT / "selected_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        {
            "patient_id": external["patient_id"],
            "label": y_external,
            "prob_COP": selected_external,
            "locked_threshold": locked_threshold,
            "prediction": (
                selected_external >= locked_threshold
            ).astype(int),
        }
    ).to_csv(
        OUT / "selected_external_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lock = {
        "iteration": "transportability_v34_train_fold_augmentation",
        "selected_key": list(selected_key),
        "selected_features": FEATURE_SETS[selected_key[0]],
        "augmentation_modes": AUGMENTATION_MODES,
        "jitter_scale": JITTER_SCALE,
        "selection_rule": (
            "internal repeated patient-level 5-fold OOF only; composite 0.5 "
            "normal AUROC + 0.5 mean external-missingness stress AUROC, then "
            "worst stress AUROC, normal AUROC, mean scenario Brier"
        ),
        "augmentation_scope": (
            "median imputation and augmentation fitted only within each "
            "training fold; validation folds and external cohort were never "
            "synthetically augmented"
        ),
        "external_covariates_used": (
            "unlabeled missingness pattern only for predefined training masks"
        ),
        "external_labels_used_for_selection": False,
        "external_labels_used_for_threshold": False,
        "external_labels_used_for_training": False,
        "locked_threshold": locked_threshold,
        "internal_sha256": sha256_file(INTERNAL_PATH),
        "external_sha256": sha256_file(EXTERNAL_PATH),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (OUT / "prediction_lock.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT / "metadata.json").write_text(
        json.dumps(
            {
                **lock,
                "feature_sets": FEATURE_SETS,
                "model_specs": MODEL_SPECS,
                "scenarios": SCENARIOS,
                "seeds": SEEDS,
                "n_splits": N_SPLITS,
                "selected_internal_metrics": performance.iloc[0].to_dict(),
                "selected_external_metrics": performance.iloc[1].to_dict(),
                "scientific_status": (
                    "Exploratory label-blind training-fold augmentation analysis."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Top candidates:")
    print(comparison.head(20).to_string(index=False))
    print("\nSelected performance:")
    print(
        performance[
            ["cohort", "AUROC", "AUPRC", "Brier", "ACC", "SE", "SP", "F1"]
        ].to_string(index=False)
    )
    print(f"\nOUTPUT_DIR={OUT}")


if __name__ == "__main__":
    main()
