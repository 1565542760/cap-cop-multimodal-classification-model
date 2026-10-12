from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import RobustScaler

from cap_cop.clinical import clinical_transportability_protocol as v3


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "transportability_v30_multiple_imputation_20260908"
INTERNAL_PATH = v3.INTERNAL_PATH
EXTERNAL_PATH = v3.EXTERNAL_PATH
SEEDS = [20260908, 20260909, 20260910]
N_SPLITS = 5
N_IMPUTATIONS = 5
N_BOOT = 5000

FEATURE_SETS = {
    "full16": v3.ALL_FEATURES,
    "transportable15_no_CRP": [
        feature for feature in v3.ALL_FEATURES if feature != "c_reactive_protein"
    ],
    "transportable11": v3.TRANSPORTABLE11,
}
IMPUTERS = ["median", "mice_bayesian_posterior"]
MODEL_SPECS = ["logistic", "random_forest"]
SCENARIOS = ["normal", "external_mask_1", "external_mask_2", "external_mask_3"]
CATEGORICAL = {
    "fever",
    "cough",
    "sputum_category",
    "chest_tightness",
    "dyspnoea",
    "haemoptysis",
    "weight_loss",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean(frame, features):
    return v3.FrameCleaner().fit_transform(frame[features].copy())


def apply_external_mask(frame, external, features, scenario, seed):
    out = frame.copy()
    if scenario == "normal":
        return out
    pattern = external[features].isna().to_numpy()
    rng = np.random.default_rng(seed + int(scenario.rsplit("_", 1)[1]))
    sampled = pattern[rng.integers(0, len(pattern), size=len(out))]
    for index, feature in enumerate(features):
        out.loc[sampled[:, index], feature] = np.nan
    return out


class FoldImputer:
    def __init__(self, features, method, seed):
        self.features = list(features)
        self.method = method
        self.seed = seed

    def fit(self, frame):
        self.categories = {}
        clean_frame = clean(frame, self.features)
        values = clean_frame[self.features].apply(pd.to_numeric, errors="coerce")
        self.medians = {
            feature: float(values[feature].median())
            if values[feature].notna().any()
            else 0.0
            for feature in self.features
        }
        self.lower = {
            feature: float(values[feature].quantile(0.01))
            if values[feature].notna().any()
            else 0.0
            for feature in self.features
        }
        self.upper = {
            feature: float(values[feature].quantile(0.99))
            if values[feature].notna().any()
            else 0.0
            for feature in self.features
        }
        for feature in self.features:
            if feature in CATEGORICAL:
                observed = sorted(values[feature].dropna().unique().tolist())
                self.categories[feature] = observed or [0.0]
        if self.method == "mice_bayesian_posterior":
            self.imputer = IterativeImputer(
                estimator=BayesianRidge(),
                initial_strategy="median",
                max_iter=20,
                tol=1e-3,
                sample_posterior=True,
                keep_empty_features=True,
                random_state=self.seed,
            )
            self.imputer.fit(values.to_numpy(dtype=float, copy=True))
        else:
            self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            self.imputer.fit(values.to_numpy(dtype=float, copy=True))
        return self

    def transform(self, frame, imputation_index=0):
        clean_frame = clean(frame, self.features)
        values = clean_frame[self.features].apply(pd.to_numeric, errors="coerce")
        if self.method == "mice_bayesian_posterior":
            if imputation_index:
                self.imputer.random_state = self.seed + imputation_index
            imputed = self.imputer.transform(
                values.to_numpy(dtype=float, copy=True)
            )
        else:
            imputed = self.imputer.transform(
                values.to_numpy(dtype=float, copy=True)
            )
        result = pd.DataFrame(imputed, columns=self.features, index=frame.index)
        for feature in self.features:
            result[feature] = result[feature].clip(
                lower=self.lower[feature],
                upper=self.upper[feature],
            )
            if feature in self.categories:
                categories = np.asarray(self.categories[feature])
                result[feature] = result[feature].map(
                    lambda value: float(categories[np.argmin(np.abs(categories - value))])
                )
        return result


def make_design(imputed, original, features):
    result = imputed.copy()
    missing = original[features].isna()
    for feature in features:
        result[f"__missing_{feature}"] = missing[feature].astype(float)
    clinical = [
        feature
        for feature in ["fever", "cough", "sputum_category", "weight_loss"]
        if feature in features
    ]
    labs = [feature for feature in v3.LAB_FEATURES if feature in features]
    anthropometric = [
        feature
        for feature in ["height_cm", "weight_kg", "body_mass_index"]
        if feature in features
    ]
    result["__missing_count_selected"] = missing.sum(axis=1).astype(float)
    result["__missing_count_clinical"] = missing[clinical].sum(axis=1).astype(float)
    result["__missing_count_laboratory"] = missing[labs].sum(axis=1).astype(float)
    result["__missing_count_anthropometric"] = missing[anthropometric].sum(axis=1).astype(float)
    return result


def fit_one_model(train_design, y, features, model_spec, seed):
    categorical = [feature for feature in features if feature in CATEGORICAL]
    numeric = [column for column in train_design.columns if column not in categorical]
    encoded_train = pd.get_dummies(
        train_design,
        columns=categorical,
        drop_first=False,
        dtype=float,
    )
    if model_spec == "logistic":
        scaler = RobustScaler()
        matrix = scaler.fit_transform(encoded_train[numeric + [
            column for column in encoded_train.columns if column not in numeric
        ]])
        model = LogisticRegression(
            C=0.3,
            penalty="l2",
            solver="liblinear",
            max_iter=5000,
        )
        model.fit(matrix, y)
        return ("logistic", model, scaler, encoded_train.columns.tolist())
    model = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(encoded_train, y)
    return ("random_forest", model, None, encoded_train.columns.tolist())


def predict_one_model(fitted, design, features):
    kind, model, scaler, columns = fitted
    categorical = [feature for feature in features if feature in CATEGORICAL]
    encoded = pd.get_dummies(design, columns=categorical, drop_first=False, dtype=float)
    encoded = encoded.reindex(columns=columns, fill_value=0.0)
    if kind == "logistic":
        return model.predict_proba(scaler.transform(encoded))[:, 1]
    return model.predict_proba(encoded)[:, 1]


def fit_predict(
    train_frame,
    eval_frame,
    y,
    features,
    imputer_name,
    model_spec,
    seed,
):
    train_clean = clean(train_frame, features)
    eval_clean = clean(eval_frame, features)
    imputer = FoldImputer(features, imputer_name, seed)
    imputer.fit(train_clean)
    count = N_IMPUTATIONS if imputer_name == "mice_bayesian_posterior" else 1
    predictions = []
    for imputation_index in range(count):
        train_imputed = imputer.transform(train_clean, imputation_index)
        eval_imputed = imputer.transform(eval_clean, imputation_index)
        train_design = make_design(train_imputed, train_clean, features)
        eval_design = make_design(eval_imputed, eval_clean, features)
        fitted = fit_one_model(
            train_design,
            y,
            features,
            model_spec,
            seed + imputation_index,
        )
        predictions.append(predict_one_model(fitted, eval_design, features))
    return np.mean(np.vstack(predictions), axis=0)


def cross_fitted(
    X,
    external,
    y,
    features,
    imputer_name,
    model_spec,
    scenario,
    seed,
):
    from sklearn.model_selection import StratifiedKFold

    masked = apply_external_mask(X, external, features, scenario, seed)
    p = np.full(len(X), np.nan)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(masked, y)):
        p[val_idx] = fit_predict(
            masked.iloc[train_idx],
            masked.iloc[val_idx],
            y[train_idx],
            features,
            imputer_name,
            model_spec,
            seed * 100 + fold,
        )
    return p


def full_external(X, external, y, features, imputer_name, model_spec):
    predictions = []
    for seed in SEEDS:
        predictions.append(
            fit_predict(
                X,
                external,
                y,
                features,
                imputer_name,
                model_spec,
                seed,
            )
        )
    return np.mean(np.vstack(predictions), axis=0)


def metric_point(y, p, threshold):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    pred = (p >= threshold).astype(int)
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    return {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "LogLoss": float(log_loss(y, p, labels=[0, 1])),
        "ACC": float(accuracy_score(y, pred)),
        "SE": float(recall_score(y, pred, zero_division=0)),
        "SP": float(tn / (tn + fp)) if tn + fp else np.nan,
        "F1": float(f1_score(y, pred, zero_division=0)),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def threshold(y, p):
    fpr, tpr, values = roc_curve(y, p)
    j = tpr - fpr
    candidates = values[np.flatnonzero(j == np.nanmax(j))]
    finite = candidates[np.isfinite(candidates)]
    return float(finite[np.argmin(np.abs(finite - 0.5))]) if len(finite) else 0.5


def bootstrap_ci(y, p, cutoff, seed):
    pos = np.flatnonzero(np.asarray(y) == 1)
    neg = np.flatnonzero(np.asarray(y) == 0)
    rng = np.random.default_rng(seed)
    names = ["AUROC", "AUPRC", "Brier", "ACC", "SE", "SP", "F1"]
    draws = {name: [] for name in names}
    for _ in range(N_BOOT):
        idx = np.concatenate(
            [
                rng.choice(pos, len(pos), replace=True),
                rng.choice(neg, len(neg), replace=True),
            ]
        )
        values = metric_point(np.asarray(y)[idx], np.asarray(p)[idx], cutoff)
        for name in names:
            draws[name].append(values[name])
    return {
        f"{name}_CI_low": float(np.nanpercentile(values, 2.5))
        for name, values in draws.items()
    } | {
        f"{name}_CI_high": float(np.nanpercentile(values, 97.5))
        for name, values in draws.items()
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    internal = pd.read_csv(INTERNAL_PATH)
    external = pd.read_csv(EXTERNAL_PATH)
    X = internal[v3.ALL_FEATURES].copy()
    y = internal["label_binary"].astype(int).to_numpy()
    y_external = external["label"].astype(int).to_numpy()

    candidate_rows = []
    oof_store = {}
    external_store = {}
    for feature_name, features in FEATURE_SETS.items():
        for imputer_name in IMPUTERS:
            for model_spec in MODEL_SPECS:
                scenario_predictions = {}
                for scenario in SCENARIOS:
                    seed_predictions = [
                        cross_fitted(
                            X,
                            external,
                            y,
                            features,
                            imputer_name,
                            model_spec,
                            scenario,
                            seed,
                        )
                        for seed in SEEDS
                    ]
                    scenario_predictions[scenario] = np.mean(
                        np.vstack(seed_predictions), axis=0
                    )
                scenario_auroc = {
                    scenario: roc_auc_score(y, scenario_predictions[scenario])
                    for scenario in SCENARIOS
                }
                scenario_brier = {
                    scenario: brier_score_loss(y, scenario_predictions[scenario])
                    for scenario in SCENARIOS
                }
                key = (feature_name, imputer_name, model_spec)
                oof_store[key] = scenario_predictions
                external_store[key] = full_external(
                    X,
                    external,
                    y,
                    features,
                    imputer_name,
                    model_spec,
                )
                candidate_rows.append(
                    {
                        "feature_set": feature_name,
                        "imputer": imputer_name,
                        "model_spec": model_spec,
                        "normal_AUROC": scenario_auroc["normal"],
                        "mean_stress_AUROC": float(
                            np.mean(
                                [
                                    scenario_auroc[s]
                                    for s in SCENARIOS
                                    if s != "normal"
                                ]
                            )
                        ),
                        "worst_stress_AUROC": float(
                            np.min(
                                [
                                    scenario_auroc[s]
                                    for s in SCENARIOS
                                    if s != "normal"
                                ]
                            )
                        ),
                        "composite_AUROC": float(
                            0.5 * scenario_auroc["normal"]
                            + 0.5
                            * np.mean(
                                [
                                    scenario_auroc[s]
                                    for s in SCENARIOS
                                    if s != "normal"
                                ]
                            )
                        ),
                        "mean_scenario_Brier": float(np.mean(list(scenario_brier.values()))),
                    }
                )

    candidates = pd.DataFrame(candidate_rows).sort_values(
        [
            "composite_AUROC",
            "worst_stress_AUROC",
            "normal_AUROC",
            "mean_scenario_Brier",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    selected = candidates.iloc[0]
    selected_key = (
        str(selected["feature_set"]),
        str(selected["imputer"]),
        str(selected["model_spec"]),
    )
    selected_oof = oof_store[selected_key]["normal"]
    locked_threshold = round(threshold(y, selected_oof), 6)
    selected_external = external_store[selected_key]
    performance = pd.DataFrame(
        [
            {
                "cohort": "internal_OOF_normal",
                **metric_point(y, selected_oof, locked_threshold),
                **bootstrap_ci(y, selected_oof, locked_threshold, 20260909),
            },
            {
                "cohort": "external_frozen_all",
                **metric_point(y_external, selected_external, locked_threshold),
                **bootstrap_ci(
                    y_external, selected_external, locked_threshold, 20260910
                ),
            },
        ]
    )

    candidates.to_csv(
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
            "prediction": (selected_external >= locked_threshold).astype(int),
        }
    ).to_csv(
        OUT / "selected_external_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lock = {
        "iteration": "transportability_v30_multiple_imputation",
        "selected_key": list(selected_key),
        "selected_features": FEATURE_SETS[selected_key[0]],
        "selection_rule": "internal repeated patient-level 5-fold OOF only; composite 0.5 normal AUROC + 0.5 mean external-missingness stress AUROC, then worst stress AUROC, normal AUROC, mean scenario Brier",
        "external_covariates_used": "missingness matrix only for stress scenarios",
        "external_labels_used_for_selection": False,
        "external_labels_used_for_threshold": False,
        "external_labels_used_for_training": False,
        "locked_threshold": locked_threshold,
        "n_imputations": N_IMPUTATIONS,
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
                "imputers": IMPUTERS,
                "model_specs": MODEL_SPECS,
                "scenarios": SCENARIOS,
                "seeds": SEEDS,
                "n_splits": N_SPLITS,
                "selected_internal_metrics": performance.iloc[0].to_dict(),
                "selected_external_metrics": performance.iloc[1].to_dict(),
                "scientific_status": "Exploratory label-blind training-fold multiple-imputation analysis.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Top candidates:")
    print(candidates.head(20).to_string(index=False))
    print("\nSelected performance:")
    print(
        performance[
            ["cohort", "AUROC", "AUPRC", "Brier", "ACC", "SE", "SP", "F1"]
        ].to_string(index=False)
    )
    print(f"\nOUTPUT_DIR={OUT}")


if __name__ == "__main__":
    main()
