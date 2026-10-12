from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def binary_metrics(y_true, y_probability, threshold: float = 0.50) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_probability = np.asarray(y_probability, dtype=float)
    y_pred = (y_probability >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, y_probability)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
