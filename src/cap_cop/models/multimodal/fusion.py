from __future__ import annotations

import numpy as np


def weighted_average_fusion(
    clinical_probability: np.ndarray,
    imaging_probability: np.ndarray,
    clinical_weight: float,
    imaging_weight: float,
) -> np.ndarray:
    if not np.isclose(clinical_weight + imaging_weight, 1.0):
        raise ValueError("Fusion weights must sum to 1.0.")
    clinical_probability = np.asarray(clinical_probability, dtype=float)
    imaging_probability = np.asarray(imaging_probability, dtype=float)
    if clinical_probability.shape != imaging_probability.shape:
        raise ValueError("Clinical and imaging probability arrays must have the same shape.")
    return clinical_weight * clinical_probability + imaging_weight * imaging_probability


def classify(probability: np.ndarray, threshold: float = 0.50) -> np.ndarray:
    return (np.asarray(probability) >= threshold).astype(int)
