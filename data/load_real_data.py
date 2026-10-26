"""Offline-friendly multivariate real dataset loader."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer


def load_real_data(
    random_state: int = 17, max_samples: int | None = 1200
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[str]]:
    """Load Wisconsin breast-cancer data; malignant cases are anomaly class 1.

    The dataset is bundled with scikit-learn, so no network access is required.
    This binary target is a labeled anomaly-detection proxy, not a claim that
    cancer diagnosis is inherently an anomaly-detection problem.
    """
    dataset = load_breast_cancer()
    features = dataset.data.astype(float)
    labels = (dataset.target == 0).astype(np.int64)  # malignant is target 0

    if max_samples is not None and max_samples < len(labels):
        rng = np.random.default_rng(random_state)
        chosen = np.sort(rng.choice(len(labels), size=max_samples, replace=False))
        features, labels = features[chosen], labels[chosen]
    return features, labels, list(dataset.feature_names)
