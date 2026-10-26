"""Synthetic multivariate stream with controlled anomalies and drift."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class StreamData:
    """Samples, anomaly labels, and a concept-drift indicator."""

    features: NDArray[np.float64]
    labels: NDArray[np.int64]
    drift: NDArray[np.bool_]


def generate_synthetic_stream(
    n_samples: int = 2400,
    n_features: int = 4,
    anomaly_rate: float = 0.08,
    drift_point: float = 0.55,
    random_state: int = 17,
) -> StreamData:
    """Generate a normally operating stream with anomalies and mean drift.

    Syllabus units I/II: samples come from multivariate Gaussian regimes;
    drift changes the normal regime's mean at a known point.
    """
    if n_samples < 2 or n_features < 1:
        raise ValueError("n_samples must be >= 2 and n_features must be >= 1")
    if not 0.0 <= anomaly_rate < 1.0 or not 0.0 < drift_point < 1.0:
        raise ValueError("anomaly_rate must be in [0, 1) and drift_point in (0, 1)")

    rng = np.random.default_rng(random_state)
    change_at = int(n_samples * drift_point)
    drift = np.arange(n_samples) >= change_at
    means = np.zeros((n_samples, n_features))
    means[drift] = np.linspace(1.25, 2.0, n_features)
    covariance = 0.65 * np.eye(n_features) + 0.35 * np.ones((n_features, n_features))
    features = rng.multivariate_normal(np.zeros(n_features), covariance, size=n_samples)
    features += means

    labels = (rng.random(n_samples) < anomaly_rate).astype(np.int64)
    anomaly_indices = np.flatnonzero(labels)
    if anomaly_indices.size:
        directions = rng.choice((-1.0, 1.0), size=(anomaly_indices.size, n_features))
        magnitudes = rng.uniform(3.0, 5.0, size=(anomaly_indices.size, 1))
        features[anomaly_indices] += directions * magnitudes

    missing_mask = rng.random(features.shape) < 0.025
    features[missing_mask] = np.nan
    return StreamData(features=features, labels=labels, drift=drift)
