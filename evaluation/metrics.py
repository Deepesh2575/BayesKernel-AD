"""Metrics for classification quality, operational risk, and drift adaptation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix


def detection_metrics(
    labels: NDArray[np.int64],
    decisions: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    cost_matrix: NDArray[np.float64],
    drift_points: NDArray[np.bool_] | None = None,
) -> dict[str, float]:
    """Compute detection, false-alarm, soft accuracy, and expected-risk metrics."""
    y = np.asarray(labels, dtype=int)
    pred = np.asarray(decisions, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    costs = np.asarray(cost_matrix, dtype=float)
    expected = np.mean(costs[y, pred]) if len(y) else float("nan")
    result = {
        "detection_rate": float(tp / max(tp + fn, 1)),
        "false_alarm_rate": float(fp / max(fp + tn, 1)),
        "soft_classification_accuracy": float(np.mean(p[np.arange(len(y)), y])),
        "expected_risk": float(expected),
    }
    if drift_points is not None and np.any(drift_points):
        indices = np.flatnonzero(drift_points)
        result["adaptation_speed"] = float(indices[0])
    else:
        result["adaptation_speed"] = float("nan")
    return result


def adaptation_delay(
    detected: NDArray[np.bool_], true_drift: NDArray[np.bool_]
) -> float:
    """Samples from true drift onset until the first detection, or NaN."""
    actual = np.flatnonzero(true_drift)
    found = np.flatnonzero(detected)
    if not len(actual) or not len(found):
        return float("nan")
    after = found[found >= actual[0]]
    return float(after[0] - actual[0]) if len(after) else float("nan")
