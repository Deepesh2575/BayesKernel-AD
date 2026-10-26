"""Bayesian posterior fusion and minimum expected-loss decisions (Unit I)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def normalize_probabilities(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize rows safely into categorical distributions."""
    values = np.asarray(probabilities, dtype=float)
    values = np.clip(values, 1e-12, None)
    return values / values.sum(axis=1, keepdims=True)


def bayesian_decision(
    posterior: NDArray[np.float64], cost_matrix: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Choose the action with minimum posterior expected loss.

    Cost-matrix rows index true classes and columns index candidate actions.
    """
    probabilities = normalize_probabilities(posterior)
    costs = np.asarray(cost_matrix, dtype=float)
    if costs.shape[0] != probabilities.shape[1] or costs.ndim != 2:
        raise ValueError("cost_matrix rows must match the number of classes")
    risk = probabilities @ costs
    return np.argmin(risk, axis=1).astype(np.int64), risk


def fuse_posteriors(
    generative: NDArray[np.float64],
    discriminative: NDArray[np.float64],
    generative_weight: float = 0.5,
) -> NDArray[np.float64]:
    """Fuse calibrated class probabilities with a weighted log-opinion pool."""
    if not 0.0 <= generative_weight <= 1.0:
        raise ValueError("generative_weight must be in [0, 1]")
    p_g, p_k = normalize_probabilities(generative), normalize_probabilities(discriminative)
    if p_g.shape != p_k.shape:
        raise ValueError("generative and discriminative probability shapes must match")
    logits = generative_weight * np.log(p_g) + (1.0 - generative_weight) * np.log(p_k)
    logits -= logits.max(axis=1, keepdims=True)
    return normalize_probabilities(np.exp(logits))
