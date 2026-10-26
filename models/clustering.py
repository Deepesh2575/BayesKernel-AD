"""Fuzzy c-means, EM clustering, and lightweight drift detection (Unit IV)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture


@dataclass
class FuzzyCMeans:
    """Fuzzy c-means clustering with missing-value support."""

    n_clusters: int = 3
    fuzziness: float = 2.0
    max_iter: int = 150
    tolerance: float = 1e-5
    random_state: int = 17
    centers_: NDArray[np.float64] | None = None
    impute_values_: NDArray[np.float64] | None = None

    def fit(self, features: NDArray[np.float64]) -> "FuzzyCMeans":
        """Fit cluster centers; missing values use per-feature training means."""
        raw = np.asarray(features, dtype=float)
        if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] == 0:
            raise ValueError("features must be a non-empty 2D array")
        if self.n_clusters < 1 or self.fuzziness <= 1:
            raise ValueError("n_clusters must be positive and fuzziness > 1")
        self.impute_values_ = np.nanmean(raw, axis=0)
        self.impute_values_ = np.where(np.isfinite(self.impute_values_), self.impute_values_, 0.0)
        x = self._impute(raw)
        rng = np.random.default_rng(self.random_state)
        membership = rng.dirichlet(np.ones(self.n_clusters), size=len(x))
        exponent = 2.0 / (self.fuzziness - 1.0)

        for _ in range(self.max_iter):
            weights = membership**self.fuzziness
            centers = (weights.T @ x) / np.maximum(weights.sum(axis=0)[:, None], 1e-12)
            distances = np.maximum(
                np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2), 1e-10
            )
            inverse = distances ** (-exponent)
            updated = inverse / inverse.sum(axis=1, keepdims=True)
            if np.linalg.norm(updated - membership) < self.tolerance:
                membership = updated
                break
            membership = updated

        self.centers_ = centers
        return self

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return soft cluster memberships for each row."""
        if self.centers_ is None:
            raise RuntimeError("Call fit before predict_proba")
        x = self._impute(np.asarray(features, dtype=float))
        distances = np.maximum(
            np.linalg.norm(x[:, None, :] - self.centers_[None, :, :], axis=2), 1e-10
        )
        inverse = distances ** (-2.0 / (self.fuzziness - 1.0))
        return inverse / inverse.sum(axis=1, keepdims=True)

    def _impute(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.impute_values_ is None:
            raise RuntimeError("Imputation values are unavailable before fitting")
        return np.where(np.isfinite(features), features, self.impute_values_)


@dataclass
class EMClusterer:
    """Gaussian-mixture EM clustering baseline."""

    n_clusters: int = 3
    random_state: int = 17
    model_: GaussianMixture | None = None

    def fit(self, features: NDArray[np.float64]) -> "EMClusterer":
        self.model_ = GaussianMixture(
            n_components=min(self.n_clusters, len(features)),
            random_state=self.random_state,
            reg_covar=1e-5,
        ).fit(np.asarray(features, dtype=float))
        return self

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.model_ is None:
            raise RuntimeError("Call fit before predict_proba")
        return self.model_.predict_proba(np.asarray(features, dtype=float))


@dataclass
class FuzzyEMClusterer:
    """Hybrid fuzzy c-means / Gaussian-mixture EM clustering.

    Fuzzy memberships initialize the GMM component means, weights, and
    covariance matrices; final soft memberships combine fuzzy and EM scores.
    """

    n_clusters: int = 3
    fuzziness: float = 2.0
    random_state: int = 17
    fuzzy_: FuzzyCMeans | None = None
    model_: GaussianMixture | None = None

    @property
    def centers_(self) -> NDArray[np.float64] | None:
        return None if self.fuzzy_ is None else self.fuzzy_.centers_

    def fit(self, features: NDArray[np.float64]) -> "FuzzyEMClusterer":
        """Initialize and fit the hybrid fuzzy-EM clustering model."""
        raw = np.asarray(features, dtype=float)
        self.n_clusters = min(self.n_clusters, len(raw))
        self.fuzzy_ = FuzzyCMeans(
            self.n_clusters, fuzziness=self.fuzziness, random_state=self.random_state
        ).fit(raw)
        assert self.fuzzy_.centers_ is not None
        assert self.fuzzy_.impute_values_ is not None
        clean = np.where(np.isfinite(raw), raw, self.fuzzy_.impute_values_)
        membership = self.fuzzy_.predict_proba(clean)
        weights = membership**self.fuzziness
        covariances = []
        for cluster in range(self.n_clusters):
            difference = clean - self.fuzzy_.centers_[cluster]
            covariance = (difference.T * weights[:, cluster]) @ difference
            covariance /= max(weights[:, cluster].sum(), 1e-12)
            covariance += 1e-5 * np.eye(clean.shape[1])
            covariances.append(covariance)
        self.model_ = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type="full",
            weights_init=np.maximum(weights.mean(axis=0), 1e-8)
            / np.maximum(weights.mean(axis=0), 1e-8).sum(),
            means_init=self.fuzzy_.centers_,
            precisions_init=np.linalg.inv(np.asarray(covariances)),
            init_params="random",
            n_init=1,
            random_state=self.random_state,
            reg_covar=1e-5,
        ).fit(clean)
        return self

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Combine fuzzy memberships and GMM posterior responsibilities."""
        if self.fuzzy_ is None or self.model_ is None:
            raise RuntimeError("Call fit before predict_proba")
        raw = np.asarray(features, dtype=float)
        assert self.fuzzy_.impute_values_ is not None
        clean = np.where(np.isfinite(raw), raw, self.fuzzy_.impute_values_)
        fuzzy = self.fuzzy_.predict_proba(clean)
        em = self.model_.predict_proba(clean)
        combined = fuzzy * em
        return combined / np.maximum(combined.sum(axis=1, keepdims=True), 1e-12)


@dataclass
class DriftDetector:
    """Detect sustained feature-distribution shifts from a reference window."""

    window_size: int = 40
    threshold: float = 0.35
    reference_: NDArray[np.float64] | None = None
    recent_: deque[NDArray[np.float64]] | None = None

    def update(self, sample: NDArray[np.float64]) -> bool:
        """Add one observation and return True once a mean shift is detected."""
        row = np.asarray(sample, dtype=float).reshape(1, -1)
        if not np.isfinite(row).all():
            return False
        if self.recent_ is None:
            self.recent_ = deque(maxlen=self.window_size)
        self.recent_.append(row[0].copy())
        if len(self.recent_) < self.window_size:
            return False
        current = np.asarray(self.recent_)
        if self.reference_ is None:
            self.reference_ = current.copy()
            return False
        scale = np.std(self.reference_, axis=0) + 1e-6
        distance = np.linalg.norm((current.mean(axis=0) - self.reference_.mean(axis=0)) / scale)
        if distance > self.threshold:
            self.reference_ = current.copy()
            return True
        self.reference_ = 0.95 * self.reference_ + 0.05 * current
        return False
