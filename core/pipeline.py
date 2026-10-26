"""Online BayesKernel-AD hybrid detection pipeline."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture

from config import PipelineConfig
from core.utils import OnlinePreprocessor
from models.clustering import DriftDetector, FuzzyEMClusterer
from models.decision import bayesian_decision, fuse_posteriors, normalize_probabilities
from models.generative import GaussianRegimeModel
from models.kernel_classifier import SoftKernelClassifier
from models.model_selection import (
    select_cluster_count,
    select_kernel,
    select_mixture_components,
    should_update_model,
)


@dataclass
class BatchPrediction:
    """Soft predictions and diagnostics for a stream batch."""

    probabilities: NDArray[np.float64]
    decisions: NDArray[np.int64]
    expected_risk: NDArray[np.float64]
    uncertainty: NDArray[np.float64]
    drift_detected: bool


class BayesKernelAD:
    """Batch-updated streaming hybrid with adaptive refitting on detected drift."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.preprocessor = OnlinePreprocessor()
        self.drift_detector = DriftDetector(
            window_size=self.config.drift_window,
            threshold=self.config.drift_threshold,
        )
        self._raw_x: list[NDArray[np.float64]] = []
        self._labels: list[int] = []
        self._recent_predictions: deque[tuple[NDArray[np.float64], int, float]] = deque(
            maxlen=self.config.drift_window
        )
        self._fitted = False
        self._generative: dict[int, GaussianRegimeModel] = {}
        self._classifier: SoftKernelClassifier | None = None
        self._clusterer: FuzzyEMClusterer | None = None
        self._normal_log_likelihood: GaussianMixture | None = None
        self._cluster_anomaly_threshold = 0.0
        self._cluster_anomaly_scale = 1.0
        self._feature_count: int | None = None

    def fit(self, features: NDArray[np.float64], labels: NDArray[np.int64]) -> "BayesKernelAD":
        """Fit all components from labeled warm-up data."""
        raw = np.asarray(features, dtype=float)
        target = np.asarray(labels, dtype=np.int64)
        if raw.ndim != 2 or len(raw) != len(target) or len(raw) == 0:
            raise ValueError("features and labels must be non-empty and row-aligned")
        if not set(np.unique(target)).issubset({0, 1}):
            raise ValueError("labels must use 0 for normal and 1 for anomaly")
        self._feature_count = raw.shape[1]
        self.preprocessor = OnlinePreprocessor()
        x = self.preprocessor.fit_transform(raw)
        self._raw_x = [row.copy() for row in raw]
        self._labels = target.tolist()
        chosen_kernel = self.config.kernel
        if chosen_kernel == "auto":
            chosen_kernel, _ = select_kernel(
                x, target, self.config.kernel_candidates, self.config.random_state
            )
        self._fit_components(x, target, chosen_kernel)
        self._fitted = True
        return self

    def predict(self, features: NDArray[np.float64]) -> BatchPrediction:
        """Predict, then monitor the unlabeled stream for distribution drift."""
        if not self._fitted:
            raise RuntimeError("Call fit with a labeled warm-up batch before predict")
        raw = np.asarray(features, dtype=float)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if self._feature_count != raw.shape[1]:
            raise ValueError("incoming feature count does not match fitted model")
        x = self.preprocessor.transform(raw)
        gen = self._generative_probabilities(x)
        disc = self._classifier_probabilities(x)
        supervised = fuse_posteriors(gen, disc, generative_weight=0.55)
        posterior = fuse_posteriors(
            supervised, self._cluster_probabilities(x), generative_weight=0.8
        )
        decisions, risk = bayesian_decision(posterior, self.config.cost_matrix)
        drift = False
        for raw_row, probability in zip(raw, posterior):
            predicted_class = int(np.argmax(probability))
            confidence = float(np.max(probability))
            self._recent_predictions.append((raw_row.copy(), predicted_class, confidence))
            processed = self.preprocessor.transform(raw_row.reshape(1, -1))[0]
            drift = self.drift_detector.update(processed) or drift
        if drift:
            self._refit_after_drift()
        entropy = -np.sum(posterior * np.log(np.clip(posterior, 1e-12, 1.0)), axis=1)
        return BatchPrediction(posterior, decisions, risk, entropy, drift)

    def partial_fit(
        self, features: NDArray[np.float64], labels: NDArray[np.int64] | None = None
    ) -> None:
        """Append observed samples and refit once enough labeled data is present.

        Online adaptation needs labels for supervised probability calibration.
        Use ``predict`` for unlabeled stream batches; this method requires labels.
        """
        raw = np.asarray(features, dtype=float)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if labels is None:
            raise ValueError("partial_fit requires labels; use predict for unlabeled data")
        target = np.asarray(labels, dtype=np.int64).reshape(-1)
        if len(raw) != len(target):
            raise ValueError("features and labels must have matching row counts")
        if not self._fitted:
            raise RuntimeError("Call fit with a labeled warm-up batch before partial_fit")
        if not set(np.unique(target)).issubset({0, 1}):
            raise ValueError("labels must use 0 for normal and 1 for anomaly")
        self._raw_x.extend(row.copy() for row in raw)
        self._labels.extend(target.tolist())
        if len(self._labels) >= self.config.min_train_size:
            self._consider_labeled_update()

    def _fit_components(
        self,
        x: NDArray[np.float64],
        labels: NDArray[np.int64],
        kernel: str | None = None,
    ) -> None:
        self._generative.clear()
        for regime in (0, 1):
            rows = x[labels == regime]
            if len(rows) >= 2:
                if self.config.n_components is None:
                    max_components = min(6, max(1, len(rows) // 10))
                    selected, _ = select_mixture_components(
                        rows, range(1, max_components + 1), self.config.random_state + regime
                    )
                    component_count = selected.n_components
                else:
                    component_count = min(
                        self.config.n_components, max(1, len(rows) // 10)
                    )
                model = GaussianRegimeModel(
                    n_components=component_count,
                    random_state=self.config.random_state + regime,
                ).fit_map(rows)
                self._generative[regime] = model
        if 0 not in self._generative or 1 not in self._generative:
            raise ValueError("warm-up labels must contain at least two samples per class")

        self._classifier = SoftKernelClassifier(
            kernel or ("rbf" if self.config.kernel == "auto" else self.config.kernel),
            random_state=self.config.random_state,
        ).fit(x, labels)
        normal = x[labels == 0]
        if self.config.n_clusters is None:
            selected_clusterer, _ = select_cluster_count(
                x, labels, random_state=self.config.random_state
            )
            cluster_count = selected_clusterer.n_clusters
        else:
            cluster_count = self.config.n_clusters
        self._clusterer = FuzzyEMClusterer(
            cluster_count, random_state=self.config.random_state
        ).fit(normal)
        distances = self._nearest_cluster_distances(normal)
        self._cluster_anomaly_threshold = float(np.quantile(distances, 0.95))
        self._cluster_anomaly_scale = float(np.std(distances) + 1e-6)
        if self.config.n_components is None:
            max_components = min(6, max(1, len(normal) // 10))
            selected_normal_model, _ = select_mixture_components(
                normal, range(1, max_components + 1), self.config.random_state
            )
            normal_components = selected_normal_model.n_components
        else:
            normal_components = min(
                self.config.n_components, max(1, len(normal) // 10)
            )
        self._normal_log_likelihood = GaussianMixture(
            n_components=normal_components,
            random_state=self.config.random_state,
            reg_covar=1e-5,
        ).fit(normal)

    def _generative_probabilities(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        log_likelihood = np.column_stack(
            [self._generative[regime].score_samples(x) for regime in (0, 1)]
        )
        log_likelihood -= log_likelihood.max(axis=1, keepdims=True)
        likelihood = np.exp(log_likelihood)
        return normalize_probabilities(likelihood)

    def _classifier_probabilities(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self._classifier is not None
        source = self._classifier.predict_proba(x)
        result = np.zeros((len(x), 2))
        for index, label in enumerate(self._classifier.classes_):
            result[:, int(label)] = source[:, index]
        return normalize_probabilities(result)

    def _cluster_probabilities(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        distance = self._nearest_cluster_distances(x)
        score = (distance - self._cluster_anomaly_threshold) / self._cluster_anomaly_scale
        anomaly = 1.0 / (1.0 + np.exp(-np.clip(score, -30.0, 30.0)))
        return np.column_stack([1.0 - anomaly, anomaly])

    def _nearest_cluster_distances(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self._clusterer is not None and self._clusterer.centers_ is not None
        return np.min(
            np.linalg.norm(x[:, None, :] - self._clusterer.centers_[None, :, :], axis=2),
            axis=1,
        )

    def _refit_after_drift(self) -> None:
        if not self._raw_x or not self._labels:
            return
        # Add only high-confidence pseudo-labels from the detected drift window.
        pseudo_labeled = [
            item for item in self._recent_predictions if item[2] >= 0.8
        ]
        self._raw_x.extend(row.copy() for row, _, _ in pseudo_labeled)
        self._labels.extend(label for _, label, _ in pseudo_labeled)
        self._recent_predictions.clear()
        # Keep recent labeled data while retaining a capped history.
        self._raw_x = self._raw_x[-5000:]
        self._labels = self._labels[-5000:]
        raw = np.asarray(self._raw_x)
        labels = np.asarray(self._labels, dtype=np.int64)
        self.preprocessor = OnlinePreprocessor()
        x = self.preprocessor.fit_transform(raw)
        self.drift_detector = DriftDetector(
            window_size=self.config.drift_window,
            threshold=self.config.drift_threshold,
        )
        chosen_kernel = self.config.kernel
        if chosen_kernel == "auto":
            chosen_kernel, _ = select_kernel(
                x, labels, self.config.kernel_candidates, self.config.random_state
            )
        self._fit_components(x, labels, chosen_kernel)

    def _consider_labeled_update(self) -> None:
        """Use BIC improvement to decide whether newly labeled data merits a refit."""
        if self._normal_log_likelihood is None:
            self._refit_after_drift()
            return
        raw = np.asarray(self._raw_x)
        labels = np.asarray(self._labels, dtype=np.int64)
        x = self.preprocessor.transform(raw)
        normal = x[labels == 0]
        if len(normal) < 2:
            return
        if self.config.n_components is None:
            max_components = min(6, max(1, len(normal) // 10))
            selected, _ = select_mixture_components(
                normal, range(1, max_components + 1), self.config.random_state
            )
            count = selected.n_components
        else:
            count = min(self.config.n_components, max(1, len(normal) // 10))
        candidate = GaussianMixture(
            n_components=count, random_state=self.config.random_state, reg_covar=1e-5
        ).fit(normal)
        if should_update_model(
            self._normal_log_likelihood.bic(normal), candidate.bic(normal)
        ):
            self._refit_after_drift()
