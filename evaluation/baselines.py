"""Pure generative, discriminative, and clustering comparison baselines."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from models.clustering import FuzzyCMeans
from models.decision import bayesian_decision, normalize_probabilities
from models.generative import GaussianRegimeModel


class GenerativeBaseline:
    def fit(self, features: NDArray[np.float64], labels: NDArray[np.int64]) -> "GenerativeBaseline":
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        x = self.scaler.fit_transform(self.imputer.fit_transform(features))
        self.models = {}
        for label in (0, 1):
            subset = x[labels == label]
            self.models[label] = GaussianRegimeModel(
                n_components=min(2, max(1, len(subset) // 10))
            ).fit_mle(subset)
        return self

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        x = self.scaler.transform(self.imputer.transform(features))
        scores = np.column_stack([self.models[k].score_samples(x) for k in (0, 1)])
        scores -= scores.max(axis=1, keepdims=True)
        return normalize_probabilities(np.exp(scores))


class DiscriminativeBaseline:
    def fit(self, features: NDArray[np.float64], labels: NDArray[np.int64]) -> "DiscriminativeBaseline":
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        x = self.scaler.fit_transform(self.imputer.fit_transform(features))
        base_model = SVC(kernel="rbf", class_weight="balanced", random_state=17)
        folds = min(5, int(np.bincount(labels.astype(int)).min()))
        if folds < 2:
            raise ValueError("each class needs at least two samples for calibration")
        self.model = CalibratedClassifierCV(
            estimator=base_model, method="sigmoid", cv=folds
        )
        self.model.fit(x, labels)
        return self

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        x = self.scaler.transform(self.imputer.transform(features))
        source = self.model.predict_proba(x)
        result = np.zeros((len(x), 2))
        for i, label in enumerate(self.model.classes_):
            result[:, int(label)] = source[:, i]
        return normalize_probabilities(result)


class ClusteringBaseline:
    """Fuzzy-clustering novelty detector; threshold learned on training normals."""

    def fit(self, features: NDArray[np.float64], labels: NDArray[np.int64]) -> "ClusteringBaseline":
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        x = self.scaler.fit_transform(self.imputer.fit_transform(features))
        normal = x[labels == 0]
        self.clusterer = FuzzyCMeans(n_clusters=3).fit(normal)
        memberships = self.clusterer.predict_proba(normal)
        entropy = -np.sum(memberships * np.log(np.clip(memberships, 1e-12, 1)), axis=1)
        self.threshold = float(np.quantile(entropy, 0.95))
        return self

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        x = self.scaler.transform(self.imputer.transform(features))
        memberships = self.clusterer.predict_proba(x)
        entropy = -np.sum(memberships * np.log(np.clip(memberships, 1e-12, 1)), axis=1)
        anomaly = 1.0 / (1.0 + np.exp(-8.0 * (entropy - self.threshold)))
        return np.column_stack([1.0 - anomaly, anomaly])


def decisions_from_probabilities(
    probabilities: NDArray[np.float64], cost_matrix: NDArray[np.float64]
) -> NDArray[np.int64]:
    return bayesian_decision(probabilities, cost_matrix)[0]
