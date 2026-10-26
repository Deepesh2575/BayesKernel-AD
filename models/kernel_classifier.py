"""Soft-margin kernel classifier, including a GMM Fisher kernel (Unit V)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from models.generative import GaussianRegimeModel


class SoftKernelClassifier:
    """SVM probability classifier with linear, RBF, polynomial, or Fisher kernels."""

    SUPPORTED = {"linear", "rbf", "poly", "fisher"}

    def __init__(
        self, kernel: str = "rbf", c: float = 1.0, random_state: int = 17
    ) -> None:
        if kernel not in self.SUPPORTED:
            raise ValueError(f"kernel must be one of {sorted(self.SUPPORTED)}")
        self.kernel = kernel
        self.c = c
        self.random_state = random_state
        self.model_: CalibratedClassifierCV | None = None
        self.fisher_model_: GaussianRegimeModel | None = None
        self._train_fisher: NDArray[np.float64] | None = None

    def fit(
        self, features: NDArray[np.float64], labels: NDArray[np.int64]
    ) -> "SoftKernelClassifier":
        x = np.asarray(features, dtype=float)
        y = np.asarray(labels)
        if len(np.unique(y)) < 2:
            raise ValueError("kernel classifier requires at least two classes")
        if self.kernel == "fisher":
            self.fisher_model_ = GaussianRegimeModel(
                n_components=min(3, max(1, len(x) // 10)),
                random_state=self.random_state,
            ).fit_mle(x)
            self._train_fisher = self._fisher_features(x)
            # A linear SVM on Fisher feature vectors is equivalent to the
            # corresponding precomputed inner-product Fisher kernel.
            kernel, fit_x = "linear", self._train_fisher
        else:
            kernel, fit_x = self.kernel, x
        base_estimator = SVC(
            C=self.c,
            kernel=kernel,
            random_state=self.random_state,
            class_weight="balanced",
        )
        folds = min(5, int(np.bincount(y.astype(int)).min()))
        if folds < 2:
            raise ValueError("each class needs at least two samples for calibration")
        self.model_ = CalibratedClassifierCV(
            estimator=base_estimator, method="sigmoid", cv=folds
        ).fit(fit_x, y)
        return self

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        model = self._require_model()
        x = np.asarray(features, dtype=float)
        if self.kernel == "fisher":
            x = self._fisher_features(x)
        return model.predict_proba(x)

    def decision_function(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        model = self._require_model()
        x = np.asarray(features, dtype=float)
        if self.kernel == "fisher":
            x = self._fisher_features(x)
        return model.decision_function(x)

    @property
    def classes_(self) -> NDArray[np.int64]:
        return self._require_model().classes_

    def _fisher_features(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self.fisher_model_ is not None
        responsibilities = self.fisher_model_.predict_component_proba(x)
        assert self.fisher_model_.model is not None
        residuals = x[:, None, :] - self.fisher_model_.model.means_[None, :, :]
        scales = np.sqrt(
            np.maximum(np.diagonal(self.fisher_model_.model.covariances_, axis1=1, axis2=2), 1e-8)
        )
        feature_map = responsibilities[:, :, None] * residuals / scales[None, :, :]
        return np.concatenate([responsibilities, feature_map.reshape(len(x), -1)], axis=1)

    def _require_model(self) -> CalibratedClassifierCV:
        if self.model_ is None:
            raise RuntimeError("Call fit before prediction")
        return self.model_
