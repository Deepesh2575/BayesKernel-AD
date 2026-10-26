"""Gaussian and Gaussian-mixture regime models (syllabus Units I and II)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture


@dataclass
class GaussianRegimeModel:
    """Class-conditional Gaussian mixture density estimator.

    ``fit_mle`` uses the standard EM maximum-likelihood estimator.
    ``fit_map`` adds a weak empirical-Bayes shrinkage prior on component means
    and covariances after the EM fit, providing a regularized MAP-style estimate.
    """

    n_components: int = 2
    covariance_type: str = "full"
    reg_covar: float = 1e-5
    random_state: int = 17
    model: GaussianMixture | None = None
    global_mean_: NDArray[np.float64] | None = None
    global_covariance_: NDArray[np.float64] | None = None

    def fit_mle(self, features: NDArray[np.float64]) -> "GaussianRegimeModel":
        """Fit a Gaussian mixture by maximum likelihood using EM."""
        x = self._validate(features)
        self.model = GaussianMixture(
            n_components=min(self.n_components, len(x)),
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            random_state=self.random_state,
            n_init=2,
        ).fit(x)
        self.global_mean_ = np.mean(x, axis=0)
        self.global_covariance_ = np.atleast_2d(np.cov(x, rowvar=False))
        return self

    def fit_map(
        self, features: NDArray[np.float64], prior_strength: float = 0.05
    ) -> "GaussianRegimeModel":
        """Fit EM then shrink parameters toward a global Gaussian prior.

        This lightweight empirical-Bayes MAP approximation is stable for small
        batches; it is not a full posterior sampler.
        """
        if not 0.0 <= prior_strength < 1.0:
            raise ValueError("prior_strength must be in [0, 1)")
        self.fit_mle(features)
        assert self.model is not None
        assert self.global_mean_ is not None and self.global_covariance_ is not None
        strength = prior_strength
        self.model.means_ = (
            (1.0 - strength) * self.model.means_ + strength * self.global_mean_
        )
        if self.model.covariance_type == "full":
            self.model.covariances_ = (
                (1.0 - strength) * self.model.covariances_
                + strength * self.global_covariance_[None, :, :]
            )
            self.model.precisions_cholesky_ = np.linalg.cholesky(
                np.linalg.inv(self.model.covariances_)
            ).transpose(0, 2, 1)
        return self

    def fit_maximum_entropy(
        self, features: NDArray[np.float64]
    ) -> "GaussianRegimeModel":
        """Fit the max-entropy Gaussian under empirical mean/covariance constraints.

        A Gaussian maximizes differential entropy among continuous
        distributions with fixed first and second moments, so the one-component
        Gaussian fit is the maximum-entropy estimate for these constraints.
        """
        self.n_components = 1
        return self.fit_mle(features)

    def score_samples(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return per-sample log likelihoods."""
        return self._require_model().score_samples(self._validate(features))

    def predict_component_proba(
        self, features: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return posterior component responsibilities."""
        return self._require_model().predict_proba(self._validate(features))

    def _require_model(self) -> GaussianMixture:
        if self.model is None:
            raise RuntimeError("Call fit_mle or fit_map before prediction")
        return self.model

    @staticmethod
    def _validate(features: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("features must be a non-empty 2D array")
        if not np.isfinite(x).all():
            raise ValueError("features must be imputed before density estimation")
        return x
