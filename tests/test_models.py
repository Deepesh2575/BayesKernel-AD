"""Focused correctness tests for the core hybrid components."""

from __future__ import annotations

import unittest

import numpy as np

from config import PipelineConfig
from core.pipeline import BayesKernelAD
from data.generate_synthetic_stream import generate_synthetic_stream
from models.clustering import FuzzyCMeans, FuzzyEMClusterer
from models.decision import bayesian_decision, fuse_posteriors
from models.generative import GaussianRegimeModel
from models.kernel_classifier import SoftKernelClassifier
from models.model_selection import select_cluster_count, select_mixture_components


class ModelTests(unittest.TestCase):
    def test_decision_minimizes_expected_cost(self) -> None:
        posterior = np.array([[0.8, 0.2], [0.1, 0.9]])
        costs = np.array([[0.0, 2.0], [4.0, 0.0]])
        decision, risk = bayesian_decision(posterior, costs)
        self.assertEqual(decision.tolist(), [0, 1])
        self.assertTrue(np.allclose(risk, [[0.8, 1.6], [3.6, 0.2]]))

    def test_fusion_is_normalized(self) -> None:
        result = fuse_posteriors(np.array([[0.8, 0.2]]), np.array([[0.5, 0.5]]))
        self.assertTrue(np.allclose(result.sum(axis=1), 1.0))
        self.assertGreater(result[0, 0], result[0, 1])

    def test_fuzzy_memberships_and_missing_values(self) -> None:
        x = np.array([[0.0, 1.0], [0.1, np.nan], [5.0, 5.0], [5.1, 5.2]])
        model = FuzzyCMeans(n_clusters=2, random_state=2).fit(x)
        probabilities = model.predict_proba(x)
        self.assertEqual(probabilities.shape, (4, 2))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))

    def test_fuzzy_em_hybrid_produces_soft_memberships(self) -> None:
        rng = np.random.default_rng(5)
        x = np.vstack([rng.normal(-2, 0.3, size=(30, 2)), rng.normal(2, 0.3, size=(30, 2))])
        model = FuzzyEMClusterer(n_clusters=2, random_state=5).fit(x)
        memberships = model.predict_proba(x[:5])
        self.assertEqual(memberships.shape, (5, 2))
        self.assertTrue(np.allclose(memberships.sum(axis=1), 1.0))

    def test_map_density_returns_finite_scores(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.normal(size=(60, 3))
        model = GaussianRegimeModel(n_components=2).fit_map(x)
        self.assertTrue(np.isfinite(model.score_samples(x[:5])).all())

    def test_maximum_entropy_fit_uses_one_gaussian(self) -> None:
        x = np.random.default_rng(4).normal(size=(50, 2))
        model = GaussianRegimeModel(n_components=3).fit_maximum_entropy(x)
        self.assertEqual(model.model.n_components, 1)

    def test_fisher_kernel_returns_calibrated_probabilities(self) -> None:
        rng = np.random.default_rng(13)
        labels = np.repeat([0, 1], 30)
        features = rng.normal(size=(60, 2)) + labels[:, None] * 1.5
        model = SoftKernelClassifier(kernel="fisher").fit(features, labels)
        probabilities = model.predict_proba(features[:4])
        self.assertEqual(probabilities.shape, (4, 2))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))

    def test_information_criteria_and_mutual_information_selectors(self) -> None:
        rng = np.random.default_rng(21)
        labels = np.repeat([0, 1], 40)
        features = rng.normal(size=(80, 2)) + labels[:, None] * 3.0
        mixture, bic_scores = select_mixture_components(features, range(1, 3))
        clusterer, mi_scores = select_cluster_count(features, labels, range(2, 4))
        self.assertIn(mixture.n_components, bic_scores)
        self.assertIn(clusterer.n_clusters, mi_scores)

    def test_stream_pipeline_returns_soft_decisions(self) -> None:
        stream = generate_synthetic_stream(n_samples=500, random_state=8)
        model = BayesKernelAD(
            PipelineConfig(
                n_components=1,
                n_clusters=2,
                min_train_size=80,
                drift_window=15,
                drift_threshold=2.0,
            )
        ).fit(stream.features[:100], stream.labels[:100])
        prediction = model.predict(stream.features[100:110])
        self.assertEqual(prediction.probabilities.shape, (10, 2))
        self.assertTrue(np.allclose(prediction.probabilities.sum(axis=1), 1.0))
        self.assertTrue(set(np.unique(prediction.decisions)).issubset({0, 1}))
        model.partial_fit(stream.features[110:115], stream.labels[110:115])

    def test_pipeline_supports_automatic_model_selection(self) -> None:
        stream = generate_synthetic_stream(n_samples=360, random_state=15)
        model = BayesKernelAD(
            PipelineConfig(
                n_components=None,
                n_clusters=None,
                kernel="auto",
                kernel_candidates=("linear", "rbf"),
                min_train_size=60,
                drift_window=15,
            )
        ).fit(stream.features[:120], stream.labels[:120])
        prediction = model.predict(stream.features[120:123])
        self.assertEqual(prediction.probabilities.shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
