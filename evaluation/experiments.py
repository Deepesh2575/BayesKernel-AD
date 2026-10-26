"""Reproducible synthetic-stream and real-data model comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import PipelineConfig
from core.pipeline import BayesKernelAD
from data.generate_synthetic_stream import generate_synthetic_stream
from data.load_real_data import load_real_data
from evaluation.baselines import (
    ClusteringBaseline,
    DiscriminativeBaseline,
    GenerativeBaseline,
    decisions_from_probabilities,
)
from evaluation.metrics import adaptation_delay, detection_metrics


def _evaluate_split(
    features: np.ndarray, labels: np.ndarray, random_state: int = 17
) -> pd.DataFrame:
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.3, random_state=random_state, stratify=labels
    )
    cost = PipelineConfig().cost_matrix
    models = {
        "pure generative": GenerativeBaseline().fit(train_x, train_y),
        "pure discriminative": DiscriminativeBaseline().fit(train_x, train_y),
        "pure clustering": ClusteringBaseline().fit(train_x, train_y),
    }
    results = []
    for name, model in models.items():
        probability = model.predict_proba(test_x)
        decisions = decisions_from_probabilities(probability, cost)
        results.append(
            {"model": name, **detection_metrics(test_y, decisions, probability, cost)}
        )
    hybrid = BayesKernelAD(PipelineConfig(min_train_size=40)).fit(train_x, train_y)
    prediction = hybrid.predict(test_x)
    results.append(
        {
            "model": "BayesKernel-AD",
            **detection_metrics(test_y, prediction.decisions, prediction.probabilities, cost),
        }
    )
    return pd.DataFrame(results)


def run_experiments(random_state: int = 17) -> pd.DataFrame:
    """Compare all four systems on synthetic and bundled real-world data."""
    stream = generate_synthetic_stream(random_state=random_state)
    synthetic = _evaluate_split(stream.features, stream.labels, random_state)
    synthetic.loc[synthetic["model"] == "BayesKernel-AD", "adaptation_speed"] = (
        evaluate_drift_adaptation(random_state)
    )
    synthetic.insert(0, "dataset", "synthetic stream")
    real_x, real_y, _ = load_real_data(random_state=random_state)
    real = _evaluate_split(real_x, real_y, random_state)
    real.insert(0, "dataset", "breast cancer (binary proxy)")
    return pd.concat([synthetic, real], ignore_index=True)


def evaluate_drift_adaptation(random_state: int = 17) -> float:
    """Measure detector delay in stream samples from known synthetic drift onset."""
    stream = generate_synthetic_stream(random_state=random_state)
    warmup = 200
    model = BayesKernelAD(
        PipelineConfig(
            min_train_size=warmup,
            drift_window=30,
            drift_threshold=1.25,
        )
    ).fit(stream.features[:warmup], stream.labels[:warmup])
    detected = np.zeros(len(stream.labels) - warmup, dtype=bool)
    for i in range(warmup, len(stream.labels)):
        detected[i - warmup] = model.predict(stream.features[i : i + 1]).drift_detected
    return adaptation_delay(detected, stream.drift[warmup:])
