"""Default configuration for BayesKernel-AD."""

from dataclasses import dataclass

import numpy as np


@dataclass
class PipelineConfig:
    """Settings for the online hybrid detector."""

    n_components: int | None = 2
    n_clusters: int | None = 3
    kernel: str = "rbf"
    kernel_candidates: tuple[str, ...] = ("linear", "rbf", "poly")
    batch_size: int = 64
    min_train_size: int = 80
    drift_window: int = 40
    drift_threshold: float = 0.35
    random_state: int = 17
    cost_matrix: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.n_components is not None and self.n_components < 1:
            raise ValueError("n_components must be positive or None for BIC selection")
        if self.n_clusters is not None and self.n_clusters < 1:
            raise ValueError("n_clusters must be positive or None for MI selection")
        if self.cost_matrix is None:
            # Rows are true classes [normal, anomaly], columns are decisions.
            self.cost_matrix = np.array([[0.0, 1.0], [5.0, 0.0]])
