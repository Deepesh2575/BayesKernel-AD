"""Small shared preprocessing helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class OnlinePreprocessor:
    """Training-fitted imputation and scaling for incoming batches."""

    def __init__(self) -> None:
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.fitted = False

    def fit_transform(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        clean = self.imputer.fit_transform(features)
        result = self.scaler.fit_transform(clean)
        self.fitted = True
        return result

    def transform(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        return self.scaler.transform(self.imputer.transform(features))
