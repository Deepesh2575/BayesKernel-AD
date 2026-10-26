"""Information-criterion model selection utilities (syllabus Unit II)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC

from models.clustering import FuzzyCMeans


def select_mixture_components(
    features: NDArray[np.float64], candidates: range = range(1, 7), random_state: int = 17
) -> tuple[GaussianMixture, dict[int, tuple[float, float]]]:
    """Select by combined AIC/BIC rank and report both scores per candidate."""
    x = np.asarray(features, dtype=float)
    scores: dict[int, tuple[float, float]] = {}
    models: dict[int, GaussianMixture] = {}
    for count in candidates:
        if count < 1 or count > len(x):
            continue
        model = GaussianMixture(
            n_components=count, random_state=random_state, reg_covar=1e-5, n_init=2
        ).fit(x)
        scores[count] = (float(model.aic(x)), float(model.bic(x)))
        models[count] = model
    if not models:
        raise ValueError("no valid component candidates")
    aic_order = {
        count: rank
        for rank, count in enumerate(sorted(scores, key=lambda item: scores[item][0]))
    }
    bic_order = {
        count: rank
        for rank, count in enumerate(sorted(scores, key=lambda item: scores[item][1]))
    }
    best = min(models, key=lambda count: (aic_order[count] + bic_order[count], scores[count][1]))
    return models[best], scores


def mutual_information_scores(
    features: NDArray[np.float64], labels: NDArray[np.int64], random_state: int = 17
) -> NDArray[np.float64]:
    """Estimate per-feature mutual information with a discrete target."""
    return mutual_info_classif(features, labels, random_state=random_state)


def select_cluster_count(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    candidates: range = range(2, 7),
    random_state: int = 17,
) -> tuple[FuzzyCMeans, dict[int, float]]:
    """Choose the fuzzy cluster count with highest membership/label MI."""
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)
    scores: dict[int, float] = {}
    models: dict[int, FuzzyCMeans] = {}
    for count in candidates:
        if count < 2 or count >= len(x):
            continue
        model = FuzzyCMeans(n_clusters=count, random_state=random_state).fit(x)
        memberships = model.predict_proba(x)
        score = mutual_info_classif(
            memberships, y, discrete_features=False, random_state=random_state
        )
        scores[count] = float(np.mean(score))
        models[count] = model
    if not models:
        raise ValueError("no valid cluster-count candidates")
    best = max(models, key=scores.get)
    return models[best], scores


def select_kernel(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    candidates: tuple[str, ...] = ("linear", "rbf", "poly", "fisher"),
    random_state: int = 17,
) -> tuple[str, dict[str, float]]:
    """Choose a kernel by stratified cross-validated accuracy.

    Kernel choice uses held-out predictive utility; mutual information is
    separately used above for cluster-count selection.
    """
    from models.kernel_classifier import SoftKernelClassifier

    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)
    folds = min(5, int(np.bincount(y).min()))
    if folds < 2:
        raise ValueError("each class needs at least two examples for kernel selection")
    scores: dict[str, float] = {}
    for kernel in candidates:
        if kernel not in SoftKernelClassifier.SUPPORTED:
            raise ValueError(f"unsupported kernel: {kernel}")
        if kernel == "fisher":
            # Fisher transformations are learned from each training fold, so
            # evaluate it directly rather than leaking the full-data GMM.
            fold_scores = []
            from sklearn.model_selection import StratifiedKFold

            for train, test in StratifiedKFold(
                n_splits=folds, shuffle=True, random_state=random_state
            ).split(x, y):
                classifier = SoftKernelClassifier(kernel, random_state=random_state)
                classifier.fit(x[train], y[train])
                fold_scores.append(float(np.mean(classifier.predict_proba(x[test]).argmax(axis=1) == y[test])))
            scores[kernel] = float(np.mean(fold_scores))
        else:
            estimator = SVC(kernel=kernel, class_weight="balanced")
            scores[kernel] = float(
                np.mean(
                    cross_val_score(
                        estimator, x, y, cv=folds, scoring="accuracy"
                    )
                )
            )
    return max(scores, key=scores.get), scores


def should_update_model(
    previous_bic: float, current_bic: float, relative_improvement: float = 0.1
) -> bool:
    """Signal refit when a candidate improves BIC by a meaningful fraction."""
    if previous_bic <= 0:
        return current_bic < previous_bic
    return current_bic < previous_bic * (1.0 - relative_improvement)
