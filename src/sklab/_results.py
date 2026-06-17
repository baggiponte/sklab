"""Result dataclasses returned by Experiment methods."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

RawT = TypeVar("RawT")


@dataclass(slots=True)
class FitResult:
    """Result of a single fit run.

    Attributes:
        estimator: The fitted pipeline/estimator.
        metrics: Empty dict (fit doesn't compute metrics).
        params: Merged parameters used for fitting.
        raw: The fitted estimator (same as estimator, for API consistency).
    """

    estimator: Any
    metrics: Mapping[str, float]
    params: Mapping[str, Any]
    raw: Any


@dataclass(slots=True)
class EvalResult:
    """Result of evaluating a fitted estimator on a dataset.

    Attributes:
        metrics: Computed metric scores.
        estimator: The fitted estimator used for evaluation.
        raw: The metrics dict (same as metrics, for API consistency).
    """

    metrics: Mapping[str, float]
    estimator: Any
    raw: Mapping[str, float]


@dataclass(slots=True)
class CVResult:
    """Result of a cross-validation run.

    Attributes:
        metrics: Aggregated metrics (mean/std across folds).
        fold_metrics: Per-fold metric values.
        estimator: Final refitted estimator (if refit=True), else None.
        raw: Full sklearn cross_validate() dict, including fit_time,
            score_time, and test scores for each fold.
    """

    metrics: Mapping[str, float]
    fold_metrics: Mapping[str, list[float]]
    estimator: Any | None
    raw: Mapping[str, Any]


@dataclass(slots=True)
class SearchResult(Generic[RawT]):
    """Result of a hyperparameter search run.

    Attributes:
        best_params: Best hyperparameters found.
        best_score: Best cross-validation score achieved.
        estimator: Best estimator refitted on full data (if refit=True).
        raw: The underlying search object. For OptunaConfig, this is the
            Optuna Study with full trial history. For sklearn searchers
            (GridSearchCV, RandomizedSearchCV), this is the fitted searcher
            with cv_results_ and other attributes.
    """

    best_params: Mapping[str, Any]
    best_score: float | None
    estimator: Any | None
    raw: RawT
