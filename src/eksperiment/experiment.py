"""Experiment runner core types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from sklearn.base import clone
from sklearn.model_selection import cross_validate as sklearn_cross_validate
from sklearn.metrics import get_scorer

from eksperiment.logging.adapters import NoOpLogger
from eksperiment.logging.interfaces import LoggerProtocol
from eksperiment.search import SearchConfigProtocol, SearcherProtocol

MetricFunc = Callable[[Any, Any, Any], float]
Scorer = MetricFunc | str
Scorers = Mapping[str, Scorer]


@dataclass(slots=True)
class FitResult:
    """Result of a single fit run."""

    estimator: Any
    metrics: Mapping[str, float]
    params: Mapping[str, Any]


@dataclass(slots=True)
class EvalResult:
    """Result of evaluating a fitted estimator on a dataset."""

    metrics: Mapping[str, float]


@dataclass(slots=True)
class CVResult:
    """Result of a cross-validation run."""

    metrics: Mapping[str, float]
    fold_metrics: Mapping[str, list[float]]
    estimator: Any | None


@dataclass(slots=True)
class SearchResult:
    """Result of a hyperparameter search run."""

    best_params: Mapping[str, Any]
    best_score: float | None
    estimator: Any | None


@dataclass(slots=True)
class Experiment:
    """Bundle experiment inputs for an sklearn-style run."""

    pipeline: Any
    logger: LoggerProtocol | None = None
    scorers: Scorers | None = None
    name: str | None = None
    tags: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if self.logger is None:
            self.logger = NoOpLogger()

    def fit(
        self,
        X: Any,
        y: Any | None = None,
        *,
        params: Mapping[str, Any] | None = None,
        run_name: str | None = None,
    ) -> FitResult:
        """Fit the pipeline on the provided data and log the run."""
        estimator = clone(self.pipeline)
        merged_params = _merge_params(estimator, params)
        if params:
            estimator.set_params(**params)
        with self.logger.start_run(
            name=run_name or self.name,
            config=merged_params,
            tags=self.tags,
        ) as run:
            estimator.fit(X, y)
            run.log_model(estimator, name="model")
            run.finish()
        return FitResult(estimator=estimator, metrics={}, params=merged_params)

    def evaluate(
        self,
        estimator: Any,
        X: Any,
        y: Any | None = None,
        *,
        run_name: str | None = None,
    ) -> EvalResult:
        """Evaluate a fitted estimator using experiment scorers and log metrics."""
        scorers = _require_scorers(self.scorers)
        metrics = _score_estimator(estimator, X, y, scorers)
        with self.logger.start_run(
            name=run_name or self.name,
            config=None,
            tags=self.tags,
        ) as run:
            run.log_metrics(metrics)
            run.finish()
        return EvalResult(metrics=metrics)

    def cross_validate(
        self,
        X: Any,
        y: Any | None = None,
        *,
        cv: Any,
        refit: bool = True,
        run_name: str | None = None,
    ) -> CVResult:
        """Run sklearn cross-validation, aggregate metrics, and optionally refit."""
        scorers = _require_scorers(self.scorers)
        scoring = _sklearn_scoring(scorers)
        scores = sklearn_cross_validate(
            self.pipeline,
            X,
            y,
            scoring=scoring,
            cv=cv,
            return_train_score=False,
        )
        fold_metrics = {
            name: list(scores[f"test_{name}"]) for name in scoring.keys()
        }
        metrics = _aggregate_cv_metrics(fold_metrics)
        final_estimator = None
        if refit:
            final_estimator = clone(self.pipeline)
            final_estimator.fit(X, y)
        with self.logger.start_run(
            name=run_name or self.name,
            config=None,
            tags=self.tags,
        ) as run:
            run.log_metrics(metrics)
            if final_estimator is not None:
                run.log_model(final_estimator, name="model")
            run.finish()
        return CVResult(
            metrics=metrics,
            fold_metrics=fold_metrics,
            estimator=final_estimator,
        )

    def search(
        self,
        search: SearcherProtocol | SearchConfigProtocol,
        X: Any,
        y: Any | None = None,
        *,
        cv: Any | None = None,
        n_trials: int | None = None,
        timeout: float | None = None,
        run_name: str | None = None,
    ) -> SearchResult:
        """Run a hyperparameter search using a searcher or config object."""
        searcher = _build_searcher(
            search,
            pipeline=self.pipeline,
            scorers=self.scorers,
            cv=cv,
            n_trials=n_trials,
            timeout=timeout,
        )
        with self.logger.start_run(
            name=run_name or self.name,
            config=None,
            tags=self.tags,
        ) as run:
            searcher.fit(X, y)
            best_params = getattr(searcher, "best_params_", {})
            best_score = getattr(searcher, "best_score_", None)
            run.log_params(best_params)
            if best_score is not None:
                run.log_metrics({"best_score": float(best_score)})
            best_estimator = getattr(searcher, "best_estimator_", None)
            if best_estimator is not None:
                run.log_model(best_estimator, name="model")
            run.finish()
        return SearchResult(
            best_params=best_params,
            best_score=best_score,
            estimator=getattr(searcher, "best_estimator_", None),
        )


def _merge_params(estimator: Any, params: Mapping[str, Any] | None) -> dict[str, Any]:
    if hasattr(estimator, "get_params"):
        merged = dict(estimator.get_params())
    else:
        merged = {}
    if params:
        merged.update(params)
    return merged


def _require_scorers(scorers: Scorers | None) -> Scorers:
    if not scorers:
        raise ValueError("Experiment scorers are required for evaluation.")
    return scorers


def _score_estimator(
    estimator: Any, X: Any, y: Any | None, scorers: Scorers
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, scorer in scorers.items():
        scorer_fn = _resolve_scorer(scorer)
        metrics[name] = float(scorer_fn(estimator, X, y))
    return metrics


def _resolve_scorer(scorer: Scorer) -> MetricFunc:
    if isinstance(scorer, str):
        return get_scorer(scorer)
    return scorer


def _sklearn_scoring(scorers: Scorers) -> dict[str, Scorer]:
    scoring: dict[str, Scorer] = {}
    for name, scorer in scorers.items():
        scoring[name] = scorer
    return scoring


def _aggregate_cv_metrics(fold_metrics: Mapping[str, list[float]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, values in fold_metrics.items():
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        metrics[f"cv/{name}_mean"] = float(mean)
        metrics[f"cv/{name}_std"] = float(variance**0.5)
    return metrics


def _build_searcher(
    search: SearcherProtocol | SearchConfigProtocol,
    *,
    pipeline: Any,
    scorers: Scorers | None,
    cv: Any | None,
    n_trials: int | None,
    timeout: float | None,
) -> Any:
    if hasattr(search, "create_searcher"):
        return search.create_searcher(
            pipeline=pipeline,
            scorers=scorers,
            cv=cv,
            n_trials=n_trials,
            timeout=timeout,
        )
    if hasattr(search, "fit"):
        return search
    raise TypeError("search must provide create_searcher(...) or fit(...).")
