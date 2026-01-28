"""Experiment runner core types."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast, overload

import numpy as np
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_validate as sklearn_cross_validate
from sklearn.utils.validation import check_is_fitted

from sklab._explain import (
    ExplainerModel,
    ExplainerOutput,
    ExplainerPlotKind,
    ExplainResult,
    compute_shap_explanation,
)
from sklab._results import CVResult, EvalResult, FitResult, SearchResult
from sklab._search.optuna import OptunaConfig, OptunaSearcher
from sklab._search.sklearn import GridSearchConfig, RandomSearchConfig
from sklab.adapters.logging import LoggerProtocol
from sklab.logging import NoOpLogger
from sklab.search import SearchConfigProtocol, SearcherProtocol
from sklab.type_aliases import ScorerFunc, Scoring

if TYPE_CHECKING:
    from collections.abc import Sequence as SequenceType

    from numpy.typing import ArrayLike
    from optuna.study import Study

# Re-export result classes for public API
__all__ = [
    "Experiment",
    "FitResult",
    "EvalResult",
    "CVResult",
    "SearchResult",
    "ExplainResult",
    "ExplainerModel",
    "ExplainerOutput",
    "ExplainerPlotKind",
]


@dataclass(slots=True)
class Experiment:
    """Bundle experiment inputs for an sklearn-style run."""

    pipeline: Any
    logger: LoggerProtocol = field(default_factory=NoOpLogger)
    scoring: Scoring | Sequence[Scoring] | None = None
    name: str | None = None
    tags: Mapping[str, str] | None = None
    _fitted_estimator: Any | None = None

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
        self._fitted_estimator = estimator
        return FitResult(
            estimator=estimator, metrics={}, params=merged_params, raw=estimator
        )

    def evaluate(
        self,
        X: Any,
        y: Any | None = None,
        *,
        run_name: str | None = None,
    ) -> EvalResult:
        """Evaluate the fitted estimator using experiment scoring and log metrics."""
        check_is_fitted(self._fitted_estimator)
        scoring = _require_scoring(self.scoring)
        metrics = _score_estimator(self._fitted_estimator, X, y, scoring)
        with self.logger.start_run(
            name=run_name or self.name,
            config=None,
            tags=self.tags,
        ) as run:
            run.log_metrics(metrics)
        return EvalResult(metrics=metrics, raw=metrics)

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
        scoring_dict = _require_scoring(self.scoring)
        scoring = _sklearn_scoring(scoring_dict)
        scores = sklearn_cross_validate(
            self.pipeline,
            X,
            y,
            scoring=scoring,
            cv=cv,
            return_train_score=False,
        )
        fold_metrics = {name: list(scores[f"test_{name}"]) for name in scoring.keys()}
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
        if final_estimator is not None:
            self._fitted_estimator = final_estimator
        return CVResult(
            metrics=metrics,
            fold_metrics=fold_metrics,
            estimator=final_estimator,
            raw=scores,
        )

    @overload
    def search(
        self,
        search: OptunaConfig | OptunaSearcher,
        X: Any,
        y: Any | None = None,
        *,
        cv: Any | None = None,
        n_trials: int | None = None,
        timeout: float | None = None,
        run_name: str | None = None,
    ) -> SearchResult[Study]: ...

    @overload
    def search(
        self,
        search: GridSearchConfig | GridSearchCV,
        X: Any,
        y: Any | None = None,
        *,
        cv: Any | None = None,
        n_trials: int | None = None,
        timeout: float | None = None,
        run_name: str | None = None,
    ) -> SearchResult[GridSearchCV]: ...

    @overload
    def search(
        self,
        search: RandomSearchConfig | RandomizedSearchCV,
        X: Any,
        y: Any | None = None,
        *,
        cv: Any | None = None,
        n_trials: int | None = None,
        timeout: float | None = None,
        run_name: str | None = None,
    ) -> SearchResult[RandomizedSearchCV]: ...

    @overload
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
    ) -> SearchResult[Any]: ...

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
    ) -> SearchResult[Any]:
        """Run a hyperparameter search using a searcher or config object."""
        searcher = _build_searcher(
            search,
            pipeline=self.pipeline,
            scoring=self.scoring,
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
        if best_estimator is not None:
            self._fitted_estimator = best_estimator
        # Expose Study for Optuna searches, searcher for sklearn searches
        raw = (
            searcher.study
            if isinstance(search, (OptunaConfig, OptunaSearcher))
            else searcher
        )
        return SearchResult(
            best_params=best_params,
            best_score=best_score,
            estimator=best_estimator,
            raw=raw,
        )

    def explain(
        self,
        X: ArrayLike,
        *,
        method: ExplainerModel | str = ExplainerModel.AUTO,
        model_output: ExplainerOutput | str = ExplainerOutput.AUTO,
        background: ArrayLike | int | None = None,
        feature_names: SequenceType[str] | None = None,
        run_name: str | None = None,
    ) -> ExplainResult:
        """Compute SHAP values for the fitted estimator.

        Args:
            X: Samples to explain.
            method: Explainer type. "auto" selects based on estimator structure:
                - Tree models (RandomForest, XGBoost, etc.) -> TreeExplainer
                - Linear models (LogisticRegression, Ridge) -> LinearExplainer
                - Neural networks (Keras, PyTorch) -> DeepExplainer
                - Everything else -> KernelExplainer (slower)
            model_output: What model output to explain. "auto" uses probability for
                classifiers with predict_proba, raw output otherwise. Use "log_odds"
                when comparing SHAP values to logistic regression coefficients.
            background: Background data for KernelExplainer/etc. If int, samples that
                many rows from X. If None, uses X as background.
            feature_names: Feature names to use. If None, attempts to infer from
                pipeline transformers (best-effort; may fall back to generic names
                like x0, x1).
            run_name: Name for the logged run.

        Returns:
            SHAP explanation with values, base values, and feature names. Access the
            raw shap.Explanation via result.raw for advanced use.

        Raises:
            ValueError: If the estimator has not been fitted yet.
            ValueError: If model_output is incompatible with the estimator type.

        Examples:
            >>> exp = Experiment(pipeline=LogisticRegression())
            >>> exp.fit(X_train, y_train)
            >>> result = exp.explain(X_test[:10])
            >>> result.plot("beeswarm")  # Visualize
        """
        if self._fitted_estimator is None:
            raise ValueError(
                "No fitted estimator. Call fit() or cross_validate(refit=True) "
                "before explain()."
            )

        # Compute SHAP explanation
        result = compute_shap_explanation(
            self._fitted_estimator,
            X,
            method=method,
            model_output=model_output,
            background=background,
            feature_names=feature_names,
        )

        # Log metrics
        with self.logger.start_run(
            name=run_name or self.name,
            config=None,
            tags=self.tags,
        ) as run:
            # Log mean |SHAP| per feature as importance metrics
            # Use the normalized 3D values and average over samples and outputs
            values_3d = result.values  # (n_samples, n_features, n_outputs)
            mean_abs_shap = np.abs(values_3d).mean(axis=(0, 2))  # (n_features,)
            if result.feature_names:
                metrics = {
                    f"shap_importance/{name}": float(val)
                    for name, val in zip(result.feature_names, mean_abs_shap)
                }
            else:
                metrics = {
                    f"shap_importance/x{i}": float(val)
                    for i, val in enumerate(mean_abs_shap)
                }
            run.log_metrics(metrics)

        return result


def _merge_params(estimator: Any, params: Mapping[str, Any] | None) -> dict[str, Any]:
    if hasattr(estimator, "get_params"):
        merged = dict(estimator.get_params())
    else:
        merged = {}
    if params:
        merged.update(params)
    return merged


def _normalize_scoring(
    scoring: Scoring | Sequence[Scoring] | None,
) -> dict[str, Scoring]:
    """Convert scoring input to a dict of {name: scorer}."""
    if scoring is None:
        return {}
    if isinstance(scoring, str):
        return {scoring: scoring}
    if not isinstance(scoring, Sequence):
        # Must be ScorerFunc (callable)
        name = getattr(scoring, "__name__", "custom_scorer")
        return {name: scoring}
    scorers = cast(Sequence[Scoring], scoring)
    return {_scorer_name(s): s for s in scorers}


def _scorer_name(scorer: Scoring) -> str:
    """Get the name for a scorer."""
    if isinstance(scorer, str):
        return scorer
    if callable(scorer):
        return getattr(scorer, "__name__", "custom_scorer")
    return str(scorer)


def _require_scoring(
    scoring: Scoring | Sequence[Scoring] | None,
) -> dict[str, Scoring]:
    """Normalize scoring and raise if empty."""
    normalized = _normalize_scoring(scoring)
    if not normalized:
        raise ValueError("Experiment scoring is required for evaluation.")
    return normalized


def _score_estimator(
    estimator: Any, X: Any, y: Any | None, scoring: dict[str, Scoring]
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, scorer in scoring.items():
        scorer_fn = _resolve_scorer(scorer)
        metrics[name] = float(scorer_fn(estimator, X, y))
    return metrics


def _resolve_scorer(scorer: Scoring) -> ScorerFunc:
    if isinstance(scorer, str):
        return get_scorer(scorer)
    return scorer


def _sklearn_scoring(scoring: dict[str, Scoring]) -> dict[str, Scoring]:
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
    scoring: Scoring | Sequence[Scoring] | None,
    cv: Any | None,
    n_trials: int | None,
    timeout: float | None,
) -> Any:
    if isinstance(search, SearchConfigProtocol):
        return search.create_searcher(
            pipeline=pipeline,
            scoring=scoring,
            cv=cv,
            n_trials=n_trials,
            timeout=timeout,
        )
    if hasattr(search, "fit"):
        return search
    raise TypeError("search must provide create_searcher(...) or fit(...).")
