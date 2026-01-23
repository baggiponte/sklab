"""sklearn adapter for hyperparameter search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklab.type_aliases import Scoring


@dataclass(slots=True)
class GridSearchConfig:
    """Quick config for sklearn GridSearchCV."""

    param_grid: Mapping[str, Any]
    scoring: Scoring | Sequence[Scoring] | None = None
    cv: Any | None = None
    refit: bool | str = True
    n_jobs: int | None = None
    verbose: int = 0
    pre_dispatch: str | int | None = "2*n_jobs"
    error_score: float | str = "raise"

    def create_searcher(
        self,
        *,
        pipeline: Any,
        scoring: Scoring | Sequence[Scoring] | None,
        cv: Any | None,
        n_trials: int | None,
        timeout: float | None,
    ) -> GridSearchCV:
        resolved = _resolve_scoring(self.scoring, scoring)
        return GridSearchCV(
            pipeline,
            param_grid=self.param_grid,
            scoring=resolved,
            cv=self.cv if self.cv is not None else cv,
            refit=self.refit,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            pre_dispatch=self.pre_dispatch,
            error_score=self.error_score,
        )


@dataclass(slots=True)
class RandomSearchConfig:
    """Quick config for sklearn RandomizedSearchCV."""

    param_distributions: Mapping[str, Any]
    n_iter: int | None = None
    scoring: Scoring | Sequence[Scoring] | None = None
    cv: Any | None = None
    refit: bool | str = True
    n_jobs: int | None = None
    random_state: int | None = None
    verbose: int = 0
    pre_dispatch: str | int | None = "2*n_jobs"
    error_score: float | str = "raise"

    def create_searcher(
        self,
        *,
        pipeline: Any,
        scoring: Scoring | Sequence[Scoring] | None,
        cv: Any | None,
        n_trials: int | None,
        timeout: float | None,
    ) -> RandomizedSearchCV:
        resolved = _resolve_scoring(self.scoring, scoring)
        resolved_n_iter = self.n_iter or n_trials or 20
        return RandomizedSearchCV(
            pipeline,
            param_distributions=self.param_distributions,
            n_iter=resolved_n_iter,
            scoring=resolved,
            cv=self.cv if self.cv is not None else cv,
            refit=self.refit,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            pre_dispatch=self.pre_dispatch,
            error_score=self.error_score,
        )


def _resolve_scoring(
    config_scoring: Scoring | Sequence[Scoring] | None,
    experiment_scoring: Scoring | Sequence[Scoring] | None,
) -> Scoring | dict[str, Scoring]:
    """Resolve scoring from config or experiment, preferring config."""
    scoring = config_scoring if config_scoring is not None else experiment_scoring
    if scoring is None:
        raise ValueError("scoring or experiment scoring is required for search.")
    return _normalize_scoring(scoring)


def _normalize_scoring(
    scoring: Scoring | Sequence[Scoring],
) -> Scoring | dict[str, Scoring]:
    """Normalize scoring to what sklearn expects."""
    if isinstance(scoring, str):
        return scoring
    if not isinstance(scoring, Sequence):
        # Must be ScorerFunc (callable)
        return scoring
    # Sequence of scorers -> dict
    scorers = cast(Sequence[Scoring], scoring)
    result = {_scorer_name(s): s for s in scorers}
    if len(result) == 1:
        return next(iter(result.values()))
    return result


def _scorer_name(scorer: Scoring) -> str:
    """Get the name for a scorer."""
    if isinstance(scorer, str):
        return scorer
    if callable(scorer):
        return getattr(scorer, "__name__", "custom_scorer")
    return str(scorer)
