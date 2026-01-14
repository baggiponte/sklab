"""sklearn adapter for hyperparameter search."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklab.type_aliases import Scorer, Scorers


@dataclass(slots=True)
class GridSearchConfig:
    """Quick config for sklearn GridSearchCV."""

    param_grid: Mapping[str, Any]
    scoring: Scorer | Mapping[str, Scorer] | None = None
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
        scorers: Scorers | None,
        cv: Any | None,
        n_trials: int | None,
        timeout: float | None,
    ) -> GridSearchCV:
        scoring = _resolve_scoring(self.scoring, scorers)
        return GridSearchCV(
            pipeline,
            param_grid=self.param_grid,
            scoring=scoring,
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
    scoring: Scorer | Mapping[str, Scorer] | None = None
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
        scorers: Scorers | None,
        cv: Any | None,
        n_trials: int | None,
        timeout: float | None,
    ) -> RandomizedSearchCV:
        scoring = _resolve_scoring(self.scoring, scorers)
        resolved_n_iter = self.n_iter or n_trials or 20
        return RandomizedSearchCV(
            pipeline,
            param_distributions=self.param_distributions,
            n_iter=resolved_n_iter,
            scoring=scoring,
            cv=self.cv if self.cv is not None else cv,
            refit=self.refit,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            pre_dispatch=self.pre_dispatch,
            error_score=self.error_score,
        )


def _resolve_scoring(
    scoring: Scorer | Mapping[str, Scorer] | None,
    scorers: Scorers | None,
) -> Scorer | Mapping[str, Scorer]:
    if scoring is not None:
        return scoring
    if scorers is None:
        raise ValueError("scoring or experiment scorers are required for search.")
    resolved = dict(scorers)
    if len(resolved) == 1:
        return next(iter(resolved.values()))
    return resolved
