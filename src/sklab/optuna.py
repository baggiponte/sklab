"""Optuna adapter for hyperparameter search."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from sklab.search import Scorer, Scorers


@dataclass(slots=True)
class OptunaConfig:
    """Quick Optuna config for Experiment.search()."""

    search_space: Callable[[Any], Mapping[str, Any]]
    n_trials: int = 50
    direction: str = "maximize"
    callbacks: Sequence[Callable[[Any, Any], None]] | None = None
    study_factory: Callable[..., Any] | None = None
    scoring: Scorer | Mapping[str, Scorer] | None = None

    def create_searcher(
        self,
        *,
        pipeline: Any,
        scorers: Scorers | None,
        cv: Any | None,
        n_trials: int | None,
        timeout: float | None,
    ) -> OptunaSearcher:
        return OptunaSearcher(
            pipeline=pipeline,
            scorers=scorers,
            cv=cv,
            n_trials=n_trials or self.n_trials,
            timeout=timeout,
            search_space=self.search_space,
            direction=self.direction,
            callbacks=self.callbacks,
            study_factory=self.study_factory,
            scoring=self.scoring,
        )


@dataclass(slots=True)
class OptunaSearcher:
    pipeline: Any
    scorers: Scorers | None
    cv: Any | None
    n_trials: int
    timeout: float | None
    search_space: Callable[[Any], Mapping[str, Any]]
    direction: str
    callbacks: Sequence[Callable[[Any, Any], None]] | None
    study_factory: Callable[..., Any] | None
    scoring: Scorer | Mapping[str, Scorer] | None

    best_params_: Mapping[str, Any] | None = None
    best_score_: float | None = None
    best_estimator_: Any | None = None

    def fit(self, X: Any, y: Any | None = None) -> OptunaSearcher:  # noqa: N803
        optuna = _require_optuna()
        scoring = _resolve_scoring(self.scoring, self.scorers)
        scorer = _pick_primary_scorer(scoring)

        def objective(trial: Any) -> float:
            params = dict(self.search_space(trial))
            estimator = clone(self.pipeline).set_params(**params)
            score = cross_val_score(
                estimator,
                X,
                y,
                scoring=scorer,
                cv=self.cv,
            ).mean()
            trial.set_user_attr("params", params)
            return float(score)

        if self.study_factory is None:
            study = optuna.create_study(direction=self.direction)
        else:
            study = self.study_factory(direction=self.direction)
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            callbacks=list(self.callbacks or ()),
        )

        self.best_score_ = float(study.best_value)
        self.best_params_ = dict(study.best_trial.user_attrs["params"])
        self.best_estimator_ = (
            clone(self.pipeline).set_params(**self.best_params_).fit(X, y)
        )
        return self


def _require_optuna() -> Any:
    try:
        import optuna
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
        raise ModuleNotFoundError(
            "Optuna is required. Install it with `uv add optuna`."
        ) from exc
    return optuna


def _resolve_scoring(
    scoring: Scorer | Mapping[str, Scorer] | None,
    scorers: Scorers | None,
) -> Scorer | Mapping[str, Scorer]:
    if scoring is not None:
        return scoring
    if scorers is None:
        raise ValueError("scoring or experiment scorers are required for search.")
    return dict(scorers)


def _pick_primary_scorer(scoring: Scorer | Mapping[str, Scorer]) -> Scorer:
    if isinstance(scoring, dict):
        if not scoring:
            raise ValueError("At least one scorer is required for search.")
        return next(iter(scoring.values()))
    return scoring
