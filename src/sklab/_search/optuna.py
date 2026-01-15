"""Optuna adapter for hyperparameter search."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from sklab._lazy import LazyModule
from sklab.type_aliases import Direction, Scorer, Scorers

if TYPE_CHECKING:
    from optuna.study import Study
    from optuna.trial import FrozenTrial, Trial

optuna = LazyModule("optuna", install_hint="Install it with `uv add optuna`.")


@dataclass(slots=True)
class OptunaConfig:
    """Configuration for Optuna-based hyperparameter search.

    Use this to configure how ``Experiment.search()`` explores the hyperparameter
    space using Optuna's optimization algorithms.

    Parameters
    ----------
    search_space
        A callable that defines the hyperparameter search space. Receives an
        Optuna `Trial`_ object and returns a mapping of parameter names to
        suggested values. Use ``trial.suggest_*`` methods to sample values.

        Example::

            def search_space(trial: Trial) -> dict[str, Any]:
                return {
                    "classifier__C": trial.suggest_float("C", 0.01, 100, log=True),
                    "classifier__kernel": trial.suggest_categorical("kernel", ["rbf", "linear"]),
                }

    n_trials
        Number of trials to run. Each trial evaluates one hyperparameter
        configuration. Default: 50.

    direction
        Optimization direction: ``Direction.MAXIMIZE`` (default) or
        ``Direction.MINIMIZE``. Since ``Direction`` is a ``StrEnum``, you can
        also pass ``"maximize"`` or ``"minimize"`` directly. Use maximize for
        metrics like accuracy; minimize for metrics like log_loss.

    callbacks
        Optional sequence of callbacks invoked after each trial completes.
        Each callback receives the `Study`_ and `FrozenTrial`_ objects.
        Useful for early stopping, logging, or custom pruning logic.
        See `Optuna callbacks tutorial`_.

    study_factory
        Optional factory function to create a custom `Study`_. Receives
        ``direction`` as a keyword argument and returns a Study. Use this
        when you need:

        - A custom `sampler`_ (e.g., ``RandomSampler``, ``CmaEsSampler``)
        - A custom `pruner`_ (e.g., ``HyperbandPruner``)
        - Persistent storage (database URL for resumable studies)
        - A named study for tracking across runs

        Example::

            def my_study_factory(direction: str) -> Study:
                return optuna.create_study(
                    direction=direction,
                    sampler=optuna.samplers.TPESampler(seed=42),
                    pruner=optuna.pruners.HyperbandPruner(),
                    storage="sqlite:///optuna.db",
                    study_name="my-experiment",
                    load_if_exists=True,
                )

        If None, uses ``optuna.create_study(direction=direction)`` with
        defaults (TPE sampler, median pruner, in-memory storage).

    scoring
        Scorer to use for evaluating trials. If None, uses the first scorer
        from the Experiment's scorers. Can be a string (e.g., ``"accuracy"``),
        a callable, or a mapping of names to scorers (first one is used).

    References
    ----------
    .. _Trial: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
    .. _Study: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html
    .. _FrozenTrial: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.FrozenTrial.html
    .. _sampler: https://optuna.readthedocs.io/en/stable/reference/samplers/index.html
    .. _pruner: https://optuna.readthedocs.io/en/stable/reference/pruners.html
    .. _Optuna callbacks tutorial: https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/007_optuna_callback.html
    """

    search_space: Callable[[Trial], Mapping[str, Any]]
    n_trials: int = 50
    direction: Direction = Direction.MAXIMIZE
    callbacks: Sequence[Callable[[Study, FrozenTrial], None]] | None = None
    study_factory: Callable[..., Study] | None = None
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
    search_space: Callable[[Trial], Mapping[str, Any]]
    direction: str
    callbacks: Sequence[Callable[[Study, FrozenTrial], None]] | None
    study_factory: Callable[..., Study] | None
    scoring: Scorer | Mapping[str, Scorer] | None

    best_params_: Mapping[str, Any] | None = None
    best_score_: float | None = None
    best_estimator_: Any | None = None

    def fit(self, X: Any, y: Any | None = None) -> OptunaSearcher:  # noqa: N803
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
    if isinstance(scoring, Mapping):
        if not scoring:
            raise ValueError("At least one scorer is required for search.")
        return next(iter(scoring.values()))
    return scoring
