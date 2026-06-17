"""Tests for Experiment class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, cast

import optuna
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from sklab._search.optuna import OptunaSearcher
from sklab.experiment import Experiment
from sklab.search import (
    GridSearchConfig,
    OptunaConfig,
    RandomSearchConfig,
    SearcherProtocol,
)
from .conftest import InMemoryLogger


class TestFit:
    def test_returns_fitted_estimator(
        self,
        data: tuple[Any, Any],
        pipeline: Pipeline,
    ) -> None:
        X, y = data
        experiment = Experiment(pipeline=pipeline)
        result = experiment.fit(X, y)

        assert result.estimator is not None
        preds = result.estimator.predict(X[:3])
        assert len(preds) == 3
        assert result.raw is result.estimator

    def test_clones_pipeline(
        self,
        data: tuple[Any, Any],
        pipeline: Pipeline,
    ) -> None:
        X, y = data
        experiment = Experiment(pipeline=pipeline)
        result = experiment.fit(X, y)

        with pytest.raises(NotFittedError):
            check_is_fitted(pipeline)
        check_is_fitted(result.estimator)

    def test_sets_fitted_estimator(
        self,
        data: tuple[Any, Any],
        pipeline: Pipeline,
    ) -> None:
        X, y = data
        experiment = Experiment(pipeline=pipeline)
        result = experiment.fit(X, y)

        assert experiment._fitted_estimator is result.estimator

    def test_merges_params(
        self,
        data: tuple[Any, Any],
        pipeline: Pipeline,
    ) -> None:
        X, y = data
        experiment = Experiment(pipeline=pipeline)
        result = experiment.fit(X, y, params={"model__C": 0.5})

        assert result.params["model__C"] == 0.5
        assert "scale__with_mean" in result.params

    def test_logs_run_with_config_and_model(
        self,
        data: tuple[Any, Any],
        pipeline: Pipeline,
        logger: InMemoryLogger,
    ) -> None:
        X, y = data
        experiment = Experiment(
            pipeline=pipeline,
            logger=logger,
            name="test-fit",
            tags={"env": "test"},
        )
        result = experiment.fit(X, y)

        assert len(logger.runs) == 1
        run = logger.runs[0]
        assert run.name == "test-fit"
        assert run.tags == {"env": "test"}
        assert run.config is not None
        assert "model__C" in run.config
        assert len(run.model_calls) == 1
        assert run.model_calls[0] is result.estimator
        assert len(run.metrics_calls) == 0
        assert len(run.params_calls) == 0


class TestEvaluate:
    def test_requires_prior_fit(
        self,
        data: tuple[Any, Any],
        pipeline_factory: Callable[[], Pipeline],
    ) -> None:
        X, y = data
        experiment = Experiment(pipeline=pipeline_factory(), scoring="accuracy")

        with pytest.raises((NotFittedError, TypeError)):
            experiment.evaluate(X, y)

    def test_requires_scoring(
        self,
        data: tuple[Any, Any],
        pipeline_factory: Callable[[], Pipeline],
    ) -> None:
        X, y = data
        experiment = Experiment(pipeline=pipeline_factory())
        experiment.fit(X, y)

        with pytest.raises(ValueError, match="scoring is required"):
            experiment.evaluate(X, y)

    def test_uses_init_scoring(
        self,
        data: tuple[Any, Any],
        pipeline_factory: Callable[[], Pipeline],
    ) -> None:
        X, y = data
        experiment = Experiment(pipeline=pipeline_factory(), scoring="accuracy")
        experiment.fit(X, y)

        result = experiment.evaluate(X, y)

        assert "accuracy" in result.metrics
        assert 0.0 <= result.metrics["accuracy"] <= 1.0
        assert result.estimator is experiment._fitted_estimator
        assert result.raw is result.metrics

    def test_works_with_callable_scorer(
        self,
        data: tuple[Any, Any],
        pipeline_factory: Callable[[], Pipeline],
    ) -> None:
        def dummy_scorer(estimator, X, y):
            return 0.42

        X, y = data
        experiment = Experiment(
            pipeline=pipeline_factory(),
            scoring=["accuracy", dummy_scorer],
        )
        experiment.fit(X, y)

        result = experiment.evaluate(X, y)

        assert result.metrics["dummy_scorer"] == 0.42
        assert "accuracy" in result.metrics

    def test_logs_metrics_only(
        self,
        data: tuple[Any, Any],
        pipeline_factory: Callable[[], Pipeline],
        logger: InMemoryLogger,
    ) -> None:
        X, y = data
        experiment = Experiment(
            pipeline=pipeline_factory(),
            logger=logger,
            scoring="accuracy",
            name="test-eval",
        )
        experiment.fit(X, y)
        experiment.evaluate(X, y)

        assert len(logger.runs) == 2
        eval_run = logger.runs[1]
        assert eval_run.name == "test-eval"
        assert eval_run.config is None
        assert len(eval_run.metrics_calls) == 1
        assert "accuracy" in eval_run.metrics_calls[0][0]
        assert len(eval_run.model_calls) == 0
        assert len(eval_run.params_calls) == 0


class TestCrossValidate:
    def test_requires_scoring(
        self,
        data: tuple[Any, Any],
        pipeline_factory: Callable[[], Pipeline],
    ) -> None:
        X, y = data
        experiment = Experiment(pipeline=pipeline_factory())

        with pytest.raises(ValueError, match="scoring is required"):
            experiment.cross_validate(X, y, cv=2)

    def test_returns_fold_metrics(
        self,
        data: tuple[Any, Any],
        pipeline_factory: Callable[[], Pipeline],
    ) -> None:
        X, y = data
        experiment = Experiment(pipeline=pipeline_factory(), scoring="accuracy")

        result = experiment.cross_validate(X, y, cv=3, refit=False)

        assert len(result.fold_metrics["accuracy"]) == 3
        assert "cv/accuracy_mean" in result.metrics
        assert "cv/accuracy_std" in result.metrics
        assert "test_accuracy" in result.raw

    def test_refit_true_returns_estimator(
        self,
        data: tuple[Any, Any],
        pipeline_factory: Callable[[], Pipeline],
        logger: InMemoryLogger,
    ) -> None:
        X, y = data
        experiment = Experiment(
            pipeline=pipeline_factory(),
            logger=logger,
            scoring="accuracy",
        )

        result = experiment.cross_validate(X, y, cv=2, refit=True)

        assert result.estimator is not None
        assert experiment._fitted_estimator is result.estimator
        cv_run = logger.runs[0]
        assert len(cv_run.model_calls) == 1

    def test_refit_false_no_estimator(
        self,
        data: tuple[Any, Any],
        pipeline_factory: Callable[[], Pipeline],
        logger: InMemoryLogger,
    ) -> None:
        X, y = data
        experiment = Experiment(
            pipeline=pipeline_factory(),
            logger=logger,
            scoring="accuracy",
        )
        original_fitted = experiment._fitted_estimator

        result = experiment.cross_validate(X, y, cv=2, refit=False)

        assert result.estimator is None
        assert experiment._fitted_estimator is original_fitted
        cv_run = logger.runs[0]
        assert len(cv_run.model_calls) == 0

    def test_logs_metrics(
        self,
        data: tuple[Any, Any],
        pipeline_factory: Callable[[], Pipeline],
        logger: InMemoryLogger,
    ) -> None:
        X, y = data
        experiment = Experiment(
            pipeline=pipeline_factory(),
            logger=logger,
            scoring="accuracy",
        )

        experiment.cross_validate(X, y, cv=2, refit=False)

        cv_run = logger.runs[0]
        assert len(cv_run.metrics_calls) == 1
        metrics = cv_run.metrics_calls[0][0]
        assert "cv/accuracy_mean" in metrics
        assert "cv/accuracy_std" in metrics


class TestSearch:
    @dataclass
    class DummySearcher:
        estimator: Pipeline
        best_params_: dict[str, float] | None = None
        best_score_: float | None = None
        best_estimator_: Pipeline | None = None

        def fit(self, X, y=None) -> TestSearch.DummySearcher:
            self.best_params_ = {"model__C": 1.0}
            self.best_score_ = 0.95
            self.best_estimator_ = clone(self.estimator).fit(X, y)
            return self

    def test_accepts_searcher_protocol(
        self,
        data: tuple[Any, Any],
        pipeline_factory: Callable[[], Pipeline],
    ) -> None:
        X, y = data
        pipeline = pipeline_factory()
        experiment = Experiment(pipeline=pipeline, scoring="accuracy")
        searcher = cast(SearcherProtocol, self.DummySearcher(pipeline))

        result = experiment.search(searcher, X, y)

        assert result.best_params == {"model__C": 1.0}
        assert result.best_score == 0.95
        assert result.estimator is not None
        assert result.raw is searcher

    def test_raw_falls_back_when_study_missing(
        self,
        data: tuple[Any, Any],
        pipeline_factory: Callable[[], Pipeline],
    ) -> None:
        X, y = data
        pipeline = pipeline_factory()
        experiment = Experiment(pipeline=pipeline, scoring="accuracy")
        searcher = cast(SearcherProtocol, self.DummySearcher(pipeline))

        result = experiment.search(searcher, X, y)

        assert result.raw is searcher

    def test_raw_falls_back_when_study_none(
        self,
        data: tuple[Any, Any],
        pipeline_factory: Callable[[], Pipeline],
    ) -> None:
        @dataclass
        class StudyNoneSearcher:
            estimator: Pipeline
            study: None = None
            best_params_: dict[str, float] | None = None
            best_score_: float | None = None
            best_estimator_: Pipeline | None = None

            def fit(self, X, y=None) -> StudyNoneSearcher:
                self.best_params_ = {"model__C": 1.0}
                self.best_score_ = 0.9
                self.best_estimator_ = clone(self.estimator).fit(X, y)
                return self

        X, y = data
        pipeline = pipeline_factory()
        experiment = Experiment(pipeline=pipeline, scoring="accuracy")
        searcher = cast(SearcherProtocol, StudyNoneSearcher(pipeline))

        result = experiment.search(searcher, X, y)

        assert result.raw is searcher

    def test_logs_params_metrics_model(
        self,
        data: tuple[Any, Any],
        pipeline_factory: Callable[[], Pipeline],
        logger: InMemoryLogger,
    ) -> None:
        X, y = data
        pipeline = pipeline_factory()
        experiment = Experiment(
            pipeline=pipeline,
            logger=logger,
            scoring="accuracy",
        )
        searcher = cast(SearcherProtocol, self.DummySearcher(pipeline))

        result = experiment.search(searcher, X, y)

        assert len(logger.runs) == 1
        run = logger.runs[0]
        assert len(run.params_calls) == 1
        assert run.params_calls[0] == {"model__C": 1.0}
        assert len(run.metrics_calls) == 1
        assert run.metrics_calls[0][0] == {"best_score": 0.95}
        assert len(run.model_calls) == 1
        assert run.model_calls[0] is result.estimator

    def test_updates_fitted_estimator(
        self,
        data: tuple[Any, Any],
        pipeline_factory: Callable[[], Pipeline],
    ) -> None:
        X, y = data
        pipeline = pipeline_factory()
        experiment = Experiment(pipeline=pipeline, scoring="accuracy")
        searcher = cast(SearcherProtocol, self.DummySearcher(pipeline))

        result = experiment.search(searcher, X, y)

        assert experiment._fitted_estimator is result.estimator

    def test_no_score_skips_metrics_logging(
        self,
        data: tuple[Any, Any],
        pipeline_factory: Callable[[], Pipeline],
        logger: InMemoryLogger,
    ) -> None:
        @dataclass
        class NoScoreSearcher:
            estimator: Pipeline
            best_params_: dict[str, float] | None = None
            best_score_: None = None
            best_estimator_: Pipeline | None = None

            def fit(self, X, y=None) -> NoScoreSearcher:
                self.best_params_ = {"model__C": 1.0}
                self.best_estimator_ = clone(self.estimator).fit(X, y)
                return self

        X, y = data
        pipeline = pipeline_factory()
        experiment = Experiment(pipeline=pipeline, logger=logger)
        searcher = cast(SearcherProtocol, NoScoreSearcher(pipeline))

        experiment.search(searcher, X, y)

        run = logger.runs[0]
        assert len(run.metrics_calls) == 0

    def test_no_estimator_skips_model_logging(
        self,
        data: tuple[Any, Any],
        pipeline_factory: Callable[[], Pipeline],
        logger: InMemoryLogger,
    ) -> None:
        @dataclass
        class NoEstimatorSearcher:
            best_params_: dict[str, float] | None = None
            best_score_: float | None = None
            best_estimator_: None = None

            def fit(self, X, y=None) -> NoEstimatorSearcher:
                self.best_params_ = {"model__C": 1.0}
                self.best_score_ = 0.5
                return self

        X, y = data
        experiment = Experiment(pipeline=pipeline_factory(), logger=logger)
        original_fitted = experiment._fitted_estimator
        searcher = cast(SearcherProtocol, NoEstimatorSearcher())

        experiment.search(searcher, X, y)

        run = logger.runs[0]
        assert len(run.model_calls) == 0
        assert experiment._fitted_estimator is original_fitted

    def test_search_config_protocol(
        self,
        data: tuple[Any, Any],
        pipeline_factory: Callable[[], Pipeline],
    ) -> None:
        @dataclass
        class DummyConfig:
            created_with: dict[str, Any] | None = None
            last_searcher: TestSearch.DummySearcher | None = None

            def create_searcher(self, pipeline, scoring, cv, n_trials, timeout):
                self.created_with = {
                    "pipeline": pipeline,
                    "scoring": scoring,
                    "cv": cv,
                    "n_trials": n_trials,
                    "timeout": timeout,
                }
                self.last_searcher = TestSearch.DummySearcher(pipeline)
                return self.last_searcher

        X, y = data
        pipeline = pipeline_factory()
        config = DummyConfig()
        experiment = Experiment(
            pipeline=pipeline,
            scoring="accuracy",
        )

        result = experiment.search(config, X, y, cv=3, n_trials=10, timeout=60.0)

        assert config.created_with is not None
        assert config.created_with["pipeline"] is pipeline
        assert config.created_with["scoring"] == "accuracy"
        assert config.created_with["cv"] == 3
        assert config.created_with["n_trials"] == 10
        assert config.created_with["timeout"] == 60.0
        assert result.best_params == {"model__C": 1.0}
        assert config.last_searcher is not None
        assert result.raw is config.last_searcher

    def test_raw_grid_search_config(
        self,
        data: tuple[Any, Any],
        pipeline_factory: Callable[[], Pipeline],
    ) -> None:
        X, y = data
        experiment = Experiment(pipeline=pipeline_factory(), scoring="accuracy")
        config = GridSearchConfig(param_grid={"model__C": [0.1, 1.0]})

        result = experiment.search(config, X, y, cv=2)

        assert isinstance(result.raw, GridSearchCV)

    def test_raw_random_search_config(
        self,
        data: tuple[Any, Any],
        pipeline_factory: Callable[[], Pipeline],
    ) -> None:
        X, y = data
        experiment = Experiment(pipeline=pipeline_factory(), scoring="accuracy")
        config = RandomSearchConfig(
            param_distributions={"model__C": [0.1, 1.0]},
            n_iter=1,
            random_state=42,
        )

        result = experiment.search(config, X, y, cv=2)

        assert isinstance(result.raw, RandomizedSearchCV)

    def test_raw_grid_searcher(
        self,
        data: tuple[Any, Any],
        pipeline_factory: Callable[[], Pipeline],
    ) -> None:
        X, y = data
        experiment = Experiment(pipeline=pipeline_factory(), scoring="accuracy")
        searcher = GridSearchCV(
            pipeline_factory(),
            param_grid={"model__C": [0.1, 1.0]},
            scoring="accuracy",
            cv=2,
        )

        result = experiment.search(searcher, X, y)

        assert result.raw is searcher

    def test_raw_random_searcher(
        self,
        data: tuple[Any, Any],
        pipeline_factory: Callable[[], Pipeline],
    ) -> None:
        X, y = data
        experiment = Experiment(pipeline=pipeline_factory(), scoring="accuracy")
        searcher = RandomizedSearchCV(
            pipeline_factory(),
            param_distributions={"model__C": [0.1, 1.0]},
            n_iter=1,
            scoring="accuracy",
            cv=2,
            random_state=42,
        )

        result = experiment.search(searcher, X, y)

        assert result.raw is searcher

    def test_raw_optuna_config_returns_study(
        self,
        data: tuple[Any, Any],
        pipeline_factory: Callable[[], Pipeline],
    ) -> None:
        def search_space(trial: optuna.trial.Trial) -> dict[str, float]:
            return {"model__C": trial.suggest_float("C", 0.1, 1.0)}

        X, y = data
        experiment = Experiment(pipeline=pipeline_factory(), scoring="accuracy")
        config = OptunaConfig(search_space=search_space, n_trials=2)

        result = experiment.search(config, X, y, cv=2)

        assert isinstance(result.raw, optuna.study.Study)

    def test_raw_optuna_searcher_returns_study(
        self,
        data: tuple[Any, Any],
        pipeline_factory: Callable[[], Pipeline],
    ) -> None:
        def search_space(trial: optuna.trial.Trial) -> dict[str, float]:
            return {"model__C": trial.suggest_float("C", 0.1, 1.0)}

        X, y = data
        experiment = Experiment(pipeline=pipeline_factory(), scoring="accuracy")
        searcher = OptunaSearcher(
            pipeline=pipeline_factory(),
            experiment_scoring="accuracy",
            cv=2,
            n_trials=2,
            timeout=None,
            search_space=search_space,
            direction="maximize",
            callbacks=None,
            study_factory=None,
            config_scoring=None,
        )

        result = experiment.search(searcher, X, y)

        assert isinstance(result.raw, optuna.study.Study)

    def test_invalid_search_raises_type_error(
        self,
        data: tuple[Any, Any],
        pipeline_factory: Callable[[], Pipeline],
    ) -> None:
        X, y = data
        experiment = Experiment(pipeline=pipeline_factory())

        with pytest.raises(TypeError, match="create_searcher.*fit"):
            experiment.search("not a searcher", X, y)  # type: ignore[arg-type]


class TestPromote:
    def test_promote_after_fit_supports_inference(
        self,
        data: tuple[Any, Any],
        pipeline_factory: Callable[[], Pipeline],
    ) -> None:
        X, y = data
        experiment = Experiment(pipeline=pipeline_factory(), name="iris-fit")
        experiment.fit(X, y)

        model = experiment.promote()

        assert model.name == "iris-fit"
        assert model.source == "Experiment"
        assert "model__C" in model.params
        preds = model.predict(X[:3])
        proba = model.predict_proba(X[:3])
        assert len(preds) == 3
        assert len(proba) == 3

    def test_promote_after_search_uses_latest_fitted_estimator(
        self,
        data: tuple[Any, Any],
        pipeline_factory: Callable[[], Pipeline],
    ) -> None:
        X, y = data
        experiment = Experiment(pipeline=pipeline_factory(), scoring="accuracy")
        search = GridSearchConfig(param_grid={"model__C": [0.1, 1.0]})
        experiment.search(search, X, y, cv=2)

        model = experiment.promote(name="best-grid")

        assert model.name == "best-grid"
        assert model.source == "Experiment"
        assert model.estimator is experiment._fitted_estimator

    def test_promote_raises_before_fit(
        self,
        pipeline_factory: Callable[[], Pipeline],
    ) -> None:
        experiment = Experiment(pipeline=pipeline_factory(), scoring="accuracy")

        with pytest.raises(ValueError, match="Cannot promote an unfitted experiment"):
            experiment.promote()
