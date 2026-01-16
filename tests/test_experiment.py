"""Tests for Experiment class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from sklab.experiment import Experiment
from sklab.search import SearcherProtocol

from .conftest import InMemoryLogger, make_data, make_pipeline


class TestFit:
    def test_returns_fitted_estimator(self) -> None:
        X, y = make_data()
        experiment = Experiment(pipeline=make_pipeline())
        result = experiment.fit(X, y)

        assert result.estimator is not None
        preds = result.estimator.predict(X[:3])
        assert len(preds) == 3

    def test_clones_pipeline(self) -> None:
        X, y = make_data()
        pipeline = make_pipeline()
        experiment = Experiment(pipeline=pipeline)
        result = experiment.fit(X, y)

        with pytest.raises(NotFittedError):
            check_is_fitted(pipeline)
        check_is_fitted(result.estimator)

    def test_sets_fitted_estimator(self) -> None:
        X, y = make_data()
        experiment = Experiment(pipeline=make_pipeline())
        result = experiment.fit(X, y)

        assert experiment._fitted_estimator is result.estimator

    def test_merges_params(self) -> None:
        X, y = make_data()
        experiment = Experiment(pipeline=make_pipeline())
        result = experiment.fit(X, y, params={"model__C": 0.5})

        assert result.params["model__C"] == 0.5
        assert "scale__with_mean" in result.params

    def test_logs_run_with_config_and_model(self) -> None:
        X, y = make_data()
        logger = InMemoryLogger()
        experiment = Experiment(
            pipeline=make_pipeline(),
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
    def test_requires_prior_fit(self) -> None:
        X, y = make_data()
        experiment = Experiment(pipeline=make_pipeline(), scoring="accuracy")

        with pytest.raises((NotFittedError, TypeError)):
            experiment.evaluate(X, y)

    def test_requires_scoring(self) -> None:
        X, y = make_data()
        experiment = Experiment(pipeline=make_pipeline())
        experiment.fit(X, y)

        with pytest.raises(ValueError, match="scoring is required"):
            experiment.evaluate(X, y)

    def test_uses_init_scoring(self) -> None:
        X, y = make_data()
        experiment = Experiment(pipeline=make_pipeline(), scoring="accuracy")
        experiment.fit(X, y)

        result = experiment.evaluate(X, y)

        assert "accuracy" in result.metrics
        assert 0.0 <= result.metrics["accuracy"] <= 1.0

    def test_works_with_callable_scorer(self) -> None:
        def dummy_scorer(estimator, X, y):
            return 0.42

        X, y = make_data()
        experiment = Experiment(
            pipeline=make_pipeline(),
            scoring=["accuracy", dummy_scorer],
        )
        experiment.fit(X, y)

        result = experiment.evaluate(X, y)

        assert result.metrics["dummy_scorer"] == 0.42
        assert "accuracy" in result.metrics

    def test_logs_metrics_only(self) -> None:
        X, y = make_data()
        logger = InMemoryLogger()
        experiment = Experiment(
            pipeline=make_pipeline(),
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
    def test_requires_scoring(self) -> None:
        X, y = make_data()
        experiment = Experiment(pipeline=make_pipeline())

        with pytest.raises(ValueError, match="scoring is required"):
            experiment.cross_validate(X, y, cv=2)

    def test_returns_fold_metrics(self) -> None:
        X, y = make_data()
        experiment = Experiment(pipeline=make_pipeline(), scoring="accuracy")

        result = experiment.cross_validate(X, y, cv=3, refit=False)

        assert len(result.fold_metrics["accuracy"]) == 3
        assert "cv/accuracy_mean" in result.metrics
        assert "cv/accuracy_std" in result.metrics

    def test_refit_true_returns_estimator(self) -> None:
        X, y = make_data()
        logger = InMemoryLogger()
        experiment = Experiment(
            pipeline=make_pipeline(),
            logger=logger,
            scoring="accuracy",
        )

        result = experiment.cross_validate(X, y, cv=2, refit=True)

        assert result.estimator is not None
        assert experiment._fitted_estimator is result.estimator
        cv_run = logger.runs[0]
        assert len(cv_run.model_calls) == 1

    def test_refit_false_no_estimator(self) -> None:
        X, y = make_data()
        logger = InMemoryLogger()
        experiment = Experiment(
            pipeline=make_pipeline(),
            logger=logger,
            scoring="accuracy",
        )
        original_fitted = experiment._fitted_estimator

        result = experiment.cross_validate(X, y, cv=2, refit=False)

        assert result.estimator is None
        assert experiment._fitted_estimator is original_fitted
        cv_run = logger.runs[0]
        assert len(cv_run.model_calls) == 0

    def test_logs_metrics(self) -> None:
        X, y = make_data()
        logger = InMemoryLogger()
        experiment = Experiment(
            pipeline=make_pipeline(),
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

    def test_accepts_searcher_protocol(self) -> None:
        X, y = make_data()
        pipeline = make_pipeline()
        experiment = Experiment(pipeline=pipeline, scoring="accuracy")
        searcher = cast(SearcherProtocol, self.DummySearcher(pipeline))

        result = experiment.search(searcher, X, y)

        assert result.best_params == {"model__C": 1.0}
        assert result.best_score == 0.95
        assert result.estimator is not None

    def test_logs_params_metrics_model(self) -> None:
        X, y = make_data()
        logger = InMemoryLogger()
        pipeline = make_pipeline()
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

    def test_updates_fitted_estimator(self) -> None:
        X, y = make_data()
        pipeline = make_pipeline()
        experiment = Experiment(pipeline=pipeline, scoring="accuracy")
        searcher = cast(SearcherProtocol, self.DummySearcher(pipeline))

        result = experiment.search(searcher, X, y)

        assert experiment._fitted_estimator is result.estimator

    def test_no_score_skips_metrics_logging(self) -> None:
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

        X, y = make_data()
        logger = InMemoryLogger()
        pipeline = make_pipeline()
        experiment = Experiment(pipeline=pipeline, logger=logger)
        searcher = cast(SearcherProtocol, NoScoreSearcher(pipeline))

        experiment.search(searcher, X, y)

        run = logger.runs[0]
        assert len(run.metrics_calls) == 0

    def test_no_estimator_skips_model_logging(self) -> None:
        @dataclass
        class NoEstimatorSearcher:
            best_params_: dict[str, float] | None = None
            best_score_: float | None = None
            best_estimator_: None = None

            def fit(self, X, y=None) -> NoEstimatorSearcher:
                self.best_params_ = {"model__C": 1.0}
                self.best_score_ = 0.5
                return self

        X, y = make_data()
        logger = InMemoryLogger()
        experiment = Experiment(pipeline=make_pipeline(), logger=logger)
        original_fitted = experiment._fitted_estimator
        searcher = cast(SearcherProtocol, NoEstimatorSearcher())

        experiment.search(searcher, X, y)

        run = logger.runs[0]
        assert len(run.model_calls) == 0
        assert experiment._fitted_estimator is original_fitted

    def test_search_config_protocol(self) -> None:
        @dataclass
        class DummyConfig:
            created_with: dict[str, Any] | None = None

            def create_searcher(self, pipeline, scoring, cv, n_trials, timeout):
                self.created_with = {
                    "pipeline": pipeline,
                    "scoring": scoring,
                    "cv": cv,
                    "n_trials": n_trials,
                    "timeout": timeout,
                }
                return TestSearch.DummySearcher(pipeline)

        X, y = make_data()
        pipeline = make_pipeline()
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

    def test_invalid_search_raises_type_error(self) -> None:
        X, y = make_data()
        experiment = Experiment(pipeline=make_pipeline())

        with pytest.raises(TypeError, match="create_searcher.*fit"):
            experiment.search("not a searcher", X, y)  # type: ignore[arg-type]
