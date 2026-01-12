from __future__ import annotations

from dataclasses import dataclass

import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eksperiment.experiment import Experiment


def _make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=200)),
        ]
    )


def _make_data():
    data = load_iris()
    return data.data, data.target


def test_fit_returns_estimator() -> None:
    X, y = _make_data()
    experiment = Experiment(pipeline=_make_pipeline())
    result = experiment.fit(X, y)

    assert result.estimator is not None
    preds = result.estimator.predict(X[:3])
    assert len(preds) == 3


def test_evaluate_uses_init_scorers() -> None:
    X, y = _make_data()
    experiment = Experiment(pipeline=_make_pipeline(), scorers={"acc": "accuracy"})
    fit_result = experiment.fit(X, y)

    eval_result = experiment.evaluate(fit_result.estimator, X, y)
    assert "acc" in eval_result.metrics
    assert 0.0 <= eval_result.metrics["acc"] <= 1.0


def test_evaluate_requires_scorers() -> None:
    X, y = _make_data()
    experiment = Experiment(pipeline=_make_pipeline())
    fit_result = experiment.fit(X, y)

    with pytest.raises(ValueError, match="scorers are required"):
        experiment.evaluate(fit_result.estimator, X, y)


def test_cross_validate_logs_metrics_and_refit() -> None:
    X, y = _make_data()
    experiment = Experiment(pipeline=_make_pipeline(), scorers={"acc": "accuracy"})

    result = experiment.cross_validate(X, y, cv=3, refit=True)

    assert "cv/acc_mean" in result.metrics
    assert "cv/acc_std" in result.metrics
    assert result.estimator is not None


def test_search_accepts_searcher() -> None:
    X, y = _make_data()
    pipeline = _make_pipeline()

    @dataclass
    class DummySearch:
        estimator: Pipeline
        best_params_: dict[str, float] | None = None
        best_score_: float | None = None
        best_estimator_: Pipeline | None = None

        def fit(self, X, y=None):
            self.best_params_ = {"model__C": 1.0}
            self.best_score_ = 0.5
            self.best_estimator_ = clone(self.estimator).fit(X, y)
            return self

    experiment = Experiment(pipeline=pipeline, scorers={"acc": "accuracy"})
    searcher = DummySearch(pipeline)

    result = experiment.search(searcher, X, y)

    assert result.best_params == {"model__C": 1.0}
    assert result.best_score == 0.5
    assert result.estimator is not None
