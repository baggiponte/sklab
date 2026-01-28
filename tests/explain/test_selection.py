"""Tests for explainer and output selection logic."""

from __future__ import annotations

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklab import ExplainerModel, ExplainerOutput
from sklab._explain import _default_model_output, _select_explainer_method


class TestExplainerSelection:
    """Tests for automatic explainer selection."""

    @pytest.mark.parametrize(
        "estimator,expected",
        [
            (DecisionTreeClassifier(), ExplainerModel.TREE),
            (RandomForestClassifier(n_estimators=2), ExplainerModel.TREE),
            (LogisticRegression(), ExplainerModel.LINEAR),
            (Ridge(), ExplainerModel.LINEAR),
            (SVC(), ExplainerModel.KERNEL),
        ],
    )
    def test_select_explainer_method(self, estimator, expected, binary_data):
        X, y = binary_data
        estimator.fit(X, y)
        assert _select_explainer_method(estimator) == expected


class TestOutputSelection:
    """Tests for automatic model output selection."""

    @pytest.mark.parametrize(
        "estimator,expected",
        [
            (LogisticRegression(), ExplainerOutput.PROBABILITY),
            (SVC(probability=False), ExplainerOutput.RAW),
            (SVC(probability=True), ExplainerOutput.PROBABILITY),
            (Ridge(), ExplainerOutput.RAW),
        ],
    )
    def test_default_model_output(self, estimator, expected):
        assert _default_model_output(estimator) == expected
