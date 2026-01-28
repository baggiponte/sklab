"""Tests for ExplainResult and helper functions."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklab import Experiment, ExplainResult
from sklab._explain import _compute_mean_abs_shap


class TestShapValues:
    """Tests for SHAP value shapes and normalization."""

    def test_binary_values_shape(self, binary_data):
        """Binary classification: values should be (n_samples, n_features, 1)."""
        X, y = binary_data
        exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
        exp.fit(X, y)
        result = exp.explain(X[:5])

        assert result.values.ndim == 3
        assert result.values.shape[0] == 5
        assert result.values.shape[1] == X.shape[1]

    def test_regression_values_shape(self, regression_data):
        """Regression: values should be 3D."""
        from sklearn.linear_model import Ridge

        X, y = regression_data
        exp = Experiment(pipeline=Ridge())
        exp.fit(X, y)
        result = exp.explain(X[:5])

        assert result.values.ndim == 3
        assert result.values.shape[0] == 5
        assert result.values.shape[1] == X.shape[1]

    def test_multiclass_values_shape(self, multiclass_data):
        """Multiclass: values should be (n_samples, n_features, n_classes)."""
        X, y = multiclass_data
        exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
        exp.fit(X, y)
        result = exp.explain(X[:5])

        assert result.values.ndim == 3
        assert result.values.shape[0] == 5
        assert result.values.shape[2] == 3  # 3 classes in iris


class TestMeanAbsShap:
    """Tests for mean |SHAP| aggregation."""

    def test_2d_input(self):
        shap_values = np.array([[0.1, -0.2], [0.3, -0.4]])
        result = _compute_mean_abs_shap(shap_values)
        expected = np.array([0.2, 0.3])
        np.testing.assert_array_almost_equal(result, expected)

    def test_multiclass_input(self):
        # 2 classes, 3 samples, 2 features
        shap_values = [
            np.array([[0.1, -0.2], [0.3, -0.4], [0.5, -0.6]]),
            np.array([[0.2, -0.3], [0.4, -0.5], [0.6, -0.7]]),
        ]
        result = _compute_mean_abs_shap(shap_values)
        assert result.shape == (2,)


class TestFeatureNames:
    """Tests for feature name recovery."""

    def test_user_override(self, binary_data):
        X, y = binary_data
        n_features = X.shape[1]
        custom_names = [f"feature_{i}" for i in range(n_features)]

        exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
        exp.fit(X, y)
        result = exp.explain(X[:5], feature_names=custom_names)

        assert result.feature_names == custom_names

    def test_fallback_generic(self, binary_data):
        X, y = binary_data
        exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
        exp.fit(X, y)
        result = exp.explain(X[:5])

        expected = [f"x{i}" for i in range(X.shape[1])]
        assert result.feature_names == expected

    def test_pandas_dataframe(self, binary_data):
        """Should get feature names from DataFrame columns."""
        pd = pytest.importorskip("pandas")

        X, y = binary_data
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

        exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
        exp.fit(X_df, y)
        result = exp.explain(X_df.iloc[:10])

        assert isinstance(result, ExplainResult)
        assert result.feature_names == list(X_df.columns)


class TestPlot:
    """Tests for ExplainResult.plot()."""

    def test_passthrough_does_not_raise(self, binary_data):
        plt = pytest.importorskip("matplotlib.pyplot")

        X, y = binary_data
        exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
        exp.fit(X, y)
        result = exp.explain(X[:5])

        result.plot("bar")
        plt.close("all")

    def test_invalid_kind_raises(self, binary_data):
        X, y = binary_data
        exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
        exp.fit(X, y)
        result = exp.explain(X[:5])

        with pytest.raises(ValueError, match="invalid_plot"):
            result.plot("invalid_plot")


class TestDeterminism:
    """Tests for reproducibility."""

    def test_deterministic_with_seed(self, binary_data):
        """Same model + data should produce identical SHAP values."""
        X, y = binary_data

        exp = Experiment(pipeline=LogisticRegression(random_state=42, max_iter=1000))
        exp.fit(X, y)

        result1 = exp.explain(X[:10])
        result2 = exp.explain(X[:10])

        np.testing.assert_array_almost_equal(result1.values, result2.values)
        np.testing.assert_array_almost_equal(result1.base_values, result2.base_values)
