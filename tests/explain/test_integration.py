"""Integration tests for Experiment.explain()."""

from __future__ import annotations

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklab import Experiment, ExplainerModel, ExplainerOutput, ExplainResult
from sklab.search import GridSearchConfig


class TestExplainIntegration:
    """Integration tests for various estimator types."""

    def test_logistic_regression(self, binary_data):
        X, y = binary_data
        exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
        exp.fit(X, y)
        result = exp.explain(X[:10])

        assert isinstance(result, ExplainResult)
        assert result.values.shape[0] == 10
        assert result.base_values is not None
        assert result.raw is not None

    def test_random_forest(self, binary_data):
        X, y = binary_data
        exp = Experiment(
            pipeline=RandomForestClassifier(n_estimators=10, random_state=42)
        )
        exp.fit(X, y)
        result = exp.explain(X[:10])

        assert isinstance(result, ExplainResult)

    def test_pipeline_with_preprocessing(self, binary_data):
        X, y = binary_data
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ])
        exp = Experiment(pipeline=pipeline)
        exp.fit(X, y)
        result = exp.explain(X[:10])

        assert isinstance(result, ExplainResult)
        assert result.feature_names is not None
        assert len(result.feature_names) == X.shape[1]

    def test_fallback_to_kernel_with_warning(self, binary_data):
        """Unsupported estimator should fall back to KernelExplainer."""
        X, y = binary_data
        exp = Experiment(pipeline=SVC(kernel="rbf"))
        exp.fit(X, y)

        with pytest.warns(UserWarning, match="KernelExplainer"):
            result = exp.explain(X[:3])

        assert isinstance(result, ExplainResult)


class TestExplainAfterWorkflows:
    """Tests for explain() after other experiment workflows."""

    def test_after_cross_validate_with_refit(self, binary_data):
        X, y = binary_data
        exp = Experiment(
            pipeline=LogisticRegression(max_iter=1000),
            scoring="accuracy",
        )
        exp.cross_validate(X, y, cv=3, refit=True)
        result = exp.explain(X[:10])

        assert isinstance(result, ExplainResult)

    def test_after_cross_validate_without_refit_raises(self, binary_data):
        X, y = binary_data
        exp = Experiment(
            pipeline=LogisticRegression(max_iter=1000),
            scoring="accuracy",
        )
        exp.cross_validate(X, y, cv=3, refit=False)

        with pytest.raises(ValueError, match="fit"):
            exp.explain(X[:10])

    def test_after_search(self, binary_data):
        X, y = binary_data
        exp = Experiment(
            pipeline=LogisticRegression(max_iter=1000),
            scoring="accuracy",
        )
        exp.search(
            GridSearchConfig(param_grid={"C": [0.1, 1.0]}),
            X, y,
            cv=2,
        )
        result = exp.explain(X[:10])

        assert isinstance(result, ExplainResult)


class TestExplainLogging:
    """Tests for logger integration."""

    def test_logs_metrics(self, binary_data, logger):
        X, y = binary_data
        exp = Experiment(
            pipeline=LogisticRegression(max_iter=1000),
            logger=logger,
        )
        exp.fit(X, y)
        exp.explain(X[:10])

        # Check that shap_importance metrics were logged
        all_metrics = {}
        for run in logger.runs:
            for metrics, _ in run.metrics_calls:
                all_metrics.update(metrics)

        assert any("shap_importance" in key for key in all_metrics)

    def test_logs_correct_metric_count(self, binary_data, logger):
        """Should log one shap_importance metric per feature."""
        X, y = binary_data
        n_features = X.shape[1]
        exp = Experiment(
            pipeline=LogisticRegression(max_iter=1000),
            logger=logger,
        )
        exp.fit(X, y)
        exp.explain(X[:10])

        all_metrics = {}
        for run in logger.runs:
            for metrics, _ in run.metrics_calls:
                all_metrics.update(metrics)

        importance_metrics = [k for k in all_metrics if "shap_importance" in k]
        assert len(importance_metrics) == n_features

    def test_works_without_logger(self, binary_data):
        """explain() should work fine with NoOpLogger (default)."""
        X, y = binary_data
        exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
        exp.fit(X, y)
        result = exp.explain(X[:10])

        assert isinstance(result, ExplainResult)


class TestExplainParameters:
    """Tests for explain() parameters."""

    @pytest.mark.parametrize("method", [ExplainerModel.LINEAR, "linear"])
    def test_accepts_method_string_and_enum(self, binary_data, method):
        X, y = binary_data
        exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
        exp.fit(X, y)
        result = exp.explain(X[:5], method=method)

        assert isinstance(result, ExplainResult)

    @pytest.mark.parametrize("model_output", [ExplainerOutput.PROBABILITY, "probability"])
    def test_accepts_model_output_string_and_enum(self, binary_data, model_output):
        X, y = binary_data
        exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
        exp.fit(X, y)
        result = exp.explain(X[:5], model_output=model_output)

        assert isinstance(result, ExplainResult)

    def test_single_sample(self, binary_data):
        """Explaining a single sample should work."""
        X, y = binary_data
        exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
        exp.fit(X, y)
        result = exp.explain(X[:1])

        assert result.values.shape[0] == 1
        assert isinstance(result, ExplainResult)


class TestExplainErrors:
    """Tests for error handling."""

    def test_unfitted_raises(self, binary_data):
        X, _ = binary_data
        exp = Experiment(pipeline=LogisticRegression())

        with pytest.raises(ValueError, match="fit"):
            exp.explain(X[:5])

    def test_invalid_method_raises(self, binary_data):
        X, y = binary_data
        exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
        exp.fit(X, y)

        with pytest.raises(ValueError):
            exp.explain(X[:5], method="invalid")

    def test_invalid_model_output_raises(self, binary_data):
        X, y = binary_data
        exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
        exp.fit(X, y)

        with pytest.raises(ValueError):
            exp.explain(X[:5], model_output="invalid")

    def test_probability_on_regressor_raises(self, regression_data):
        X, y = regression_data
        exp = Experiment(pipeline=Ridge())
        exp.fit(X, y)

        with pytest.raises(ValueError, match="regressor"):
            exp.explain(X[:5], model_output="probability")

    def test_log_odds_on_regressor_raises(self, regression_data):
        X, y = regression_data
        exp = Experiment(pipeline=Ridge())
        exp.fit(X, y)

        with pytest.raises(ValueError, match="regressor"):
            exp.explain(X[:5], model_output="log_odds")

    def test_feature_names_length_mismatch_raises(self, binary_data):
        X, y = binary_data
        exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
        exp.fit(X, y)

        with pytest.raises(ValueError, match="feature_names"):
            exp.explain(X[:5], feature_names=["a", "b"])

    def test_empty_x_raises(self, binary_data):
        X, y = binary_data
        exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
        exp.fit(X, y)

        with pytest.raises(ValueError, match="empty"):
            exp.explain(X[:0])

    def test_background_larger_than_x_raises(self, binary_data):
        X, y = binary_data
        exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
        exp.fit(X, y)

        with pytest.raises(ValueError, match="background"):
            exp.explain(X[:5], background=100)
