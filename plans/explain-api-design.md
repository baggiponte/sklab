---
title: Explain API Design
description: SHAP integration via Experiment.explain() method
date: 2025-01-23
---

# Explain API Design

## Goal

Add model explanation capabilities to sklab through a new `Experiment.explain()` method that computes SHAP values with zero boilerplate, following the same inject-once-forget philosophy as the rest of the library.

## The Problem

Computing SHAP values requires:
1. Choosing the right explainer for your model type
2. Managing background data sampling
3. Handling the explainer → values → visualization pipeline
4. Integrating with experiment tracking (logging plots/values)

This is tedious boilerplate that sklab should remove.

## Design Principles

1. **One method, one job** — `explain()` does SHAP. Nothing else.
2. **SHAP does SHAP well** — Expose the `shap.Explanation` object. The `plot()` method is a thin passthrough to `shap.plots`, not a wrapper.
3. **Separate concerns** — Explanation (why this prediction?) is distinct from diagnostics (how well does it perform?) and feature analysis (what features matter globally?).
4. **Compute by default, plot on demand** — `explain()` computes values only. Plotting is user-initiated via `result.plot()`. No automatic plot logging (avoids matplotlib dependency issues in headless environments).

## API Design

### Type Definitions

```python
from enum import StrEnum, auto

class ExplainerMethod(StrEnum):
    """SHAP explainer selection strategy."""
    AUTO = auto()        # Select based on estimator type
    TREE = auto()        # TreeExplainer (RF, GBM, XGBoost, LightGBM, CatBoost)
    LINEAR = auto()      # LinearExplainer (LogisticRegression, Ridge, Lasso)
    KERNEL = auto()      # KernelExplainer (model-agnostic, slow)
    DEEP = auto()        # DeepExplainer (neural networks)

class ModelOutput(StrEnum):
    """What model output to explain."""
    AUTO = auto()        # probability for classifiers with predict_proba, raw otherwise
    RAW = auto()         # Raw model output (logits, regression values)
    PROBABILITY = auto() # Class probabilities (classifiers only)
    LOG_ODDS = auto()    # Log-odds (classifiers only, for coefficient comparison)
```

### Core Method

```python
class Experiment:
    def explain(
        self,
        X,
        *,
        method: ExplainerMethod | str = ExplainerMethod.AUTO,
        model_output: ModelOutput | str = ModelOutput.AUTO,
        background: ArrayLike | int | None = None,
        feature_names: Sequence[str] | None = None,
        run_name: str | None = None,
    ) -> ExplainResult:
        """
        Compute SHAP values for the fitted estimator.

        Parameters
        ----------
        X : array-like
            Samples to explain.
        method : ExplainerMethod or str, default="auto"
            Explainer type. "auto" selects based on estimator structure.
        model_output : ModelOutput or str, default="auto"
            What model output to explain. "auto" uses probability for classifiers
            with predict_proba, raw output otherwise. Use "log_odds" when comparing
            SHAP values to logistic regression coefficients.
        background : array-like or int, optional
            Background data for KernelExplainer/etc. If int, samples that many
            rows from X. If None, uses shap.maskers.Independent or model-appropriate default.
        feature_names : sequence of str, optional
            Feature names to use. If None, attempts to infer from pipeline
            transformers (best-effort; may fall back to generic names).
        run_name : str, optional
            Name for the logged run.

        Returns
        -------
        ExplainResult
            SHAP explanation with values, base values, and feature names.
        """
```

### Result Type

```python
@dataclass(slots=True)
class ExplainResult:
    """SHAP explanation result."""

    values: np.ndarray
    """SHAP values array. Shape: (n_samples, n_features) or (n_samples, n_features, n_classes)."""

    base_values: np.ndarray
    """Expected value(s) of the model output."""

    data: np.ndarray
    """The input data that was explained."""

    feature_names: list[str] | None
    """Feature names if available."""

    raw: shap.Explanation
    """The underlying shap.Explanation object for advanced use."""

    def plot(self, kind: str = "summary", **kwargs) -> None:
        """
        Convenience wrapper around SHAP plotting functions.

        Parameters
        ----------
        kind : str
            Plot type: "summary", "bar", "beeswarm", "waterfall", "force", "dependence".
        **kwargs
            Passed to the underlying shap plot function.
        """
        # Delegates to shap.plots.{kind}(self.raw, **kwargs)
```

### Explainer Selection ("auto" mode)

| Estimator Type | Explainer |
|----------------|-----------|
| Tree-based (RF, GBM, XGBoost, LightGBM, CatBoost) | `TreeExplainer` |
| Linear models (LogisticRegression, LinearRegression, Ridge, Lasso) | `LinearExplainer` |
| Neural networks (Keras, PyTorch via skorch) | `DeepExplainer` (if available) |
| Everything else | `KernelExplainer` (with background sampling) |

Detection uses **structural checks** (attributes like `tree_`, `coef_`) and sklearn's `is_classifier`/`is_regressor`, not brittle class name matching:

```python
from sklearn.base import is_classifier, is_regressor

def _select_explainer_method(estimator) -> ExplainerMethod:
    """Select SHAP explainer based on estimator structure."""

    # 1. Check for tree structure (sklearn trees and ensembles)
    if hasattr(estimator, "tree_"):
        return ExplainerMethod.TREE
    if hasattr(estimator, "estimators_"):
        first = estimator.estimators_[0]
        check = first.flat[0] if isinstance(first, np.ndarray) else first
        if hasattr(check, "tree_"):
            return ExplainerMethod.TREE

    # 2. Check for linear structure
    if hasattr(estimator, "coef_"):
        return ExplainerMethod.LINEAR

    # 3. Check external libraries by module (not class name)
    module = type(estimator).__module__
    if any(lib in module for lib in ("xgboost", "lightgbm", "catboost")):
        return ExplainerMethod.TREE
    if any(lib in module for lib in ("keras", "tensorflow", "torch")):
        return ExplainerMethod.DEEP

    # 4. Fallback to kernel (model-agnostic)
    return ExplainerMethod.KERNEL


def _default_model_output(estimator) -> ModelOutput:
    """Select model output based on estimator type."""
    if is_classifier(estimator) and hasattr(estimator, "predict_proba"):
        return ModelOutput.PROBABILITY
    return ModelOutput.RAW
```

### Pipeline Handling

For sklearn Pipelines:
1. Extract the final estimator
2. Transform X through all preprocessing steps
3. Apply explainer to final estimator with transformed X
4. Attempt to recover feature names (best-effort)

```python
# Internal logic
if hasattr(estimator, "steps"):  # It's a Pipeline
    preprocessor = Pipeline(estimator.steps[:-1])
    final_estimator = estimator.steps[-1][1]
    X_transformed = preprocessor.transform(X)
    explainer = select_explainer(final_estimator, X_transformed, method, background)
else:
    explainer = select_explainer(estimator, X, method, background)
```

#### Feature Name Recovery (Best-Effort)

Feature names after transformation are recovered using `get_feature_names_out()` (sklearn 1.0+):

```python
def _get_feature_names(preprocessor, X, user_provided: Sequence[str] | None) -> list[str] | None:
    """Attempt to recover feature names after preprocessing."""
    # 1. User override takes precedence
    if user_provided is not None:
        return list(user_provided)

    # 2. Try sklearn's get_feature_names_out
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            return list(preprocessor.get_feature_names_out())
        except (ValueError, AttributeError):
            pass  # Some transformers don't support this

    # 3. Fall back to generic names
    n_features = X.shape[1] if hasattr(X, "shape") else len(X[0])
    return [f"x{i}" for i in range(n_features)]
```

**Limitations:**
- `ColumnTransformer` with `passthrough` and missing `get_feature_names_out` may fail
- Some custom transformers don't implement `get_feature_names_out`
- When recovery fails, generic names (`x0`, `x1`, ...) are used

**Mitigation:** Users can always pass explicit `feature_names` to override inference.

### Logger Integration

When a logger is configured, `explain()` logs **metrics only** (no plots):

```python
# Inside explain()
with self.logger.start_run(name=run_name, config={}, tags=self.tags):
    # ... compute shap_values ...

    # Log mean |SHAP| per feature as importance metrics
    mean_abs_shap = _compute_mean_abs_shap(shap_values)
    self.logger.log_metrics({
        f"shap_importance/{name}": float(val)
        for name, val in zip(feature_names, mean_abs_shap)
    })
```

#### Multi-class SHAP Value Handling

SHAP returns different shapes for binary vs multiclass:
- Binary/regression: `ndarray` of shape `(n_samples, n_features)`
- Multiclass: `list[ndarray]` of length `n_classes`, each `(n_samples, n_features)`

Normalize before aggregating:

```python
def _compute_mean_abs_shap(shap_values) -> np.ndarray:
    """Compute mean |SHAP| per feature, handling multiclass."""
    if isinstance(shap_values, list):
        # Multiclass: stack to (n_classes, n_samples, n_features)
        stacked = np.stack(shap_values)
        return np.abs(stacked).mean(axis=(0, 1))  # Mean over classes and samples
    return np.abs(shap_values).mean(axis=0)
```

#### No Automatic Plot Logging

Plots are **not** logged automatically because:
- SHAP plotting returns `Axes`, not `Figure` (inconsistent API)
- Requires matplotlib (fails in headless/minimal environments)
- Users may want different plot types or customizations

To log plots manually:

```python
result = exp.explain(X_test)

# User controls plotting and logging
import matplotlib.pyplot as plt
result.plot("beeswarm")
plt.savefig("shap_summary.png")
logger.log_artifact("shap_summary.png")
```

**Future iteration:** Investigate adding optional automatic plot logging with a unified plotting API. See [Open Questions: Unified Plotting API](#unified-plotting-api).

## What This Is NOT

### Not Feature Importance

Feature importance answers: "What features matter globally?"
- Permutation importance (model-agnostic, based on performance drop)
- Coefficient magnitude (linear models)
- Tree-based importance (impurity decrease / gain)

SHAP values answer: "Why did the model make *this* prediction?"
- Local explanations that can be aggregated globally
- Accounts for feature interactions
- Theoretically grounded (Shapley values)

**Recommendation:** Feature importance belongs in a separate `Experiment.feature_importance()` method or a diagnostics API, not in `explain()`.

### Not Partial Dependence Plots

PDPs answer: "How does changing feature X affect predictions on average?"
- Shows marginal effect of a feature
- Useful for understanding learned relationships
- sklearn has `PartialDependenceDisplay`

SHAP dependence plots show similar information but with interaction coloring and are SHAP-specific.

**Recommendation:** PDPs belong in a diagnostics API alongside learning curves, calibration plots, etc.

## Proposed API Landscape

```
Experiment.fit()           → FitResult           # Train
Experiment.evaluate()      → EvalResult          # Test metrics
Experiment.cross_validate() → CVResult           # CV metrics
Experiment.search()        → SearchResult        # Hyperparameter search
Experiment.explain()       → ExplainResult       # SHAP explanations (NEW)

# Future (separate from explain):
Experiment.diagnose()      → DiagnosticsResult   # Performance plots, calibration, etc.
Experiment.analyze()       → AnalysisResult      # Feature importance, PDPs, ICE
```

Or, if we want to keep the API minimal:

```
# Alternative: Single diagnostics entry point
Experiment.diagnose(
    X, y,
    include=["metrics", "importance", "pdp", "calibration", ...]
) → DiagnosticsResult
```

## Edge Cases

### Multi-output / Multi-class Models

Store `values` consistently as 3D array `(n_samples, n_features, n_outputs)`:
- Binary classification: `n_outputs=1` (single column)
- Multiclass: `n_outputs=n_classes`
- Regression: `n_outputs=1`

This avoids users having to handle both shapes. The `raw` shap.Explanation preserves SHAP's native format for advanced use.

### No Fitted Estimator
Raise `ValueError("Call fit() or cross_validate(refit=True) before explain()")`.

### Unsupported Estimator
Fall back to `KernelExplainer` with a warning about computational cost.

### Large X
For `KernelExplainer`, default background to `shap.kmeans(X, 100)` or similar summarization.

## Dependencies

SHAP is an optional dependency:
```toml
[project.optional-dependencies]
shap = ["shap>=0.42"]
```

Use the existing `LazyModule` pattern (see `src/sklab/_lazy.py`):

```python
# In src/sklab/_explain.py (or wherever explain logic lives)
from sklab._lazy import LazyModule

shap = LazyModule("shap", install_hint="Install with: pip install sklab[shap]")

# Usage - import happens on first attribute access
def _create_explainer(estimator, X, method):
    if method == "tree" or _is_tree_model(estimator):
        return shap.TreeExplainer(estimator)
    # ...
```

This matches how `mlflow` and `wandb` are handled in the logging adapters.

## Related: Composite Runs

To group `search()` + `explain()` under a single logged run, see `plans/run-context-api-design.md` for the `Experiment.run()` context manager proposal.

```python
# With run context (proposed)
with exp.run(name="grid-search-with-explanation") as e:
    result = e.search(GridSearchConfig(...), X, y)
    explanation = e.explain(X_test)
    # Both logged as nested runs under one parent
```

## How to Test

### 1. Explainer Selection (`_select_explainer_method`)

| Estimator | Expected Method | Structural Check |
|-----------|-----------------|------------------|
| `DecisionTreeClassifier` | `TREE` | `tree_` attribute |
| `RandomForestClassifier` | `TREE` | `estimators_[0].tree_` |
| `GradientBoostingClassifier` | `TREE` | `estimators_[0][0].tree_` |
| `LogisticRegression` | `LINEAR` | `coef_` attribute |
| `Ridge` | `LINEAR` | `coef_` attribute |
| `SVC(kernel='rbf')` | `KERNEL` | fallback |
| `MLPClassifier` | `KERNEL` | fallback (no torch module) |

External libraries (guarded by `pytest.importorskip`):

| Estimator | Expected Method |
|-----------|-----------------|
| `xgboost.XGBClassifier` | `TREE` |
| `lightgbm.LGBMClassifier` | `TREE` |
| `catboost.CatBoostClassifier` | `TREE` |

```python
@pytest.mark.parametrize("estimator,expected", [
    (DecisionTreeClassifier(), ExplainerMethod.TREE),
    (RandomForestClassifier(n_estimators=2), ExplainerMethod.TREE),
    (LogisticRegression(), ExplainerMethod.LINEAR),
    (Ridge(), ExplainerMethod.LINEAR),
    (SVC(), ExplainerMethod.KERNEL),
])
def test_select_explainer_method(estimator, expected, data):
    X, y = data
    estimator.fit(X, y)
    assert _select_explainer_method(estimator) == expected
```

### 2. Model Output Selection (`_default_model_output`)

| Estimator | Has `predict_proba`? | Expected Output |
|-----------|---------------------|-----------------|
| `LogisticRegression` | Yes | `PROBABILITY` |
| `SVC(probability=False)` | No | `RAW` |
| `SVC(probability=True)` | Yes | `PROBABILITY` |
| `Ridge` (regressor) | No | `RAW` |

```python
@pytest.mark.parametrize("estimator,expected", [
    (LogisticRegression(), ModelOutput.PROBABILITY),
    (SVC(probability=False), ModelOutput.RAW),
    (SVC(probability=True), ModelOutput.PROBABILITY),
    (Ridge(), ModelOutput.RAW),
])
def test_default_model_output(estimator, expected):
    assert _default_model_output(estimator) == expected
```

### 3. Feature Name Recovery

| Scenario | Expected Behavior |
|----------|-------------------|
| User provides `feature_names` | Use user-provided |
| Pipeline with `get_feature_names_out()` | Use recovered names |
| Pipeline without `get_feature_names_out()` | Fall back to `x0`, `x1`, ... |
| Raw estimator (no pipeline) | Fall back to generic names |

```python
def test_feature_names_user_override(data):
    X, y = data
    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X, y)
    result = exp.explain(X[:5], feature_names=["a", "b", "c", "d"])
    assert result.feature_names == ["a", "b", "c", "d"]

def test_feature_names_from_pipeline(data):
    X, y = data
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression())
    ])
    exp = Experiment(pipeline=pipeline)
    exp.fit(X, y)
    result = exp.explain(X[:5])
    # StandardScaler supports get_feature_names_out in sklearn 1.0+
    assert result.feature_names is not None
    assert len(result.feature_names) == X.shape[1]

def test_feature_names_fallback_generic(data):
    X, y = data
    # Raw estimator, no pipeline
    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X, y)
    result = exp.explain(X[:5])
    # Should fall back to x0, x1, ...
    assert result.feature_names == [f"x{i}" for i in range(X.shape[1])]
```

### 4. Multi-class SHAP Value Handling

| Case | Input Shape | Expected `values` Shape |
|------|-------------|-------------------------|
| Binary classification | `(n, f)` | `(n, f, 1)` |
| Multiclass (3 classes) | `list[3]` of `(n, f)` | `(n, f, 3)` |
| Regression | `(n, f)` | `(n, f, 1)` |

```python
def test_explain_binary_classification(binary_data):
    X, y = binary_data
    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X, y)
    result = exp.explain(X[:5])

    assert result.values.ndim == 3
    assert result.values.shape == (5, X.shape[1], 1)

def test_explain_multiclass(iris_data):
    X, y = iris_data  # 3 classes
    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X, y)
    result = exp.explain(X[:5])

    assert result.values.shape == (5, X.shape[1], 3)

def test_explain_regression(regression_data):
    X, y = regression_data
    exp = Experiment(pipeline=Ridge())
    exp.fit(X, y)
    result = exp.explain(X[:5])

    assert result.values.shape == (5, X.shape[1], 1)
```

### 5. Mean |SHAP| Aggregation

```python
def test_compute_mean_abs_shap_binary():
    shap_values = np.array([[0.1, -0.2], [0.3, -0.4]])
    result = _compute_mean_abs_shap(shap_values)
    expected = np.array([0.2, 0.3])  # mean of abs values per feature
    np.testing.assert_array_almost_equal(result, expected)

def test_compute_mean_abs_shap_multiclass():
    # 2 classes, 3 samples, 2 features
    shap_values = [
        np.array([[0.1, -0.2], [0.3, -0.4], [0.5, -0.6]]),
        np.array([[0.2, -0.3], [0.4, -0.5], [0.6, -0.7]]),
    ]
    result = _compute_mean_abs_shap(shap_values)
    # Should be (2,) - one value per feature
    assert result.shape == (2,)
```

### 6. Integration Tests (End-to-End)

```python
def test_explain_logistic_regression(data):
    X, y = data
    exp = Experiment(pipeline=LogisticRegression(), scoring="accuracy")
    exp.fit(X, y)
    result = exp.explain(X[:10])

    assert isinstance(result, ExplainResult)
    assert result.values.shape[0] == 10
    assert result.base_values is not None
    assert result.raw is not None  # shap.Explanation

def test_explain_random_forest(data):
    X, y = data
    exp = Experiment(pipeline=RandomForestClassifier(n_estimators=10, random_state=42))
    exp.fit(X, y)
    result = exp.explain(X[:10])

    assert isinstance(result, ExplainResult)

def test_explain_pipeline_with_preprocessing(data):
    X, y = data
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression())
    ])
    exp = Experiment(pipeline=pipeline)
    exp.fit(X, y)
    result = exp.explain(X[:10])

    assert isinstance(result, ExplainResult)
    assert result.feature_names is not None
    assert len(result.feature_names) == X.shape[1]

def test_explain_fallback_to_kernel_with_warning(data):
    """Unsupported estimator should fall back to KernelExplainer with warning."""
    X, y = data
    exp = Experiment(pipeline=SVC(kernel="rbf"))  # No tree_, no coef_
    exp.fit(X, y)

    with pytest.warns(UserWarning, match="KernelExplainer"):
        result = exp.explain(X[:5])

    assert isinstance(result, ExplainResult)

def test_explain_background_sampling(data):
    """Background int should sample from X."""
    X, y = data
    exp = Experiment(pipeline=SVC(kernel="rbf"))
    exp.fit(X, y)

    # Should not raise - samples 10 background points from X
    result = exp.explain(X[:20], background=10)
    assert isinstance(result, ExplainResult)
```

### 6a. SHAP Output Semantics Validation

Verify the shape and semantics of SHAP outputs are correct:

```python
def test_explain_binary_values_shape(binary_data):
    """Binary classification: values should be (n_samples, n_features, 1)."""
    X, y = binary_data
    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X, y)
    result = exp.explain(X[:5])

    assert result.values.shape == (5, X.shape[1], 1)
    assert result.base_values.shape == (5, 1) or result.base_values.shape == (1,)

def test_explain_multiclass_values_shape(iris_data):
    """Multiclass: values should be (n_samples, n_features, n_classes)."""
    X, y = iris_data
    n_classes = len(np.unique(y))
    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X, y)
    result = exp.explain(X[:5])

    assert result.values.shape == (5, X.shape[1], n_classes)
    # base_values: one per class
    assert result.base_values.shape[-1] == n_classes

def test_explain_regression_values_shape(regression_data):
    """Regression: values should be (n_samples, n_features, 1)."""
    X, y = regression_data
    exp = Experiment(pipeline=Ridge())
    exp.fit(X, y)
    result = exp.explain(X[:5])

    assert result.values.shape == (5, X.shape[1], 1)

def test_explain_values_sum_to_prediction_diff():
    """SHAP values should sum to (prediction - base_value) for linear models."""
    X, y = load_iris(return_X_y=True)
    # Use only 2 classes for binary
    mask = y < 2
    X, y = X[mask], y[mask]

    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X, y)
    result = exp.explain(X[:5], model_output="raw")

    # For each sample, sum of SHAP values ≈ prediction - base_value
    predictions = exp._fitted_estimator.decision_function(X[:5])
    for i in range(5):
        shap_sum = result.values[i, :, 0].sum()
        expected_diff = predictions[i] - result.base_values[i, 0]
        np.testing.assert_almost_equal(shap_sum, expected_diff, decimal=4)

def test_explain_model_output_probability_semantics(data):
    """With model_output=probability, values should explain probabilities."""
    X, y = data
    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X, y)

    result_prob = exp.explain(X[:5], model_output="probability")
    result_raw = exp.explain(X[:5], model_output="raw")

    # Probability explanations should have different scale than raw
    assert not np.allclose(result_prob.values, result_raw.values)
    # Base values should be in [0, 1] for probability
    assert np.all(result_prob.base_values >= 0)
    assert np.all(result_prob.base_values <= 1)
```

### 6b. Feature Name Alignment

```python
def test_feature_names_match_transformed_columns():
    """Feature names should match the actual transformed feature count."""
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer

    # Create data with categorical column
    X = np.array([[1, "a"], [2, "b"], [3, "a"], [4, "c"]], dtype=object)
    y = np.array([0, 1, 0, 1])

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), [0]),
        ("cat", OneHotEncoder(), [1]),
    ])
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression())
    ])

    exp = Experiment(pipeline=pipeline)
    exp.fit(X, y)
    result = exp.explain(X[:2])

    # Should have 1 (scaled) + 3 (one-hot) = 4 features
    assert len(result.feature_names) == result.values.shape[1]
    assert result.values.shape[1] == 4

def test_feature_names_order_matches_values():
    """Feature names order should correspond to SHAP values columns."""
    X, y = load_iris(return_X_y=True)
    feature_names = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"]

    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X, y)
    result = exp.explain(X[:5], feature_names=feature_names)

    assert result.feature_names == feature_names
    assert result.values.shape[1] == len(feature_names)
```

### 7. Logger Integration

```python
def test_explain_logs_metrics(data, in_memory_logger):
    X, y = data
    exp = Experiment(
        pipeline=LogisticRegression(),
        logger=in_memory_logger,
    )
    exp.fit(X, y)
    exp.explain(X[:10])

    # Check that shap_importance metrics were logged
    logged_metrics = in_memory_logger.metrics
    assert any("shap_importance" in key for key in logged_metrics)

def test_explain_does_not_log_artifacts(data, in_memory_logger):
    X, y = data
    exp = Experiment(pipeline=LogisticRegression(), logger=in_memory_logger)
    exp.fit(X, y)
    exp.explain(X[:10])

    # No plot artifacts should be logged (plots are manual)
    # Only model artifact from fit() should exist
    plot_artifacts = [a for a in in_memory_logger.artifacts if "shap" in a]
    assert len(plot_artifacts) == 0

def test_explain_logs_correct_metric_count(data, in_memory_logger):
    """Should log one shap_importance metric per feature."""
    X, y = data
    n_features = X.shape[1]
    exp = Experiment(pipeline=LogisticRegression(), logger=in_memory_logger)
    exp.fit(X, y)
    exp.explain(X[:10])

    importance_metrics = [k for k in in_memory_logger.metrics if "shap_importance" in k]
    assert len(importance_metrics) == n_features

def test_explain_works_without_logger(data):
    """explain() should work fine with NoOpLogger (default)."""
    X, y = data
    exp = Experiment(pipeline=LogisticRegression())  # No logger
    exp.fit(X, y)
    result = exp.explain(X[:10])

    assert isinstance(result, ExplainResult)
```

### 7a. Logger Edge Environments

```python
def test_explain_in_headless_environment(data, in_memory_logger, monkeypatch):
    """explain() should not fail in headless environments (no DISPLAY)."""
    monkeypatch.delenv("DISPLAY", raising=False)

    X, y = data
    exp = Experiment(pipeline=LogisticRegression(), logger=in_memory_logger)
    exp.fit(X, y)

    # Should compute and log without plotting
    result = exp.explain(X[:10])
    assert isinstance(result, ExplainResult)
    assert any("shap_importance" in k for k in in_memory_logger.metrics)

def test_explain_logging_without_matplotlib(data, in_memory_logger, monkeypatch):
    """explain() logging should work even if matplotlib is not available."""
    # Hide matplotlib
    import sys
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

    X, y = data
    exp = Experiment(pipeline=LogisticRegression(), logger=in_memory_logger)
    exp.fit(X, y)

    # Should still compute and log metrics (no plots)
    result = exp.explain(X[:10])
    assert isinstance(result, ExplainResult)
```

### 8. Edge Cases

```python
def test_explain_unfitted_raises(data):
    X, y = data
    exp = Experiment(pipeline=LogisticRegression())
    with pytest.raises(ValueError, match="fit()"):
        exp.explain(X[:5])

def test_explain_after_cross_validate_with_refit(data):
    X, y = data
    exp = Experiment(pipeline=LogisticRegression(), scoring="accuracy")
    exp.cross_validate(X, y, cv=3, refit=True)
    result = exp.explain(X[:10])  # Should work
    assert isinstance(result, ExplainResult)

def test_explain_after_cross_validate_without_refit_raises(data):
    X, y = data
    exp = Experiment(pipeline=LogisticRegression(), scoring="accuracy")
    exp.cross_validate(X, y, cv=3, refit=False)
    with pytest.raises(ValueError, match="fit()"):
        exp.explain(X[:10])

def test_explain_after_search(data):
    X, y = data
    exp = Experiment(pipeline=LogisticRegression(), scoring="accuracy")
    exp.search(GridSearchConfig({"C": [0.1, 1.0]}), X, y, cv=3)
    result = exp.explain(X[:10])  # Should use best_estimator_
    assert isinstance(result, ExplainResult)
```

### 8a. Input Variations

```python
def test_explain_single_sample(data):
    """Explaining a single sample should work."""
    X, y = data
    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X, y)
    result = exp.explain(X[:1])  # Single row

    assert result.values.shape[0] == 1
    assert isinstance(result, ExplainResult)

def test_explain_pandas_dataframe(data):
    """Should accept pandas DataFrame input."""
    import pandas as pd

    X, y = data
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X_df, y)
    result = exp.explain(X_df.iloc[:10])

    assert isinstance(result, ExplainResult)
    # Feature names should come from DataFrame columns
    assert result.feature_names == list(X_df.columns)

def test_explain_polars_dataframe(data):
    """Should accept polars DataFrame input."""
    pytest.importorskip("polars")
    import polars as pl

    X, y = data
    X_pl = pl.DataFrame({f"f{i}": X[:, i] for i in range(X.shape[1])})

    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X_pl, y)
    result = exp.explain(X_pl.head(10))

    assert isinstance(result, ExplainResult)

def test_explain_sparse_matrix(data):
    """Should handle sparse matrix input."""
    from scipy import sparse

    X, y = data
    X_sparse = sparse.csr_matrix(X)

    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X_sparse, y)
    result = exp.explain(X_sparse[:10])

    assert isinstance(result, ExplainResult)
    assert result.values.shape[0] == 10

def test_explain_with_nan_in_background(data):
    """Background data with NaN should be handled or raise clear error."""
    X, y = data
    X_with_nan = X.copy()
    X_with_nan[0, 0] = np.nan

    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X, y)  # Fit on clean data

    # Explaining data with NaN - behavior depends on SHAP
    # Should either handle gracefully or raise clear error
    with pytest.raises((ValueError, Exception)):
        exp.explain(X_with_nan[:5])

def test_explain_high_dimensional_data():
    """Should work with high-dimensional data (more features than samples)."""
    np.random.seed(42)
    X = np.random.randn(50, 200)  # 50 samples, 200 features
    y = (X[:, 0] > 0).astype(int)

    exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
    exp.fit(X, y)
    result = exp.explain(X[:5])

    assert result.values.shape == (5, 200, 1)
```

### 9. StrEnum Parameter Acceptance

```python
@pytest.mark.parametrize("method", [
    ExplainerMethod.LINEAR,
    "linear",
])
def test_explain_accepts_method_string_and_enum(data, method):
    X, y = data
    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X, y)
    result = exp.explain(X[:5], method=method)
    assert isinstance(result, ExplainResult)

@pytest.mark.parametrize("model_output", [
    ModelOutput.PROBABILITY,
    "probability",
])
def test_explain_accepts_model_output_string_and_enum(data, model_output):
    X, y = data
    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X, y)
    result = exp.explain(X[:5], model_output=model_output)
    assert isinstance(result, ExplainResult)
```

### 10. Plot Passthrough (Smoke Test)

```python
def test_plot_passthrough_does_not_raise(data):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    X, y = data
    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X, y)
    result = exp.explain(X[:5])

    # Should not raise
    result.plot("bar")
    plt.close()

@pytest.mark.parametrize("kind", ["summary", "bar", "beeswarm"])
def test_plot_kinds(data, kind):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    X, y = data
    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X, y)
    result = exp.explain(X[:5])

    result.plot(kind)
    plt.close()
```

### 11. Unhappy Paths

| Scenario | Expected Behavior |
|----------|-------------------|
| Invalid `method` string | `ValueError` with valid options |
| Invalid `model_output` string | `ValueError` with valid options |
| `model_output="probability"` on regressor | `ValueError` (no `predict_proba`) |
| `model_output="log_odds"` on regressor | `ValueError` |
| `feature_names` length mismatch | `ValueError` with expected vs actual |
| Empty `X` (0 samples) | `ValueError` |
| `X` with wrong number of features | `ValueError` |
| SHAP not installed | `ModuleNotFoundError` with install hint |
| `background` int larger than X | `ValueError` |
| Invalid `plot()` kind | `ValueError` or `AttributeError` |
| `plot()` without matplotlib | `ImportError` with hint |

```python
def test_explain_invalid_method_raises(data):
    X, y = data
    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X, y)
    with pytest.raises(ValueError, match="method"):
        exp.explain(X[:5], method="invalid")

def test_explain_invalid_model_output_raises(data):
    X, y = data
    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X, y)
    with pytest.raises(ValueError, match="model_output"):
        exp.explain(X[:5], model_output="invalid")

def test_explain_probability_on_regressor_raises(regression_data):
    X, y = regression_data
    exp = Experiment(pipeline=Ridge())
    exp.fit(X, y)
    with pytest.raises(ValueError, match="probability.*regressor"):
        exp.explain(X[:5], model_output="probability")

def test_explain_log_odds_on_regressor_raises(regression_data):
    X, y = regression_data
    exp = Experiment(pipeline=Ridge())
    exp.fit(X, y)
    with pytest.raises(ValueError, match="log_odds.*regressor"):
        exp.explain(X[:5], model_output="log_odds")

def test_explain_feature_names_length_mismatch_raises(data):
    X, y = data
    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X, y)
    with pytest.raises(ValueError, match="feature_names.*expected.*got"):
        exp.explain(X[:5], feature_names=["a", "b"])  # Wrong length

def test_explain_empty_x_raises(data):
    X, y = data
    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X, y)
    with pytest.raises(ValueError, match="empty"):
        exp.explain(X[:0])  # Zero samples

def test_explain_wrong_feature_count_raises(data):
    X, y = data
    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X, y)
    with pytest.raises(ValueError):
        exp.explain(X[:5, :2])  # Wrong number of features

def test_explain_shap_not_installed(data, monkeypatch):
    # Simulate SHAP not being installed
    monkeypatch.setattr("sklab._explain.shap._module", None)
    monkeypatch.setattr("sklab._explain.shap._name", "shap")
    X, y = data
    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X, y)
    with pytest.raises(ModuleNotFoundError, match="pip install"):
        exp.explain(X[:5])

def test_explain_background_larger_than_x_raises(data):
    X, y = data
    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X, y)
    with pytest.raises(ValueError, match="background.*larger"):
        exp.explain(X[:5], background=100)  # Can't sample 100 from 5

def test_plot_invalid_kind_raises(data):
    X, y = data
    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X, y)
    result = exp.explain(X[:5])
    with pytest.raises((ValueError, AttributeError)):
        result.plot("invalid_plot")

def test_plot_without_matplotlib_raises(data, monkeypatch):
    """plot() should raise ImportError with hint when matplotlib unavailable."""
    import sys
    # Hide matplotlib
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

    X, y = data
    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X, y)
    result = exp.explain(X[:5])
    with pytest.raises(ImportError, match="matplotlib"):
        result.plot("bar")
```

### 12. Regression Tests (Correctness Verification)

Compare SHAP outputs against known values to catch regressions in sign/magnitude:

```python
def test_explain_linear_model_known_output():
    """
    For a simple linear model, SHAP values should match coefficients direction.
    Positive coefficient + positive feature value = positive SHAP contribution.
    """
    # Simple 2-feature dataset
    np.random.seed(42)
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]] * 25, dtype=float)
    y = (X[:, 0] > 0.5).astype(int)  # Only feature 0 matters

    exp = Experiment(pipeline=LogisticRegression(C=1000, max_iter=1000))
    exp.fit(X, y)
    result = exp.explain(X[:4], model_output="raw")

    # Feature 0 should have much larger |SHAP| than feature 1
    mean_abs_f0 = np.abs(result.values[:, 0, 0]).mean()
    mean_abs_f1 = np.abs(result.values[:, 1, 0]).mean()
    assert mean_abs_f0 > mean_abs_f1 * 5  # Feature 0 dominates

def test_explain_tree_feature_importance_correlation():
    """
    SHAP importance (mean |SHAP|) should correlate with tree feature importance.
    """
    X, y = load_iris(return_X_y=True)
    # Binary for simplicity
    mask = y < 2
    X, y = X[mask], y[mask]

    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    exp = Experiment(pipeline=rf)
    exp.fit(X, y)
    result = exp.explain(X)

    # Compute SHAP importance
    shap_importance = np.abs(result.values[:, :, 0]).mean(axis=0)
    shap_rank = np.argsort(-shap_importance)

    # Compute sklearn feature importance
    sklearn_importance = rf.feature_importances_
    sklearn_rank = np.argsort(-sklearn_importance)

    # Top 2 features should overlap (not necessarily same order)
    assert len(set(shap_rank[:2]) & set(sklearn_rank[:2])) >= 1

def test_explain_deterministic_with_seed():
    """Same model + data + seed should produce identical SHAP values."""
    X, y = load_breast_cancer(return_X_y=True)

    exp = Experiment(pipeline=LogisticRegression(random_state=42))
    exp.fit(X, y)

    result1 = exp.explain(X[:10])
    result2 = exp.explain(X[:10])

    np.testing.assert_array_almost_equal(result1.values, result2.values)
    np.testing.assert_array_almost_equal(result1.base_values, result2.base_values)

def test_explain_symmetry_for_symmetric_data():
    """
    For symmetric data, SHAP values should be symmetric.
    If X1 and X2 are swapped and features are swapped, SHAP should swap too.
    """
    # Symmetric dataset: swapping features swaps prediction
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=float)
    y = np.array([1, 0, 1, 0])  # XOR-ish

    exp = Experiment(pipeline=LogisticRegression())
    exp.fit(X, y)

    # Explain [1, 0] and [0, 1]
    result = exp.explain(X[:2], model_output="raw")

    # SHAP for [1, 0] feature 0 should ≈ SHAP for [0, 1] feature 1
    np.testing.assert_almost_equal(
        result.values[0, 0, 0],
        result.values[1, 1, 0],
        decimal=2
    )
```

### Test Fixtures

```python
@pytest.fixture
def data():
    """Binary classification dataset."""
    return load_breast_cancer(return_X_y=True)

@pytest.fixture
def binary_data():
    """Binary classification dataset (alias for clarity)."""
    return load_breast_cancer(return_X_y=True)

@pytest.fixture
def iris_data():
    """Multiclass classification dataset."""
    return load_iris(return_X_y=True)

@pytest.fixture
def regression_data():
    """Regression dataset."""
    return load_diabetes(return_X_y=True)

@pytest.fixture
def in_memory_logger():
    """Logger that captures all calls for assertions."""
    return InMemoryLogger()
```

## Future Considerations

1. **Caching** — SHAP computation is expensive. Consider optional result caching.
2. **Sampling** — For large datasets, option to explain a sample.
3. **Batch explanations** — Stream explanations for very large X.
4. **Interaction values** — `shap_interaction_values` for tree models.
5. **Text/image data** — Would require different maskers and explainers.

## References

- [SHAP documentation](https://shap.readthedocs.io/)
- [Lundberg & Lee, 2017 - "A Unified Approach to Interpreting Model Predictions"](https://arxiv.org/abs/1705.07874)
- [sklearn inspection module](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.inspection) (permutation importance, PDPs)

---

## Open Questions

### Unified Plotting API

Currently, `result.plot()` is a thin passthrough to `shap.plots`. However, SHAP's plotting API has inconsistencies:
- Some functions return `Axes`, others return `Figure`
- Different functions have different parameter names
- Behavior varies between plot types

**Question:** Should sklab provide a unified plotting API that normalizes these inconsistencies?

**Option A: Thin passthrough (current design)**
- Pro: No maintenance burden, users get full SHAP flexibility
- Con: Users deal with SHAP's inconsistencies

**Option B: Unified wrapper**
- Pro: Consistent return types, consistent parameters, easier logging
- Con: Maintenance burden, may lag behind SHAP updates, limits advanced usage
- Would require matplotlib as a dependency (with shap extra)

**Decision:** Deferred. Start with thin passthrough, evaluate user feedback.

---

## Summary

| Question | Answer |
|----------|--------|
| What does `explain()` do? | Computes SHAP values for the fitted estimator |
| What does it return? | `ExplainResult` with values, base_values, data, feature_names, and raw `shap.Explanation` |
| Does it wrap SHAP plotting? | Thin passthrough only; exposes `raw` for full control |
| How is the explainer selected? | Structural checks (`tree_`, `coef_`, module name), returns `ExplainerMethod` StrEnum |
| How is model output selected? | `is_classifier()` + `predict_proba` check, returns `ModelOutput` StrEnum |
| What about feature names? | Best-effort via `get_feature_names_out()`, user can override |
| What about multiclass? | Values normalized to 3D array `(n_samples, n_features, n_outputs)` |
| Does it log plots? | No (future iteration). Users log plots manually. |
| What about feature importance? | Separate concern → separate method or diagnostics API |
| What about PDPs? | Separate concern → diagnostics API |
| Is SHAP required? | Optional dependency via `LazyModule` |
