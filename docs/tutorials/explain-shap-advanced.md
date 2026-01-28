# Advanced SHAP Usage

**What you'll learn:**

- How to control model output interpretation (probability, raw, log-odds)
- How to use background data for better explanations
- How to explain models after cross-validation and search
- How SHAP metrics are logged automatically

**Prerequisites:** Read [Model Explanations with SHAP](explain-shap.md) first.

---

## Controlling model output

By default, sklab explains the most interpretable model output:
- **Classifiers with `predict_proba`**: Explains probability output
- **Regressors and other models**: Explains raw output

You can override this with the `model_output` parameter:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklab import Experiment, ExplainerOutput

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000)),
])

experiment = Experiment(pipeline=pipeline, scoring="accuracy")
experiment.fit(X_train, y_train)

# Explain probability output (default for classifiers)
result_prob = experiment.explain(X_test[:5], model_output=ExplainerOutput.PROBABILITY)
print(f"Probability output - base value: {result_prob.base_values[0]:.4f}")

# Explain raw decision function
result_raw = experiment.explain(X_test[:5], model_output=ExplainerOutput.RAW)
print(f"Raw output - base value: {result_raw.base_values[0]:.4f}")
```

### Available model outputs

| Value | Description | Use case |
|-------|-------------|----------|
| `ExplainerOutput.AUTO` | Auto-select based on model type | Default behavior |
| `ExplainerOutput.PROBABILITY` | Explain `predict_proba` output | Intuitive probability scale |
| `ExplainerOutput.RAW` | Explain raw output (`decision_function` or `predict`) | Compare to coefficients |
| `ExplainerOutput.LOG_ODDS` | Explain log-odds (logit of probability) | Additive on log scale |

!!! note "Concept: Log-Odds"

    Log-odds are the natural logarithm of odds: `log(p / (1-p))`. They're
    useful because SHAP values become additive—the sum of SHAP values plus
    the base value equals the log-odds prediction.

    **When to use:** Log-odds make sense when comparing SHAP values to
    logistic regression coefficients, which operate on the log-odds scale.

```{.python continuation}
# Log-odds for direct coefficient comparison
result_logodds = experiment.explain(X_test[:5], model_output=ExplainerOutput.LOG_ODDS)
print(f"Log-odds output - base value: {result_logodds.base_values[0]:.4f}")
```

---

## Background data

Some SHAP explainers (KernelExplainer, LinearExplainer) need background data
to estimate the expected value. By default, sklab uses all of X as background,
but you can control this:

```{.python continuation}
# Sample 50 random points as background (faster for large datasets)
result = experiment.explain(X_test[:10], background=50)

# Use specific background data
result = experiment.explain(X_test[:10], background=X_train[:100])
```

!!! note "Concept: Background Data"

    Background data represents "typical" inputs. SHAP compares the actual
    prediction to what the model would predict for "average" features.
    Using too little background can make explanations unstable; too much
    slows computation.

    **Rule of thumb:** 100-500 samples is usually sufficient. For large
    datasets, sample randomly rather than using everything.

---

## Explaining after cross-validation

After `cross_validate()` with `refit=True`, you can explain the refitted model:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklab import Experiment

X, y = load_breast_cancer(return_X_y=True)

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000)),
])

experiment = Experiment(pipeline=pipeline, scoring="accuracy")

# Cross-validate with refit=True (default)
cv_result = experiment.cross_validate(X, y, cv=5)
print(f"CV accuracy: {cv_result.metrics['cv/accuracy_mean']:.4f}")

# Now explain the refitted model
result = experiment.explain(X[:10])
print(f"SHAP values computed for {result.values.shape[0]} samples")
```

If `refit=False`, calling `explain()` raises an error:

```{.python continuation}
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import pytest

# Create a fresh experiment
pipeline2 = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000)),
])
exp_no_refit = Experiment(pipeline=pipeline2, scoring="accuracy")

# Cross-validate without refitting
exp_no_refit.cross_validate(X, y, cv=3, refit=False)

# This raises ValueError - no fitted estimator
with pytest.raises(ValueError, match="fit"):
    exp_no_refit.explain(X[:5])
```

---

## Explaining after hyperparameter search

After `search()`, the best estimator is automatically available for explanation:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklab import Experiment
from sklab.search import GridSearchConfig

X, y = load_breast_cancer(return_X_y=True)

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000)),
])

experiment = Experiment(pipeline=pipeline, scoring="accuracy")

# Search finds the best hyperparameters
search_result = experiment.search(
    GridSearchConfig(param_grid={"model__C": [0.1, 1.0, 10.0]}),
    X, y,
    cv=3,
)
print(f"Best C: {search_result.best_params['model__C']}")

# Explain the best model
result = experiment.explain(X[:10])
print(f"Explaining model with C={search_result.best_params['model__C']}")
```

---

## Logger integration

When you configure a logger, `explain()` automatically logs SHAP importance
metrics:

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from sklab import Experiment
from sklab.logging import NoOpLogger


class MetricCapturingLogger(NoOpLogger):
    """Logger that captures metrics for demonstration."""

    def __init__(self):
        self.captured_metrics = {}

    def start_run(self, name=None, config=None, tags=None, nested=False):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def log_metrics(self, metrics, step=None):
        self.captured_metrics.update(metrics)


X, y = load_iris(return_X_y=True)

logger = MetricCapturingLogger()
experiment = Experiment(
    pipeline=LogisticRegression(max_iter=1000),
    logger=logger,
)
experiment.fit(X, y)
experiment.explain(X[:10])

# Check what was logged
shap_metrics = {k: v for k, v in logger.captured_metrics.items() if "shap" in k}
print(f"Logged {len(shap_metrics)} SHAP importance metrics")
print(f"First 3: {list(shap_metrics.items())[:3]}")
```

**What gets logged:**

- `shap_importance/{feature_name}`: Mean |SHAP| value for each feature
- These metrics appear in your MLflow/W&B dashboard for tracking

---

## Handling multiclass classification

For multiclass problems, SHAP values have shape `(n_samples, n_features, n_classes)`:

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from sklab import Experiment

X, y = load_iris(return_X_y=True)  # 3 classes

experiment = Experiment(pipeline=LogisticRegression(max_iter=1000))
experiment.fit(X, y)
result = experiment.explain(X[:5])

print(f"Shape: {result.values.shape}")
print(f"  - {result.values.shape[0]} samples")
print(f"  - {result.values.shape[1]} features")
print(f"  - {result.values.shape[2]} classes")
```

The third dimension contains SHAP values for each class. To focus on a
specific class, slice the array:

```{.python continuation}
# SHAP values for class 0
class_0_shap = result.values[:, :, 0]
print(f"Class 0 SHAP shape: {class_0_shap.shape}")
```

---

## Using the raw shap.Explanation

For advanced use cases, access the underlying `shap.Explanation` object:

```{.python continuation}
# Access raw shap explanation
raw_explanation = result.raw

# Use SHAP's native plotting
import shap

shap.plots.bar(raw_explanation)
```

This gives you full access to SHAP's API for specialized visualizations or
analysis.

---

## Performance tips

1. **Sample your data.** Explaining 10,000 samples is slow. Start with 100.

2. **Use tree models.** TreeExplainer runs in milliseconds; KernelExplainer
   can take minutes per sample.

3. **Reduce background size.** For KernelExplainer, `background=100` is
   usually sufficient.

4. **Cache explanations.** If you need to replot, save the `ExplainResult`
   rather than recomputing.

5. **Use `shap.sample()` for background.** For very large datasets:
   ```text
   import shap
   background = shap.sample(X_train, 100)
   result = experiment.explain(X_test[:10], background=background)
   ```

---

## Error handling

Common errors and how to fix them:

| Error | Cause | Fix |
|-------|-------|-----|
| `ValueError: fit()` | No fitted estimator | Call `fit()` or `cross_validate(refit=True)` first |
| `ValueError: regressor` | `model_output='probability'` on regressor | Use `ExplainerOutput.RAW` for regressors |
| `ValueError: feature_names` | Wrong number of feature names | Match feature count after preprocessing |
| `ValueError: background` | Background size > X size | Use smaller background or `None` |

---

## Next steps

- [Logger Adapters](logger-adapters.md) — Track experiments with MLflow or W&B
- [Hyperparameter Search](sklearn-search.md) — Find better configurations
- [Optuna Integration](optuna-search.md) — Bayesian optimization
