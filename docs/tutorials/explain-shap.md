# Model Explanations with SHAP

**What you'll learn:**

- How to compute SHAP values for your fitted models
- How to interpret the `ExplainResult` object
- How to visualize feature importance with built-in plots

**Prerequisites:** Familiarity with the [Experiment class](experiment.md) and basic
sklearn pipelines. Install the SHAP extra: `pip install sklab[shap]`.

## Why explain your models?

Machine learning models often act as black boxes—they make predictions, but
understanding *why* they make those predictions can be challenging. SHAP
(SHapley Additive exPlanations) provides a principled way to attribute each
feature's contribution to a prediction.

!!! note "Concept: SHAP Values"

    SHAP values come from game theory. They measure how much each feature
    contributes to pushing the prediction away from a baseline (the average
    prediction). Positive SHAP values push the prediction higher; negative
    values push it lower.

    **Why it matters:** SHAP values let you debug models, explain predictions
    to stakeholders, and identify which features drive decisions.

---

## Basic usage

After fitting an experiment, call `explain()` to compute SHAP values:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklab import Experiment

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build and fit experiment
pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000)),
])

experiment = Experiment(pipeline=pipeline, scoring="accuracy")
experiment.fit(X_train, y_train)

# Explain predictions on test samples
result = experiment.explain(X_test[:10])

print(f"SHAP values shape: {result.values.shape}")
print(f"Number of features: {len(result.feature_names)}")
```

**What just happened:**

1. The experiment fitted the pipeline on training data
2. `explain()` computed SHAP values for 10 test samples
3. Each sample gets a SHAP value per feature, showing how much that feature
   contributed to the prediction

---

## Understanding ExplainResult

The `explain()` method returns an `ExplainResult` with these attributes:

| Attribute | Description |
|-----------|-------------|
| `values` | SHAP values array: (n_samples, n_features, n_outputs) |
| `base_values` | Expected model output (the baseline prediction) |
| `data` | The transformed input data that was explained |
| `feature_names` | List of feature names (inferred or user-provided) |
| `raw` | The underlying `shap.Explanation` for advanced use |

```{.python continuation}
# Inspect the result
print(f"Base value (expected output): {result.base_values[0]:.4f}")
print(f"First sample, first 5 features:")
for i, (name, val) in enumerate(zip(result.feature_names[:5], result.values[0, :5, 0])):
    print(f"  {name}: {val:+.4f}")
```

!!! note "Concept: Base Values"

    The base value is the model's average prediction across the background
    dataset. SHAP values show how each feature moves the prediction away
    from this baseline. The sum of all SHAP values plus the base value
    equals the model's prediction.

---

## Visualizing explanations

`ExplainResult` provides a `plot()` method that passes through to SHAP's
visualization functions:

```{.python continuation}
# Bar plot - global feature importance
result.plot("bar")
```

Available plot types:

| Plot | Description | Best for |
|------|-------------|----------|
| `"bar"` | Mean absolute SHAP value per feature | Global importance |
| `"beeswarm"` | SHAP values colored by feature value | Feature-value relationships |
| `"waterfall"` | Single prediction breakdown | Explaining one prediction |
| `"summary"` | Dot plot of SHAP values | Overview of all samples |

```{.python continuation}
import matplotlib.pyplot as plt

# Beeswarm - see how feature values affect predictions
result.plot("beeswarm")
plt.close("all")
```

---

## Working with pipelines

sklab automatically handles sklearn pipelines. When you have preprocessing
steps, `explain()` extracts the final estimator and transforms the data:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklab import Experiment

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline with preprocessing
pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=50, random_state=42)),
])

experiment = Experiment(pipeline=pipeline, scoring="accuracy")
experiment.fit(X_train, y_train)
result = experiment.explain(X_test[:10])

# Feature names are recovered automatically
print(f"Features explained: {len(result.feature_names)}")
print(f"First 5: {result.feature_names[:5]}")
```

**What just happened:**

1. sklab extracted the `RandomForestClassifier` from the pipeline
2. The scaler transformed X_test before computing SHAP values
3. Feature names fall back to generic names (x0, x1, ...) when not available

---

## Automatic explainer selection

sklab automatically selects the best SHAP explainer for your model:

| Model type | Explainer | Why |
|------------|-----------|-----|
| Tree-based (RandomForest, XGBoost, etc.) | TreeExplainer | Fast, exact |
| Linear (LogisticRegression, Ridge, etc.) | LinearExplainer | Exact, fast |
| Neural networks (Keras, PyTorch) | DeepExplainer | Gradient-based |
| Everything else | KernelExplainer | Model-agnostic (slower) |

You can override this with the `method` parameter:

```{.python continuation}
from sklab import ExplainerModel

# Force a specific explainer
result = experiment.explain(X_test[:5], method=ExplainerModel.TREE)

# String values also work
result = experiment.explain(X_test[:5], method="tree")
```

---

## Providing custom feature names

If generic names aren't sufficient, provide your own:

```{.python continuation}
from sklearn.datasets import load_breast_cancer

# Get the real feature names
data = load_breast_cancer()
feature_names = list(data.feature_names)

result = experiment.explain(X_test[:5], feature_names=feature_names)
print(f"First feature: {result.feature_names[0]}")
```

---

## Best practices

1. **Explain a sample, not the whole dataset.** SHAP computation can be slow
   for large datasets. Start with 10-100 samples.

2. **Use tree models when possible.** TreeExplainer is exact and fast. Linear
   models are also fast. KernelExplainer (the fallback) is slow.

3. **Provide meaningful feature names.** Generic names (x0, x1) make plots
   hard to interpret. Pass DataFrame columns or explicit names.

4. **Check the base value.** If the base value doesn't match your intuition
   about average model output, something may be wrong.

5. **Compare explanations across models.** SHAP values help you understand
   why different models make different predictions.

---

## Next steps

- [Advanced SHAP Usage](explain-shap-advanced.md) — Background data, model output modes, and more
- [Hyperparameter Search](sklearn-search.md) — Find better model configurations
- [Logger Adapters](logger-adapters.md) — Track SHAP importance metrics automatically
