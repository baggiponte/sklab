# Regression Workflow: Diabetes Progression

**What you'll learn:**

- How to structure a regression experiment with sklab
- Why cross-validation gives more reliable estimates than a single holdout split
- How to interpret MAE and RMSE metrics
- The importance of scaling features for linear models

**Prerequisites:** Basic Python and sklearn familiarity. If you're new to pipelines,
read [Why Pipelines Matter](why-pipelines.md) first.

## The problem: predicting disease progression

Predicting continuous values—prices, temperatures, sales figures—is fundamentally
different from classification. Instead of "which category?", we ask "how much?".

This difference affects everything: which metrics make sense, how to interpret
errors, and what pitfalls to avoid. A classifier that's wrong is just wrong,
but a regressor can be wrong by a little (off by 5 points) or a lot (off by 200 points).

We'll use the scikit-learn diabetes dataset to predict a quantitative disease
progression score. This is a classic regression task: given features like age,
BMI, blood pressure, and blood serum measurements, estimate a continuous target.

---

## Step 1: Load the data

```python
import polars as pl
from sklearn.datasets import load_diabetes

# Load the dataset (bundled with sklearn, no download needed)
diabetes = load_diabetes()

# Keep data in Polars for exploration, convert to NumPy for sklearn
diabetes_df = pl.DataFrame(diabetes.data, schema=diabetes.feature_names)
diabetes_df = diabetes_df.with_columns(pl.Series("target", diabetes.target))

print(diabetes_df.head())
print(f"\nSamples: {diabetes_df.shape[0]}, Features: {len(diabetes.feature_names)}")
print(f"Target range: {diabetes_df['target'].min():.1f} - {diabetes_df['target'].max():.1f}")

X = diabetes_df.select(diabetes.feature_names).to_numpy()
y = diabetes_df["target"].to_numpy()
```

> **Concept: Regression Targets**
>
> Unlike classification (discrete labels like "spam" or "not spam"), regression
> predicts continuous values. The diabetes target is a disease progression score
> in arbitrary units—so a prediction of 150 means a higher expected progression
> than a prediction of 100.
>
> **Why it matters:** Regression errors are distances, not just "right or wrong."
> A prediction of 145 when the true value is 150 is much better than predicting 80.

---

## Step 2: Build the pipeline

```{.python continuation}
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", Ridge(alpha=1.0)),
])
```

**What this does:**

- `StandardScaler()` centers features to mean=0 and std=1
- `Ridge` is linear regression with L2 regularization (prevents overfitting)
- The `alpha` parameter controls regularization strength

> **Concept: Why Scale for Linear Models?**
>
> Linear models are sensitive to feature scales. If "age" ranges from
> 18-80 while a blood serum feature spans 0-300, the model will weight them unevenly
> based on magnitude, not importance.
>
> **Why it matters:** Without scaling, features with larger ranges dominate
> the model. Scaling puts all features on equal footing, letting the model
> learn which features actually matter.

---

## Step 3: Set up the experiment

```{.python continuation}
from sklab.experiment import Experiment

experiment = Experiment(
    pipeline=pipeline,
    scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error"],
    name="diabetes-progression",
)
```

> **Concept: Regression Metrics**
>
> - **MAE (Mean Absolute Error):** Average of |predicted - actual|. Intuitive:
>   "on average, predictions are off by X dollars."
> - **RMSE (Root Mean Squared Error):** Square root of average squared errors.
>   Penalizes large errors more heavily than MAE.
>
> **Why it matters:** MAE treats all errors equally; RMSE punishes big mistakes
> more. If being off by $100,000 is much worse than being off by $10,000 ten times,
> optimize RMSE. If all errors matter equally, use MAE.
>
> Note: sklearn's scorers are *negated* by convention (higher is better), so we
> flip the sign when interpreting results.

---

## Step 4: Cross-validate

A single train/test split is noisy—you might get lucky or unlucky with which
samples end up in the holdout set. Cross-validation averages over multiple splits
for a more robust estimate.

```{.python continuation}
from sklearn.model_selection import KFold

cv = KFold(n_splits=5, shuffle=True, random_state=42)
result = experiment.cross_validate(X, y, cv=cv, run_name="diabetes-cv")

# Flip signs for readability (sklearn uses negative scores)
mae = -result.metrics["cv/neg_mean_absolute_error_mean"]
rmse = -result.metrics["cv/neg_root_mean_squared_error_mean"]

print(f"MAE:  {mae:.1f} (average error)")
print(f"RMSE: {rmse:.1f} (penalizes large errors)")
```

**What just happened:**

1. Split the data into 5 folds
2. For each fold: trained on 4 folds, evaluated on the held-out fold
3. Computed MAE and RMSE on each fold's predictions
4. Averaged the metrics across all 5 folds

> **Concept: Cross-Validation for Regression**
>
> Unlike classification, regression doesn't need stratified splits (there are no
> classes to balance). Standard `KFold` works well. However, if your target has
> outliers or is heavily skewed, consider stratified regression splits.
>
> **Why it matters:** A single holdout split might accidentally put all the
> expensive houses in the test set. Cross-validation averages over many splits,
> giving you a more reliable performance estimate.

---

## Interpreting the results

The metrics tell you how well your model generalizes:

- **MAE of ~50** means predictions are typically off by about 50 units
- **RMSE higher than MAE** indicates some predictions have large errors

For the diabetes dataset (targets ~25–350), an MAE of 50 means
~15–20% average error. Whether that's acceptable depends on your use case.

### What to try next

If results aren't good enough:

1. **Try a more complex model:** Gradient boosting often outperforms linear models
2. **Engineer features:** Add interactions or non-linear transformations
3. **Tune hyperparameters:** Search over `alpha` values with `experiment.search()`

---

## Complete example

Here's the full workflow in one block:

```python
import polars as pl
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklab.experiment import Experiment

# 1. Load data
diabetes = load_diabetes()
diabetes_df = pl.DataFrame(diabetes.data, schema=diabetes.feature_names)
diabetes_df = diabetes_df.with_columns(pl.Series("target", diabetes.target))

X = diabetes_df.select(diabetes.feature_names).to_numpy()
y = diabetes_df["target"].to_numpy()

# 2. Build pipeline
pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", Ridge(alpha=1.0)),
])

# 3. Create experiment
experiment = Experiment(
    pipeline=pipeline,
    scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error"],
    name="diabetes-progression",
)

# 4. Cross-validate
cv = KFold(n_splits=5, shuffle=True, random_state=42)
result = experiment.cross_validate(X, y, cv=cv, run_name="cv")

mae = -result.metrics["cv/neg_mean_absolute_error_mean"]
print(f"Cross-validated MAE: {mae:.1f}")
```

---

## Best practices

1. **Always cross-validate.** A single split is unreliable for estimating
   generalization performance.

2. **Scale features for linear models.** Tree-based models don't need scaling,
   but linear models do.

3. **Use appropriate metrics.** MAE for interpretability, RMSE if large errors
   are especially costly.

4. **Shuffle before splitting.** Unless you have time-series data—then use
   `TimeSeriesSplit` instead.

5. **Compare to a baseline.** A model that predicts the mean value for everything
   gives you a floor. If your model barely beats this, something's wrong.

## Next steps

- [Hyperparameter Search](sklearn-search.md) — Find better `alpha` values
- [Time Series Forecasting](time-series-forecasting.md) — When temporal order matters
- [Logger Adapters](logger-adapters.md) — Track experiments with MLflow or W&B
