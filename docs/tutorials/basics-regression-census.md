# Regression Workflow: California Housing

**What you'll learn:**

- How to structure a regression experiment with sklab
- Why cross-validation gives more reliable estimates than a single holdout split
- How to interpret MAE and RMSE metrics
- The importance of scaling features for linear models

**Prerequisites:** Basic Python and sklearn familiarity. If you're new to pipelines,
read [Why Pipelines Matter](why-pipelines.md) first.

## The problem: predicting house prices

Predicting continuous values—prices, temperatures, sales figures—is fundamentally
different from classification. Instead of "which category?", we ask "how much?".

This difference affects everything: which metrics make sense, how to interpret
errors, and what pitfalls to avoid. A classifier that's wrong is just wrong,
but a regressor can be wrong by a little (off by $1,000) or a lot (off by $1,000,000).

We'll use California housing data to predict median house values for districts.
This is a classic regression task: given features like median income, average
rooms, and location, estimate the median house value.

---

## Step 1: Load the data

```python
import polars as pl
from sklearn.datasets import fetch_california_housing

# Load the dataset
housing = fetch_california_housing()

# Keep data in Polars for exploration, convert to NumPy for sklearn
housing_df = pl.DataFrame(housing.data, schema=housing.feature_names)
housing_df = housing_df.with_columns(pl.Series("target", housing.target))

print(housing_df.head())
print(f"\nSamples: {housing_df.shape[0]}, Features: {len(housing.feature_names)}")
print(f"Target range: ${housing_df['target'].min() * 100_000:.0f} - ${housing_df['target'].max() * 100_000:.0f}")

X = housing_df.select(housing.feature_names).to_numpy()
y = housing_df["target"].to_numpy()
```

The first run downloads the dataset and caches it in the scikit-learn data
directory (defaults to `~/scikit_learn_data`).

> **Concept: Regression Targets**
>
> Unlike classification (discrete labels like "spam" or "not spam"), regression
> predicts continuous values. The California housing target is median house value
> in $100,000s—so a prediction of 2.5 means $250,000.
>
> **Why it matters:** Regression errors are distances, not just "right or wrong."
> A prediction of $245,000 when the true value is $250,000 is much better than
> predicting $150,000.

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
> Linear models are sensitive to feature scales. If "median income" ranges from
> 0-15 while "latitude" ranges from 32-42, the model will weight them unevenly
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
    scorers={
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    },
    name="california-housing",
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
result = experiment.cross_validate(X, y, cv=cv, run_name="california-cv")

# Flip signs for readability (sklearn uses negative scores)
mae = -result.metrics["cv/mae_mean"]
rmse = -result.metrics["cv/rmse_mean"]

print(f"MAE:  ${mae * 100_000:,.0f} (average error)")
print(f"RMSE: ${rmse * 100_000:,.0f} (penalizes large errors)")
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

- **MAE of ~$50,000** means predictions are typically off by about $50,000
- **RMSE higher than MAE** indicates some predictions have large errors

For California housing (median values ~$200,000), an MAE of $50,000 means
~25% average error. Whether that's acceptable depends on your use case.

### What to try next

If results aren't good enough:

1. **Try a more complex model:** Gradient boosting often outperforms linear models
2. **Engineer features:** Combine latitude/longitude into distance-from-coast
3. **Tune hyperparameters:** Search over `alpha` values with `experiment.search()`

---

## Complete example

Here's the full workflow in one block:

```python
import polars as pl
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklab.experiment import Experiment

# 1. Load data
housing = fetch_california_housing()
housing_df = pl.DataFrame(housing.data, schema=housing.feature_names)
housing_df = housing_df.with_columns(pl.Series("target", housing.target))

X = housing_df.select(housing.feature_names).to_numpy()
y = housing_df["target"].to_numpy()

# 2. Build pipeline
pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", Ridge(alpha=1.0)),
])

# 3. Create experiment
experiment = Experiment(
    pipeline=pipeline,
    scorers={
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    },
    name="california-housing",
)

# 4. Cross-validate
cv = KFold(n_splits=5, shuffle=True, random_state=42)
result = experiment.cross_validate(X, y, cv=cv, run_name="cv")

mae = -result.metrics["cv/mae_mean"]
print(f"Cross-validated MAE: ${mae * 100_000:,.0f}")
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
