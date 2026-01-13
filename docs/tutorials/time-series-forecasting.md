# Time Series Forecasting: Mauna Loa CO2

**What you'll learn:**

- Why time series data requires special cross-validation strategies
- How to encode seasonal patterns with sine/cosine features
- The difference between temporal and random splits
- How to avoid "predicting the past with the future"

**Prerequisites:** [Regression Workflow](basics-regression-census.md), basic understanding
of cross-validation.

## The problem: time has a direction

Most machine learning assumes observations are independent—shuffling the data
shouldn't affect predictions. Time series data breaks this assumption. Tomorrow's
temperature depends on today's. Next month's sales depend on this month's trends.

This creates a unique pitfall: **temporal leakage**. If you train on data from
2024 and test on data from 2023, you're using the future to predict the past.
Your model looks great in validation but fails in production, where it must
predict genuinely unseen futures.

We'll use atmospheric CO2 measurements from Mauna Loa Observatory—one of the
longest continuous environmental records in existence. The data has a clear
trend (rising CO2) and strong seasonality (annual cycles from plant growth).

---

## Step 1: Load and prepare the data

```python
import math
import polars as pl

from sklearn.datasets import fetch_openml

co2 = fetch_openml(data_id=41187, as_frame=False, parser="liac-arff")
series = pl.DataFrame(co2.data, schema=co2.feature_names)
series = series.with_columns(pl.Series("co2", co2.target))
series = series.with_columns(
    pl.date(
        pl.col("year").cast(pl.Int32),
        pl.col("month").cast(pl.Int32),
        pl.col("day").cast(pl.Int32),
    ).alias("date")
).select(["date", "co2"])

series = series.sort("date")
print(f"Date range: {series['date'].min()} to {series['date'].max()}")
print(f"CO2 range: {series['co2'].min():.1f} to {series['co2'].max():.1f} ppm")
```

> **Concept: The Mauna Loa Dataset**
>
> Scientists at Mauna Loa, Hawaii have measured atmospheric CO2 since 1958.
> The data shows two patterns: a long-term upward trend (from fossil fuels)
> and an annual cycle (plants absorb CO2 in summer, release it in winter).
>
> **Why it matters:** This combination of trend and seasonality is common in
> real-world time series—sales, traffic, energy usage. Learning to model it
> here transfers to many practical problems.

---

## Step 2: Engineer time features

Raw dates aren't useful for regression. We need numeric features that capture
temporal patterns.

```{.python continuation}
# Add time index and seasonal features
series = series.with_columns(pl.Series("t", range(series.height)))
series = series.with_columns(pl.col("date").dt.month().alias("month"))

# Encode month as sine/cosine to capture annual cycle
series = series.with_columns(
    (
        pl.col("month").map_elements(
            lambda m: math.sin(m / 12 * 2 * math.pi), return_dtype=pl.Float64
        )
    ).alias("month_sin"),
    (
        pl.col("month").map_elements(
            lambda m: math.cos(m / 12 * 2 * math.pi), return_dtype=pl.Float64
        )
    ).alias("month_cos"),
)

feature_cols = ["t", "month_sin", "month_cos"]
X = series.select(feature_cols).to_numpy()
y = series["co2"].to_numpy()

print(f"Features: {feature_cols}")
print(f"Samples: {X.shape[0]}")
```

**What this does:**

- `t`: Linear time index (captures long-term trend)
- `month_sin`, `month_cos`: Encode the annual cycle as coordinates on a circle

> **Concept: Sine/Cosine Encoding**
>
> Months aren't linear—December (12) is close to January (1), not far from it.
> Encoding month as a number treats 12 and 1 as distant, which breaks cyclical
> relationships.
>
> Sine and cosine transform the cycle into circular coordinates. On a circle,
> December and January are neighbors. The model sees smooth, continuous seasonal
> patterns instead of artificial jumps.
>
> **Why it matters:** Linear models can't learn "month 12 is near month 1" from
> raw numbers. Sine/cosine encoding hands the model this relationship for free.

---

## Step 3: Use temporal cross-validation

This is where time series differs critically from other ML problems.

```{.python continuation}
from sklearn.model_selection import TimeSeriesSplit

ts_cv = TimeSeriesSplit(n_splits=3)

# Visualize the splits
for i, (train_idx, test_idx) in enumerate(ts_cv.split(X)):
    print(f"Fold {i+1}: Train on {len(train_idx)} samples, test on {len(test_idx)}")
```

> **Concept: Why Not Random Splits?**
>
> Random k-fold cross-validation shuffles data, then splits. For time series,
> this means training on 2020 data and testing on 2018 data—using the future
> to predict the past.
>
> `TimeSeriesSplit` respects temporal order: always train on earlier data,
> test on later data. Each fold expands the training set forward in time.
>
> **Why it matters:** A model validated with random splits will look great but
> fail in production. It's seen the future during training—something impossible
> in real deployment.

---

## Step 4: Build the pipeline and experiment

```{.python continuation}
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eksperiment.experiment import Experiment

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", Ridge(alpha=1.0)),
])

experiment = Experiment(
    pipeline=pipeline,
    scorers={"mae": "neg_mean_absolute_error"},
    name="mauna-loa-co2",
)
```

---

## Step 5: Cross-validate with temporal splits

```{.python continuation}
result = experiment.cross_validate(X, y, cv=ts_cv, run_name="co2-cv")

mae = -result.metrics["cv/mae_mean"]
print(f"Cross-validated MAE: {mae:.2f} ppm")
```

**What just happened:**

1. Fold 1: Trained on earliest data, tested on next segment
2. Fold 2: Trained on larger early segment, tested on following data
3. Fold 3: Trained on even more data, tested on most recent data
4. Each fold simulates real deployment: model sees only past, predicts future

### Interpreting the results

An MAE of ~2 ppm means predictions are typically off by about 2 parts per million.
Given CO2 levels around 350-400 ppm, this is reasonable for a simple linear model.

---

## The danger of ignoring temporal order

Let's see what happens with random splits—the wrong approach:

```{.python continuation}
from sklearn.model_selection import KFold

# WRONG for time series: random splits
wrong_cv = KFold(n_splits=3, shuffle=True, random_state=42)
wrong_result = experiment.cross_validate(X, y, cv=wrong_cv, run_name="wrong-cv")

wrong_mae = -wrong_result.metrics["cv/mae_mean"]
print(f"Random CV MAE: {wrong_mae:.2f} ppm (misleadingly good!)")
```

The random-split MAE is likely lower—but it's a lie. The model saw future data
during training for each fold. In production, where you can only predict forward,
performance will match the temporal CV estimate.

---

## Complete example

```python
import math
import polars as pl

from sklearn.datasets import fetch_openml
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eksperiment.experiment import Experiment

# 1. Load data
co2 = fetch_openml(data_id=41187, as_frame=False, parser="liac-arff")
series = pl.DataFrame(co2.data, schema=co2.feature_names)
series = series.with_columns(pl.Series("co2", co2.target))
series = series.with_columns(
    pl.date(
        pl.col("year").cast(pl.Int32),
        pl.col("month").cast(pl.Int32),
        pl.col("day").cast(pl.Int32),
    ).alias("date")
).select(["date", "co2"])
series = series.sort("date")

# 2. Engineer features
series = series.with_columns(pl.Series("t", range(series.height)))
series = series.with_columns(pl.col("date").dt.month().alias("month"))
series = series.with_columns(
    (
        pl.col("month").map_elements(
            lambda m: math.sin(m / 12 * 2 * math.pi), return_dtype=pl.Float64
        )
    ).alias("month_sin"),
    (
        pl.col("month").map_elements(
            lambda m: math.cos(m / 12 * 2 * math.pi), return_dtype=pl.Float64
        )
    ).alias("month_cos"),
)

feature_cols = ["t", "month_sin", "month_cos"]
X = series.select(feature_cols).to_numpy()
y = series["co2"].to_numpy()

# 3. Set up experiment with temporal CV
pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", Ridge(alpha=1.0)),
])

experiment = Experiment(
    pipeline=pipeline,
    scorers={"mae": "neg_mean_absolute_error"},
    name="mauna-loa-co2",
)

ts_cv = TimeSeriesSplit(n_splits=3)
result = experiment.cross_validate(X, y, cv=ts_cv, run_name="final")

print(f"MAE: {-result.metrics['cv/mae_mean']:.2f} ppm")
```

---

## Best practices

1. **Always use temporal splits.** `TimeSeriesSplit`, expanding window, or rolling
   window—never random splits for time series.

2. **Encode cyclical features properly.** Use sine/cosine for hours, days, months.
   See [Cyclical Feature Engineering](cyclical-feature-engineering.md) for advanced
   techniques.

3. **Include a trend feature.** Many time series have underlying trends. A simple
   time index captures linear trends; more complex trends need domain modeling.

4. **Hold out a true future test set.** Cross-validation estimates performance,
   but keep the most recent data completely untouched for final evaluation.

5. **Consider lag features.** For true forecasting, include lagged values of the
   target (yesterday's CO2 to predict today's). This requires careful feature
   engineering to avoid leakage.

## Tradeoffs

| Approach | Pros | Cons |
|----------|------|------|
| Linear + sine/cosine | Fast, interpretable | Can't capture complex patterns |
| Gradient boosting | Handles nonlinearity | May overfit short series |
| ARIMA/Prophet | Built for time series | Different API, less sklearn-compatible |
| LSTMs/Transformers | Learns complex patterns | Needs lots of data, slow to train |

## Next steps

- [Cyclical Feature Engineering](cyclical-feature-engineering.md) — Compare encoding methods
- [Hyperparameter Search](sklearn-search.md) — Find optimal model settings
- [Why Pipelines Matter](why-pipelines.md) — Prevent leakage in preprocessing
