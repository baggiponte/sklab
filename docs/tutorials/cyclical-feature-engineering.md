# Cyclical Feature Engineering

**What you'll learn:**

- Why cyclical features (hour, weekday, month) break ordinary encoding
- How to encode cycles with sine/cosine transforms
- When periodic splines outperform trigonometric features
- How tree-based models handle cyclical data differently

**Prerequisites:** [Time Series Forecasting](time-series-forecasting.md),
understanding of feature engineering.

## The problem: cycles don't have edges

Hour 23 is one hour from hour 0. December is one month from January. But if you
encode hour as a number (0-23) or month as a number (1-12), the model sees 23 as
far from 0, and 12 as far from 1. This artificial "edge" at the cycle boundary
breaks relationships that should be smooth and continuous.

Consider predicting bike rentals by hour. Demand at 11 PM likely resembles demand
at midnight—both are late-night hours. But a linear model with raw hour encoding
sees them as maximally distant (23 vs. 0), completely missing the pattern.

This tutorial compares five encoding strategies on synthetic data with known
cyclical patterns, so you can see which approaches recover the true signal.

---

## Setup: synthetic data with cyclical patterns

We'll create data where the true signal depends on hour-of-day, day-of-week, and
month. This lets us measure how well each encoding recovers the underlying cycles.

```python
import numpy as np
import polars as pl

from sklearn.model_selection import TimeSeriesSplit

from sklab.experiment import Experiment

rng = np.random.default_rng(42)
n_samples = 360

# Time features
hours = np.arange(n_samples) % 24
weekday = np.arange(n_samples) % 7
month = (np.arange(n_samples) % 12) + 1

# Other features
weather = rng.integers(0, 4, size=n_samples)
temp = rng.normal(20, 5, size=n_samples)
humidity = rng.uniform(0.2, 0.9, size=n_samples)

# True signal: cyclical patterns + linear effects
signal = (
    10
    + 2 * np.sin(hours / 24 * 2 * np.pi)      # hourly cycle
    + 1.5 * np.cos(weekday / 7 * 2 * np.pi)   # weekly cycle
    - 0.5 * weather
    + 0.1 * temp
    - 0.2 * humidity
)
y = signal + rng.normal(0, 0.5, size=n_samples)

features = pl.DataFrame({
    "hour": hours,
    "weekday": weekday,
    "month": month,
    "weather": weather,
    "temp": temp,
    "humidity": humidity,
})

X = features.to_numpy()
ts_cv = TimeSeriesSplit(n_splits=3)
```

!!! note "Concept: Why Synthetic Data?"

    With real data, we don't know the true underlying patterns. Synthetic data
    lets us embed known cycles and measure how well each encoding recovers them.
    If an encoding works on synthetic data with the right structure, it will
    generalize to real data with similar patterns.

---

## Experiment setup

We'll compare all five approaches using the same scoring and CV strategy.

```{.python continuation}
scoring = [
    "neg_mean_absolute_error",
    "neg_root_mean_squared_error",
]

experiment = Experiment(
    pipeline=None,  # set per model
    scoring=scoring,
    name="cyclical-features",
)

# Column indices
categorical_columns = [3]  # weather
hour_column = [0]
weekday_column = [1]
month_column = [2]


def show_metrics(result):
    mae = -result.metrics["cv/neg_mean_absolute_error_mean"]
    rmse = -result.metrics["cv/neg_root_mean_squared_error_mean"]
    print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}")
```

---

## Model 1: Gradient Boosting baseline

Tree-based models can consume ordinal features directly—they split on thresholds.
This baseline shows what you get without explicit cyclical encoding.

```{.python continuation}
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(
    ColumnTransformer(
        transformers=[
            ("categorical", "passthrough", categorical_columns),
        ],
        remainder="passthrough",
    ),
    HistGradientBoostingRegressor(
        categorical_features=categorical_columns,
        random_state=42,
    ),
)

experiment.pipeline = pipeline
result = experiment.cross_validate(X, y, cv=ts_cv, run_name="gbrt")
print("1. Gradient Boosting (raw ordinal):")
show_metrics(result)
```

!!! note "Concept: How Trees Handle Cycles"

    Decision trees split on threshold comparisons: "hour < 12?" They can learn
    that hours 22, 23, 0, 1 share similar patterns by creating multiple splits.
    But this requires the model to "discover" the cycle from data, rather than
    encoding it directly.

    **Why it matters:** Trees work decently on cyclical data but waste capacity
    re-learning patterns that simple encoding could provide for free.

---

## Model 2: Linear regression with ordinal encoding

A linear model that treats hour/weekday/month as raw numbers—the naive approach.

```{.python continuation}
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

pipeline = make_pipeline(
    ColumnTransformer(
        transformers=[
            ("categorical", one_hot, categorical_columns),
        ],
        remainder=MinMaxScaler(),  # scales time features to 0-1
    ),
    RidgeCV(alphas=np.logspace(-6, 6, 25)),
)

experiment.pipeline = pipeline
result = experiment.cross_validate(X, y, cv=ts_cv, run_name="linear-ordinal")
print("\n2. Linear (ordinal time):")
show_metrics(result)
```

This typically performs poorly because linear models can only learn monotonic
relationships with numeric features. Hour 23 being "high" and hour 0 being "low"
breaks the actual pattern.

---

## Model 3: One-hot encoded time

Treat each hour, weekday, and month as a separate category. No assumptions about
relationships between values.

```{.python continuation}
pipeline = make_pipeline(
    ColumnTransformer(
        transformers=[
            ("categorical", one_hot, categorical_columns),
            ("hour_one_hot", one_hot, hour_column),
            ("weekday_one_hot", one_hot, weekday_column),
            ("month_one_hot", one_hot, month_column),
        ],
        remainder=MinMaxScaler(),
    ),
    RidgeCV(alphas=np.logspace(-6, 6, 25)),
)

experiment.pipeline = pipeline
result = experiment.cross_validate(X, y, cv=ts_cv, run_name="linear-one-hot-time")
print("\n3. Linear (one-hot time):")
show_metrics(result)
```

!!! note "Concept: One-Hot Trade-offs"

    One-hot encoding creates 24 features for hour, 7 for weekday, 12 for month.
    Each time point gets its own coefficient—maximum flexibility.

    **The catch:** The model doesn't know hour 23 and hour 0 are similar. It must
    learn this from data, if it can at all. And with 43 time features, you need
    enough data to estimate them all reliably.

---

## Model 4: Trigonometric (sine/cosine) encoding

Transform cyclical features into coordinates on a circle. This explicitly encodes
that the cycle wraps around.

```{.python continuation}
from sklearn.preprocessing import FunctionTransformer


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


pipeline = make_pipeline(
    ColumnTransformer(
        transformers=[
            ("categorical", one_hot, categorical_columns),
            ("month_sin", sin_transformer(12), month_column),
            ("month_cos", cos_transformer(12), month_column),
            ("weekday_sin", sin_transformer(7), weekday_column),
            ("weekday_cos", cos_transformer(7), weekday_column),
            ("hour_sin", sin_transformer(24), hour_column),
            ("hour_cos", cos_transformer(24), hour_column),
        ],
        remainder=MinMaxScaler(),
    ),
    RidgeCV(alphas=np.logspace(-6, 6, 25)),
)

experiment.pipeline = pipeline
result = experiment.cross_validate(X, y, cv=ts_cv, run_name="linear-trig-time")
print("\n4. Linear (sine/cosine time):")
show_metrics(result)
```

!!! note "Concept: The Circle Trick"

    Sine and cosine map a cycle onto a unit circle. Hour 0 and hour 23 are now
    geometrically close—they're neighbors on the circle. The Euclidean distance
    between their (sin, cos) coordinates reflects their true temporal distance.

    **Why it matters:** With just 2 features per cycle (sin + cos), you encode
    the wraparound structure explicitly. The model doesn't need to discover it.

---

## Model 5: Periodic spline features

Splines create smooth basis functions that respect periodicity. More expressive
than sine/cosine, but more features.

```{.python continuation}
from sklearn.preprocessing import SplineTransformer


def periodic_spline_transformer(period, n_splines=None, degree=3):
    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )


pipeline = make_pipeline(
    ColumnTransformer(
        transformers=[
            ("categorical", one_hot, categorical_columns),
            ("month_spline", periodic_spline_transformer(12, n_splines=6), month_column),
            ("weekday_spline", periodic_spline_transformer(7, n_splines=6), weekday_column),
            ("hour_spline", periodic_spline_transformer(24, n_splines=12), hour_column),
        ],
        remainder=MinMaxScaler(),
    ),
    RidgeCV(alphas=np.logspace(-6, 6, 25)),
)

experiment.pipeline = pipeline
result = experiment.cross_validate(X, y, cv=ts_cv, run_name="linear-spline-time")
print("\n5. Linear (periodic splines):")
show_metrics(result)
```

!!! note "Concept: Splines vs. Sine/Cosine"

    Sine/cosine can only model single-frequency patterns. Real hourly effects
    might peak at 8 AM and 6 PM—a two-peak pattern that needs multiple harmonics.

    Periodic splines create flexible basis functions that can capture arbitrary
    shapes while still wrapping smoothly around the cycle boundary. They need
    more features but can model complex patterns.

    **When to use which:** Sine/cosine for simple, single-peak cycles. Splines
    when you expect multi-modal or asymmetric patterns.

---

## Comparison summary

| Approach | Features | Captures cycles? | Flexibility |
|----------|----------|------------------|-------------|
| Raw ordinal | 1 per cycle | No | Low (linear only) |
| One-hot | Period per cycle | Implicitly | High |
| Sine/cosine | 2 per cycle | Yes | Low (single frequency) |
| Periodic splines | Configurable | Yes | High |
| Tree (raw) | 1 per cycle | Learns from data | High |

---

## Best practices

1. **Use sine/cosine as a default.** For most cyclical features, sine/cosine
   provides good performance with minimal features.

2. **Consider splines for complex patterns.** If your cycle has multiple peaks
   or asymmetric shapes, splines capture more detail.

3. **Trees can work without encoding.** If you're using gradient boosting,
   raw ordinal features often suffice—but explicit encoding can still help.

4. **Match encoding to model.** Linear models benefit most from cyclical
   encoding. Trees are more forgiving of raw values.

5. **Don't forget the period.** The period (24 for hours, 7 for days, 12 for
   months) must match your data's actual cycle length.

## Tradeoffs

| Choice | Pros | Cons |
|--------|------|------|
| Ordinal | Simple, compact | Breaks at cycle boundaries |
| One-hot | Maximum flexibility | High dimensionality, no smoothness |
| Sine/cosine | Compact, smooth | Single frequency only |
| Periodic splines | Flexible, smooth | More features, harder to tune |

## Next steps

- [Time Series Forecasting](time-series-forecasting.md) — Apply these techniques
- [Hyperparameter Search](sklearn-search.md) — Tune spline parameters
- [Why Pipelines Matter](why-pipelines.md) — Keep encoding in the pipeline
