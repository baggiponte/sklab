# Cyclical feature engineering with Experiment

This tutorial shows how to handle cyclical time features (hour, weekday, month)
using sklearn primitives and `Experiment` for consistent evaluation. We use a
small synthetic dataset to keep the example runnable, but the approach is the
same for real time-series data.

## Setup and data

```python
import numpy as np
import polars as pl

from sklearn.model_selection import TimeSeriesSplit

from eksperiment.experiment import Experiment

rng = np.random.default_rng(42)
n_samples = 360

hours = np.arange(n_samples) % 24
weekday = np.arange(n_samples) % 7
month = (np.arange(n_samples) % 12) + 1
weather = rng.integers(0, 4, size=n_samples)
temp = rng.normal(20, 5, size=n_samples)
humidity = rng.uniform(0.2, 0.9, size=n_samples)

signal = (
    10
    + 2 * np.sin(hours / 24 * 2 * np.pi)
    + 1.5 * np.cos(weekday / 7 * 2 * np.pi)
    - 0.5 * weather
    + 0.1 * temp
    - 0.2 * humidity
)
y = signal + rng.normal(0, 0.5, size=n_samples)

features = pl.DataFrame(
    {
        "hour": hours,
        "weekday": weekday,
        "month": month,
        "weather": weather,
        "temp": temp,
        "humidity": humidity,
    }
)

X = features.to_numpy()

# Time-based CV setup
ts_cv = TimeSeriesSplit(n_splits=3)
```

Scorers can be strings or callables; here we use sklearn scorer names. Note that
sklearn’s MAE/RMSE scorers are **negative** by convention, so we flip the sign
for readability when printing.

```{.python continuation}
scorers = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
}

experiment = Experiment(
    pipeline=None,  # filled per model below
    scorers=scorers,
    name="bike-sharing",
)

categorical_columns = [3]  # weather index
hour_column = [0]
weekday_column = [1]
month_column = [2]


def show_metrics(result):
    mae = -result.metrics["cv/mae_mean"]
    rmse = -result.metrics["cv/rmse_mean"]
    print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}")
```

## Model 1: Gradient Boosting (baseline)
Tree-based models can consume ordinal time features directly, while categorical
variables are marked as categorical.

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
show_metrics(result)
```

## Model 2: Naive linear regression (ordinal time)
A linear model with one-hot encoding for categorical features, but time features
left as ordinal.

```{.python continuation}
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

pipeline = make_pipeline(
    ColumnTransformer(
        transformers=[
            ("categorical", one_hot, categorical_columns),
        ],
        remainder=MinMaxScaler(),
    ),
    RidgeCV(alphas=np.logspace(-6, 6, 25)),
)

experiment.pipeline = pipeline
result = experiment.cross_validate(X, y, cv=ts_cv, run_name="linear-ordinal")
show_metrics(result)
```

## Model 3: One-hot time steps
Treat hour/weekday/month as categorical to avoid imposing monotonic ordering.

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
show_metrics(result)
```

## Model 4: Trigonometric (sine/cosine) time features
Encode periodic features with sine/cosine transforms to respect circularity.

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
show_metrics(result)
```

## Model 5: Periodic spline features
Spline features provide a smooth, periodic encoding with more expressivity than
sine/cosine.

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
            (
                "month_spline",
                periodic_spline_transformer(12, n_splines=6),
                month_column,
            ),
            (
                "weekday_spline",
                periodic_spline_transformer(7, n_splines=6),
                weekday_column,
            ),
            (
                "hour_spline",
                periodic_spline_transformer(24, n_splines=12),
                hour_column,
            ),
        ],
        remainder=MinMaxScaler(),
    ),
    RidgeCV(alphas=np.logspace(-6, 6, 25)),
)

experiment.pipeline = pipeline
result = experiment.cross_validate(X, y, cv=ts_cv, run_name="linear-spline-time")
show_metrics(result)
```

## Tradeoffs

- One-hot encodings can be high-dimensional but are simple and robust.
- Sine/cosine and splines model cyclical structure more smoothly.
- Tree models can handle raw ordinal features but may miss periodic structure.
