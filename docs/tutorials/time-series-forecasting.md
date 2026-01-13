# Time Series Forecasting (AirPassengers)

This tutorial uses a small slice of the AirPassengers dataset to demonstrate
best practices for time series cross-validation.

## Prepare data with Polars

```python
import math
import polars as pl

from sklearn.model_selection import TimeSeriesSplit

from eksperiment.experiment import Experiment

passengers = [
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
]

n = len(passengers)
months = [(i % 12) + 1 for i in range(n)]
idx = list(range(n))

series = pl.DataFrame({"t": idx, "month": months, "passengers": passengers})
series = series.with_columns(
    (
        pl.col("month").map_elements(lambda m: math.sin(m / 12 * 2 * math.pi))
    ).alias("month_sin"),
    (
        pl.col("month").map_elements(lambda m: math.cos(m / 12 * 2 * math.pi))
    ).alias("month_cos"),
)

feature_cols = ["t", "month_sin", "month_cos"]
X = series.select(feature_cols).to_numpy()
y = series["passengers"].to_numpy()

ts_cv = TimeSeriesSplit(n_splits=3)
```

## Build the pipeline

```{.python continuation}
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline(
    [
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ]
)

experiment = Experiment(
    pipeline=pipeline,
    scorers={"mae": "neg_mean_absolute_error"},
    name="airpassengers",
)

result = experiment.cross_validate(X, y, cv=ts_cv, run_name="air-cv")
print(result.metrics)
```

## Best practices

- Use `TimeSeriesSplit` or explicit temporal splits; never shuffle time series.
- Add seasonal features (sine/cosine) for periodic data.
- For real forecasting, build lag features and test with a true holdout horizon.

## Tradeoffs

- Simple linear models are fast but may miss nonlinear effects.
- More complex models (GBDT, ARIMA, LSTM) need careful validation and tuning.
