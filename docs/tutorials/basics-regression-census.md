# Basics: Census Housing Regression

This tutorial uses a small excerpt inspired by the California housing census
features to show a regression workflow. The goal is to demonstrate best
practices with pipelines and cross-validation while keeping the example
runnable offline.

## Prepare data with Polars

```python
import polars as pl

rows = [
    {"median_income": 8.3, "housing_median_age": 41, "total_rooms": 880, "population": 322,
     "households": 126, "latitude": 37.88, "longitude": -122.23, "median_house_value": 4.526},
    {"median_income": 8.3, "housing_median_age": 21, "total_rooms": 7099, "population": 2401,
     "households": 1138, "latitude": 37.86, "longitude": -122.22, "median_house_value": 3.585},
    {"median_income": 7.2, "housing_median_age": 52, "total_rooms": 1467, "population": 496,
     "households": 177, "latitude": 37.85, "longitude": -122.24, "median_house_value": 3.521},
    {"median_income": 5.6, "housing_median_age": 52, "total_rooms": 1274, "population": 558,
     "households": 219, "latitude": 37.85, "longitude": -122.25, "median_house_value": 3.413},
    {"median_income": 4.5, "housing_median_age": 52, "total_rooms": 1627, "population": 565,
     "households": 259, "latitude": 37.85, "longitude": -122.25, "median_house_value": 3.422},
    {"median_income": 4.2, "housing_median_age": 52, "total_rooms": 919, "population": 413,
     "households": 193, "latitude": 37.85, "longitude": -122.25, "median_house_value": 2.697},
    {"median_income": 3.6, "housing_median_age": 52, "total_rooms": 2535, "population": 1094,
     "households": 485, "latitude": 37.84, "longitude": -122.25, "median_house_value": 2.992},
    {"median_income": 3.1, "housing_median_age": 52, "total_rooms": 3104, "population": 1157,
     "households": 584, "latitude": 37.84, "longitude": -122.25, "median_house_value": 2.414},
    {"median_income": 3.2, "housing_median_age": 52, "total_rooms": 2555, "population": 1206,
     "households": 595, "latitude": 37.84, "longitude": -122.25, "median_house_value": 2.267},
    {"median_income": 2.8, "housing_median_age": 42, "total_rooms": 3549, "population": 2208,
     "households": 1041, "latitude": 37.84, "longitude": -122.25, "median_house_value": 2.611},
    {"median_income": 2.5, "housing_median_age": 52, "total_rooms": 2202, "population": 803,
     "households": 325, "latitude": 37.84, "longitude": -122.25, "median_house_value": 2.804},
    {"median_income": 2.8, "housing_median_age": 52, "total_rooms": 3503, "population": 1660,
     "households": 731, "latitude": 37.84, "longitude": -122.25, "median_house_value": 2.67},
    {"median_income": 3.0, "housing_median_age": 52, "total_rooms": 2491, "population": 877,
     "households": 355, "latitude": 37.84, "longitude": -122.24, "median_house_value": 2.701},
    {"median_income": 3.1, "housing_median_age": 52, "total_rooms": 696, "population": 191,
     "households": 90, "latitude": 37.84, "longitude": -122.25, "median_house_value": 2.863},
    {"median_income": 2.6, "housing_median_age": 52, "total_rooms": 2643, "population": 1194,
     "households": 412, "latitude": 37.84, "longitude": -122.24, "median_house_value": 2.252},
]

# Repeat rows to create a small, stable dataset.
rows = rows * 3
census_df = pl.DataFrame(rows)

feature_cols = [
    "median_income",
    "housing_median_age",
    "total_rooms",
    "population",
    "households",
    "latitude",
    "longitude",
]

X = census_df.select(feature_cols).to_numpy()
y = census_df["median_house_value"].to_numpy()
```

## Pipeline + cross-validation

```{.python continuation}
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eksperiment.experiment import Experiment

pipeline = Pipeline(
    [
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ]
)

experiment = Experiment(
    pipeline=pipeline,
    scorers={
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    },
    name="census-housing",
)

cv = KFold(n_splits=3, shuffle=True, random_state=42)
result = experiment.cross_validate(X, y, cv=cv, run_name="census-cv")

print(result.metrics)
```

## Best practices

- Use cross-validation to estimate generalization error.
- Keep data prep inside the pipeline.
- Standardize numeric features for linear models.
