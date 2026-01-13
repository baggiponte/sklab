# Titanic Classification

This tutorial walks through a compact Titanic-style classification task, using
categorical encoding and a pipeline. The dataset below is a small excerpt to
keep the example runnable.

## Prepare data with Polars

```python
import polars as pl

rows = [
    {"pclass": 1, "sex": "female", "age": 29, "fare": 211.3, "embarked": "C", "survived": 1},
    {"pclass": 1, "sex": "male", "age": 42, "fare": 151.6, "embarked": "S", "survived": 1},
    {"pclass": 2, "sex": "female", "age": 17, "fare": 26.0, "embarked": "S", "survived": 1},
    {"pclass": 3, "sex": "male", "age": 21, "fare": 7.25, "embarked": "S", "survived": 0},
    {"pclass": 3, "sex": "female", "age": 15, "fare": 7.75, "embarked": "Q", "survived": 1},
    {"pclass": 2, "sex": "male", "age": 35, "fare": 13.0, "embarked": "S", "survived": 0},
    {"pclass": 1, "sex": "female", "age": 38, "fare": 80.0, "embarked": "C", "survived": 1},
    {"pclass": 3, "sex": "male", "age": 28, "fare": 7.9, "embarked": "S", "survived": 0},
    {"pclass": 2, "sex": "female", "age": 24, "fare": 26.0, "embarked": "S", "survived": 1},
    {"pclass": 3, "sex": "male", "age": 20, "fare": 8.05, "embarked": "S", "survived": 0},
]

rows = rows * 4

titanic_df = pl.DataFrame(rows)

feature_cols = ["pclass", "sex", "age", "fare", "embarked"]
X = titanic_df.select(feature_cols).to_numpy()
y = titanic_df["survived"].to_numpy()

categorical_cols = [0, 1, 4]
numeric_cols = [2, 3]
```

## Build the pipeline

```{.python continuation}
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from eksperiment.experiment import Experiment

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ],
    remainder="drop",
)

pipeline = Pipeline(
    [
        ("prep", preprocess),
        ("model", LogisticRegression(max_iter=200)),
    ]
)

experiment = Experiment(
    pipeline=pipeline,
    scorers={"accuracy": "accuracy", "f1": "f1"},
    name="titanic",
)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
result = experiment.cross_validate(X, y, cv=cv, run_name="titanic-cv")

print(result.metrics)
```

## Best practices

- Use stratified CV for classification to preserve class balance.
- Keep preprocessing in the pipeline to avoid leakage.
- Add more features (family size, titles, cabin info) before tuning.

## Tradeoffs

- Linear models are easy to interpret but may underfit nonlinear interactions.
- Tree-based models handle mixed data well but are harder to calibrate.
