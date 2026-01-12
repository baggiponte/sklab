# Experiment tutorial

This tutorial shows a minimal experiment run with sklearn, using the default no-op logger.

## Fit and evaluate

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eksperiment.experiment import Experiment

X, y = load_iris(return_X_y=True)

pipeline = Pipeline(
    [
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=200)),
    ]
)

experiment = Experiment(
    pipeline=pipeline,
    scorers={"accuracy": "accuracy"},
    name="baseline-iris",
)

fit_result = experiment.fit(X, y, run_name="baseline-fit")

holdout_metrics = experiment.evaluate(
    fit_result.estimator,
    X,
    y,
    run_name="baseline-eval",
)

print(holdout_metrics.metrics)
```

## Cross-validate

```python
cv_result = experiment.cross_validate(
    X,
    y,
    cv=5,
    run_name="baseline-cv",
)

print(cv_result.metrics)
```

## Tune with an external searcher

`tune()` expects an explicit searcher or config object. Here is a minimal searcher sketch:

```python
from dataclasses import dataclass
from sklearn.base import clone

@dataclass
class DummySearch:
    estimator: Pipeline
    best_params_: dict | None = None
    best_score_: float | None = None
    best_estimator_: Pipeline | None = None

    def fit(self, X, y=None):
        self.best_params_ = {"model__C": 1.0}
        self.best_score_ = 0.5
        self.best_estimator_ = clone(self.estimator).fit(X, y)
        return self

searcher = DummySearch(pipeline)
result = experiment.tune(searcher, X, y, run_name="baseline-tune")
print(result.best_params, result.best_score)
```
