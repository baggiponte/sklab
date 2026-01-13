# Logger adapters

Eksperiment is backend-agnostic: you can pass a logger adapter that conforms to
`LoggerProtocol` (a protocol, so inheritance isn't required). This tutorial
shows the built-ins and a simple custom logger.

## Default: No-op logger

If you do nothing, Eksperiment uses a no-op logger.

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eksperiment.experiment import Experiment

X, y = load_iris(return_X_y=True)

experiment = Experiment(
    pipeline=Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=200)),
        ]
    ),
    scorers={"accuracy": "accuracy"},
    name="noop-run",
)

fit_result = experiment.fit(X, y, run_name="noop-fit")
metrics = experiment.evaluate(fit_result.estimator, X, y, run_name="noop-eval")
print(metrics.metrics)
```

## W&B adapter (optional)

```python
import pytest
pytest.importorskip("wandb")

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eksperiment.experiment import Experiment
from eksperiment.logging.adapters import WandbLogger

X, y = load_iris(return_X_y=True)

experiment = Experiment(
    pipeline=Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=200)),
        ]
    ),
    scorers={"accuracy": "accuracy"},
    logger=WandbLogger(project="eksperiment-demo"),
    name="wandb-demo",
)

fit_result = experiment.fit(X, y, run_name="wandb-fit")
metrics = experiment.evaluate(fit_result.estimator, X, y, run_name="wandb-eval")
print(metrics.metrics)
```

## MLflow adapter (optional)

```python
import pytest
pytest.importorskip("mlflow")

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eksperiment.experiment import Experiment
from eksperiment.logging.adapters import MLflowLogger

X, y = load_iris(return_X_y=True)

experiment = Experiment(
    pipeline=Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=200)),
        ]
    ),
    scorers={"accuracy": "accuracy"},
    logger=MLflowLogger(experiment_name="eksperiment-demo"),
    name="mlflow-demo",
)

fit_result = experiment.fit(X, y, run_name="mlflow-fit")
metrics = experiment.evaluate(fit_result.estimator, X, y, run_name="mlflow-eval")
print(metrics.metrics)
```

## Custom logger (quick stub)

Use this as a starting point for your own logger integration.

```python
from dataclasses import dataclass
from typing import Any

from eksperiment.logging.interfaces import LoggerProtocol, RunProtocol

@dataclass
class PrintRun:
    def __enter__(self) -> "PrintRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool | None:
        return None

    def log_params(self, params) -> None:
        print("params", params)

    def log_metrics(self, metrics, step=None) -> None:
        print("metrics", metrics)

    def set_tags(self, tags) -> None:
        print("tags", tags)

    def log_artifact(self, path: str, name: str | None = None) -> None:
        print("artifact", path, name)

    def log_model(self, model: Any, name: str | None = None) -> None:
        print("model", name)

    def finish(self, status: str = "success") -> None:
        print("finish", status)


@dataclass
class PrintLogger:
    def start_run(self, name=None, config=None, tags=None, nested=False) -> PrintRun:
        run = PrintRun()
        if config:
            run.log_params(config)
        if tags:
            run.set_tags(tags)
        return run
```
