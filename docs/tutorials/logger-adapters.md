# Logger Adapters: Tracking Your Experiments

**What you'll learn:**

- Why experiment tracking matters for reproducibility
- How sklab's logging works with MLflow and W&B
- When to use each logging backend
- How to build custom loggers for other backends

**Prerequisites:** [The Experiment Class](experiment.md), basic understanding
of ML workflows.

## The problem: experiments are easy to lose

You run 50 experiments over two weeks. Some use different hyperparameters, some
use different preprocessing, some use different data splits. At the end, you
know one worked well—but which one? What were its settings?

Manual tracking (spreadsheets, notes, file names) breaks down at scale. You
forget to update the sheet. You overwrite a file. You can't remember if "model_v3"
was before or after you changed the learning rate.

Experiment tracking solves this by automatically logging:
- **Parameters:** Every hyperparameter and setting
- **Metrics:** Training and validation scores
- **Artifacts:** Models, plots, predictions
- **Metadata:** Timestamps, run names, tags

sklab integrates with logging backends through **adapters**—pluggable
components that translate experiment events into backend-specific API calls.

---

## How sklab logging works

Every `Experiment` method (`fit`, `evaluate`, `cross_validate`, `search`) logs
automatically when you provide a logger:

```
experiment.fit(X, y)
    └── logger.start_run()
        └── log_params(pipeline params)
        └── log_metrics(training metrics)
        └── log_model(fitted pipeline) [if enabled]
        └── finish()
```

Without a logger (the default), nothing is logged. With a logger, everything
is captured consistently across all operations.

---

## Default: No-op logger

If you don't specify a logger, sklab uses a no-op that does nothing.
This is useful for development and testing when you don't need tracking.

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklab.experiment import Experiment

X, y = load_iris(return_X_y=True)

# No logger specified = no-op logging
experiment = Experiment(
    pipeline=Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=200)),
    ]),
    scorers={"accuracy": "accuracy"},
    name="no-logging",
)

fit_result = experiment.fit(X, y, run_name="noop-fit")
eval_result = experiment.evaluate(fit_result.estimator, X, y, run_name="noop-eval")
print(eval_result.metrics)
```

---

## Weights & Biases adapter

W&B provides cloud-based experiment tracking with rich visualization. The
adapter logs everything to your W&B project.

```python
import pytest
pytest.importorskip("wandb")

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklab.experiment import Experiment
from sklab.logging.adapters import WandbLogger

X, y = load_iris(return_X_y=True)

experiment = Experiment(
    pipeline=Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=200)),
    ]),
    scorers={"accuracy": "accuracy"},
    logger=WandbLogger(project="sklab-demo"),
    name="wandb-demo",
)

fit_result = experiment.fit(X, y, run_name="wandb-fit")
eval_result = experiment.evaluate(fit_result.estimator, X, y, run_name="wandb-eval")
print(eval_result.metrics)
```

> **Concept: W&B Projects**
>
> W&B organizes runs into projects. Each run tracks one experiment execution.
> The project dashboard shows all runs with their parameters and metrics,
> making comparison easy.
>
> **Why it matters:** You can filter, sort, and compare runs across days or
> weeks of experimentation. The web UI handles visualization so you don't
> have to build custom dashboards.

---

## MLflow adapter

MLflow provides open-source experiment tracking with local or remote storage.
Good for teams that want control over their tracking infrastructure.

```python
import pytest
pytest.importorskip("mlflow")

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklab.experiment import Experiment
from sklab.logging.adapters import MLflowLogger

X, y = load_iris(return_X_y=True)

experiment = Experiment(
    pipeline=Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=200)),
    ]),
    scorers={"accuracy": "accuracy"},
    logger=MLflowLogger(experiment_name="sklab-demo"),
    name="mlflow-demo",
)

fit_result = experiment.fit(X, y, run_name="mlflow-fit")
eval_result = experiment.evaluate(fit_result.estimator, X, y, run_name="mlflow-eval")
print(eval_result.metrics)
```

> **Concept: MLflow Tracking Server**
>
> MLflow can store runs locally (default) or on a remote tracking server.
> Local storage is simple but team collaboration requires a server.
>
> **Why it matters:** For personal projects, local MLflow "just works." For
> teams, deploy a tracking server to share experiments.

---

## Decision guide: which logger to use

| Situation | Recommendation |
|-----------|----------------|
| Quick experiments, no tracking needed | No logger (default) |
| Personal projects, cloud convenience | W&B |
| Team projects, need control over infrastructure | MLflow |
| Already using a specific platform | Use that platform's adapter |
| Need something custom | Build a custom logger |

### W&B vs. MLflow

| Feature | W&B | MLflow |
|---------|-----|--------|
| Hosting | Cloud (SaaS) | Self-hosted or local |
| Setup | Sign up, done | Install, run server (for teams) |
| Cost | Free tier, paid for teams | Free, open source |
| UI | Rich, polished | Functional, simpler |
| Collaboration | Built-in | Requires tracking server |

---

## Custom logger: build your own

Loggers are simple to build. Implement the protocol and you can log to any
backend—databases, cloud storage, custom dashboards.

```python
from dataclasses import dataclass
from typing import Any

from sklab.logging.interfaces import LoggerProtocol, RunProtocol


@dataclass
class PrintRun:
    """A run that prints everything to stdout."""

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
    """A logger that uses PrintRun for all runs."""

    def start_run(
        self, name=None, config=None, tags=None, nested=False
    ) -> PrintRun:
        run = PrintRun()
        if config:
            run.log_params(config)
        if tags:
            run.set_tags(tags)
        return run
```

> **Concept: The Logger Protocol**
>
> sklab uses structural typing (protocols) rather than inheritance.
> Any object with `start_run()` that returns an object with `log_params()`,
> `log_metrics()`, etc. is a valid logger.
>
> **Why it matters:** You don't need to inherit from a base class. Just
> implement the methods and it works.

### Using the custom logger

```{.python continuation}
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklab.experiment import Experiment

X, y = load_iris(return_X_y=True)

experiment = Experiment(
    pipeline=Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=200)),
    ]),
    scorers={"accuracy": "accuracy"},
    logger=PrintLogger(),  # Uses our custom logger
    name="custom-logger",
)

result = experiment.fit(X, y, run_name="custom-fit")
```

---

## What gets logged

Every sklab operation logs specific data:

| Method | Logged Data |
|--------|-------------|
| `fit()` | Pipeline parameters, fit timing |
| `evaluate()` | Metrics, predictions (optional) |
| `cross_validate()` | Per-fold metrics, mean/std metrics |
| `search()` | All trial parameters, best params, best score |

The exact data depends on your configuration—some loggers support model
artifacts, others only capture metrics.

---

## Best practices

1. **Start with no logging.** Get your experiment working first. Add logging
   when you need to compare runs.

2. **Use consistent naming.** Run names should describe the experiment:
   `"ridge-alpha-0.1"` not `"test3"`.

3. **Add tags for filtering.** Tags like `"baseline"`, `"production-candidate"`,
   or `"debugging"` make it easier to find runs later.

4. **Log early, log often.** Once you have logging set up, use it for all
   experiments—even quick tests. You never know which one will be important.

5. **Don't log secrets.** Hyperparameters are fine. API keys and credentials
   are not.

## Next steps

- [The Experiment Class](experiment.md) — Full API reference
- [Hyperparameter Search](sklearn-search.md) — Track all search trials
- [Optuna Search](optuna-search.md) — Advanced search with logging
