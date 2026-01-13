# Glossary

A reference for the core concepts in eksperiment. Each term includes a brief
definition and, where relevant, why it matters for experiment design.

---

## Experiment

A lightweight wrapper around a sklearn pipeline that provides a consistent API
for `fit`, `evaluate`, `cross_validate`, and `search`. The Experiment object
holds your pipeline, scorers, and logger configuration, ensuring that every
operation uses the same settings.

**Why it matters:** Without a central experiment object, it's easy to use
different scorers for training vs. evaluation, forget to log certain runs, or
accidentally change preprocessing between experiments. The Experiment class
prevents these inconsistencies.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eksperiment.experiment import Experiment
from eksperiment.logging.adapters import NoOpLogger

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=200)),
])
mlflow_logger = NoOpLogger()

experiment = Experiment(
    pipeline=pipeline,
    scorers={"accuracy": "accuracy", "f1": "f1"},
    logger=mlflow_logger,
    name="my-experiment",
)
```

---

## Pipeline

A sklearn `Pipeline` that bundles preprocessing and modeling steps into a
single estimator. When you call `fit()` on a pipeline, each step is fit in
sequence. When you call `predict()`, each step transforms the data before
passing it to the next.

**Why it matters:** Pipelines prevent data leakage. If you scale your features
before splitting data, the scaler "sees" test set statistics during training.
A pipeline ensures that preprocessing is refit on each fold's training data
during cross-validation. See [Why Pipelines Matter](tutorials/why-pipelines.md)
for a detailed explanation.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ("scale", StandardScaler()),      # preprocessing
    ("model", LogisticRegression()),  # estimator
])
```

---

## Data Leakage

Information from outside the training set influencing the training process.
Common sources include:

- Fitting a scaler on the full dataset before splitting
- Selecting features based on the full dataset's statistics
- Using future data to predict past values in time series

**Why it matters:** Leakage causes overly optimistic evaluation metrics. Your
model appears to perform well in development but fails in production because
it was "cheating" during evaluation.

**Prevention:** Keep all data-dependent preprocessing inside the pipeline.
eksperiment enforces this by requiring a Pipeline object.

---

## Scorer

A metric definition passed to sklearn. Can be a string (e.g., `"accuracy"`) or
a callable that takes `(y_true, y_pred)` and returns a float. eksperiment
accepts a mapping of names to scorers, and uses them consistently across all
operations.

**Why it matters:** Using different metrics for training, cross-validation,
and final evaluation leads to misleading comparisons. By defining scorers once
on the Experiment, you ensure consistent evaluation everywhere.

```text
scorers = {
    "accuracy": "accuracy",
    "f1": "f1_macro",
    "custom": make_scorer(my_metric_fn),
}
```

**Note:** sklearn's regression metrics like MAE and RMSE are negated by
convention (`neg_mean_absolute_error`). This allows sklearn to maximize scores
uniformly. eksperiment follows this convention.

---

## Cross-Validation

A technique for estimating how well a model generalizes to unseen data. The
dataset is split into k folds; the model is trained on k-1 folds and evaluated
on the remaining fold, rotating through all folds. The final metric is the
average across all folds.

**Why it matters:** A single train/test split is noisy—you might get lucky or
unlucky with the split. Cross-validation averages over multiple splits,
providing a more stable estimate of model performance and its variance.

**Variants:**

| Splitter | Use Case |
|----------|----------|
| `KFold` | General purpose, regression |
| `StratifiedKFold` | Classification (preserves class balance) |
| `TimeSeriesSplit` | Time series (respects temporal order) |
| `GroupKFold` | Grouped data (keeps groups together) |

```text
cv_result = experiment.cross_validate(X, y, cv=5, run_name="cv")
```

---

## Hyperparameter Search

The process of finding optimal hyperparameters—settings that control learning
but aren't learned from data. Examples: regularization strength (`C`), tree
depth, learning rate.

**Why it matters:** Default hyperparameters rarely give best performance.
Search systematically explores the parameter space to find better
configurations. eksperiment's `search()` method wraps various search strategies
with consistent logging.

**Strategies:**

| Strategy | How It Works | When to Use |
|----------|--------------|-------------|
| Grid Search | Exhaustive, tries all combinations | Small spaces, need reproducibility |
| Random Search | Samples randomly from distributions | Medium spaces, cheap evaluations |
| Bayesian (Optuna) | Learns which regions are promising | Large spaces, expensive evaluations |

---

## Searcher

An object that conforms to the Searcher protocol: it provides `fit(X, y)` and
exposes `best_params_`, `best_score_`,
and `best_estimator_` after fitting. Searchers encapsulate hyperparameter
search logic.

**Why it matters:** eksperiment is searcher-agnostic. You can use sklearn's
`GridSearchCV`, Optuna, or your own custom searcher. As long as it follows
the protocol (no inheritance required), eksperiment will log results consistently.

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eksperiment.experiment import Experiment

X, y = load_iris(return_X_y=True)

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=200)),
])
param_grid = {"model__C": [0.1, 1.0, 10.0]}

experiment = Experiment(
    pipeline=pipeline,
    scorers={"accuracy": "accuracy"},
    name="search-demo",
)

searcher = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy")
result = experiment.search(searcher, X, y, run_name="grid-search")
```

---

## Search Config

An object that conforms to the SearchConfig protocol: it provides
`create_searcher(...)` and returns a Searcher. This provides a clean API for
common configurations while allowing full
customization when needed.

**Why it matters:** Configs keep the call site simple. Instead of constructing
a complex searcher manually, you pass a config object and eksperiment handles
the rest.

```text
from eksperiment.search import GridSearchConfig

result = experiment.search(
    GridSearchConfig(param_grid={"model__C": [0.1, 1.0, 10.0]}),
    X, y, cv=3, run_name="search",
)
```

---

## Logger

A backend-agnostic logger that creates runs. It conforms to `LoggerProtocol`
and returns a run handle from `start_run(...)`.

**Why it matters:** Different teams use different experiment tracking tools
(MLflow, Weights & Biases, Neptune, custom databases). eksperiment's logger
protocol lets you swap backends without changing modeling code. No inheritance
is required because it uses protocols.

```python
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from eksperiment.experiment import Experiment
from eksperiment.logging.adapters import MLflowLogger, WandbLogger

pipeline = Pipeline([("model", DummyClassifier(strategy="most_frequent"))])
scorers = {"accuracy": "accuracy"}

experiment = Experiment(
    pipeline=pipeline,
    scorers=scorers,
    logger=MLflowLogger(experiment_name="my-project"),
)
```

---

## Run

A context-managed handle for logging params, metrics, tags, artifacts, and
models. Runs are created by logger adapters and used internally by eksperiment
methods.

**Why it matters:** Runs provide isolation and organization. Each call to
`fit()`, `evaluate()`, `cross_validate()`, or `search()` creates a separate
run with its own logged data. This makes it easy to compare experiments and
reproduce results.

---

## Adapter

A thin wrapper that bridges eksperiment's logging protocols to an external
tool (MLflow, W&B, etc.) without coupling the core API to that SDK.

**Why it matters:** Adapters let eksperiment stay lightweight while supporting
multiple backends. Each adapter translates eksperiment's simple protocol calls
into the specific API of an external tool.

**Built-in adapters:**

- `NoOpLogger` — Logs nothing (default)
- `MLflowLogger` — Logs to MLflow tracking server
- `WandbLogger` — Logs to Weights & Biases

**Custom adapters:** Implement `LoggerProtocol` and `RunProtocol`. See
[Logger Plugins](developer/logging-plugins.md) for details.

---

## Further Reading

- [Why Pipelines Matter](tutorials/why-pipelines.md) — Data leakage explained
- [Hyperparameter Search](tutorials/sklearn-search.md) — Search strategies compared
- [References](references.md) — Academic papers and external resources
