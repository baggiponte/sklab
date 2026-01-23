# sklab

sklab is a lightweight experiment runner for sklearn pipelines. It keeps
modeling code focused on data and pipelines while standardizing the fit/evaluate
loop, logging, and hyperparameter search workflows.

## Why sklab?

Machine learning code tends to accumulate boilerplate: fitting models, computing
metrics, logging results, running cross-validation, searching hyperparameters.
Each project reinvents this wheel slightly differently, making code harder to
review, test, and reproduce.

sklab solves this by providing a thin, opinionated wrapper around sklearn
that enforces good practices while staying out of your way.

### The problem it solves

Consider a typical ML workflow without sklab:

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
import mlflow

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
model = LogisticRegression(max_iter=200)
param_grid = {"C": [0.1, 1.0, 10.0]}

# Fit
model.fit(X_train, y_train)

# Evaluate (hope you remembered all the metrics)
y_pred = model.predict(X_test)
print("accuracy:", accuracy_score(y_test, y_pred))
print("f1:", f1_score(y_test, y_pred, average="macro"))
acc = accuracy_score(y_test, y_pred)

# Cross-validate (different API)
scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print("cv mean:", scores.mean())

# Search (yet another API)
grid = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
grid.fit(X, y)
print("best:", grid.best_params_)

# Log somewhere (MLflow? W&B? CSV?)
mlflow.log_metric("accuracy", acc)
mlflow.log_params(grid.best_params_)
mlflow.end_run()  # Don't forget to clean up!
```

Each operation has a different API. Logging is tightly coupled to a specific
backend. Metrics are computed ad-hoc. It's easy to forget a step or log
inconsistently.

### How sklab helps

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklab.experiment import Experiment
from sklab.logging import NoOpLogger
from sklab.search import GridSearchConfig

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=200)),
])
config = GridSearchConfig(
    param_grid={"model__C": [0.1, 1.0, 10.0]},
    refit="accuracy",
)
mlflow_logger = NoOpLogger()

experiment = Experiment(
    pipeline=pipeline,
    scoring=["accuracy", "f1_macro"],
    logger=mlflow_logger,  # or wandb, or none
    name="my-experiment",
)

# Consistent API for everything
experiment.fit(X_train, y_train, run_name="fit")
eval_result = experiment.evaluate(X_test, y_test, run_name="eval")
cv_result = experiment.cross_validate(X, y, cv=5, run_name="cv")
search_result = experiment.search(config, X, y, cv=5, run_name="search")
```

Every method:

- Uses the same scoring you defined once
- Logs automatically to whatever backend you configured
- Returns structured results you can inspect programmatically
- Works with any sklearn-compatible pipeline

### Core benefits

**1. Standardized workflow**

One API for fit, evaluate, cross-validate, and search. Define your scoring once;
they're used consistently everywhere.

**2. Backend-agnostic logging**

Swap between MLflow, Weights & Biases, or no logging at all without changing
your modeling code. sklab uses a simple protocol—write your own adapter
in 20 lines if needed.

**3. Pipeline-first design**

sklab requires sklearn Pipelines, not raw estimators. This isn't
arbitrary—pipelines prevent data leakage by keeping preprocessing inside the
cross-validation loop. If you're not using pipelines, you're probably leaking.

**4. Searchable, comparable runs**

Every run logs its parameters, metrics, and tags consistently. Compare
experiments across time, across team members, across hyperparameter
configurations.

**5. Testable tutorials**

Because sklab tutorials are runnable Python, they double as integration
tests. Your documentation stays in sync with your code.

## Quick example

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklab.experiment import Experiment

X, y = load_iris(return_X_y=True)

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=200)),
])

experiment = Experiment(
    pipeline=pipeline,
    scoring="accuracy",
    name="iris-quickstart",
)

# Fit and evaluate
fit_result = experiment.fit(X, y, run_name="fit")
print("fit complete")

# Cross-validate for robust estimation
cv_result = experiment.cross_validate(X, y, cv=3, run_name="cv")
print(f"CV accuracy: {cv_result.metrics['cv/accuracy_mean']:.3f}")
```

## What's next

- [Tutorials](tutorials/experiment.md): Learn sklab step by step
- [API Reference](api/index.md): Detailed method documentation
- [Glossary](glossary.md): Core concepts explained

## Installation

```bash
pip install sklab
```

For Optuna support:

```bash
pip install sklab[optuna]
```
