# Quickstart: The Experiment Class

**What you'll learn:**

- The four core operations: `fit`, `evaluate`, `cross_validate`, `search`
- When to use each operation
- How sklab standardizes the ML workflow

**Time to complete:** 5 minutes.

## The Experiment class at a glance

The `Experiment` class is the heart of sklab. It wraps a sklearn pipeline
with consistent scoring, logging, and methods for the full ML lifecycle:

| Method | Purpose | When to use |
|--------|---------|-------------|
| `fit()` | Train the pipeline | Initial training, final model |
| `evaluate()` | Score on held-out data | Holdout evaluation |
| `cross_validate()` | k-fold cross-validation | Robust performance estimate |
| `search()` | Hyperparameter search | Finding better configurations |

Each method logs results automatically using the configured logger (MLflow,
W&B, or no-op by default).

---

## Setup: Create an experiment

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklab.experiment import Experiment

X, y = load_iris(return_X_y=True)

# Build a pipeline
pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=200)),
])

# Create the experiment
experiment = Experiment(
    pipeline=pipeline,
    scorers={"accuracy": "accuracy"},
    name="quickstart",
)
```

**What this does:**

- `pipeline`: The sklearn pipeline to train and evaluate
- `scorers`: A dict of metric names to sklearn scorer strings or callables
- `name`: A human-readable name for logging and identification

---

## Operation 1: `fit()`

Train the pipeline on your data.

```{.python continuation}
fit_result = experiment.fit(X, y, run_name="fit")

print(f"Estimator type: {type(fit_result.estimator).__name__}")
print(f"Logged params: {fit_result.params}")
```

**Returns:** `FitResult` with:
- `estimator`: The fitted pipeline (cloned from original)
- `params`: Parameters logged for the run

**Use when:** Training on your full training set, or training a final model.

---

## Operation 2: `evaluate()`

Score a fitted model on held-out data.

```{.python continuation}
# In practice, you'd evaluate on a separate test set
# Here we reuse the data for demonstration
eval_result = experiment.evaluate(
    fit_result.estimator,
    X, y,
    run_name="eval",
)

print(f"Metrics: {eval_result.metrics}")
```

**Returns:** `EvalResult` with:
- `metrics`: Dict of metric names to values

**Use when:** Evaluating on a holdout test set after training.

---

## Operation 3: `cross_validate()`

Get a robust performance estimate by training and evaluating across multiple
folds.

```{.python continuation}
cv_result = experiment.cross_validate(
    X, y,
    cv=5,  # 5-fold cross-validation
    run_name="cv",
)

print(f"CV accuracy: {cv_result.metrics['cv/accuracy_mean']:.4f}")
print(f"CV std: {cv_result.metrics['cv/accuracy_std']:.4f}")
```

**Returns:** `CVResult` with:
- `metrics`: Mean and std for each scorer (prefixed with `cv/`)
- `fold_metrics`: Per-fold scores

**Use when:** Estimating model performance, comparing model variants, model
selection.

### Cross-validation variants

sklab accepts any sklearn splitter:

```{.python continuation}
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

# Stratified (preserves class balance) - good for classification
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
strat_result = experiment.cross_validate(X, y, cv=skf, run_name="strat-cv")
print(f"Stratified CV: {strat_result.metrics['cv/accuracy_mean']:.4f}")
```

```{.python continuation}
# Time series split - respects temporal order
import numpy as np

rng = np.random.default_rng(42)
X_ts = np.arange(100).reshape(-1, 1)
y_ts = np.sin(X_ts[:, 0] / 10) + rng.normal(0, 0.1, size=100)

ts_experiment = Experiment(
    pipeline=Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=200)),
    ]),
    scorers={"accuracy": "accuracy"},
    name="ts-demo",
)

# Binarize for classification demo
y_ts_binary = (y_ts > 0).astype(int)

tscv = TimeSeriesSplit(n_splits=3)
ts_result = ts_experiment.cross_validate(X_ts, y_ts_binary, cv=tscv, run_name="ts-cv")
print(f"Time series CV: {ts_result.metrics['cv/accuracy_mean']:.4f}")
```

> **Concept: Choosing a Splitter**
>
> - **Classification:** Use `StratifiedKFold` to preserve class balance
> - **Regression:** Use `KFold` (or pass an integer like `cv=5`)
> - **Time series:** Use `TimeSeriesSplit` to avoid using future data to predict past
> - **Grouped data:** Use `GroupKFold` to keep groups together

---

## Operation 4: `search()`

Find better hyperparameters by searching over a parameter space.

```{.python continuation}
from sklab.search import GridSearchConfig

search_result = experiment.search(
    GridSearchConfig(param_grid={"model__C": [0.1, 1.0, 10.0]}),
    X, y,
    cv=3,
    run_name="search",
)

print(f"Best params: {search_result.best_params}")
print(f"Best score: {search_result.best_score:.4f}")
```

**Returns:** `SearchResult` with:
- `best_params`: Best parameter combination found
- `best_score`: Score achieved by best params
- `best_estimator`: Fitted pipeline with best params (if `refit=True`)

**Use when:** Tuning hyperparameters, exploring the parameter space.

### Search options

sklab supports multiple search strategies:

```{.python continuation}
# Grid search via config
from sklab.search import GridSearchConfig

grid_result = experiment.search(
    GridSearchConfig(param_grid={"model__C": [0.1, 1.0, 10.0]}),
    X, y, cv=3, run_name="grid",
)
```

```{.python continuation}
# Random search via sklearn searcher
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

random_searcher = RandomizedSearchCV(
    pipeline,
    param_distributions={"model__C": loguniform(0.01, 100)},
    n_iter=10,
    cv=3,
    random_state=42,
    refit=True,
)
random_result = experiment.search(random_searcher, X, y, run_name="random")
```

See [Hyperparameter Search](sklearn-search.md) for a complete guide to search
strategies.

---

## Multiple scorers

Track multiple metrics simultaneously:

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

# Define multiple scorers
experiment = Experiment(
    pipeline=pipeline,
    scorers={
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "precision": "precision_macro",
    },
    name="multi-scorer",
)

cv_result = experiment.cross_validate(X, y, cv=5, run_name="multi-cv")

for key, value in cv_result.metrics.items():
    print(f"{key}: {value:.4f}")
```

---

## Putting it all together

A typical workflow combines these operations:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklab.experiment import Experiment
from sklab.search import GridSearchConfig

# 1. Load and split data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Create experiment
pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=200)),
])

experiment = Experiment(
    pipeline=pipeline,
    scorers={"accuracy": "accuracy"},
    name="full-workflow",
)

# 3. Cross-validate to estimate baseline performance
cv_result = experiment.cross_validate(X_train, y_train, cv=5, run_name="baseline-cv")
print(f"Baseline CV: {cv_result.metrics['cv/accuracy_mean']:.4f}")

# 4. Search for better hyperparameters
search_result = experiment.search(
    GridSearchConfig(param_grid={"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]}),
    X_train, y_train,
    cv=5,
    run_name="search",
)
print(f"Best params: {search_result.best_params}")
print(f"Search CV: {search_result.best_score:.4f}")

# 5. Final evaluation on holdout
if search_result.estimator is not None:
    eval_result = experiment.evaluate(
        search_result.estimator,
        X_test, y_test,
        run_name="final-eval",
    )
    print(f"Holdout accuracy: {eval_result.metrics['accuracy']:.4f}")
```

---

## Next steps

- [Classification Workflow](basics-classification-iris.md) — Detailed classification tutorial
- [Why Pipelines Matter](why-pipelines.md) — Understanding data leakage
- [Hyperparameter Search](sklearn-search.md) — Grid, random, and halving search
- [Bayesian Optimization](optuna-search.md) — Intelligent search with Optuna
- [Logger Adapters](logger-adapters.md) — Track experiments with MLflow or W&B
