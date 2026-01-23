# Hyperparameter Search: From Exhaustive to Intelligent

**What you'll learn:**

- Why hyperparameter tuning matters for model performance
- How grid search works and when it fails
- Why random search often beats grid search
- When to use halving search for expensive models
- How to choose the right strategy for your problem

**Prerequisites:** [Why Pipelines Matter](why-pipelines.md), basic sklearn familiarity.

## The problem: finding good hyperparameters

Most machine learning models have hyperparameters—settings that control
learning but aren't learned from data. A decision tree's `max_depth`, a
regularized model's penalty strength `C`, a neural network's learning rate.

These parameters matter enormously. The wrong regularization strength can
cause underfitting (too strong) or overfitting (too weak). Default values
are reasonable starting points, but rarely optimal for your specific data.

Hyperparameter search systematically explores the parameter space to find
better configurations. sklab's `search()` method wraps various search
strategies with consistent logging.

---

## Strategy 1: Grid search

Grid search is the simplest approach: specify a set of values for each
parameter, and try every combination.

### How grid search works

Given these parameter values:

- `C`: [0.1, 1.0, 10.0]
- `gamma`: [0.01, 0.1]

Grid search evaluates all 6 combinations:

```
(C=0.1, gamma=0.01), (C=0.1, gamma=0.1),
(C=1.0, gamma=0.01), (C=1.0, gamma=0.1),
(C=10.0, gamma=0.01), (C=10.0, gamma=0.1)
```

**Complexity:** O(∏ᵢ |Vᵢ|) where Vᵢ is the set of values for parameter i.

### Grid search with sklab

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklab.experiment import Experiment
from sklab.search import GridSearchConfig

X, y = load_iris(return_X_y=True)

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=200)),
])

experiment = Experiment(
    pipeline=pipeline,
    scoring="accuracy",
    name="iris-grid",
)

# Grid search over regularization strength
result = experiment.search(
    GridSearchConfig(
        param_grid={"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        refit=True,
    ),
    X, y,
    cv=5,
    run_name="grid-search",
)

print(f"Best params: {result.best_params}")
print(f"Best score: {result.best_score:.4f}")
```

### When grid search works

- **Small parameter spaces:** 2-3 parameters with a few values each
- **Need exact reproducibility:** Grid search is deterministic
- **All parameters matter equally:** Grid allocates equal attention to each

### When grid search fails: the curse of dimensionality

The curse of dimensionality kills grid search in high dimensions:

| Parameters | Values each | Total combinations |
|------------|-------------|-------------------|
| 2 | 10 | 100 |
| 3 | 10 | 1,000 |
| 5 | 10 | 100,000 |
| 10 | 10 | 10,000,000,000 |

With expensive model training, exhaustive search becomes intractable.

---

## Strategy 2: Random search

Random search samples parameters independently from specified distributions.

### How random search works

Instead of a grid, you define distributions:

- `C`: log-uniform(0.01, 100)
- `gamma`: log-uniform(0.001, 1)

Each trial draws random values. After n trials, you keep the best.

### The key insight: why random beats grid

In 2012, Bergstra and Bengio published a surprising result: random search
often finds better hyperparameters than grid search with the same budget.

**Why?** In most problems, only a few parameters actually matter. If 2 of 10
parameters drive performance, grid search wastes most of its budget on
irrelevant dimensions. Random search samples the important dimensions densely
regardless of how many unimportant dimensions exist.

Consider a 2D search when only one dimension matters:

```
Grid (9 points):          Random (9 points):
x x x                     x   x    x
x x x                       x   x
x x x                     x    x  x x

Grid tests 3 unique        Random tests 9 unique
values on the important    values on the important
dimension.                 dimension.
```

The grid wastes 6 evaluations testing the same 3 values repeatedly. Random
search explores 9 unique values on the dimension that matters.

### Random search with sklab

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform

from sklab.experiment import Experiment

X, y = load_iris(return_X_y=True)

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=200)),
])

# Define distributions instead of fixed values
param_distributions = {
    "model__C": loguniform(0.01, 100),  # log-uniform from 0.01 to 100
}

searcher = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=20,  # number of random samples
    scoring="accuracy",
    cv=5,
    random_state=42,
    refit=True,
)

experiment = Experiment(
    pipeline=pipeline,
    scoring="accuracy",
    name="iris-random",
)

result = experiment.search(searcher, X, y, run_name="random-search")
print(f"Best params: {result.best_params}")
print(f"Best score: {result.best_score:.4f}")
```

### When to use random search

- **Medium-to-high dimensional spaces:** When you can't afford exhaustive search
- **Cheap evaluations:** When you can afford many trials
- **Low effective dimensionality:** When only a few parameters matter (common)

---

## Strategy 3: Halving search

Halving search is a budget-aware strategy that quickly discards unpromising
candidates.

### How halving search works

The idea: don't give every configuration a full evaluation. Start with many
candidates using small budgets (few samples, few iterations), then
progressively increase the budget while keeping only the best performers.

1. Start with n candidates, evaluate each with budget b
2. Keep the top 1/factor candidates
3. Multiply budget by factor
4. Repeat until one candidate remains

This is related to the **successive halving** algorithm, which inspired
Hyperband (used in Optuna's pruning).

### Halving search with sklab

```python
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform

from sklab.experiment import Experiment

X, y = load_iris(return_X_y=True)

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=200)),
])

param_distributions = {
    "model__C": loguniform(0.01, 100),
}

searcher = HalvingRandomSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_candidates=16,  # start with 16 candidates
    factor=2,         # halve candidates each round
    scoring="accuracy",
    cv=3,
    random_state=42,
    refit=True,
)

experiment = Experiment(
    pipeline=pipeline,
    scoring="accuracy",
    name="iris-halving",
)

result = experiment.search(searcher, X, y, run_name="halving-search")
print(f"Best params: {result.best_params}")
print(f"Best score: {result.best_score:.4f}")
```

### When halving search works

- **Large candidate pools:** When you have many configurations to try
- **Scalable budget:** When you can meaningfully vary training budget
  (more data, more iterations, more trees)
- **Early differentiation:** When bad configurations show poor performance
  early

---

## Comparison: same problem, three strategies

Let's compare all three strategies on the same dataset to see how they behave.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingRandomSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint

from sklab.experiment import Experiment

X, y = load_breast_cancer(return_X_y=True)

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", RandomForestClassifier(random_state=42)),
])

experiment = Experiment(
    pipeline=pipeline,
    scoring="accuracy",
    name="search-comparison",
)

# Define the search space
param_grid = {
    "model__n_estimators": [10, 50, 100],
    "model__max_depth": [3, 5, 10, None],
    "model__min_samples_split": [2, 5, 10],
}

param_distributions = {
    "model__n_estimators": randint(10, 150),
    "model__max_depth": [3, 5, 10, 20, None],
    "model__min_samples_split": randint(2, 20),
}
```

```{.python continuation}
# Grid search: 3 × 4 × 3 = 36 combinations
grid_searcher = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    refit=True,
)
grid_result = experiment.search(grid_searcher, X, y, run_name="grid")
print(f"Grid: {grid_result.best_score:.4f} (36 combinations)")
```

```{.python continuation}
# Random search: 20 random samples
random_searcher = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=20,
    scoring="accuracy",
    cv=5,
    random_state=42,
    refit=True,
)
random_result = experiment.search(random_searcher, X, y, run_name="random")
print(f"Random: {random_result.best_score:.4f} (20 samples)")
```

```{.python continuation}
# Halving: starts with many, progressively eliminates
halving_searcher = HalvingRandomSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_candidates=32,
    factor=2,
    scoring="accuracy",
    cv=3,
    random_state=42,
    refit=True,
)
halving_result = experiment.search(halving_searcher, X, y, run_name="halving")
print(f"Halving: {halving_result.best_score:.4f} (32 initial candidates)")
```

---

## Bring your own searcher

sklab doesn't lock you into specific searchers. Any object that conforms to
the Searcher protocol (structural typing, no inheritance required) and exposes
`fit(X, y)`, `best_params_`, `best_score_`, and `best_estimator_` works.

```python
from dataclasses import dataclass
import random

from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklab.experiment import Experiment

@dataclass
class SimpleRandomSearch:
    """A minimal random searcher for demonstration."""
    estimator: Pipeline
    param_sampler: callable  # function that returns random params
    n_iter: int = 10
    cv: int = 5
    scoring: str = "accuracy"

    best_params_: dict | None = None
    best_score_: float | None = None
    best_estimator_: Pipeline | None = None

    def fit(self, X, y=None):
        best_score = float("-inf")
        best_params = None

        for _ in range(self.n_iter):
            params = self.param_sampler()
            estimator = clone(self.estimator).set_params(**params)
            score = cross_val_score(
                estimator, X, y,
                scoring=self.scoring,
                cv=self.cv,
            ).mean()

            if score > best_score:
                best_score = score
                best_params = params

        self.best_params_ = best_params
        self.best_score_ = float(best_score)
        self.best_estimator_ = (
            clone(self.estimator)
            .set_params(**best_params)
            .fit(X, y)
        )
        return self

X, y = load_iris(return_X_y=True)

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=200)),
])

def sample_params():
    return {"model__C": 10 ** random.uniform(-2, 2)}

searcher = SimpleRandomSearch(
    estimator=pipeline,
    param_sampler=sample_params,
    n_iter=15,
)

experiment = Experiment(
    pipeline=pipeline,
    scoring="accuracy",
    name="custom-searcher",
)

result = experiment.search(searcher, X, y, run_name="custom")
print(f"Best params: {result.best_params}")
print(f"Best score: {result.best_score:.4f}")
```

---

## Accessing the underlying searcher

The `SearchResult` returned by `experiment.search()` exposes the underlying
sklearn searcher via the `.raw` attribute. This gives you access to detailed
cross-validation results, timing information, and other sklearn-specific data.

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklab.experiment import Experiment
from sklab.search import GridSearchConfig

X, y = load_iris(return_X_y=True)

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=200)),
])

experiment = Experiment(
    pipeline=pipeline,
    scoring="accuracy",
    name="iris-raw-access",
)

result = experiment.search(
    GridSearchConfig(
        param_grid={"model__C": [0.01, 0.1, 1.0, 10.0]},
        refit=True,
    ),
    X, y,
    cv=5,
)

# Access the underlying GridSearchCV via .raw
searcher = result.raw
print(f"Number of candidates: {len(searcher.cv_results_['params'])}")
print(f"Best index: {searcher.best_index_}")
```

### Inspecting CV results

The `cv_results_` attribute contains detailed information about every
parameter combination tested:

```{.python continuation}
# View results for each candidate
for i, params in enumerate(searcher.cv_results_["params"]):
    mean_score = searcher.cv_results_["mean_test_score"][i]
    std_score = searcher.cv_results_["std_test_score"][i]
    print(f"{params}: {mean_score:.4f} (+/- {std_score:.4f})")
```

For more convenient analysis, convert to a DataFrame:

```{.python continuation}
import polars as pl

cv_df = pl.DataFrame({
    "C": [p["model__C"] for p in searcher.cv_results_["params"]],
    "mean_score": searcher.cv_results_["mean_test_score"],
    "std_score": searcher.cv_results_["std_test_score"],
    "mean_fit_time": searcher.cv_results_["mean_fit_time"],
})
print(cv_df)
```

This works the same way for `RandomizedSearchCV` and `HalvingRandomSearchCV`.

---

## Decision guide: which strategy to use

| Situation | Recommended Strategy |
|-----------|---------------------|
| Small grid (< 100 combinations) | Grid search |
| Need exact reproducibility | Grid search |
| Medium space, cheap evaluations | Random search |
| High-dimensional space | Random search |
| Large candidate pool, scalable budget | Halving search |
| Expensive evaluations | Optuna (see [Optuna Search](optuna-search.md)) |
| Complex search logic | Custom searcher |

## Best practices

1. **Start with random search.** Unless your space is tiny, random search
   is a safe default that works well across many problems.

2. **Use log-uniform for scale parameters.** Parameters like learning rate,
   regularization strength, and kernel width often span orders of magnitude.
   Log-uniform sampling explores this space more evenly.

3. **Set a budget, not a grid size.** Decide how many evaluations you can
   afford, then choose a strategy that uses that budget well.

4. **Log everything.** sklab logs all search results automatically.
   Review them to understand which parameters matter.

5. **Don't over-tune.** Hyperparameter optimization has diminishing returns.
   If your model is fundamentally wrong, no amount of tuning will save it.

## Notes

- Config classes default to the experiment scoring when `scoring` is not set.
- Config classes use `cv` from `Experiment.search()` unless you set `cv` on the config.
- When you pass a searcher instance directly, `Experiment.search()` will call its
  `fit()` method and log `best_params_`, `best_score_`, and `best_estimator_` if
  the searcher exposes them.
- Use `step__param` names to target pipeline steps (for example, `model__C`).

## Further reading

- Bergstra & Bengio (2012), [Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/v13/bergstra12a.html) — foundational paper on random vs. grid search
- [sklearn User Guide: Tuning hyperparameters](https://scikit-learn.org/stable/modules/grid_search.html)
- [Optuna Search](optuna-search.md) — Bayesian optimization for expensive searches
