# Bayesian Optimization with Optuna

**What you'll learn:**

- Why Bayesian optimization outperforms random search for expensive models
- How Optuna's TPE sampler works at a conceptual level
- When to choose Optuna over simpler search strategies
- How to use OptunaConfig and custom searchers with eksperiment

**Prerequisites:** [Hyperparameter Search](sklearn-search.md), understanding of cross-validation.

## The problem: expensive hyperparameter evaluations

Grid and random search treat every trial independently—each evaluation learns
nothing from previous ones. This works fine when evaluations are cheap (seconds),
but becomes wasteful when training takes minutes or hours.

Imagine searching over 5 hyperparameters with 20 trials. Random search might
waste 10 trials in regions of parameter space that a human could tell are
unpromising after seeing just 3 poor results.

**Bayesian optimization** solves this by learning *during* the search. It builds
a model of which parameter regions are promising, then uses that model to decide
what to try next.

## How Bayesian optimization works

At its core, Bayesian optimization asks: "Given what I've observed so far, which
parameters should I try next?"

It answers this in two steps:

1. **Surrogate model**: Build a probabilistic model of the objective function
   based on observed (params, score) pairs. This model predicts both the
   expected score and the *uncertainty* for any new parameters.

2. **Acquisition function**: Use the surrogate model to decide which parameters
   to try next, balancing:
   - **Exploitation**: Try parameters similar to the best seen so far
   - **Exploration**: Try uncertain regions that might contain better solutions

The magic is in the balance. Pure exploitation gets stuck in local optima.
Pure exploration wastes budget on unpromising regions. Good acquisition
functions find the sweet spot.

## Optuna's TPE sampler

Optuna uses **TPE (Tree-structured Parzen Estimator)** as its default sampler.
TPE differs from classic Gaussian Process-based Bayesian optimization in how
it builds the surrogate model.

### How TPE works

Instead of modeling the objective function directly, TPE models the *density*
of parameters that lead to good vs. bad results:

1. **Split observations**: Divide previous trials into "good" (top γ quantile)
   and "bad" (the rest). Typically γ = 0.25, so top 25% are "good."

2. **Build density estimates**: For each parameter, estimate two probability
   densities:
   - p(param | good): density of parameter values in good trials
   - p(param | bad): density of parameter values in bad trials

3. **Suggest parameters**: Sample parameters that maximize p(good) / p(bad).
   This ratio is related to Expected Improvement, a classic acquisition function.

### Why TPE works well

TPE has several practical advantages:

- **Scales to high dimensions**: Unlike Gaussian Processes, TPE treats each
  parameter independently, avoiding the curse of dimensionality.
- **Handles mixed types**: Naturally supports continuous, integer, and
  categorical parameters without special encoding.
- **Computationally cheap**: Sampling is fast even with many trials.

The tradeoff: TPE may miss complex parameter interactions that a GP would
capture. In practice, this rarely matters—most hyperparameter landscapes
have relatively simple structure.

---

## Quick Optuna search with OptunaConfig

The easiest way to use Optuna with eksperiment is through `OptunaConfig`:

```python
import pytest
pytest.importorskip("optuna")

import optuna

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eksperiment.experiment import Experiment
from eksperiment.optuna import OptunaConfig

optuna.logging.set_verbosity(optuna.logging.WARNING)

X, y = load_iris(return_X_y=True)

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=200)),
])

# Define a search space function
def search_space(trial):
    return {
        "model__C": trial.suggest_float("model__C", 1e-3, 1e2, log=True),
    }

experiment = Experiment(
    pipeline=pipeline,
    scorers={"accuracy": "accuracy"},
    name="iris-optuna",
)

result = experiment.search(
    OptunaConfig(search_space=search_space, n_trials=20, direction="maximize"),
    X, y,
    cv=5,
    run_name="optuna-quick",
)

print(f"Best params: {result.best_params}")
print(f"Best score: {result.best_score:.4f}")
```

### Defining the search space

The `search_space` function receives an Optuna `trial` object and returns a
dict of parameters. Use the `trial.suggest_*` methods:

```python
def search_space(trial):
    return {
        # Continuous, log-scale (good for regularization, learning rates)
        "model__C": trial.suggest_float("model__C", 1e-4, 1e2, log=True),

        # Continuous, linear scale
        "model__tol": trial.suggest_float("model__tol", 1e-5, 1e-2),

        # Integer
        "model__max_iter": trial.suggest_int("model__max_iter", 100, 1000),

        # Categorical
        "model__solver": trial.suggest_categorical(
            "model__solver", ["lbfgs", "liblinear", "saga"]
        ),
    }
```

**Tip**: Use `log=True` for parameters that span orders of magnitude. This
ensures equal exploration across the scale.

---

## Custom searcher for full control

For advanced use cases—custom samplers, pruning, multi-objective optimization—
implement a custom searcher:

```python
from dataclasses import dataclass

import pytest
pytest.importorskip("optuna")

import optuna
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eksperiment.experiment import Experiment

optuna.logging.set_verbosity(optuna.logging.WARNING)

@dataclass
class OptunaSearcher:
    """Custom Optuna searcher with full study control."""
    pipeline: Pipeline
    cv: int = 5
    n_trials: int = 20

    best_params_: dict | None = None
    best_score_: float | None = None
    best_estimator_: Pipeline | None = None

    def fit(self, X, y=None):
        def objective(trial):
            params = {
                "model__C": trial.suggest_float("model__C", 1e-3, 1e2, log=True),
            }
            estimator = clone(self.pipeline).set_params(**params)
            score = cross_val_score(
                estimator, X, y,
                scoring="accuracy",
                cv=self.cv,
            ).mean()
            trial.set_user_attr("params", params)
            return score

        # Create study with custom configuration
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=self.n_trials)

        self.best_score_ = float(study.best_value)
        self.best_params_ = study.best_trial.user_attrs["params"]
        self.best_estimator_ = (
            clone(self.pipeline)
            .set_params(**self.best_params_)
            .fit(X, y)
        )
        return self


X, y = load_iris(return_X_y=True)

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=200)),
])

experiment = Experiment(
    pipeline=pipeline,
    scorers={"accuracy": "accuracy"},
    name="iris-optuna-custom",
)

searcher = OptunaSearcher(pipeline=pipeline, cv=5, n_trials=20)
result = experiment.search(searcher, X, y, run_name="optuna-custom")
print(f"Best params: {result.best_params}")
print(f"Best score: {result.best_score:.4f}")
```

---

## When to use Optuna vs. simpler strategies

| Situation | Recommendation |
|-----------|----------------|
| < 50 evaluations possible | Random search (Optuna needs data to learn) |
| Evaluations take seconds | Random search (overhead not worth it) |
| Evaluations take minutes+ | **Optuna** |
| 5+ hyperparameters | **Optuna** |
| Need exact reproducibility | Grid search |
| Complex conditional parameters | **Optuna** (supports dynamic search spaces) |

### The overhead of Bayesian optimization

Optuna adds computational overhead per trial:

- Building density estimates from history
- Sampling and ranking candidates
- Managing study state

For fast models (< 10 seconds), this overhead may exceed the savings from
smarter sampling. For slow models (> 1 minute), the overhead is negligible
compared to training time, and smarter sampling pays off.

### A rule of thumb

**Use Optuna when**: (training time) × (number of trials) > 10 minutes

---

## Comparison: Random search vs. Optuna

Let's compare convergence on a problem with a clear optimal region:

```python
import pytest
pytest.importorskip("optuna")

import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform, randint

from eksperiment.experiment import Experiment
from eksperiment.optuna import OptunaConfig

optuna.logging.set_verbosity(optuna.logging.WARNING)

X, y = load_breast_cancer(return_X_y=True)

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", GradientBoostingClassifier(random_state=42)),
])

experiment = Experiment(
    pipeline=pipeline,
    scorers={"accuracy": "accuracy"},
    name="optuna-vs-random",
)

# Random search: 30 trials
random_searcher = RandomizedSearchCV(
    pipeline,
    param_distributions={
        "model__n_estimators": randint(10, 200),
        "model__learning_rate": loguniform(0.01, 1.0),
        "model__max_depth": randint(2, 10),
    },
    n_iter=30,
    scoring="accuracy",
    cv=5,
    random_state=42,
    refit=True,
)
random_result = experiment.search(random_searcher, X, y, run_name="random-30")
print(f"Random (30 trials): {random_result.best_score:.4f}")
```

```{.python continuation}
# Optuna: 30 trials
def search_space(trial):
    return {
        "model__n_estimators": trial.suggest_int("model__n_estimators", 10, 200),
        "model__learning_rate": trial.suggest_float("model__learning_rate", 0.01, 1.0, log=True),
        "model__max_depth": trial.suggest_int("model__max_depth", 2, 10),
    }

optuna_result = experiment.search(
    OptunaConfig(search_space=search_space, n_trials=30, direction="maximize"),
    X, y,
    cv=5,
    run_name="optuna-30",
)
print(f"Optuna (30 trials): {optuna_result.best_score:.4f}")
```

With enough trials, both methods find good solutions. The difference shows in
*how quickly* they converge—Optuna typically reaches a good solution faster.

---

## Best practices

1. **Set `log=True` for scale parameters.** Learning rates, regularization
   strengths, and kernel widths often span orders of magnitude.

2. **Use enough trials.** TPE needs ~10-20 trials before its suggestions
   become meaningfully better than random. Budget at least 30 trials.

3. **Seed the sampler for reproducibility.** Pass `seed` to `TPESampler` in
   custom searchers.

4. **Check the study history.** Optuna stores all trials; review them to
   understand which parameters matter.

5. **Consider pruning for iterative models.** See [Optuna Advanced](optuna-advanced.md)
   for early stopping unpromising trials.

## Notes

- `OptunaConfig` uses the experiment's first scorer when `scoring` is not set.
- If you pass multiple scorers, OptunaConfig optimizes the first one.
- eksperiment only requires `fit()` and optionally `best_params_`, `best_score_`,
  `best_estimator_` attributes on custom searchers.

## Further reading

- [Optuna Paper](https://arxiv.org/abs/1907.10902) — Original TPE-based framework
- Bergstra et al. (2011), [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization) — TPE algorithm
- [Optuna Documentation](https://optuna.readthedocs.io/) — Official docs
- [Optuna Advanced](optuna-advanced.md) — Custom samplers, pruning, study factories
