# Optuna Advanced: Pruning and Custom Samplers

**What you'll learn:**

- How pruning saves compute by stopping unpromising trials early
- How Hyperband and successive halving work
- When and how to use different samplers
- How to configure studies for reproducibility and persistence

**Prerequisites:** [Bayesian Optimization with Optuna](optuna-search.md).

## The problem: wasting compute on bad trials

When training expensive models (deep networks, gradient boosting with many
trees), most hyperparameter configurations are clearly bad. You can often
tell after a few epochs or iterations that a configuration won't compete
with the best ones.

Standard Bayesian optimization runs every trial to completion—even the ones
that are obviously failing. This wastes significant compute.

**Pruning** solves this by stopping unpromising trials early based on
intermediate results.

---

## How pruning works

Pruning requires a model that reports intermediate results during training—
for example, validation loss after each epoch.

The basic idea:

1. At each checkpoint (epoch, iteration), report the current score
2. The pruner compares this intermediate score to other trials at the same step
3. If the trial is clearly behind, stop it early

### A simple example

Imagine 5 trials training for 100 epochs each. After epoch 10:

| Trial | Score at epoch 10 |
|-------|-------------------|
| A | 0.75 |
| B | 0.82 |
| C | 0.61 |
| D | 0.79 |
| E | 0.58 |

Trials C and E are far behind. A good pruner would stop them, saving 90%
of their remaining compute.

---

## Hyperband: principled early stopping

Optuna's `HyperbandPruner` implements **Hyperband**, a principled algorithm
for early stopping.

### How Hyperband works

Hyperband combines two ideas:

1. **Successive halving**: Run many configurations with small budgets.
   Keep the best half, double their budget, repeat until one remains.

2. **Multiple brackets**: Since successive halving is sensitive to the
   initial budget, Hyperband runs multiple "brackets" with different
   starting budgets and halving rates.

The result: configurations get budgets proportional to their promise.
Good ones get full training; bad ones get stopped quickly.

### The successive halving algorithm

Given n configurations and budget B:

```
1. Train all n configurations for B/n resources each
2. Keep the top n/2 configurations
3. Double their budget (now B/n × 2 each)
4. Repeat until 1 configuration remains
```

For example, with n=16 configurations and total budget B=16:

| Round | Configs | Budget each | Total resources |
|-------|---------|-------------|-----------------|
| 1 | 16 | 1 | 16 |
| 2 | 8 | 2 | 16 |
| 3 | 4 | 4 | 16 |
| 4 | 2 | 8 | 16 |
| 5 | 1 | 16 | 16 |

Total resources: 80 (vs. 256 for running all 16 to completion)

### Hyperband brackets

The catch: successive halving's performance depends on the initial budget.
Too small, and you prune good configurations that start slow. Too large,
and you waste budget on bad ones.

Hyperband hedges by running multiple "brackets" with different initial
budgets:

- Bracket 0: Start with many configs, small initial budget (aggressive)
- Bracket 1: Fewer configs, larger initial budget (conservative)
- Bracket 2: Even fewer configs, even larger budget (very conservative)

This ensures good performance regardless of whether your objective
converges quickly or slowly.

---

## Using pruners with sklab

### With OptunaConfig and study_factory

```python
import optuna

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklab.experiment import Experiment
from sklab.search import OptunaConfig

optuna.logging.set_verbosity(optuna.logging.WARNING)

X, y = load_iris(return_X_y=True)

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=200)),
])

def search_space(trial):
    return {
        "model__C": trial.suggest_float("model__C", 1e-3, 1e2, log=True),
    }

# Configure sampler and pruner
sampler = optuna.samplers.TPESampler(seed=42)
pruner = optuna.pruners.HyperbandPruner(
    min_resource=1,      # minimum budget (e.g., epochs)
    max_resource=100,    # maximum budget
    reduction_factor=3,  # keep top 1/3 each round
)

def study_factory(*, direction):
    return optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
    )

experiment = Experiment(
    pipeline=pipeline,
    scoring="accuracy",
    name="optuna-pruning",
)

config = OptunaConfig(
    search_space=search_space,
    n_trials=20,
    direction="maximize",
    study_factory=study_factory,
)

result = experiment.search(config, X, y, cv=3, run_name="pruning-demo")
print(f"Best params: {result.best_params}")
print(f"Best score: {result.best_score:.4f}")
```

### Custom searcher with pruning integration

For models that support incremental training (e.g., neural networks,
gradient boosting with warm_start), you can report intermediate scores:

```python
import optuna
from dataclasses import dataclass
import numpy as np

from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklab.experiment import Experiment

optuna.logging.set_verbosity(optuna.logging.WARNING)

@dataclass
class PruningSearcher:
    """Searcher with Hyperband pruning for iterative models."""
    pipeline: Pipeline
    cv: int = 3
    n_trials: int = 20
    max_n_estimators: int = 100

    best_params_: dict | None = None
    best_score_: float | None = None
    best_estimator_: Pipeline | None = None

    def fit(self, X, y=None):
        def objective(trial):
            # Suggest hyperparameters
            params = {
                "model__learning_rate": trial.suggest_float(
                    "model__learning_rate", 0.01, 0.3, log=True
                ),
                "model__max_depth": trial.suggest_int("model__max_depth", 2, 8),
            }

            # Train incrementally, reporting progress
            for n_estimators in [10, 25, 50, 75, 100]:
                if n_estimators > self.max_n_estimators:
                    break

                full_params = {**params, "model__n_estimators": n_estimators}
                estimator = clone(self.pipeline).set_params(**full_params)

                score = cross_val_score(
                    estimator, X, y,
                    scoring="accuracy",
                    cv=self.cv,
                ).mean()

                # Report intermediate score for pruning decision
                trial.report(score, step=n_estimators)

                # Check if we should stop this trial
                if trial.should_prune():
                    raise optuna.TrialPruned()

            trial.set_user_attr("params", full_params)
            return score

        # Create study with Hyperband pruner
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=10,
                max_resource=self.max_n_estimators,
                reduction_factor=2,
            ),
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


X, y = load_breast_cancer(return_X_y=True)

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", GradientBoostingClassifier(random_state=42)),
])

experiment = Experiment(
    pipeline=pipeline,
    scoring="accuracy",
    name="pruning-custom",
)

searcher = PruningSearcher(pipeline=pipeline, cv=3, n_trials=30)
result = experiment.search(searcher, X, y, run_name="pruning-custom")
print(f"Best params: {result.best_params}")
print(f"Best score: {result.best_score:.4f}")
```

---

## Available pruners

Optuna provides several pruning strategies:

| Pruner | Strategy | Best For |
|--------|----------|----------|
| `MedianPruner` | Prune if below median at step | Simple, robust default |
| `HyperbandPruner` | Successive halving with brackets | Most iterative models |
| `PercentilePruner` | Prune if below percentile | Aggressive pruning |
| `SuccessiveHalvingPruner` | Single successive halving | When Hyperband is overkill |
| `ThresholdPruner` | Prune if below fixed threshold | Known minimum acceptable |
| `NopPruner` | Never prune | Baseline comparison |

### Choosing a pruner

- **Start with `HyperbandPruner`** for iterative models—it's robust across
  different convergence patterns.
- **Use `MedianPruner`** for a simpler baseline that works well in practice.
- **Use `ThresholdPruner`** when you know the minimum acceptable score.

---

## Custom samplers

While TPE is the default, Optuna offers other sampling strategies:

### Available samplers

| Sampler | Strategy | Best For |
|---------|----------|----------|
| `TPESampler` | Tree-structured Parzen Estimator | Default, general purpose |
| `CmaEsSampler` | Covariance Matrix Adaptation | Continuous parameters only |
| `RandomSampler` | Uniform random | Baseline comparison |
| `GridSampler` | Exhaustive grid | Small discrete spaces |
| `NSGAIISampler` | Multi-objective evolutionary | Pareto optimization |

### CMA-ES for continuous spaces

**CMA-ES (Covariance Matrix Adaptation Evolution Strategy)** is a powerful
optimizer for continuous parameter spaces. It adapts a multivariate normal
distribution to the objective landscape.

```python
import optuna
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklab.experiment import Experiment
from sklab.search import OptunaConfig

optuna.logging.set_verbosity(optuna.logging.WARNING)

X, y = load_iris(return_X_y=True)

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", SVC()),
])

def search_space(trial):
    return {
        "model__C": trial.suggest_float("model__C", 1e-3, 1e3, log=True),
        "model__gamma": trial.suggest_float("model__gamma", 1e-4, 1e1, log=True),
    }

# CMA-ES works well for continuous spaces
sampler = optuna.samplers.CmaEsSampler(seed=42)

def study_factory(*, direction):
    return optuna.create_study(direction=direction, sampler=sampler)

experiment = Experiment(
    pipeline=pipeline,
    scoring="accuracy",
    name="cmaes-demo",
)

config = OptunaConfig(
    search_space=search_space,
    n_trials=30,
    direction="maximize",
    study_factory=study_factory,
)

result = experiment.search(config, X, y, cv=5, run_name="cmaes")
print(f"Best params: {result.best_params}")
print(f"Best score: {result.best_score:.4f}")
```

### When to use CMA-ES

- All parameters are continuous (no categoricals or integers)
- Search space is relatively low-dimensional (< 10 parameters)
- You have budget for ~50+ trials

---

## Study persistence and resumption

For long-running searches, persist the study to disk or database:

```python
import optuna
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklab.experiment import Experiment
from sklab.search import OptunaConfig

optuna.logging.set_verbosity(optuna.logging.WARNING)

X, y = load_iris(return_X_y=True)

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=200)),
])

def search_space(trial):
    return {
        "model__C": trial.suggest_float("model__C", 1e-3, 1e2, log=True),
    }

# Persist to SQLite (can resume if interrupted)
storage = "sqlite:///optuna_study.db"
study_name = "iris-persistent"

def study_factory(*, direction):
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        load_if_exists=True,  # Resume if study exists
        sampler=optuna.samplers.TPESampler(seed=42),
    )

experiment = Experiment(
    pipeline=pipeline,
    scoring="accuracy",
    name="persistent-study",
)

config = OptunaConfig(
    search_space=search_space,
    n_trials=10,
    direction="maximize",
    study_factory=study_factory,
)

result = experiment.search(config, X, y, cv=3, run_name="persistent")
print(f"Best params: {result.best_params}")
print(f"Best score: {result.best_score:.4f}")
```

---

## Best practices

1. **Use pruning for iterative models.** Neural networks, gradient boosting,
   and anything with epochs benefits from early stopping.

2. **Match pruner to convergence.** Hyperband handles varying convergence
   speeds well; MedianPruner is simpler but less adaptive.

3. **Seed your samplers.** For reproducibility, always pass `seed` to
   samplers.

4. **Persist long studies.** Use SQLite storage for studies that might be
   interrupted.

5. **Start simple.** TPESampler with MedianPruner is a robust default.
   Only switch to CMA-ES or Hyperband if you have specific needs.

## Decision guide

| Need | Solution |
|------|----------|
| Basic Optuna search | TPESampler (default) |
| Iterative model, budget-conscious | HyperbandPruner |
| Continuous parameters, low-dim | CmaEsSampler |
| Resume interrupted search | SQLite storage |
| Multi-objective optimization | NSGAIISampler |

## Further reading

- Li et al. (2018), [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://arxiv.org/abs/1603.06560) — Hyperband algorithm
- Jamieson & Talwalkar (2016), [Non-stochastic Best Arm Identification and Hyperparameter Optimization](https://arxiv.org/abs/1502.07943) — Successive halving theory
- [Optuna Pruners Documentation](https://optuna.readthedocs.io/en/stable/reference/pruners.html)
- [Optuna Samplers Documentation](https://optuna.readthedocs.io/en/stable/reference/samplers.html)
