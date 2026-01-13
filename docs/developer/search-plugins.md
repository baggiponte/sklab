# Search plugins

Sklab’s `Experiment.search()` accepts either a searcher object or a config
object that can create one. This gives you a simple path and a power-user path.

## Protocols

- `SearcherProtocol`: must provide `fit(X, y)` and may expose
  `best_params_`, `best_score_`, `best_estimator_`.
- `SearchConfigProtocol`: must provide `create_searcher(...)` and return a
  `SearcherProtocol`.
- These are protocols (structural typing), so inheritance is not required.

## Custom searcher (power user)

```python
from dataclasses import dataclass
from typing import Any

from sklearn.base import clone
from sklearn.model_selection import cross_val_score

@dataclass
class MySearcher:
    estimator: Any
    cv: int = 3

    best_params_: dict | None = None
    best_score_: float | None = None
    best_estimator_: Any | None = None

    def fit(self, X, y=None):
        params = {"model__C": 1.0}
        estimator = clone(self.estimator).set_params(**params)
        score = cross_val_score(estimator, X, y, scoring="accuracy", cv=self.cv).mean()

        self.best_params_ = params
        self.best_score_ = float(score)
        self.best_estimator_ = estimator.fit(X, y)
        return self
```

## Config wrapper (clean API)

```{.python continuation}
from dataclasses import dataclass

@dataclass
class MySearchConfig:
    def create_searcher(self, *, pipeline, scorers, cv, n_trials, timeout):
        return MySearcher(estimator=pipeline, cv=cv or 3)
```

## Best practices

- Use `cv` and `scorers` from Experiment when possible.
- Keep searcher state on the instance (`best_*` attributes).
- Favor reproducibility (set seeds, track versions).

## Optuna custom searcher

Use a custom searcher when you need Optuna features beyond the quick config.

```python
import optuna
from dataclasses import dataclass
from typing import Any

from sklearn.base import clone
from sklearn.model_selection import cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

@dataclass
class OptunaSearcher:
    pipeline: Any
    cv: int = 3
    n_trials: int = 6

    best_params_: dict | None = None
    best_score_: float | None = None
    best_estimator_: Any | None = None

    def fit(self, X, y=None):
        def objective(trial):
            params = {
                "model__C": trial.suggest_float("model__C", 1e-3, 1e2, log=True),
            }
            estimator = clone(self.pipeline).set_params(**params)
            score = cross_val_score(
                estimator,
                X,
                y,
                scoring="accuracy",
                cv=self.cv,
            ).mean()
            trial.set_user_attr("params", params)
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)

        self.best_score_ = float(study.best_value)
        self.best_params_ = dict(study.best_trial.user_attrs["params"])
        self.best_estimator_ = (
            clone(self.pipeline)
            .set_params(**self.best_params_)
            .fit(X, y)
        )
        return self
```
