---
date: 2026-01-13
title: Simplified Logger Protocol
description: Remove adapter classes, use @contextmanager to wrap native run lifecycle
tags:
  - logging
  - mlflow
  - wandb
  - refactor
---

# Goal

Simplify the logger protocol by removing the separate `RunProtocol` / adapter classes. The logger itself provides logging methods; `start_run()` is a `@contextmanager` that yields `self`.

This supersedes [logger-protocol.md](logger-protocol.md).

# Motivation

The original design had two abstractions:
- `LoggerProtocol` with `start_run() -> RunProtocol`
- `RunProtocol` implementing `__enter__/__exit__` plus logging methods

This led to:
1. **Adapter classes** (`MLflowRunAdapter`, `WandbRunAdapter`) that wrap native run objects
2. **Eager run creation** in MLflow adapter, causing runs to start before `__enter__` and leak if not used with `with`
3. **Extra indirection** when the native libraries already provide context managers

# Design

Both `mlflow.start_run()` and `wandb.init()` return context managers. We wrap them with `@contextmanager` and yield `self`, so the logger is both the factory and the run interface.

```
with logger.start_run(name="exp-1") as run:
    run.log_params({"lr": 0.01})
    run.log_metrics({"accuracy": 0.95})
```

Here `run` is the logger itself. Logging methods delegate to the native API.

# Protocol signature

```python
from contextlib import contextmanager
from typing import Any, ContextManager, Mapping, Protocol, Self

Metrics = Mapping[str, float]
Params = Mapping[str, Any]
Tags = Mapping[str, str]


class LoggerProtocol(Protocol):
    def start_run(
        self,
        name: str | None = None,
        config: Params | None = None,
        tags: Tags | None = None,
        nested: bool = False,
    ) -> ContextManager[Self]: ...

    def log_params(self, params: Params) -> None: ...
    def log_metrics(self, metrics: Metrics, step: int | None = None) -> None: ...
    def set_tags(self, tags: Tags) -> None: ...
    def log_artifact(self, path: str, name: str | None = None) -> None: ...
    def log_model(self, model: Any, name: str | None = None) -> None: ...
```

Note: `finish()` is removed. The `@contextmanager` handles cleanup via `finally` or the native context manager's `__exit__`.

# Implementations

## NoOpLogger

```python
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass
class NoOpLogger:
    @contextmanager
    def start_run(self, name=None, config=None, tags=None, nested=False):
        yield self

    def log_params(self, params) -> None:
        pass

    def log_metrics(self, metrics, step: int | None = None) -> None:
        pass

    def set_tags(self, tags) -> None:
        pass

    def log_artifact(self, path: str, name: str | None = None) -> None:
        pass

    def log_model(self, model: Any, name: str | None = None) -> None:
        pass
```

## MLflowLogger

MLflow uses module-level functions that operate on the active run. No need to store run state.

```python
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass
class MLflowLogger:
    experiment_name: str | None = None

    @contextmanager
    def start_run(self, name=None, config=None, tags=None, nested=False):
        import mlflow

        if self.experiment_name:
            mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=name, nested=nested):
            if config:
                self.log_params(config)
            if tags:
                self.set_tags(tags)
            yield self

    def log_params(self, params) -> None:
        import mlflow
        mlflow.log_params(dict(params))

    def log_metrics(self, metrics, step: int | None = None) -> None:
        import mlflow
        mlflow.log_metrics(dict(metrics), step=step)

    def set_tags(self, tags) -> None:
        import mlflow
        mlflow.set_tags(dict(tags))

    def log_artifact(self, path: str, name: str | None = None) -> None:
        import mlflow
        if name is None:
            mlflow.log_artifact(path)
        else:
            mlflow.log_artifact(path, name=name)

    def log_model(self, model: Any, name: str | None = None) -> None:
        import mlflow
        mlflow.sklearn.log_model(model, name=name or "model")
```

## WandbLogger

W&B requires calling methods on the run object. Store it in `self._run`.

```python
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WandbLogger:
    project: str | None = None
    entity: str | None = None
    _run: Any = field(default=None, init=False, repr=False)

    @contextmanager
    def start_run(self, name=None, config=None, tags=None, nested=False):
        import wandb

        with wandb.init(
            project=self.project,
            entity=self.entity,
            name=name,
            config=config or {},
            tags=list(tags.values()) if tags else None,
            reinit=nested,
        ) as run:
            self._run = run
            yield self
        self._run = None

    def log_params(self, params) -> None:
        self._run.config.update(dict(params), allow_val_change=True)

    def log_metrics(self, metrics, step: int | None = None) -> None:
        self._run.log(dict(metrics), step=step)

    def set_tags(self, tags) -> None:
        existing = set(self._run.tags or [])
        self._run.tags = sorted(existing | set(tags.values()))

    def log_artifact(self, path: str, name: str | None = None) -> None:
        import wandb
        artifact = wandb.Artifact(name or "artifact", type="file")
        artifact.add_file(path)
        self._run.log_artifact(artifact)

    def log_model(self, model: Any, name: str | None = None) -> None:
        if isinstance(model, str):
            self.log_artifact(model, name=name or "model")
```

# Changes required

1. **Delete** `RunProtocol` from `src/sklab/adapters/interfaces.py`
2. **Update** `LoggerProtocol` to match new signature
3. **Rewrite** `src/sklab/_logging/mlflow.py` (delete `MLflowRunAdapter`)
4. **Rewrite** `src/sklab/_logging/wandb.py` (delete `WandbRunAdapter`)
5. **Rewrite** `src/sklab/_logging/noop.py` (delete `NoOpRun`)
6. **Update** `Experiment` to remove `run.finish()` calls
7. **Update** tests and docs

# How to test

- Unit tests for each logger: verify `start_run` is a working context manager
- Integration tests (guarded by optional deps):
  - MLflow: run starts/ends correctly, params/metrics logged
  - W&B: run starts/ends correctly (offline mode), params/metrics logged
- Doctest in `docs/tutorials/logger-adapters.md` should pass without race conditions
