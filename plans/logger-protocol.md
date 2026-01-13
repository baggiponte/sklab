---
date: 2026-01-12
title: Logger Protocol With Context-Managed Runs
description: Minimal logger and run handle protocol aligned with sklearn workflows
tags:
  - logging
  - mlflow
  - wandb
  - sklearn
  - experiment-runner
---

# Goal
Define a logger protocol that preserves context-managed run semantics (e.g., `with mlflow.start_run()` / `with wandb.init()`), while keeping a small, stable surface for experiment logging.

# Design sources
- HuggingFace Transformers callback system (event-driven logging hooks such as `on_log`, `on_train_begin`, `on_train_end` and a plugin-style integration list).
- MLflow run lifecycle (`mlflow.start_run()` returns a context-managed `ActiveRun` and auto-ends on `with` exit).
- Weights & Biases run lifecycle (`wandb.init()` returns a context-managed `Run` and auto-finishes on `with` exit).

# Design constraints
- The logger must return a context-managed run handle.
- The run handle must expose minimal, backend-agnostic logging methods.
- The protocol should support backends that expect config/params and tags at run creation time.

# Conceptual shape
- `LoggerProtocol.start_run(...) -> RunProtocol`
- `RunProtocol` implements `__enter__/__exit__` plus logging methods (`log_params`, `log_metrics`, `log_artifact`, `set_tags`, `log_model`, `finish(status=...)`).
- `Experiment` uses: `with logger.start_run(...) as run: ... run.log_metrics(...)`.

# Notes
- For simple local logging (stdout / JSONL), `RunProtocol` can be a no-op context manager.
- If finer-grained event hooks are needed, add a separate callback interface rather than inflating the core run API.

# Protocol signatures (draft)

```python
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Self


Metrics = Mapping[str, float]
Params = Mapping[str, Any]
Tags = Mapping[str, str]


class RunProtocol(Protocol):
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type, exc, tb) -> bool | None: ...

    def log_params(self, params: Params) -> None: ...
    def log_metrics(self, metrics: Metrics, step: int | None = None) -> None: ...
    def set_tags(self, tags: Tags) -> None: ...
    def log_artifact(self, path: str, name: str | None = None) -> None: ...
    def log_model(self, model: Any, name: str | None = None) -> None: ...
    def finish(self, status: str = "success") -> None: ...


class LoggerProtocol(Protocol):
    def start_run(
        self,
        name: str | None = None,
        config: Params | None = None,
        tags: Tags | None = None,
        nested: bool = False,
    ) -> RunProtocol: ...


```

# Minimal adapter sketches (draft)

```python
from dataclasses import dataclass
from typing import Any
from contextlib import contextmanager

# Optional import paths; adapters should guard imports in real code.
import mlflow
import wandb


@dataclass
class NoOpRun:
    def __enter__(self) -> "NoOpRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool | None:
        return None

    def log_params(self, params) -> None:
        return None

    def log_metrics(self, metrics, step: int | None = None) -> None:
        return None

    def set_tags(self, tags) -> None:
        return None

    def log_artifact(self, path: str, name: str | None = None) -> None:
        return None

    def log_model(self, model: Any, name: str | None = None) -> None:
        return None

    def finish(self, status: str = "success") -> None:
        return None


@dataclass
class NoOpLogger:
    def start_run(self, name=None, config=None, tags=None, nested=False) -> NoOpRun:
        return NoOpRun()


@dataclass
class MLflowRunAdapter:
    run: mlflow.ActiveRun

    def __enter__(self) -> "MLflowRunAdapter":
        self.run.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool | None:
        return self.run.__exit__(exc_type, exc, tb)

    def log_params(self, params) -> None:
        mlflow.log_params(dict(params))

    def log_metrics(self, metrics, step: int | None = None) -> None:
        mlflow.log_metrics(dict(metrics), step=step)

    def set_tags(self, tags) -> None:
        mlflow.set_tags(dict(tags))

    def log_artifact(self, path: str, name: str | None = None) -> None:
        if name is None:
            mlflow.log_artifact(path)
        else:
            mlflow.log_artifact(path, name=name)

    def log_model(self, model: Any, name: str | None = None) -> None:
        if name is None:
            name = "model"
        # Minimal: let callers provide their own mlflow flavor wrappers.
        mlflow.pyfunc.log_model(name=name, python_model=model)

    def finish(self, status: str = "success") -> None:
        # MLflow uses "FINISHED" or "FAILED".
        if status == "failed":
            mlflow.end_run(status="FAILED")
        else:
            mlflow.end_run(status="FINISHED")


@dataclass
class MLflowLogger:
    experiment_name: str | None = None

    def start_run(self, name=None, config=None, tags=None, nested=False) -> MLflowRunAdapter:
        if self.experiment_name:
            mlflow.set_experiment(self.experiment_name)
        active = mlflow.start_run(run_name=name, nested=nested)
        run = MLflowRunAdapter(active)
        if config:
            run.log_params(config)
        if tags:
            run.set_tags(tags)
        return run


@dataclass
class WandbRunAdapter:
    run: wandb.sdk.wandb_run.Run

    def __enter__(self) -> "WandbRunAdapter":
        self.run.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool | None:
        return self.run.__exit__(exc_type, exc, tb)

    def log_params(self, params) -> None:
        # W&B config is the conventional place for params.
        self.run.config.update(dict(params), allow_val_change=True)

    def log_metrics(self, metrics, step: int | None = None) -> None:
        if step is None:
            self.run.log(dict(metrics))
        else:
            self.run.log(dict(metrics), step=step)

    def set_tags(self, tags) -> None:
        # W&B has tags on the run; merge with existing tags.
        existing = set(self.run.tags or [])
        self.run.tags = sorted(existing | set(tags.values()))

    def log_artifact(self, path: str, name: str | None = None) -> None:
        artifact_name = name or "artifact"
        artifact = wandb.Artifact(artifact_name, type="file")
        artifact.add_file(path)
        self.run.log_artifact(artifact)

    def log_model(self, model: Any, name: str | None = None) -> None:
        # Minimal placeholder: store as artifact via serialization handled by caller.
        # For now, expect `model` to be a path or file-like identifier.
        if isinstance(model, str):
            self.log_artifact(model, name=name or "model")

    def finish(self, status: str = "success") -> None:
        # W&B finishes on exit, but explicit finish is ok.
        self.run.finish(exit_code=1 if status == "failed" else 0)


@dataclass
class WandbLogger:
    project: str | None = None
    entity: str | None = None

    def start_run(self, name=None, config=None, tags=None, nested=False) -> WandbRunAdapter:
        run = wandb.init(project=self.project, entity=self.entity, name=name, config=config or {}, tags=list(tags.values()) if tags else None, reinit=nested)
        return WandbRunAdapter(run)
```

# How to test
- Integration tests with a stub logger and run adapter that record calls:
  - `start_run()` returns a context-managed run; `__enter__`/`__exit__` are invoked.
  - `log_params`, `log_metrics`, `set_tags`, `log_artifact`, `log_model`, `finish` are called with expected payload shapes.
- Adapter smoke tests (guarded by optional deps):
  - MLflow: start/finish a run and log params/metrics without errors.
  - W&B: init/finish a run and log params/metrics without errors (offline mode).
