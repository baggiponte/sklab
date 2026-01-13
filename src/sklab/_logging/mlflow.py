"""MLflow logger."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


def _require_mlflow() -> Any:
    try:
        import mlflow
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "mlflow is not installed. Install mlflow to use MLflowLogger."
        ) from exc
    return mlflow


@dataclass
class MLflowLogger:
    """Logger that tracks experiments with MLflow.

    MLflow uses module-level functions that operate on the active run,
    so we don't need to store run state.
    """

    experiment_name: str | None = None

    @contextmanager
    def start_run(self, name=None, config=None, tags=None, nested=False):
        mlflow = _require_mlflow()
        if self.experiment_name:
            mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=name, nested=nested):
            if config:
                self.log_params(config)
            if tags:
                self.set_tags(tags)
            yield self

    def log_params(self, params) -> None:
        _require_mlflow().log_params(dict(params))

    def log_metrics(self, metrics, step: int | None = None) -> None:
        _require_mlflow().log_metrics(dict(metrics), step=step)

    def set_tags(self, tags) -> None:
        _require_mlflow().set_tags(dict(tags))

    def log_artifact(self, path: str, name: str | None = None) -> None:
        mlflow = _require_mlflow()
        if name is None:
            mlflow.log_artifact(path)
        else:
            mlflow.log_artifact(path, name=name)

    def log_model(self, model: Any, name: str | None = None) -> None:
        mlflow = _require_mlflow()
        mlflow.sklearn.log_model(model, name=name or "model")
