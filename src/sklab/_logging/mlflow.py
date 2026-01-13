"""MLFlow logger adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _require_mlflow() -> Any:
    try:
        import mlflow
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "mlflow is not installed. Install mlflow to use MLflow adapters."
        ) from exc
    return mlflow


@dataclass
class MLflowRunAdapter:
    """Run adapter that forwards logging calls to MLflow."""

    run: Any

    def __enter__(self) -> MLflowRunAdapter:
        self.run.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool | None:
        return self.run.__exit__(exc_type, exc, tb)

    def log_params(self, params) -> None:
        mlflow = _require_mlflow()
        mlflow.log_params(dict(params))

    def log_metrics(self, metrics, step: int | None = None) -> None:
        mlflow = _require_mlflow()
        mlflow.log_metrics(dict(metrics), step=step)

    def set_tags(self, tags) -> None:
        mlflow = _require_mlflow()
        mlflow.set_tags(dict(tags))

    def log_artifact(self, path: str, name: str | None = None) -> None:
        mlflow = _require_mlflow()
        if name is None:
            mlflow.log_artifact(path)
        else:
            mlflow.log_artifact(path, artifact_path=name)

    def log_model(self, model: Any, name: str | None = None) -> None:
        mlflow = _require_mlflow()
        if name is None:
            name = "model"
        mlflow.pyfunc.log_model(artifact_path=name, python_model=model)

    def finish(self, status: str = "success") -> None:
        mlflow = _require_mlflow()
        if status == "failed":
            mlflow.end_run(status="FAILED")
        else:
            mlflow.end_run(status="FINISHED")


@dataclass
class MLflowLogger:
    """Logger adapter that creates MLflow runs."""

    experiment_name: str | None = None

    def start_run(
        self, name=None, config=None, tags=None, nested=False
    ) -> MLflowRunAdapter:
        mlflow = _require_mlflow()
        if self.experiment_name:
            mlflow.set_experiment(self.experiment_name)
        active = mlflow.start_run(run_name=name, nested=nested)
        run = MLflowRunAdapter(active)
        if config:
            run.log_params(config)
        if tags:
            run.set_tags(tags)
        return run
