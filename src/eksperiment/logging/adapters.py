"""Optional adapters for common experiment loggers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _require_mlflow() -> Any:
    try:
        import mlflow  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "mlflow is not installed. Install mlflow to use MLflow adapters."
        ) from exc
    return mlflow


def _require_wandb() -> Any:
    try:
        import wandb  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "wandb is not installed. Install eksperiment with the 'wandb' extra."
        ) from exc
    return wandb


@dataclass
class MLflowRunAdapter:
    """Run adapter that forwards logging calls to MLflow."""

    run: Any

    def __enter__(self) -> "MLflowRunAdapter":
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

    def start_run(self, name=None, config=None, tags=None, nested=False) -> MLflowRunAdapter:
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


@dataclass
class WandbRunAdapter:
    """Run adapter that forwards logging calls to W&B."""

    run: Any

    def __enter__(self) -> "WandbRunAdapter":
        self.run.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool | None:
        return self.run.__exit__(exc_type, exc, tb)

    def log_params(self, params) -> None:
        self.run.config.update(dict(params), allow_val_change=True)

    def log_metrics(self, metrics, step: int | None = None) -> None:
        if step is None:
            self.run.log(dict(metrics))
        else:
            self.run.log(dict(metrics), step=step)

    def set_tags(self, tags) -> None:
        existing = set(self.run.tags or [])
        self.run.tags = sorted(existing | set(tags.values()))

    def log_artifact(self, path: str, name: str | None = None) -> None:
        wandb = _require_wandb()
        artifact_name = name or "artifact"
        artifact = wandb.Artifact(artifact_name, type="file")
        artifact.add_file(path)
        self.run.log_artifact(artifact)

    def log_model(self, model: Any, name: str | None = None) -> None:
        if isinstance(model, str):
            self.log_artifact(model, name=name or "model")

    def finish(self, status: str = "success") -> None:
        self.run.finish(exit_code=1 if status == "failed" else 0)


@dataclass
class WandbLogger:
    """Logger adapter that creates W&B runs."""

    project: str | None = None
    entity: str | None = None

    def start_run(self, name=None, config=None, tags=None, nested=False) -> WandbRunAdapter:
        wandb = _require_wandb()
        run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=name,
            config=config or {},
            tags=list(tags.values()) if tags else None,
            reinit=nested,
        )
        return WandbRunAdapter(run)
