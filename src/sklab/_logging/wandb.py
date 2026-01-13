"""Weights and Biases logger adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _require_wandb() -> Any:
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "wandb is not installed. Install sklab with the 'wandb' extra."
        ) from exc
    return wandb


@dataclass
class WandbRunAdapter:
    """Run adapter that forwards logging calls to W&B."""

    run: Any

    def __enter__(self) -> WandbRunAdapter:
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

    def start_run(
        self, name=None, config=None, tags=None, nested=False
    ) -> WandbRunAdapter:
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
