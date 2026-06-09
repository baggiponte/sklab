"""Weights & Biases logger."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from sklab._lazy import LazyModule

wandb = LazyModule("wandb", install_hint="Install scikit-lab with the 'wandb' extra.")


@dataclass
class WandbLogger:
    """Logger that tracks experiments with Weights & Biases.

    W&B requires calling methods on the run object, so we store
    a reference to the active run in `self._run`.
    """

    project: str | None = None
    entity: str | None = None
    _run: Any = field(default=None, init=False, repr=False)

    @contextmanager
    def start_run(self, name=None, config=None, tags=None, nested=False):
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
        artifact = wandb.Artifact(name or "artifact", type="file")
        artifact.add_file(path)
        self._run.log_artifact(artifact)

    def log_model(self, model: Any, name: str | None = None) -> None:
        if isinstance(model, str):
            self.log_artifact(model, name=name or "model")
