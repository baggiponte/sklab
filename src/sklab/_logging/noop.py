from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class NoOpRun:
    """Run adapter that drops logging calls.

    Useful as the default run when no external logging backend is configured.
    """

    def __enter__(self) -> NoOpRun:
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
    """Logger adapter that produces no-op runs."""

    def start_run(self, name=None, config=None, tags=None, nested=False) -> NoOpRun:
        return NoOpRun()
