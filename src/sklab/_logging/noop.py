"""No-op logger that drops all logging calls."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass
class NoOpLogger:
    """Logger that drops all logging calls.

    Useful as the default logger when no external tracking backend is configured.
    """

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
