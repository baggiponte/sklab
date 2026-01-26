"""Logger protocol used by experiment runs."""

from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Any, Protocol, Self, runtime_checkable

Metrics = Mapping[str, float]
Params = Mapping[str, Any]
Tags = Mapping[str, str]


@runtime_checkable
class LoggerProtocol(Protocol):
    """Context-managed logger for experiment tracking.

    The logger itself provides logging methods. `start_run()` is a context
    manager that yields `self`, so usage is:

        with logger.start_run(name="exp-1") as run:
            run.log_params({"lr": 0.01})
            run.log_metrics({"accuracy": 0.95})
    """

    def start_run(
        self,
        name: str | None = None,
        config: Params | None = None,
        tags: Tags | None = None,
        nested: bool = False,
    ) -> AbstractContextManager[Self]:
        """Start a run and return a context manager for logging."""
        ...

    def log_params(self, params: Params) -> None:
        """Log parameters for the current run."""
        ...

    def log_metrics(self, metrics: Metrics, step: int | None = None) -> None:
        """Log metrics for the current run."""
        ...

    def set_tags(self, tags: Tags) -> None:
        """Set tags for the current run."""
        ...

    def log_artifact(self, path: str, name: str | None = None) -> None:
        """Log an artifact file for the current run."""
        ...

    def log_model(self, model: Any, name: str | None = None) -> None:
        """Log a model for the current run."""
        ...
