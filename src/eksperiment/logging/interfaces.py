"""Logger protocols used by experiment runs."""

from typing import Any, Mapping, Protocol, Self

Metrics = Mapping[str, float]
Params = Mapping[str, Any]
Tags = Mapping[str, str]


class RunProtocol(Protocol):
    """Minimal run handle for experiment logging."""

    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type, exc, tb) -> bool | None: ...

    def log_params(self, params: Params) -> None: ...

    def log_metrics(self, metrics: Metrics, step: int | None = None) -> None: ...

    def set_tags(self, tags: Tags) -> None: ...

    def log_artifact(self, path: str, name: str | None = None) -> None: ...

    def log_model(self, model: Any, name: str | None = None) -> None: ...

    def finish(self, status: str = "success") -> None: ...


class LoggerProtocol(Protocol):
    """Factory for context-managed logging runs."""

    def start_run(
        self,
        name: str | None = None,
        config: Params | None = None,
        tags: Tags | None = None,
        nested: bool = False,
    ) -> RunProtocol: ...
