# Logger plugins

Sklab loggers are defined by a small protocol. This keeps the core API
backend-agnostic while letting you integrate any tracker.

## Protocols

- `LoggerProtocol.start_run(...) -> RunProtocol`
- `RunProtocol` is a context manager with methods for params, metrics, tags,
  artifacts, model, and finish.
- These are protocols (structural typing), so you do **not** need to inherit
  from them to be compatible.

## Minimal implementation

```python
from dataclasses import dataclass
from typing import Any

from sklab.logging.interfaces import LoggerProtocol, RunProtocol

@dataclass
class ConsoleRun:
    def __enter__(self) -> "ConsoleRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool | None:
        return None

    def log_params(self, params) -> None:
        print("params", params)

    def log_metrics(self, metrics, step=None) -> None:
        print("metrics", metrics)

    def set_tags(self, tags) -> None:
        print("tags", tags)

    def log_artifact(self, path: str, name: str | None = None) -> None:
        print("artifact", path, name)

    def log_model(self, model: Any, name: str | None = None) -> None:
        print("model", name)

    def finish(self, status: str = "success") -> None:
        print("finish", status)


@dataclass
class ConsoleLogger:
    def start_run(self, name=None, config=None, tags=None, nested=False) -> ConsoleRun:
        run = ConsoleRun()
        if config:
            run.log_params(config)
        if tags:
            run.set_tags(tags)
        return run
```

## Best practices

- Keep logging I/O light; avoid blocking in `log_metrics`.
- Log params once at run start, and metrics at evaluation time.
- Use `finish(status="failed")` if an exception bubbles up.
