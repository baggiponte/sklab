# Logger plugins

Sklab loggers are defined by a small protocol. This keeps the core API
backend-agnostic while letting you integrate any tracker.

## Protocol

- `LoggerProtocol.start_run(...)` is a context manager that yields `self`
- The logger provides methods for params, metrics, tags, artifacts, and model
- This is a protocol (structural typing), so you do **not** need to inherit
  from it to be compatible

## Minimal implementation

```python
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass
class ConsoleLogger:
    @contextmanager
    def start_run(self, name=None, config=None, tags=None, nested=False):
        print("start_run", name)
        if config:
            self.log_params(config)
        if tags:
            self.set_tags(tags)
        try:
            yield self
        finally:
            print("end_run")

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
```

## Best practices

- Keep logging I/O light; avoid blocking in `log_metrics`.
- Log params once at run start, and metrics at evaluation time.
- Use the context manager's `finally` block for cleanup on errors.
