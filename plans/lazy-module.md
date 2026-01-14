---
title: LazyModule Pattern
description: Replace _require_X() helpers with a familiar module-like interface
date: 2026-01-14
---

# LazyModule Pattern

## Goal

Replace the repetitive `_require_X()` pattern with a `LazyModule` class that:
1. Looks like a normal module import at the call site
2. Defers actual import until first attribute access
3. Provides clear error messages when dependencies are missing
4. Makes adding new optional-dependency integrations trivial

## Problem

Current implementation requires calling `_require_mlflow()` at every use site:

```python
def log_params(self, params) -> None:
    _require_mlflow().log_params(dict(params))

def log_metrics(self, metrics, step=None) -> None:
    _require_mlflow().log_metrics(dict(metrics), step=step)

def set_tags(self, tags) -> None:
    _require_mlflow().set_tags(dict(tags))
```

Issues:
- **Repetitive** - Every method must remember to call the helper
- **Unfamiliar** - Doesn't look like normal Python imports
- **Error-prone** - Forgetting the call produces `NameError` instead of helpful message
- **Re-imports** - Each call does a fresh import (though Python caches it)
- **Copy-paste extensibility** - New integrations must duplicate the pattern

## References

- [Polars `_dependencies.py`](https://github.com/pola-rs/polars/blob/34e5b3333a729a0c71ea2ff4027e63fadfd46382/py-polars/src/polars/_dependencies.py#L219) - Production-grade lazy loading with `find_spec`, self-replacement in globals
- Python `importlib.import_module` - Dynamic import mechanism

## Design

### Core: `LazyModule` Class

```python
from importlib import import_module
from types import ModuleType
from typing import Any


class LazyModule:
    """Deferred module import - loads on first attribute access."""

    def __init__(self, name: str, *, install_hint: str) -> None:
        self._name = name
        self._install_hint = install_hint
        self._module: ModuleType | None = None

    def __getattr__(self, attr: str) -> Any:
        if self._module is None:
            try:
                self._module = import_module(self._name)
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    f"{self._name} is not installed. {self._install_hint}"
                ) from exc
        return getattr(self._module, attr)

    def __repr__(self) -> str:
        status = "loaded" if self._module else "not loaded"
        return f"<LazyModule {self._name!r} ({status})>"
```

### Usage in Logger Modules

```python
# src/sklab/_logging/mlflow.py
from sklab._utils import LazyModule

mlflow = LazyModule("mlflow", install_hint="Install mlflow to use MLflowLogger.")


class MLflowLogger:
    def log_params(self, params) -> None:
        mlflow.log_params(dict(params))  # Clean!

    def log_metrics(self, metrics, step=None) -> None:
        mlflow.log_metrics(dict(metrics), step=step)
```

### Design Decisions

#### 1. No `find_spec` Pre-check

Polars uses `find_spec()` to check module availability without importing. We skip this because:
- Our optional deps are only accessed when the user explicitly uses them
- The import error is informative and happens at the right time
- Simpler implementation

#### 2. No Self-Replacement in Globals

Polars replaces the proxy with the real module in `globals()` for zero overhead after first access. We skip this because:
- Logger methods aren't called in hot loops
- One `if self._module is None` check per access is negligible
- Keeps the implementation simple and predictable

#### 3. No `_AVAILABLE` Boolean

Polars returns `(module, available)` tuples. We skip this because:
- We don't need to branch on availability - we just use it or fail
- The error message guides users to install the dependency
- Could add later via `@property` if needed

#### 4. Keyword-Only `install_hint`

```python
LazyModule("mlflow", install_hint="...")  # Explicit
LazyModule("mlflow", "...")               # Error - forces clarity
```

Forces explicit naming, improving readability when skimming code.

#### 5. Module Location

Place in `src/sklab/_utils.py` (new file) or `src/sklab/_lazy.py`:
- Private module (underscore prefix) - internal utility
- Single responsibility - just the LazyModule class
- Importable by all adapter modules

Recommendation: `src/sklab/_lazy.py` - clear purpose, easy to find.

### Extensibility Story

Someone adding a `CometLogger`:

```python
# src/sklab/_logging/comet.py
from sklab._lazy import LazyModule

comet = LazyModule("comet_ml", install_hint="Install comet-ml to use CometLogger.")


class CometLogger:
    def start_run(self, name=None, config=None, **kwargs):
        self._experiment = comet.Experiment(project_name=name)
        if config:
            self._experiment.log_parameters(config)
        ...
```

No boilerplate function to copy. Just one line declaring the lazy module.

## Files to Change

1. **Create** `src/sklab/_lazy.py` - LazyModule class
2. **Update** `src/sklab/_logging/mlflow.py` - Use LazyModule
3. **Update** `src/sklab/_logging/wandb.py` - Use LazyModule
4. **Update** `src/sklab/optuna.py` - Use LazyModule
5. **Delete** `_require_X()` functions from each module

## How to Test

### Unit Tests for LazyModule

```python
# tests/test_lazy.py
import sys
import pytest
from sklab._lazy import LazyModule


def test_lazy_module_defers_import():
    """Module not imported until attribute access."""
    lazy = LazyModule("json", install_hint="...")
    assert lazy._module is None
    _ = lazy.dumps  # Trigger import
    assert lazy._module is not None


def test_lazy_module_caches_import():
    """Same module instance reused."""
    lazy = LazyModule("json", install_hint="...")
    _ = lazy.dumps
    module1 = lazy._module
    _ = lazy.loads
    module2 = lazy._module
    assert module1 is module2


def test_lazy_module_missing_dependency():
    """Clear error when module not installed."""
    lazy = LazyModule("nonexistent_module_xyz", install_hint="pip install xyz")
    with pytest.raises(ModuleNotFoundError, match="pip install xyz"):
        _ = lazy.something


def test_lazy_module_repr():
    lazy = LazyModule("json", install_hint="...")
    assert "not loaded" in repr(lazy)
    _ = lazy.dumps
    assert "loaded" in repr(lazy)
```

### Integration: Existing Logger Tests Still Pass

The existing tests in `tests/test_logging_adapters.py` should pass unchanged - they test the logger behavior, not the import mechanism.

## Future Considerations

### Optional: `available` Property

If we need to check availability without triggering import:

```python
@property
def available(self) -> bool:
    if self._module is not None:
        return True
    from importlib.util import find_spec
    spec = find_spec(self._name)
    return spec is not None and spec.loader is not None
```

### Optional: Type Stubs

For better IDE support, could add `py.typed` marker and stub files. Low priority - the `Any` return from `__getattr__` is acceptable for optional deps.
