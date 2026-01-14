"""Lazy module loading for optional dependencies."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any


class LazyModule:
    """Deferred module import - loads on first attribute access.

    Usage:
        mlflow = LazyModule("mlflow", install_hint="Install mlflow to use MLflowLogger.")

        # Later, in any method:
        mlflow.log_params(...)  # Import happens here, on first access
    """

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
