"""Tests for LazyModule."""

from __future__ import annotations

import pytest

from sklab._lazy import LazyModule


def test_lazy_module_defers_import() -> None:
    """Module not imported until attribute access."""
    lazy = LazyModule("json", install_hint="...")
    assert lazy._module is None
    _ = lazy.dumps  # Trigger import
    assert lazy._module is not None


def test_lazy_module_caches_import() -> None:
    """Same module instance reused after first access."""
    lazy = LazyModule("json", install_hint="...")
    _ = lazy.dumps
    module1 = lazy._module
    _ = lazy.loads
    module2 = lazy._module
    assert module1 is module2


def test_lazy_module_attribute_works() -> None:
    """Can actually use the lazy-loaded module."""
    lazy = LazyModule("json", install_hint="...")
    result = lazy.dumps({"a": 1})
    assert result == '{"a": 1}'


def test_lazy_module_missing_dependency() -> None:
    """Clear error when module not installed."""
    lazy = LazyModule("nonexistent_module_xyz", install_hint="pip install xyz")
    with pytest.raises(ModuleNotFoundError, match="pip install xyz"):
        _ = lazy.something


def test_lazy_module_repr_not_loaded() -> None:
    lazy = LazyModule("json", install_hint="...")
    assert "not loaded" in repr(lazy)
    assert "json" in repr(lazy)


def test_lazy_module_repr_loaded() -> None:
    lazy = LazyModule("json", install_hint="...")
    _ = lazy.dumps
    assert "loaded" in repr(lazy)
    assert "not loaded" not in repr(lazy)


def test_lazy_module_requires_keyword_hint() -> None:
    """install_hint must be keyword-only."""
    with pytest.raises(TypeError):
        LazyModule("json", "positional hint")  # type: ignore[misc]
