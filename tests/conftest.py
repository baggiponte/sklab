"""Shared test fixtures and helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def data() -> tuple[Any, Any]:
    """Dataset fixture for experiments."""
    dataset = load_iris()
    return dataset.data, dataset.target


@pytest.fixture
def pipeline() -> Pipeline:
    """Pipeline fixture for experiments."""
    return _pipeline()


@pytest.fixture
def pipeline_factory() -> Callable[[], Pipeline]:
    """Factory fixture for creating new pipelines on demand."""
    return _pipeline


@pytest.fixture
def logger() -> InMemoryLogger:
    """Logger fixture for experiments."""
    return InMemoryLogger()


@dataclass
class RunRecord:
    """Record of a single start_run context."""

    name: str | None
    config: Mapping[str, Any] | None
    tags: Mapping[str, str] | None
    params_calls: list[Mapping[str, Any]] = field(default_factory=list)
    metrics_calls: list[tuple[Mapping[str, float], int | None]] = field(
        default_factory=list
    )
    model_calls: list[Any] = field(default_factory=list)


@dataclass
class InMemoryLogger:
    """Logger that records all calls for testing."""

    runs: list[RunRecord] = field(default_factory=list)
    _current_run: RunRecord | None = None

    @contextmanager
    def start_run(self, name=None, config=None, tags=None, nested=False):
        record = RunRecord(
            name=name,
            config=dict(config) if config else None,
            tags=dict(tags) if tags else None,
        )
        self.runs.append(record)
        self._current_run = record
        try:
            yield self
        finally:
            self._current_run = None

    def log_params(self, params) -> None:
        assert self._current_run is not None
        self._current_run.params_calls.append(dict(params))

    def log_metrics(self, metrics, step=None) -> None:
        assert self._current_run is not None
        self._current_run.metrics_calls.append((dict(metrics), step))

    def set_tags(self, tags) -> None:
        pass

    def log_artifact(self, path, name=None) -> None:
        pass

    def log_model(self, model, name=None) -> None:
        assert self._current_run is not None
        self._current_run.model_calls.append(model)


def _pipeline() -> Pipeline:
    """Pipeline fixture for experiments."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=200, random_state=42)),
        ]
    )


