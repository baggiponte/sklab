---
title: Test Coverage Improvements
description: Add InMemoryLogger, test logging behavior, error paths, and coverage instrumentation
date: 2026-01-15
---

## Goal

Improve test coverage to verify the library's core value proposition: "log everything transparently." Current tests are smoke-level; they don't assert on logging behavior, error paths, or internal invariants.

## References

- [tests/test_experiment.py](../tests/test_experiment.py)
- [src/sklab/experiment.py](../src/sklab/experiment.py)
- [src/sklab/adapters/logging.py](../src/sklab/adapters/logging.py)
- [AGENTS.md](../AGENTS.md) — "Integration tests over mocks"

## Design

### 1. InMemoryLogger test helper

A logger that implements `LoggerProtocol` and records all calls for assertion. Key insight: `start_run` yields `self` (the logger *is* the run handle), not a separate Run object.

```python
@dataclass
class RunRecord:
    """Record of a single start_run context."""
    name: str | None
    config: Mapping[str, Any] | None
    tags: Mapping[str, str] | None
    params_calls: list[Mapping[str, Any]]
    metrics_calls: list[tuple[Mapping[str, float], int | None]]
    model_calls: list[Any]


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
            params_calls=[],
            metrics_calls=[],
            model_calls=[],
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
        pass  # Not currently used by Experiment

    def log_artifact(self, path, name=None) -> None:
        pass  # Not currently used by Experiment

    def log_model(self, model, name=None) -> None:
        assert self._current_run is not None
        self._current_run.model_calls.append(model)
```

### 2. Tests to add

#### A. Experiment.fit logging tests

- `start_run` called with correct `name`, `config` (merged params), `tags`
- `log_model` called once with the fitted estimator
- No `log_metrics` or `log_params` calls
- Pipeline is cloned (original unfitted, result fitted)
- `_fitted_estimator` is set to the result

#### B. Experiment.evaluate tests

- Raises `NotFittedError` if called before `fit`
- `start_run` called with `config=None`
- `log_metrics` called once with all scorer results
- No `log_model` or `log_params` calls
- Works with both string and callable scorers

#### C. Experiment.cross_validate tests

- Raises `ValueError` if no scorers configured
- `fold_metrics` has correct length per scorer
- `metrics` has `cv/<name>_mean` and `cv/<name>_std` keys
- When `refit=True`: `log_model` called, `_fitted_estimator` updated
- When `refit=False`: no `log_model`, `_fitted_estimator` unchanged, `estimator` is None

#### D. Experiment.search tests

- `SearchConfigProtocol` path: `create_searcher` called with correct args
- `log_params` called with `best_params`
- When `best_score_` exists: `log_metrics` called
- When `best_score_` is None: no `log_metrics`
- When `best_estimator_` exists: `log_model` called, `_fitted_estimator` updated
- When `best_estimator_` is None: no `log_model`, `_fitted_estimator` unchanged
- Invalid search object raises `TypeError`

### 3. Coverage instrumentation

Add `pytest-cov` to dev dependencies and configure:

```toml
[tool.pytest.ini_options]
addopts = "--cov=sklab --cov-report=term-missing --cov-fail-under=85"
```

Use coverage to find dead code, not as a vanity metric.

## How to test

```bash
just test
```

All new tests should pass. Coverage report should show improved coverage of `experiment.py` and logging paths.

## Future considerations

- Property-based tests for `_aggregate_cv_metrics` edge cases
- Contract tests for custom logger implementations
- Tests for nested runs when that feature is used
