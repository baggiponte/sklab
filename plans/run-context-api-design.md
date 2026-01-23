---
title: Run Context API Design
description: Experiment.run() context manager for composite runs
date: 2025-01-23
---

# Run Context API Design

## Goal

Add an `Experiment.run()` context manager that groups multiple operations (search, fit, evaluate, explain) under a single logged run with nested child runs.

## The Problem

Currently, each `Experiment` method creates its own logger run:

```python
exp.search(config, X, y)   # Run 1: logs best_params, best_score, model
exp.explain(X_test)        # Run 2: logs shap_summary, shap_importance
```

Users often want a single run containing related operations. The logger protocol supports `nested=True`, but the current design doesn't expose this.

### Current Limitations

1. Each method calls `self.logger.start_run()` independently
2. Methods don't know if they're inside an outer run
3. No way to group related operations under one run

## API Design

### Core API

```python
with exp.run(name="grid-search-with-explanation") as e:
    result = e.search(GridSearchConfig(...), X, y)
    explanation = e.explain(X_test)
    # Both logged as nested runs under one parent
```

### Implementation

```python
@dataclass(slots=True)
class Experiment:
    # ... existing fields ...
    _active_run: bool = field(default=False, repr=False)

    @contextmanager
    def run(
        self,
        name: str | None = None,
        *,
        tags: Mapping[str, str] | None = None,
    ) -> Iterator[Self]:
        """
        Context manager for grouping operations under a single logged run.

        All methods called within this context create nested runs under
        the parent run.

        Parameters
        ----------
        name : str, optional
            Name for the parent run. Defaults to experiment name.
        tags : dict, optional
            Additional tags for this run. Merged with experiment tags.

        Yields
        ------
        Self
            The experiment instance (for method chaining).
        """
        merged_tags = {**(self.tags or {}), **(tags or {})}
        with self.logger.start_run(
            name=name or self.name,
            config=None,
            tags=merged_tags or None,
        ):
            self._active_run = True
            try:
                yield self
            finally:
                self._active_run = False

    def fit(self, X, y=None, *, params=None, run_name=None) -> FitResult:
        # ... existing logic ...
        with self.logger.start_run(
            name=run_name or self.name,
            config=merged_params,
            tags=self.tags,
            nested=self._active_run,  # <-- Key change
        ) as run:
            # ...
```

### Behavior

| Context | `nested` value | Result |
|---------|----------------|--------|
| Outside `run()` | `False` | Standalone run (current behavior) |
| Inside `run()` | `True` | Nested under parent run |

### What Gets Logged Where

**Parent run** (from `run()` context):
- Tags (merged experiment + context tags)
- No metrics/params directly (those go in nested runs)

**Nested runs** (from methods inside context):
- All metrics, params, artifacts as usual
- Linked to parent run in MLflow/W&B UI

## Alternative: No Nesting, Just Grouping

If nested runs prove problematic (UI clutter, query complexity), consider a simpler approach:

```python
@contextmanager
def run(self, name: str | None = None) -> Iterator[Self]:
    """All operations in this context share the same run."""
    with self.logger.start_run(name=name or self.name, ...) as active_run:
        self._current_run = active_run
        try:
            yield self
        finally:
            self._current_run = None

def fit(self, ...):
    if self._current_run:
        # Log to existing run, don't create new one
        self._current_run.log_params(...)
    else:
        # Create new run (current behavior)
        with self.logger.start_run(...):
            ...
```

This puts everything in one flat run instead of nested runs. Simpler, but loses the ability to see individual operation boundaries in the UI.

**Recommendation:** Start with nested runs (Option B from the original design). It's more informative and matches how MLflow presents parent/child relationships. Fall back to flat runs only if users report UX issues.

## Edge Cases

### Nested `run()` Contexts

```python
with exp.run(name="outer"):
    with exp.run(name="inner"):  # What happens?
        exp.fit(X, y)
```

**Option 1:** Raise `RuntimeError("Already in a run context")`
**Option 2:** Allow arbitrary nesting (MLflow supports this)

Recommend Option 1 for v1—keep it simple. Users who need deeper nesting can use the logger directly.

### Exception Handling

```python
with exp.run(name="might-fail"):
    exp.fit(X, y)          # Succeeds, logged
    exp.evaluate(X, y)     # Raises exception
```

The parent run should still close properly (context manager handles this). MLflow/W&B will show the run as failed or interrupted depending on their semantics.

### Thread Safety

`_active_run` is instance state. If users share an `Experiment` across threads, they'll have issues. This is already true for `_fitted_estimator`. Document that `Experiment` instances are not thread-safe.

## Logger Protocol Impact

No changes needed. The protocol already has `nested: bool = False`:

```python
def start_run(
    self,
    name: str | None = None,
    config: Params | None = None,
    tags: Tags | None = None,
    nested: bool = False,  # Already exists
) -> AbstractContextManager[Self]: ...
```

All three logger implementations (NoOp, MLflow, W&B) already support this parameter.

## How to Test

1. **Unit test**: Verify `_active_run` flag toggles correctly
2. **Integration test with InMemoryLogger**:
   - Outside context: methods create standalone runs
   - Inside context: methods create nested runs
3. **MLflow integration test**:
   - Verify parent/child relationship in MLflow tracking
   - Verify nested runs appear under parent in UI
4. **Exception test**: Parent run closes even when child raises
5. **Nested context test**: Verify error raised for nested `run()` calls

## Future Considerations

1. **Run metadata**: Allow logging custom params/metrics to parent run
   ```python
   with exp.run(name="experiment") as e:
       e.log_params({"dataset": "iris"})  # To parent run
       e.fit(X, y)  # Nested run
   ```

2. **Run result**: Return a summary object from the context
   ```python
   with exp.run(name="experiment") as e:
       e.fit(X, y)
       e.evaluate(X_test, y_test)
   # e.summary contains all nested results?
   ```

3. **Automatic naming**: Child runs could auto-name based on method
   ```python
   with exp.run(name="experiment"):
       exp.fit(X, y)      # nested run named "experiment/fit"
       exp.evaluate(...)  # nested run named "experiment/evaluate"
   ```

## References

- [MLflow Nested Runs](https://mlflow.org/docs/latest/tracking.html#organizing-runs-into-experiments)
- [W&B Run Groups](https://docs.wandb.ai/guides/runs/grouping)
- Related: `plans/explain-api-design.md` (uses this for search + explain grouping)
