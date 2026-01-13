---
title: Search choices
description: Options and tradeoffs for hyperparameter search integration
date: 2026-01-12
---

# Search choices

## Goal
Capture the search API choices for Sklab: sklearn-native searchers, a simple Optuna config, and a power-user escape hatch that keeps the core API backend-agnostic.

## References
- `src/sklab/experiment.py` (Experiment.search contract)
- `docs/tutorials/sklearn-search.md` (sklearn searcher usage)

## Design
### Current contract (baseline)
- `Experiment.search(search, X, y, cv, n_trials, timeout, run_name)`
- `search` is either:
  - a searcher instance with `fit(X, y)` and optional `best_params_`, `best_score_`, `best_estimator_`, or
  - a config object with `create_searcher(...)` that returns such a searcher.

### Option A: sklearn primitives (lowest friction)
- Users pass `GridSearchCV`, `RandomizedSearchCV`, or halving searchers directly.
- Sklab does not override `scoring`, `refit`, or `cv` on the searcher. Users configure those explicitly.
- Sklab logs `best_params_`, `best_score_`, and `best_estimator_` when available.

### Option B: OptunaConfig (simple, batteries-included)
Provide a thin config that removes boilerplate without coupling core types to Optuna.

Proposed shape:
- `OptunaConfig(search_space: Callable[[Trial], dict[str, Any]], n_trials: int = 50, direction: str = "maximize", callbacks: list[Callable] | None = None, study_factory: Callable | None = None)`
- `create_searcher(...)` builds an internal Optuna-backed searcher that:
  - clones and configures the pipeline per trial,
  - evaluates via `cv` + scorers,
  - logs the best params/score/estimator.

User-facing example:
```python
result = experiment.search(
    OptunaConfig(search_space=search_space, n_trials=50, direction="maximize"),
    X,
    y,
    cv=5,
)
```

### Option C: Power-user escape hatch (bring your own searcher)
- Users can pass a custom searcher object that implements `fit()` and optional `best_*` attributes.
- This keeps advanced use cases possible (custom samplers/pruners, multi-objective studies, per-trial logging).

Blueprint to document:
```python
class SearcherProtocol:
    def fit(self, X, y=None): ...
    best_params_: dict | None
    best_score_: float | None
    best_estimator_: Any | None
```

### Tradeoffs
- **Option A**: simplest, but grid/halving limitations and no adaptive search.
- **Option B**: removes boilerplate while staying backend-agnostic; may hide some Optuna knobs unless surfaced.
- **Option C**: full flexibility with minimal core surface area; requires more user code.

### Implementation split (non-spaghetti)
- Centralize branching in `_build_searcher(...)`:
  - if object has `create_searcher(...)`, treat as config and build a searcher.
  - else if object has `fit(...)`, treat as searcher.
  - otherwise raise a TypeError with a helpful message.
- `Experiment.search()` stays small: resolve searcher → `fit()` → log `best_*` if present.

### Recommendation
- Keep Option A and Option C (already aligned with the current contract).
- Add Option B as an optional adapter module (e.g., `sklab/optuna.py`) that implements `OptunaConfig`.
- Document Option C as the escape hatch to preserve hackability.

## How to test
- Unit test: ensure `Experiment.search()` logs `best_*` for a dummy searcher.
- Integration test: sklearn `GridSearchCV` with a small dataset.
- Integration test (optional extra): OptunaConfig searcher returns best params/estimator.

## Future considerations
- Multi-objective Optuna support and richer trial metadata hooks.
- Standardized `SearcherProtocol` type alias in the public API for static typing.
- Expose trial-level logging callbacks across adapters.
