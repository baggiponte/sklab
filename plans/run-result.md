---
title: RunResult foundation
description: Core result object shape for predictions, metrics, and artifacts
date: 2026-01-12
---

# RunResult foundation

## Goal
Define a unified `RunResult` object that captures metrics, predictions, and
artifacts so downstream features (reports, feature importance, SHAP, conformal
predictions via MAPIE, etc.) can be built on top of a consistent result API.

## References
- `src/sklab/experiment.py` (FitResult/EvalResult/CVResult/SearchResult)
- `docs/tutorials/*` (expected user workflows and outputs)

## Design
### Core intent
- Provide a single, extensible result object for any run type (fit/eval/cv/search).
- Capture **predictions** alongside metrics and model references.
- Enable report generation without re-running models.

### Proposed shape (draft)
- `RunResult` dataclass
  - `kind: Literal["fit", "evaluate", "cross_validate", "search"]`
  - `estimator: Any | None`
  - `metrics: Mapping[str, float]`
  - `params: Mapping[str, Any]`
  - `predictions: Any | None`
  - `targets: Any | None`
  - `fold_predictions: Mapping[str, list[Any]] | None`
  - `fold_metrics: Mapping[str, list[float]] | None`
  - `artifacts: Mapping[str, Any] | None` (e.g., model paths, plots)

### Prediction strategy
- `evaluate()` should optionally compute and store predictions.
- `cross_validate()` should optionally store per-fold predictions and targets.
- `search()` should store predictions for the best estimator (if requested).
- Keep prediction storage optional to avoid heavy memory usage.

### Reporting hooks
- Provide a `report()` method or helper that can format metrics, params, and
  predictions into a summary artifact (JSON/markdown/HTML).
- Reports should be pure and not re-run training.

### Feature importance / SHAP / conformal hooks
- Provide minimal API surface: `result.estimator`, `result.predictions`,
  `result.targets` to let downstream utilities compute explanations.
- Keep SHAP/MAPIE integrations out of core but document usage patterns.

## How to test
- Unit tests for RunResult serialization (metrics, params, predictions).
- Integration tests: `evaluate()` and `cross_validate()` returning predictions
  when explicitly requested.
- Doc tests: add small examples that use predictions from results.

## Future considerations
- Add result caching to disk for large predictions.
- Add multi-output / probabilistic predictions support.
- Add standardized plotting utilities for common reports.
