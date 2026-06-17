---
title: Result Predict API
description: Add explicit prediction access on result objects via estimator passthrough
date: 2026-02-19
---
# Result Predict API

## Goal

Expose an explicit, sklearn-like way to run predictions from a fitted result object, so users can continue experimentation flows without manually reaching into private state.

## References

- `src/sklab/_results.py`
- `src/sklab/experiment.py`
- `tests/test_experiment.py`
- `plans/run-result.md`
- `plans/run-result-design-v2.md`

## Design

- Keep `result.estimator` as the primary transparent escape hatch.
- Add thin convenience methods on result objects:
  - `result.predict(X)`
  - `result.predict_proba(X)`
- Add clear runtime errors when no estimator is available (for example, `cross_validate(refit=False)`).
- Extend `EvalResult` to include the fitted estimator so `evaluate()` results can also predict.

## How to Test

- Add tests that:
  - verify `FitResult.predict()` and `predict_proba()`
  - verify `EvalResult` now carries estimator and supports prediction
  - verify `CVResult` and `SearchResult` raise useful errors when estimator is unavailable
- Run experiment tests to ensure no behavior regression.

## Future Considerations

- Add task-specific conveniences only if they prove broadly useful (for example calibrated probabilities), keeping core result API slim.
