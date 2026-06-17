---
title: Model Promotion
description: Promote experiment results to a lightweight inference model abstraction
date: 2026-02-19
---
# Model Promotion

## Goal

Add a minimal, explicit API to promote fitted experiment results into a reusable model object for inference.

## References

- `src/sklab/experiment.py`
- `src/sklab/_results.py`
- `tests/test_experiment.py`

## Design

- Introduce `PromotedModel` dataclass with:
  - `estimator`
  - `name`
  - `source` (`Experiment`)
  - `metrics` (if available)
  - `params` (if available)
- Add `Experiment.promote(name=None)`:
  - promotes `Experiment._fitted_estimator`
  - requires experiment to already be fitted
  - returns `PromotedModel`
- Keep scope intentionally small:
  - no registry, no stages, no deployment concerns
  - pure in-memory abstraction aligned with sklearn feel

## How to Test

- Fit an experiment, call `promote()`, and run `predict`/`predict_proba`.
- Search with refit, call `promote()`, and verify it uses latest fitted estimator.
- Verify clear error when calling `promote()` before fitting.

## Future Considerations

- Optional logger adapters may store promoted metadata in backend-specific model registries without changing core API.
