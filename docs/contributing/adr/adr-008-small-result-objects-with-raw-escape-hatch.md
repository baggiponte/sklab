# ADR-008: Return Small Method-Specific Result Objects with a Raw Escape Hatch

Source: This ADR follows the structure from [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) and synthesizes the decisions explored in `plans/run-result.md` and `plans/run-result-design-v2.md` against the current result API in `src/sklab/_results.py` and `src/sklab/experiment.py`.

## Status

Accepted.

## Context

sklab needs result objects that make experiment outputs usable without forcing users to learn a new framework-specific result model. Early planning explored a broader `RunResult` direction that would always capture predictions, targets, probabilities, timing, logger references, and downstream analysis hooks. That direction promised a single object for reporting and analysis, but it also risked turning a small experiment runner into a large result platform with higher memory cost, more conditional behavior, and more coupling to optional integrations.

The current codebase already leans toward a narrower shape. `fit()`, `evaluate()`, `cross_validate()`, and `search()` each return a dedicated result dataclass with a small number of fields that directly reflect the operation that produced them. At the same time, advanced users still need a way to drop down to the underlying estimator, sklearn return payload, fitted searcher, or Optuna study when sklab's convenience surface is not enough.

## Decision

We will keep method-specific result objects rather than introducing one unified result type. `FitResult`, `EvalResult`, `CVResult`, and `SearchResult` will remain small data containers whose primary job is to expose the most useful high-level outputs for each experiment method.

We will preserve a `raw` attribute on every result object as the explicit escape hatch to the underlying object produced or wrapped by that method. For fit results, `raw` may be the fitted estimator. For evaluation results, it may be the metrics mapping. For cross-validation results, it may be the full sklearn `cross_validate()` payload. For search results, it may be the fitted sklearn searcher or the Optuna study. We will not adopt the richer "always capture everything" ambitions from the earlier plans as the current default API.

## Consequences

The public result API stays small, predictable, and easy to explain. Users can learn the return type for each operation quickly, and sklab avoids taking ownership of a large cross-cutting analysis surface that sklearn or downstream tools may already provide better.

This choice also means some data proposed in the earlier plans is not guaranteed to be available directly on every result. Predictions, probabilities, run references, and other richer artifacts are not universally stored as first-class fields today, so users who need deeper access must rely on the `raw` escape hatch or compute those artifacts separately. That is a deliberate tradeoff in favor of a leaner core API.
