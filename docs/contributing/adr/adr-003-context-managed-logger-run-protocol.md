# ADR-003: Introduce a context-managed logger run protocol

Source: [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions); extracted from `plans/logger-protocol.md`.

## Status

Superseded by [ADR-004: Make the logger itself the run handle returned by `start_run()`](./adr-004-logger-is-the-run-handle.md).

## Context

sklab needs to log experiment parameters, metrics, artifacts, tags, and fitted models without binding the core experiment API to one tracking backend. The initial design work focused on matching the run lifecycle that users already know from MLflow and Weights & Biases, where a run is opened with a context manager and closed automatically when the block exits. At the same time, the library needed a backend-agnostic surface that `Experiment` could call consistently across no-op logging, MLflow, and W&B.

The first direction separated the logger factory from the active run object. In that model, `LoggerProtocol.start_run(...)` would create and return a dedicated `RunProtocol` object, and that run handle would implement the context manager methods together with the logging methods. This design preserved the familiar `with logger.start_run(...) as run:` shape and gave adapters a place to translate backend-specific lifecycle details into a common interface.

## Decision

We first decided to model logging around two protocols: a `LoggerProtocol` responsible for starting runs and a `RunProtocol` responsible for the active context-managed run. We would require `start_run(...)` to return a run handle that implemented `__enter__`, `__exit__`, `log_params`, `log_metrics`, `set_tags`, `log_artifact`, `log_model`, and `finish`. We would keep that run API intentionally small so the core experiment runner could depend on a stable contract while adapters handled backend differences.

This ADR now remains only as historical context. The later design in [ADR-004](./adr-004-logger-is-the-run-handle.md) replaced the separate run-handle abstraction with a simpler protocol in which `start_run()` yields the logger itself.

## Consequences

This original decision would have made the run lifecycle explicit and familiar to users coming from existing experiment trackers. It also would have let adapters prepare configuration and tags at run creation time while keeping the `Experiment` API backend-agnostic.

The tradeoff was additional abstraction. Every backend needed an extra adapter layer to wrap its native run object, and the split between logger and run handle introduced indirection without adding much user-facing capability. In practice, that made implementations more complex than necessary and created pressure around cleanup semantics and eager run creation. Those costs led to the replacement recorded in ADR-004.
