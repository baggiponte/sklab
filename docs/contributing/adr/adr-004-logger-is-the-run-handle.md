# ADR-004: Make the logger itself the run handle returned by `start_run()`

Source: [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions); extracted from `plans/logger-protocol-v2.md`.

## Status

Accepted.

## Context

The original logger design, captured in [ADR-003](./adr-003-context-managed-logger-run-protocol.md), introduced a separate `RunProtocol` returned by `LoggerProtocol.start_run(...)`. That approach preserved a familiar context-managed run shape, but it also required backend-specific adapter objects to wrap native MLflow and W&B run lifecycles. The extra layer did not materially improve the public API, and it made the implementation harder to reason about.

The revised design work showed that sklab did not need a distinct run object. MLflow already exposes module-level logging functions against the active run, and W&B can be handled by storing the active run on the logger during the context-managed block. The experiment runner only needs one small interface for starting a run and then logging params, metrics, tags, artifacts, and models. That made the earlier direction unnecessarily indirect.

## Decision

We will make the logger itself the run handle. `LoggerProtocol.start_run(...)` will return a context manager that yields `self`, and the logger object will provide the logging methods used inside the run block. This replaces the earlier direction in ADR-003 and removes the standalone `RunProtocol` abstraction from the core design.

We will keep the public usage shape familiar by continuing to write `with logger.start_run(...) as run:`, but `run` will now be the logger itself rather than a separate wrapper object. Implementations may rely on native backend context managers internally, but the protocol exposed to `Experiment` remains one object with one small method surface.

## Consequences

This decision simplifies the protocol, removes a layer of adapter classes, and keeps the experiment runner aligned with the code that now exists in the repository. It reduces lifecycle mistakes because runs are opened inside the context manager itself instead of being created eagerly and then wrapped after the fact. It also keeps custom loggers easy to implement because a compatible logger only needs `start_run()` plus the small set of logging methods.

The tradeoff is that the logger now carries any backend-specific active-run state during the context-managed block. Backends like W&B need to store and clear that state carefully, and the protocol no longer models the active run as a distinct type. That is acceptable here because sklab only needs a minimal, backend-agnostic logging surface, not a full run object hierarchy.
