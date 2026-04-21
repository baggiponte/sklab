# Architecture Decision Records

This project keeps lightweight Architecture Decision Records (ADRs) for
architecturally significant decisions. The format follows Michael Nygard's
["Documenting Architecture Decisions"](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions).

Status values follow that article's terminology:

- `Accepted` means the decision is embodied by the current repository.
- `Proposed` means the decision was captured but not yet adopted.
- `Superseded by ADR-XXX` means the decision was once active but has been
  replaced by a later ADR.

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-001](adr-001-experiment-runner-scope.md) | Keep sklab focused on experiment running, not MLOps platform features | Accepted |
| [ADR-002](adr-002-sklearn-first-experiment-api.md) | Use a sklearn-first, pipeline-first Experiment API | Accepted |
| [ADR-003](adr-003-context-managed-logger-run-protocol.md) | Introduce a context-managed logger run protocol | Superseded by ADR-004 |
| [ADR-004](adr-004-logger-is-the-run-handle.md) | Make the logger itself the run handle returned by `start_run()` | Accepted |
| [ADR-005](adr-005-lazy-module-for-optional-integrations.md) | Use LazyModule for optional integrations | Accepted |
| [ADR-006](adr-006-search-tiers-sklearn-and-optuna.md) | Support two search tiers: sklearn-native search configs and Optuna searchers | Accepted |
| [ADR-007](adr-007-type-safe-scorer-enum.md) | Provide a type-safe scorer enum without removing raw scorer flexibility | Accepted |
| [ADR-008](adr-008-small-result-objects-with-raw-escape-hatch.md) | Return small method-specific result objects with a raw escape hatch | Accepted |
| [ADR-009](adr-009-documentation-philosophy-as-project-decision.md) | Treat documentation philosophy as a project decision | Accepted |
