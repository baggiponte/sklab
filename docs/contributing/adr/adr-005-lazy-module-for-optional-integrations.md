# ADR-005: Use LazyModule for optional integrations

Source: Follows [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) and extracts the decision from `plans/lazy-module.md`.

## Status

Accepted

## Context

sklab supports optional integrations such as MLflow, Weights & Biases, and Optuna. Those integrations should feel available from the public API without forcing every installation to include every dependency. The earlier `_require_x()` helper pattern pushed import checks into every call site, repeated the same boilerplate across modules, and made optional adapters harder to extend. It also produced an interface that looked unlike normal Python module usage, which increased friction for contributors and made mistakes easier when new integrations were added.

The project needs one internal mechanism for optional dependencies that keeps adapter modules small, delays import work until the integration is actually used, and raises a clear installation error at the moment a missing package matters. That mechanism also needs to stay lightweight because these integrations are not performance-critical hot paths.

## Decision

We will represent optional third-party dependencies with a shared `LazyModule` helper. Adapter modules will bind a module-like proxy once at import time and then use that proxy exactly as they would use the real dependency. The proxy will defer the real import until first attribute access, cache the imported module after the first successful load, and raise a `ModuleNotFoundError` with an installation hint when the dependency is absent.

We will keep this helper minimal. It will not preflight availability with separate boolean flags, and it will not replace itself in module globals after loading. The accepted design favors a familiar call site, predictable behavior, and simple extension over micro-optimizations that do not matter for logging adapters and optional search integrations.

## Consequences

Optional integrations now share one import pattern instead of repeating per-module guard helpers. Adapter code reads more like ordinary Python because call sites use `mlflow.log_params(...)`, `wandb.init(...)`, or `optuna.create_study(...)` directly through the proxy. Adding a new optional integration becomes a small, local change because the contributor only needs to declare one `LazyModule` instance and write the adapter logic.

This decision also means import failures move to first use rather than module import time. That is the intended behavior, but it can delay discovery of a missing dependency until the relevant feature is exercised. The helper also keeps a small amount of proxy indirection on every attribute access after the first import. The project accepts that tradeoff because the affected code is not latency-sensitive, while the simpler and more consistent adapter design materially improves maintainability.
