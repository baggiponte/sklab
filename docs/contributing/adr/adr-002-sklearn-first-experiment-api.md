# ADR-002: Use a sklearn-first, pipeline-first Experiment API

Source: Following the structure from [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions), extracted from `plans/experiment-v1.md` with supporting context from `plans/feature-vision.md`.

## Status

Accepted.

## Context

The project aims to reduce experiment boilerplate without introducing a new workflow that competes with sklearn. The source plans describe a tension between convenience and familiarity: sklab needs a public API that is higher level than raw sklearn primitives, but it cannot hide the estimator lifecycle so aggressively that users lose track of what is being fit, evaluated, or searched. The feature vision also assumes that users already have pipelines and want answers from them, which makes the pipeline itself the natural center of the API.

The experiment plan resolves this by separating the main experiment actions instead of collapsing them into one overloaded entry point. Training, evaluation, cross-validation, and search have different inputs, logging expectations, and result shapes. The same plan also makes data ownership explicit by keeping train-test splitting and dataset storage out of the `Experiment` object. That preserves sklearn's mental model, keeps the wrapper slim, and avoids turning `Experiment` into a stateful orchestration container.

## Decision

We will make `Experiment` the primary public wrapper around an existing sklearn pipeline and define its API in sklearn-first terms. An `Experiment` will hold the pipeline together with lightweight experiment concerns such as logging configuration, scorers, and optional metadata, but it will not become the owner of datasets or split logic. Its core methods will remain explicit and separate: `fit()` for training, `evaluate()` for metrics on a provided estimator and dataset, `cross_validate()` for fold-based evaluation with an explicit splitter, and `search()` for explicit hyperparameter search.

We will keep the API pipeline-first by requiring users to bring their own sklearn-compatible pipeline and by preserving explicit control over evaluation strategy and search configuration. We will not introduce implicit dataset splitting, silent fallback search behavior, or a framework-specific execution model that obscures the underlying sklearn concepts. Logging and result handling may be standardized around these methods, but the lifecycle they represent will remain recognizable to sklearn users.

## Consequences

This decision makes sklab easier to adopt because the core abstraction matches what users already understand: they start with a pipeline and then choose whether to fit, evaluate, cross-validate, or search it. It also keeps branching logic centralized inside a small number of methods instead of scattering experiment behavior across helper layers. That structure makes logging integration and result typing easier to reason about because each operation has a clear purpose and a bounded contract.

The tradeoff is that the API remains intentionally explicit. Users still need to choose their splitters, provide the right datasets to each method, and understand the distinction between evaluation and search instead of relying on a single magic entry point. This limits convenience in some cases, but it avoids surprising behavior and preserves compatibility with the surrounding sklearn ecosystem, which is the more important long-term constraint for the library.
