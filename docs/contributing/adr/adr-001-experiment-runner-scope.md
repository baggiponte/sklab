# ADR-001: Keep sklab focused on experiment running, not MLOps platform features

Source: Following the structure from [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions), extracted from `plans/feature-vision.md`.

## Status

Accepted.

## Context

sklab exists to remove the repetitive work around running sklearn experiments, not to become a general machine learning platform. The source vision document defines the core problem narrowly: data scientists already have pipelines, datasets, and modeling tools, but they lose time to repeated experiment boilerplate, inconsistent logging, scattered diagnostics, and opaque search and cross-validation outputs. That same document also draws a hard boundary around the product so the library stays familiar and useful instead of expanding into adjacent concerns that other tools already cover well.

This boundary matters because the surrounding ecosystem is already crowded with tools for data loading, experiment tracking backends, model registries, deployment, and distributed training. If sklab tries to absorb those responsibilities, it stops being a thin, dependable layer over sklearn workflows and starts forcing users into a new framework and a larger mental model. The project philosophy in the source material therefore treats familiarity as a feature and keeps the library centered on one job: taking an existing pipeline and returning useful experimental results.

## Decision

We will keep sklab narrowly focused on experiment running for sklearn-style pipelines. The library will help users fit, evaluate, cross-validate, and search experiments while automatically capturing the outputs that make those runs interpretable and reproducible. We will design it as a library, not a framework, so users can continue to rely on the surrounding sklearn ecosystem for data preparation, split strategy selection, estimator composition, and backend-specific tracking infrastructure.

We will not expand the core product into dataset loaders, a CLI, pipeline template generation, calibration tooling, experiment registry features, model serving, deployment workflows, or distributed training orchestration. When users need those capabilities, sklab will defer to established tools or to project documentation and examples rather than duplicating them in the core package.

## Consequences

This decision keeps the public surface area small and makes the project easier to explain, test, and maintain. It also protects the main value proposition: users can bring an existing sklearn pipeline to sklab without learning a new platform or surrendering control over the rest of their workflow. The library can remain opinionated about experiment outputs and ergonomics while still feeling familiar to users who already know sklearn.

The tradeoff is that some requests that seem adjacent to experimentation will remain out of scope, even when they might be convenient additions. Users who want a single tool for the entire machine learning lifecycle will need to combine sklab with other libraries and services. The documentation therefore carries more responsibility, because examples and guides must show how sklab fits into a larger workflow without trying to replace the rest of that workflow.
