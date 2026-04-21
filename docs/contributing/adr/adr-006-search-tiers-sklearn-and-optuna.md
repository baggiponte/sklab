# ADR-006: Support two search tiers: sklearn-native search configs and Optuna searchers

Source: Follows [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) and extracts the decision from `plans/search-choices.md`.

## Status

Accepted

## Context

Hyperparameter search is a core part of experiment running, but sklab is not meant to become a search framework of its own. Users need a low-friction path that feels familiar when they are already working with scikit-learn, and they also need a stronger adaptive-search path for cases where grid or random search is too limited. At the same time, the core `Experiment.search()` API must stay small, backend-agnostic, and easy to reason about.

The design pressure is to support common search workflows without scattering conditional logic across the codebase or forcing users into one backend. A search API that only accepts raw sklearn objects leaves too much boilerplate for Optuna users. A search API that bakes Optuna concepts directly into the core would make the library less neutral and less familiar. The system needs a narrow contract that can support both tiers while keeping branching centralized.

## Decision

We will support two first-class search tiers behind the same `Experiment.search()` entry point. The first tier will cover sklearn-native search through lightweight config objects and compatible sklearn searchers such as `GridSearchCV` and `RandomizedSearchCV`. The second tier will cover Optuna through an `OptunaConfig` convenience path and an Optuna-backed searcher that fits the same search protocol.

We will keep the core contract structural rather than backend-specific. `Experiment.search()` will accept either a searcher object with `fit()` and optional `best_*` attributes or a config object that can build such a searcher. The branching to resolve configs into concrete searchers will stay centralized in the search-building path, and the rest of the experiment flow will treat both tiers the same: fit the searcher, log `best_params_`, `best_score_`, and `best_estimator_` when available, and return a `SearchResult` with a raw escape hatch.

## Consequences

sklab now offers a simple search path for the common sklearn case and a more capable adaptive-search path for Optuna without splitting the public API into unrelated workflows. Users can start with familiar sklearn search tools, move to `OptunaConfig` when they need better exploration, and still rely on the same experiment-level logging and result behavior. The core stays backend-agnostic because it depends on protocols and centralized branching rather than backend-specific code spread across the experiment runner.

This decision also makes the search surface intentionally uneven in capability. sklearn-backed search remains the easiest path, but it inherits sklearn’s limits. Optuna offers more power, but its configuration and study behavior are necessarily richer and less uniform than the sklearn tier. The project accepts that asymmetry because it preserves familiarity for the default case while keeping a clear power-user path, and it avoids building a custom search abstraction that would duplicate behavior users already know from sklearn and Optuna.
