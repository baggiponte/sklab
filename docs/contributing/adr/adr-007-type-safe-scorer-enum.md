# ADR-007: Provide a Type-Safe Scorer Enum Without Removing Raw Scorer Flexibility

Source: This ADR follows the structure from [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) and preserves the decision originally captured in `plans/scorer-types.md`.

## Status

Accepted.

## Context

sklab accepts sklearn scorers across evaluation, cross-validation, and search APIs. Raw scorer strings match sklearn and stay familiar, but they are easy to mistype and hard to discover from an editor alone. The project also does not want to replace sklearn's scoring model with a custom abstraction that users must learn before they can run experiments.

The tension was between improving autocomplete and discoverability for common scorer names and preserving sklearn compatibility. A stricter wrapper type would improve guidance in the editor, but it would also block valid sklearn scorers expressed as strings and make it harder to pass scorer callables. A documentation-only answer would not address the ergonomics problem in actual API use.

## Decision

We will provide a `ScorerName` `StrEnum` containing supported sklearn scorer names and use it as an accepted input type alongside raw strings and scorer callables. Public scoring types will continue to accept `ScorerName | str | ScorerFunc` rather than forcing callers onto the enum.

We will keep the enum as an aid for autocomplete and readability, not as a replacement for sklearn's scoring interface. The enum values will remain valid sklearn scorer strings so they can flow through the existing scoring path without conversion or a parallel compatibility layer.

## Consequences

Callers get a typed, discoverable set of common scorer names while retaining the ability to pass raw sklearn strings and custom scorer callables. This keeps the public API familiar to sklearn users and avoids trapping advanced users behind a narrower abstraction.

The enum now becomes curated public surface area that must stay aligned with sklearn's scorer names. That creates a maintenance obligation when sklearn adds, renames, or removes scorers. It also means the type surface cannot express every possible valid scorer configuration by enum alone, so the raw-string and callable escape hatches remain necessary by design.
