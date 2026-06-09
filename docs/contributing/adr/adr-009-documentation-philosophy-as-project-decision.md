# ADR-009: Treat Documentation Philosophy as a Project Decision

Source: This ADR follows the structure from [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) and preserves the documentation direction explored in `plans/docs-critique-and-improvement.md`, together with the standing contributor guidance in [docs/developer/writing-docs.md](../../developer/writing-docs.md).

## Status

Accepted.

## Context

sklab's documentation is part of the product, not supporting material around it. The project promises to reduce the friction of running experiments, but that promise fails if users can run the code only after bringing substantial prior knowledge or reverse-engineering the design from examples. Earlier documentation planning identified a recurring problem: technically correct docs can still fail as teaching material when they start with mechanics, assume theory, and leave readers to infer why the workflow exists.

The repo already treats runnable examples as a quality gate, but execution alone does not guarantee that the docs teach well. The project needed an explicit decision that documentation should explain the problem being solved, introduce theory only when needed, connect abstractions to concrete consequences, and direct readers toward primary sources for deeper study.

## Decision

We will treat the documentation philosophy as an explicit project decision and write docs accordingly. Tutorials and conceptual docs will start with the problem before the solution, explain concepts at the point of use, show what the code just did after examples, and link outward for deeper theory instead of assuming prior knowledge or reproducing whole textbooks.

We will keep runnable examples as a non-negotiable requirement and pair that requirement with writing guidance that is intentionally instructional rather than purely referential. The contributor guide in `docs/developer/writing-docs.md` remains the operational expression of this decision.

## Consequences

Documentation work now has a clearer bar for acceptance. Authors are expected to justify why a feature matters, teach the minimum theory needed for correct use, and keep examples executable. That should make tutorials more useful to less experienced practitioners and keep the docs aligned with the library's product philosophy.

This decision also raises the cost of writing and reviewing docs. Pages need stronger narrative structure, more deliberate explanation, and careful linkage to primary sources. Some concise reference-style writing that would be cheaper to produce is no longer sufficient for tutorials and conceptual guides, because correctness alone is not the standard the project has chosen.
