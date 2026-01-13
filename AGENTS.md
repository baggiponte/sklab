# AGENTS.md

## What This Project Is

Sklab is an experiment runner for sklearn pipelines. One thing, done well: **run experiments**.

It removes the tedious parts of ML experimentation so you can iterate faster.

**The promise:** Give me a pipeline, I'll give you answers.

## What We Do

1. **Fit** — Train a pipeline, get params logged automatically
2. **Evaluate** — Get metrics + predictions + diagnostics, no boilerplate
3. **Cross-validate** — Get per-fold transparency, not just a number
4. **Search** — Get all trials logged, not just the winner

## Philosophy

> **Be useful. No bloat. So elegant it's familiar. Abstractions that are not obstructions. Provide value.**

**BE USEFUL.** Every feature must solve a real pain point. If you can't explain who needs it and why, don't build it.

**NO BLOAT.** If sklearn, Polars, or the logger backend already does it, we don't duplicate it. No distributed training, no deployment pipelines, no MLOps platform features. Just experiments, done well.

**SO ELEGANT IT'S FAMILIAR.** The API feels like sklearn because sklearn got it right. No new mental models. No surprising behaviors. Muscle memory works here.

**ABSTRACTIONS, NOT OBSTRUCTIONS.** We remove tedium, not control. You can always drop down to raw sklearn when needed. We wrap, we don't trap.

**PROVIDE VALUE.** Every line of code must earn its place. We ship what helps data scientists iterate faster, not what sounds impressive.

**DOCS ARE CODE.** Documentation is tested with the same rigor as source code. Every code example runs. If the docs lie, the build fails. High-quality docs are not optional — they're part of the product.

**ZERO SPRINKLING.** Inject the logger once. Everything else logs automatically. No `mlflow.log_*` scattered through your code.

## What We Always Capture

Every result includes:
- `y_pred` — Predictions (always)
- `y_proba` — Probabilities (classification)
- `y_true` — Ground truth
- `params` — What ran
- `estimator` — The fitted model
- Per-fold data (CV) — Debug variance, find bad samples

No flags to enable. No forgetting. It just happens.

## What We Do NOT Build

| Feature | Why Not |
|---------|---------|
| Dataset loaders | sklearn, HF Datasets, Polars do this |
| CLI | This is a library |
| Pipeline templates | Docs/examples, not core |
| Calibration tools | Analysis, not experimentation |
| Experiment registry | Logger backend's job |
| Distributed training | Different problem, different tool |
| Deployment | Out of scope — we run experiments |
| Model serving | Out of scope — we run experiments |

## How to Collaborate

1. **Spec first.** Draft a plan in `plans/` before coding. Keep it small, concrete, goal-driven.
2. **Plans must include:** Frontmatter header (`title`, `description`, `date`), then Goal, References, Design, How to test. Add "Future considerations" for scoped-out items. Note: Only plans have frontmatter; docs do not.
3. **Sketch before coding.** Interfaces, usage snippets, tradeoffs first. Code only after the API is clear.
4. **Small steps.** Prefer small, reviewable PRs over big jumps.
5. **Ask if blocked.** One brief clarifying question, not a brainstorm session.

## Style and Conventions

- Python 3.11+ type hints (`|`, `Self`, `list[str]`)
- Minimal public API surface
- Empty `__init__.py` files
- Polars over pandas in examples
- Integration tests over mocks
- Centralized branching logic

## Logger Design

- Logger protocol returns a context-managed run handle
- Minimal run API: `log_params`, `log_metrics`, `set_tags`, `log_artifact`, `log_model`, `finish`
- Adapters live in optional modules, not core

## Tooling

- `uv` for Python env and packages (not `uv pip`)
- `ruff` for linting
- `prek` for git hooks
- `pytest` with `pytest-markdown-docs` (docs are tests)
- `just test` runs with Optuna extra

## Testing Strategy

- **Docs are code.** All tutorial code fences are executed by pytest. If the docs lie, the build fails. This is non-negotiable.
- Integration tests exercise real sklearn flows.
- Fast and deterministic: small datasets, fixed seeds, low CV folds.
- Skip gracefully when optional deps missing.

---

## Documentation Philosophy

> **Docs teach. Code shows. Neither assumes.**

Documentation is a product. We ship it like one.

### The Three Principles

1. **Problem first, solution second.** Every explanation starts with *why* before *how*.
2. **Explain at point of use.** Don't front-load theory. Introduce concepts when the reader needs them.
3. **Link for depth, explain for correctness.** Provide enough context to use the feature correctly; link to authoritative sources for deeper dives.

### Writing Rules

**Never assume prior knowledge.** A junior data scientist should follow any tutorial. If you mention a concept, explain it or link to an explanation.

**Start with the problem.** Before explaining what something does, explain what problem it solves.

**Use concept boxes for theory:**
```markdown
> **Concept: Data Leakage**
>
> Data leakage occurs when information from outside the training set
> influences model training. sklab prevents this by requiring pipelines.
```

**Show "what just happened."** After code blocks, explain what the code did.

**Provide decision guides.** When multiple approaches exist, give a decision table:
```markdown
| Situation | Recommendation |
|-----------|----------------|
| Small grid | Grid search |
| Expensive evals | Optuna |
```

**Include "why it matters."** Connect concepts to practical consequences.

**End with next steps.** Link to related tutorials.

**Link to primary sources.** Cite papers for algorithms.

### Tone

- **Be direct:** "sklab requires pipelines" not "you might want to consider..."
- **Be confident:** "This prevents leakage" not "This can help prevent..."
- **Use second person:** "You'll notice..." not "One notices..."
- **Acknowledge tradeoffs:** Don't just praise; note limitations

### Code Examples

- **Must run.** Every code fence is tested. Broken examples fail the build.
- **Show imports.** Don't assume the reader knows which module a class comes from.
- **Use real datasets.** Prefer `load_iris`, `load_breast_cancer` for reproducibility.
- **Set random seeds.** Always `random_state=42` or `rng = np.random.default_rng(42)`.

### Tutorial Template

```markdown
# Title

**What you'll learn:**
- Bullet 1
- Bullet 2

**Prerequisites:** [Links]

## The Problem
[Why this matters]

## Concept: X (if needed)
[2-4 paragraphs with "Why it matters"]

## Implementation
[Code + "What just happened"]

## Best Practices
[Numbered list]

## Further Reading
[Links to papers, sklearn docs]
```

See [docs/developer/writing-docs.md](docs/developer/writing-docs.md) for the full style guide.

---

## Key Lessons

- Clean split between quick configs and power-user escape hatches
- Backend-agnostic core; adapters in optional modules
- Doc examples are runnable tests
- Explicit branching beats scattered conditionals
- Document tradeoffs directly in plans
