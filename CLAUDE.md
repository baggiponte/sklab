# CLAUDE.md

Quick reference for AI agents working on this codebase.

## Project in One Sentence

Eksperiment is a zero-boilerplate experiment runner for sklearn pipelines. One thing, done well: **run experiments**. Automatically captures predictions, metrics, and diagnostics.

## The Four Methods

```python
experiment.fit(X, y)              # Train, log params
experiment.evaluate(est, X, y)    # Metrics + predictions + plots
experiment.cross_validate(X, y)   # Per-fold everything
experiment.search(config, X, y)   # All trials logged
```

## Philosophy

> **Be useful. No bloat. So elegant it's familiar. Abstractions that are not obstructions. Provide value.**

- **BE USEFUL** — Solve real pain points or don't build it
- **NO BLOAT** — No distributed, no deployment, no MLOps platform. Just experiments.
- **SO ELEGANT IT'S FAMILIAR** — Feel like sklearn, no new abstractions to learn
- **ABSTRACTIONS, NOT OBSTRUCTIONS** — Remove tedium, not control
- **PROVIDE VALUE** — Every feature must earn its place
- **DOCS ARE CODE** — Every code example runs. If the docs lie, the build fails.

## Results Are Not Dumb Containers

Results must:
- Always include predictions, probabilities (clf), targets
- Print as formatted metric tables
- Offer `.plot()` for task-appropriate diagnostics
- Support `Result.compare([r1, r2, r3])`
- Expose per-fold data for CV

## Logger = Inject Once, Forget

```python
Experiment(pipeline=p, logger=MLflowLogger(...))
# Everything logged automatically from here
```

No `mlflow.log_param()` scattered through code. Ever.

## Do Not Build

- Dataset loaders (sklearn does this)
- CLI (we're a library)
- Templates (docs/examples)
- Calibration tools (analysis, not experimentation)
- Experiment registry (logger backend's job)
- Distributed training (different problem, different tool)
- Deployment / model serving (we run experiments, not production)

## Before Writing Code

1. Check `plans/` for existing specs
2. If new feature, write a plan first with frontmatter + sections:
   ```
   ---
   title: Feature Name
   description: One-line summary
   ---
   # Feature Name
   ## Goal / ## References / ## Design / ## How to Test / ## Future Considerations
   ```
   Note: Only `plans/` files have frontmatter. Docs do not.
3. Sketch the API before implementing

## Commands

```bash
just test      # Run tests with optuna extra
just lint      # Ruff check
just format    # Ruff format
just docs      # Serve docs locally
```

## Code Style

- Python 3.11+ types: `list[str]`, `X | None`, `Self`
- Empty `__init__.py`
- Polars over pandas in examples
- Integration tests, not mocks
- **Docs are code** — all code fences run via `pytest-markdown-docs`. Broken examples fail the build.

## Key Files

- `src/eksperiment/experiment.py` — Core Experiment class
- `src/eksperiment/search.py` — Search configs (Grid, Random)
- `src/eksperiment/optuna.py` — Optional Optuna integration
- `plans/feature-vision.md` — Product vision and feature scope

## When in Doubt

1. Does this feature remove tedium for data scientists? If no, skip it.
2. Does sklearn/Polars/the logger already do this? If yes, don't duplicate.
3. Can you explain who needs this and why in one sentence? If no, rethink it.
