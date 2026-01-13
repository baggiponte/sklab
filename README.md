![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Ruff](https://img.shields.io/badge/lint-ruff-2f6feb)
![Ruff Format](https://img.shields.io/badge/format-ruff-2f6feb)
![Ty](https://img.shields.io/badge/type%20check-ty-6e56cf)

# Sklab

A zero-boilerplate experiment runner for sklearn pipelines. One thing, done well: **run experiments**.

**The promise:** Give me a pipeline, I'll give you answers.

## What It Does

```python
from sklab import Experiment
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression()),
])

experiment = Experiment(
    pipeline=pipeline,
    scorers={"accuracy": "accuracy", "f1": "f1_macro"},
)

# Fit — train, get params logged
result = experiment.fit(X_train, y_train)

# Evaluate — metrics + predictions + diagnostics, no boilerplate
result = experiment.evaluate(result.estimator, X_test, y_test)

# Cross-validate — per-fold transparency, not just a number
result = experiment.cross_validate(X, y, cv=5)

# Search — all trials logged, not just the winner
result = experiment.search(GridSearchConfig(param_grid={...}), X, y, cv=5)
```

## Why

Data scientists waste time on:
- Writing the same logging code for every experiment
- Forgetting to save predictions, then needing them later
- Copy-pasting matplotlib code for confusion matrices and ROC curves
- Getting a single number from `cross_val_score` with no insight into fold variance

Sklab removes this friction. Results include predictions, probabilities, and diagnostics automatically. Inject a logger once, everything gets tracked. No sprinkling `mlflow.log_*` through your code.

## Philosophy

> **Be useful. No bloat. So elegant it's familiar. Abstractions that are not obstructions. Provide value.**

- **BE USEFUL** — Every feature solves a real pain point. If it doesn't help you iterate faster, it doesn't belong.
- **NO BLOAT** — No distributed training, no deployment, no MLOps platform. Just experiments, done well.
- **SO ELEGANT IT'S FAMILIAR** — The API feels like sklearn because sklearn got it right. No new abstractions to learn.
- **ABSTRACTIONS, NOT OBSTRUCTIONS** — We remove tedium, not control. You can always drop down to raw sklearn.
- **PROVIDE VALUE** — Every line of code must earn its place. We ship what helps, not what's clever.
- **DOCS ARE CODE** — Every code example runs. If the docs lie, the build fails.

## Install

```bash
pip install sklab

# With optional integrations
pip install sklab[optuna]   # Optuna search
pip install sklab[mlflow]   # MLflow logging
pip install sklab[wandb]    # W&B logging
```

## Documentation

See the [docs](docs/) for tutorials and API reference.

Our docs follow a strict philosophy: **problem first, solution second**. Every tutorial explains *why* before *how*, never assumes prior knowledge, and includes runnable code examples tested by CI.

## Contributing

### Process

1. Draft a plan in `plans/` before coding (Goal, Design, How to test)
2. Sketch the API before implementing
3. Keep changes small and reviewable

### Setup

```bash
# Prerequisites: Python 3.11+, uv, just
uv sync                    # Install deps (dev + docs groups)
```

### Commands

```bash
just test      # Run tests with optuna extra
just lint      # Ruff check + type check
just format    # Ruff format
just docs      # Serve docs locally
```

### Testing

- **Docs are code**: every code fence in `docs/` is executed by pytest. If the docs lie, the build fails.
- Integration tests over mocks
- Fast and deterministic: small datasets, fixed seeds

### Writing Documentation

Documentation is a product. We ship it like one.

> **Docs teach. Code shows. Neither assumes.**

**The three principles:**

1. **Problem first, solution second.** Every explanation starts with *why* before *how*.
2. **Explain at point of use.** Don't front-load theory. Introduce concepts when the reader needs them.
3. **Link for depth, explain for correctness.** Provide enough context to use the feature correctly; link to authoritative sources for deeper dives.

**Quick rules:**

| Rule | Example |
|------|---------|
| Never assume | Don't say "avoid leakage"—explain what leakage is |
| Start with the problem | "Most models have hyperparameters that need tuning..." not "Grid search evaluates..." |
| Use concept boxes | `> **Concept: Data Leakage**` blockquotes for theory |
| Show "what happened" | Explain what code did after each block |
| Provide decision tables | When to use X vs Y in table format |
| Include "why it matters" | Connect concepts to practical consequences |
| End with next steps | Link to related tutorials |
| Cite sources | Link to papers for algorithms |

**Code examples must:**

- Run without modification (tested by pytest)
- Show all imports
- Use sklearn's built-in datasets
- Set random seeds for reproducibility

See [docs/developer/writing-docs.md](docs/developer/writing-docs.md) for the full style guide.

---

See [AGENTS.md](AGENTS.md) for full contributing guidelines.
