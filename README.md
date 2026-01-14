![Python](https://img.shields.io/badge/python-3.11%2B-blue)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)

# 🧪 sklab

A zero-boilerplate experiment runner for sklearn pipelines. One thing, done well: **run experiments**.

**The promise:** Give me a pipeline, I'll give you answers.

## What It Does

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklab import Experiment
from sklab.search import GridSearchConfig

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression()),
])

experiment = Experiment(
    pipeline=pipeline,
    scorers={"accuracy": "accuracy", "f1": "f1_macro"},
)

experiment.fit(X_train, y_train)

result = experiment.evaluate(X_test, y_test)

result = experiment.cross_validate(X, y, cv=5)

result = experiment.search(GridSearchConfig(param_grid={...}), X, y, cv=5)
```

## 🪄 Why

sklab wants to help data scientist avoid:
- Writing the same logging code for every experiment
- Forgetting to save predictions, then needing them later
- Copy-pasting matplotlib code for confusion matrices and ROC curves
- Getting a single number from `cross_val_score` with no insight into fold variance

Sklab removes this friction. Results include predictions, probabilities, and diagnostics automatically. Inject a logger once, everything gets tracked. No sprinkling `mlflow.log_*` through your code.

## Install

```bash
uv add sklab

# With optional integrations
uv add "sklab[optuna]"   # Optuna search
uv add "sklab[mlflow]"   # MLflow logging
uv add "sklab[wandb]"    # W&B logging
```

## 🤗 Contributing

### Philosophy

Documentation, code and abstraction strive to adhere to the following principles:

- **Be useful** — Every feature solves a real pain point. If it doesn't help you iterate faster, it doesn't belong.
- **Provide value** — Every line of code must earn its place. We ship what helps, not what's clever.
- **Abstractions, not obstructions** — We remove tedium, not control. You can always drop down to raw sklearn.
- **Docs are code** — Every code example runs. If the docs lie, the build fails.
- **No bloat** — No distributed training, no deployment, no MLOps platform. Just experiments, done well.
- **Elegance stems from familiarity** — The API feels like sklearn because sklearn got it right, and that's what everybody uses. Don't make people learn new abstractions.
- **A library, not a framework** — Libraries use familiar concepts; frameworks invent new ones. Study what works in sklearn, HuggingFace, PyTorch - then adopt, don't reinvent. Every new abstraction must earn its place. Design slim wrappers users can see through.

### Coding guidelines

1. Disclose usage of AI Agents. You are free to use them to contribute. We strive to keep this codebase as agent-friendly as possible. However, you **must** own every line of code the agent writes. This means, as a starter, that you must be able to explain and justify the choice. No slop.
2. Start your feature request in the [discussions](https://github.com/baggiponte/sklab/discussions) tab. Once the core details are ironed out, we'll move it to the issue tracker.
3. Agents are encouraged to explore the [plans/](plans/) folder to get a sense of the big picture of the ongoing/relevant developments and to create a new plan if needed.
4. Code is now free to write. The value we bring is in the ideas, taste and judgment to assert the adherence to the principles above. Let's discuss thoroughly those ideas - including the final API - and code will follow naturally. In other words, treat code as an implementation detail, not as the end goal - this is, and always has been, the ideas we bring.
5. Keep changes small and reviewable

### Setup

```bash
uv sync
```

### Commands

```bash
just format    # Ruff format
just test      # Run tests with optuna extra
just lint      # Ruff check + type check
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
