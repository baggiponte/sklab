# Writing Documentation

This guide defines how we write documentation for sklab. Follow it
religiously. Good docs are not optional—they're part of the product.

---

## The Philosophy

> **Docs teach. Code shows. Neither assumes.**

Our documentation philosophy in three sentences:

1. **Problem first, solution second.** Every explanation starts with *why*
   before *how*.
2. **Explain at point of use.** Don't front-load theory. Introduce concepts
   when the reader needs them.
3. **Link for depth, explain for correctness.** Provide enough context to use
   the feature correctly; link to authoritative sources for deeper dives.

---

## The Rules

### 1. Never Assume Prior Knowledge

A junior data scientist with basic Python and sklearn experience should be
able to follow any tutorial. If you mention a concept, explain it or link
to an explanation.

**Bad:**
```markdown
Use `TimeSeriesSplit` to avoid leakage.
```

**Good:**
```markdown
Use `TimeSeriesSplit` to avoid leakage—using future data to predict the
past. Unlike random k-fold splits, `TimeSeriesSplit` respects temporal
ordering by always training on earlier data and validating on later data.
```

### 2. Start With the Problem

Before explaining what something does, explain what problem it solves. The
reader needs motivation before mechanics.

**Bad:**
````markdown
## Grid Search

Grid search evaluates all combinations of parameter values.

```python
GridSearchConfig(param_grid={"model__C": [0.1, 1.0, 10.0]})
```
````

**Good:**
````markdown
## The Problem: Finding Good Hyperparameters

Most models have hyperparameters—settings that control learning but aren't
learned from data. Default values are reasonable starting points, but
rarely optimal for your specific data.

## Grid Search

Grid search solves this by systematically trying every combination of
parameter values you specify.

```python
GridSearchConfig(param_grid={"model__C": [0.1, 1.0, 10.0]})
```
````

### 3. Use Concept Boxes for Theory

When introducing a concept, use a blockquote "Concept box" that explains
the idea in 2-4 sentences:

```markdown
> **Concept: Data Leakage**
>
> Data leakage occurs when information from outside the training set
> influences model training. A common example: fitting a StandardScaler
> on all data before splitting. The scaler "sees" test set statistics,
> giving artificially optimistic results.
>
> sklab prevents this by requiring pipelines.
```

### 4. Show "What Just Happened"

After code blocks, explain what the code did. Don't assume the reader
understood everything from the code alone.

**Bad:**
````markdown
```python
result = experiment.cross_validate(X, y, cv=5)
```
````

**Good:**
````markdown
```python
result = experiment.cross_validate(X, y, cv=5)
```

**What this does:**

1. Splits data into 5 folds
2. Trains on 4 folds, evaluates on 1, rotating through all combinations
3. Returns mean and std of all metrics across folds
4. Logs each fold's results to the configured logger
````

### 5. Provide Decision Guides

When multiple approaches exist, give the reader a decision table:

```markdown
| Situation | Recommended Strategy |
|-----------|---------------------|
| Small grid (< 100 combinations) | Grid search |
| Medium space, cheap evaluations | Random search |
| Expensive evaluations | Optuna |
```

### 6. Always Include "Why It Matters"

For glossary entries and concept explanations, include a "Why it matters"
section that connects the concept to practical consequences:

```markdown
## Pipeline

A sklearn `Pipeline` that bundles preprocessing and modeling steps.

**Why it matters:** Pipelines prevent data leakage. If you scale features
before splitting, the scaler "sees" test set statistics during training.
A pipeline ensures preprocessing is refit on each fold's training data
during cross-validation.
```

### 7. End With Next Steps

Every tutorial should end with links to related content:

```markdown
## Next steps

- [Cross-Validation](experiment.md) — Robust evaluation with multiple splits
- [Hyperparameter Search](sklearn-search.md) — Find better configurations
- [References](../references.md) — Papers and external resources
```

### 8. Link to Primary Sources

For algorithms and theory, link to the original papers or authoritative
documentation:

```markdown
**Further reading:**
- Bergstra & Bengio (2012), [Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/v13/bergstra12a.html)
- [sklearn User Guide: Tuning hyperparameters](https://scikit-learn.org/stable/modules/grid_search.html)
```

---

## Tutorial Structure

Every tutorial follows this template:

```markdown
# Title

**What you'll learn:**

- Bullet point 1
- Bullet point 2
- Bullet point 3

**Prerequisites:** [Links to prior tutorials or concepts]

## The Problem / Motivation

[1-2 paragraphs explaining what problem this tutorial solves]

## Concept: [Key Concept] (if needed)

[2-4 paragraph explanation with "Why it matters"]

## Implementation

[Code with inline commentary]

### What Just Happened

[Explain what the code did]

## Best Practices

[Numbered list of actionable recommendations]

## Tradeoffs (if applicable)

[When to use this approach vs. alternatives]

## Further Reading

[Links to papers, sklearn docs, related tutorials]
```

---

## Tone Guidelines

### Do

- **Be direct:** "sklab requires pipelines" not "you might want to consider using pipelines"
- **Be confident:** "This prevents leakage" not "This can help prevent leakage"
- **Use second person:** "You'll notice that..." not "One notices that..."
- **Acknowledge tradeoffs:** "TPE may miss complex interactions" not just praise

### Don't

- **Don't hedge:** Avoid "might", "could", "perhaps" when stating facts
- **Don't oversell:** Don't claim sklab is "revolutionary" or "game-changing"
- **Don't use jargon without explanation:** Define terms at first use
- **Don't be cute:** No jokes, puns, or cleverness that obscures meaning

### Examples

**Bad tone:**
> You might want to consider using cross-validation, which could potentially
> give you a more robust estimate of your model's amazing performance!

**Good tone:**
> Use cross-validation for robust performance estimates. A single train/test
> split is noisy—you might get lucky or unlucky with which samples end up
> in the holdout set. Cross-validation averages over multiple splits.

---

## Code Examples

### Must Be Runnable

Every code fence in `docs/` is executed by pytest. If it doesn't run, the
build fails. This is non-negotiable.

### Use Continuation Blocks

For multi-part examples, use the Superfences format with
`{.python continuation}` to indicate code that continues from the previous
block:

```python
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from sklab.experiment import Experiment

X = np.zeros((10, 2))
y = np.zeros(10)
pipeline = Pipeline([("model", DummyClassifier(strategy="most_frequent"))])
scorers = {"accuracy": "accuracy"}

experiment = Experiment(pipeline=pipeline, scorers=scorers)
```

```{.python continuation}
result = experiment.fit(X, y, run_name="fit")
print(result.metrics)
```

### Include Imports

Always show imports. Don't assume the reader knows which module a class
comes from:

**Bad:**
````markdown
```python
result = experiment.search(GridSearchConfig(...), X, y)
```
````

**Good:**
````markdown
```python
from sklab.search import GridSearchConfig

result = experiment.search(GridSearchConfig(...), X, y)
```
````

### Use Real Datasets

Prefer sklearn's built-in datasets (`load_iris`, `load_breast_cancer`) for
reproducibility. When creating synthetic data, always set `random_state`:

```python
import numpy as np

rng = np.random.default_rng(42)
X = rng.normal(0, 1, size=(100, 5))
```

### Skip Optional Dependencies Gracefully

For optional features (Optuna, MLflow), guard with pytest.importorskip:

```python
import pytest
pytest.importorskip("optuna")

import optuna
# ... rest of example
```

---

## Glossary Entries

Every glossary entry follows this structure:

```markdown
## Term Name

[1-2 sentence definition]

**Why it matters:** [1-2 sentences connecting to practical consequences]

[Optional: code example]

[Optional: link to tutorial for more detail]
```

---

## What NOT to Document

- **Implementation details:** Don't explain how the code works internally
  unless it affects usage.
- **Edge cases:** Don't document every possible error; document the happy path.
- **Exhaustive API reference:** That's what docstrings and auto-generated
  API docs are for.
- **Features we don't have:** Don't apologize for missing features or
  promise future work.

---

## Checklist for Every Tutorial

Before submitting a tutorial, verify:

- [ ] Starts with "What you'll learn" and "Prerequisites"
- [ ] Explains the problem before the solution
- [ ] Every concept has a "Why it matters" or concept box
- [ ] All code runs without modification
- [ ] Includes "What just happened" after complex code blocks
- [ ] Has decision table if multiple approaches exist
- [ ] Ends with "Best practices" and "Next steps"
- [ ] Links to primary sources for algorithms/theory
- [ ] No unexplained jargon

---

## The Test

Read your documentation and ask:

1. **Would a junior data scientist understand this?** If no, add context.
2. **Does it explain WHY, not just HOW?** If no, add motivation.
3. **Does the code run?** If no, fix it.
4. **Would I want to read this?** If no, rewrite it.

Documentation is a product. Ship it like one.
