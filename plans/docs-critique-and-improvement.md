---
title: Documentation critique and improvement plan
description: Assessment of docs gaps and a roadmap to improve tutorials and theory
date: 2026-01-12
---

# Documentation Critique and Improvement Plan

## Executive Summary

The current documentation is **technically solid and well-organized**, but reads more like **reference material than learning material**. It assumes significant prior knowledge and jumps straight into code without motivating _why_ certain patterns matter. For a library whose core value proposition is "standardizing experiments," the docs don't yet tell a compelling story about what problems you're solving or why the design decisions matter.

---

## Part 1: Critique of Current State

### Strengths

1. **Clean, runnable examples**: Every tutorial is executable. This is excellent.
2. **Consistent structure**: Best practices, tradeoffs sections are valuable.
3. **Protocol-based extensibility**: Developer docs clearly explain the plugin architecture.
4. **Progressive disclosure**: Beginner → advanced flow exists.

### Weaknesses

#### 1. Missing "Why" Layer

The tutorials jump into _how_ without establishing _why_. For example:

- **Why use Experiment over raw sklearn?** The index mentions "standardizes," "logging," "reproducibility"—but these are assertions, not demonstrations.
- **Why keep preprocessing in the pipeline?** The glossary says "avoids leakage during cross-validation" but never explains what leakage is or why it's catastrophic.
- **Why Optuna over GridSearch?** The tutorials show both but don't explain when to choose which.

**Impact**: Users who already understand these concepts don't need the tutorials. Users who don't understand them won't learn from the current docs.

#### 2. Theory Vacuum

Critical concepts are mentioned but never explained:

| Concept | Current Treatment | What's Missing |
|---------|------------------|----------------|
| Data leakage | "avoids leakage" | What leakage is, how it manifests, why pipelines prevent it |
| Cross-validation | Shows `cv=3` | Why CV matters, variance vs. bias, when k-fold vs. stratified vs. time-series |
| Hyperparameter search | Shows code | Search space design, curse of dimensionality, sample efficiency |
| TPE sampler | Shows `TPESampler()` | What TPE is, why it's better than random for expensive objectives |
| Pruning | Shows `HyperbandPruner()` | What early stopping achieves, when to use it |

#### 3. Tone is Flat and Procedural

Current tone: "Use X to do Y. Here is code."

This is fine for API reference but makes tutorials feel like checklists rather than guided learning experiences. Compare:

**Current**:
> Use `fit()` for training and `evaluate()` for a quick metric on held-out data.

**Better**:
> After fitting, you need to know if your model generalizes. `evaluate()` scores the fitted estimator on held-out data—data the model has never seen. If the holdout score is much lower than training performance, you're likely overfitting.

#### 4. Repetition Without Progression

The same pipeline (StandardScaler + LogisticRegression on Iris) appears in almost every tutorial. This:
- Makes tutorials feel copy-pasted
- Doesn't demonstrate sklab's value across diverse problems
- Misses opportunities to show real complexity

#### 5. "Best Practices" Are Underjustified

Every tutorial ends with bullet points like:
> - Keep preprocessing inside the pipeline to avoid leakage.

But this is stated, not taught. A user who doesn't understand leakage won't be convinced by this line. A user who does understand leakage doesn't need the reminder.

#### 6. Missing Comparative Analysis

The sklearn-search tutorial shows Grid, Random, and Halving searches side-by-side—but doesn't compare their behavior on the same problem. Users can't see:
- How many evaluations each requires
- How close they get to the optimum
- When one dominates another

---

## Part 2: Philosophy for Theory Coverage

### Guiding Principle

**Explain concepts at the point of use, to the depth needed for correct usage, with links for deeper dives.**

This means:

1. **Don't assume prior knowledge** of ML concepts. A junior data scientist should be able to follow along.
2. **Don't write textbooks**. Provide enough context to use the feature correctly, then link to authoritative sources.
3. **Always connect theory to sklab**. Don't explain cross-validation in the abstract—explain it in terms of what `experiment.cross_validate()` is doing.

### Depth Guidelines

| Topic | Depth in Tutorials | Link Out To |
|-------|-------------------|-------------|
| Data leakage | 2-3 paragraphs with diagram | sklearn docs on pipelines, blog posts |
| Cross-validation | Explain variance reduction, show fold structure | sklearn user guide |
| Grid vs. Random search | Explain curse of dimensionality, when random wins | Bergstra & Bengio (2012) |
| Bayesian optimization (Optuna) | Explain surrogate model concept, acquisition functions at high level | Optuna docs, Shahriari et al. survey |
| TPE sampler | 1 paragraph on tree-structured Parzen estimators | Optuna paper |
| Pruning | Explain early stopping, Hyperband concept | Li et al. Hyperband paper |
| Time series CV | Explain why shuffling breaks causality | sklearn TimeSeriesSplit docs |

### Implementation: "Concept Boxes"

Introduce collapsible or highlighted "Concept" sections:

```markdown
> **Concept: Data Leakage**
>
> Data leakage occurs when information from outside the training set
> influences the model during training. A common example: fitting a
> StandardScaler on the full dataset before splitting into train/test.
> The scaler "sees" test data statistics, giving artificially optimistic
> results.
>
> sklab prevents this by keeping preprocessing inside the pipeline.
> sklearn's cross-validation fits the scaler separately on each fold's
> training data.
>
> Further reading: [sklearn Pipeline documentation](...)
```

---

## Part 3: Tutorial Restructuring Plan

### New Tutorial Hierarchy

```
tutorials/
├── 01-quickstart.md              # 5-minute intro, single fit/evaluate
├── 02-why-pipelines.md           # Leakage demo, pipeline motivation
├── 03-cross-validation.md        # CV concepts, when to use which splitter
├── 04-hyperparameter-search.md   # Grid → Random → Bayesian progression
├── 05-optuna-deep-dive.md        # TPE, pruning, study configuration
├── 06-logging-runs.md            # Why track experiments, adapter usage
├── 07-time-series.md             # Temporal CV, feature engineering
├── 08-classification-workflow.md # End-to-end classification project
├── 09-regression-workflow.md     # End-to-end regression project
└── 10-custom-extensions.md       # Searchers, loggers, advanced patterns
```

### Tutorial Template

Each tutorial should follow this structure:

```markdown
# Title

**What you'll learn**: [2-3 bullet points]

**Prerequisites**: [Links to prior tutorials or concepts]

## Motivation

[1-2 paragraphs explaining the problem this tutorial solves]

## Concept: [Key Concept]

[Explanation of the underlying theory, 2-4 paragraphs]

## Implementation

[Code with inline commentary]

## What Just Happened

[Explain what the code did, connecting back to the concept]

## Tradeoffs

[When to use this approach vs. alternatives]

## Further Reading

[Links to papers, sklearn docs, blog posts]
```

---

## Part 4: Specific Tutorial Rewrites

### Tutorial: Why Pipelines (NEW)

**Goal**: Demonstrate data leakage concretely, then show how pipelines prevent it.

```markdown
# Why Pipelines Matter

**What you'll learn**:
- What data leakage is and why it destroys model validity
- How sklearn pipelines prevent leakage automatically
- Why sklab enforces pipeline-first design

## The Problem: Data Leakage

Imagine you're building a model to predict house prices. You scale your
features using StandardScaler, then split into train and test sets.
Your model scores 95% on the test set. Ship it!

But in production, predictions are wildly wrong. What happened?

The scaler was fit on *all* the data, including the test set. When you
scaled the training data, you used statistics (mean, variance) that
included information from the test set. Your model "cheated" by seeing
the future.

This is **data leakage**: information from outside the training set
influencing the training process.

## Demonstration: Leakage vs. No Leakage

Let's see this concretely. We'll create a dataset where leakage
dramatically inflates apparent performance.

[Code showing same dataset with/without leakage, comparing scores]

## The Solution: Pipelines

sklearn Pipelines bundle preprocessing and modeling together. During
cross-validation, the *entire* pipeline is refit on each fold's training
data. The scaler never sees validation data.

[Code showing pipeline-based approach]

## How sklab Enforces This

sklab requires a Pipeline object. This isn't arbitrary—it's a
forcing function for correct experimental methodology.

[Code showing sklab usage]
```

### Tutorial: Hyperparameter Search (REWRITE)

**Goal**: Progress from Grid → Random → Bayesian with clear motivation at each step.

```markdown
# Hyperparameter Search: From Exhaustive to Intelligent

**What you'll learn**:
- Why hyperparameter tuning matters
- When grid search works and when it fails
- How random search beats grid search for high-dimensional spaces
- When Bayesian optimization (Optuna) is worth the complexity

## The Problem: Finding Good Hyperparameters

Most ML models have hyperparameters—settings that control learning but
aren't learned from data. A decision tree's `max_depth`, a neural
network's learning rate, a regularized model's penalty strength.

Bad hyperparameters → bad models. But the search space is often huge.

## Strategy 1: Grid Search

Grid search evaluates every combination of specified values.

**Pros**: Exhaustive, reproducible, easy to understand
**Cons**: Scales exponentially with parameters

[Code + visual showing grid]

### The Curse of Dimensionality

With 2 parameters and 10 values each: 100 evaluations.
With 5 parameters and 10 values each: 100,000 evaluations.

For expensive models, this is intractable.

## Strategy 2: Random Search

Random search samples randomly from the parameter space.

> **Key Insight**: If only a few parameters actually matter (which is
> common), random search finds good regions faster than grid search.
> See Bergstra & Bengio (2012) for the theory.

[Code + visual comparing random vs. grid coverage]

## Strategy 3: Bayesian Optimization (Optuna)

What if we could learn *during* the search which regions are promising?

Bayesian optimization builds a **surrogate model** of the objective
function, then uses it to decide which parameters to try next. It
balances:
- **Exploitation**: Try parameters similar to the best so far
- **Exploration**: Try uncertain regions that might be better

Optuna uses **TPE (Tree-structured Parzen Estimator)**, a variant that
models good and bad parameter regions separately.

[Code showing Optuna search]

### When to Use Each

| Strategy | Use When |
|----------|----------|
| Grid | Small space (≤3 params, ≤5 values each), need reproducibility |
| Random | Medium space, each evaluation is cheap |
| Optuna | Large space, expensive evaluations, can afford some overhead |

## Comparison: Same Problem, Three Strategies

[Code running all three on same dataset, comparing iterations to reach good score]
```

---

## Part 5: Tone and Style Guidelines

### Current Tone Problems

1. **Passive, procedural**: "Use X to do Y"
2. **No personality**: Reads like auto-generated docs
3. **No narrative arc**: Sections feel disconnected

### Recommended Tone

1. **Direct and confident**: "sklab does X because Y"
2. **Second person**: "You'll notice that..." "When you run this..."
3. **Problem-first**: Start with the pain, then the solution
4. **Honest about tradeoffs**: Don't oversell; acknowledge limitations

### Examples

**Before**:
> Sklab can search hyperparameters using native sklearn searchers.

**After**:
> You've built a pipeline and it works. But is it *good*? Hyperparameter
> search systematically tries different configurations to find better
> ones. sklab wraps sklearn's searchers to keep your code clean
> and your results logged.

**Before**:
> Use `TimeSeriesSplit` or explicit temporal splits; never shuffle time series.

**After**:
> Time series data has a hidden variable: time. If you shuffle before
> splitting, your model might train on January data and validate on
> December data—learning to predict the past from the future. Always
> use `TimeSeriesSplit` to respect temporal ordering.

---

## Part 6: Theory Deep-Dives for Search Algorithms

These sections should be comprehensive because search is where sklab adds the most value over raw sklearn.

### Grid Search Theory Section

```markdown
## How Grid Search Works

Grid search is exhaustive: it evaluates every point in a Cartesian
product of parameter values.

Given:
- `C`: [0.1, 1.0, 10.0]
- `gamma`: [0.01, 0.1]

Grid search evaluates: (0.1, 0.01), (0.1, 0.1), (1.0, 0.01), ...

**Complexity**: O(∏ᵢ |Vᵢ|) where Vᵢ is the set of values for parameter i.

**When it works**: Low-dimensional spaces where you can afford to be
exhaustive. Grid search guarantees you'll find the best combination
*among the points you specified*.

**When it fails**: High-dimensional spaces where the grid becomes
astronomically large. Also fails when optimal values fall between
grid points.
```

### Random Search Theory Section

```markdown
## How Random Search Works

Random search samples parameters independently from their distributions.

Given:
- `C`: log-uniform(0.01, 100)
- `gamma`: log-uniform(0.001, 1)

Each trial draws random values. After n trials, you keep the best.

**Key insight** (Bergstra & Bengio, 2012): In high dimensions, most
parameters don't matter much. If 2 of 10 parameters drive performance,
random search samples those 2 dimensions densely while grid search
wastes budget on irrelevant dimensions.

[Include figure from the paper or recreate it]

**When it works**: Medium-to-high dimensional spaces, especially when
you suspect only a few parameters matter.

**When it fails**: When all parameters matter equally and interactions
are complex.
```

### Optuna/Bayesian Optimization Theory Section

```markdown
## How Optuna Works: Bayesian Optimization with TPE

Bayesian optimization treats hyperparameter tuning as a sequential
decision problem. At each step, it asks: "Given what I've learned,
which parameters should I try next?"

### The Surrogate Model

Instead of modeling the objective directly, Optuna uses **TPE
(Tree-structured Parzen Estimator)**:

1. Split previous trials into "good" (top quantile) and "bad" (rest)
2. Build probability density estimates: p(params | good) and p(params | bad)
3. Suggest params that maximize p(good) / p(bad)

This differs from classic Gaussian Process-based BO, which models the
objective function directly. TPE scales better to high dimensions.

### Acquisition Functions

The ratio p(good)/p(bad) is related to the **Expected Improvement**
acquisition function. It balances:
- Exploiting known good regions
- Exploring uncertain regions

### Pruning with Hyperband

Optuna can stop unpromising trials early using **pruning**. Hyperband
is a principled early-stopping strategy:

1. Start many trials with small budgets (few epochs, small data)
2. Keep the best fraction, increase their budgets
3. Repeat until one trial remains

This dramatically reduces wasted computation on bad configurations.

### When to Use Optuna

- Evaluations are expensive (training takes minutes+)
- Search space is large (5+ parameters)
- You can define an intermediate metric for pruning

### When Grid/Random is Better

- Evaluations are cheap (seconds)
- You need exact reproducibility
- Space is small and you want exhaustive coverage

**Further reading**:
- [Optuna Paper](https://arxiv.org/abs/1907.10902)
- [TPE Paper](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization)
- [Hyperband Paper](https://arxiv.org/abs/1603.06560)
```

---

## Part 7: Links and Further Reading Strategy

### Principle

Every concept introduction should end with 1-2 links for users who want more.

### Link Categories

1. **Primary sources**: Papers (Bergstra & Bengio, Optuna paper, etc.)
2. **sklearn docs**: For sklearn-specific concepts
3. **Optuna docs**: For Optuna-specific features
4. **Curated blog posts**: For accessible explanations (e.g., Distill articles)

### Implementation

Create a `docs/references.md` with a bibliography, then link to anchors:

```markdown
See [Bergstra & Bengio, 2012](references.md#bergstra2012) for the
theoretical foundation of random search.
```

---

## Part 8: Prioritized Action Plan

### Phase 1: Foundation (High Impact)

1. **Rewrite index.md**: Add a "Why sklab?" section with concrete benefits
2. **Create "Why Pipelines" tutorial**: Demonstrate leakage concretely
3. **Expand glossary**: Add 2-3 sentences of explanation per term
4. **Add concept boxes**: To existing tutorials for leakage, CV, etc.

### Phase 2: Search Deep-Dive (Core Value)

5. **Rewrite sklearn-search.md**: Add theory sections, comparative analysis
6. **Rewrite optuna-search.md**: Add TPE explanation, acquisition functions
7. **Expand optuna-advanced.md**: Pruning theory, study configuration

### Phase 3: Polish

8. **Diversify examples**: Use different datasets across tutorials
9. **Add visualizations**: Search space coverage, CV fold diagrams
10. **Create references.md**: Centralized bibliography

---

## Part 9: Checklist for Each Tutorial

When writing or reviewing a tutorial, verify:

- [ ] **Motivation**: Does the intro explain *why* this matters?
- [ ] **Prerequisites**: Are required concepts linked or explained?
- [ ] **Concept boxes**: Is each new concept introduced with context?
- [ ] **Runnable code**: Does the code execute without modification?
- [ ] **Inline commentary**: Do code comments explain *why*, not just *what*?
- [ ] **Tradeoffs**: Are limitations and alternatives acknowledged?
- [ ] **Further reading**: Are there links for deeper exploration?
- [ ] **Connection to sklab**: Does the tutorial show sklab's value?

---

## Conclusion

Your documentation has a solid foundation—runnable examples, consistent structure, and good coverage of features. The main gap is the **explanatory layer**: helping users understand not just how to use sklab, but why its patterns lead to better experiments.

The recommendations above prioritize:
1. Teaching concepts at point of use
2. Providing deep coverage of search algorithms (sklab's differentiator)
3. Linking out for theory that's tangential to sklab itself
4. Maintaining a confident, problem-first tone

This approach keeps tutorials focused on sklab while ensuring users can follow along regardless of their ML background.
