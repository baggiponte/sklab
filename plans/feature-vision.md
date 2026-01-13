---
title: Feature Vision
description: Essential feature set for sklab based on real data scientist needs
date: 2026-01-12
---

# Feature Vision: What Sklab Should Be

## Goal
Define the essential feature set for sklab based on real data scientist needs. Strip away bloat. Focus on what makes experimentation faster and less tedious.

## References
- `AGENTS.md`
- `plans/experiment-v1.md`
- `plans/run-result.md`
- Research: Kaggle grandmaster workflows, MLflow/W&B pain points, sklearn limitations

## The Problem We Solve

Data scientists waste time on:
1. **Boilerplate** — Writing the same logging, plotting, and prediction-saving code for every experiment
2. **Forgotten artifacts** — Not saving predictions, then needing them later for analysis
3. **Scattered diagnostics** — Copy-pasting matplotlib code for confusion matrices, ROC curves, residual plots
4. **Opaque CV results** — `cross_val_score` returns numbers, not insights about fold variance or problematic samples
5. **Logger coupling** — Sprinkling `mlflow.log_*` calls throughout code, coupling experiment logic to a specific backend

## Core Principle

> "The faster you iterate, the better, because you try more things."

Sklab removes friction between "I have a pipeline" and "I understand what happened."

## Design

### What We Always Capture (Opinionated Defaults)

Every result object includes:

| Artifact | Why |
|----------|-----|
| `y_pred` | Post-hoc analysis always needs predictions |
| `y_proba` | Threshold tuning, calibration, ROC (classification only) |
| `y_true` | Can't analyze without ground truth |
| `params` | Know exactly what ran |
| `estimator` | Reproduce inference |
| Per-fold data (CV) | Debug variance, find bad samples |

No flags to enable. No forgetting. It just happens.

### Result Objects That Do Work

Results are not dumb data containers. They:

1. **Print cleanly** — `print(result)` shows a formatted metrics table
2. **Plot on demand** — `result.plot()` shows task-appropriate diagnostics
3. **Compare easily** — `Result.compare([r1, r2, r3])` returns a comparison DataFrame
4. **Export simply** — `result.to_dict()`, `result.to_json()`

### Automatic Diagnostics (Task-Aware)

Classification results offer:
- Confusion matrix (normalized + raw)
- ROC curve + AUC
- Precision-Recall curve
- Calibration curve
- Per-class metrics table

Regression results offer:
- Residuals vs predicted
- Residuals distribution
- Actual vs predicted scatter
- Q-Q plot

Both offer:
- Feature importances (when available)
- Learning curve (CV results)

API:
```python
result.plot()              # All "obvious" plots for this task
result.plot("confusion")   # Specific plot
result.plots               # Dict of available figures
```

### Per-Fold Transparency (CV)

CV results expose everything:
```python
result.fold_metrics       # Per-fold scores
result.fold_predictions   # Predictions on each held-out fold
result.fold_indices       # Which samples in each fold
result.metrics            # Aggregated mean +/- std
```

Use cases:
- Find samples the model consistently misclassifies
- Identify folds with data quality issues
- Debug stratification problems

### Search Trial Visibility

Search results expose all trials, not just the winner:
```python
result.best_params
result.best_score
result.trials             # DataFrame: all param combos + scores
result.plot("parallel")   # Parallel coordinates visualization
```

When a logger is attached, every trial is logged automatically.

### Zero-Sprinkling Logger Integration

Inject once, forget:
```python
experiment = Experiment(
    pipeline=pipeline,
    logger=MLflowLogger(...),
    scorers={"accuracy": "accuracy"},
)
# All methods now log automatically
```

What gets logged:
- All hyperparameters
- All metrics
- Predictions (configurable)
- Model artifact (configurable)
- Diagnostic plots (configurable)

No more:
```python
with mlflow.start_run():
    mlflow.log_param("C", C)
    # ... 20 lines of logging boilerplate
```

### Baseline Comparison Built-In

Every result can compare against a dummy baseline:
```python
result.vs_baseline        # Lift over DummyClassifier/DummyRegressor
```

Or shown in metrics output:
```
Metric      Score    vs Baseline
accuracy    0.847    +0.512
f1_macro    0.823    +0.489
```

### Sensible Defaults, Full Control

| Aspect | Default | Override |
|--------|---------|----------|
| Scorers | Inferred from task | `scorers={...}` |
| CV strategy | 5-fold stratified (clf) / 5-fold (reg) | `cv=TimeSeriesSplit(5)` |
| Refit after CV | No | `refit=True` |
| Predictions saved | Yes | `save_predictions=False` |
| Plots generated | On-demand | `result.plot()` |

## What We Do NOT Build

Guided by "DO ONE THING WELL. NO BLOAT":

| Feature | Why Not |
|---------|---------|
| Dataset loaders | sklearn, HF Datasets, Polars already do this |
| CLI | This is a library, not a framework |
| Pipeline templates | Belongs in docs/examples |
| Calibration/threshold tuning | Analysis tool, not experiment runner |
| Experiment registry/lookup | Logger backend's job (MLflow UI, W&B dashboard) |
| Distributed training | Different problem space |
| Model cards | Separate concern |

## How to Test

- Integration tests: Result objects contain predictions, probabilities, fold data
- Plotting tests: `result.plot()` returns figure objects without errors
- Comparison tests: `Result.compare()` produces valid DataFrame
- Logger tests: All artifacts logged when logger attached
- Doc tests: All tutorial code fences runnable

## Implementation Order

1. **Predictions in results** — Add `predictions`, `probabilities`, `targets` to result objects
2. **Per-fold data** — Expose `fold_predictions`, `fold_indices` in CVResult
3. **Result printing** — Clean `__str__` / `__repr__` for all result types
4. **Result comparison** — `Result.compare()` class method
5. **Automatic diagnostics** — Task-aware `plot()` method on results
6. **Search trial visibility** — `trials` DataFrame in SearchResult
7. **Baseline comparison** — `vs_baseline` property
8. **Automatic logging** — Logger captures predictions/plots when attached

## Future Considerations

- Caching large predictions to disk
- Multi-output prediction support
- Probabilistic predictions (prediction intervals)
- Export to common formats (ONNX, PMML) — but only if it stays simple
