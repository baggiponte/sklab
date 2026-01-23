---
title: Result object design
description: Brainstorm what Result objects should contain, their API shape, and logger platform integration
date: 2026-01-16
---

# Result object design

## Goal

Define a consistent Result API across `fit()`, `evaluate()`, `cross_validate()`, and `search()` that:
1. Contains everything needed to plot/analyze without re-running
2. Stays lean—no bloat, just what's useful
3. Optionally exposes logger run references for downstream artifact retrieval

## References

- `src/sklab/experiment.py` (current FitResult/EvalResult/CVResult/SearchResult)
- `plans/run-result.md` (earlier draft)
- `src/sklab/adapters/logging.py` (LoggerProtocol)

---

## Current state

| Result | Fields |
|--------|--------|
| `FitResult` | `estimator`, `metrics`, `params` |
| `EvalResult` | `metrics` |
| `CVResult` | `metrics`, `fold_metrics`, `estimator` |
| `SearchResult` | `best_params`, `best_score`, `estimator` |

**What's missing:**
- Predictions (`y_pred`, `y_proba`)
- Ground truth (`y_true`)
- Logger run reference (for artifact retrieval)
- Timing info (fit time, score time)

---

## Design questions

### 1. Data container vs. methods?

**Option A: Pure dataclass (data container)**
```python
@dataclass(slots=True)
class CVResult:
    metrics: Mapping[str, float]
    fold_metrics: Mapping[str, list[float]]
    estimator: Any | None
    y_true: ArrayLike | None
    y_pred: ArrayLike | None
    y_proba: ArrayLike | None
    run_ref: RunRef | None  # for logger lookups
```

Pros:
- Simple, transparent, serializable
- No hidden behavior
- Matches sklearn's `cross_validate()` returning a dict

Cons:
- Plotting/reporting must live elsewhere

**Option B: Dataclass with convenience methods**
```python
@dataclass(slots=True)
class CVResult:
    # ... fields ...

    def confusion_matrix(self) -> ConfusionMatrix:
        """Compute confusion matrix from stored predictions."""
        ...

    def classification_report(self) -> dict:
        """Compute classification report from stored predictions."""
        ...
```

Pros:
- Convenient for common tasks
- Discoverable API

Cons:
- Grows over time (scope creep)
- Mixes data and behavior
- Classification-specific methods don't make sense for regression

**Recommendation: Pure dataclass + external utilities**

Keep results as dumb data containers. Provide optional utilities:
```python
from sklab.reports import confusion_matrix, classification_report

result = exp.cross_validate(X, y, cv=5)
cm = confusion_matrix(result)  # Uses result.y_true, result.y_pred
```

This keeps the core lean and lets users use sklearn's built-in functions directly:
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(result.y_true, result.y_pred)
```

---

### 2. What predictions to store?

| Field | When | Type |
|-------|------|------|
| `y_pred` | Always (for eval/cv) | `ArrayLike` |
| `y_proba` | Classification + `predict_proba` available | `ArrayLike \| None` |
| `y_true` | Always (for eval/cv) | `ArrayLike` |

**For cross-validation:**
- `y_pred` and `y_true` should be the **out-of-fold** predictions/targets
- These enable proper error analysis without leakage
- Store as concatenated arrays (aligned by sample order)

**For search:**
- Only store predictions for the **best estimator** on the final refit
- Don't store all trial predictions (memory bloat)

---

### 3. Logger run reference

**Problem:** User runs an experiment, logs to MLflow/W&B. Later wants to retrieve artifacts (plots, model file) from the platform. Currently no connection info is preserved.

**Proposed solution: `RunRef` dataclass**
```python
@dataclass(frozen=True, slots=True)
class RunRef:
    """Reference to a logged experiment run."""
    run_id: str
    experiment_id: str | None = None
    experiment_name: str | None = None
    backend: str | None = None  # "mlflow", "wandb", etc.
```

**How it gets populated:**
- Logger adapters return run info when exiting context
- `LoggerProtocol` gains an optional property: `run_ref: RunRef | None`

```python
class MLflowLogger:
    def start_run(...) -> AbstractContextManager[Self]:
        ...
        # On __exit__, capture:
        self._run_ref = RunRef(
            run_id=mlflow.active_run().info.run_id,
            experiment_id=mlflow.active_run().info.experiment_id,
            experiment_name=...,
            backend="mlflow",
        )

    @property
    def run_ref(self) -> RunRef | None:
        return self._run_ref
```

**Usage:**
```python
result = exp.cross_validate(X, y, cv=5)
if result.run_ref:
    # Fetch artifacts from MLflow
    import mlflow
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(result.run_ref.run_id)
```

**Tradeoff:** This couples Result to logger implementation details. Alternative: just store `run_id: str | None` and let users figure out the backend.

---

### 4. Timing info

sklearn's `cross_validate()` returns `fit_time` and `score_time`. Should we expose?

**Recommendation:** Yes, but aggregated only (mean/std). Store in `metrics`:
```python
metrics = {
    "cv/accuracy_mean": 0.95,
    "cv/accuracy_std": 0.02,
    "cv/fit_time_mean": 1.23,
    "cv/fit_time_std": 0.15,
    "cv/score_time_mean": 0.05,
    "cv/score_time_std": 0.01,
}
```

Per-fold times go in `fold_metrics` alongside scores.

---

## Proposed unified Result shape

```python
@dataclass(slots=True)
class FitResult:
    estimator: Any
    params: Mapping[str, Any]
    metrics: Mapping[str, float]  # Empty for fit, but consistent shape
    run_ref: RunRef | None = None


@dataclass(slots=True)
class EvalResult:
    metrics: Mapping[str, float]
    y_true: ArrayLike
    y_pred: ArrayLike
    y_proba: ArrayLike | None = None
    run_ref: RunRef | None = None


@dataclass(slots=True)
class CVResult:
    metrics: Mapping[str, float]  # Aggregated: cv/accuracy_mean, cv/accuracy_std, etc.
    fold_metrics: Mapping[str, list[float]]  # Per-fold: accuracy, fit_time, etc.
    estimator: Any | None  # Final refitted estimator (if refit=True)
    y_true: ArrayLike  # Out-of-fold ground truth (ordered by fold)
    y_pred: ArrayLike  # Out-of-fold predictions (ordered by fold)
    y_proba: ArrayLike | None = None
    run_ref: RunRef | None = None


@dataclass(slots=True)
class SearchResult:
    best_params: Mapping[str, Any]
    best_score: float | None
    estimator: Any | None  # Best estimator (if refit=True)
    # Optional: predictions from final refit on full data
    y_pred: ArrayLike | None = None
    y_proba: ArrayLike | None = None
    run_ref: RunRef | None = None
```

---

## What we do NOT include

| Feature | Why not |
|---------|---------|
| Feature importances | Computed from `result.estimator`; many methods exist |
| SHAP values | Heavy dependency, user's choice of explainer |
| Plots | Generated from predictions; not serializable |
| Full trial history (search) | Logger backend's job (Optuna study, MLflow runs) |
| Raw fold estimators | Memory bloat; use callback if needed |

---

## Open questions

1. **Should `y_pred`/`y_true` be mandatory or optional?**
   - Mandatory = always useful, but memory cost
   - Optional = flag like `return_predictions=True`

2. **Should `RunRef` be a field or accessible via `result.run_ref`?**
   - Field = explicit, but couples to logger
   - Could be `None` when no logger or logger doesn't support it

3. **Generic `Result` base vs. separate classes?**
   - Separate is cleaner, no type narrowing needed
   - Could share a mixin/protocol for common fields

4. **Fold indices for CV?**
   - Store `fold_indices: list[ArrayLike]` to map predictions back to original samples?
   - Or just document that predictions are ordered by fold appearance?

---

## How to test

- Unit: Result dataclasses serialize/deserialize correctly
- Integration: `evaluate()` populates `y_pred`, `y_true`
- Integration: `cross_validate()` returns out-of-fold predictions aligned with targets
- Integration: `RunRef` is populated when using MLflow/W&B loggers

## Future considerations

- `result.to_polars()` / `result.to_pandas()` for predictions + metrics as DataFrame
- Caching large predictions to disk (with lazy loading)
- Multi-output prediction support

---

# Oracle Review (2026-01-16)

## TL;DR
The overall shape is sound and aligned with sklab's "sklearn-feel / no-bloat" philosophy: **results as simple dataclasses + external utilities** is the right default, and **storing out-of-fold predictions** is the correct call for CV. The two biggest gaps are (1) **missing "always captured" fields (params, estimator, y\_true/y\_pred) across all Result types**, and (2) **`RunRef` + `LoggerProtocol(start_run)->Self` is likely to become awkward/buggy for nested/multiple runs**—a tiny "run handle" protocol avoids that trap.

---

## 1) Pure dataclass + external utilities — right call?
**Yes.** It matches your principles:
- *Feels like sklearn*: results are transparent containers (like dicts from `cross_validate`) but typed.
- *No bloat*: avoids accumulating classification/regression-specific convenience methods on every result type.
- *Escape hatch friendly*: users can call sklearn metrics directly on `result.y_true / result.y_pred`.

**Recommendation (simple path):**
- Keep dataclasses "dumb".
- Put conveniences in `sklab.reports` / `sklab.viz` / `sklab.utils` functions that accept a `Result` protocol (duck-typing).
- Allow one minimal ergonomic helper if you really want discoverability without scope creep: e.g. `sklab.reports.available(result)` or `sklab.reports.summary(result)` (function, not method).

**Minor caveat:** "pure dataclass" doesn't automatically mean "serializable". Numpy arrays are fine; estimators can be pickled but are not always stable. That's OK—just be explicit in docs/tests.

---

## 2) Is the `RunRef` design appropriate for logger integration?
**Mostly, but adjust the interface boundary.** Storing a backend-agnostic run reference on the result is useful and keeps your promise: "later I can retrieve artifacts without re-running."

### What's good
- `RunRef` as a frozen dataclass is simple, explicit, and optional (`None` when not logged).
- Keeping the actual MLflow/W&B client usage out of sklab core is correct.

### Main issue: `LoggerProtocol.start_run(...) -> AbstractContextManager[Self]`
Yielding `Self` makes "per-run" metadata (like `run_ref`) awkward because:
- It encourages storing `self._run_ref` on the logger object, which is **global mutable state**.
- Nested runs (`nested=True`) or sequential runs can overwrite `_run_ref`.
- It's ambiguous whether `result.run_ref` should point to the *outer* run, a *nested* run, etc.

### Recommendation (still minimal, but safer)
Introduce a **separate run-handle protocol**:

```python
class RunProtocol(Protocol):
    run_ref: RunRef | None
    def log_params(self, params: Params) -> None: ...
    def log_metrics(self, metrics: Metrics, step: int | None = None) -> None: ...
    def set_tags(self, tags: Tags) -> None: ...
    def log_artifact(self, path: str, name: str | None = None) -> None: ...
    def log_model(self, model: Any, name: str | None = None) -> None: ...

class LoggerProtocol(Protocol):
    def start_run(...) -> AbstractContextManager[RunProtocol]: ...
```

This avoids state bleed and makes it obvious where `run_ref` lives.

### RunRef fields: what to change
Your proposed `RunRef` is MLflow-shaped (`experiment_id`, `experiment_name`). W&B doesn't map cleanly. Keep it **minimal and extensible**:

**Primary recommendation:**
```python
@dataclass(frozen=True, slots=True)
class RunRef:
    backend: str           # "mlflow", "wandb", "noop", etc.
    run_id: str            # backend-native id (mlflow run_id, wandb run.id)
    url: str | None = None # direct link if available
    extra: Mapping[str, str] = field(default_factory=dict)  # backend-specific (project, entity, experiment_id, tracking_uri...)
```

This prevents you from chasing every backend's naming scheme while still enabling artifact retrieval.

Effort: **M (1–3h)** to refactor the protocol cleanly; **S (<1h)** if you accept the risk and keep `Self` + mutable `_run_ref`.

---

## 3) Missing fields / problematic choices in the proposed Result shapes
### A. You currently violate your own "What We Always Capture" contract
`AGENTS.md` says every result includes:
- `y_pred`, `y_proba`, `y_true`, `params`, `estimator`, per-fold data (CV)

But the plan's unified shapes omit `params` and `estimator` in several places:
- `EvalResult` lacks **estimator** and **params**
- `CVResult` lacks **params**
- `SearchResult` lacks **params**
- `FitResult` has `metrics` but no predictions (fine), but then your "always y_pred/y_true" statement conflicts with "fit".

**Recommendation: decide and make it consistent:**
- Either soften the "always capture" statement ("always for evaluate/cv/search"), **or**
- Make `FitResult` include training-set predictions too (usually not desirable; can be misleading and expensive).

Given sklearn norms, I'd do:
- **FitResult:** estimator + params + run_ref + metrics (empty ok)
- **EvalResult/CVResult/SearchResult:** include estimator + params + y_true/y_pred (+ optional y_proba)

### B. CV predictions ordering: "ordered by fold appearance" is a footgun
Users typically want to align predictions back to the original sample order (and/or a pandas index). If you only concatenate by fold order, users must reconstruct alignment themselves.

**Simplest fix:** store OOF predictions **in original sample order**, not fold order.
- Preallocate `y_pred_oof` and fill by `test_idx`.
- This avoids needing `fold_indices` for basic plotting.

If you also want fold-level debugging, add one lightweight field:
- `fold_id: ArrayLike[int]` length `n_samples` (or `None` for samples never predicted).

This is simpler than storing `fold_indices: list[array]` and is more directly usable.

### C. Classification probabilities need `classes_` to be interpretable
`y_proba` without class order is ambiguous. For sklearn classifiers, probabilities are aligned to `estimator.classes_`.

**Recommendation:** add:
- `classes: ArrayLike | None = None`

Only set when `y_proba` is present.

### D. Consider a generic "score" field for non-proba classifiers (optional)
Many sklearn models expose `decision_function` instead of `predict_proba`, which is what you want for ROC/PR curves.

Minimal option (if you want it now): add
- `y_score: ArrayLike | None = None`  (decision function output)

If you want to stay extremely lean, skip it for now; but note it's a common "why can't I plot ROC?" pain point.

### E. SearchResult predictions: be explicit what dataset they're on
"predictions from final refit on full data" is fine, but ambiguous:
- Is it predictions on training data? on a provided eval set? Both?

**Recommendation:**
- Either don't include predictions in `SearchResult` by default (and let users call `evaluate()`), **or**
- Include `y_true` too and document the source dataset ("refit_predictions_on='train' | 'eval'").
Given your promise "ready to plot without rerunning", the best minimalist pattern is:
- Keep `SearchResult` about search metadata + best estimator + run_ref
- Encourage calling `exp.evaluate(X_eval, y_eval)` for predictions (clear semantics)

This avoids accidental training-set plots.

### F. Metrics schema: keep it consistent and non-duplicative
Storing CV mean/std inside `metrics` is good. Just ensure naming is consistent across APIs:
- For eval: `eval/accuracy`, etc (optional but nice)
- For CV: `cv/accuracy_mean`, `cv/accuracy_std`
- For search: `search/best_score` (or keep `best_score` field only)

I'd avoid having the same concept in both `best_score` and `metrics["best_score"]` unless there's a clear reason.

---

## 4) Open questions — clear answers
### Q1. Should `y_pred`/`y_true` be mandatory or optional?
Given your documented promise and the product goal ("plot/analyze without re-running"):
**Make them mandatory for `EvalResult` and `CVResult`.**

For memory concerns, the least surprising compromise is:
- Default: store predictions (no flags)
- Add an escape hatch later only if real users hit memory pain: `store_predictions=False` (explicit opt-out)

So: **mandatory now**, add opt-out only when needed.

### Q2. Should `RunRef` be a field or accessible via `result.run_ref`?
Field is fine; `result.run_ref` *is* the field.
**Yes: `run_ref: RunRef | None` on every result type** (or at least those that log runs).

Key improvement is not "field vs property", it's ensuring the logger API can produce a stable per-run ref (run-handle protocol).

### Q3. Generic `Result` base vs separate classes?
**Separate classes** is correct for sklearn-feel and clarity.
If you need polymorphism for utilities, use a lightweight `Protocol`:

```python
class PredictionResult(Protocol):
    y_true: ArrayLike
    y_pred: ArrayLike
    y_proba: ArrayLike | None
```

No inheritance required.

### Q4. Fold indices for CV?
Best answer: **don't store fold indices unless you need them.**
Instead:
- Store OOF predictions in original order
- Optionally store `fold_id` array for per-fold debugging

If you cannot/choose not to store in original order, then yes: store `test_indices` so alignment is recoverable. But "ordered by fold appearance" alone is not enough.

---

## Risks and guardrails
- **RunRef correctness risk** if you keep `start_run()->Self` and store `_run_ref` on the logger: nested/sequential runs can give wrong `run_ref`. Guardrail: move to run-handle protocol early.
- **Prediction memory cost**: can be large for big datasets / large proba matrices. Guardrail: document that results hold arrays; consider a future "log predictions as artifact + lazy load" only if users actually need it.
- **Estimator pickling/serialization**: don't promise perfect serialization; test pickling for common sklearn estimators only.

---

## When to consider the advanced path
Revisit a more complex design only if:
- Users routinely run on datasets where storing predictions is infeasible (hundreds of MB+),
- You need to support multiple logger backends simultaneously,
- You need robust persistence/reloading of results independent of Python environment.

Optional advanced path (outline only): store predictions as an artifact (parquet/npy) and keep only a pointer + `RunRef` + lazy loader. This is unnecessary now.

---

**Net: keep the plan, but (1) make "always captured" fields truly consistent across results, (2) fix CV alignment semantics, and (3) adjust logger protocol to return a per-run handle so `RunRef` is reliable.**
