---
title: Explain API Design
description: SHAP integration via Experiment.explain() method
date: 2025-01-23
---

# Explain API Design

## Goal

Add model explanation capabilities to sklab through a new `Experiment.explain()` method that computes SHAP values with zero boilerplate, following the same inject-once-forget philosophy as the rest of the library.

## The Problem

Computing SHAP values requires:
1. Choosing the right explainer for your model type
2. Managing background data sampling
3. Handling the explainer → values → visualization pipeline
4. Integrating with experiment tracking (logging plots/values)

This is tedious boilerplate that sklab should remove.

## Design Principles

1. **One method, one job** — `explain()` does SHAP. Nothing else.
2. **SHAP does SHAP well** — Don't wrap SHAP's plotting. Expose the `shap.Explanation` object.
3. **Separate concerns** — Explanation (why this prediction?) is distinct from diagnostics (how well does it perform?) and feature analysis (what features matter globally?).

## API Design

### Core Method

```python
class Experiment:
    def explain(
        self,
        X,
        *,
        method: Literal["auto", "tree", "kernel", "linear", "deep"] = "auto",
        background: ArrayLike | int | None = None,
        run_name: str | None = None,
    ) -> ExplainResult:
        """
        Compute SHAP values for the fitted estimator.

        Parameters
        ----------
        X : array-like
            Samples to explain.
        method : str, default="auto"
            Explainer type. "auto" selects based on estimator type.
        background : array-like or int, optional
            Background data for KernelExplainer/etc. If int, samples that many
            rows from X. If None, uses shap.maskers.Independent or model-appropriate default.
        run_name : str, optional
            Name for the logged run.

        Returns
        -------
        ExplainResult
            SHAP explanation with values, base values, and feature names.
        """
```

### Result Type

```python
@dataclass(slots=True)
class ExplainResult:
    """SHAP explanation result."""

    values: np.ndarray
    """SHAP values array. Shape: (n_samples, n_features) or (n_samples, n_features, n_classes)."""

    base_values: np.ndarray
    """Expected value(s) of the model output."""

    data: np.ndarray
    """The input data that was explained."""

    feature_names: list[str] | None
    """Feature names if available."""

    raw: shap.Explanation
    """The underlying shap.Explanation object for advanced use."""

    def plot(self, kind: str = "summary", **kwargs) -> None:
        """
        Convenience wrapper around SHAP plotting functions.

        Parameters
        ----------
        kind : str
            Plot type: "summary", "bar", "beeswarm", "waterfall", "force", "dependence".
        **kwargs
            Passed to the underlying shap plot function.
        """
        # Delegates to shap.plots.{kind}(self.raw, **kwargs)
```

### Explainer Selection ("auto" mode)

| Estimator Type | Explainer |
|----------------|-----------|
| Tree-based (RF, GBM, XGBoost, LightGBM, CatBoost) | `TreeExplainer` |
| Linear models (LogisticRegression, LinearRegression, Ridge, Lasso) | `LinearExplainer` |
| Neural networks (Keras, PyTorch via skorch) | `DeepExplainer` (if available) |
| Everything else | `KernelExplainer` (with background sampling) |

Detection uses `hasattr` checks and string matching on class names—no explicit type registry.

### Pipeline Handling

For sklearn Pipelines:
1. Extract the final estimator
2. Transform X through all preprocessing steps
3. Apply explainer to final estimator with transformed X
4. Map feature names back through transformers where possible

```python
# Internal logic
if hasattr(estimator, "steps"):  # It's a Pipeline
    preprocessor = Pipeline(estimator.steps[:-1])
    final_estimator = estimator.steps[-1][1]
    X_transformed = preprocessor.transform(X)
    explainer = select_explainer(final_estimator, X_transformed, method, background)
else:
    explainer = select_explainer(estimator, X, method, background)
```

### Logger Integration

When a logger is configured:
- Log SHAP summary plot as artifact
- Log mean absolute SHAP values as metrics (feature importance proxy)

```python
# Inside explain()
with self.logger.start_run(name=run_name, config={}, tags=self.tags):
    # ... compute shap_values ...

    # Log summary plot
    fig = shap.plots.beeswarm(explanation, show=False)
    self.logger.log_artifact(fig, "shap_summary.png")

    # Log mean |SHAP| per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    self.logger.log_metrics({
        f"shap_importance/{name}": val
        for name, val in zip(feature_names, mean_abs_shap)
    })
```

## What This Is NOT

### Not Feature Importance

Feature importance answers: "What features matter globally?"
- Permutation importance (model-agnostic, based on performance drop)
- Coefficient magnitude (linear models)
- Tree-based importance (impurity decrease / gain)

SHAP values answer: "Why did the model make *this* prediction?"
- Local explanations that can be aggregated globally
- Accounts for feature interactions
- Theoretically grounded (Shapley values)

**Recommendation:** Feature importance belongs in a separate `Experiment.feature_importance()` method or a diagnostics API, not in `explain()`.

### Not Partial Dependence Plots

PDPs answer: "How does changing feature X affect predictions on average?"
- Shows marginal effect of a feature
- Useful for understanding learned relationships
- sklearn has `PartialDependenceDisplay`

SHAP dependence plots show similar information but with interaction coloring and are SHAP-specific.

**Recommendation:** PDPs belong in a diagnostics API alongside learning curves, calibration plots, etc.

## Proposed API Landscape

```
Experiment.fit()           → FitResult           # Train
Experiment.evaluate()      → EvalResult          # Test metrics
Experiment.cross_validate() → CVResult           # CV metrics
Experiment.search()        → SearchResult        # Hyperparameter search
Experiment.explain()       → ExplainResult       # SHAP explanations (NEW)

# Future (separate from explain):
Experiment.diagnose()      → DiagnosticsResult   # Performance plots, calibration, etc.
Experiment.analyze()       → AnalysisResult      # Feature importance, PDPs, ICE
```

Or, if we want to keep the API minimal:

```
# Alternative: Single diagnostics entry point
Experiment.diagnose(
    X, y,
    include=["metrics", "importance", "pdp", "calibration", ...]
) → DiagnosticsResult
```

## Edge Cases

### Multi-output Models
Return SHAP values with shape `(n_samples, n_features, n_outputs)`. Let users slice as needed.

### No Fitted Estimator
Raise `ValueError("Call fit() or cross_validate(refit=True) before explain()")`.

### Unsupported Estimator
Fall back to `KernelExplainer` with a warning about computational cost.

### Large X
For `KernelExplainer`, default background to `shap.kmeans(X, 100)` or similar summarization.

## Dependencies

SHAP is an optional dependency:
```toml
[project.optional-dependencies]
shap = ["shap>=0.42"]
```

Use the existing `LazyModule` pattern (see `src/sklab/_lazy.py`):

```python
# In src/sklab/_explain.py (or wherever explain logic lives)
from sklab._lazy import LazyModule

shap = LazyModule("shap", install_hint="Install with: pip install sklab[shap]")

# Usage - import happens on first attribute access
def _create_explainer(estimator, X, method):
    if method == "tree" or _is_tree_model(estimator):
        return shap.TreeExplainer(estimator)
    # ...
```

This matches how `mlflow` and `wandb` are handled in the logging adapters.

## Related: Composite Runs

To group `search()` + `explain()` under a single logged run, see `plans/run-context-api-design.md` for the `Experiment.run()` context manager proposal.

```python
# With run context (proposed)
with exp.run(name="grid-search-with-explanation") as e:
    result = e.search(GridSearchConfig(...), X, y)
    explanation = e.explain(X_test)
    # Both logged as nested runs under one parent
```

## How to Test

1. **Unit tests** for explainer selection logic (mock estimators with specific attributes)
2. **Integration tests** with real pipelines:
   - `LogisticRegression` → `LinearExplainer`
   - `RandomForestClassifier` → `TreeExplainer`
   - `SVC(kernel='rbf')` → `KernelExplainer`
3. **Pipeline tests** verifying preprocessing is handled correctly
4. **Logger tests** verifying artifacts and metrics are logged
5. **Edge case tests** for unfitted estimator, empty X, etc.

## Future Considerations

1. **Caching** — SHAP computation is expensive. Consider optional result caching.
2. **Sampling** — For large datasets, option to explain a sample.
3. **Batch explanations** — Stream explanations for very large X.
4. **Interaction values** — `shap_interaction_values` for tree models.
5. **Text/image data** — Would require different maskers and explainers.

## References

- [SHAP documentation](https://shap.readthedocs.io/)
- [Lundberg & Lee, 2017 - "A Unified Approach to Interpreting Model Predictions"](https://arxiv.org/abs/1705.07874)
- [sklearn inspection module](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.inspection) (permutation importance, PDPs)

---

## Summary

| Question | Answer |
|----------|--------|
| What does `explain()` do? | Computes SHAP values for the fitted estimator |
| What does it return? | `ExplainResult` with values, base_values, data, and raw `shap.Explanation` |
| Does it wrap SHAP plotting? | Minimal convenience wrapper; exposes `raw` for full control |
| What about feature importance? | Separate concern → separate method or diagnostics API |
| What about PDPs? | Separate concern → diagnostics API |
| Is SHAP required? | Optional dependency, guarded import |
