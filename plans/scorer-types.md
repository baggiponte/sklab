---
title: Type-safe Scorer StrEnum
description: Add a StrEnum with all sklearn scorer names for autocomplete while keeping raw string flexibility
date: 2026-01-15
---

# Goal

Provide IDE autocomplete and discoverability for sklearn scorer names without sacrificing flexibility. Users can use `Scorer.ACCURACY` or `"accuracy"` interchangeably.

# References

- [sklearn model evaluation docs](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)
- `sklearn.metrics.get_scorer_names()` — returns all valid scorer strings
- `sklearn.metrics.get_scorer(name)._sign` — internal attribute: `1` = higher is better, `-1` = negated

# Design

## 1. Single `Scorer` StrEnum

One enum containing all sklearn scorers. No classification/regression split — simpler, more flexible:

```python
from enum import StrEnum

class Scorer(StrEnum):
    """sklearn scorer names with IDE autocomplete.
    
    All values are valid sklearn scorer strings. Use directly or pass raw strings.
    """
    # Classification
    ACCURACY = "accuracy"
    BALANCED_ACCURACY = "balanced_accuracy"
    TOP_K_ACCURACY = "top_k_accuracy"
    AVERAGE_PRECISION = "average_precision"
    NEG_BRIER_SCORE = "neg_brier_score"
    F1 = "f1"
    F1_MICRO = "f1_micro"
    F1_MACRO = "f1_macro"
    F1_WEIGHTED = "f1_weighted"
    F1_SAMPLES = "f1_samples"
    NEG_LOG_LOSS = "neg_log_loss"
    PRECISION = "precision"
    PRECISION_MICRO = "precision_micro"
    PRECISION_MACRO = "precision_macro"
    PRECISION_WEIGHTED = "precision_weighted"
    PRECISION_SAMPLES = "precision_samples"
    RECALL = "recall"
    RECALL_MICRO = "recall_micro"
    RECALL_MACRO = "recall_macro"
    RECALL_WEIGHTED = "recall_weighted"
    RECALL_SAMPLES = "recall_samples"
    JACCARD = "jaccard"
    JACCARD_MICRO = "jaccard_micro"
    JACCARD_MACRO = "jaccard_macro"
    JACCARD_WEIGHTED = "jaccard_weighted"
    JACCARD_SAMPLES = "jaccard_samples"
    ROC_AUC = "roc_auc"
    ROC_AUC_OVR = "roc_auc_ovr"
    ROC_AUC_OVO = "roc_auc_ovo"
    ROC_AUC_OVR_WEIGHTED = "roc_auc_ovr_weighted"
    ROC_AUC_OVO_WEIGHTED = "roc_auc_ovo_weighted"
    MATTHEWS_CORRCOEF = "matthews_corrcoef"
    
    # Regression
    R2 = "r2"
    EXPLAINED_VARIANCE = "explained_variance"
    NEG_MAX_ERROR = "neg_max_error"
    NEG_MEAN_ABSOLUTE_ERROR = "neg_mean_absolute_error"
    NEG_MEAN_SQUARED_ERROR = "neg_mean_squared_error"
    NEG_ROOT_MEAN_SQUARED_ERROR = "neg_root_mean_squared_error"
    NEG_MEAN_SQUARED_LOG_ERROR = "neg_mean_squared_log_error"
    NEG_ROOT_MEAN_SQUARED_LOG_ERROR = "neg_root_mean_squared_log_error"
    NEG_MEDIAN_ABSOLUTE_ERROR = "neg_median_absolute_error"
    NEG_MEAN_ABSOLUTE_PERCENTAGE_ERROR = "neg_mean_absolute_percentage_error"
    NEG_MEAN_POISSON_DEVIANCE = "neg_mean_poisson_deviance"
    NEG_MEAN_GAMMA_DEVIANCE = "neg_mean_gamma_deviance"
    D2_ABSOLUTE_ERROR_SCORE = "d2_absolute_error_score"
    
    # Clustering
    ADJUSTED_MUTUAL_INFO_SCORE = "adjusted_mutual_info_score"
    ADJUSTED_RAND_SCORE = "adjusted_rand_score"
    COMPLETENESS_SCORE = "completeness_score"
    FOWLKES_MALLOWS_SCORE = "fowlkes_mallows_score"
    HOMOGENEITY_SCORE = "homogeneity_score"
    MUTUAL_INFO_SCORE = "mutual_info_score"
    NORMALIZED_MUTUAL_INFO_SCORE = "normalized_mutual_info_score"
    RAND_SCORE = "rand_score"
    V_MEASURE_SCORE = "v_measure_score"
```

## 2. Updated type aliases

```python
from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

# Scorer callable: what get_scorer/make_scorer returns
# scorer(estimator, X, y, **kwargs) -> float
ScorerFunc: TypeAlias = Callable[..., float]

# What users can pass as a single scorer
Scoring: TypeAlias = ScorerName | str | ScorerFunc
```

**Usage in signatures:**
```python
def evaluate(
    estimator,
    X,
    y,
    scoring: Scoring | Sequence[Scoring] = ScorerName.ACCURACY,
) -> EvalResult:
    ...
```

**Changes from current design:**
- Drop `MetricFunc` — sklab expects scorer callables, not raw metrics
- Drop `Scorers` mapping — accept `Sequence[Scoring]` directly in signatures
- Rename `Scorer` enum to `ScorerName` to avoid collision with existing alias

## 3. Runtime utilities

### Validation helper

```python
from sklearn.metrics import get_scorer_names

def is_valid_scorer(name: str) -> bool:
    """Check if a scorer name is valid in sklearn."""
    return name in get_scorer_names()
```

### Scorer metadata (best-effort)

```python
from dataclasses import dataclass
from sklearn.metrics import get_scorer

@dataclass(frozen=True, slots=True)
class ScorerInfo:
    """Metadata about a scorer."""
    name: str
    greater_is_better: bool | None  # None if unknown

def get_scorer_info(name: str) -> ScorerInfo:
    """Get metadata about a scorer. Raises if invalid."""
    scorer = get_scorer(name)
    sign = getattr(scorer, "_sign", None)
    greater_is_better = None if sign is None else (sign == 1)
    return ScorerInfo(name=name, greater_is_better=greater_is_better)
```

## 4. Sync test

Ensures enum values remain valid sklearn scorers:

```python
import pytest
from sklearn.metrics import get_scorer_names
from sklab import Scorer

def test_scorer_enum_values_are_valid_sklearn_scorers():
    """All Scorer enum values must be valid sklearn scorer names."""
    valid = set(get_scorer_names())
    invalid = [s.value for s in Scorer if s.value not in valid]
    assert not invalid, f"Invalid scorer names: {invalid}"
```

# How to test

1. Unit test: `test_scorer_enum_values_are_valid_sklearn_scorers` — fails if sklearn removes a scorer
2. Unit test: `test_is_valid_scorer` — returns True for known scorers, False for garbage
3. Unit test: `test_get_scorer_info` — returns correct `greater_is_better` for known scorers
4. Integration: existing experiment tests should continue to work with both `Scorer.ACCURACY` and `"accuracy"`

# Future considerations

- **Autocomplete degradation:** If `str` in the union degrades IDE completion, consider task-specific overloads on public APIs
- **New sklearn scorers:** The sync test only checks our enum is valid, not complete. New sklearn scorers won't be in the enum until manually added. This is intentional (curated list).
- **`_sign` stability:** If sklearn removes/renames `_sign`, `get_scorer_info` returns `greater_is_better=None` gracefully
