# Why Pipelines Matter

**What you'll learn:**

- What data leakage is and why it invalidates your model
- How sklearn pipelines prevent leakage automatically
- Why eksperiment enforces pipeline-first design

**Prerequisites:** Basic familiarity with sklearn estimators and train/test splits.

## The problem: data leakage

Imagine you're building a model to predict house prices. Your workflow looks
reasonable:

1. Load the data
2. Scale features with StandardScaler
3. Split into train and test sets
4. Train a model on the training set
5. Evaluate on the test set

You get 95% accuracy on the test set. Ship it!

But in production, predictions are wildly wrong. What happened?

The scaler was fit on *all* the data—including the test set. When you scaled
the training data, you used statistics (mean, variance) computed from data
your model should never have seen. The model "cheated" by learning from the
future.

This is **data leakage**: information from outside the training set influencing
the training process. It's one of the most common and dangerous mistakes in
machine learning.

## Demonstration: leakage vs. no leakage

Let's see this concretely. We'll create a scenario where leakage dramatically
inflates apparent performance.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create data where test set has different distribution
rng = np.random.default_rng(42)

# Training distribution: centered at 0
X_train_raw = rng.normal(0, 1, size=(100, 5))
y_train = (X_train_raw[:, 0] > 0).astype(int)

# Test distribution: centered at 3 (different!)
X_test_raw = rng.normal(3, 1, size=(50, 5))
y_test = (X_test_raw[:, 0] > 3).astype(int)

X_all = np.vstack([X_train_raw, X_test_raw])
y_all = np.hstack([y_train, y_test])
```

### The wrong way: scale before splitting

```{.python continuation}
# WRONG: Fit scaler on ALL data (including test)
scaler_wrong = StandardScaler()
X_all_scaled = scaler_wrong.fit_transform(X_all)

# Now split
X_train_leaked = X_all_scaled[:100]
X_test_leaked = X_all_scaled[100:]

# Train and evaluate
model_leaked = LogisticRegression()
model_leaked.fit(X_train_leaked, y_train)
score_leaked = model_leaked.score(X_test_leaked, y_test)

print(f"Score with leakage: {score_leaked:.3f}")
```

### The right way: scale after splitting

```{.python continuation}
# RIGHT: Fit scaler only on training data
scaler_right = StandardScaler()
X_train_clean = scaler_right.fit_transform(X_train_raw)
X_test_clean = scaler_right.transform(X_test_raw)  # transform only!

# Train and evaluate
model_clean = LogisticRegression()
model_clean.fit(X_train_clean, y_train)
score_clean = model_clean.score(X_test_clean, y_test)

print(f"Score without leakage: {score_clean:.3f}")
```

The leaked model appears to perform better, but this is an illusion. In
production, you won't have access to future data to inform your scaling.

## Why this matters for cross-validation

The problem gets worse with cross-validation. If you scale before CV, *every*
fold's validation data leaks into *every* fold's training data through the
scaler statistics.

```{.python continuation}
from sklearn.model_selection import cross_val_score

# WRONG: Scale all data, then cross-validate
scaler = StandardScaler()
X_scaled_all = scaler.fit_transform(X_all)

scores_leaked = cross_val_score(
    LogisticRegression(),
    X_scaled_all,
    y_all,
    cv=5,
)

print(f"CV with leakage: {scores_leaked.mean():.3f} (+/- {scores_leaked.std():.3f})")
```

## The solution: sklearn Pipelines

Pipelines bundle preprocessing and modeling into a single estimator. When you
call `fit()` on a pipeline, it fits each step in sequence. When sklearn's
cross-validation fits a pipeline, it refits *the entire pipeline* on each
fold's training data.

This means the scaler never sees validation data.

```{.python continuation}
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression()),
])

# RIGHT: Cross-validate the pipeline
scores_clean = cross_val_score(pipeline, X_all, y_all, cv=5)

print(f"CV without leakage: {scores_clean.mean():.3f} (+/- {scores_clean.std():.3f})")
```

## How eksperiment enforces this

eksperiment requires a Pipeline object—not a raw estimator. This isn't a
limitation; it's a forcing function for correct methodology.

```{.python continuation}
from eksperiment.experiment import Experiment

experiment = Experiment(
    pipeline=pipeline,
    scorers={"accuracy": "accuracy"},
    name="leakage-demo",
)

# Every eksperiment method uses the pipeline correctly
cv_result = experiment.cross_validate(X_all, y_all, cv=5, run_name="cv")

print(f"eksperiment CV: {cv_result.metrics['cv/accuracy_mean']:.3f}")
```

When you use eksperiment:

- `fit()` fits the entire pipeline
- `evaluate()` uses the fitted pipeline to transform and predict
- `cross_validate()` refits the pipeline on each fold
- `search()` searches over pipeline parameters, refitting on each trial

You can't accidentally leak because the pipeline encapsulates the correct
order of operations.

## Common leakage patterns

Beyond scaling, watch out for these leakage sources:

| Pattern | Problem | Solution |
|---------|---------|----------|
| Feature selection on full data | Selected features "know" about test labels | Use `SelectKBest` inside pipeline |
| Target encoding on full data | Encoded values include test target info | Use `TargetEncoder` inside pipeline |
| Imputation on full data | Imputed values use test statistics | Use `SimpleImputer` inside pipeline |
| Oversampling before split | Synthetic samples from test distribution | Use `imblearn.Pipeline` with SMOTE |

## Best practices

1. **Always use pipelines.** Every preprocessing step that learns from data
   (scaling, encoding, imputation, feature selection) belongs in the pipeline.

2. **Split early.** If you must do exploratory analysis, split first and only
   look at training data.

3. **Be paranoid about temporal data.** Time series adds another dimension of
   leakage—you can't use future data to predict the past. Use `TimeSeriesSplit`.

4. **Validate with holdout.** Even with correct CV, keep a final holdout set
   that you only touch once, at the very end.

## Further reading

- [sklearn Pipeline documentation](https://scikit-learn.org/stable/modules/compose.html#pipeline)
- [sklearn Cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Common pitfalls in ML](https://scikit-learn.org/stable/common_pitfalls.html)
- Kaufman et al., "Leakage in Data Mining" (2012) — foundational paper on leakage taxonomy
