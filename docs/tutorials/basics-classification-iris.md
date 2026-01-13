# Classification Workflow

**What you'll learn:**

- How to structure a classification experiment with sklab
- The importance of stratified splits for balanced evaluation
- How to interpret holdout vs. training metrics

**Prerequisites:** Basic Python and sklearn familiarity. If you're new to pipelines,
read [Why Pipelines Matter](why-pipelines.md) first.

## The workflow at a glance

Every classification experiment follows the same pattern:

1. **Load and prepare data** — Get features (X) and labels (y)
2. **Split data** — Separate into training and holdout sets
3. **Build a pipeline** — Bundle preprocessing and model together
4. **Fit** — Train on the training set
5. **Evaluate** — Score on the holdout set

sklab standardizes steps 3-5 while keeping your code focused on the
data-specific parts (1-2).

---

## Step 1: Prepare data

We'll use the classic Iris dataset—150 samples of three flower species,
with 4 features each. Small and fast, perfect for learning the workflow.

```python
import polars as pl
from sklearn.datasets import load_iris

# Load the dataset
iris = load_iris()
feature_names = [name.replace(" (cm)", "") for name in iris.feature_names]

# Keep data in Polars for exploration, convert to NumPy for sklearn
iris_df = pl.DataFrame(iris.data, schema=feature_names)
iris_df = iris_df.with_columns(pl.Series("target", iris.target))

print(iris_df.head())
print(f"\nShape: {iris_df.shape}")
print(f"Classes: {iris_df['target'].unique().to_list()}")

X = iris_df.select(feature_names).to_numpy()
y = iris_df["target"].to_numpy()
```

---

## Step 2: Split into train and holdout

> **Concept: Train/Test Splits**
>
> The fundamental rule of ML evaluation: never evaluate on data you trained on.
> A model that memorizes training data looks perfect on that data but fails
> on new data. The holdout set simulates "new data" by keeping it hidden
> during training.
>
> **Stratification** ensures the class distribution in training and holdout
> sets matches the original data. Without it, you might accidentally put all
> examples of one class in the training set.

```{.python continuation}
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,        # 25% for holdout
    random_state=42,       # reproducibility
    stratify=y,            # preserve class balance
)

print(f"Training set: {len(X_train)} samples")
print(f"Holdout set: {len(X_test)} samples")
```

---

## Step 3: Build a pipeline and experiment

> **Concept: Why Pipelines?**
>
> If you scale features before splitting, the scaler "learns" statistics from
> the holdout data—this is data leakage. Pipelines prevent this by ensuring
> preprocessing is refit on each split's training data.
>
> See [Why Pipelines Matter](why-pipelines.md) for a detailed explanation.

```{.python continuation}
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklab.experiment import Experiment

# Bundle preprocessing and model
pipeline = Pipeline([
    ("scale", StandardScaler()),          # normalize features
    ("model", LogisticRegression(max_iter=200)),
])

# Create experiment with pipeline and scorers
experiment = Experiment(
    pipeline=pipeline,
    scorers={"accuracy": "accuracy"},
    name="iris-classification",
)
```

**What just happened:**

- `StandardScaler()` will center features to mean=0 and std=1
- `LogisticRegression` handles multi-class classification automatically
- The experiment wraps the pipeline with consistent scoring and logging

---

## Step 4: Fit the model

```{.python continuation}
fit_result = experiment.fit(X_train, y_train, run_name="iris-fit")

print(f"Fitted estimator: {type(fit_result.estimator).__name__}")
print(f"Logged params: {fit_result.params}")
```

**What `fit()` does:**

1. Clones the pipeline (so the original stays unchanged)
2. Fits the pipeline on (X_train, y_train)
3. Logs parameters and timing to the configured logger
4. Returns a `FitResult` with the fitted estimator and metadata

---

## Step 5: Evaluate on holdout data

```{.python continuation}
eval_result = experiment.evaluate(
    X_test,
    y_test,
    run_name="iris-eval",
)

print(f"Holdout accuracy: {eval_result.metrics['accuracy']:.4f}")
```

> **Concept: Holdout Evaluation**
>
> The holdout score estimates how well your model will perform on new,
> unseen data. If holdout accuracy is much lower than training accuracy,
> your model is overfitting—memorizing training patterns rather than
> learning generalizable ones.

---

## Putting it all together

Here's the complete workflow in one block:

```python
import polars as pl
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklab.experiment import Experiment

# 1. Load data
iris = load_iris()
feature_names = [name.replace(" (cm)", "") for name in iris.feature_names]
iris_df = pl.DataFrame(iris.data, schema=feature_names)
iris_df = iris_df.with_columns(pl.Series("target", iris.target))

X = iris_df.select(feature_names).to_numpy()
y = iris_df["target"].to_numpy()

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3. Build pipeline and experiment
pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=200)),
])

experiment = Experiment(
    pipeline=pipeline,
    scorers={"accuracy": "accuracy"},
    name="iris-classification",
)

# 4. Fit
fit_result = experiment.fit(X_train, y_train, run_name="fit")

# 5. Evaluate
eval_result = experiment.evaluate(X_test, y_test, run_name="eval")

print(f"Holdout accuracy: {eval_result.metrics['accuracy']:.4f}")
```

---

## Going further: cross-validation

A single holdout split is noisy—you might get lucky or unlucky with which
samples end up in the holdout set. Cross-validation averages over multiple
splits for a more robust estimate:

```{.python continuation}
cv_result = experiment.cross_validate(X, y, cv=5, run_name="cv")

print(f"CV accuracy: {cv_result.metrics['cv/accuracy_mean']:.4f}")
print(f"CV std: {cv_result.metrics['cv/accuracy_std']:.4f}")
```

> **Concept: Cross-Validation**
>
> Cross-validation splits the data into k folds, trains on k-1 folds, and
> evaluates on the remaining fold. This rotates through all folds, giving
> k scores that are averaged. The result is less sensitive to a single
> lucky/unlucky split.
>
> For classification, use `StratifiedKFold` (sklab uses this by default
> when you pass `cv=5` to `cross_validate()`).

---

## Best practices

1. **Always use stratified splits for classification.** Pass `stratify=y` to
   `train_test_split` and use `StratifiedKFold` for cross-validation.

2. **Keep preprocessing in the pipeline.** This prevents data leakage and
   ensures reproducible results.

3. **Start simple.** Logistic regression is a solid baseline. Only move to
   complex models if the baseline is insufficient.

4. **Check for overfitting.** If training accuracy is much higher than holdout
   accuracy, your model is memorizing rather than learning.

5. **Use cross-validation for model selection.** Reserve the holdout set for
   final evaluation only.

## Next steps

- [Cross-Validation](experiment.md#cross-validate) — Robust evaluation with multiple splits
- [Hyperparameter Search](sklearn-search.md) — Find better model configurations
- [Logging Adapters](logger-adapters.md) — Track experiments with MLflow or W&B
