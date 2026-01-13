# Mixed Data Types: Titanic Classification

**What you'll learn:**

- How to handle datasets with both numeric and categorical features
- Why `ColumnTransformer` is essential for mixed-type preprocessing
- How to deal with missing values in a pipeline-safe way
- The importance of stratified CV for imbalanced classification

**Prerequisites:** [Classification Workflow](basics-classification-iris.md),
understanding of pipelines.

## The problem: real data is messy

The Iris dataset is clean—numeric features, no missing values, balanced classes.
Real-world data rarely cooperates like this.

The Titanic dataset represents reality better: passenger age (numeric, sometimes
missing), ticket fare (numeric), gender (categorical), embarkation port (categorical,
sometimes missing), and class (ordinal). Different feature types need different
preprocessing. Missing values must be handled. Class imbalance (more deaths than
survivals) affects evaluation.

This tutorial shows how to build a pipeline that handles all of this correctly,
without leaking information between training and test data.

---

## Step 1: Load and explore the data

```python
import pytest
pytest.importorskip("pandas")

from sklearn.datasets import fetch_openml

titanic = fetch_openml(data_id=40945, as_frame=True)
titanic_df = titanic.frame

print(f"Shape: {titanic_df.shape}")
print(f"\nFeatures:")
for col in ["pclass", "sex", "age", "fare", "embarked"]:
    missing = titanic_df[col].isna().sum()
    print(f"  {col}: {titanic_df[col].dtype}, {missing} missing")

print(f"\nTarget distribution:")
print(titanic_df["survived"].value_counts())
```

> **Concept: Mixed Data Types**
>
> Datasets often contain different types of features:
> - **Numeric:** age, fare, continuous measurements
> - **Categorical:** gender, port of embarkation, discrete labels
> - **Ordinal:** passenger class (1st > 2nd > 3rd), education level
>
> Each type needs different preprocessing. Scalers work on numbers. Encoders
> work on categories. Applying the wrong transform corrupts your data.
>
> **Why it matters:** A pipeline that treats "sex" as numeric will try to
> compute its mean—nonsense that sklearn might not catch.

---

## Step 2: Define feature groups

We need to tell sklearn which columns are which type, so it applies the
correct preprocessing to each.

```{.python continuation}
feature_cols = ["pclass", "sex", "age", "fare", "embarked"]
X = titanic_df[feature_cols].to_numpy()
y = titanic_df["survived"].to_numpy().astype(int)  # Convert string labels to int

# Column indices by type (after selecting feature_cols)
categorical_cols = [0, 1, 4]  # pclass, sex, embarked
numeric_cols = [2, 3]         # age, fare
```

---

## Step 3: Build the preprocessing pipeline

This is where `ColumnTransformer` shines. It routes different columns through
different transformers.

```{.python continuation}
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Categorical pipeline: impute missing, then one-hot encode
categorical_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

# Numeric pipeline: impute missing, then scale
numeric_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

# Combine with ColumnTransformer
preprocess = ColumnTransformer(
    transformers=[
        ("cat", categorical_pipeline, categorical_cols),
        ("num", numeric_pipeline, numeric_cols),
    ],
    remainder="drop",  # drop any columns not explicitly handled
)
```

**What this does:**

- **Categorical columns:** Fill missing with most frequent value, then create
  binary columns for each category
- **Numeric columns:** Fill missing with median, then scale to mean=0, std=1
- **Other columns:** Dropped (we've selected only the features we want)

> **Concept: Imputation Inside the Pipeline**
>
> Missing values must be handled *inside* the pipeline, not before. Why?
>
> If you impute before splitting, the imputer sees test data statistics—leakage.
> The median age in your training set shouldn't include test passengers.
>
> **Why it matters:** Inside the pipeline, imputation is refit on each fold's
> training data. The test fold's missing values are filled using only training
> statistics—the correct approach.

> **Concept: OneHotEncoder Options**
>
> `handle_unknown="ignore"` prevents errors when the test set contains
> categories not seen during training. Instead of crashing, it creates a
> row of zeros for that observation.
>
> **Why it matters:** In production, you might see a new embarkation port
> or edge case. The model should handle it gracefully, not crash.

---

## Step 4: Build the full pipeline and experiment

```{.python continuation}
from sklearn.linear_model import LogisticRegression

from sklab.experiment import Experiment

pipeline = Pipeline([
    ("prep", preprocess),
    ("model", LogisticRegression(max_iter=200)),
])

experiment = Experiment(
    pipeline=pipeline,
    scorers={"accuracy": "accuracy", "f1": "f1"},
    name="titanic",
)
```

---

## Step 5: Cross-validate with stratification

```{.python continuation}
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
result = experiment.cross_validate(X, y, cv=cv, run_name="titanic-cv")

print(f"Accuracy: {result.metrics['cv/accuracy_mean']:.3f} (+/- {result.metrics['cv/accuracy_std']:.3f})")
print(f"F1 Score: {result.metrics['cv/f1_mean']:.3f} (+/- {result.metrics['cv/f1_std']:.3f})")
```

> **Concept: Stratified Splits for Imbalanced Data**
>
> The Titanic dataset is imbalanced—more passengers died than survived. Random
> splits might accidentally put most survivors in one fold, distorting metrics.
>
> `StratifiedKFold` ensures each fold has the same proportion of survivors and
> non-survivors as the original data.
>
> **Why it matters:** Without stratification, fold metrics vary wildly based on
> random chance rather than model quality.

---

## Understanding the metrics

For imbalanced classification, accuracy alone is misleading. If 60% of passengers
died, a model that always predicts "died" gets 60% accuracy—useless but high-scoring.

**F1 score** balances precision and recall:
- **Precision:** Of passengers predicted to survive, how many actually did?
- **Recall:** Of passengers who survived, how many did we predict correctly?

A good model has high F1, not just high accuracy.

---

## Complete example

```python
import pytest
pytest.importorskip("pandas")

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklab.experiment import Experiment

# 1. Load data
titanic = fetch_openml(data_id=40945, as_frame=True)
titanic_df = titanic.frame

feature_cols = ["pclass", "sex", "age", "fare", "embarked"]
X = titanic_df[feature_cols].to_numpy()
y = titanic_df["survived"].to_numpy().astype(int)  # Convert string labels to int

categorical_cols = [0, 1, 4]
numeric_cols = [2, 3]

# 2. Build preprocessing
preprocess = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_cols,
        ),
        (
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            numeric_cols,
        ),
    ],
    remainder="drop",
)

# 3. Build full pipeline
pipeline = Pipeline([
    ("prep", preprocess),
    ("model", LogisticRegression(max_iter=200)),
])

# 4. Create experiment and cross-validate
experiment = Experiment(
    pipeline=pipeline,
    scorers={"accuracy": "accuracy", "f1": "f1"},
    name="titanic",
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
result = experiment.cross_validate(X, y, cv=cv, run_name="cv")

print(f"Accuracy: {result.metrics['cv/accuracy_mean']:.3f}")
print(f"F1 Score: {result.metrics['cv/f1_mean']:.3f}")
```

---

## Best practices

1. **Use ColumnTransformer for mixed types.** Don't try to preprocess everything
   the same way—different features need different handling.

2. **Impute inside the pipeline.** Missing value handling must be part of the
   pipeline to prevent leakage.

3. **Use stratified splits for imbalanced data.** Ensures each fold has
   representative class proportions.

4. **Look beyond accuracy.** F1, precision, recall, and AUC-ROC tell you more
   about real-world performance on imbalanced problems.

5. **Handle unknown categories.** Use `handle_unknown="ignore"` for production
   robustness.

## Tradeoffs

| Choice | Pros | Cons |
|--------|------|------|
| OneHotEncoder | No ordinal assumptions | High dimensionality |
| OrdinalEncoder | Compact | Implies ordering that may not exist |
| Median imputation | Robust to outliers | Ignores feature relationships |
| Model-based imputation | Uses correlations | Adds complexity, may overfit |

## Next steps

- [Hyperparameter Search](sklearn-search.md) — Find better model parameters
- [Why Pipelines Matter](why-pipelines.md) — Deeper dive into leakage prevention
- [Logger Adapters](logger-adapters.md) — Track these experiments
