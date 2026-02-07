# References

A curated list of papers, documentation, and resources referenced throughout
the sklab documentation. Organized by topic for easy browsing.

---

## Hyperparameter Optimization

### Random Search

<a name="bergstra2012"></a>
**Bergstra, J., & Bengio, Y. (2012).** Random Search for Hyper-Parameter Optimization.
*Journal of Machine Learning Research, 13*, 281-305.

[Paper](https://www.jmlr.org/papers/v13/bergstra12a.html)

The foundational paper showing that random search often outperforms grid search,
especially in high-dimensional spaces where only a few parameters matter.

**Key insight:** Random search samples each parameter independently, so it
explores important dimensions densely regardless of how many unimportant
dimensions exist.

---

### Bayesian Optimization and TPE

<a name="bergstra2011"></a>
**Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011).** Algorithms for
Hyper-Parameter Optimization. *Advances in Neural Information Processing Systems, 24*.

[Paper](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization)

Introduces the **TPE (Tree-structured Parzen Estimator)** algorithm, which
models the density of good vs. bad configurations rather than the objective
function directly.

**Key insight:** TPE scales better to high dimensions than Gaussian Process-based
Bayesian optimization because it treats parameters independently.

---

<a name="optuna"></a>
**Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019).** Optuna: A
Next-generation Hyperparameter Optimization Framework. *Proceedings of the 25th
ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*.

[Paper](https://arxiv.org/abs/1907.10902) | [Documentation](https://optuna.readthedocs.io/)

The Optuna framework paper, describing its define-by-run API and efficient
TPE implementation.

---

### Early Stopping and Successive Halving

<a name="hyperband"></a>
**Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2018).**
Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization.
*Journal of Machine Learning Research, 18*, 1-52.

[Paper](https://arxiv.org/abs/1603.06560)

Introduces **Hyperband**, which combines successive halving with multiple
brackets to robustly handle different convergence rates.

**Key insight:** Running many configurations with small budgets, then
progressively eliminating the worst performers, uses resources more efficiently
than running all configurations to completion.

---

<a name="sha"></a>
**Jamieson, K., & Talwalkar, A. (2016).** Non-stochastic Best Arm Identification
and Hyperparameter Optimization. *Proceedings of the 19th International
Conference on Artificial Intelligence and Statistics (AISTATS)*.

[Paper](https://arxiv.org/abs/1502.07943)

Formalizes **successive halving** as a multi-armed bandit problem and proves
theoretical guarantees.

---

### Gaussian Process Bayesian Optimization

<a name="shahriari2016"></a>
**Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & de Freitas, N. (2016).**
Taking the Human Out of the Loop: A Review of Bayesian Optimization.
*Proceedings of the IEEE, 104*(1), 148-175.

[Paper](https://ieeexplore.ieee.org/document/7352306)

Comprehensive survey of Bayesian optimization, covering acquisition functions,
surrogate models, and practical considerations.

---

## Cross-Validation and Model Evaluation

<a name="sklearn-cv"></a>
**scikit-learn User Guide: Cross-validation**

[Documentation](https://scikit-learn.org/stable/modules/cross_validation.html)

Comprehensive guide to cross-validation strategies in sklearn, including
k-fold, stratified, time series, and grouped variants.

---

<a name="sklearn-pipeline"></a>
**scikit-learn User Guide: Pipelines and Composite Estimators**

[Documentation](https://scikit-learn.org/stable/modules/compose.html)

Official documentation on sklearn Pipelines, ColumnTransformers, and avoiding
data leakage.

---

## Data Leakage

<a name="kaufman2012"></a>
**Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012).** Leakage in
Data Mining: Formulation, Detection, and Avoidance. *ACM Transactions on
Knowledge Discovery from Data, 6*(4), 1-21.

[Paper](https://dl.acm.org/doi/10.1145/2382577.2382579)

Foundational paper defining and categorizing data leakage in machine learning.

**Key insight:** Leakage can occur through many subtle mechanisms—feature
engineering, sampling, temporal ordering—not just obvious train/test contamination.

---

<a name="sklearn-pitfalls"></a>
**scikit-learn: Common pitfalls and recommended practices**

[Documentation](https://scikit-learn.org/stable/common_pitfalls.html)

Official guide to common mistakes in ML workflows, including data leakage,
overfitting, and evaluation errors.

---

## External Tools

### Experiment Tracking

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html) — Open-source platform for ML lifecycle management
- [Weights & Biases Documentation](https://docs.wandb.ai/) — ML experiment tracking and visualization

### Hyperparameter Optimization

- [Optuna Documentation](https://optuna.readthedocs.io/) — Hyperparameter optimization framework
- [scikit-learn GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) — Exhaustive grid search
- [scikit-learn RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) — Randomized search
- [scikit-learn HalvingRandomSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingRandomSearchCV.html) — Successive halving

---

## How to cite sklab

If you use sklab in your research, please cite this repository:

```bibtex
@software{sklab,
  title = {sklab: A lightweight experiment runner for sklearn pipelines},
  url = {https://github.com/your-username/scikit-lab},
  year = {2024}
}
```
