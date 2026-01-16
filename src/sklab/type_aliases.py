"""Type aliases."""

from collections.abc import Callable
from enum import StrEnum
from typing import TypeAlias


class Direction(StrEnum):
    """Optimization direction for Optuna hyperparameter search."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ScorerName(StrEnum):
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
    D2_BRIER_SCORE = "d2_brier_score"
    D2_LOG_LOSS_SCORE = "d2_log_loss_score"
    POSITIVE_LIKELIHOOD_RATIO = "positive_likelihood_ratio"
    NEG_NEGATIVE_LIKELIHOOD_RATIO = "neg_negative_likelihood_ratio"

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


# Scorer callable: what get_scorer/make_scorer returns
# scorer(estimator, X, y, **kwargs) -> float
ScorerFunc: TypeAlias = Callable[..., float]

# What users can pass as a single scorer
Scoring: TypeAlias = ScorerName | str | ScorerFunc
