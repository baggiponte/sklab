"""Tests for ScorerName enum."""

from sklearn.metrics import get_scorer_names

from sklab.type_aliases import ScorerName


def test_scorer_name_enum_values_are_valid_sklearn_scorers():
    """All ScorerName enum values must be valid sklearn scorer names."""
    valid = set(get_scorer_names())
    invalid = [s.value for s in ScorerName if s.value not in valid]
    assert not invalid, f"Invalid scorer names in ScorerName enum: {invalid}"


def test_scorer_name_is_string():
    """ScorerName values can be used as strings."""
    assert ScorerName.ACCURACY == "accuracy"
    assert ScorerName.F1 == "f1"
    assert ScorerName.ROC_AUC == "roc_auc"
    assert ScorerName.R2 == "r2"
