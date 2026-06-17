"""Lightweight promoted model abstraction for inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class PromotedModel:
    """A promoted, inference-ready model derived from an experiment."""

    estimator: Any
    name: str | None
    source: str
    metrics: Mapping[str, float]
    params: Mapping[str, Any]

    def predict(self, X: Any) -> Any:  # noqa: N803
        """Run predict on the promoted estimator."""
        return self.estimator.predict(X)

    def predict_proba(self, X: Any) -> Any:  # noqa: N803
        """Run predict_proba on the promoted estimator."""
        if not hasattr(self.estimator, "predict_proba"):
            raise ValueError("This estimator does not implement predict_proba().")
        return self.estimator.predict_proba(X)
