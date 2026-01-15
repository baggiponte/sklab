"""Type aliases."""

from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Any, TypeAlias


class Direction(StrEnum):
    """Optimization direction for Optuna hyperparameter search."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


MetricFunc: TypeAlias = Callable[[Any, Any, Any], float]
Scorer: TypeAlias = MetricFunc | str
Scorers: TypeAlias = Mapping[str, Scorer]
