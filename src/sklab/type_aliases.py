"""Type aliases."""

from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

MetricFunc: TypeAlias = Callable[[Any, Any, Any], float]
Scorer: TypeAlias = MetricFunc | str
Scorers: TypeAlias = Mapping[str, Scorer]
