"""Experiment runner core types."""

from dataclasses import dataclass
from typing import Any, Mapping

from eksperiment.logging.interfaces import LoggerProtocol


@dataclass(slots=True)
class Experiment:
    """Bundle experiment inputs for an sklearn-style run."""

    pipeline: Any
    logger: LoggerProtocol | None = None
    optuna_config: Mapping[str, Any] | None = None
