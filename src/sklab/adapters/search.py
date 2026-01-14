"""Protocols to add new searchers that are not supported by sklab."""

from collections.abc import Mapping
from typing import Any, Protocol

from sklab.type_aliases import Scorers


class SearcherProtocol(Protocol):
    """Minimal interface required by Experiment.search."""

    def fit(self, X: Any, y: Any | None = None) -> Any:  # noqa: N803
        ...

    best_params_: Mapping[str, Any] | None
    best_score_: float | None
    best_estimator_: Any | None


class SearchConfigProtocol(Protocol):
    """Config that can build a searcher for Experiment.search."""

    def create_searcher(
        self,
        *,
        pipeline: Any,
        scorers: Scorers | None,
        cv: Any | None,
        n_trials: int | None,
        timeout: float | None,
    ) -> SearcherProtocol: ...
