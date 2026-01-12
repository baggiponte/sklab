# Experiment v1 API

## Goal
Define a minimal, sklearn-first `Experiment` API that separates training, evaluation, and tuning while keeping data handling and splits outside the core class. The spec should be explicit about intent (no silent hyperparameter search defaults) and align with the logger protocol.

## References
- `plans/logger-protocol.md`
- `AGENTS.md`

## Design
### Core principles
- Keep `Experiment` state minimal: pipeline + logger + optional name/tags. Do not store datasets.
- Make lifecycle explicit: `fit()` trains, `evaluate()` measures, `tune()` searches. No implicit GridSearchCV fallback.
- Keep split logic out of the core API; callers pass pre-split data or CV strategies.
- Logger uses a context-managed run handle with minimal run API.

### Proposed dataclass
- Fields:
- `pipeline: sklearn.pipeline.Pipeline`
- `logger: LoggerProtocol | None` (defaults to NoOpLogger when `None`)
- `scorers: Mapping[str, Scorer] | None = None`
- `name: str | None = None` (experiment name; run name can be provided per method)
- `tags: dict[str, str] | None = None`

### Methods
- `fit(X, y=None, *, params=None) -> FitResult`
  - Fits `pipeline` on provided data, applies optional `params` overrides.
  - Logs params and fitted model artifact.
  - No train/val split occurs here.

- `evaluate(estimator, X, y=None) -> EvalResult`
  - Computes metrics for a provided estimator (or optionally uses `self.pipeline` if already fitted; decide in implementation).
  - Uses `self.scorers` configured at init.
  - Logs metrics for the run.

- `cross_validate(X, y=None, *, cv, refit=True) -> CVResult`
  - Runs sklearn CV (e.g., `sklearn.model_selection.cross_validate`) with explicit `cv` splitter.
  - Uses `self.scorers` configured at init; logs fold metrics and aggregates.
  - If `refit=True`, fits a final estimator on full data and logs it.
  - Implementation should rely on sklearn primitives rather than calling `fit()`/`evaluate()` directly, but may share internal helpers for scorer resolution, logging shape, and final refit.

- `tune(search_config, X, y=None, *, cv, n_trials=None, timeout=None) -> TuneResult`
  - Runs hyperparameter search using an explicit `search_config`.
  - Uses `self.scorers` configured at init.
  - No implicit fallback to sklearn search if Optuna config is missing.
  - `search_config` should be a backend-agnostic union (e.g., Optuna config vs sklearn search config). Keep Optuna details in integration layer, not in core types.

### Search config shape (conceptual)
- `SearchConfig` union:
  - `OptunaSearchConfig`: study parameters + search space + objective config
  - `SklearnSearchConfig`: Grid/Randomized search params
- `tune()` dispatches based on config type; integration modules hold vendor-specific behavior.

### Data handling and splits
- `Experiment` does not perform train/test splitting.
- Callers provide `X_train`, `X_val`, etc., or pass `cv` splitters to `cross_validate()`/`tune()`.
- Provide optional helper utilities outside core API if needed later.

### Logging expectations
- Each method starts a logger run context and finishes it.
- Log `params`, `metrics`, `tags`, and `artifacts` in a consistent shape across methods.

## How to test
- Add integration tests covering:
  - `fit()` logs params + artifact and returns a fitted estimator.
  - `evaluate()` logs metrics for a provided estimator.
  - `cross_validate()` respects `cv` splitter and logs metrics; `refit=True` logs final model.
  - `tune()` requires explicit `search_config`; verify no fallback behavior.
- Use simple sklearn pipelines (e.g., `StandardScaler` + `LogisticRegression`) with small datasets.

## Future considerations
- `run_id` or backend run references on result objects (best-effort only, depends on logger support).
