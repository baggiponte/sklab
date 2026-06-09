# ADR-010: Accept native tracker objects through integration adapters

## Status

Proposed.

## Context

ADR-004 simplified logging by making the logger itself the run handle returned by
`start_run()`. That kept the core protocol small, but it still asks users to
learn sklab-specific classes such as `MLflowLogger` or `WandbLogger` before they
can use tools they already know. This is friction in the first five minutes:
someone who already uses MLflow naturally reaches for `import mlflow`, not
`from sklab.logging import MLflowLogger`.

The user-facing API should preserve sklab's main abstraction. Users should
configure an `Experiment`, give it the tracker they already use, and let sklab
translate experiment events behind the scenes:

```text
import mlflow

from sklab.experiment import Experiment

with Experiment(pipeline=pipeline, scoring="accuracy").start(
    logger=mlflow,
    run_name="baseline",
) as exp:
    fit_result = exp.fit(X_train, y_train)
    eval_result = exp.evaluate(X_test, y_test)
```

The exact context-manager shape is not the important part. The important part is
that MLflow, W&B, or another tracker remains recognizable at the call site. sklab
should not make ordinary users instantiate a second wrapper class just to turn on
logging.

Hugging Face Transformers solves a related problem with `Trainer` callbacks and
reporting integrations. `TrainingArguments.report_to` selects integrations,
`get_reporting_integration_callbacks(...)` maps those selections to callback
classes, and concrete callbacks such as `WandbCallback` and `MLflowCallback`
hide backend-specific setup behind the main `Trainer` workflow. This confirms
the useful separation: the main experiment object owns the workflow, while
integration classes own backend translation.

That design also shows what sklab should avoid. Transformers uses string names
such as `"wandb"`, `"mlflow"`, `"all"`, and `"none"` for integration selection.
Those names are convenient for CLIs and JSON configs, but sklab is a Python
library without a CLI, and string dispatch adds another vocabulary users must
remember. sklab should prefer Python objects and types over string selectors,
using `StrEnum` only when a fixed closed set is genuinely part of the domain.

## Decision

We will separate the user-facing tracker input from the implementor-facing
integration interface.

The recommended design is native-object dispatch backed by a small adapter
registry. It keeps the everyday API focused on `Experiment`, while giving
integration authors a concrete class to subclass.

The public `Experiment` API will accept tracker objects that users already have:

- `None`, meaning no external tracking.
- Native tracker modules or clients, such as the imported `mlflow` module or a
  W&B run/client object.
- Advanced integration objects for custom or unusual backends.

The normal path should not require importing `sklab.logging.MLflowLogger` or
`sklab.logging.WandbLogger`. Those names may remain as compatibility shims during
the transition, but they should stop being the primary documented path.

Internally, sklab will resolve tracker objects through a small registry of
integration adapter classes. Each adapter class will answer whether it supports a
given object and will create a context-managed run bridge for sklab's standard
experiment events.

The implementor-facing API should be explicit and subclassable, for example:

```text
from abc import ABC, abstractmethod
from typing import Any


class TrackerIntegration(ABC):
    @classmethod
    @abstractmethod
    def supports(cls, tracker: Any) -> bool:
        """Return True when this integration can adapt the tracker object."""

    @abstractmethod
    def adapt(self, tracker: Any) -> LoggerProtocol:
        """Return sklab's logging bridge for this tracker object."""
```

`LoggerProtocol` remains the small internal run bridge for params, metrics,
tags, artifacts, models, and cleanup. If adapters eventually need operation
metadata, introduce a small typed context object carrying only data sklab owns:
run name, experiment name, tags, operation kind, and any initial params. If
operation kind needs a closed set, represent it as a `StrEnum`, not raw strings.

Resolution should be centralized:

1. If the value already implements the internal run protocol, use it.
2. Otherwise, find the first registered `TrackerIntegration` whose
   `supports(...)` method accepts the value.
3. If none match, raise a clear error explaining that the object is not a
   supported tracker and pointing custom integrations to the developer docs.

The core `Experiment` methods will continue to be the main workflow entry
points. A long-lived `Experiment.start(...)` context may group multiple
operations into one backend run. Outside such a context, individual operations
may keep the existing behavior of starting and closing their own run. The
important invariant is that fitted estimators, params, metrics, predictions,
probabilities, targets, and per-fold data remain captured automatically by the
experiment methods.

## Design Options

### Option A: Keep explicit sklab logger classes

This is the current shape:

```text
from sklab.logging import MLflowLogger

experiment = Experiment(
    pipeline=pipeline,
    scoring="accuracy",
    logger=MLflowLogger(experiment_name="demo"),
)
```

This is simple to implement and resembles PyTorch Lightning, where users create
`MLFlowLogger(...)` or `WandbLogger(...)` and pass it to `Trainer(logger=...)`.
It also gives each backend a typed configuration surface.

The cost is user-facing vocabulary. A user who already knows MLflow or W&B must
learn a sklab wrapper before logging works. That is acceptable in a framework
with a large trainer abstraction, but it is too much ceremony for sklab's thinner
experiment runner.

### Option B: Use string selectors

This mirrors Transformers:

```text
experiment = Experiment(
    pipeline=pipeline,
    scoring="accuracy",
    logger=Tracker.MLFLOW,
)
```

Using a `StrEnum` is safer than raw strings and gives a tidy closed set for
built-in integrations. It can work well when the user is loading settings from a
configuration file.

The cost is that it still asks Python users to learn sklab names for backends
they already imported. It also creates pressure to encode backend configuration
inside sklab, for example `tracker_config={...}`, which quickly becomes a second
configuration API for MLflow and W&B.

### Option C: Use callback-style integrations

This mirrors Keras and Transformers:

```text
experiment = Experiment(
    pipeline=pipeline,
    scoring="accuracy",
    callbacks=[MLflowCallback()],
)
```

Callbacks are useful when a framework owns a long training loop and exposes many
events. They are also familiar to users of Keras, where TensorBoard and W&B
integrations plug into `model.fit(..., callbacks=[...])`.

The cost is a new event model. sklab has four explicit operations, not a hidden
training loop. A callback system would make simple experiment tracking look like
a framework extension point and would invite unrelated hooks beyond the project
scope.

### Option D: Accept native tracker objects and adapt them internally

This is the recommended shape:

```text
import mlflow

experiment = Experiment(
    pipeline=pipeline,
    scoring="accuracy",
    logger=mlflow,
)
```

It is compatible with a grouped run shape:

```text
import mlflow

with Experiment(pipeline=pipeline, scoring="accuracy").start(logger=mlflow) as exp:
    exp.fit(X_train, y_train)
    exp.evaluate(X_test, y_test)
```

The value at the call site is the backend the user already knows. sklab owns only
the translation layer. This matches the project principle that sklab should wrap,
not trap.

The cost is runtime resolution. sklab must decide whether an object is MLflow,
W&B, a compatible custom logger, or unsupported. That branching should be
centralized in one resolver instead of leaking into experiment methods.

### Option E: Accept explicit integration instances for advanced users

This is a lower-level escape hatch:

```text
experiment = Experiment(
    pipeline=pipeline,
    scoring="accuracy",
    logger=CustomTrackerIntegration(client),
)
```

This is useful for custom backends, private tracker clients, or cases where
native object detection is ambiguous. It should be supported, but not be the
primary tutorial path.

## Recommended Implementation

Implement native-object dispatch in small steps.

First, preserve the existing `LoggerProtocol` as the internal run bridge. The
protocol is already the minimal surface `Experiment` needs: start a run, log
params, log metrics, set tags, log artifacts, and log models. Renaming it can
wait; the immediate improvement is to stop exposing sklab-specific logger
classes as the main user path.

Second, introduce a tiny resolver module:

```text
src/sklab/_logging/resolve.py
```

The resolver should expose one function:

```text
def resolve_logger(logger: LoggerLike | None) -> LoggerProtocol:
    ...
```

`LoggerLike` should include `None`, `LoggerProtocol`, native tracker modules or
clients, and `TrackerIntegration` instances. `Experiment.__post_init__()` should
call this once so the operation methods can keep using
`self.logger.start_run(...)` without new branching.

Third, introduce an integration base in the public adapter module:

```text
src/sklab/adapters/logging.py
```

The base class should be for extension authors, not ordinary users:

```text
class TrackerIntegration(ABC):
    @classmethod
    @abstractmethod
    def supports(cls, tracker: Any) -> bool: ...

    @abstractmethod
    def adapt(self, tracker: Any) -> LoggerProtocol: ...
```

Returning `LoggerProtocol` from `adapt(...)` is simpler than making every
integration implement the run methods directly. It lets built-in integrations
reuse the existing MLflow and W&B logger implementations internally, and it keeps
the first patch small.

Fourth, add built-in integrations:

```text
src/sklab/_logging/integrations.py
```

Initial integrations:

- `NoOpIntegration` for `None`.
- `ExistingLoggerIntegration` for objects that already satisfy
  `LoggerProtocol`.
- `MLflowIntegration` for the imported `mlflow` module.
- `WandbIntegration` for the imported `wandb` module and, if practical, active
  W&B run objects.

Detection should avoid importing optional dependencies just to check a value.
For module objects, use the module name:

```text
getattr(tracker, "__name__", None) == "mlflow"
getattr(tracker, "__name__", None) == "wandb"
```

If the user passed an actual module, the dependency is already present. Existing
compatibility classes can keep using `LazyModule` so `from sklab.logging import
MLflowLogger` does not eagerly import MLflow.

Fifth, keep operation methods unchanged except for a run kind. Add a small
`RunKind(StrEnum)` only if the adapter needs to distinguish operations:

```text
class RunKind(StrEnum):
    FIT = "fit"
    EVALUATE = "evaluate"
    CROSS_VALIDATE = "cross_validate"
    SEARCH = "search"
```

This is a good use of `StrEnum`: it is a closed sklab-owned vocabulary, and it
prevents scattering raw operation strings through adapters.

Sixth, defer grouped runs until the resolver is stable. The first user-facing
win is:

```text
experiment = Experiment(pipeline=pipeline, scoring="accuracy", logger=mlflow)
experiment.fit(X_train, y_train)
```

After that works, add `Experiment.start(...)` as a convenience context manager
that temporarily overrides the active logger/run context and lets multiple
operations share one backend run. That keeps the first implementation small and
avoids mixing tracker resolution with operation grouping.

The first implementation patch should touch only:

- `src/sklab/adapters/logging.py` for `TrackerIntegration` and type aliases.
- `src/sklab/_logging/resolve.py` for centralized resolution.
- `src/sklab/_logging/integrations.py` for built-in adapter registration.
- `src/sklab/experiment.py` for `logger` typing and `__post_init__`.
- `tests/test_logging_adapters.py` and `tests/test_experiment.py` for native
  module acceptance and backward compatibility.
- `docs/tutorials/logger-adapters.md` only after behavior exists.

## Consequences

Users get a more familiar first interaction:

```text
import mlflow

experiment = Experiment(pipeline=pipeline, scoring="accuracy", logger=mlflow)
result = experiment.fit(X_train, y_train)
```

The integration implementation becomes slightly more structured than the current
single `LoggerProtocol`, but that structure is on the contributor side, not the
ordinary user side. This is an acceptable tradeoff because adding a new tracker
should be a deliberate extension point with a clear contract.

The adapter registry gives sklab one place to handle optional dependency checks,
backend quirks, and error messages. It also keeps branching logic out of
`Experiment.fit()`, `evaluate()`, `cross_validate()`, and `search()`.

The design should not grow into a general callback framework. Transformers needs
callbacks because `Trainer` owns a long training loop with many events. sklab has
four explicit experiment operations over sklearn workflows. A broad callback
system would add mental model cost without solving the narrow tracker problem.

The design should not use string names such as `logger="mlflow"` for normal
Python usage. String selectors can be reconsidered only if sklab later adds a
configuration-file surface, and even then they should be typed through a
`StrEnum` at the boundary.

## Alternatives Considered

### Keep `MLflowLogger(...)` and `WandbLogger(...)` as the main API

This keeps the implementation simple, but it makes sklab's wrapper classes part
of the user's first interaction with tools they already know. That conflicts
with the project goal of wrapping without trapping.

### Use Transformers-style `report_to`

This is proven in a larger framework and works well for command-line training
arguments. It is less appropriate for sklab because the library does not need a
string-based configuration layer. Passing `logger=mlflow` is more Pythonic and
requires less new vocabulary than `report_to="mlflow"`.

### Require only structural protocols

Structural protocols are flexible, but they do not solve native tracker modules
cleanly. The imported `mlflow` module will not grow sklab-specific methods, and
contributors still need a documented place to translate backend semantics. A
subclassable integration base class gives extension authors that place while
preserving object-based user input.

## References

- [Transformers callback documentation](https://huggingface.co/docs/transformers/v4.55.4/main_classes/callback)
- [Transformers `TrainingArguments.report_to`](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py)
- [Transformers integration callback registry](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/integration_utils.py)
- [Transformers callback base and handler](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py)
- [Lightning logging documentation](https://lightning.ai/docs/pytorch/stable/extensions/logging.html)
- [Lightning MLFlowLogger documentation](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.mlflow.html)
- [Lightning WandbLogger documentation](https://lightning.ai/docs/pytorch/stable/extensions/generated/pytorch_lightning.loggers.WandbLogger.html)
- [W&B Keras integration documentation](https://docs.wandb.ai/models/integrations/keras)
- [MLflow Python API documentation](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html)
- [W&B `init` documentation](https://docs.wandb.ai/ref/python/functions/init)
