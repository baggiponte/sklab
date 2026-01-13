from __future__ import annotations

from sklab.logging import MLflowLogger, NoOpLogger, WandbLogger


def test_noop_logger_run_context_and_methods() -> None:
    logger = NoOpLogger()
    with logger.start_run(name="noop", config={"a": 1}, tags={"x": "y"}) as run:
        assert run is logger
        assert run.log_params({"a": 1}) is None
        assert run.log_metrics({"acc": 0.9}) is None
        assert run.log_metrics({"acc": 0.9}, step=1) is None
        assert run.set_tags({"k": "v"}) is None
        assert run.log_artifact("/tmp/file.txt") is None
        assert run.log_model("/tmp/model.bin") is None


def test_mlflow_logger_requires_dependency() -> None:
    logger = MLflowLogger()
    with logger.start_run():
        pass


def test_wandb_logger_requires_dependency() -> None:
    logger = WandbLogger()
    with logger.start_run():
        pass
