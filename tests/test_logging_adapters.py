from __future__ import annotations

import pytest

from eksperiment.logging.adapters import MLflowLogger, NoOpLogger, WandbLogger


def test_noop_logger_run_context_and_methods() -> None:
    logger = NoOpLogger()
    run = logger.start_run(name="noop", config={"a": 1}, tags={"x": "y"})
    with run as active:
        assert active is run
        assert active.log_params({"a": 1}) is None
        assert active.log_metrics({"acc": 0.9}) is None
        assert active.log_metrics({"acc": 0.9}, step=1) is None
        assert active.set_tags({"k": "v"}) is None
        assert active.log_artifact("/tmp/file.txt") is None
        assert active.log_model("/tmp/model.bin") is None
        assert active.finish() is None


def test_mlflow_logger_requires_dependency() -> None:
    logger = MLflowLogger()
    with pytest.raises(ModuleNotFoundError, match="mlflow is not installed"):
        logger.start_run()


def test_wandb_logger_requires_dependency() -> None:
    logger = WandbLogger()
    with pytest.raises(ModuleNotFoundError, match="wandb is not installed"):
        logger.start_run()
