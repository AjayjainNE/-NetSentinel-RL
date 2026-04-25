"""
NetSentinel-RL — Experiment Tracker
Thin wrapper around MLflow for consistent experiment naming and metric logging.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional

log = logging.getLogger(__name__)

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    log.warning("mlflow not installed — experiment tracking disabled.")


class ExperimentTracker:
    """
    Centralised experiment tracking for all NetSentinel training runs.

    Provides a clean context-manager interface over MLflow so individual
    training scripts don't need to manage run lifecycle directly.

    Usage
    -----
    tracker = ExperimentTracker("netsentinel-detector")
    with tracker.run("ppo_attention_v2"):
        tracker.log_params({"lr": 3e-4, "window": 5})
        tracker.log_metric("f1", 0.94, step=1000)
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "http://localhost:5000",
        artifact_root: Optional[str] = None,
    ) -> None:
        self.experiment_name = experiment_name
        self._run_id: Optional[str] = None

        if not _MLFLOW_AVAILABLE:
            return

        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            kwargs: Dict[str, Any] = {}
            if artifact_root:
                kwargs["artifact_location"] = artifact_root
            mlflow.create_experiment(experiment_name, **kwargs)
        mlflow.set_experiment(experiment_name)
        log.info("MLflow experiment: %s @ %s", experiment_name, tracking_uri)

    @contextmanager
    def run(
        self,
        run_name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> Generator[None, None, None]:
        """Context manager that wraps an MLflow active run."""
        if not _MLFLOW_AVAILABLE:
            yield
            return

        with mlflow.start_run(run_name=run_name, tags=tags or {}) as active_run:
            self._run_id = active_run.info.run_id
            log.info("MLflow run started: %s (id=%s)", run_name, self._run_id)
            try:
                yield
            finally:
                log.info("MLflow run finished: %s", self._run_id)

    def log_params(self, params: Dict[str, Any]) -> None:
        if _MLFLOW_AVAILABLE:
            mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        if _MLFLOW_AVAILABLE:
            mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if _MLFLOW_AVAILABLE:
            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        if _MLFLOW_AVAILABLE:
            mlflow.log_artifact(local_path, artifact_path)

    def log_model(self, model, artifact_path: str) -> None:
        if _MLFLOW_AVAILABLE:
            try:
                import mlflow.sklearn
                mlflow.sklearn.log_model(model, artifact_path)
            except Exception:
                try:
                    import mlflow.pytorch
                    mlflow.pytorch.log_model(model, artifact_path)
                except Exception as exc:
                    log.warning("Could not log model: %s", exc)

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id
