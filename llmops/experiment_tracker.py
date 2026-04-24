"""MLflow experiment tracking wrapper for NetSentinel-RL."""
import mlflow, os, time
from typing import Dict, Any, Optional
from pathlib import Path
import logging
log = logging.getLogger(__name__)

class ExperimentTracker:
    def __init__(self, experiment_name="netsentinel-rl", tracking_uri="./mlruns"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = None

    def start_run(self, run_name: Optional[str]=None, tags: Optional[Dict]=None):
        self._run = mlflow.start_run(run_name=run_name or f"run_{int(time.time())}", tags=tags)
        log.info(f"MLflow run started: {self._run.info.run_id}")
        return self

    def log_params(self, params: Dict[str,Any]):
        mlflow.log_params({k:str(v)[:250] for k,v in params.items()})

    def log_metrics(self, metrics: Dict[str,float], step: Optional[int]=None):
        for k,v in metrics.items():
            try: mlflow.log_metric(k, float(v), step=step)
            except Exception as e: log.warning(f"Metric log failed {k}: {e}")

    def log_artifact(self, path: str):
        if Path(path).exists(): mlflow.log_artifact(path)

    def log_model_sb3(self, model, artifact_path="model"):
        try:
            import mlflow.sklearn
            path = f"/tmp/sb3_model_{int(time.time())}"
            model.save(path)
            mlflow.log_artifact(path+".zip", artifact_path)
        except Exception as e: log.warning(f"Model log failed: {e}")

    def end_run(self): mlflow.end_run()

    def __enter__(self): return self.start_run()
    def __exit__(self, *_): self.end_run()
