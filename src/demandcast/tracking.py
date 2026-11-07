"""MLflow: the backtest metrics, policy checks and published model of each run."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
from mlflow import MlflowClient

from demandcast.backtest import Result
from demandcast.policy import Check, Policy

EXPERIMENT = "demand-forecast"
REGISTERED = "demand-forecast-global-lightgbm"


def log_run(tracking_uri: str, out: Path, result: Result, checks: list[Check], policy: Policy) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    experiment_id = (
        experiment.experiment_id if experiment else client.create_experiment(EXPERIMENT, artifact_location=(out / "mlartifacts").resolve().as_uri())
    )
    passed = all(c.passed for c in checks)
    with mlflow.start_run(experiment_id=experiment_id, run_name="weekly-forecast") as run:
        mlflow.set_tags({"policy_passed": str(passed)})
        mlflow.log_params(
            {
                "validation_origins": str(policy.backtest.validation_origins),
                "test_origins": str(policy.backtest.test_origins),
                "interval": policy.backtest.interval,
            }
        )
        for level, methods in result.metrics.items():
            for method, values in methods.items():
                mlflow.log_metrics({f"{level}.{method}.{k}": v for k, v in values.items()})
            mlflow.log_metric(f"{level}.coverage", result.coverage[level])
        mlflow.log_metric("checks_passed", sum(c.passed for c in checks))
        mlflow.log_artifact(str(out / "backtest.json"))
        mlflow.log_artifacts(str(out / "forecast"), artifact_path="forecast")
    if passed:
        try:
            client.get_registered_model(REGISTERED)
        except mlflow.exceptions.MlflowException:
            client.create_registered_model(REGISTERED, description="Global LightGBM over the store-product hierarchy, MinT reconciled")
        version = client.create_model_version(REGISTERED, f"runs:/{run.info.run_id}/forecast", run.info.run_id)
        client.set_registered_model_alias(REGISTERED, "champion", version.version)
    return str(run.info.run_id)
