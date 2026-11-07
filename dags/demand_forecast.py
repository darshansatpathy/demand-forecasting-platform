"""Weekly demand forecast on Airflow: backtest, gate, publish, record.

The tasks call the same functions as `dfc run`. The policy gate is its own task, so a forecast
that fails backtesting stops the run before anything is published, and the failure is visible
in the Airflow UI. Set DFC_WORKDIR for where outputs go and MLFLOW_TRACKING_URI to record runs.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

try:  # Airflow 3
    from airflow.sdk import dag, task
except ImportError:  # pragma: no cover - Airflow 2
    from airflow.decorators import dag, task  # type: ignore[no-redef]

WORKDIR = Path(os.environ.get("DFC_WORKDIR", "/tmp/demand-forecast"))  # noqa: S108 - default for local runs


@dag(
    dag_id="demand_forecast",
    schedule="0 6 * * 1",
    start_date=datetime(2026, 1, 5),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["forecasting"],
)
def demand_forecast() -> None:
    @task
    def backtest() -> str:
        from demandcast import backtest as B
        from demandcast import forecast as F
        from demandcast import policy
        from demandcast.pipeline import load_inputs

        pol = policy.load()
        data = F.prepare(*load_inputs())
        result = B.run(data, pol.backtest.validation_origins, pol.backtest.test_origins, pol.backtest.interval)
        gate = policy.checks(result, pol)
        WORKDIR.mkdir(parents=True, exist_ok=True)
        path = WORKDIR / "backtest.json"
        path.write_text(json.dumps(result.to_dict() | {"checks": [c.__dict__ for c in gate]}, indent=1))
        return str(path)

    @task
    def gate(path: str) -> str:
        failed = [c for c in json.loads(Path(path).read_text())["checks"] if not c["passed"]]
        if failed:
            raise ValueError("forecast policy failed: " + "; ".join(c["name"] for c in failed))
        return path

    @task
    def publish(_: str) -> str:
        from demandcast import pipeline, policy

        result = pipeline.run(policy.load(), WORKDIR, tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"))
        if result.published is None:
            raise ValueError("forecast policy failed at publish time")
        return str(WORKDIR / "forecast")

    publish(gate(backtest()))


demand_forecast()
