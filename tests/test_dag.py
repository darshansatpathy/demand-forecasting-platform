"""Runs where Airflow is installed (the CI airflow job); skipped elsewhere."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_dag_imports() -> None:
    pytest.importorskip("airflow")
    from airflow.models import DagBag

    bag = DagBag(dag_folder=str(Path(__file__).parents[1] / "dags"))
    assert not bag.import_errors
    dag = bag.get_dag("demand_forecast")
    assert dag is not None and [t.task_id for t in dag.topological_sort()] == ["backtest", "gate", "publish"]
