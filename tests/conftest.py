from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from demandcast import forecast as F
from demandcast.cli import main
from demandcast.generate import generate


@pytest.fixture(scope="session")
def inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    return generate()


@pytest.fixture(scope="session")
def data(inputs: tuple[pd.DataFrame, pd.DataFrame]) -> F.Data:
    return F.prepare(*inputs)


@pytest.fixture(scope="session")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One full weekly run through the CLI: backtest, gate, publish, MLflow."""
    out = tmp_path_factory.mktemp("run")
    assert main(["run", "--out", str(out), "--report", str(out / "report.md"), "--tracking-uri", f"sqlite:///{out}/mlflow.db", "--check"]) == 0
    return out
