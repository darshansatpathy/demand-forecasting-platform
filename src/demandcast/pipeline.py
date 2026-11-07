"""One weekly run: data, backtest, policy gate, publish, record."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from demandcast import backtest as B
from demandcast import forecast as F
from demandcast.generate import generate
from demandcast.policy import Check, Policy, checks
from demandcast.publish import Published, publish


@dataclass
class RunResult:
    backtest: B.Result
    checks: list[Check]
    published: Published | None
    run_id: str | None = None

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def load_inputs(sales: Path | None = None, promotions: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sales is None:
        return generate()
    s = pd.read_csv(sales, parse_dates=["week"])
    p = pd.read_csv(promotions, parse_dates=["week"]) if promotions else pd.DataFrame(columns=["sku", "week", "discount", "base_price"])
    return s, p


def run(policy: Policy, out: Path, sales: Path | None = None, promotions: Path | None = None, tracking_uri: str | None = None) -> RunResult:
    data = F.prepare(*load_inputs(sales, promotions))
    result = B.run(data, policy.backtest.validation_origins, policy.backtest.test_origins, policy.backtest.interval)
    gate = checks(result, policy)
    out.mkdir(parents=True, exist_ok=True)
    (out / "backtest.json").write_text(json.dumps(result.to_dict() | {"checks": [c.__dict__ for c in gate]}, indent=1))
    published = None
    if all(c.passed for c in gate):
        published = publish(data, result, policy.backtest.interval)
        published.save(out / "forecast")
    run_id = None
    if tracking_uri and published is not None:
        from demandcast.tracking import log_run

        run_id = log_run(tracking_uri, out, result, gate, policy)
    return RunResult(result, gate, published, run_id)
