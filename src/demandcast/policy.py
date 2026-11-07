"""The forecast policy file and the checks it defines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from demandcast.backtest import Result
from demandcast.resources import resource


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Backtest(_Strict):
    validation_origins: list[int]
    test_origins: list[int]
    interval: float


class Coverage(_Strict):
    min: float
    max: float
    levels: list[str]


class Thresholds(_Strict):
    min_improvement_vs_seasonal_naive: dict[str, float]
    no_worse_than_moving_average: bool
    max_abs_bias: dict[str, float]
    interval_coverage: Coverage
    max_incoherence: float


class Policy(_Strict):
    backtest: Backtest
    thresholds: Thresholds


def load(path: Path | None = None) -> Policy:
    return Policy.model_validate(yaml.safe_load((path or resource("policy") / "forecast_policy.yaml").read_text()))


@dataclass
class Check:
    name: str
    value: float
    threshold: str
    passed: bool


PUBLISHED = "lightgbm_mint"


def checks(result: Result, policy: Policy) -> list[Check]:
    th = policy.thresholds
    m = result.metrics
    out = []
    for level, need in th.min_improvement_vs_seasonal_naive.items():
        gain = 1 - m[level][PUBLISHED]["wape"] / m[level]["seasonal_naive"]["wape"]
        out.append(Check(f"accuracy: WAPE improvement over seasonal naive ({level})", gain, f">= {need:.0%}", gain >= need))
    if th.no_worse_than_moving_average:
        worst = max(m[lv][PUBLISHED]["wape"] - m[lv]["moving_average"]["wape"] for lv in m)
        out.append(Check("accuracy: no level worse than an 8-week moving average", worst, "<= 0", worst <= 0))
    for level, limit in th.max_abs_bias.items():
        bias = m[level][PUBLISHED]["bias"]
        out.append(Check(f"bias ({level})", bias, f"within ±{limit:.0%}", abs(bias) <= limit))
    cov = th.interval_coverage
    for level in cov.levels:
        c = result.coverage[level]
        out.append(
            Check(f"{policy.backtest.interval:.0%} interval coverage ({level})", c, f"{cov.min:.0%} to {cov.max:.0%}", cov.min <= c <= cov.max)
        )
    out.append(
        Check("coherence: forecasts add up the hierarchy", result.incoherence, f"<= {th.max_incoherence}", result.incoherence <= th.max_incoherence)
    )
    negative = min(float(w.forecasts[PUBLISHED].min()) for w in result.windows)
    out.append(Check("no negative forecasts", negative, ">= 0", negative >= 0))
    return out
