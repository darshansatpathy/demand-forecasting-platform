"""The backtest's checks pass for the published method and fail for planted problems."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from demandcast import backtest as B
from demandcast import forecast as F
from demandcast import policy as P


@pytest.fixture(scope="module")
def short(data: F.Data) -> B.Result:
    return B.run(data, [91, 99, 107, 115], [139, 147])


def failing(result: B.Result) -> set[str]:
    return {c.name for c in P.checks(result, P.load()) if not c.passed}


def test_full_run_passes_every_check(run_dir: Path) -> None:
    summary = json.loads((run_dir / "backtest.json").read_text())
    assert all(c["passed"] for c in summary["checks"]) and len(summary["checks"]) == 10
    m = summary["metrics"]
    for level in m:
        assert m[level]["lightgbm_mint"]["wape"] < m[level]["seasonal_naive"]["wape"]
    assert m["total"]["lightgbm_mint"]["wape"] < m["total"]["lightgbm_bottom_up"]["wape"], "MinT helps at the top"
    assert abs(m["total"]["lightgbm_mint"]["bias"]) < abs(m["total"]["lightgbm_bottom_up"]["bias"])


def test_baseline_in_place_of_the_model_fails_accuracy(short: B.Result, data: F.Data) -> None:
    bad = copy.deepcopy(short)
    for w in bad.windows:
        w.forecasts["lightgbm_mint"] = w.forecasts["seasonal_naive"]
    B.score(data, bad)
    assert {c for c in failing(bad) if c.startswith("accuracy")} == {
        "accuracy: WAPE improvement over seasonal naive (store)",
        "accuracy: WAPE improvement over seasonal naive (store_sku)",
        "accuracy: no level worse than an 8-week moving average",
    }


def test_narrow_intervals_fail_coverage(short: B.Result, data: F.Data) -> None:
    bad = copy.deepcopy(short)
    bad.intervals = B.Intervals({k: v * 0.3 for k, v in bad.intervals.lower.items()}, {k: v * 0.3 for k, v in bad.intervals.upper.items()})
    B.score(data, bad)
    assert {"80% interval coverage (store)", "80% interval coverage (store_sku)"} <= failing(bad)


def test_biased_and_incoherent_forecasts_fail(short: B.Result, data: F.Data) -> None:
    bad = copy.deepcopy(short)
    for w in bad.windows:
        w.forecasts["lightgbm_mint"] = w.forecasts["lightgbm_mint"] * 1.10
        w.forecasts["lightgbm_mint"][0] += 50.0
        w.forecasts["lightgbm_mint"][-1] = -1.0
    B.score(data, bad)
    names = failing(bad)
    assert {"bias (total)", "bias (region)", "coherence: forecasts add up the hierarchy", "no negative forecasts"} <= names


def test_intervals_are_wider_lower_down(short: B.Result) -> None:
    width = {lv: short.intervals.upper[(lv, 1)] - short.intervals.lower[(lv, 1)] for lv in ["total", "store", "store_sku"]}
    assert width["total"] < width["store"] < width["store_sku"]
    assert np.isfinite(list(short.intervals.lower.values())).all()
