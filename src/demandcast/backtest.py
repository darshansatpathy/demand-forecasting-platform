"""Rolling-origin backtest: retrain at each origin, forecast eight weeks, score against what happened.

Methods compared at every level of the hierarchy:

- seasonal naive (same week last year) and an 8-week moving average, the baselines;
- LightGBM base forecasts, made at each level independently (not coherent);
- LightGBM bottom-up: bottom forecasts summed up;
- LightGBM MinT: every level combined by MinT with a shrunk covariance, the published method.

80% intervals for the published forecasts come from the validation origins' scaled errors by
level and horizon (split conformal), and their coverage is measured on the test origins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from demandcast import forecast as F
from demandcast import reconcile as R
from demandcast.hierarchy import LEVELS

METHODS = ["seasonal_naive", "moving_average", "lightgbm_base", "lightgbm_bottom_up", "lightgbm_mint"]


@dataclass
class Window:
    origin: int
    forecasts: dict[str, np.ndarray]  # method -> series × horizon
    actual: np.ndarray


@dataclass
class Intervals:
    lower: dict[tuple[str, int], float]  # (level, horizon) -> scaled error quantile
    upper: dict[tuple[str, int], float]

    def apply(self, data: F.Data, origin: int, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scale = data.Y[:, max(0, origin - 51) : origin + 1].mean(axis=1) + 1.0
        lo, hi = np.empty_like(point), np.empty_like(point)
        for level in LEVELS:
            ix = data.hierarchy.index(level)
            for h in range(F.HORIZON):
                lo[ix, h] = point[ix, h] + self.lower[(level, h + 1)] * scale[ix]
                hi[ix, h] = point[ix, h] + self.upper[(level, h + 1)] * scale[ix]
        return np.clip(lo, 0, None), hi


@dataclass
class Result:
    windows: list[Window]
    validation: list[Window]
    intervals: Intervals
    metrics: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)  # level -> method -> metric
    coverage: dict[str, float] = field(default_factory=dict)
    width: dict[str, float] = field(default_factory=dict)
    incoherence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "coverage": self.coverage,
            "interval_width": self.width,
            "incoherence": self.incoherence,
            "origins": [w.origin for w in self.windows],
        }


def _window(data: F.Data, origin: int, residuals: np.ndarray | None) -> Window:
    booster = F.train(data, origin)
    base = F.predict(booster, data, origin)
    S = data.hierarchy.S
    forecasts = {
        "seasonal_naive": F.seasonal_naive(data, origin),
        "moving_average": F.moving_average(data, origin),
        "lightgbm_base": base,
        "lightgbm_bottom_up": R.bottom_up(S, base),
    }
    if residuals is not None:
        forecasts["lightgbm_mint"] = np.clip(R.mint(S, base, residuals), 0, None)
        # Clipping negatives can break coherence; restore it from the clipped bottom level.
        forecasts["lightgbm_mint"] = R.bottom_up(S, forecasts["lightgbm_mint"])
    return Window(origin, forecasts, F.actuals(data, origin))


def run(data: F.Data, validation_origins: list[int], test_origins: list[int], interval: float = 0.8) -> Result:
    validation = [_window(data, o, None) for o in validation_origins]
    # Base-forecast errors from the validation windows: observations (origin × horizon) × series.
    residuals = np.hstack([w.actual - w.forecasts["lightgbm_base"] for w in validation]).T
    S = data.hierarchy.S
    for w in validation:
        w.forecasts["lightgbm_mint"] = R.bottom_up(S, np.clip(R.mint(S, w.forecasts["lightgbm_base"], residuals), 0, None))
    intervals = calibrate_intervals(data, validation, interval)
    windows = [_window(data, o, residuals) for o in test_origins]
    result = Result(windows, validation, intervals)
    score(data, result)
    return result


def calibrate_intervals(data: F.Data, windows: list[Window], interval: float) -> Intervals:
    """Split-conformal bounds from scaled errors, pooled over horizons within each level.

    Pooling matters at the top of the hierarchy: there is one total series, so each validation
    window gives one error per horizon. The quantile uses the finite-sample correction.
    """
    alpha = (1 - interval) / 2
    lower, upper = {}, {}
    for level in LEVELS:
        ix = data.hierarchy.index(level)
        errs = []
        for w in windows:
            scale = data.Y[ix, max(0, w.origin - 51) : w.origin + 1].mean(axis=1) + 1.0
            errs.append(((w.actual[ix] - w.forecasts["lightgbm_mint"][ix]) / scale[:, None]).ravel())
        e = np.sort(np.concatenate(errs))
        n = len(e)
        lo_q = max(np.floor(alpha * (n + 1)) / n, 0.0)
        hi_q = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
        lo, hi = float(np.quantile(e, lo_q)), float(np.quantile(e, hi_q))
        for h in range(F.HORIZON):
            lower[(level, h + 1)] = lo
            upper[(level, h + 1)] = hi
    return Intervals(lower, upper)


def score(data: F.Data, result: Result) -> None:
    h = data.hierarchy
    for level in LEVELS:
        ix = h.index(level)
        result.metrics[level] = {}
        actual = np.hstack([w.actual[ix] for w in result.windows])
        for method in METHODS:
            fc = np.hstack([w.forecasts[method][ix] for w in result.windows])
            result.metrics[level][method] = {
                "wape": float(np.abs(fc - actual).sum() / actual.sum()),
                "bias": float((fc - actual).sum() / actual.sum()),
            }
        inside, widths = [], []
        for w in result.windows:
            lo, hi = result.intervals.apply(data, w.origin, w.forecasts["lightgbm_mint"])
            inside.append((w.actual[ix] >= lo[ix]) & (w.actual[ix] <= hi[ix]))
            widths.append((hi[ix] - lo[ix]).sum() / max(w.actual[ix].sum(), 1.0))
        result.coverage[level] = float(np.mean(np.concatenate([i.ravel() for i in inside])))
        result.width[level] = float(np.mean(widths))
    result.incoherence = max(R.incoherence(h.S, w.forecasts["lightgbm_mint"]) for w in result.windows)
