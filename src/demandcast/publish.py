"""The published forecast: the next eight weeks for every series, with intervals, the promotion
uplift behind each product's forecast, and what drives it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from demandcast import backtest as B
from demandcast import forecast as F
from demandcast import reconcile as R

DRIVER_GROUPS = {
    "promotions": ["discount", "discount_prev", "discount_at_origin"],
    "seasonality and holidays": ["week_of_year", "holiday", "holiday_prev", "last_year", "last_year_window", "last_year_at_origin"],
    "recent sales level": ["last", "mean_4", "mean_8", "mean_13", "mean_26", "std_8", "zero_share_13"],
}


@dataclass
class Published:
    forecast: pd.DataFrame  # one row per series and week: p10, p50, p90, no_promo (bottom), drivers (bottom)
    history: pd.DataFrame  # one row per series and observed week: units
    series: pd.DataFrame
    origin_week: pd.Timestamp
    booster: lgb.Booster | None = None

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        self.forecast.to_parquet(out / "forecast.parquet", index=False)
        self.history.to_parquet(out / "history.parquet", index=False)
        self.series.to_parquet(out / "series.parquet", index=False)
        (out / "origin.txt").write_text(self.origin_week.date().isoformat())
        if self.booster is not None:
            self.booster.save_model(str(out / "model.txt"))

    @classmethod
    def load(cls, out: Path) -> Published:
        return cls(
            pd.read_parquet(out / "forecast.parquet"),
            pd.read_parquet(out / "history.parquet"),
            pd.read_parquet(out / "series.parquet"),
            pd.Timestamp((out / "origin.txt").read_text().strip()),
        )


def _drivers(booster: lgb.Booster, data: F.Data, origin: int) -> dict[str, np.ndarray]:
    """TreeSHAP contributions grouped into drivers, series × horizon, relative to the model's average."""
    X, _, _ = F.design(data, origin, with_target=False)
    contrib = np.asarray(booster.predict(X, pred_contrib=True))[:, :-1]
    out = {}
    for group, cols in DRIVER_GROUPS.items():
        idx = [F.FEATURES.index(c) for c in cols]
        out[group] = contrib[:, idx].sum(axis=1).reshape(F.HORIZON, data.n_series).T
    return out


def publish(data: F.Data, backtest: B.Result, interval: float = 0.8) -> Published:
    origin = data.observed - 1
    booster = F.train(data, origin)
    base = F.predict(booster, data, origin)
    windows = backtest.validation + backtest.windows
    residuals = np.hstack([w.actual - w.forecasts["lightgbm_base"] for w in windows]).T
    S = data.hierarchy.S
    p50 = R.bottom_up(S, np.clip(R.mint(S, base, residuals), 0, None))
    intervals = B.calibrate_intervals(data, windows, interval)
    p10, p90 = intervals.apply(data, origin, p50)

    # The same forecast without any planned promotion: the difference is the promotion uplift.
    saved = data.discount_bottom.copy()
    try:
        data.discount_bottom[:, origin + 1 :] = 0.0
        no_promo = R.bottom_up(S, np.clip(R.mint(S, F.predict(booster, data, origin), residuals), 0, None))
    finally:
        data.discount_bottom[:] = saved
    drivers = _drivers(booster, data, origin)

    weeks = data.weeks[origin + 1 : origin + 1 + F.HORIZON]
    series = data.hierarchy.series
    rows = []
    for h, week in enumerate(weeks):
        frame = series[["id", "level", "region", "store", "category", "sku"]].copy()
        frame["week"] = week
        frame["horizon"] = h + 1
        frame["p10"], frame["p50"], frame["p90"] = p10[:, h], p50[:, h], p90[:, h]
        frame["no_promo"] = no_promo[:, h]
        frame["discount"] = np.r_[np.zeros(len(series) - S.shape[1]), data.discount_bottom[:, origin + 1 + h]]
        for group, values in drivers.items():
            frame["driver_" + group.replace(" ", "_")] = values[:, h]
        rows.append(frame)
    hist = []
    for t in range(data.observed):
        frame = series[["id", "level", "region", "store", "category", "sku"]].copy()
        frame["week"] = data.weeks[t]
        frame["units"] = data.Y[:, t]
        hist.append(frame)
    return Published(pd.concat(rows, ignore_index=True), pd.concat(hist, ignore_index=True), series.copy(), data.weeks[origin], booster)
