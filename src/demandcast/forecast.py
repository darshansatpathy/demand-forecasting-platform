"""One global LightGBM model across every series and level, forecasting 1 to 8 weeks ahead.

Each training row is (series, forecast origin, horizon). Features are taken from the history
up to the origin, scaled by the series' recent level, plus what is known about the target week
in advance: the calendar, holidays and the planned promotions. Because every lag is measured
from the origin, the same model forecasts every horizon without feeding forecasts back in.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from demandcast.hierarchy import LEVELS, Hierarchy, bottom_matrix
from demandcast.hierarchy import build as build_hierarchy

HORIZON = 8
CATEGORY_CODES = {"Beverages": 0, "Snacks": 1, "Household": 2, "Personal Care": 3}
FEATURES = [
    "level",
    "category",
    "horizon",
    "week_of_year",
    "holiday",
    "holiday_prev",
    "last",
    "mean_4",
    "mean_8",
    "mean_13",
    "mean_26",
    "std_8",
    "zero_share_13",
    "last_year",
    "last_year_window",
    "last_year_at_origin",
    "discount",
    "discount_prev",
    "discount_at_origin",
    "age",
]


@dataclass
class Data:
    hierarchy: Hierarchy
    Y: np.ndarray  # all series × observed weeks
    weeks: pd.DatetimeIndex  # observed weeks, then the planned future weeks
    discount_bottom: np.ndarray  # bottom × (observed + future) weeks
    holiday: np.ndarray  # (observed + future) weeks
    observed: int

    @property
    def n_series(self) -> int:
        return int(self.Y.shape[0])


def prepare(sales: pd.DataFrame, promos: pd.DataFrame, future_weeks: int = HORIZON) -> Data:
    from demandcast.generate import holiday_weeks

    h = build_hierarchy(sales)
    bottom, observed_weeks = bottom_matrix(sales, h)
    weeks = observed_weeks.append(pd.date_range(observed_weeks[-1] + pd.Timedelta(weeks=1), periods=future_weeks, freq="W-MON"))
    disc = np.zeros((len(h.bottom), len(weeks)))
    index = {w: i for i, w in enumerate(weeks)}
    sku_rows: dict[str, np.ndarray] = {s: np.where(h.bottom["sku"].to_numpy() == s)[0] for s in h.bottom["sku"].unique()}
    for row in promos.itertuples():
        col = index.get(pd.Timestamp(row.week))
        if col is not None and row.sku in sku_rows:
            disc[sku_rows[row.sku], col] = row.discount
    return Data(h, h.aggregate(bottom), weeks, disc, holiday_weeks(weeks), len(observed_weeks))


def _weighted_discount(data: Data, origin: int) -> np.ndarray:
    """Discount for every series: a bottom series' own, or for an aggregate the share of its
    recent volume that is on promotion, weighted by depth."""
    n_bottom = data.discount_bottom.shape[0]
    recent = data.Y[-n_bottom:, max(0, origin - 12) : origin + 1].mean(axis=1) + 1e-6
    S = data.hierarchy.S
    return np.asarray((S @ (recent[:, None] * data.discount_bottom)) / (S @ recent)[:, None], dtype=float)


def design(data: Data, origin: int, with_target: bool = True) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Rows for every series and horizon 1..HORIZON from one origin.

    Returns (features, scaled target, scale per row). The target is nan where it is not observed.
    """
    Y = data.Y
    n = data.n_series
    scale = Y[:, max(0, origin - 51) : origin + 1].mean(axis=1) + 1.0
    hist = Y[:, : origin + 1]
    level = np.array([LEVELS.index(lv) for lv in data.hierarchy.levels], dtype=float)
    category = np.array([CATEGORY_CODES.get(c, -1) for c in data.hierarchy.series["category"]], dtype=float)
    disc = _weighted_discount(data, origin)
    nonzero = hist > 0
    first = np.where(nonzero.any(axis=1), nonzero.argmax(axis=1), origin)
    age = np.clip(origin - first, 0, 104).astype(float)

    def mean_last(k: int) -> np.ndarray:
        return np.asarray(hist[:, max(0, origin - k + 1) :].mean(axis=1), dtype=float)

    blocks, targets, scales = [], [], []
    week_of_year = data.weeks.isocalendar().week.to_numpy().astype(float)
    for h in range(1, HORIZON + 1):
        t = origin + h
        ly = t - 52
        f = {
            "level": level,
            "category": category,
            "horizon": np.full(n, float(h)),
            "week_of_year": np.full(n, week_of_year[t]),
            "holiday": np.full(n, data.holiday[t]),
            "holiday_prev": np.full(n, data.holiday[t - 1]),
            "last": hist[:, -1] / scale,
            "mean_4": mean_last(4) / scale,
            "mean_8": mean_last(8) / scale,
            "mean_13": mean_last(13) / scale,
            "mean_26": mean_last(26) / scale,
            "std_8": hist[:, max(0, origin - 7) :].std(axis=1) / scale,
            "zero_share_13": (hist[:, max(0, origin - 12) :] == 0).mean(axis=1),
            "last_year": Y[:, ly] / scale if ly >= 0 else np.full(n, np.nan),
            "last_year_window": Y[:, max(ly - 1, 0) : ly + 2].mean(axis=1) / scale if ly >= 1 else np.full(n, np.nan),
            "last_year_at_origin": Y[:, origin - 52] / scale if origin >= 52 else np.full(n, np.nan),
            "discount": disc[:, t],
            "discount_prev": disc[:, t - 1],
            "discount_at_origin": disc[:, origin],
            "age": age,
        }
        blocks.append(pd.DataFrame(f))
        observed = with_target and t < data.observed
        targets.append(Y[:, t] / scale if observed else np.full(n, np.nan))
        scales.append(scale)
    return pd.concat(blocks, ignore_index=True)[FEATURES], np.concatenate(targets), np.concatenate(scales)


def train(data: Data, last_origin: int, first_origin: int = 56, step: int = 2, seed: int = 3) -> lgb.Booster:
    """Fit on every origin whose whole horizon is observed by `last_origin`."""
    frames, targets = [], []
    for origin in range(first_origin, last_origin - HORIZON + 1, step):
        X, y, _ = design(data, origin)
        frames.append(X)
        targets.append(y)
    X = pd.concat(frames, ignore_index=True)
    y = np.concatenate(targets)
    params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.2,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 2,
        "verbose": -1,
    }
    return lgb.train(params, lgb.Dataset(X, y, categorical_feature=["level", "category"]), num_boost_round=400)


def predict(booster: lgb.Booster, data: Data, origin: int) -> np.ndarray:
    """Base forecasts for every series, series × horizon, in units."""
    X, _, scale = design(data, origin, with_target=False)
    pred = np.clip(booster.predict(X), 0, None) * scale
    return np.asarray(pred, dtype=float).reshape(HORIZON, data.n_series).T


def actuals(data: Data, origin: int) -> np.ndarray:
    return data.Y[:, origin + 1 : origin + 1 + HORIZON]


def seasonal_naive(data: Data, origin: int) -> np.ndarray:
    """Same week last year; the last observed value where there is no last year."""
    out = np.empty((data.n_series, HORIZON))
    for h in range(1, HORIZON + 1):
        ly = origin + h - 52
        out[:, h - 1] = data.Y[:, ly] if ly >= 0 else data.Y[:, origin]
    return out


def moving_average(data: Data, origin: int, window: int = 8) -> np.ndarray:
    return np.repeat(data.Y[:, origin - window + 1 : origin + 1].mean(axis=1, keepdims=True), HORIZON, axis=1)
