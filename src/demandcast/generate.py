"""Deterministic weekly retail sales with the structure real demand has.

Three regions, twelve stores, four categories and sixty products: 720 store-product series over
three years. Demand has category seasonality, holiday peaks, trends, planned promotions with
price elasticity and a post-promotion dip, slow sellers with intermittent demand, product
launches part way through, and count noise. The data is synthetic so that the drivers are
known; the forecasting code never sees how it was made.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

START = dt.date(2023, 1, 2)  # a Monday
WEEKS = 156
REGIONS = {"North": 4, "South": 4, "West": 4}
CATEGORIES = {
    # name: (price range, elasticity, seasonal amplitude, peak week, holiday uplift)
    "Beverages": ((1.0, 4.0), -2.2, 0.35, 28, 0.25),
    "Snacks": ((1.5, 5.0), -1.8, 0.10, 50, 0.60),
    "Household": ((3.0, 12.0), -1.2, 0.05, 10, 0.05),
    "Personal Care": ((2.0, 9.0), -1.5, 0.12, 22, 0.15),
}
SKUS_PER_CATEGORY = 15


def holiday_weeks(weeks: pd.DatetimeIndex) -> np.ndarray:
    """Weeks containing Thanksgiving, the week before Christmas, Christmas week and New Year."""
    flags = np.zeros(len(weeks))
    for i, start in enumerate(weeks):
        days = [start + pd.Timedelta(days=d) for d in range(7)]
        for day in days:
            thanksgiving = day.month == 11 and day.weekday() == 3 and 22 <= day.day <= 28
            christmas = day.month == 12 and 18 <= day.day <= 31
            new_year = day.month == 1 and day.day <= 3
            if thanksgiving or christmas or new_year:
                flags[i] = 1.0
    return flags


def products(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for c, ((lo, hi), elasticity, *_rest) in CATEGORIES.items():
        for k in range(SKUS_PER_CATEGORY):
            slow = rng.random() < 0.2
            rows.append(
                {
                    "sku": f"{c[:3].upper()}-{k + 1:02d}",
                    "category": c,
                    "base_price": round(float(rng.uniform(lo, hi)), 2),
                    "elasticity": elasticity * float(rng.uniform(0.8, 1.2)),
                    "base_level": float(rng.uniform(0.6, 2.0) if slow else rng.lognormal(3.0, 0.5)),
                    "trend": float(rng.normal(0, 0.0025)),
                    "launch_week": int(rng.integers(40, 110)) if rng.random() < 0.1 else 0,
                }
            )
    return pd.DataFrame(rows)


def stores(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    n = 0
    for region, count in REGIONS.items():
        for _ in range(count):
            n += 1
            rows.append({"store": f"S{n:02d}", "region": region, "size": float(rng.lognormal(0, 0.35))})
    return pd.DataFrame(rows)


def promotions(rng: np.random.Generator, skus: pd.DataFrame, weeks: int, horizon_extra: int = 0) -> pd.DataFrame:
    """Planned promotions: a SKU on discount chain-wide for a week, about one week in twelve."""
    rows = []
    for sku in skus["sku"]:
        for w in range(weeks + horizon_extra):
            if rng.random() < 0.08:
                rows.append({"sku": sku, "week_index": w, "discount": round(float(rng.choice([0.1, 0.15, 0.2, 0.25, 0.3])), 2)})
    return pd.DataFrame(rows, columns=["sku", "week_index", "discount"])


def generate(seed: int = 11, weeks: int = WEEKS, future_weeks: int = 13) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sales history, planned promotions including weeks beyond the history)."""
    rng = np.random.default_rng(seed)
    skus, shops = products(rng), stores(rng)
    promos = promotions(rng, skus, weeks, future_weeks)
    calendar = pd.date_range(START, periods=weeks, freq="W-MON")
    holiday = holiday_weeks(calendar)
    week_of_year = calendar.isocalendar().week.to_numpy().astype(float)
    discount = np.zeros((len(skus), weeks))
    for row in promos.itertuples():
        if row.week_index < weeks:
            discount[skus.index[skus["sku"] == row.sku][0], row.week_index] = row.discount

    frames = []
    t = np.arange(weeks)
    for i, p in skus.iterrows():
        _, _, amp, peak, uplift = CATEGORIES[p["category"]]
        season = 1 + amp * np.cos(2 * np.pi * (week_of_year - peak) / 52)
        trend = np.exp(p["trend"] * t)
        promo = (1 - discount[i]) ** p["elasticity"]
        dip = np.r_[1.0, np.where(discount[i][:-1] > 0, 0.9, 1.0)]
        live = (t >= p["launch_week"]).astype(float)
        ramp = np.clip((t - p["launch_week"] + 1) / 6, 0, 1) if p["launch_week"] else np.ones(weeks)
        shape = p["base_level"] * season * trend * promo * dip * (1 + uplift * holiday) * live * ramp
        price = np.round(p["base_price"] * (1 - discount[i]), 2)
        for s in shops.itertuples():
            mean = shape * s.size * float(rng.lognormal(0, 0.15))
            # Gamma-Poisson (negative binomial) counts: overdispersed like real unit sales.
            units = rng.poisson(rng.gamma(4.0, mean / 4.0))
            frames.append(
                pd.DataFrame(
                    {
                        "week": calendar,
                        "store": s.store,
                        "region": s.region,
                        "category": p["category"],
                        "sku": p["sku"],
                        "units": units.astype(int),
                        "price": price,
                        "discount": discount[i],
                        "holiday": holiday,
                    }
                )
            )
    sales = pd.concat(frames, ignore_index=True)
    promos["week"] = [pd.Timestamp(START) + pd.Timedelta(weeks=int(w)) for w in promos["week_index"]]
    promos = promos.merge(skus[["sku", "base_price"]], on="sku")
    return sales, promos[["sku", "week", "discount", "base_price"]]
