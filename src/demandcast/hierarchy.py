"""The forecasting hierarchy and its summing matrix.

Levels: total, region, store, store × category, store × product (the 720 bottom series). Every
series is a sum of bottom series, so S @ bottom gives all 784 series, and a set of forecasts is
coherent when it can be written that way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LEVELS = ["total", "region", "store", "store_category", "store_sku"]


@dataclass
class Hierarchy:
    bottom: pd.DataFrame  # one row per bottom series: store, region, category, sku
    series: pd.DataFrame  # one row per series: id, level, store, region, category, sku
    S: np.ndarray  # summing matrix, series × bottom

    @property
    def levels(self) -> np.ndarray:
        return np.asarray(self.series["level"].to_numpy())

    def index(self, level: str) -> np.ndarray:
        return np.asarray(np.where(self.levels == level)[0])

    def aggregate(self, bottom_values: np.ndarray) -> np.ndarray:
        return np.asarray(self.S @ bottom_values, dtype=float)


def build(sales: pd.DataFrame) -> Hierarchy:
    bottom = sales[["store", "region", "category", "sku"]].drop_duplicates().sort_values(["store", "sku"]).reset_index(drop=True)
    rows, masks = [], []

    def add(level: str, mask: np.ndarray, **keys: str) -> None:
        label = "|".join(f"{k}={v}" for k, v in keys.items() if k != "category" or level == "store_category") or "total"
        rows.append({"id": label, "level": level, **keys})
        masks.append(mask)

    add("total", np.ones(len(bottom), dtype=bool))
    for r in sorted(bottom["region"].unique()):
        add("region", (bottom["region"] == r).to_numpy(), region=r)
    for s in sorted(bottom["store"].unique()):
        add("store", (bottom["store"] == s).to_numpy(), store=s)
    for (s, c), _ in bottom.groupby(["store", "category"]):
        add("store_category", ((bottom["store"] == s) & (bottom["category"] == c)).to_numpy(), store=s, category=c)
    for b in bottom.itertuples():
        add("store_sku", ((bottom["store"] == b.store) & (bottom["sku"] == b.sku)).to_numpy(), store=b.store, category=b.category, sku=b.sku)
    series = pd.DataFrame(rows)
    for col in ["region", "store", "category", "sku"]:
        if col not in series:
            series[col] = ""
    series[["region", "store", "category", "sku"]] = series[["region", "store", "category", "sku"]].fillna("")
    # Regions for every store-based series, so answers can filter by region at any level.
    region_of = dict(zip(bottom["store"], bottom["region"], strict=False))
    has_store = series["store"] != ""
    series.loc[has_store, "region"] = series.loc[has_store, "store"].map(region_of)
    return Hierarchy(bottom, series, np.vstack(masks).astype(float))


def bottom_matrix(sales: pd.DataFrame, h: Hierarchy, column: str = "units") -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Bottom series × weeks."""
    wide = sales.pivot_table(index=["store", "sku"], columns="week", values=column, aggfunc="sum", fill_value=0)
    wide = wide.reindex(pd.MultiIndex.from_frame(h.bottom[["store", "sku"]]), fill_value=0)
    return wide.to_numpy(dtype=float), pd.DatetimeIndex(wide.columns)
