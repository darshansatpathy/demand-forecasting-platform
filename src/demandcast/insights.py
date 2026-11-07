"""Plain-language questions over the published forecast.

The parser understands a fixed set of question shapes and refuses anything else. Every number
in an answer is computed from the forecast tables, and an optional language model may only
rephrase an answer: a rephrasing that introduces a number not in the answer's facts is discarded.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from demandcast.generate import CATEGORIES, REGIONS
from demandcast.publish import DRIVER_GROUPS, Published

HELP = (
    "I can answer: the forecast for an entity ('forecast for Snacks in North next 4 weeks'), rankings ('top 5 products by "
    "growth in South'), comparisons with last year ('how does S03 compare with last year'), planned promotions and their "
    "uplift ('which promotions are planned for Beverages'), and drivers ('why is BEV-03 at S01 forecast to change')."
)


@dataclass
class Answer:
    intent: str
    text: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    facts: dict[str, float] = field(default_factory=dict)


@dataclass
class Scope:
    region: str | None = None
    store: str | None = None
    category: str | None = None
    sku: str | None = None
    weeks: int = 8
    capped: bool = False

    def describe(self) -> str:
        parts = [p for p in (self.sku, self.category, self.store and f"store {self.store}", self.region) if p]
        return " / ".join(parts) if parts else "all stores"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        m = pd.Series(True, index=df.index)
        for col in ("region", "store", "category", "sku"):
            value = getattr(self, col)
            if value:
                m &= df[col] == value
        return m


def _scope(q: str, published: Published) -> Scope:
    s = Scope()
    for region in REGIONS:
        if re.search(rf"\b{region.lower()}\b", q):
            s.region = region
    for category in CATEGORIES:
        if category.lower() in q:
            s.category = category
    if m := re.search(r"\bs(\d{2})\b", q):
        s.store = f"S{m[1]}"
    skus = set(published.series["sku"]) - {""}
    for sku in skus:
        if re.search(rf"\b{sku.lower()}\b", q):
            s.sku = sku
    if m := re.search(r"next (\d+) weeks?", q):
        s.weeks = int(m[1])
    elif "next month" in q:
        s.weeks = 4
    elif "next week" in q:
        s.weeks = 1
    elif "next quarter" in q or "next 3 months" in q:
        s.weeks = 13
    if s.weeks > 8:
        s.weeks, s.capped = 8, True
    return s


def _bottom(published: Published, scope: Scope) -> pd.DataFrame:
    f = published.forecast
    f = f[(f["level"] == "store_sku") & (f["horizon"] <= scope.weeks)]
    return f[scope.mask(f)]


def _history(published: Published, scope: Scope, weeks: pd.Series) -> float:
    """Actual units in the same weeks a year earlier."""
    h = published.history
    h = h[(h["level"] == "store_sku") & scope.mask(h)]
    last_year = set(weeks - pd.Timedelta(weeks=52))
    return float(h[h["week"].isin(last_year)]["units"].sum())


def _cap_note(scope: Scope) -> str:
    return " The forecast runs eight weeks, so the answer covers those." if scope.capped else ""


def answer(question: str, published: Published) -> Answer:
    q = question.lower().strip()
    if not q or len(q) > 300:
        return Answer("unsupported", "Questions must be between 1 and 300 characters. " + HELP)
    scope = _scope(q, published)
    if re.search(r"\b(why|driver|drivers|explain)\b", q):
        return _drivers(published, scope)
    if re.search(r"\bpromo(tion)?s?\b|\buplift\b", q):
        return _promotions(published, scope)
    if m := re.search(r"\b(top|bottom|best|worst)\s*(\d+)?\s*(stores?|regions?|categories|category|products?|skus?)\b", q):
        return _ranking(published, scope, m[1], int(m[2] or 5), m[3], "growth" if "growth" in q or "grow" in q else "forecast")
    if re.search(r"last year|year over year|vs\.? last|compare", q):
        return _versus_last_year(published, scope)
    if re.search(r"\bforecast|\bdemand|\bsell|\bsales|\bunits|how many|how much", q):
        return _forecast(published, scope)
    return Answer("unsupported", "I can't answer that. " + HELP)


def _forecast(published: Published, scope: Scope) -> Answer:
    b = _bottom(published, scope)
    if b.empty:
        return Answer("forecast", f"Nothing in the forecast matches {scope.describe()}.")
    weekly = b.groupby("week", as_index=False)["p50"].sum()
    total = float(weekly["p50"].sum())
    rows = [{"week": w.date().isoformat(), "units": round(float(u))} for w, u in zip(weekly["week"], weekly["p50"], strict=True)]
    node = _node(published, scope)
    if node is not None:
        nf = published.forecast[(published.forecast["id"] == node) & (published.forecast["horizon"] <= scope.weeks)]
        for row, lo, hi in zip(rows, nf["p10"], nf["p90"], strict=True):
            row["p10"], row["p90"] = round(float(lo)), round(float(hi))
    text = f"Forecast for {scope.describe()} over the next {scope.weeks} weeks: {total:,.0f} units." + _cap_note(scope)
    if node is not None:
        text += " Weekly 80% intervals are in the table."
    return Answer("forecast", text, rows, {"total": round(total)})


def _node(published: Published, scope: Scope) -> str | None:
    """The hierarchy series that matches the scope exactly, if any (intervals exist only there)."""
    keys = {k: v for k, v in (("region", scope.region), ("store", scope.store), ("category", scope.category), ("sku", scope.sku)) if v}
    s = published.series
    if not keys:
        return "total"
    if keys.keys() == {"region"}:
        return f"region={scope.region}"
    if "store" in keys and set(keys) <= {"store", "region"}:
        return f"store={scope.store}"
    if {"store", "category"} <= keys.keys() and "sku" not in keys:
        return f"store={scope.store}|category={scope.category}"
    if {"store", "sku"} <= keys.keys():
        ids = s[(s["store"] == scope.store) & (s["sku"] == scope.sku) & (s["level"] == "store_sku")]["id"]
        return str(ids.iloc[0]) if len(ids) else None
    return None


def _ranking(published: Published, scope: Scope, word: str, n: int, unit: str, measure: str) -> Answer:
    column = {"store": "store", "region": "region", "categor": "category", "product": "sku", "sku": "sku"}
    by = next(v for k, v in column.items() if unit.startswith(k))
    b = _bottom(published, scope)
    fc = b.groupby(by)["p50"].sum()
    if measure == "growth":
        h = published.history
        h = h[(h["level"] == "store_sku") & scope.mask(h)]
        last_year = set(b["week"].unique() - pd.Timedelta(weeks=52))
        ly = h[h["week"].isin(last_year)].groupby(by)["units"].sum()
        value = (fc / ly.reindex(fc.index)).dropna() - 1
        value = value[ly.reindex(value.index) >= 20]  # growth on a handful of units is noise
    else:
        value = fc
    ascending = word in ("bottom", "worst")
    top = value.sort_values(ascending=ascending).head(n)
    label = "growth vs the same weeks last year" if measure == "growth" else "forecast units"
    rows = [{by: k, "value": round(float(v), 4 if measure == "growth" else 0)} for k, v in top.items()]

    def fmt(v: float) -> str:
        return f"{v:+.1%}" if measure == "growth" else f"{v:,.0f}"

    listed = ", ".join(f"{k} ({fmt(float(v))})" for k, v in top.items())
    plural = {"store": "stores", "region": "regions", "category": "categories", "sku": "products"}[by]
    text = f"{'Top' if not ascending else 'Bottom'} {len(top)} {plural} by {label} over the next {scope.weeks} weeks"
    text += f" in {scope.describe()}: {listed}." if scope.describe() != "all stores" else f": {listed}."
    return Answer("ranking", text + _cap_note(scope), rows, {str(k): round(float(v), 4) for k, v in top.items()})


def _versus_last_year(published: Published, scope: Scope) -> Answer:
    b = _bottom(published, scope)
    fc = float(b["p50"].sum())
    ly = _history(published, scope, pd.Series(b["week"].unique()))
    if ly <= 0:
        return Answer("versus_last_year", f"There are no sales for {scope.describe()} in the same weeks last year to compare with.")
    change = fc / ly - 1
    text = (
        f"{scope.describe().capitalize()}: {fc:,.0f} units forecast for the next {scope.weeks} weeks against {ly:,.0f} in the "
        f"same weeks last year ({change:+.1%})."
    ) + _cap_note(scope)
    return Answer("versus_last_year", text, [], {"forecast": round(fc), "last_year": round(ly), "change": round(change, 4)})


def _promotions(published: Published, scope: Scope) -> Answer:
    b = _bottom(published, scope)
    promo = b[b["discount"] > 0]
    if promo.empty:
        return Answer("promotions", f"No promotions are planned for {scope.describe()} in the next {scope.weeks} weeks.")
    g = promo.groupby(["sku", "week", "discount"], as_index=False)[["p50", "no_promo"]].sum()
    g["uplift"] = g["p50"] - g["no_promo"]
    rows = [
        {"sku": r.sku, "week": r.week.date().isoformat(), "discount": r.discount, "units": round(r.p50), "uplift_units": round(r.uplift)}
        for r in g.sort_values(["week", "sku"]).itertuples()
    ]
    total = float(g["uplift"].sum())
    text = (
        f"{g['sku'].nunique()} products have promotions planned for {scope.describe()} in the next {scope.weeks} weeks, "
        f"adding about {total:,.0f} units over what they would sell without them (forecast with and without the discount)."
    )
    return Answer("promotions", text + _cap_note(scope), rows, {"products": float(g["sku"].nunique()), "uplift": round(total)})


def _drivers(published: Published, scope: Scope) -> Answer:
    if not scope.sku:
        return Answer("drivers", "Drivers are given per product: name a product, for example 'why is BEV-03 at S01 forecast to change'.")
    b = _bottom(published, scope)
    if b.empty:
        return Answer("drivers", f"Nothing in the forecast matches {scope.describe()}.")
    cols = {g: "driver_" + g.replace(" ", "_") for g in DRIVER_GROUPS}
    contrib = {g: float(b[c].mean()) for g, c in cols.items()}
    ranked = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))
    recent = _recent(published, scope)
    fc_week = float(b.groupby("week")["p50"].sum().mean())
    direction = "up" if fc_week > recent else "down"
    parts = [f"{g} ({'raises' if v > 0 else 'lowers'} it)" for g, v in ranked if abs(v) > 0.02]
    text = (
        f"{scope.describe()}: about {fc_week:,.1f} units a week forecast against {recent:,.1f} a week over the last eight weeks "
        f"({direction}). Largest drivers: " + (", ".join(parts) if parts else "none stands out") + "."
    )
    return Answer(
        "drivers",
        text,
        [{"driver": g, "contribution": round(v, 4)} for g, v in ranked],
        {"forecast_per_week": round(fc_week, 1), "recent_per_week": round(recent, 1)},
    )


def _recent(published: Published, scope: Scope) -> float:
    h = published.history
    h = h[(h["level"] == "store_sku") & scope.mask(h)]
    last = sorted(h["week"].unique())[-8:]
    return float(h[h["week"].isin(last)].groupby("week")["units"].sum().mean())


def narrate(ans: Answer, llm: Callable[[str], str] | None) -> str:
    """Optionally let a language model rephrase an answer, keeping it only if it adds no numbers."""
    if llm is None or ans.intent == "unsupported":
        return ans.text
    rewritten = llm(f"Rephrase for a merchandising manager, keep every number exactly:\n{ans.text}")
    allowed = set(re.findall(r"\d[\d,.]*", ans.text)) | {str(v) for v in ans.facts.values()}
    used = set(re.findall(r"\d[\d,.]*", rewritten))
    return rewritten if used <= allowed else ans.text
