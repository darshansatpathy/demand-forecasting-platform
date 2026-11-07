"""The backtest report, generated from a run."""

from __future__ import annotations

from demandcast.backtest import METHODS, Result
from demandcast.hierarchy import LEVELS
from demandcast.policy import Check, Policy
from demandcast.publish import Published

NAMES = {
    "seasonal_naive": "Seasonal naive",
    "moving_average": "8-week moving average",
    "lightgbm_base": "LightGBM, each level alone",
    "lightgbm_bottom_up": "LightGBM, bottom-up",
    "lightgbm_mint": "LightGBM, MinT (published)",
}
LEVEL_NAMES = {
    "total": "Total",
    "region": "Region (3)",
    "store": "Store (12)",
    "store_category": "Store × category (48)",
    "store_sku": "Store × product (720)",
}


def render(result: Result, checks: list[Check], policy: Policy, published: Published | None = None) -> str:
    passed = all(c.passed for c in checks)
    lines = ["# Backtest report", ""]
    lines.append(f"**{sum(c.passed for c in checks)} / {len(checks)} policy checks pass.**" + ("" if passed else " The forecast is not published."))
    lines += [
        "",
        f"Rolling origin: the model is retrained at each of {len(result.windows)} test origins (weeks "
        f"{', '.join(str(w.origin) for w in result.windows)}) and forecasts the following eight weeks, so the last "
        f"{8 * len(result.windows)} weeks are scored out of sample. {len(result.validation)} earlier origins provide the errors "
        "used for MinT and for the intervals.",
        "",
        "## Weighted absolute percentage error",
        "",
        "| Level | " + " | ".join(NAMES[m] for m in METHODS) + " |",
        "|---|" + "---|" * len(METHODS),
    ]
    for level in LEVELS:
        cells = []
        best = min(result.metrics[level][m]["wape"] for m in METHODS)
        for m in METHODS:
            v = result.metrics[level][m]["wape"]
            cells.append(f"**{v:.1%}**" if v == best else f"{v:.1%}")
        lines.append(f"| {LEVEL_NAMES[level]} | " + " | ".join(cells) + " |")
    lines += ["", "## Bias", "", "| Level | " + " | ".join(NAMES[m] for m in METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for level in LEVELS:
        lines.append(f"| {LEVEL_NAMES[level]} | " + " | ".join(f"{result.metrics[level][m]['bias']:+.1%}" for m in METHODS) + " |")
    lines += [
        "",
        f"## {policy.backtest.interval:.0%} intervals",
        "",
        "| Level | Test observations | Coverage | Average width (share of actual) |",
        "|---|---|---|---|",
    ]
    for level in LEVELS:
        n = {"total": 1, "region": 3, "store": 12, "store_category": 48, "store_sku": 720}[level] * 8 * len(result.windows)
        lines.append(f"| {LEVEL_NAMES[level]} | {n:,} | {result.coverage[level]:.1%} | {result.width[level]:.1%} |")
    lines += [
        "",
        "Coverage is gated only where there are enough observations to measure it; at the total level each test point moves "
        "coverage by three percentage points.",
        "",
        "## Policy checks",
        "",
        "| Check | Value | Threshold | Result |",
        "|---|---|---|---|",
    ]
    lines += [f"| {c.name} | {c.value:.4f} | {c.threshold} | {'pass' if c.passed else '**fail**'} |" for c in checks]
    if published is not None:
        f = published.forecast
        total = f[f["level"] == "total"]
        bottom = f[(f["level"] == "store_sku") & (f["discount"] > 0)]
        lines += [
            "",
            "## Published forecast",
            "",
            f"From week {published.origin_week.date().isoformat()}, eight weeks ahead:",
            "",
            "| Week | Low (10%) | Forecast | High (90%) | Without promotions |",
            "|---|---|---|---|---|",
        ]
        lines += [f"| {r.week.date().isoformat()} | {r.p10:,.0f} | {r.p50:,.0f} | {r.p90:,.0f} | {r.no_promo:,.0f} |" for r in total.itertuples()]
        if len(bottom):
            ratio = (bottom["p50"] / bottom["no_promo"].clip(lower=1e-9)).median()
            lines += [
                "",
                f"{len(bottom):,} store-product-weeks carry a planned promotion; the median forecast uplift on them is "
                f"{ratio:.2f} times the no-promotion forecast.",
            ]
    return "\n".join(lines) + "\n"
