# Demand Forecasting Platform

[![CI](https://github.com/darshansatpathy/demand-forecasting-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/darshansatpathy/demand-forecasting-platform/actions/workflows/ci.yml)
[![CodeQL](https://github.com/darshansatpathy/demand-forecasting-platform/actions/workflows/codeql.yml/badge.svg)](https://github.com/darshansatpathy/demand-forecasting-platform/actions/workflows/codeql.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![Airflow](https://img.shields.io/badge/Airflow-3.4-017CEE)
![Policy checks](https://img.shields.io/badge/forecast%20policy-10%2F10-success)
![License](https://img.shields.io/badge/license-MIT-green)

**Weekly demand forecasts for every store, product and level above them that add up, beat
the baselines in backtesting before they are published, come with calibrated intervals, show
what each promotion adds, and answer plain-language questions.**

```
sales history ─┐                                                  ┌─▶ policy gate (10 checks) ─▶ publish ─▶ forecast tables
planned promos ┼─▶ 784 series ─▶ one global LightGBM ─▶ MinT ──────┤                                        ├─▶ "top 5 products by growth in South"
calendar ──────┘   (5 levels)    every series, 1 to 8 weeks ahead │                                        └─▶ API, MLflow
                                 rolling-origin backtest ─────────┘
Airflow: backtest ─▶ gate ─▶ publish
```

## Results

Rolling-origin backtest: retrained at each of four origins and scored on the last 32 weeks,
which the model never saw ([full report](docs/backtest-report.md)). Weighted absolute
percentage error:

| Level | Seasonal naive | 8-week moving average | LightGBM, bottom-up | **LightGBM, MinT (published)** |
|---|---|---|---|---|
| Total | 6.1% | 5.4% | 4.0% | **3.0%** |
| Region (3) | 7.6% | 6.9% | 5.2% | **4.5%** |
| Store (12) | 11.5% | 9.7% | 8.3% | **7.9%** |
| Store × category (48) | 22.0% | 20.0% | 16.1% | **16.1%** |
| Store × product (720) | 59.4% | 47.2% | 42.6% | **43.0%** |

- **About 30% better than seasonal naive** at store level (31.7%) and product level (27.7%),
  and better than the moving average at every level.
- **MinT reconciliation** uses every level's forecast rather than only the bottom one. It cuts
  total-level error by a quarter against bottom-up (4.0% to 3.0%) and bias from −2.9% to −0.9%.
  At product level it costs 0.4 points, which the report shows.
- **80% intervals cover 81% to 85%** of outcomes at store, store × category and product level,
  measured on 384 to 23,040 test observations.
- **Promotion uplift is measured, not assumed:** each forecast is also made with the planned
  discounts removed. On promoted product-weeks the median uplift is 1.44×. The data is
  synthetic, so the true uplift implied by its price elasticities is known, and a test checks
  the estimate against it to within 15%.

## How it works

- **One global model.** A single LightGBM model forecasts all 784 series (total, 3 regions,
  12 stores, 48 store-categories, 720 store-products) for 1 to 8 weeks ahead. Every lag is
  measured from the forecast origin, so each horizon is forecast directly, without feeding
  forecasts back in. Features: recent levels scaled by each series' own level, same weeks last
  year, calendar and holidays, and the promotions planned for the target week.
- **Reconciled.** Forecasts made level by level don't add up. MinT (Wickramasuriya,
  Athanasopoulos and Hyndman, 2019) combines them using a shrunk estimate of their error
  covariance, and the result is coherent to machine precision.
- **Backtested before publishing.** The policy in
  [`policy/forecast_policy.yaml`](policy/forecast_policy.yaml) sets ten checks: improvement over
  seasonal naive, no level worse than a moving average, bias, interval coverage, coherence and
  no negative forecasts. Tests feed in a baseline in place of the model, narrowed intervals and
  biased, incoherent forecasts, and each one fails its check.
- **Calibrated intervals.** Split-conformal: scaled errors from seven earlier origins, pooled
  across horizons within each level. Pooling matters at the top, where one series gives one
  error per window.
- **Explained.** TreeSHAP contributions are grouped into promotions, seasonality and holidays,
  and recent sales level.

## Ask it

```console
$ dfc ask "What is the forecast for Snacks in North next 4 weeks?"
Forecast for Snacks / North over the next 4 weeks: 10,130 units.

$ dfc ask "Top 5 products by growth in South"
Top 5 products by growth vs the same weeks last year over the next 8 weeks in South:
BEV-02 (+35.4%), HOU-12 (+29.6%), HOU-09 (+29.3%), BEV-10 (+26.5%), SNA-07 (+25.1%).

$ dfc ask "Which promotions are planned for Household next month?"
3 products have promotions planned for Household in the next 4 weeks, adding about 1,177 units
over what they would sell without them (forecast with and without the discount).

$ dfc ask "Why is BEV-03 at S01 forecast to change?"
BEV-03 / store S01: about 57.0 units a week forecast against 67.4 a week over the last eight
weeks (down). Largest drivers: seasonality and holidays (lowers it), recent sales level (raises it).

$ dfc ask "what is the weather"
I can't answer that. I can answer: the forecast for an entity (...), rankings (...), ...
```

Every number comes from the forecast tables, and tests recompute each answer independently
with pandas. A language model can be plugged in to rephrase answers (`insights.narrate`). Any
rephrasing that contains a number not in the original answer is discarded.

## Run it

```bash
pip install -e ".[dev]"
dfc run --report backtest-report.md --tracking-uri sqlite:///output/mlflow.db --check   # about a minute
dfc ask "how does S03 compare with last year"
dfc serve                                     # GET /forecast?series=region=North, POST /ask
```

With your own data: `dfc run --sales sales.csv --promotions promotions.csv`, where sales have
`week, store, region, category, sku, units` and promotions have `sku, week, discount`.

On Airflow, [`dags/demand_forecast.py`](dags/demand_forecast.py) runs backtest → gate → publish
every Monday. The gate is its own task, so a forecast that fails backtesting stops before
anything is published. CI runs the DAG with `airflow dags test` on Airflow 3.4.

## Engineering

- 20 tests, 98% line coverage (gate: 90%), plus the DAG test under Airflow.
- `ruff`, `mypy --strict`, Python 3.11 to 3.13.
- CI:
  - lint, types and tests on three Python versions;
  - the forecast policy gate, with the backtest report in the job summary;
  - the DAG run end to end on Airflow 3.4;
  - a wheel install outside the repository;
  - CodeQL, pip-audit and Dependabot.
- Design notes: [architecture](docs/architecture.md) and [decision records](docs/adr).

## Layout

```
policy/forecast_policy.yaml   backtest origins and the ten checks
dags/demand_forecast.py       Airflow DAG: backtest, gate, publish
src/demandcast/
  generate.py                 synthetic weekly sales with known drivers
  hierarchy.py                five levels, the summing matrix
  forecast.py                 features from the origin, the global LightGBM, baselines
  reconcile.py                bottom-up, MinT with shrinkage, coherence
  backtest.py                 rolling origin, metrics by level, conformal intervals
  policy.py                   the policy and its checks
  publish.py                  next eight weeks: intervals, no-promotion forecast, drivers
  insights.py                 plain-language questions over the forecast
  tracking.py, report.py      MLflow and the backtest report
  api.py, cli.py              HTTP API and the dfc command
```

## Known limits

- The data is synthetic: three years, 720 store-products. The structure is realistic, but real
  data adds stock-outs, which censor demand, and assortment changes. Neither is modelled.
- Promotions are chain-wide per product. Store-level promotions and cannibalisation between
  products are not modelled.
- Intervals for arbitrary groupings (all Snacks across stores) are not provided. Questions about
  them get point forecasts; nodes of the hierarchy get intervals.

## License

MIT. See [LICENSE](LICENSE).
