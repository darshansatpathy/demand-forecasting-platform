# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.1] - 2026-11-08

### Fixed

- The forecast feature helper returns a float array explicitly, so mypy passes with the NumPy stubs installed on Python 3.11.

## [0.1.0] - 2026-11-08

### Added

- Synthetic weekly retail sales with seasonality, holidays, promotions with price elasticity,
  launches and intermittent demand.
- Five-level hierarchy (784 series), a global direct multi-horizon LightGBM model and baselines.
- Bottom-up and MinT reconciliation with shrinkage; split-conformal intervals.
- Rolling-origin backtest, a ten-check forecast policy and a generated backtest report.
- Published forecasts with intervals, no-promotion counterfactuals and grouped drivers.
- Plain-language questions over the forecast, an HTTP API, the `dfc` command, MLflow tracking
  and registry, and an Airflow 3 DAG.
