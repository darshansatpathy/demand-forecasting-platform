# ADR-0001: One global model, direct multi-horizon

**Status:** accepted

## Context

720 product-store series are too many to tune one model each, and many are short or
intermittent. A recursive model feeds its own forecasts back in, so errors compound with
horizon.

## Decision

Train one LightGBM model across every series and level, with features scaled by each series'
level, and a horizon feature with lags measured from the origin, so each of the eight horizons
is forecast directly.

## Consequences

- Slow sellers borrow strength from similar series; new products are forecast from their
  short history and their category's patterns.
- Training is about three seconds per origin, which makes retraining at every backtest origin
  affordable.
