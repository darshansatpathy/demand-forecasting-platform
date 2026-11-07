# ADR-0002: Publish MinT-reconciled forecasts

**Status:** accepted

## Context

Planners use forecasts at every level. If the store forecast is not the sum of its products,
the plans contradict each other. Bottom-up is coherent but discards the better signal available
at aggregate levels.

## Decision

Reconcile with MinT using a shrunk covariance of base forecast errors from validation windows.

## Consequences

- In backtesting, MinT is the most accurate at total, region and store level and removes the
  bottom-up bias (−2.9% to −0.9%). At product level it is 0.4 points worse than bottom-up; the
  report shows this, and the policy gates on store and product accuracy against the baselines.
