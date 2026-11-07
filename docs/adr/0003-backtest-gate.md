# ADR-0003: A backtest gate before anything is published

**Status:** accepted

## Context

A forecast that is worse than last year's sales, biased, or overconfident does more harm than
no forecast, because people act on it.

## Decision

Every run retrains at four rolling origins over the last 32 weeks and must pass ten checks
before publishing: accuracy against seasonal naive and a moving average, bias, interval
coverage, coherence and no negative values. On Airflow the gate is its own task.

## Consequences

- The policy's thresholds are in a reviewed file, and tests show each check failing on a
  planted problem.
- Coverage is gated only where it can be measured: 32 test points at the total level cannot
  distinguish 80% from 90%.
