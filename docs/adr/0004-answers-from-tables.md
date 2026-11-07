# ADR-0004: Answers are computed from the forecast tables

**Status:** accepted

## Context

A language model asked about forecasts will produce fluent numbers that may not exist.

## Decision

Parse a fixed set of question shapes, compute every number from the forecast tables, and refuse
other questions with a list of what can be asked. A language model may rephrase an answer;
the rephrasing is discarded if it contains any number not in the answer.

## Consequences

- Tests recompute each answer independently.
- Open-ended questions need a new question shape, not a better prompt.
