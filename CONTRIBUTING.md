# Contributing

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
make check
make report
```

## Changing the model or features

Run `make report` before and after, and paste both WAPE tables in the pull request. A change
that only makes a policy check pass by loosening `policy/forecast_policy.yaml` needs its own
justification.

## Adding a question shape

Add it to `insights.py` and a test in `tests/test_publish_insights.py` that recomputes the
answer with pandas.
