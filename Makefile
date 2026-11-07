.PHONY: install check lint test report

install:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .
	mypy

test:
	pytest --cov --cov-report=term-missing

check: lint test

report:
	dfc run --out output --report docs/backtest-report.md --check
