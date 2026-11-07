"""Command line: run (backtest, gate, publish), ask, serve."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from demandcast import __version__
from demandcast import policy as policy_mod


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dfc", description="Hierarchical demand forecasting.")
    parser.add_argument("--version", action="version", version=f"dfc {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="backtest, check against the policy, publish the forecast")
    p.add_argument("--out", type=Path, default=Path("output"))
    p.add_argument("--sales", type=Path, help="weekly sales CSV (default: the bundled synthetic generator)")
    p.add_argument("--promotions", type=Path, help="planned promotions CSV")
    p.add_argument("--policy", type=Path)
    p.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    p.add_argument("--report", type=Path)
    p.add_argument("--check", action="store_true", help="exit 1 if a policy check fails")

    p = sub.add_parser("ask", help="ask a question about the published forecast")
    p.add_argument("question")
    p.add_argument("--forecast", type=Path, default=Path("output/forecast"))
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("serve", help="run the HTTP API")
    p.add_argument("--forecast", type=Path, default=Path("output/forecast"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    return int(COMMANDS[args.command](args))


def _run(args: argparse.Namespace) -> int:
    from demandcast import pipeline, report

    policy = policy_mod.load(args.policy)
    result = pipeline.run(policy, args.out, args.sales, args.promotions, args.tracking_uri)
    for c in result.checks:
        print(f"{'PASS' if c.passed else 'FAIL'}  {c.name}: {c.value:.4f} ({c.threshold})")
    text = report.render(result.backtest, result.checks, policy, result.published)
    if args.report:
        args.report.write_text(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write(text)
    print("published" if result.published else "not published: a policy check failed")
    return 1 if args.check and not result.passed else 0


def _ask(args: argparse.Namespace) -> int:
    from demandcast.insights import answer
    from demandcast.publish import Published

    a = answer(args.question, Published.load(args.forecast))
    if args.json:
        print(json.dumps({"intent": a.intent, "answer": a.text, "rows": a.rows, "facts": a.facts}, indent=1, default=str))
    else:
        print(a.text)
        for row in a.rows[:20]:
            print("  " + ", ".join(f"{k}: {v}" for k, v in row.items()))
    return 0 if a.intent != "unsupported" else 2


def _serve(args: argparse.Namespace) -> int:  # pragma: no cover - runs a server
    import uvicorn

    from demandcast.api import create_app

    uvicorn.run(create_app(args.forecast), host=args.host, port=args.port)
    return 0


COMMANDS = {"run": _run, "ask": _ask, "serve": _serve}
