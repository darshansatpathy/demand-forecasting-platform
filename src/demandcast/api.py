"""HTTP API over the published forecast."""

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from demandcast.insights import answer
from demandcast.publish import Published


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=300)


def create_app(forecast_dir: Path) -> FastAPI:
    published = Published.load(forecast_dir)
    app = FastAPI(title="Demand forecast", version="0.1.1")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "origin": published.origin_week.date().isoformat()}

    @app.get("/forecast")
    def forecast(series: str = Query("total", max_length=100)) -> list[dict[str, Any]]:
        f = published.forecast[published.forecast["id"] == series]
        if f.empty:
            raise HTTPException(404, "unknown series; ids look like total, region=North, store=S01, store=S01|sku=BEV-01")
        return [{"week": r.week.date().isoformat(), "p10": round(r.p10, 1), "p50": round(r.p50, 1), "p90": round(r.p90, 1)} for r in f.itertuples()]

    @app.post("/ask")
    def ask(body: Question) -> dict[str, Any]:
        a = answer(body.question, published)
        return {"intent": a.intent, "answer": a.text, "rows": a.rows, "facts": a.facts}

    return app
