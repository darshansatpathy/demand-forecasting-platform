from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient

from demandcast.api import create_app
from demandcast.cli import main
from demandcast.generate import products
from demandcast.insights import answer, narrate
from demandcast.publish import Published


@pytest.fixture(scope="module")
def pub(run_dir: Path) -> Published:
    return Published.load(run_dir / "forecast")


def test_published_forecast_is_coherent_and_ordered(pub: Published) -> None:
    f = pub.forecast
    assert len(f) == 784 * 8 and (f["p10"] <= f["p50"]).all() and (f["p50"] <= f["p90"]).all()
    total = f[f["level"] == "total"].set_index("week")["p50"]
    bottom = f[f["level"] == "store_sku"].groupby("week")["p50"].sum()
    assert np.allclose(total, bottom)
    assert (pub.forecast["week"].min() - pub.origin_week).days == 7


def test_promotion_uplift_matches_the_generators_elasticities(pub: Published) -> None:
    """The data is synthetic, so the true uplift is known: (1 - discount) ** elasticity."""
    truth = products(np.random.default_rng(11)).set_index("sku")["elasticity"]
    b = pub.forecast[(pub.forecast["level"] == "store_sku") & (pub.forecast["discount"] > 0)]
    true_ratio = ((1 - b["discount"]) ** b["sku"].map(truth)).median()
    estimated = (b["p50"] / b["no_promo"]).median()
    assert estimated == pytest.approx(true_ratio, rel=0.15)


def test_answers_match_independent_calculations(pub: Published) -> None:
    f, h = pub.forecast, pub.history
    b = f[f["level"] == "store_sku"]
    a = answer("What is the forecast for Snacks in North next 4 weeks?", pub)
    expected = b[(b["category"] == "Snacks") & (b["region"] == "North") & (b["horizon"] <= 4)]["p50"].sum()
    assert a.intent == "forecast" and a.facts["total"] == round(expected) and len(a.rows) == 4 and "p10" not in a.rows[0]
    a = answer("forecast for S03", pub)
    assert a.facts["total"] == round(b[b["store"] == "S03"]["p50"].sum()) and "p90" in a.rows[0]
    a = answer("bottom 3 stores by forecast", pub)
    by_store = b.groupby("store")["p50"].sum().sort_values()
    assert [r["store"] for r in a.rows] == list(by_store.index[:3])
    a = answer("How does Beverages compare with last year?", pub)
    weeks = set(b["week"].unique() - pd.Timedelta(weeks=52))
    ly = h[(h["level"] == "store_sku") & (h["category"] == "Beverages") & h["week"].isin(weeks)]["units"].sum()
    assert a.facts["last_year"] == round(ly)
    a = answer("Which promotions are planned for Household next month?", pub)
    promo = b[(b["category"] == "Household") & (b["horizon"] <= 4) & (b["discount"] > 0)]
    assert a.facts["products"] == promo["sku"].nunique() and all(r["uplift_units"] > 0 for r in a.rows)
    a = answer("Top 5 products by growth in South", pub)
    assert a.intent == "ranking" and len(a.rows) == 5 and a.rows[0]["value"] >= a.rows[-1]["value"]
    a = answer("Why is BEV-03 at S01 forecast to change?", pub)
    assert a.intent == "drivers" and {r["driver"] for r in a.rows} == {"promotions", "seasonality and holidays", "recent sales level"}


def test_scope_limits_and_refusals(pub: Published) -> None:
    assert "covers those" in answer("forecast for the next quarter", pub).text
    assert answer("what is the weather", pub).intent == "unsupported"
    assert answer("", pub).intent == "unsupported"
    assert "name a product" in answer("why is demand changing", pub).text
    assert "Nothing in the forecast matches" in answer("forecast for S99", pub).text
    f = pub.forecast
    quiet = f[(f["level"] == "store_sku") & (f["store"] == "S01")].groupby("sku")["discount"].sum()
    sku = str(quiet[quiet == 0].index[0])
    assert answer(f"promotions for {sku} at S01", pub).text.startswith("No promotions are planned")


def test_narration_cannot_add_numbers(pub: Published) -> None:
    a = answer("forecast for S03", pub)
    assert narrate(a, None) == a.text
    same = narrate(a, lambda prompt: "Store S03 should sell " + f"{a.facts['total']:,}" + " units over 8 weeks.")
    assert same.startswith("Store S03 should sell")
    invented = narrate(a, lambda prompt: "Store S03 should sell 99,999 units, up 42%.")
    assert invented == a.text


def test_cli_ask_and_api(run_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ask", "top 2 regions by forecast", "--forecast", str(run_dir / "forecast")]) == 0
    assert "Top 2 regions" in capsys.readouterr().out
    assert main(["ask", "tell me a joke", "--forecast", str(run_dir / "forecast"), "--json"]) == 2
    assert '"unsupported"' in capsys.readouterr().out
    client = TestClient(create_app(run_dir / "forecast"))
    assert client.get("/healthz").json()["status"] == "ok"
    assert len(client.get("/forecast", params={"series": "region=North"}).json()) == 8
    assert client.get("/forecast", params={"series": "nope"}).status_code == 404
    assert client.post("/ask", json={"question": "forecast for West"}).json()["intent"] == "forecast"
    assert "Rolling origin" in (run_dir / "report.md").read_text()


def test_mlflow_registry(run_dir: Path) -> None:
    v = MlflowClient().get_model_version_by_alias("demand-forecast-global-lightgbm", "champion")
    assert MlflowClient().get_run(v.run_id).data.tags["policy_passed"] == "True"
    assert (run_dir / "forecast" / "model.txt").exists()


def test_failed_policy_is_not_published(tmp_path: Path) -> None:
    strict = tmp_path / "p.yaml"
    text = (Path(__file__).parents[1] / "policy" / "forecast_policy.yaml").read_text().replace("store: 0.15", "store: 0.90")
    strict.write_text(text.replace("[67, 75, 83, 91, 99, 107, 115]", "[99, 107, 115]").replace("[123, 131, 139, 147]", "[147]"))
    assert main(["run", "--out", str(tmp_path / "o"), "--policy", str(strict), "--check"]) == 1
    assert not (tmp_path / "o" / "forecast").exists()
