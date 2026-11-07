from __future__ import annotations

import numpy as np
import pytest

from demandcast import forecast as F
from demandcast import reconcile as R


def test_hierarchy_shape_and_sums(data: F.Data) -> None:
    h = data.hierarchy
    counts = {lv: len(h.index(lv)) for lv in ["total", "region", "store", "store_category", "store_sku"]}
    assert counts == {"total": 1, "region": 3, "store": 12, "store_category": 48, "store_sku": 720}
    assert h.S.shape == (784, 720) and (h.S[-720:] == np.eye(720)).all()
    assert np.allclose(data.Y[0], data.Y[-720:].sum(axis=0))
    north = h.index("region")[0]
    assert np.allclose(data.Y[north], data.Y[h.index("store")][:4].sum(axis=0))
    assert set(h.series.loc[h.index("store_sku"), "region"]) == {"North", "South", "West"}


def test_reconciliation_is_coherent_and_preserves_coherent_input(data: F.Data) -> None:
    S = data.hierarchy.S
    rng = np.random.default_rng(0)
    bottom = rng.gamma(2.0, 5.0, size=(720, 8))
    coherent = S @ bottom
    noisy = coherent * rng.normal(1, 0.1, size=coherent.shape)
    residuals = rng.normal(0, 1, size=(40, 784)) * np.sqrt(coherent.mean(axis=1))
    assert R.incoherence(S, noisy) > 1
    for out in (R.bottom_up(S, noisy), R.mint(S, noisy, residuals)):
        assert R.incoherence(S, out) < 1e-6
    assert np.allclose(R.mint(S, coherent, residuals), coherent, rtol=1e-6, atol=1e-6)


def test_shrunk_covariance() -> None:
    rng = np.random.default_rng(1)
    few = rng.normal(size=(10, 50))
    W = R.shrunk_covariance(few)
    assert np.allclose(W, W.T) and np.linalg.eigvalsh(W).min() > 0, "positive definite even with more series than observations"
    sample_corr = np.corrcoef(few.T)
    shrunk_corr = W / np.sqrt(np.outer(np.diag(W), np.diag(W)))
    off = ~np.eye(50, dtype=bool)
    assert np.abs(shrunk_corr[off]).mean() < np.abs(sample_corr[off]).mean()


def test_generator_structure(inputs: tuple) -> None:  # type: ignore[type-arg]
    sales, promos = inputs
    assert len(sales) == 156 * 720 and sales["units"].min() >= 0
    on_promo = sales[sales["discount"] > 0]["units"].mean()
    off_promo = sales[(sales["discount"] == 0) & (sales["units"] > 0)]["units"].mean()
    assert on_promo > 1.2 * off_promo
    thanksgiving = sales[sales["week"] == "2023-11-20"]["holiday"].iloc[0]
    assert thanksgiving == 1.0 and sales[sales["week"] == "2023-11-13"]["holiday"].iloc[0] == 0.0
    assert promos["week"].max() > sales["week"].max(), "promotions are planned beyond the history"
    again, _ = __import__("demandcast.generate", fromlist=["generate"]).generate()
    assert again["units"].equals(sales["units"])


def test_features_use_only_the_past(data: F.Data) -> None:
    X1, y1, s1 = F.design(data, 100)
    future = data.Y.copy()
    data.Y[:, 101:] *= 3.0
    try:
        X2, y2, s2 = F.design(data, 100)
    finally:
        data.Y[:] = future
    assert np.allclose(s1, s2)
    past = [c for c in F.FEATURES if c not in ("last_year", "last_year_window")]
    assert np.allclose(X1[past].to_numpy(), X2[past].to_numpy(), equal_nan=True)
    assert not np.allclose(y1, y2), "only the target looks forward"
    Xf, yf, _ = F.design(data, data.observed - 1, with_target=False)
    assert np.isnan(yf).all() and len(Xf) == 784 * F.HORIZON


def test_baselines(data: F.Data) -> None:
    sn = F.seasonal_naive(data, 100)
    assert np.allclose(sn[:, 0], data.Y[:, 49])
    early = F.seasonal_naive(data, 20)
    assert np.allclose(early[:, 0], data.Y[:, 20])
    ma = F.moving_average(data, 100)
    assert ma.shape == (784, 8) and np.allclose(ma[:, 3], data.Y[:, 93:101].mean(axis=1))
    with pytest.raises(IndexError):
        F.actuals(data, 10_000)[0, 0]
