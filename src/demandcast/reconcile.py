"""Forecast reconciliation: make forecasts add up across the hierarchy.

Base forecasts made independently at each level do not add up. Reconciliation maps them to
coherent forecasts y~ = S G y^. Bottom-up uses only the bottom forecasts; MinT (Wickramasuriya,
Athanasopoulos and Hyndman, 2019) uses every level, weighted by a shrunk estimate of the base
forecast errors' covariance.
"""

from __future__ import annotations

import numpy as np


def bottom_up(S: np.ndarray, base: np.ndarray) -> np.ndarray:
    n_bottom = S.shape[1]
    return np.asarray(S @ base[-n_bottom:], dtype=float)


def shrunk_covariance(residuals: np.ndarray) -> np.ndarray:
    """Schäfer-Strimmer shrinkage of the residual covariance towards its diagonal.

    residuals: observations × series.
    """
    n, _ = residuals.shape
    centered = residuals - residuals.mean(axis=0)
    sample = centered.T @ centered / n
    sd = np.sqrt(np.clip(np.diag(sample), 1e-12, None))
    std = centered / sd
    corr = std.T @ std / n
    # Variance of the off-diagonal correlation estimates, for the optimal shrinkage intensity.
    w2 = (std**2).T @ (std**2) / n
    var_corr = (w2 - corr**2) * n / (n - 1) ** 2
    off = ~np.eye(len(corr), dtype=bool)
    lam = float(np.clip(var_corr[off].sum() / (corr[off] ** 2).sum(), 0.0, 1.0))
    shrunk_corr = (1 - lam) * corr
    np.fill_diagonal(shrunk_corr, 1.0)
    return np.asarray(shrunk_corr * np.outer(sd, sd), dtype=float)


def mint(S: np.ndarray, base: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """MinT with a shrunk covariance. base: series (× columns); residuals: observations × series."""
    W = shrunk_covariance(residuals)
    W_inv = np.linalg.pinv(W, hermitian=True)
    G = np.linalg.solve(S.T @ W_inv @ S, S.T @ W_inv)
    return np.asarray(S @ (G @ base), dtype=float)


def incoherence(S: np.ndarray, forecasts: np.ndarray) -> float:
    """Largest gap between a forecast and the sum of its bottom series."""
    return float(np.max(np.abs(forecasts - S @ forecasts[-S.shape[1] :])))
