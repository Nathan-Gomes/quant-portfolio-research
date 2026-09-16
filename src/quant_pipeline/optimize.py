from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _validate_max_weight(asset_count: int, maximum_weight: float) -> None:
    if asset_count * maximum_weight < 1.0:
        raise ValueError("maximum_asset_weight is too low to create a fully invested portfolio")


def minimum_variance_weights(returns: pd.DataFrame, maximum_weight: float, periods: int = 252) -> pd.Series:
    """Solve a long-only, fully invested minimum-variance allocation."""
    clean = returns.dropna()
    assets = list(clean.columns)
    _validate_max_weight(len(assets), maximum_weight)
    covariance = clean.cov().to_numpy() * periods
    initial = np.repeat(1.0 / len(assets), len(assets))
    result = minimize(
        lambda weights: float(weights @ covariance @ weights),
        initial,
        method="SLSQP",
        bounds=[(0.0, maximum_weight)] * len(assets),
        constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Minimum-variance optimization failed: {result.message}")
    return pd.Series(result.x, index=assets, name="weight")


def maximum_sharpe_weights(
    returns: pd.DataFrame, maximum_weight: float, risk_free_rate: float, periods: int = 252
) -> pd.Series:
    """Solve the capped long-only allocation with the highest in-sample Sharpe ratio.

    The caller controls what history reaches this optimizer. In the walk-forward
    backtest, that history ends at each rebalance date rather than the end of
    the full sample.
    """
    clean = returns.dropna()
    assets = list(clean.columns)
    _validate_max_weight(len(assets), maximum_weight)
    expected_returns = clean.mean().to_numpy() * periods
    covariance = clean.cov().to_numpy() * periods
    initial = np.repeat(1.0 / len(assets), len(assets))

    def negative_sharpe(weights: np.ndarray) -> float:
        volatility = np.sqrt(weights @ covariance @ weights)
        return -float((weights @ expected_returns - risk_free_rate) / volatility)

    result = minimize(
        negative_sharpe,
        initial,
        method="SLSQP",
        bounds=[(0.0, maximum_weight)] * len(assets),
        constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Maximum-Sharpe optimization failed: {result.message}")
    return pd.Series(result.x, index=assets, name="weight")
