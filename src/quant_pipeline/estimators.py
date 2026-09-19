"""Estimating the inputs an optimizer consumes.

A mean-variance optimizer is only as good as the covariance and the expected
returns handed to it, and both are estimated from a short window of noisy data.
This module makes that estimation an explicit, switchable choice rather than an
implicit call to ``DataFrame.cov()``.

Why it matters here: the research configuration estimates from a 36-month
trailing window, which is roughly 756 daily observations for six assets. The
covariance is comfortably determined at that ratio; the *means* are not. The
standard error of a mean return estimated from three years of daily data is
roughly the annual volatility divided by the square root of three — for a 17%
volatility asset, about ten percentage points. An optimizer maximizing Sharpe
treats a 12% estimate and a 2% estimate as different facts when they are barely
distinguishable observations.

So the defaults here shrink both, and the raw estimators remain available so the
difference can be measured rather than asserted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COVARIANCE_ESTIMATORS = ("sample", "ledoit_wolf")
MEAN_ESTIMATORS = ("sample", "james_stein")


def sample_covariance(returns: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    """The plain sample covariance, annualized."""
    return returns.cov() * periods


def ledoit_wolf_covariance(returns: pd.DataFrame, periods: int = 252) -> tuple[pd.DataFrame, float]:
    """Shrinkage toward a scaled identity, with the intensity estimated from the data.

    Ledoit and Wolf (2004). The sample covariance is unbiased but noisy, and its
    error concentrates in the smallest eigenvalues — exactly the directions a
    variance minimizer loads into. Pulling the matrix toward a diagonal target
    trades a little bias for a large reduction in variance, and the amount to
    pull is estimated rather than chosen.

    Returns the matrix and the intensity, so a report can state how much
    structure was imposed.
    """
    values = returns.to_numpy(dtype=float)
    observations, assets = values.shape
    if observations < 2:
        raise ValueError("Covariance estimation needs at least two observations.")

    centred = values - values.mean(axis=0)
    sample = centred.T @ centred / observations

    mu = np.trace(sample) / assets
    target = mu * np.eye(assets)
    dispersion = float(np.sum((sample - target) ** 2) / assets)

    # Mean squared error of the sample entries, from the fourth moments.
    squared = centred ** 2
    error = float(np.sum(squared.T @ squared) / observations - np.sum(sample ** 2))
    error = max(error, 0.0) / (observations * assets)
    error = min(error, dispersion)

    intensity = 0.0 if dispersion <= 0 else float(np.clip(error / dispersion, 0.0, 1.0))
    shrunk = intensity * target + (1 - intensity) * sample
    shrunk *= observations / (observations - 1)      # match the unbiased convention
    frame = pd.DataFrame(shrunk * periods, index=returns.columns, columns=returns.columns)
    return frame, intensity


def sample_means(returns: pd.DataFrame, periods: int = 252) -> pd.Series:
    return returns.mean() * periods


def james_stein_means(returns: pd.DataFrame, periods: int = 252) -> tuple[pd.Series, float]:
    """Shrink expected returns toward their cross-sectional average.

    An optimizer handed raw historical means will concentrate in whichever asset
    happened to run hardest over the window, which is a statement about the
    window rather than about the asset. Shrinking toward the grand mean concedes
    that the sample cannot tell these assets apart as confidently as the numbers
    suggest, and the intensity follows from how dispersed the estimates are
    relative to their own standard errors.
    """
    means = returns.mean() * periods
    observations, assets = returns.shape
    grand = float(means.mean())
    spread = float(((means - grand) ** 2).sum())
    if spread <= 0 or assets < 3:
        return means, 0.0
    variance = float((returns.var(ddof=1) * periods ** 2).sum() / observations)
    intensity = float(np.clip((assets - 2) * variance / (observations * spread), 0.0, 1.0))
    return grand + (1 - intensity) * (means - grand), intensity


def estimate(returns: pd.DataFrame, covariance_estimator: str = "ledoit_wolf",
             mean_estimator: str = "james_stein", periods: int = 252) -> dict:
    """Everything an optimizer needs, with a record of how it was produced."""
    if covariance_estimator not in COVARIANCE_ESTIMATORS:
        raise ValueError(f"Covariance estimator must be one of {', '.join(COVARIANCE_ESTIMATORS)}.")
    if mean_estimator not in MEAN_ESTIMATORS:
        raise ValueError(f"Mean estimator must be one of {', '.join(MEAN_ESTIMATORS)}.")

    if covariance_estimator == "sample":
        covariance, covariance_intensity = sample_covariance(returns, periods), 0.0
    else:
        covariance, covariance_intensity = ledoit_wolf_covariance(returns, periods)

    if mean_estimator == "sample":
        means, mean_intensity = sample_means(returns, periods), 0.0
    else:
        means, mean_intensity = james_stein_means(returns, periods)

    return {
        "covariance": covariance,
        "means": means,
        "covariance_estimator": covariance_estimator,
        "mean_estimator": mean_estimator,
        "covariance_shrinkage": covariance_intensity,
        "mean_shrinkage": mean_intensity,
        "observations": len(returns),
    }
