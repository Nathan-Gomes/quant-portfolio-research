"""The allocation rules this experiment compares.

An optimizer is only interesting relative to something. Comparing it with SPY
alone answers "did picking assets beat holding US equities", which is a question
about asset mix rather than about optimization. The question a portfolio team
actually asks is narrower and harder: **did solving for weights beat not
bothering?**

So the naive rules are here as first-class strategies rather than as an
afterthought. Equal weight in particular is a serious competitor, not a straw
man — DeMiguel, Garlappi and Uppal (2009) found it beat a long list of optimized
policies out of sample across many datasets, because the estimation error in the
optimizer's inputs outweighed the benefit of optimizing them.

Every rule has the same signature, so the walk-forward engine runs them on
identical rebalance dates and identical training windows. That makes the
comparison paired: differences come from the rule, not from a luckier draw.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .estimators import estimate
from .optimize import maximum_sharpe_weights, minimum_variance_weights


@dataclass
class Settings:
    """What a rule is allowed to assume, separate from what it optimizes."""

    maximum_weight: float = 1.0
    risk_free_rate: float = 0.03
    periods: int = 252
    covariance_estimator: str = "ledoit_wolf"
    mean_estimator: str = "james_stein"
    static_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class Strategy:
    name: str
    label: str
    description: str
    solve: Callable[[pd.DataFrame, Settings], pd.Series]
    needs_expected_returns: bool = False
    optimizes: bool = True


REGISTRY: dict[str, Strategy] = {}


def register(strategy: Strategy) -> Strategy:
    if strategy.name in REGISTRY:
        raise ValueError(f"A strategy named '{strategy.name}' is already registered.")
    REGISTRY[strategy.name] = strategy
    return strategy


def get(name: str) -> Strategy:
    if name not in REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Registered: {', '.join(sorted(REGISTRY))}.")
    return REGISTRY[name]


def catalogue() -> list[dict]:
    return [
        {"name": s.name, "label": s.label, "description": s.description,
         "needs_expected_returns": s.needs_expected_returns, "optimizes": s.optimizes}
        for s in sorted(REGISTRY.values(), key=lambda s: s.name)
    ]


def _normalize(weights: pd.Series, columns) -> pd.Series:
    weights = weights.reindex(columns).fillna(0.0).clip(lower=0.0)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("An allocation rule produced weights that do not add up to anything.")
    return weights / total


# --------------------------------------------------------------------------- #
# Optimized rules
# --------------------------------------------------------------------------- #

def _maximum_sharpe(window: pd.DataFrame, settings: Settings) -> pd.Series:
    inputs = estimate(window, settings.covariance_estimator, settings.mean_estimator, settings.periods)
    return maximum_sharpe_weights(
        window, settings.maximum_weight, settings.risk_free_rate, settings.periods,
        covariance=inputs["covariance"], expected_returns=inputs["means"])


register(Strategy(
    name="maximum_sharpe",
    label="Maximum Sharpe",
    description="The capped long-only mix with the highest estimated risk-adjusted return. "
                "Depends on expected returns, which are the weakest input any optimizer takes.",
    solve=_maximum_sharpe,
    needs_expected_returns=True,
))


def _minimum_variance(window: pd.DataFrame, settings: Settings) -> pd.Series:
    inputs = estimate(window, settings.covariance_estimator, settings.mean_estimator, settings.periods)
    return minimum_variance_weights(
        window, settings.maximum_weight, settings.periods, covariance=inputs["covariance"])


register(Strategy(
    name="minimum_variance",
    label="Minimum variance",
    description="The lowest-variance capped mix. Needs no return forecast, which is why it tends to "
                "survive out of sample better than rules that do.",
    solve=_minimum_variance,
))


def _risk_parity(window: pd.DataFrame, settings: Settings) -> pd.Series:
    """Equal risk contribution, by the convex log-barrier formulation.

    Minimizing ½wᵀΣw − Σ bᵢ log wᵢ has first-order condition Σw = b / w, which is
    exactly the statement that every holding contributes the same share of risk.
    Fixing the other weights leaves a quadratic in wᵢ with one positive root, so
    cyclical coordinate descent solves it without a solver. Written instead as
    "minimize the dispersion of risk contributions" the same problem is not
    convex. Spinu (2013); Maillard, Roncalli and Teïletche (2010).
    """
    inputs = estimate(window, settings.covariance_estimator, settings.mean_estimator, settings.periods)
    covariance = inputs["covariance"].to_numpy(dtype=float)
    assets = len(covariance)
    budget = np.full(assets, 1.0 / assets)
    weights = np.full(assets, 1.0 / np.sqrt(assets))

    for _ in range(3000):
        movement = 0.0
        for i in range(assets):
            cross = float(covariance[i] @ weights - covariance[i, i] * weights[i])
            a = float(covariance[i, i])
            if a <= 0:
                continue
            updated = (-cross + np.sqrt(cross * cross + 4 * a * budget[i])) / (2 * a)
            movement = max(movement, abs(updated - weights[i]))
            weights[i] = updated
        if movement < 1e-14:
            break

    solution = pd.Series(weights / weights.sum(), index=inputs["covariance"].columns)
    if solution.max() > settings.maximum_weight + 1e-9:
        # Exact equal contribution and a binding cap cannot both hold; the cap is
        # the mandate, so the solution is projected onto it.
        solution = _project_to_cap(solution, settings.maximum_weight)
    return solution


def _project_to_cap(weights: pd.Series, maximum_weight: float) -> pd.Series:
    """Nearest fully invested, capped, long-only vector to the one given."""
    values = weights.to_numpy(dtype=float)
    low, high = values.min() - maximum_weight - 1, values.max() + 1
    for _ in range(200):
        shift = (low + high) / 2
        clipped = np.clip(values - shift, 0.0, maximum_weight)
        total = clipped.sum()
        if abs(total - 1) < 1e-13:
            break
        if total > 1:
            low = shift
        else:
            high = shift
    return pd.Series(clipped / clipped.sum(), index=weights.index)


register(Strategy(
    name="risk_parity",
    label="Risk parity",
    description="Weights at which every holding supplies the same share of portfolio volatility. "
                "Uses the covariance but no return forecast.",
    solve=_risk_parity,
))


# --------------------------------------------------------------------------- #
# Rules that do not optimize
# --------------------------------------------------------------------------- #

def _equal_weight(window: pd.DataFrame, settings: Settings) -> pd.Series:
    return _normalize(pd.Series(1.0, index=window.columns), window.columns)


register(Strategy(
    name="equal_weight",
    label="Equal weight",
    description="One over N. The benchmark every optimizer should be made to beat, and frequently "
                "is not, because it has no parameters to estimate and therefore none to get wrong.",
    solve=_equal_weight,
    optimizes=False,
))


def _inverse_volatility(window: pd.DataFrame, settings: Settings) -> pd.Series:
    deviation = window.std(ddof=1)
    if (deviation <= 0).any():
        raise ValueError("Inverse-volatility weights need a non-zero volatility for every holding.")
    return _normalize(1.0 / deviation, window.columns)


register(Strategy(
    name="inverse_volatility",
    label="Inverse volatility",
    description="Weights proportional to one over each asset's own volatility. Ignores correlation, "
                "so it is a heuristic rather than a solution, and it estimates far less than one.",
    solve=_inverse_volatility,
    optimizes=False,
))


def _static(window: pd.DataFrame, settings: Settings) -> pd.Series:
    if not settings.static_weights:
        raise ValueError("A static allocation needs its weights supplied in the settings.")
    return _normalize(pd.Series(settings.static_weights, dtype=float), window.columns)


register(Strategy(
    name="static_allocation",
    label="Static allocation",
    description="A fixed policy mix held throughout, such as sixty forty. Estimates nothing at all "
                "and rebalances back to the same targets.",
    solve=_static,
    optimizes=False,
))
