"""Tests for the allocation rules and the paired comparison.

The naive rules matter as much as the optimized ones here: they are what the
optimizer has to beat, so an error in them would flatter or damn the whole
experiment. Each is checked against the property that defines it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_pipeline.backtest import (
    compare_strategies,
    walk_forward_backtest,
)
from quant_pipeline.estimators import (
    estimate,
    james_stein_means,
    ledoit_wolf_covariance,
)
from quant_pipeline.strategies import REGISTRY, Settings, get


@pytest.fixture
def window() -> pd.DataFrame:
    """Six assets with deliberately unequal volatility and a common factor."""
    generator = np.random.default_rng(7)
    factor = generator.normal(0, 0.008, (600, 2))
    loadings = generator.normal(0, 1, (6, 2))
    scale = np.array([0.003, 0.005, 0.007, 0.010, 0.013, 0.018])
    values = factor @ loadings.T + generator.normal(0, 1, (600, 6)) * scale
    dates = pd.bdate_range("2019-01-01", periods=600)
    return pd.DataFrame(values, index=dates, columns=list("ABCDEF"))


@pytest.fixture
def settings() -> Settings:
    return Settings(maximum_weight=0.30, risk_free_rate=0.03, static_weights={"A": 0.6, "B": 0.4})


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_rule_returns_a_fully_invested_long_only_portfolio(name, window, settings):
    weights = get(name).solve(window, settings)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= -1e-12).all()
    assert set(weights.index) == set(window.columns)


@pytest.mark.parametrize("name", ["maximum_sharpe", "minimum_variance", "risk_parity"])
def test_the_optimized_rules_respect_the_cap(name, window, settings):
    weights = get(name).solve(window, settings)
    assert weights.max() <= settings.maximum_weight + 1e-6


def test_equal_weight_is_one_over_n(window, settings):
    weights = get("equal_weight").solve(window, settings)
    assert np.allclose(weights.to_numpy(), 1 / len(window.columns))


def test_inverse_volatility_prefers_the_calmer_asset(window, settings):
    weights = get("inverse_volatility").solve(window, settings)
    deviation = window.std(ddof=1)
    calmest, wildest = deviation.idxmin(), deviation.idxmax()
    assert weights[calmest] > weights[wildest]
    # And the ratio is exactly the inverse ratio of their volatilities.
    assert weights[calmest] / weights[wildest] == pytest.approx(
        deviation[wildest] / deviation[calmest], rel=1e-9)


def test_risk_parity_equalizes_risk_contributions(window):
    """The property that defines the rule, checked rather than assumed."""
    loose = Settings(maximum_weight=1.0)
    weights = get("risk_parity").solve(window, loose)
    covariance = estimate(window)["covariance"].to_numpy()
    values = weights.to_numpy()
    volatility = np.sqrt(values @ covariance @ values)
    contributions = values * (covariance @ values) / volatility
    assert np.ptp(contributions / volatility) < 1e-6


def test_minimum_variance_is_the_calmest_available_mix(window, settings):
    weights = get("minimum_variance").solve(window, settings)
    covariance = estimate(window)["covariance"].to_numpy()
    best = weights.to_numpy() @ covariance @ weights.to_numpy()
    generator = np.random.default_rng(3)
    for _ in range(300):
        candidate = generator.dirichlet(np.ones(len(window.columns)))
        if candidate.max() > settings.maximum_weight:
            continue                      # not a feasible portfolio under this mandate
        assert candidate @ covariance @ candidate >= best - 1e-9


def test_an_unknown_rule_names_what_is_available():
    with pytest.raises(ValueError, match="Registered:"):
        get("no_such_rule")


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #

def test_shrinkage_pulls_the_covariance_toward_a_diagonal(window):
    shrunk, intensity = ledoit_wolf_covariance(window)
    sample = window.cov() * 252
    assert 0 < intensity <= 1
    off_diagonal = ~np.eye(len(window.columns), dtype=bool)
    assert np.abs(shrunk.to_numpy()[off_diagonal]).sum() < np.abs(sample.to_numpy()[off_diagonal]).sum()


def test_shrinkage_rises_when_observations_are_scarce(window):
    _, plenty = ledoit_wolf_covariance(window)
    _, scarce = ledoit_wolf_covariance(window.iloc[:40])
    assert scarce > plenty


def test_mean_shrinkage_pulls_toward_the_average_and_preserves_it(window):
    raw = window.mean() * 252
    shrunk, intensity = james_stein_means(window)
    assert 0 <= intensity <= 1
    assert shrunk.mean() == pytest.approx(raw.mean())
    assert shrunk.std() <= raw.std() + 1e-12


@pytest.mark.parametrize("bad", [
    {"covariance_estimator": "nonsense"},
    {"mean_estimator": "nonsense"},
])
def test_an_unknown_estimator_is_refused(window, bad):
    with pytest.raises(ValueError):
        estimate(window, **bad)


# --------------------------------------------------------------------------- #
# The paired comparison
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def market() -> pd.DataFrame:
    generator = np.random.default_rng(11)
    dates = pd.bdate_range("2015-01-01", periods=1800)
    drift = np.array([0.0002, 0.0003, 0.00025, 0.0004, 0.00012, 0.00018])
    scale = np.array([0.004, 0.011, 0.009, 0.013, 0.003, 0.008])
    values = generator.normal(drift, scale, (len(dates), 6))
    return pd.DataFrame(values, index=dates, columns=["AGG", "SPY", "EFA", "EEM", "BIL", "GLD"])


def test_every_strategy_is_scored_on_the_same_dates(market):
    """Paired, not separately run: a difference must come from the rule."""
    names = ["maximum_sharpe", "minimum_variance", "equal_weight"]
    result = compare_strategies(
        market, benchmark="SPY", strategies=names, lookback_months=24,
        rebalance_frequency_months=3, maximum_weight=0.4, transaction_cost_bps=10,
        risk_free_rate=0.03)
    assert len(result.returns.columns) == len(names) + 1
    assert not result.returns.isna().any().any()
    for name in names:
        assert result.by_strategy[name].returns.index.equals(result.returns.index)


def test_a_rule_that_never_trades_costs_less_than_one_that_does(market):
    result = compare_strategies(
        market, benchmark="SPY", strategies=["maximum_sharpe", "equal_weight"],
        lookback_months=24, rebalance_frequency_months=3, maximum_weight=0.4,
        transaction_cost_bps=10, risk_free_rate=0.03)
    assert (result.by_strategy["maximum_sharpe"].diagnostics["annual_turnover"]
            > result.by_strategy["equal_weight"].diagnostics["annual_turnover"])


@pytest.mark.parametrize("name", ["maximum_sharpe", "minimum_variance", "risk_parity", "equal_weight"])
def test_no_rule_can_see_the_future(market, name):
    """Rewrite the last month of returns; nothing decided earlier may move."""
    altered = market.copy()
    altered.iloc[-21:] += 0.05

    original = walk_forward_backtest(
        market, benchmark="SPY", strategy=name, lookback_months=24,
        rebalance_frequency_months=3, maximum_weight=0.4, transaction_cost_bps=10,
        risk_free_rate=0.03)
    changed = walk_forward_backtest(
        altered, benchmark="SPY", strategy=name, lookback_months=24,
        rebalance_frequency_months=3, maximum_weight=0.4, transaction_cost_bps=10,
        risk_free_rate=0.03)

    shared = original.weights.index.intersection(changed.weights.index)[:-1]
    assert len(shared) > 3
    pd.testing.assert_frame_equal(
        original.weights.loc[shared], changed.weights.loc[shared], atol=1e-9)
