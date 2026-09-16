import numpy as np
import pandas as pd

from quant_pipeline.backtest import walk_forward_backtest


def test_walk_forward_uses_prior_history_and_produces_normalized_weights():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2018-01-01", "2025-12-31")
    returns = pd.DataFrame(rng.normal(0.0003, 0.01, (len(dates), 6)), index=dates, columns=["EWC", "SPY", "EFA", "EEM", "AGG", "GLD"])
    result = walk_forward_backtest(returns, "SPY", 36, 3, 0.30, 10, 0.03)

    assert not result.returns.empty
    assert result.returns.index.min() > returns.index.min() + pd.DateOffset(months=36)
    assert np.allclose(result.weights.sum(axis=1), 1.0)
    assert (result.weights <= 0.30 + 1e-8).all().all()


def test_later_returns_do_not_change_already_formed_weights():
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2018-01-01", "2025-12-31")
    columns = ["EWC", "SPY", "EFA", "EEM", "AGG", "GLD"]
    returns = pd.DataFrame(rng.normal(0.0003, 0.01, (len(dates), len(columns))), index=dates, columns=columns)
    baseline = walk_forward_backtest(returns, "SPY", 36, 3, 0.30, 10, 0.03)

    changed_after = pd.Timestamp("2024-01-02")
    revised_history = returns.copy()
    revised_history.loc[revised_history.index >= changed_after] += 0.02
    rerun = walk_forward_backtest(revised_history, "SPY", 36, 3, 0.30, 10, 0.03)

    established_dates = baseline.weights.index[baseline.weights.index < changed_after]
    pd.testing.assert_frame_equal(
        baseline.weights.loc[established_dates],
        rerun.weights.loc[established_dates],
    )
