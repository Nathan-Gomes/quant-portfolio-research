"""Walk-forward engine for out-of-sample ETF portfolio research."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .optimize import maximum_sharpe_weights


@dataclass
class BacktestResult:
    returns: pd.DataFrame
    weights: pd.DataFrame
    turnover: pd.Series


def walk_forward_backtest(
    asset_returns: pd.DataFrame,
    benchmark: str,
    lookback_months: int,
    rebalance_frequency_months: int,
    maximum_weight: float,
    transaction_cost_bps: float,
    risk_free_rate: float,
    periods: int = 252,
) -> BacktestResult:
    """Fit on each trailing window and score only the returns that follow it.

    On a rebalance date, weights may use returns through that date's close, but
    the holding period begins on the next available session. This makes the
    timing explicit: a return cannot help choose the weights that earn it.
    """
    returns = asset_returns.dropna().sort_index()
    month_ends = returns.resample("ME").last().index
    eligible = month_ends[month_ends >= returns.index.min() + pd.DateOffset(months=lookback_months)]
    rebalance_dates = eligible[::rebalance_frequency_months]
    if len(rebalance_dates) < 2:
        raise ValueError("Not enough history for the requested walk-forward backtest")

    strategy = pd.Series(index=returns.index, dtype=float)
    weight_rows: list[pd.Series] = []
    turnover_values: dict[pd.Timestamp, float] = {}
    previous_weights = pd.Series(0.0, index=returns.columns)

    for index, rebalance_date in enumerate(rebalance_dates):
        training_start = rebalance_date - pd.DateOffset(months=lookback_months)
        # The optimizer sees only the completed trailing window at this date.
        training = returns.loc[(returns.index > training_start) & (returns.index <= rebalance_date)]
        weights = maximum_sharpe_weights(training, maximum_weight, risk_free_rate, periods).reindex(returns.columns)
        weights.name = rebalance_date
        weight_rows.append(weights)

        turnover = float((weights - previous_weights).abs().sum()) if index else float(weights.abs().sum())
        turnover_values[rebalance_date] = turnover
        previous_weights = weights

        next_date = rebalance_dates[index + 1] if index + 1 < len(rebalance_dates) else returns.index.max()
        # Exclude the rebalance date so selected weights apply only to later returns.
        holding = returns.loc[(returns.index > rebalance_date) & (returns.index <= next_date)]
        holding_returns = holding.mul(weights, axis=1).sum(axis=1)
        if not holding_returns.empty:
            # Pay the declared turnover cost once when the new allocation starts.
            holding_returns.iloc[0] -= turnover * transaction_cost_bps / 10_000.0
            strategy.loc[holding_returns.index] = holding_returns

    strategy = strategy.dropna()
    benchmark_returns = returns.loc[strategy.index, benchmark]
    result_returns = pd.DataFrame({"Optimized portfolio": strategy, f"Benchmark ({benchmark})": benchmark_returns})
    weights_frame = pd.DataFrame(weight_rows)
    weights_frame.index.name = "rebalance_date"
    return BacktestResult(result_returns, weights_frame, pd.Series(turnover_values, name="turnover"))
