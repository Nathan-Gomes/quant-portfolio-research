"""Does the conclusion survive the assumptions that produced it?

A backtest reports one number per configuration, and the configuration was
chosen by someone. A 36-month lookback, a 30% cap and 10 basis points are all
defensible, and all arbitrary. If the finding only holds at those settings it is
a property of the settings rather than of the strategy, and a reader deserves to
know which.

This sweeps one assumption at a time and reports how the comparison moves. It is
deliberately not an optimizer over the grid: picking the configuration with the
best Sharpe is how a backtest gets overfitted, and the output here is meant to be
read as a stability check, not as a menu.

**One row is not comparable with another across lookbacks.** A longer estimation
window needs more history before the first rebalance, so it starts the evaluation
later and scores a shorter, different period. On this dataset a 60-month lookback
raises the Sharpe ratio of every strategy *and of the benchmark*, which is a
statement about the remaining window rather than about estimation. Compare
strategies within a row; compare rows only with that in mind, and the benchmark
column is there to make the shift visible.
"""

from __future__ import annotations

import pandas as pd

from .backtest import compare_strategies
from .metrics import summarize


def sweep(
    returns: pd.DataFrame,
    benchmark: str,
    strategies: list[str],
    base: dict,
    lookbacks: tuple[int, ...] = (24, 36, 60),
    caps: tuple[float, ...] = (0.20, 0.30, 0.50),
    costs: tuple[float, ...] = (0.0, 10.0, 50.0),
) -> pd.DataFrame:
    """One row per (assumption, value, strategy), with the resulting metrics."""
    rows: list[dict] = []

    def evaluate(label: str, value, overrides: dict) -> None:
        settings = {**base, **overrides}
        try:
            result = compare_strategies(returns, benchmark=benchmark, strategies=strategies, **settings)
        except ValueError as error:      # a window too short for this lookback
            rows.append({"assumption": label, "value": value, "strategy": "—",
                         "note": str(error)})
            return
        for name in strategies:
            run = result.by_strategy[name]
            series = run.returns["Optimized portfolio"]
            metrics = summarize(series, settings["risk_free_rate"], settings["periods"])
            rows.append({
                "assumption": label,
                "value": value,
                "strategy": run.diagnostics["label"],
                "annual_return": metrics["annual_return"],
                "annual_volatility": metrics["annual_volatility"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "maximum_drawdown": metrics["maximum_drawdown"],
                "annual_turnover": run.diagnostics["annual_turnover"],
            })
        reference = result.returns[f"Benchmark ({benchmark})"]
        benchmark_metrics = summarize(reference, settings["risk_free_rate"], settings["periods"])
        rows.append({
            "assumption": label, "value": value, "strategy": f"Benchmark ({benchmark})",
            "annual_return": benchmark_metrics["annual_return"],
            "annual_volatility": benchmark_metrics["annual_volatility"],
            "sharpe_ratio": benchmark_metrics["sharpe_ratio"],
            "maximum_drawdown": benchmark_metrics["maximum_drawdown"],
            "annual_turnover": 0.0,
        })

    for lookback in lookbacks:
        evaluate("lookback_months", lookback, {"lookback_months": lookback})
    for cap in caps:
        evaluate("maximum_weight", cap, {"maximum_weight": cap})
    for cost in costs:
        evaluate("transaction_cost_bps", cost, {"transaction_cost_bps": cost})
    for estimator in ("sample", "ledoit_wolf"):
        evaluate("covariance_estimator", estimator, {"covariance_estimator": estimator})

    return pd.DataFrame(rows)


def ranking_stability(grid: pd.DataFrame, metric: str = "sharpe_ratio",
                      exclude_benchmark: bool = True) -> pd.DataFrame:
    """How often each strategy comes first, across every configuration tried.

    A rule that wins under one setting and loses under the rest has not been
    shown to be better; it has been shown to suit one configuration. Because
    every row shares its evaluation window, ranking within a row is sound even
    where comparing levels across rows is not.
    """
    usable = grid.dropna(subset=[metric])
    if exclude_benchmark:
        usable = usable[~usable["strategy"].str.startswith("Benchmark")]
    winners = (
        usable.sort_values(metric, ascending=False)
        .groupby(["assumption", "value"], sort=False)
        .head(1)
    )
    counts = winners["strategy"].value_counts()
    configurations = len(winners)
    return pd.DataFrame({
        "strategy": counts.index,
        "times_best": counts.to_numpy(),
        "share_of_configurations": counts.to_numpy() / configurations,
    })
