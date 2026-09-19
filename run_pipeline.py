import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from quant_pipeline.backtest import compare_strategies, walk_forward_backtest
from quant_pipeline.data import (
    build_database,
    load_config,
    load_price_matrix,
    run_sql_analysis,
)
from quant_pipeline.estimators import estimate
from quant_pipeline.metrics import daily_returns, summarize
from quant_pipeline.optimize import maximum_sharpe_weights, minimum_variance_weights
from quant_pipeline.report import generate_report
from quant_pipeline.sensitivity import ranking_stability, sweep


def main() -> None:
    config = load_config(ROOT / "config" / "research.yaml")
    prices_path = ROOT / "data" / "etf_prices.csv"
    if not prices_path.exists():
        raise FileNotFoundError("Frozen data is missing. Run scripts/fetch_market_data.py first.")

    database_path = ROOT / "data" / "research.sqlite"
    build_database(prices_path, database_path)
    sql_analysis = run_sql_analysis(database_path, ROOT / "sql" / "analytics.sql")
    sql_analysis.to_csv(ROOT / "outputs" / "sql_monthly_analysis.csv", index=False)

    prices = load_price_matrix(prices_path)
    returns = daily_returns(prices)

    engine = dict(
        lookback_months=config["lookback_months"],
        rebalance_frequency_months=config["rebalance_frequency_months"],
        maximum_weight=config["maximum_asset_weight"],
        transaction_cost_bps=config["transaction_cost_bps"],
        risk_free_rate=config["risk_free_rate"],
        periods=config["trading_days"],
        covariance_estimator=config.get("covariance_estimator", "ledoit_wolf"),
        mean_estimator=config.get("mean_estimator", "james_stein"),
    )
    backtest = walk_forward_backtest(returns, benchmark=config["benchmark"], **engine)

    # Every rule on identical rebalance dates, so the comparison is paired and a
    # difference between two of them comes from the rule rather than the schedule.
    comparison = compare_strategies(
        returns, benchmark=config["benchmark"],
        strategies=config.get("strategies", [
            "maximum_sharpe", "minimum_variance", "risk_parity",
            "inverse_volatility", "equal_weight"]),
        **engine,
    )
    comparison_rows = []
    for name, run in comparison.by_strategy.items():
        row = summarize(run.returns["Optimized portfolio"], config["risk_free_rate"],
                        config["trading_days"])
        row["strategy"] = run.diagnostics["label"]
        row["annual_turnover"] = run.diagnostics["annual_turnover"]
        row["optimizes"] = name not in ("equal_weight", "inverse_volatility", "static_allocation")
        comparison_rows.append(row)
    benchmark_row = summarize(comparison.returns[f"Benchmark ({config['benchmark']})"],
                              config["risk_free_rate"], config["trading_days"])
    benchmark_row.update(strategy=f"Benchmark ({config['benchmark']})",
                         annual_turnover=0.0, optimizes=False)
    comparison_rows.append(benchmark_row)
    strategy_comparison = pd.DataFrame(comparison_rows).set_index("strategy")
    strategy_comparison.to_csv(ROOT / "outputs" / "strategy_comparison.csv")

    # Does the conclusion survive the assumptions that produced it?
    grid = sweep(returns, config["benchmark"],
                 ["maximum_sharpe", "minimum_variance", "risk_parity", "equal_weight"], engine)
    grid.to_csv(ROOT / "outputs" / "sensitivity_grid.csv", index=False)
    ranking = ranking_stability(grid)
    ranking.to_csv(ROOT / "outputs" / "sensitivity_ranking.csv", index=False)

    latest_window = returns.loc[returns.index > returns.index.max() - pd.DateOffset(months=config["lookback_months"])]
    latest_inputs = estimate(latest_window, engine["covariance_estimator"],
                             engine["mean_estimator"], config["trading_days"])
    minimum_variance_weights(latest_window, config["maximum_asset_weight"],
                             config["trading_days"], covariance=latest_inputs["covariance"]).to_csv(
        ROOT / "outputs" / "minimum_variance_weights.csv", header=True
    )
    maximum_sharpe_weights(
        latest_window, config["maximum_asset_weight"], config["risk_free_rate"],
        config["trading_days"], covariance=latest_inputs["covariance"],
        expected_returns=latest_inputs["means"]
    ).to_csv(ROOT / "outputs" / "maximum_sharpe_weights.csv", header=True)

    backtest.returns.to_csv(ROOT / "outputs" / "backtest_returns.csv")
    backtest.weights.to_csv(ROOT / "outputs" / "rebalance_weights.csv")
    backtest.turnover.to_csv(ROOT / "outputs" / "turnover.csv")
    summaries = pd.DataFrame(
        {column: summarize(backtest.returns[column], config["risk_free_rate"], config["trading_days"]) for column in backtest.returns}
    ).T
    summaries.to_csv(ROOT / "outputs" / "performance_summary.csv")
    # Full-sample covariance, used for the correlation chart in the report.
    returns.cov().to_csv(ROOT / "outputs" / "daily_covariance.csv")

    # The covariance the published weights were actually solved from. Exporting
    # the full-sample matrix alongside window-fitted weights invited a
    # cross-check to compare two different problems, which is exactly what the
    # MATLAB script was doing before this was published.
    optimizer_inputs = estimate(
        latest_window, engine["covariance_estimator"], engine["mean_estimator"],
        config["trading_days"])
    (optimizer_inputs["covariance"] / config["trading_days"]).to_csv(
        ROOT / "outputs" / "optimizer_input_covariance.csv")
    optimizer_inputs["means"].rename("expected_return").to_csv(
        ROOT / "outputs" / "optimizer_input_expected_returns.csv", header=True)

    sensitivity_pivot = grid.pivot_table(
        index=["assumption", "value"], columns="strategy", values="sharpe_ratio").round(2)

    generate_report(
        ROOT / "outputs" / "research_report.html",
        config,
        backtest.returns,
        backtest.weights,
        returns.corr(),
        summaries,
        strategy_comparison=strategy_comparison,
        sensitivity=sensitivity_pivot,
        sensitivity_ranking=ranking,
    )
    print(summaries.round(4).to_string())
    print(f"\nGenerated report: {ROOT / 'outputs' / 'research_report.html'}")


if __name__ == "__main__":
    main()
