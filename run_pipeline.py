from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from quant_pipeline.backtest import walk_forward_backtest
from quant_pipeline.data import build_database, load_config, load_price_matrix, run_sql_analysis
from quant_pipeline.metrics import daily_returns, summarize
from quant_pipeline.optimize import maximum_sharpe_weights, minimum_variance_weights
from quant_pipeline.report import generate_report


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
    backtest = walk_forward_backtest(
        returns,
        benchmark=config["benchmark"],
        lookback_months=config["lookback_months"],
        rebalance_frequency_months=config["rebalance_frequency_months"],
        maximum_weight=config["maximum_asset_weight"],
        transaction_cost_bps=config["transaction_cost_bps"],
        risk_free_rate=config["risk_free_rate"],
        periods=config["trading_days"],
    )

    latest_window = returns.loc[returns.index > returns.index.max() - pd.DateOffset(months=config["lookback_months"])]
    minimum_variance_weights(latest_window, config["maximum_asset_weight"], config["trading_days"]).to_csv(
        ROOT / "outputs" / "minimum_variance_weights.csv", header=True
    )
    maximum_sharpe_weights(
        latest_window, config["maximum_asset_weight"], config["risk_free_rate"], config["trading_days"]
    ).to_csv(ROOT / "outputs" / "maximum_sharpe_weights.csv", header=True)

    backtest.returns.to_csv(ROOT / "outputs" / "backtest_returns.csv")
    backtest.weights.to_csv(ROOT / "outputs" / "rebalance_weights.csv")
    backtest.turnover.to_csv(ROOT / "outputs" / "turnover.csv")
    summaries = pd.DataFrame(
        {column: summarize(backtest.returns[column], config["risk_free_rate"], config["trading_days"]) for column in backtest.returns}
    ).T
    summaries.to_csv(ROOT / "outputs" / "performance_summary.csv")
    returns.cov().to_csv(ROOT / "outputs" / "daily_covariance.csv")

    generate_report(
        ROOT / "outputs" / "research_report.html",
        config,
        backtest.returns,
        backtest.weights,
        returns.corr(),
        summaries,
    )
    print(summaries.round(4).to_string())
    print(f"\nGenerated report: {ROOT / 'outputs' / 'research_report.html'}")


if __name__ == "__main__":
    main()
