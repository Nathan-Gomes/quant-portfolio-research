# Quantitative Portfolio Research Pipeline

A reproducible research-to-production workflow for constrained ETF portfolio optimization. The project combines Python, pandas, SQL, SciPy, MATLAB, automated testing, and a browser-based research report.

## Research question

Can a long-only, diversified ETF portfolio with a 30% asset cap improve risk-adjusted performance relative to an SPY benchmark when weights are estimated only from prior data and transaction costs are included?

## Pipeline

1. Download and freeze adjusted daily ETF prices from Yahoo Finance's chart endpoint.
2. Normalize the observations and load them into SQLite.
3. Use SQL CTEs, joins, and window functions for monthly return analysis and ranking.
4. Calculate returns, covariance, correlation, volatility, Sharpe ratio, drawdown, and historical VaR.
5. Solve minimum-variance and maximum-Sharpe allocations with long-only weight constraints.
6. Run a quarterly walk-forward backtest with a 36-month lookback and turnover costs.
7. Validate the calculations and constraints with pytest.
8. Generate an interactive HTML research report.

## Run

```bash
python3 scripts/fetch_market_data.py
python3 run_pipeline.py
pytest
```

Open `outputs/research_report.html` in a browser.

## Configuration

All research assumptions live in `config/research.yaml`. Changing the lookback, rebalance frequency, weight cap, transaction cost, or risk-free rate reruns the complete experiment.

## MATLAB

`matlab/minimum_variance_portfolio.m` reads the covariance matrix produced by Python and solves the same constrained minimum-variance problem with `quadprog`. MATLAB is not available in this development environment, so the implementation is included but its runtime output must be generated on a MATLAB-equipped machine.

## Limitations

- Historical adjusted prices are not a forecast of future returns.
- The ETF universe is intentionally small and may create selection bias.
- The backtest models transaction costs but not taxes, bid-ask variation, or market impact.
- Yahoo Finance data is used for educational research and should not be treated as an institutional market-data source.
- Results are educational and are not investment advice.
