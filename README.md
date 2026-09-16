# Quantitative Portfolio Research Pipeline

![Python](https://img.shields.io/badge/Python-pandas%20%2F%20NumPy%20%2F%20SciPy-0d1117?style=for-the-badge&logo=python&logoColor=58a6ff)
![SQL](https://img.shields.io/badge/SQL-analytics%20queries-0d1117?style=for-the-badge&logo=sqlite&logoColor=7ee787)
![MATLAB](https://img.shields.io/badge/MATLAB-optimization%20comparison-0d1117?style=for-the-badge)
![Tests](https://img.shields.io/badge/tests-pytest-0d1117?style=for-the-badge&logo=pytest&logoColor=ffffff)

A reproducible research-to-production workflow for constrained ETF portfolio optimization. The project combines Python, pandas, SQL, SciPy, MATLAB, automated testing, and a browser-based research report.

[Live report](https://www.nathan-gomes.com/quant-portfolio-report.html) | [Portfolio page](https://www.nathan-gomes.com/Project-Quant-Portfolio.dc.html)

## Review with Matt

Start with the short [review guide](docs/REVIEW_GUIDE.md). It explains the research question, the walk-forward decision flow, where the important calculations live, and the limits of the result without requiring a line-by-line code review.

## Research question

Can a long-only, diversified ETF portfolio with a 30% asset cap improve risk-adjusted performance relative to an SPY benchmark when weights are estimated only from prior data and transaction costs are included?

## What this solves

Investment research can look convincing when it is only a spreadsheet or one-off notebook. This project turns the research process into a repeatable pipeline where assumptions, data, calculations, constraints, costs, and validation checks are visible.

The goal is not to prove that one portfolio always wins. The goal is to show how a quantitative research idea can be implemented, tested, backtested, documented, and reviewed before anyone would trust it.

## Pipeline

1. Download and freeze adjusted daily ETF prices from Yahoo Finance's chart endpoint.
2. Normalize the observations and load them into SQLite.
3. Use SQL CTEs, joins, and window functions for monthly return analysis and ranking.
4. Calculate returns, covariance, correlation, volatility, Sharpe ratio, drawdown, and historical VaR.
5. Solve minimum-variance and maximum-Sharpe allocations with long-only weight constraints.
6. Run a quarterly walk-forward backtest with a 36-month lookback and turnover costs.
7. Validate the calculations and constraints with pytest.
8. Generate an interactive HTML research report.

## Key outputs

- Latest optimized portfolio weights
- Portfolio versus benchmark performance
- Annualized return, volatility, Sharpe ratio, maximum drawdown, and VaR
- Correlation heatmap across assets
- Drawdown chart for the optimized strategy and benchmark
- Rebalance history and turnover cost impact
- Python versus MATLAB minimum-variance optimization comparison

## Why it is relevant

This mirrors the work of a quantitative developer supporting a research team:

- Convert market data into clean, queryable research tables
- Implement financial calculations in Python and pandas
- Use SQL for analysis and data checks
- Solve constrained optimization problems
- Avoid look-ahead bias through walk-forward testing
- Package the results into a report that non-developers can review
- Add tests so calculations can be trusted and rerun

## Run

```bash
python3 scripts/fetch_market_data.py
python3 run_pipeline.py
pytest
```

Open `outputs/research_report.html` in a browser.

The generated report is also published on the portfolio site as the live project artifact.

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
