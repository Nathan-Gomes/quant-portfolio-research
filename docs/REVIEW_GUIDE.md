# Review guide: Quantitative Portfolio Research Pipeline

## The question

Can a diversified, long-only ETF allocation improve risk-adjusted historical results versus SPY after costs, when every rebalance is based only on information available at that time?

## The answer this project can support

It is a reproducible historical research exercise, not a return forecast. It shows how to turn a portfolio idea into a constrained optimization, test it out of sample, include turnover costs, and make the calculation trail reviewable.

## A 10-minute walkthrough

1. Read the research question and assumptions in the [README](../README.md).
2. Inspect [config/research.yaml](../config/research.yaml) for the ETF universe, 30% cap, 36-month lookback, quarterly rebalances, costs, and risk-free rate.
3. Read [backtest.py](../src/quant_pipeline/backtest.py): it fits weights on the trailing window, then scores only later returns.
4. Read [optimize.py](../src/quant_pipeline/optimize.py): SciPy solves the fully invested, long-only, capped allocation problem.
5. Open `outputs/research_report.html` for the performance, drawdown, correlation, and rebalance evidence.
6. Review [test_backtest.py](../tests/test_backtest.py) and the other tests for the calculation and timing checks.

## Decision flow

```text
Frozen daily ETF prices
        -> trailing 36-month return window
        -> constrained maximum-Sharpe weights
        -> next quarter of realized returns
        -> turnover cost at rebalance
        -> repeat through history and compare with SPY
```

## Why the timing matters

At a rebalance date, the optimizer receives only the preceding 36 months of returns. The resulting weights are held from the following trading day until the next rebalance. A later market move therefore cannot influence the allocation that earned it. The tests also verify that changing later returns does not alter already-formed weights.

## What to discuss

- Does the ETF universe represent the intended mandate?
- Is the 30% position cap appropriate for diversification and concentration control?
- Are the lookback, quarterly schedule, and cost assumption realistic for the strategy?
- Do the report's drawdown and turnover results justify the observed return trade-off?

## Important limits

This is educational historical research using a small, selected ETF universe and Yahoo Finance adjusted-price data. It does not model taxes, bid-ask spreads, market impact, or future returns. A good historical result is evidence to investigate further, not a trading recommendation.
