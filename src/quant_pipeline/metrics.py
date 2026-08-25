from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change(fill_method=None).dropna(how="all")


def annualized_return(returns: pd.Series, periods: int = 252) -> float:
    if returns.empty:
        return float("nan")
    return float((1.0 + returns).prod() ** (periods / len(returns)) - 1.0)


def annualized_volatility(returns: pd.Series, periods: int = 252) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float, periods: int = 252) -> float:
    volatility = annualized_volatility(returns, periods)
    return (annualized_return(returns, periods) - risk_free_rate) / volatility if volatility else float("nan")


def drawdown_series(returns: pd.Series) -> pd.Series:
    growth = (1.0 + returns.fillna(0.0)).cumprod()
    return growth / growth.cummax() - 1.0


def maximum_drawdown(returns: pd.Series) -> float:
    return float(drawdown_series(returns).min())


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    return float(-returns.quantile(1.0 - confidence))


def summarize(returns: pd.Series, risk_free_rate: float, periods: int = 252) -> dict[str, float]:
    return {
        "annual_return": annualized_return(returns, periods),
        "annual_volatility": annualized_volatility(returns, periods),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, periods),
        "maximum_drawdown": maximum_drawdown(returns),
        "daily_var_95": historical_var(returns),
    }
