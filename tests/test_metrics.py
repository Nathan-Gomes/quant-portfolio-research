import numpy as np
import pandas as pd
import pytest

from quant_pipeline.metrics import (
    annualized_return,
    annualized_volatility,
    drawdown_series,
    historical_var,
    maximum_drawdown,
    sharpe_ratio,
)


def test_annualized_return_compounds_observations():
    returns = pd.Series([0.01] * 252)
    assert annualized_return(returns) == pytest.approx((1.01**252) - 1)


def test_annualized_volatility_scales_daily_standard_deviation():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01])
    expected = returns.std(ddof=1) * np.sqrt(252)
    assert annualized_volatility(returns) == pytest.approx(expected)


def test_sharpe_ratio_uses_annualized_return_and_volatility():
    returns = pd.Series([0.01, 0.00, 0.02, -0.01, 0.015])
    expected = (annualized_return(returns) - 0.03) / annualized_volatility(returns)
    assert sharpe_ratio(returns, risk_free_rate=0.03) == pytest.approx(expected)


def test_drawdown_starts_at_zero_and_captures_loss():
    returns = pd.Series([0.10, -0.20, 0.05])
    drawdown = drawdown_series(returns)
    assert drawdown.iloc[0] == 0.0
    assert maximum_drawdown(returns) == pytest.approx(-0.20)


def test_historical_var_reports_loss_at_confidence_level():
    returns = pd.Series([-0.08, -0.03, 0.01, 0.02, 0.04])
    assert historical_var(returns, confidence=0.75) == pytest.approx(0.03)
