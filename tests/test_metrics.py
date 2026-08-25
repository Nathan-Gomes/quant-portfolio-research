import numpy as np
import pandas as pd

from quant_pipeline.metrics import annualized_return, drawdown_series, maximum_drawdown


def test_annualized_return_compounds_observations():
    returns = pd.Series([0.01] * 252)
    assert annualized_return(returns) == pytest.approx((1.01**252) - 1)


def test_drawdown_starts_at_zero_and_captures_loss():
    returns = pd.Series([0.10, -0.20, 0.05])
    drawdown = drawdown_series(returns)
    assert drawdown.iloc[0] == 0.0
    assert maximum_drawdown(returns) == pytest.approx(-0.20)


import pytest
