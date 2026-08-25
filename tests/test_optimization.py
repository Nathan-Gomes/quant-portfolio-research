import numpy as np
import pandas as pd
import pytest

from quant_pipeline.optimize import maximum_sharpe_weights, minimum_variance_weights


@pytest.fixture
def sample_returns():
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.normal(0.0004, 0.01, size=(600, 4)), columns=list("ABCD"))


@pytest.mark.parametrize("optimizer", [minimum_variance_weights])
def test_minimum_variance_constraints(sample_returns, optimizer):
    weights = optimizer(sample_returns, maximum_weight=0.35)
    assert weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert (weights >= -1e-10).all()
    assert (weights <= 0.35 + 1e-8).all()


def test_maximum_sharpe_constraints(sample_returns):
    weights = maximum_sharpe_weights(sample_returns, maximum_weight=0.35, risk_free_rate=0.03)
    assert weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert (weights >= -1e-10).all()
    assert (weights <= 0.35 + 1e-8).all()


def test_rejects_infeasible_maximum_weight(sample_returns):
    with pytest.raises(ValueError):
        minimum_variance_weights(sample_returns, maximum_weight=0.20)
