"""Check the production solver against an independent convex reference.

The pipeline solves with SLSQP, which is a local method: it returns a point
where it could not improve, not a proof that no better point exists. At six
assets that is almost certainly the global optimum, but "almost certainly" is
not a property a reviewer can check.

So these tests re-solve the same problems as disciplined convex programs with
cvxpy, which returns a certified global optimum, and assert the two agree.
cvxpy is a test dependency only — the pipeline does not need it at runtime, and
these tests skip when it is absent rather than failing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

cvxpy = pytest.importorskip("cvxpy", reason="convex reference solver is not installed")

from quant_pipeline.estimators import estimate
from quant_pipeline.optimize import (
    maximum_sharpe_weights,
    minimum_variance_weights,
)


def _greedy_fill(cap: float, assets: int) -> np.ndarray:
    """The weights of the best capped portfolio: cap each in turn until fully invested."""
    filled, remaining = [], 1.0
    while remaining > 1e-12 and len(filled) < assets:
        take = min(cap, remaining)
        filled.append(take)
        remaining -= take
    return np.array(filled)


@pytest.fixture(params=[(6, 0.30), (6, 1.0), (10, 0.20)])
def problem(request):
    assets, cap = request.param
    generator = np.random.default_rng(assets * 31 + int(cap * 100))
    factor = generator.normal(0, 0.009, (700, 3))
    loadings = generator.normal(0, 1, (assets, 3))
    scale = generator.uniform(0.003, 0.016, assets)
    values = factor @ loadings.T + generator.normal(0, 1, (700, assets)) * scale
    window = pd.DataFrame(values, columns=[f"A{i}" for i in range(assets)])
    return window, cap


def test_minimum_variance_matches_a_certified_optimum(problem):
    window, cap = problem
    inputs = estimate(window)
    covariance = inputs["covariance"].to_numpy()

    weights = cvxpy.Variable(len(covariance))
    reference = cvxpy.Problem(
        cvxpy.Minimize(cvxpy.quad_form(weights, cvxpy.psd_wrap(covariance))),
        [cvxpy.sum(weights) == 1, weights >= 0, weights <= cap])
    reference.solve()
    assert reference.status == "optimal"

    produced = minimum_variance_weights(window, cap, covariance=inputs["covariance"]).to_numpy()
    certified = np.asarray(weights.value).ravel()
    # The variance reached matters more than the exact vector, since a flat
    # optimum can be reached by slightly different weights.
    assert produced @ covariance @ produced == pytest.approx(
        certified @ covariance @ certified, rel=1e-6)
    assert np.abs(produced - certified).max() < 5e-3


def test_maximum_sharpe_matches_a_certified_optimum(problem):
    """The ratio is not concave, so the reference uses the Schaible transform.

    Optimizing an unnormalized vector with the excess return pinned to one turns
    the ratio into a quadratic minimization, which is convex. Rescaling the
    solution recovers the tangency portfolio, and that is valid here because
    every constraint is homogeneous in the unnormalized variable.
    """
    window, cap = problem
    inputs = estimate(window)
    covariance = inputs["covariance"].to_numpy()
    excess = inputs["means"].to_numpy() - 0.03
    # A tangency portfolio exists only if some *feasible* portfolio clears the
    # risk-free rate. Under a cap that is a stronger condition than one asset
    # doing so: the best capped mix fills the cap from the best assets down.
    best_capped = float(np.sort(excess)[::-1][:int(np.ceil(1 / cap))] @ _greedy_fill(cap, len(excess)))
    if best_capped <= 0:
        pytest.skip("no capped portfolio has an expected return above the risk-free rate here")

    y = cvxpy.Variable(len(covariance))
    kappa = cvxpy.Variable(nonneg=True)
    reference = cvxpy.Problem(
        cvxpy.Minimize(cvxpy.quad_form(y, cvxpy.psd_wrap(covariance))),
        [excess @ y == 1, cvxpy.sum(y) == kappa, y >= 0, y <= kappa * cap])
    reference.solve()
    assert reference.status == "optimal"
    certified = np.asarray(y.value).ravel()
    certified = certified / certified.sum()

    produced = maximum_sharpe_weights(
        window, cap, 0.03, covariance=inputs["covariance"],
        expected_returns=inputs["means"]).to_numpy()

    sharpe = lambda w: (w @ inputs["means"].to_numpy() - 0.03) / np.sqrt(w @ covariance @ w)
    assert sharpe(produced) == pytest.approx(sharpe(certified), rel=1e-4)
    assert np.abs(produced - certified).max() < 1e-2


def test_the_cap_is_what_makes_the_problems_differ(problem):
    """A tighter cap can only cost variance; if it did not, something is wrong."""
    window, _ = problem
    inputs = estimate(window)
    covariance = inputs["covariance"].to_numpy()
    loose = minimum_variance_weights(window, 1.0, covariance=inputs["covariance"]).to_numpy()
    tight = minimum_variance_weights(window, 0.2, covariance=inputs["covariance"]).to_numpy()
    assert tight @ covariance @ tight >= loose @ covariance @ loose - 1e-12
