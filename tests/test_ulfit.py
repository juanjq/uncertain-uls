import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).parent))
import toy  # noqa: E402

from uuls import ulfit  # noqa: E402

REF = toy.reference_simulate()
needs_ref = pytest.mark.skipif(REF is None, reason="reference repo gw-gamma-global-uls not found")
CL = 0.95


def make_curve(f0=0.02, c=1.0, t0=-11.5, s=0.15):
    """A probit detection curve in x and its exact CL crossing."""
    def curve(x):
        return f0 + (c - f0) * ndtr((np.log10(x) - t0) / s)
    x_true = 10 ** brentq(lambda t: curve(10 ** t) - CL, t0 - 2, t0 + 2)
    return curve, x_true


def bernoulli_batch(curve, offset=777):
    """simulate_batch: Lambda = +1 (above target 0) with probability curve(x), seeded by index."""
    def batch(x, r, first):
        rng = np.random.default_rng(first + offset)
        return np.where(rng.random(len(x)) < curve(x), 1.0, -1.0)
    return batch


def background(f0, n=2000, seed=3):
    return np.where(np.random.default_rng(seed).random(n) < f0, 1.0, -1.0)


def test_target_choice():
    bkg = np.linspace(-10, 0, 101)
    assert ulfit.target_lambda(-1.0, bkg)[0] == -1.0          # excess: observed
    t, p, s = ulfit.target_lambda(-8.0, bkg)                   # deficit: background median
    assert t == pytest.approx(-5.0) and s < 0 and p > 0.5


def test_no_limit_when_background_exceeds_cl():
    with pytest.raises(ValueError):
        ulfit.background_counts(np.ones(100), 0.0, CL)


def test_probability_equals_cl_at_limit():
    theta = np.array([-11.0, np.log(0.2), 0.3, 1.0])
    t_ul, s, f0, c = ulfit.probit_params(theta, CL)
    assert ulfit.detection_probability(t_ul, t_ul, s, f0, c, CL) == pytest.approx(CL, abs=1e-14)


@pytest.mark.parametrize("seed", [11, 12])
def test_adaptive_fit_recovers_known_probit(seed):
    curve, x_true = make_curve()
    res = ulfit.adaptive_fit(bernoulli_batch(curve, offset=seed * 100000), 1e-13, 1e-10, None,
                             background(0.02, seed=seed), target=0.0, n_scout=400, n_refine=800,
                             precision=0.02, max_sims=3000, design_seed=seed, verbose=False)
    pull = (res.log10_x_ul - np.log10(x_true)) / res.log10_x_ul_err
    assert abs(pull) < 3
    assert res.x_ul_lo_2sigma < res.x_ul_lo_1sigma < res.x_ul < res.x_ul_hi_1sigma < res.x_ul_hi_2sigma
    assert res.f0 == pytest.approx(0.02, abs=0.01) and res.ceiling > 0.98
    rng = np.random.default_rng(seed + 50)
    closure = np.where(rng.random(2000) < curve(res.x_ul), 1.0, -1.0)
    assert abs(ulfit.closure_pull(res, closure)["pull"]) < 3


def test_fit_measures_a_ceiling_below_one():
    curve, x_true = make_curve(c=0.99)
    rng = np.random.default_rng(1)
    x = 10 ** rng.uniform(-12.5, -10.5, 6000)
    lam = np.where(rng.random(x.size) < curve(x), 1.0, -1.0)
    res = ulfit.fit(x, lam, None, background(0.02), target=0.0)
    assert res.ceiling == pytest.approx(0.99, abs=0.005)
    assert abs((res.log10_x_ul - np.log10(x_true)) / res.log10_x_ul_err) < 3


@needs_ref
def test_regression_adaptive_fit():
    curve, _ = make_curve()
    batch, bkg = bernoulli_batch(curve), background(0.02)
    kw = dict(n_scout=300, n_refine=600, precision=0.02, max_sims=2000, max_rounds=4, design_seed=5)
    old = REF._fit_ul(batch, 1e-13, 1e-10, CL, 0.0, bkg, verbose=False, **kw)
    new = ulfit.adaptive_fit(batch, 1e-13, 1e-10, None, bkg, target=0.0, verbose=False, **kw)
    for key in ("x_ul", "x_ul_lo_1sigma", "x_ul_hi_1sigma", "x_ul_lo_2sigma", "x_ul_hi_2sigma",
                "f0", "ceiling", "width_dex", "log10_x_ul_err"):
        assert getattr(new, key) == pytest.approx(old[key], rel=1e-6), key
    assert new.n_sim_total == old["n_sim_total"] and new.converged == old["converged"]
    np.testing.assert_allclose(new.params, old["params"], rtol=1e-6)
    np.testing.assert_allclose(new.samples["x"], old["samples"]["x"], rtol=1e-12)


@needs_ref
def test_regression_single_fit():
    """fit() on the scout sample equals the reference with one round."""
    curve, _ = make_curve(c=0.995)
    batch, bkg = bernoulli_batch(curve, offset=5), background(0.02)
    old = REF._fit_ul(batch, 1e-13, 1e-10, CL, 0.0, bkg, n_scout=800, max_rounds=1,
                      design_seed=2, verbose=False)
    new = ulfit.fit(old["samples"]["x"], old["samples"]["lambda"], None, bkg, bracket=(1e-13, 1e-10),
                    target=0.0)
    for key in ("x_ul", "x_ul_lo_1sigma", "x_ul_hi_1sigma", "f0", "ceiling", "width_dex"):
        assert getattr(new, key) == pytest.approx(old[key], rel=1e-6), key
    np.testing.assert_allclose(new.cov, old["cov"], rtol=1e-6, atol=1e-12)
    assert new.gof["chi2"] == pytest.approx(old["gof"]["chi2"], rel=1e-6)


@needs_ref
def test_regression_closure_pull():
    curve, _ = make_curve()
    batch, bkg = bernoulli_batch(curve), background(0.02)
    old = REF._fit_ul(batch, 1e-13, 1e-10, CL, 0.0, bkg, n_scout=400, max_rounds=2, design_seed=1,
                      verbose=False)
    new = ulfit.adaptive_fit(batch, 1e-13, 1e-10, None, bkg, target=0.0, n_scout=400, max_rounds=2,
                             design_seed=1, verbose=False)
    lam = np.where(np.random.default_rng(8).random(1000) < 0.94, 1.0, -1.0)
    a, b = REF.closure_pull(old, lam), ulfit.closure_pull(new, lam)
    assert a["pull"] == pytest.approx(b["pull"], rel=1e-9)
