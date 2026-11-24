import sys
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import toy  # noqa: E402

from uuls import lambda_stat as ls  # noqa: E402

REF = toy.reference_simulate()
needs_ref = pytest.mark.skipif(REF is None, reason="reference repo gw-gamma-global-uls not found")


@pytest.fixture(scope="module")
def toy_data():
    reduced = toy.toy_reduced()
    gw = toy.toy_gwmap(reduced.geom)
    ds = reduced.stacked
    ds.fake(3)                                   # background-only counts
    counts = ds.counts.data.copy()
    # a bright spot, so TS has both signs
    counts[:, 18:21, 22:25] += np.random.RandomState(1).poisson(3.0, (counts.shape[0], 3, 3))
    return reduced, gw, counts


def test_signed_ts_simple_values():
    n = np.array([0.0, 2.0, 10.0])
    mu = np.array([2.0, 2.0, 2.0])
    ts = ls.signed_ts(n, mu)
    assert ts[1] == pytest.approx(0.0, abs=1e-12)
    assert ts[0] == pytest.approx(-4.0)                       # 2 mu at n = 0
    assert ts[2] == pytest.approx(2 * (10 * np.log(5) - 8))


def test_fast_counts_equal_per_plane(toy_data):
    reduced, _, counts = toy_data
    k = ls.tophat_kernel(2.0)
    np.testing.assert_array_equal(ls.correlated_counts(counts, k), ls.correlated_counts_fast(counts, k))


def test_global_lambda_is_max_in_mask():
    ts = np.zeros((5, 5)); ts[0, 0] = 100.0; ts[2, 2] = 3.0
    prob = np.full((5, 5), 0.04)
    mask = np.zeros((5, 5), bool); mask[1:4, 1:4] = True
    assert ls.global_lambda(ts, prob, mask) == pytest.approx(3.0 + 2 * np.log(0.04))


@needs_ref
def test_regression_ts_and_lambda(toy_data):
    """Same counts: the new TS map and Lambda equal _TSEngine's exactly."""
    from gammapy.estimators import ExcessMapEstimator
    from gammapy.maps import Map

    reduced, gw, counts = toy_data
    ds = reduced.stacked
    ds.counts = Map.from_geom(ds.counts.geom, data=counts)
    est = ExcessMapEstimator(correlation_radius=0.1 * u.deg, correlate_off=True)
    engine = REF._TSEngine(ds, est, gw.prob, gw.mask, 1.0)
    ts_ref, ts2_ref = engine.ts_maps(engine.counts_statistics())
    ref = engine.lambda_from(ts_ref, ts2_ref)

    r_bins = ls.correlation_radius_bins(ds.counts.geom, 0.1)
    ts_new = ls.ts_map(counts, ls.background_cube(ds), r_bins, mask=ls.safe_mask_cube(ds))
    np.testing.assert_array_equal(ts_new, ts_ref)
    assert ls.global_lambda(ts_new, gw.prob, gw.mask) == ref["lambda_data"]

    pre = ls.TSPrecomputed.build(ls.background_cube(ds), gw.prob, gw.mask, r_bins,
                                 safe_mask=ls.safe_mask_cube(ds))
    np.testing.assert_array_equal(pre.ts(counts), ts_ref)
    out = pre.evaluate(counts)
    assert out["lambda"] == ref["lambda_data"] and out["tsmax"] == ref["tsmax"]
    ra, dec = engine.bin_c_ra, engine.bin_c_dec
    assert (ra[out["lambda_iy"], out["lambda_ix"]], dec[out["lambda_iy"], out["lambda_ix"]]) == \
        (ref["lambda_ra"], ref["lambda_dec"])
