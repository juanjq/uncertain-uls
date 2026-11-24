import sys
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

sys.path.insert(0, str(Path(__file__).parent))
import toy  # noqa: E402

from uuls import realisations as R  # noqa: E402

REF = toy.reference_simulate()
needs_ref = pytest.mark.skipif(REF is None, reason="reference repo gw-gamma-global-uls not found")


@pytest.fixture(scope="module")
def setup():
    reduced = toy.toy_reduced()
    return reduced, toy.toy_gwmap(reduced.geom)


class FixedInjector:
    """Injects strength s at coords[s] (test only: strengths are used as keys)."""

    def __init__(self, coords, index=2.0):
        self.coords, self.index = coords, index

    def __call__(self, rng, strength):
        return R.Injection(self.coords[strength], R._pwl(strength, self.index, 1.0))


def test_2d_sampler_follows_prob_in_mask(setup):
    _, gw = setup
    iy, ix = R.sample_sky_2d(gw, 20000, np.random.default_rng(0))
    assert gw.mask[iy, ix].all()
    counts = np.bincount(np.ravel_multi_index((iy, ix), gw.prob.shape), minlength=gw.prob.size)
    w = np.where(gw.mask, gw.prob, 0).ravel() / gw.prob[gw.mask].sum()
    sel = w * 20000 > 20
    chi2 = np.sum((counts[sel] - 20000 * w[sel]) ** 2 / (20000 * w[sel]))
    assert chi2 / sel.sum() < 1.5


@pytest.mark.parametrize("make", [lambda gw: R.PWLInjector2D(gw), lambda gw: R.PWLInjector3D(gw)])
def test_injection_jittered_uniformly_inside_its_bin(setup, make):
    """Injected positions fall inside the sampled bin, spread as U(-0.5, 0.5) pixel on each axis;
    with jitter=False they sit on the bin centre (the old behaviour)."""
    _, gw = setup
    inj = make(gw)
    image = gw.geom.to_image()
    off = []
    for i in range(400):
        a = inj(np.random.default_rng([3, i]), 1e-12)
        x, y = image.coord_to_pix((a.coord.ra.deg * u.deg, a.coord.dec.deg * u.deg))
        off.append((np.ravel(x)[0] - a.params["ix"], np.ravel(y)[0] - a.params["iy"]))
        assert a.params["ra"] == a.coord.ra.deg
    off = np.array(off)
    assert np.all(np.abs(off) < 0.5)
    np.testing.assert_allclose(off.mean(axis=0), 0, atol=0.05)
    np.testing.assert_allclose(off.std(axis=0), np.sqrt(1 / 12), rtol=0.1)
    inj.jitter = False
    a = inj(np.random.default_rng([3, 0]), 1e-12)
    centre = R._coord(gw, a.params["iy"], a.params["ix"])
    assert a.coord.separation(centre).deg < 1e-9


def test_injector_3d_amplitude_scales_as_d2(setup):
    _, gw = setup
    inj = R.PWLInjector3D(gw)
    a = inj(np.random.default_rng([5, 1]), 1e44)
    b = inj(np.random.default_rng([5, 1]), 4e44)
    assert a.params["distance"] == b.params["distance"]
    assert b.params["amplitude"] == pytest.approx(4 * a.params["amplitude"], rel=1e-12)


def test_signal_raises_lambda(setup):
    reduced, gw = setup
    out = R.simulate(reduced, gw, R.PWLInjector2D(gw), [0.0] * 6 + [3e-11] * 6, seed=10)
    assert out["lambda"][6:].min() > out["lambda"][:6].max()
    assert np.all(out["n_signal"][6:] > 10) and np.all(out["n_signal"][:6] == 0)


def test_result_independent_of_n_jobs_and_split(setup):
    reduced, gw = setup
    s = np.r_[np.zeros(3), np.full(3, 5e-12)]
    one = R.simulate(reduced, gw, R.PWLInjector2D(gw), s, seed=7)
    two = R.simulate(reduced, gw, R.PWLInjector2D(gw), s, seed=7, n_jobs=2)
    tail = R.simulate(reduced, gw, R.PWLInjector2D(gw), s[4:], seed=7, first_index=4)
    for key in ("lambda", "tsmax", "iy", "ix", "fake_seed"):
        np.testing.assert_array_equal(one[key], two[key])
        np.testing.assert_array_equal(one[key][4:], tail[key])


def test_run_is_cached(setup, tmp_path):
    reduced, gw = setup
    kw = dict(seed=3, cache_root=tmp_path, verbose=False)
    a = R.run(reduced, gw, R.PWLInjector2D(gw), [0.0, 1e-11], **kw)
    entries = list((tmp_path / "realisations").iterdir())
    b = R.run(reduced, gw, R.PWLInjector2D(gw), [0.0, 1e-11], **kw)
    assert len(entries) == 1 and len(list((tmp_path / "realisations").iterdir())) == 1
    np.testing.assert_array_equal(a["lambda"], b["lambda"])
    R.run(reduced, gw, R.PWLInjector2D(gw, index=2.5), [0.0, 1e-11], **kw)
    assert len(list((tmp_path / "realisations").iterdir())) == 2


def test_run_spectra_without_signal_equals_background_fake(setup):
    reduced, gw = setup
    coord = SkyCoord(150, 30, unit="deg")
    zero = R.RunSpectra(np.geomspace(0.3, 40, 21), np.zeros((len(reduced.runs), 21)))
    c1, _ = R.fake_from_runs(reduced.stacked, reduced.runs, coord, zero, seed=11)
    c2, _ = R.fake_stacked(reduced.stacked, coord, None, seed=11)
    np.testing.assert_array_equal(c1, c2)


def test_run_spectra_constant_source_matches_stacked(setup):
    """A constant PWL injected run by run gives the stacked expectation (flat toy IRFs)."""
    reduced, gw = setup
    coord = SkyCoord(150.1, 30.05, unit="deg")
    e = np.geomspace(0.3, 40, 41)
    values = np.tile(1e-11 * e ** -2.0, (len(reduced.runs), 1))
    _, n_runs = R.fake_from_runs(reduced.stacked, reduced.runs, coord, R.RunSpectra(e, values), 0)
    _, n_stacked = R.fake_stacked(reduced.stacked, coord, R._pwl(1e-11, 2.0, 1.0), 0)
    assert n_runs == pytest.approx(n_stacked, rel=0.01)


@needs_ref
def test_regression_samplers(setup):
    _, gw = setup
    ra, dec = R.bin_centres(gw.geom)
    new = R.sample_sky_and_distance(gw, 500, np.random.default_rng(4))
    old = REF.sample_sky_and_distance(gw.prob, gw.cdf_valid, gw.cdf, gw.r_grid, ra, dec, 500,
                                      np.random.default_rng(4))
    for k_new, k_old in (("iy", "iy"), ("ix", "ix"), ("distance", "d_sim")):
        np.testing.assert_array_equal(new[k_new], old[k_old])
    iy, ix = R.sample_sky_2d(gw, 300, np.random.default_rng(9))
    oy, ox = REF._sample_bin_indices(gw.prob, gw.mask, 300, np.random.default_rng(9))
    np.testing.assert_array_equal(iy, oy)
    np.testing.assert_array_equal(ix, ox)


@needs_ref
def test_regression_realisations_vs_reference(setup):
    """Same positions, amplitudes and seeds: Lambda and TS max equal the old loop's."""
    from gammapy.estimators import ExcessMapEstimator

    reduced, gw = setup
    ra, dec = R.bin_centres(gw.geom)
    rng = np.random.default_rng(1)
    iy, ix = R.sample_sky_2d(gw, 6, rng)
    amps = np.array([0.0, 1e-12, 3e-12, 6e-12, 1e-11, 2e-11])
    coords = SkyCoord(ra=ra[iy, ix] * u.deg, dec=dec[iy, ix] * u.deg, frame="icrs")
    engine = REF._TSEngine(reduced.stacked, ExcessMapEstimator(correlation_radius=0.1 * u.deg),
                           gw.prob, gw.mask, 1.0)
    old = REF._simulate_realisations(engine, coords, amps, range(6), 2.0, REF.E_REF, 100,
                                     "model-simulated", False, False, False)
    reduced.stacked.models = None
    new = R.simulate(reduced, gw, FixedInjector(dict(zip(amps, coords))), amps, seed=100)
    np.testing.assert_array_equal(new["lambda"], old["lambda_data"])
    np.testing.assert_array_equal(new["tsmax"], old["tsmax"])
    np.testing.assert_array_equal(ra[new["lambda_iy"], new["lambda_ix"]], old["lambda_ra"])


@needs_ref
def test_regression_fake_from_runs(setup):
    reduced, gw = setup
    coord = SkyCoord(149.95, 30.02, unit="deg")
    e = np.geomspace(0.3, 40, 31)
    values = np.vstack([2e-11 * e ** -2.0, 5e-12 * e ** -2.5])
    c_new, n_new = R.fake_from_runs(reduced.stacked, reduced.runs, coord, R.RunSpectra(e, values), 21)
    n_old = REF._fake_from_runs(reduced.stacked, reduced.runs, coord, e, values, 21, "model-simulated")
    np.testing.assert_array_equal(c_new, reduced.stacked.counts.data)
    assert n_new == n_old


# --- region of interest: GW mask & observed region -----------------------------------------------

def _half_exposed(cut=24):
    """Toy data with no exposure, background or safe mask for ix < cut (east), where the GW blob peaks."""
    reduced = toy.toy_reduced()
    for ds in [reduced.stacked, *reduced.runs]:
        ds.exposure.data[..., :cut] = 0
        ds.background.data[..., :cut] = 0
        ds.mask_safe.data[..., :cut] = False
    return reduced, toy.toy_gwmap(reduced.geom)


def test_unexposed_bin_no_longer_sets_a_floor():
    reduced, gw = _half_exposed()
    from uuls import lambda_stat

    roi = lambda_stat.roi(gw.mask, reduced.stacked)
    dark = gw.mask & ~roi
    assert dark.any() and roi.any() and gw.prob[dark].max() > gw.prob[roi].max()
    floor = 2 * np.log(gw.prob[dark].max())
    old = R.simulate(reduced, gw, R.PWLInjector2D(gw), np.zeros(30), seed=2, roi="gw_mask")
    new = R.simulate(reduced, gw, R.PWLInjector2D(gw, reduced=reduced), np.zeros(30), seed=2)
    assert old["lambda"].min() >= floor - 1e-9 and np.any(np.isclose(old["lambda"], floor, rtol=0, atol=1e-9))
    assert new["lambda"].min() < floor
    assert np.all(roi[new["lambda_iy"], new["lambda_ix"]])
    assert np.all(new["lambda"] <= old["lambda"] + 1e-12)      # same counts, smaller region


def test_observed_and_mc_use_the_same_roi():
    reduced, gw = _half_exposed()
    from uuls import lambda_stat

    roi = lambda_stat.roi(gw.mask, reduced.stacked)
    pre = R.precompute(reduced, gw)
    np.testing.assert_array_equal(pre.mask, roi)
    inj = R.PWLInjector2D(gw, reduced=reduced)
    np.testing.assert_array_equal(inj.support, roi)
    draws = [inj(np.random.default_rng([1, i]), 1e-12).params for i in range(100)]
    assert all(roi[d["iy"], d["ix"]] for d in draws)
    reduced.stacked.counts.data = R.fake_stacked(reduced.stacked, None, None, 5)[0]
    obs = R.observed_lambda(reduced, gw)
    assert obs == pre.evaluate(reduced.stacked.counts.data) and roi[obs["lambda_iy"], obs["lambda_ix"]]
    with pytest.warns(UserWarning, match="outside the ROI"):
        R.simulate(reduced, gw, R.PWLInjector2D(gw), [0.0], seed=1)


def test_roi_default_equals_gw_mask_on_fully_exposed_data(setup):
    reduced, gw = setup
    inj = R.PWLInjector2D(gw, reduced=reduced)
    a = R.simulate(reduced, gw, inj, [0.0, 3e-12], seed=8)
    b = R.simulate(reduced, gw, R.PWLInjector2D(gw), [0.0, 3e-12], seed=8, roi="gw_mask")
    for key in ("lambda", "tsmax", "lambda_iy", "lambda_ix", "iy", "ix"):
        np.testing.assert_array_equal(a[key], b[key])
    assert inj.describe()["roi"] == "observed" and R.PWLInjector2D(gw).describe()["roi"] == "gw_mask"
