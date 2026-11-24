"""uuls.reduce on a short simulated DL3: two 5-minute wobble runs at +-0.4 deg, background only and with a
bright source at the target. The simulations and reductions are cached in the project cache."""
import time

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time

from uuls import paths, reduce, simdata

TARGET = SkyCoord(7.53, 45.81, unit="deg")
T0 = Time("2025-10-20T22:30:00", scale="utc")       # stands for the merger
T_START = [Time("2025-10-20T23:00:00", scale="utc"), Time("2025-10-20T23:06:00", scale="utc")]
SOURCE = simdata.PowerLawSource(ra=7.53, dec=45.81, amplitude=5e-11, index=2.0)
POINTINGS = Table({"obs_id": [1, 2], "ra": [7.53, 7.53], "dec": [45.81 + 0.4, 45.81 - 0.4],
                   "t_start": Time([t.isot for t in T_START], scale="utc"), "duration": [300.0, 300.0]})


def _have_simulst():
    try:
        simdata.simulst_python()
    except FileNotFoundError:
        return False
    return paths.SIMULST_IRF_ROOT.is_dir() and paths.SIMULST_LIBRARY_ROOT.is_dir()


pytestmark = [pytest.mark.simulst, pytest.mark.skipif(not _have_simulst(), reason="simulst env missing")]


@pytest.fixture(scope="module")
def geom():
    return reduce.analysis_geom(TARGET, width=4.0)


@pytest.fixture(scope="module")
def dl3_bkg():
    return simdata.simulate(POINTINGS, nsb=0.14, seed=5)


@pytest.fixture(scope="module")
def dl3_src():
    return simdata.simulate(POINTINGS, nsb=0.14, seed=5, source=SOURCE)


def _in_mask(ds, name):
    m = ds.mask_safe.data
    return getattr(ds, name).data[m].sum() if name == "counts" else ds.npred_background().data[m].sum()


def _excess_near(ds, radius=0.15):
    sep = ds.counts.geom.to_image().get_coord().skycoord.separation(TARGET).deg
    sel = (sep < radius)[None] & ds.mask_safe.data
    n, b = ds.counts.data[sel].sum(), ds.npred_background().data[sel].sum()
    return n - b, np.sqrt(max(b, 1.0))


def test_analysis_geom_defaults(geom):
    assert geom.frame == "icrs" and geom.projection == "AIR"
    assert np.allclose(geom.pixel_scales.deg, 0.08)
    e = geom.axes["energy"].edges.to_value(u.TeV)
    assert np.isclose(e[0], 0.6) and np.isclose(e[-1], 20) and len(e) - 1 == 7
    assert geom.data_shape == (7, 50, 50)


@pytest.mark.parametrize("method", ["ring", "model", "fov"])
def test_background_only_counts_match_background(dl3_bkg, geom, method):
    r = reduce.from_dl3(dl3_bkg, geom, T0, {"background": method})
    n, b = _in_mask(r.stacked, "counts"), _in_mask(r.stacked, "background")
    assert n > 200
    assert abs(n - b) < 3 * np.sqrt(b) + 0.05 * b, (method, n, b)
    excess, err = _excess_near(r.stacked)
    assert abs(excess) < 4 * err
    assert r.settings["background"] == method


def test_source_gives_an_excess_at_its_position(dl3_src, geom):
    r = reduce.from_dl3(dl3_src, geom, T0)
    excess, err = _excess_near(r.stacked)
    assert excess > 10 * err, (excess, err)
    # far from the source the counts still follow the ring background
    sep = geom.to_image().get_coord().skycoord.separation(TARGET).deg
    far = (sep > 1.0)[None] & r.stacked.mask_safe.data
    n, b = r.stacked.counts.data[far].sum(), r.stacked.npred_background().data[far].sum()
    assert abs(n - b) < 3 * np.sqrt(b) + 0.05 * b


def test_exposure_only_inside_offset_max(dl3_bkg, geom):
    offset_max = 1.0  # well inside the IRF reach, so the cut, not the IRFs, ends the exposure
    r = reduce.from_dl3(dl3_bkg, geom, T0, {"offset_max_deg": offset_max})
    coords = geom.to_image().get_coord().skycoord
    pnts = SkyCoord(POINTINGS["ra"], POINTINGS["dec"], unit="deg")
    offset = np.min([coords.separation(p).deg for p in pnts], axis=0)
    expo = r.stacked.exposure.data.sum(axis=0)
    half_diag = 0.08 / np.sqrt(2)
    assert np.all(expo[offset > offset_max + half_diag] == 0)
    assert np.all(expo[offset < offset_max - 0.2] > 0)
    for ds, p in zip(r.runs, pnts):
        own = coords.separation(p).deg
        assert np.all(ds.exposure.data.sum(axis=0)[own > offset_max + half_diag] == 0)


def test_runs_and_gti_seconds(dl3_bkg, geom):
    r = reduce.from_dl3(dl3_bkg, geom, T0)
    assert [d.name for d in r.runs] == ["run1", "run2"] and r.obs_ids == [1, 2]
    for gti, t in zip(r.run_gti_s, T_START):
        start = (t - T0).to_value(u.s)
        np.testing.assert_allclose(gti, [[start, start + 300.0]], atol=1e-3)
    n_runs = sum(_in_mask(d, "counts") for d in r.runs)
    assert n_runs == _in_mask(r.stacked, "counts")
    assert r.t0 == T0 and r.stacked.counts.geom == geom


def test_source_at_2_deg_offset_and_exposure_to_2p4_deg(geom):
    """Production 2025_2p5: a source 2.0 deg from the pointing is simulated (with "2025" it is skipped, beyond
    its 1.1 deg), and the exposure reaches out to the 2.5 deg offset-max cut."""
    pnt = SkyCoord(TARGET.ra, TARGET.dec + 2.0 * u.deg)
    one = Table({"obs_id": [1], "ra": [pnt.ra.deg], "dec": [pnt.dec.deg],
                 "t_start": Time([T_START[0].isot], scale="utc"), "duration": [300.0]})
    dl3 = simdata.simulate(one, nsb=0.14, seed=5, source=SOURCE)
    assert simdata.run_summary(dl3)["source"][0]
    old = simdata.simulate(one, nsb=0.14, seed=5, source=SOURCE, production="2025")
    assert not simdata.run_summary(old)["source"][0]

    wide = reduce.analysis_geom(pnt, width=5.6, e_reco=(0.3, 20) * u.TeV)
    r = reduce.from_dl3(dl3, wide, T0)
    offset = wide.to_image().get_coord().skycoord.separation(pnt).deg
    expo = r.stacked.exposure.data.sum(axis=0)
    half_diag = 0.08 / np.sqrt(2)
    assert np.all(expo[offset < 2.4] > 0)
    assert np.all(expo[offset > 2.5 + half_diag] == 0)
    excess, err = _excess_near(r.stacked, radius=0.2)
    assert excess > 5 * err, (excess, err)


def test_cache_round_trip(dl3_bkg, geom, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CACHE", tmp_path)
    t = time.perf_counter()
    r1 = reduce.from_dl3(dl3_bkg, geom, T0, {"background": "model"}, verbose=False)
    t1 = time.perf_counter() - t
    t = time.perf_counter()
    r2 = reduce.from_dl3(dl3_bkg, geom, T0, {"background": "model"}, verbose=False)
    t2 = time.perf_counter() - t
    assert t2 < t1
    np.testing.assert_array_equal(r1.stacked.counts.data, r2.stacked.counts.data)
    np.testing.assert_allclose(r1.stacked.background.data, r2.stacked.background.data)
    assert len(list((tmp_path / "reduce").iterdir())) == 1
    reduce.from_dl3(dl3_bkg, geom, T0 + 1 * u.s, {"background": "model"}, verbose=False)
    assert len(list((tmp_path / "reduce").iterdir())) == 2  # another t0 is another entry


def test_unknown_setting_is_refused(dl3_bkg, geom):
    with pytest.raises(ValueError, match="unknown"):
        reduce.from_dl3(dl3_bkg, geom, T0, {"ring_radius": 0.3})
    with pytest.raises(ValueError, match="background"):
        reduce.from_dl3(dl3_bkg, geom, T0, {"background": "reflected"})
