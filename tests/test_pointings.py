import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

from uuls import paths, pointings as pt

TARGET = SkyCoord(120.0 * u.deg, 30.0 * u.deg)
T = Time("2026-01-01T22:00:00", scale="utc")


def test_on_runs_back_to_back():
    tab = pt.on(TARGET, T, 3, 1200 * u.s)
    assert list(tab["obs_id"]) == [1, 2, 3]
    np.testing.assert_allclose(tab["ra"], 120.0)
    np.testing.assert_allclose(np.diff(tab["t_start"].mjd) * 86400, 1200, atol=1e-3)
    assert tab.meta["strategy"] == "on" and tab.meta["t0"] == T.isot


def test_wobble_offsets_and_angles():
    tab = pt.wobble(TARGET, T, 6, 1200, offset=0.4)
    c = pt.coords(tab)
    np.testing.assert_allclose(c.separation(TARGET).deg, 0.4, atol=1e-9)
    dpa = (TARGET.position_angle(c).deg - [0, 90, 180, 270, 0, 90] + 180) % 360 - 180
    np.testing.assert_allclose(dpa, 0, atol=1e-6)
    assert c[0].dec.deg > TARGET.dec.deg      # angle 0 is north
    assert c[1].ra.deg > TARGET.ra.deg        # angle 90 is east


def test_grid_spacing_and_centre():
    tab = pt.grid(TARGET, T, 3, 1.5, 600)
    c = pt.coords(tab)
    assert len(tab) == 9
    assert c[4].separation(TARGET).deg < 1e-9          # middle of the snake is the centre
    np.testing.assert_allclose(c[:-1].separation(c[1:]).deg, 1.5, rtol=1e-3)  # neighbours


def test_ecsv_round_trip(tmp_path):
    tab = pt.wobble(TARGET, T, 4, 1200, t0=T - 60 * u.s)
    back = pt.read(pt.write(tab, tmp_path / "p.ecsv"))
    assert isinstance(back["t_start"], Time)
    np.testing.assert_allclose(back["t_start"].mjd, tab["t_start"].mjd)
    np.testing.assert_allclose(back["dec"], tab["dec"])
    assert back.meta["t0"] == (T - 60 * u.s).isot and back.meta["strategy"] == "wobble"


def test_validate_rejects_bad_tables():
    tab = pt.on(TARGET, T, 2, 100)
    tab["obs_id"] = [1, 1]
    with pytest.raises(ValueError):
        pt.validate(tab)
    with pytest.raises(ValueError):
        pt.make([0], [0], T, 10, strategy="spiral")


@pytest.mark.slow
def test_tilepy_2d_on_s240615dg(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CACHE", tmp_path)
    t0 = Time("2026-01-01T19:50:00", scale="utc")   # peak just observable, Moon ignored below
    tab = pt.tilepy_2d(paths.GW_SKYMAPS / "S240615dg.fits", t0, settings={"moonDown": 90, "maxRuns": 3})
    assert tab.meta["strategy"] == "tilepy" and len(tab) >= 1
    assert tab["pgw"][0] > 0.5                         # one LST field covers most of this map
    assert np.all(tab["t_start"] >= t0) and np.all(tab["duration"] == 1200)
    again = pt.tilepy_2d(paths.GW_SKYMAPS / "S240615dg.fits", t0, settings={"moonDown": 90, "maxRuns": 3})
    np.testing.assert_array_equal(again["ra"], tab["ra"])
