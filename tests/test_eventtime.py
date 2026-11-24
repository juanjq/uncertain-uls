import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

from uuls import eventtime, paths

TARGET = SkyCoord(120.0 * u.deg, 30.0 * u.deg)


def test_observable_time_is_the_first_good_time():
    after = Time("2026-03-01T12:00:00", scale="utc")
    t = eventtime.observable_time(TARGET, after, max_zenith=40)
    assert t > after
    assert eventtime.altitude(TARGET, t) > 50 - 1e-3
    assert eventtime.sun_altitude(t) < -18 + 1e-3
    before = t - 5 * u.s
    bad = (eventtime.altitude(TARGET, before) <= 50) or (eventtime.sun_altitude(before) >= -18)
    assert bad


def test_returns_after_when_already_observable():
    t = eventtime.observable_time(TARGET, Time("2026-03-01T12:00:00"), max_zenith=40)
    later = t + 10 * u.min
    assert abs((eventtime.observable_time(TARGET, later) - later).to_value(u.s)) < 1e-6


def test_without_darkness_only_zenith_counts():
    after = Time("2026-03-01T12:00:00", scale="utc")
    t_any = eventtime.observable_time(TARGET, after, need_dark=False)
    t_dark = eventtime.observable_time(TARGET, after)
    assert t_any <= t_dark


def test_unreachable_target_raises():
    with pytest.raises(RuntimeError):
        eventtime.observable_time(SkyCoord(0 * u.deg, -80 * u.deg), Time("2026-01-01"), max_days=3)


def test_peak_direction_of_s240615dg():
    c = eventtime.peak_direction(paths.GW_SKYMAPS / "S240615dg.fits")
    assert c.separation(SkyCoord(8.0 * u.deg, 45.8 * u.deg)).deg < 1.0
