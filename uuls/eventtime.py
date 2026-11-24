"""When a sky position is observable from the site, and where a skymap peaks.

The simulated cases keep the skymap in equatorial coordinates and only move the merger time,
so that the field is in darkness and at low zenith angle. `observable_time` gives that time.

    observable_time(target, after)   first time after `after` with the target at zenith angle
                                     < max_zenith and (optionally) the Sun below -18 deg
    peak_direction(skymap_path)      highest-probability direction of a HEALPix skymap
    altitude(target, times)          target altitude (deg) seen from the site
    sun_altitude(times)              Sun altitude (deg) seen from the site

Nothing else is required: the Moon is ignored here (tilepy handles it when scheduling).
"""
from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time

SITES = {
    "LST-1": {"lat": 28.7616, "lon": -17.8908, "height": 2200.0},  # Roque de los Muchachos
}
SUN_DARK_DEG = -18.0


def location(site="LST-1") -> EarthLocation:
    """Site name (key of SITES) or EarthLocation -> EarthLocation."""
    if isinstance(site, EarthLocation):
        return site
    s = SITES[site]
    return EarthLocation(lat=s["lat"] * u.deg, lon=s["lon"] * u.deg, height=s["height"] * u.m)


def altitude(target: SkyCoord, times: Time, site="LST-1") -> np.ndarray:
    """Altitude (deg) of `target` at `times` (Time array) from `site`. Out: array like times."""
    frame = AltAz(obstime=times, location=location(site))
    return target.icrs.transform_to(frame).alt.deg


def sun_altitude(times: Time, site="LST-1") -> np.ndarray:
    """Altitude (deg) of the Sun at `times` from `site`. Out: array like times."""
    frame = AltAz(obstime=times, location=location(site))
    return get_sun(times).transform_to(frame).alt.deg


def _ok(target, times, site, max_zenith, need_dark):
    ok = altitude(target, times, site) > 90.0 - max_zenith
    if need_dark:
        ok &= sun_altitude(times, site) < SUN_DARK_DEG
    return ok


def observable_time(target: SkyCoord, after, max_zenith=40.0, site="LST-1", need_dark=True,
                    step_min=5.0, max_days=400, tol_s=1.0) -> Time:
    """First time after `after` when `target` is below `max_zenith` (and in darkness).

    Scans in steps of `step_min` minutes, one day at a time, then bisects the first step that
    satisfies the condition down to `tol_s` seconds. If `after` itself satisfies it, returns
    `after`. Conditions: target zenith angle < max_zenith; Sun altitude < -18 deg if need_dark.
    In:  target SkyCoord, after (Time or str, UTC), max_zenith (deg), site (name or
         EarthLocation), need_dark bool.
    Out: Time (UTC). Raises RuntimeError if not found within `max_days`.
    """
    after = Time(after).utc
    n = int(round(24 * 60 / step_min))
    dt = np.arange(n + 1) * step_min * u.min
    for day in range(max_days):
        times = after + day * u.day + dt
        ok = _ok(target, times, site, max_zenith, need_dark)
        if not ok.any():
            continue
        i = int(np.argmax(ok))
        if i == 0 and day == 0:
            return after
        lo, hi = times[i - 1] if i > 0 else times[0] - step_min * u.min, times[i]
        while (hi - lo).to_value(u.s) > tol_s:
            mid = lo + (hi - lo) / 2
            if _ok(target, Time([mid]), site, max_zenith, need_dark)[0]:
                hi = mid
            else:
                lo = mid
        return hi
    raise RuntimeError(f"target not observable within {max_days} days after {after.isot}")


def peak_direction(skymap_path, nside=128) -> SkyCoord:
    """Direction of the highest-probability pixel of a HEALPix skymap.

    The map is first summed down to `nside` (default 128, pixels of 0.46 deg) so that the
    answer is a region of about the LST PSF size rather than a single fine pixel.
    In:  skymap_path (flat or multi-order HEALPix FITS), nside (power of 2).
    Out: SkyCoord (ICRS).
    """
    import healpy as hp
    from ligo.skymap.io import read_sky_map

    prob, _ = read_sky_map(str(Path(skymap_path)), moc=False, nest=True, distances=False)
    prob = hp.ud_grade(prob, nside, order_in="NESTED", order_out="NESTED", power=-2)
    ipix = int(np.argmax(prob))
    ra, dec = hp.pix2ang(nside, ipix, nest=True, lonlat=True)
    return SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
