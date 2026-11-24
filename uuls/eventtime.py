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


def coarse_probability(skymap_path, nside=128) -> np.ndarray:
    """Probability per NESTED pixel of `nside`, summed directly from the multi-order cells.

    Cells finer than `nside` are summed into their parent pixel; cells coarser than `nside`
    spread their probability evenly over their children. The full-resolution map is never
    built, so a multi-order skymap of NSIDE 4096 costs the same as one of NSIDE 128.
    In:  skymap_path (multi-order HEALPix FITS, NUNIQ table), nside (power of 2).
    Out: float array of 12 * nside**2 probabilities, summing to the map total.
    """
    import healpy as hp
    from ligo.skymap.io import read_sky_map

    tab = read_sky_map(str(Path(skymap_path)), moc=True)
    uniq = np.asarray(tab["UNIQ"], dtype=np.int64)
    order = np.floor(np.log2(np.sqrt(uniq // 4))).astype(np.int64)   # HEALPix order of each cell
    ipix = uniq - 4 * (2 ** order) ** 2                               # NESTED index at its order
    prob = np.asarray(tab["PROBDENSITY"], float) * hp.nside2pixarea(2 ** order)
    k = int(np.log2(nside))
    out = np.zeros(12 * nside * nside)
    fine = order >= k
    np.add.at(out, ipix[fine] >> (2 * (order[fine] - k)), prob[fine])
    for o in np.unique(order[~fine]):
        m = (~fine) & (order == o)
        n = 4 ** (k - o)
        child = (ipix[m][:, None] * n + np.arange(n)[None, :]).ravel()
        out[child] += np.repeat(prob[m] / n, n)
    return out


def peak_direction(skymap_path, nside=128) -> SkyCoord:
    """Direction of the highest-probability pixel of a HEALPix skymap.

    The map is summed down to `nside` (default 128, pixels of 0.46 deg) so that the answer is
    a region of about the LST PSF size rather than a single fine pixel. Uses
    `coarse_probability`, so the cost does not depend on the finest order of the map.
    In:  skymap_path (multi-order HEALPix FITS), nside (power of 2).
    Out: SkyCoord (ICRS).
    """
    import healpy as hp

    prob = coarse_probability(skymap_path, nside)
    ra, dec = hp.pix2ang(nside, int(np.argmax(prob)), nest=True, lonlat=True)
    return SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
