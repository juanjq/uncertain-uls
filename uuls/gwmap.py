"""The GW localisation (sky and distance) on a WCS analysis grid.

    sky = read_healpix(path)                   # HealpixSkymap, multi-order or flat file
    gw = on_geom(path, geom, credible=0.95)    # GWMap on the 2D image geom, cached
    credible_area_deg2(path, 0.9)

Physics (as in gwuls.utils, docs/pipeline.md §3 of the old repo):

- Bin probability p_b: mean HEALPix probability density over the bin times the bin solid angle.
  The mean is taken over an n x n grid of points inside the bin (`OVERSAMPLE`), which replaces the
  old "mean over the pixels whose centre is in the bin" and works the same for multi-order maps.
- Credible mask: a bin is in the region if the HEALPix pixel under its centre is in the greedy
  `credible` region (pixels sorted by density, accumulated until `credible`).
- Distance CDF of a bin: the density-weighted mixture of the ansatz CDFs of the HEALPix pixels
  under the same points (`ligo.skymap.distance.marginal_cdf`). Pixels without distance
  information (DISTMU = inf) are skipped. The grid starts at 1 Mpc: a draw at d = 0 means an
  infinite flux downstream.
"""
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import astropy_healpix as ah
import healpy as hp
import numpy as np
from astropy.time import Time
from gammapy.maps import WcsGeom
from ligo.skymap.distance import marginal_cdf
from ligo.skymap.io import read_sky_map

from .cache import cached

VERSION = "1"            # bump when the output of on_geom changes
OVERSAMPLE = 8           # points per bin side used to integrate the density over a bin
R_GRID_DEFAULT = np.linspace(1.0, 10000.0, 1000)   # Mpc, same as the old notebooks


@dataclass(frozen=True, eq=False)
class HealpixSkymap:
    """A GW skymap in multi-order form (a flat map is the case of a single order).

    Arrays have one entry per HEALPix pixel. `prob` is the probability of the pixel (not a
    density), `nside` the pixel's own nside, `uniq` its NUNIQ index (NESTED scheme).
    """
    name: str
    t0: Time
    uniq: np.ndarray
    nside: np.ndarray
    prob: np.ndarray
    distmu: np.ndarray
    distsigma: np.ndarray
    distnorm: np.ndarray

    @property
    def area_sr(self) -> np.ndarray:
        return ah.nside_to_pixel_area(self.nside).to_value(u.sr)

    @property
    def density(self) -> np.ndarray:
        """Probability per steradian."""
        return self.prob / self.area_sr

    def lookup(self, ra_deg, dec_deg) -> np.ndarray:
        """Index (into the pixel arrays) of the pixel containing each (ra, dec), any shape."""
        level, ipix = ah.uniq_to_level_ipix(self.uniq)
        top = int(level.max())
        start = ipix << (2 * (top - level))          # first NESTED index at the top order
        order = np.argsort(start)
        fine = ah.lonlat_to_healpix(np.asarray(ra_deg) * u.deg, np.asarray(dec_deg) * u.deg,
                                    ah.level_to_nside(top), order="nested")
        return order[np.searchsorted(start[order], fine, side="right") - 1]


@dataclass(frozen=True, eq=False)
class GWMap:
    """The GW localisation on a 2D WCS geom. See docs/interfaces.md."""
    name: str
    t0: Time
    geom: WcsGeom
    prob: np.ndarray
    mask: np.ndarray
    credible: float
    r_grid: np.ndarray
    cdf: np.ndarray
    cdf_valid: np.ndarray


def read_healpix(path) -> HealpixSkymap:
    """Read a HEALPix FITS skymap with DISTMU/DISTSIGMA/DISTNORM.

    In: path to a multi-order (UNIQ) or flat FITS skymap. Out: HealpixSkymap, with t0 the
    merger time from the header (`gps_time`, UTC) and name the header OBJECT.
    """
    m = read_sky_map(str(path), moc=True)
    level, _ = ah.uniq_to_level_ipix(m["UNIQ"])
    nside = ah.level_to_nside(level)
    prob = np.asarray(m["PROBDENSITY"]) * ah.nside_to_pixel_area(nside).to_value(u.sr)
    return HealpixSkymap(
        name=str(m.meta.get("objid", Path(path).stem)),
        t0=Time(m.meta["gps_time"], format="gps").utc,
        uniq=np.asarray(m["UNIQ"]), nside=np.asarray(nside), prob=prob,
        distmu=np.asarray(m["DISTMU"]), distsigma=np.asarray(m["DISTSIGMA"]),
        distnorm=np.asarray(m["DISTNORM"]),
    )


def from_flat(prob, distmu, distsigma, distnorm, name="", t0=None, nest=True) -> HealpixSkymap:
    """HealpixSkymap from flat arrays (prob per pixel, RING or NESTED), e.g. a modified map."""
    nside = ah.npix_to_nside(len(prob))
    ipix = np.arange(len(prob)) if nest else hp.ring2nest(nside, np.arange(len(prob)))
    return HealpixSkymap(
        name=name, t0=t0, uniq=ah.level_ipix_to_uniq(ah.nside_to_level(nside), ipix),
        nside=np.full(len(prob), nside), prob=np.asarray(prob, dtype=float),
        distmu=np.asarray(distmu), distsigma=np.asarray(distsigma), distnorm=np.asarray(distnorm),
    )


def credible_levels(sky: HealpixSkymap) -> np.ndarray:
    """Per pixel, the probability of all pixels at least as dense (greedy credible level)."""
    order = np.argsort(sky.density)[::-1]
    levels = np.empty_like(sky.prob)
    levels[order] = np.cumsum(sky.prob[order])
    return levels


def credible_area_deg2(path, credible=0.9) -> float:
    """Area (deg2) of the greedy `credible` region of the skymap at `path`."""
    sky = read_healpix(path)
    inside = credible_levels(sky) <= credible
    return float(np.sum(sky.area_sr[inside]) * (180 / np.pi) ** 2)


def _subbin_coords(geom: WcsGeom, n: int):
    """(ra, dec) in deg of an n x n grid of points inside every bin, shape (ny, nx, n*n)."""
    ny, nx = geom.data_shape
    off = (np.arange(n) + 0.5) / n - 0.5
    dx, dy = [a.ravel() for a in np.meshgrid(off, off)]
    iy, ix = np.mgrid[0:ny, 0:nx]
    x = ix[..., None] + dx
    y = iy[..., None] + dy
    lon, lat = geom.pix_to_coord((x, y))
    return np.asarray(lon), np.asarray(lat)


def _bin_prob(sky: HealpixSkymap, idx: np.ndarray, geom: WcsGeom) -> np.ndarray:
    """Probability per bin: mean density at the sub-bin points times the bin solid angle."""
    return sky.density[idx].mean(axis=-1) * geom.solid_angle().to_value(u.sr)


def _bin_mask(sky: HealpixSkymap, geom: WcsGeom, credible: float) -> np.ndarray:
    """True for bins whose centre lies in the greedy `credible` region."""
    ra, dec = (np.asarray(c.value) for c in geom.get_coord(mode="center")[:2])
    return credible_levels(sky)[sky.lookup(ra, dec)] <= credible


def _cdf_of_points(sky: HealpixSkymap, idx: np.ndarray, r_grid: np.ndarray):
    """Distance CDF of the mixture of the pixels `idx` (one entry per sample point), or None."""
    pix, count = np.unique(idx, return_counts=True)
    w = count * sky.density[pix]
    good = np.isfinite(sky.distmu[pix]) & (w > 0)
    if not good.any():
        return None
    pix, w = pix[good], w[good]
    return marginal_cdf(r_grid, w / w.sum(), sky.distmu[pix], sky.distsigma[pix], sky.distnorm[pix])


def _bin_cdf(sky: HealpixSkymap, idx: np.ndarray, prob: np.ndarray, r_grid: np.ndarray):
    """Per-bin distance CDF table (ny, nx, n_r) and the bins where it exists."""
    cdf = np.zeros(prob.shape + r_grid.shape, dtype=np.float32)
    valid = np.zeros(prob.shape, dtype=bool)
    for iy, ix in zip(*np.nonzero(prob > 0)):
        c = _cdf_of_points(sky, idx[iy, ix], r_grid)
        if c is not None:
            cdf[iy, ix], valid[iy, ix] = c, True
    return cdf, valid


def _image_geom(geom: WcsGeom) -> WcsGeom:
    return geom.to_image() if geom.ndim > 2 else geom


def _check_r_grid(r_grid) -> np.ndarray:
    r = np.asarray(R_GRID_DEFAULT if r_grid is None else r_grid, dtype=float)
    if r.ndim != 1 or r[0] <= 0 or np.any(np.diff(r) <= 0):
        raise ValueError("r_grid must be 1D, increasing and start above 0 Mpc")
    return r


def project(sky: HealpixSkymap, geom: WcsGeom, credible=0.95, r_grid=None) -> GWMap:
    """Uncached work of `on_geom`, on a skymap already in memory."""
    geom, r = _image_geom(geom), _check_r_grid(r_grid)
    idx = sky.lookup(*_subbin_coords(geom, OVERSAMPLE))
    prob = _bin_prob(sky, idx, geom)
    cdf, valid = _bin_cdf(sky, idx, prob, r)
    return GWMap(name=sky.name, t0=sky.t0, geom=geom, prob=prob,
                 mask=_bin_mask(sky, geom, credible), credible=float(credible),
                 r_grid=r, cdf=cdf, cdf_valid=valid)


def geom_params(geom: WcsGeom) -> dict:
    """Plain description of a 2D geom (WCS header and shape), used as a cache key."""
    g = _image_geom(geom)
    return {"wcs": dict(g.wcs.to_header()), "npix": [int(n) for n in g.data_shape]}


def on_geom(path, geom: WcsGeom, credible=0.95, r_grid=None, root=None) -> GWMap:
    """GW map on a WCS geom, cached on (skymap file, geom, credible, r_grid, VERSION).

    In: skymap path; geom (WcsGeom, non-spatial axes dropped); credible level of the mask;
    r_grid (Mpc, starts > 0; default 1-10000 Mpc, 1000 points); root: cache folder override.
    Out: GWMap.
    """
    r = _check_r_grid(r_grid)
    params = {"skymap": Path(path), "geom": geom_params(geom), "credible": float(credible),
              "r_grid": r, "oversample": OVERSAMPLE, "version": VERSION}
    return cached("gwmap", params, lambda: project(read_healpix(path), geom, credible, r), root=root)
