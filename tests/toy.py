"""Small synthetic inputs for tests and tool notebooks.

Nothing here is physics: a flat instrument and a Gaussian GW blob, small enough that a
realisation takes a fraction of a second.

    reduced = toy_reduced()          # .geom, .stacked, .runs, .run_gti_s, .t0 (duck-types uuls.reduce.ReducedData)
    gw = toy_gwmap(reduced.geom)     # .prob, .mask, .cdf, .cdf_valid, .r_grid, .geom (duck-types uuls.gwmap.GWMap)

Defaults: 40 x 40 bins of 0.05 deg around (RA, Dec) = (150, 30), reco energy 0.6-20 TeV
in 3 bins, flat effective area 1e5 m2, Gaussian PSF of sigma 0.1 deg, diagonal energy
dispersion, flat background of 0.05 counts per bin and energy per run, two runs of 30 min.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from gammapy.data import GTI
from gammapy.datasets import Datasets, MapDataset
from gammapy.irf import EDispKernelMap, PSFMap
from gammapy.maps import Map, MapAxis, WcsGeom
from scipy.stats import norm

CENTER = (150.0, 30.0)
T0 = Time("2026-01-01T00:00:00", scale="utc")


def toy_geom(npix=40, binsz=0.05, center=CENTER, e_edges=(0.6, 20.0), n_e=3) -> WcsGeom:
    """3D WcsGeom (reco energy axis named "energy"), ICRS, TAN projection."""
    axis = MapAxis.from_energy_bounds(e_edges[0] * u.TeV, e_edges[1] * u.TeV, nbin=n_e,
                                      name="energy")
    return WcsGeom.create(skydir=SkyCoord(*center, unit="deg", frame="icrs"), npix=npix,
                          binsz=binsz, frame="icrs", proj="TAN", axes=[axis])


def toy_dataset(geom, livetime_s=1800.0, aeff_m2=1e5, bkg_per_bin=0.05, psf_sigma_deg=0.1,
                gti_s=(600.0, 2400.0), name="toy") -> MapDataset:
    """MapDataset on `geom` with flat exposure, flat background and a Gaussian PSF.

    gti_s: (start, stop) in seconds since T0. Counts are zero (fake them).
    """
    e_reco = geom.axes["energy"]
    e_true = MapAxis.from_energy_bounds(0.3 * u.TeV, 40.0 * u.TeV, nbin=12, name="energy_true")
    image = geom.to_image()
    exposure = Map.from_geom(image.to_cube([e_true]), unit="m2 s",
                             data=np.full(image.to_cube([e_true]).data_shape, aeff_m2 * livetime_s))
    background = Map.from_geom(geom, data=np.full(geom.data_shape, float(bkg_per_bin)))
    irf_geom = WcsGeom.create(skydir=image.center_skydir, npix=4, binsz=0.5, frame="icrs")
    psf = PSFMap.from_gauss(e_true, sigma=psf_sigma_deg * u.deg, geom=irf_geom)
    edisp = EDispKernelMap.from_diagonal_response(e_reco, e_true, geom=irf_geom)
    # IRF exposure maps are the stacking weights (Datasets.stack_reduce needs them).
    expo = aeff_m2 * livetime_s
    psf.exposure_map = Map.from_geom(psf.psf_map.geom.squash("rad"), unit="m2 s", data=expo)
    edisp.exposure_map = Map.from_geom(edisp.edisp_map.geom.squash("energy"), unit="m2 s",
                                       data=expo)
    gti = GTI.create(gti_s[0] * u.s, gti_s[1] * u.s, reference_time=T0)
    return MapDataset(counts=Map.from_geom(geom), exposure=exposure, background=background,
                      psf=psf, edisp=edisp, gti=gti, name=name,
                      mask_safe=Map.from_geom(geom, data=np.ones(geom.data_shape, bool)))


def toy_reduced(n_runs=2, run_s=1800.0, gap_s=120.0, first_s=600.0, geom=None, **kwargs):
    """Duck-typed ReducedData: per-run toy datasets and their stack.

    Returns a SimpleNamespace with geom, stacked (MapDataset), runs (list of MapDataset),
    run_gti_s (list of (1, 2) arrays, s since t0), t0. kwargs go to `toy_dataset`.
    """
    geom = geom or toy_geom()
    runs, gtis = [], []
    for j in range(n_runs):
        t1 = first_s + j * (run_s + gap_s)
        gti = (t1, t1 + run_s)
        runs.append(toy_dataset(geom, livetime_s=run_s, gti_s=gti, name=f"run{j}", **kwargs))
        gtis.append(np.array([gti]))
    stacked = Datasets(runs).stack_reduce(name="stacked")
    stacked.counts = Map.from_geom(geom)
    return SimpleNamespace(geom=geom, stacked=stacked, runs=runs, run_gti_s=gtis, t0=T0,
                           settings={"correlation_radius_deg": 0.1})


def toy_gwmap(geom, offset_deg=(0.15, -0.1), sigma_deg=0.25, credible=0.9,
              d_mean=400.0, d_sigma=80.0, r_grid=None, name="toy"):
    """Duck-typed GWMap: a Gaussian blob of probability and the same Gaussian distance CDF
    in every bin.

    prob sums to the fraction of the blob inside the map; mask is the `credible` region of
    the in-map probability.
    """
    image = geom.to_image()
    c = image.get_coord()
    centre = image.center_skydir
    lon0 = centre.ra.deg + offset_deg[0] / np.cos(np.deg2rad(centre.dec.deg))
    lat0 = centre.dec.deg + offset_deg[1]
    sep = SkyCoord(c.lon, c.lat).separation(SkyCoord(lon0 * u.deg, lat0 * u.deg)).deg
    density = np.exp(-0.5 * (sep / sigma_deg) ** 2)
    pix_sr = image.solid_angle().to_value("deg2")
    prob = density * pix_sr / (2 * np.pi * sigma_deg ** 2)
    order = np.argsort(prob, axis=None)[::-1]
    cum = np.cumsum(prob.ravel()[order]) / prob.sum()
    mask = np.zeros(prob.size, bool)
    mask[order[: int(np.searchsorted(cum, credible)) + 1]] = True
    r_grid = np.linspace(1.0, 1000.0, 400) if r_grid is None else np.asarray(r_grid, float)
    cdf_1d = norm.cdf(r_grid, d_mean, d_sigma)
    cdf_1d = (cdf_1d - cdf_1d[0]) / (cdf_1d[-1] - cdf_1d[0])
    ny, nx = prob.shape
    return SimpleNamespace(name=name, geom=image, prob=prob, mask=mask.reshape(prob.shape),
                           credible=credible, r_grid=r_grid,
                           cdf=np.broadcast_to(cdf_1d, (ny, nx, r_grid.size)).copy(),
                           cdf_valid=np.ones((ny, nx), bool), t0=T0)


REFERENCE_REPO = Path(__file__).resolve().parents[2] / "gw-gamma-global-uls"


def reference_simulate():
    """The old gwuls.simulate module, for regression tests only (None if not on disk)."""
    if not (REFERENCE_REPO / "gwuls" / "simulate.py").exists():
        return None
    if str(REFERENCE_REPO) not in sys.path:
        sys.path.insert(0, str(REFERENCE_REPO))
    from gwuls import simulate

    return simulate
