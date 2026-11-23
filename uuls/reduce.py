"""DL3 runs (real or simulated) to the reduced map datasets of the analysis.

    geom = analysis_geom(center, width=3.6)
    reduced = from_dl3(dl3_dir, geom, t0)          # cached
    reduced.stacked                                # MapDataset: counts, background, exposure, psf, edisp
    reduced.runs[k], reduced.run_gti_s[k]          # per run, for time-resolved injection

Ported from the reference notebook (gw-gamma-global-uls, notebooks/3_upper_limits.ipynb, Part 1, and
docs/pipeline.md §4.1): per run `MapDatasetMaker` (counts, background, exposure, psf, edisp), `SafeMaskMaker`
(offset-max 2.5 deg), `RingBackgroundMaker` (r_in 0.3 deg, width 0.2 deg) on the run's background model, then
stacked. The background model is the run's `bkg` IRF in the DataStore. Real LST-1 DL3 usually has none: give
`settings["bkg_pattern"]` (a file name pattern with `{obs_id}`, relative to the DL3 folder, as the reference
`utils.add_bkg` did), and the file is registered for the run. Nothing here knows whether the DL3 is simulated.

Background methods (`settings["background"]`):
    "ring"  reference behaviour. mu_bkg = alpha * n_off, n_off the counts in a ring around each pixel and
            alpha the ratio of the model acceptance inside the pixel to that inside the ring.
    "fov"   the run's background model scaled by one factor per run, fitted to the counts outside the
            exclusion regions (FoVBackgroundMaker, method "scale").
    "model" the run's background model as it is.

Two changes against the reference, both settings: `ring_within_safe_mask=True` restricts the ring to pixels
inside the safe mask (the reference let the ring reach beyond offset-max, where simulst has no events but the
model is non-zero, which biased mu_bkg low within 0.5 deg of the edge); `exclusion` lists (ra, dec, radius)
circles removed from the ring and from the FoV normalisation (the reference used none).
"""
import copy
import glob
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from gammapy.data import DataStore
from gammapy.datasets import MapDataset
from gammapy.makers import FoVBackgroundMaker, MapDatasetMaker, RingBackgroundMaker, SafeMaskMaker
from gammapy.maps import Map, MapAxis, WcsGeom

from . import cache

VERSION = "1"  # bump when the reduction would give a different result

DEFAULT_SETTINGS = {
    "e_true_tev": [0.05, 100.0],         # true-energy axis of exposure and edisp
    "true_bins_per_decade": 4.5,         # the reference uses the reco binning for the true axis too
    "offset_max_deg": 2.5,               # SafeMaskMaker offset-max
    "safe_mask": ["offset-max"],         # SafeMaskMaker methods
    "background": "ring",                # "ring" | "fov" | "model"
    "ring_r_in_deg": 0.3,
    "ring_width_deg": 0.2,
    "ring_within_safe_mask": True,
    "exclusion": [],                     # [[ra, dec, radius], ...] deg
    "obs_ids": None,                     # None: every run of the DataStore
    "bkg_pattern": None,                 # e.g. "bkg_pybkgmodel_3d_{obs_id}.fits" when the index has no bkg rows
    "correlation_radius_deg": 0.1,       # not used here; recorded for the estimators downstream
}
BACKGROUNDS = ("ring", "fov", "model")


@dataclass
class ReducedData:
    """Reduced map datasets of one set of DL3 runs on one analysis geometry.

    geom: 3D WcsGeom with the reco energy axis.
    stacked: MapDataset, counts, background (ring estimate or model), exposure, psf, edisp, mask_safe.
    runs: one MapDataset per run on `geom`, named "run<obs_id>", with its own background and GTI.
    run_gti_s: per run, (n_intervals, 2) array of GTI start/stop in seconds since t0.
    t0: reference time (the merger).
    settings: every analysis setting used (DEFAULT_SETTINGS updated by the caller, plus binsz, energy
        edges, obs_ids, background method).
    """
    geom: WcsGeom
    stacked: MapDataset
    runs: list
    run_gti_s: list
    t0: Time
    settings: dict = field(default_factory=dict)

    @property
    def obs_ids(self) -> list[int]:
        return list(self.settings["obs_ids_used"])


def analysis_geom(center: SkyCoord, width: float | tuple = 2.5 * 1.43, binsz: float = 0.08,
                  e_reco: u.Quantity = (0.6, 20) * u.TeV, bins_per_decade: float = 4.5,
                  proj: str = "AIR") -> WcsGeom:
    """3D analysis geometry: ICRS, AIR projection, reco energy axis "energy".

    center: SkyCoord of the map centre. width: deg, one value or (width_ra, width_dec); the reference used
    2.5 * 1.43 = 3.575 deg. binsz: deg. e_reco: (min, max) Quantity; bins_per_decade as in
    MapAxis.from_energy_bounds(per_decade=True) (0.6-20 TeV at 4.5 per decade gives 7 bins).
    Returns WcsGeom with npix = int(width / binsz) per side.
    """
    w = (float(width),) * 2 if np.ndim(width) == 0 else tuple(float(x) for x in width)
    e = u.Quantity(e_reco, u.TeV)
    axis = MapAxis.from_energy_bounds(e[0], e[1], nbin=bins_per_decade, per_decade=True, name="energy")
    npix = (int(round(w[0] / binsz)), int(round(w[1] / binsz)))
    c = center.icrs
    return WcsGeom.create(skydir=(c.ra.deg, c.dec.deg), npix=npix, binsz=binsz, frame="icrs", proj=proj,
                          axes=[axis])


def gti_seconds_since(gti, t0: Time) -> np.ndarray:
    """A gammapy GTI as an (n_intervals, 2) array of seconds since t0 (reference: simulate.gti_seconds_since)."""
    return np.column_stack([(gti.time_start - t0).to_value(u.s), (gti.time_stop - t0).to_value(u.s)])


def _geom_params(geom: WcsGeom) -> dict:
    return {"wcs": geom.wcs.to_header_string(), "npix": [int(n) for n in np.ravel(geom.npix)],
            "axes": {ax.name: ax.edges.to_value(u.TeV) if ax.unit.physical_type == "energy" else ax.edges.value
                     for ax in geom.axes}}


def _settings(settings: dict | None) -> dict:
    s = copy.deepcopy(DEFAULT_SETTINGS)
    unknown = set(settings or {}) - set(s)
    if unknown:
        raise ValueError(f"unknown reduce settings {sorted(unknown)}; known: {sorted(s)}")
    s.update(copy.deepcopy(settings or {}))
    if s["background"] not in BACKGROUNDS:
        raise ValueError(f"background must be one of {BACKGROUNDS}, got {s['background']!r}")
    s["e_true_tev"] = [float(x) for x in s["e_true_tev"]]
    s["exclusion"] = [[float(v) for v in reg] for reg in s["exclusion"]]
    if s["obs_ids"] is not None:
        s["obs_ids"] = [int(i) for i in s["obs_ids"]]
    return s


def _exclusion_mask(geom: WcsGeom, regions: list) -> Map:
    """Boolean image on geom: True where the pixel can be used for background (outside every circle)."""
    image = geom.to_image()
    coords = image.get_coord().skycoord
    keep = np.ones(image.data_shape, dtype=bool)
    for ra, dec, radius in regions:
        keep &= coords.separation(SkyCoord(ra, dec, unit="deg")).deg > radius
    return Map.from_geom(image, data=keep)


def _data_store(dl3_dir: Path, s: dict) -> tuple[DataStore, list[int]]:
    store = DataStore.from_dir(dl3_dir)
    obs_ids = s["obs_ids"] if s["obs_ids"] is not None else [int(i) for i in store.obs_table["OBS_ID"]]
    hdu = store.hdu_table
    for obs_id in obs_ids:
        if np.any((hdu["OBS_ID"] == obs_id) & (hdu["HDU_TYPE"] == "bkg")):
            continue
        if s["bkg_pattern"] is None:
            raise ValueError(f"run {obs_id} has no background model in {dl3_dir}/hdu-index; give "
                             "settings['bkg_pattern'] (file name with {obs_id}, relative to the DL3 folder)")
        found = sorted(glob.glob(str(Path(dl3_dir) / s["bkg_pattern"].format(obs_id=obs_id))))
        if len(found) != 1:
            raise FileNotFoundError(f"run {obs_id}: pattern {s['bkg_pattern']!r} matches {found}")
        with fits.open(found[0]) as h:
            size = h["BACKGROUND"].size
        hdu.add_row({"OBS_ID": obs_id, "HDU_TYPE": "bkg", "HDU_CLASS": "bkg_3d", "FILE_DIR": ".",
                     "FILE_NAME": Path(found[0]).name, "HDU_NAME": "BACKGROUND", "SIZE": size})
    store.hdu_table = hdu.copy()
    return store, obs_ids


def _reduce_run(obs, empty: MapDataset, s: dict, exclusion: Map) -> MapDataset:
    if obs.rad_max is not None:
        raise ValueError(f"run {obs.obs_id} has point-like IRFs (RAD_MAX): the map analysis needs full enclosure")
    name = f"run{obs.obs_id}"
    maker = MapDatasetMaker(selection=["counts", "background", "exposure", "psf", "edisp"])
    safe = SafeMaskMaker(methods=s["safe_mask"], offset_max=s["offset_max_deg"] * u.deg)
    ds = safe.run(maker.run(empty.copy(name=name), obs), obs)
    if s["background"] == "model":
        return ds
    if s["background"] == "fov":
        return FoVBackgroundMaker(method="scale", exclusion_mask=exclusion).run(ds)
    mask = Map.from_geom(ds.counts.geom, data=np.broadcast_to(exclusion.data, ds.counts.data.shape).copy())
    if s["ring_within_safe_mask"]:
        mask.data &= ds.mask_safe.data
    ring = RingBackgroundMaker(r_in=s["ring_r_in_deg"] * u.deg, width=s["ring_width_deg"] * u.deg,
                               exclusion_mask=mask)
    return ring.run(ds).to_map_dataset(name=name)


def _from_dl3(dl3_dir: Path, geom: WcsGeom, t0: Time, s: dict) -> ReducedData:
    store, obs_ids = _data_store(dl3_dir, s)
    observations = store.get_observations(obs_id=obs_ids, required_irf=["aeff", "edisp", "psf", "bkg"])
    e_true = MapAxis.from_energy_bounds(s["e_true_tev"][0] * u.TeV, s["e_true_tev"][1] * u.TeV,
                                        nbin=s["true_bins_per_decade"], per_decade=True, name="energy_true")
    empty = MapDataset.create(geom, energy_axis_true=e_true, name="empty")
    stacked = MapDataset.create(geom, energy_axis_true=e_true, name="stacked")
    exclusion = _exclusion_mask(geom, s["exclusion"])
    runs, gtis = [], []
    for obs in observations:
        ds = _reduce_run(obs, empty, s, exclusion)
        runs.append(ds)
        stacked.stack(ds)
        gtis.append(gti_seconds_since(obs.gti, t0))
    settings = {**s, "obs_ids_used": [int(o.obs_id) for o in observations], "binsz_deg": float(geom.pixel_scales[0].deg),
                "e_reco_edges_tev": geom.axes["energy"].edges.to_value(u.TeV).tolist(),
                "e_true_edges_tev": e_true.edges.to_value(u.TeV).tolist(), "version": VERSION}
    return ReducedData(geom=geom, stacked=stacked, runs=runs, run_gti_s=gtis, t0=t0, settings=settings)


def from_dl3(dl3_dir: Path, geom: WcsGeom, t0: Time, settings: dict | None = None,
             verbose: bool = True) -> ReducedData:
    """Reduce every run of a DL3 folder (with hdu-index and obs-index) to map datasets on `geom`.

    dl3_dir: DL3 folder, real (lstchain) or simulated (uuls.simdata). geom: from analysis_geom.
    t0: Time, origin of run_gti_s (the merger). settings: overrides of DEFAULT_SETTINGS (see module doc).
    Returns ReducedData. Cached on the DL3 files (path, size, mtime), geom, t0 and settings.
    """
    s = _settings(settings)
    params = {"version": VERSION, "dl3_dir": Path(dl3_dir), "geom": _geom_params(geom), "t0": Time(t0).utc.isot,
              "settings": s}

    def compute():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*outside valid IRF map range.*")
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            return _from_dl3(Path(dl3_dir), geom, Time(t0), s)

    return cache.cached("reduce", params, compute, verbose=verbose)
