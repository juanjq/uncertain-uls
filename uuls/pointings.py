"""Observation plans as `Pointings` tables: fixed patterns and the tilepy 2D scheduler.

A `Pointings` table is an `astropy.table.Table` with one row per observation run:

    obs_id    int          unique within the table
    ra, dec   float, deg   telescope pointing (ICRS)
    t_start   Time         UTC start of the run
    duration  float, s     live time asked for

    meta["t0"]        merger time, ISO string (UTC)
    meta["strategy"]  "on", "wobble", "grid" or "tilepy"

`tilepy_2d` adds a column `pgw` (GW probability tilepy assigned to the tile, counting only
probability not already covered by earlier tiles) and meta entries with its settings.

    write(table, path) / read(path)    ECSV on disk
    on, wobble, grid                   fixed patterns, runs back to back from t_start
    tilepy_2d                          greedy 2D GW-probability tiling for LST-1 (cached)
"""
from __future__ import annotations

import configparser
import contextlib
import logging
import math
import tempfile
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from astropy.table import Table
from astropy.time import Time

from . import cache

COLUMNS = ("obs_id", "ra", "dec", "t_start", "duration")
STRATEGIES = ("on", "wobble", "grid", "tilepy", "real")

# LST-1 at the Roque de los Muchachos.
LST1_SITE = {"name": "LST-1", "lat": 28.7616, "lon": -17.8908, "height": 2200.0}


# --------------------------------------------------------------------------------------------
# table helpers

def _seconds(x) -> float:
    """A duration given as a Quantity or a number of seconds -> float seconds."""
    if hasattr(x, "unit"):
        return float(x.to_value(u.s))
    return float(x)


def _t0_str(t0) -> str | None:
    if t0 is None:
        return None
    return Time(t0).utc.isot


def make(ra, dec, t_start, duration, strategy: str, t0=None, obs_id=None, **extra) -> Table:
    """Build a Pointings table from columns.

    In:  ra, dec (deg, array-like), t_start (Time array or list of Time/str), duration (s,
         scalar or array), strategy (str), t0 (Time or str, optional; defaults to the first
         t_start), obs_id (int array, optional; default 1..n), extra columns as keyword arrays.
    Out: Pointings table, rows sorted by t_start.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    ra = np.atleast_1d(np.asarray(ra, dtype=float)) % 360.0
    dec = np.atleast_1d(np.asarray(dec, dtype=float))
    n = len(ra)
    t_start = Time(t_start).utc
    if t_start.isscalar:
        t_start = Time([t_start.isot] * n, scale="utc")
    duration = np.broadcast_to(np.asarray(duration, dtype=float), (n,)).copy()
    obs_id = np.arange(1, n + 1) if obs_id is None else np.asarray(obs_id, dtype=int)
    tab = Table({"obs_id": obs_id, "ra": ra, "dec": dec, "t_start": t_start, "duration": duration})
    for k, v in extra.items():
        tab[k] = v
    tab["ra"].unit = u.deg
    tab["dec"].unit = u.deg
    tab["duration"].unit = u.s
    if n:
        tab = tab[np.argsort(tab["t_start"].mjd, kind="stable")]
    tab.meta["t0"] = _t0_str(t0) if t0 is not None else (tab["t_start"][0].isot if n else None)
    tab.meta["strategy"] = strategy
    validate(tab)
    return tab


def validate(tab: Table) -> None:
    """Raise ValueError if `tab` does not follow the Pointings contract."""
    missing = [c for c in COLUMNS if c not in tab.colnames]
    if missing:
        raise ValueError(f"Pointings table lacks columns {missing}")
    if len(set(np.asarray(tab["obs_id"]).tolist())) != len(tab):
        raise ValueError("obs_id values must be unique")
    if not isinstance(tab["t_start"], Time):
        raise ValueError("t_start must be an astropy Time column")
    if "t0" not in tab.meta or "strategy" not in tab.meta:
        raise ValueError("meta must hold 't0' and 'strategy'")


def coords(tab: Table) -> SkyCoord:
    """Pointing directions of a Pointings table as a SkyCoord (ICRS)."""
    return SkyCoord(np.asarray(tab["ra"]) * u.deg, np.asarray(tab["dec"]) * u.deg, frame="icrs")


def write(tab: Table, path) -> Path:
    """Write a Pointings table to ECSV. In: table, path. Out: the path."""
    validate(tab)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tab.write(path, format="ascii.ecsv", overwrite=True)
    return path


def read(path) -> Table:
    """Read a Pointings table from ECSV. In: path. Out: Pointings table."""
    tab = Table.read(path, format="ascii.ecsv")
    if not isinstance(tab["t_start"], Time):
        tab["t_start"] = Time(tab["t_start"], scale="utc")
    validate(tab)
    return tab


def _times(t_start, n_runs, run_s) -> Time:
    t_start = Time(t_start).utc
    return t_start + np.arange(n_runs) * run_s * u.s


# --------------------------------------------------------------------------------------------
# fixed patterns

def on(target: SkyCoord, t_start, n_runs: int, run_duration, t0=None) -> Table:
    """All runs pointed at `target`, back to back.

    In:  target SkyCoord, t_start Time (first run), n_runs int, run_duration (s or Quantity),
         t0 (merger time, default t_start).
    Out: Pointings table, strategy "on".
    """
    run_s = _seconds(run_duration)
    c = target.icrs
    return make(np.full(n_runs, c.ra.deg), np.full(n_runs, c.dec.deg), _times(t_start, n_runs, run_s),
                run_s, "on", t0=t0 if t0 is not None else t_start)


def wobble(target: SkyCoord, t_start, n_runs: int, run_duration, offset=0.4,
           angles=(0, 90, 180, 270), t0=None) -> Table:
    """Wobble runs at `offset` deg from `target`, cycling through position angles.

    The position angle is measured from north through east (0 = +Dec, 90 = +RA).
    In:  target SkyCoord, t_start Time, n_runs int, run_duration (s), offset (deg),
         angles (deg, cycled), t0 (merger time, default t_start).
    Out: Pointings table, strategy "wobble". Extra column `wobble_angle` (deg).
    """
    run_s = _seconds(run_duration)
    pa = np.array([angles[i % len(angles)] for i in range(n_runs)], dtype=float)
    c = target.icrs.directional_offset_by(pa * u.deg, np.full(n_runs, offset) * u.deg)
    return make(c.ra.deg, c.dec.deg, _times(t_start, n_runs, run_s), run_s, "wobble",
                t0=t0 if t0 is not None else t_start, wobble_angle=pa)


def grid(center: SkyCoord, t_start, n_side: int, spacing, run_duration, t0=None) -> Table:
    """Square n_side x n_side grid of pointings, `spacing` deg apart on the tangent plane.

    Runs go row by row (south to north), alternating direction (boustrophedon), so that
    consecutive pointings are neighbours.
    In:  center SkyCoord, t_start Time, n_side int, spacing (deg), run_duration (s),
         t0 (merger time, default t_start).
    Out: Pointings table, strategy "grid", n_side**2 rows.
    """
    run_s = _seconds(run_duration)
    k = (np.arange(n_side) - (n_side - 1) / 2) * float(spacing)
    dx, dy = [], []
    for j, y in enumerate(k):
        row = k if j % 2 == 0 else k[::-1]
        dx.extend(row)
        dy.extend([y] * n_side)
    frame = SkyOffsetFrame(origin=center.icrs)
    c = SkyCoord(lon=np.array(dx) * u.deg, lat=np.array(dy) * u.deg, frame=frame).icrs
    n = n_side * n_side
    return make(c.ra.deg, c.dec.deg, _times(t_start, n, run_s), run_s, "grid",
                t0=t0 if t0 is not None else t_start)


# --------------------------------------------------------------------------------------------
# tilepy

# tilepy configuration for LST-1, 2D (GW probability only) tiling. Keys are tilepy's .ini keys.
# Defaults follow tilepy's examples/config/FollowupParameters_LST.ini except where noted.
TILEPY_LST1 = {
    # observatory
    "name": "LST",
    "lat": LST1_SITE["lat"], "lon": LST1_SITE["lon"], "height": LST1_SITE["height"],
    # visibility
    "sunDown": -18,            # Sun altitude limit (deg), astronomical darkness
    "moonDown": -0.5,          # Moon below the horizon (dark time only; no grey time)
    "moonGrey": 65, "moonPhase": 60,          # used only if useGreytime
    "minMoonSourceSeparation": 30, "maxMoonSourceSeparation": 145,
    # operations
    "maxZenith": 60,           # deg; tilepy example uses 70, we keep LST-1 IRFs' better range
    "FOV": 2.5,                # deg, radius of the disc counted per tile (tilepy LST default)
    "maxRuns": 20,             # tiles; tilepy example uses 4
    "maxNights": 1,
    "duration": 20,            # min per run
    "minDuration": 10,         # min, shortest run tilepy accepts at the end of a window
    "useGreytime": False,
    # tiling
    "minProbcut": 0.01,        # tile kept only if it adds more than this; tilepy example 0.02
    "secondRound": False,
    "zenithWeighting": 0.75,
    "percentageMOC": 0.90,     # tile centres drawn from the 90% credible region
    "reducedNside": 128,       # grid of candidate centres (0.46 deg); tilepy example 256
    "HRnside": 512,            # resolution of the probability summed per tile
    "mangrove": False,
    "algorithm": "2D",
    "strategy": "integrated",
    "doPlot": False, "doRank": False,
    "countPrevious": False,
    "distCut": 500, "minimumProbCutForCatalogue": 0.01,
}

_SECTIONS = {
    "observatory": ("name", "lat", "lon", "height"),
    "visibility": ("sunDown", "moonDown", "moonGrey", "moonPhase", "minMoonSourceSeparation",
                   "maxMoonSourceSeparation"),
    "operations": ("maxZenith", "FOV", "maxRuns", "maxNights", "duration", "minDuration",
                   "useGreytime"),
    "tiling": ("minProbcut", "secondRound", "zenithWeighting", "percentageMOC", "reducedNside",
               "HRnside", "mangrove", "algorithm", "strategy", "doPlot", "doRank", "countPrevious",
               "distCut", "minimumProbCutForCatalogue"),
}


def _tilepy_obspar(skymap_path: Path, start: Time, cfg: dict, outdir: Path):
    from tilepy.include.CampaignDefinition import ObservationParameters

    unknown = set(cfg) - {k for keys in _SECTIONS.values() for k in keys}
    if unknown:
        raise ValueError(f"unknown tilepy settings {sorted(unknown)}")
    parser = configparser.ConfigParser()
    parser.optionxform = str
    for sec, keys in _SECTIONS.items():
        parser[sec] = {k: str(cfg[k]) for k in keys}
    parser["general"] = {"downloadMaxRetry": "0", "downloadWaitPeriodRetry": "0"}
    ini = outdir / "tilepy.ini"
    with open(ini, "w") as f:
        parser.write(f)
    obspar = ObservationParameters()
    obspar.add_parsed_args(str(skymap_path), start.utc.datetime, None, None, str(outdir), None,
                           None, Path(skymap_path).stem, "healpix")
    obspar.from_configfile(str(ini))
    return obspar


@contextlib.contextmanager
def _quiet(level=logging.WARNING):
    loggers = [logging.getLogger(n) for n in list(logging.root.manager.loggerDict) if n.startswith("tilepy")]
    old = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(level)
    try:
        yield
    finally:
        for lg, lv in zip(loggers, old):
            lg.setLevel(lv)


def _run_tilepy(skymap_path: Path, t0: Time, start: Time, stop: Time | None, cfg: dict) -> Table:
    import tilepy.include.TilingDetermination  # noqa: F401  (registers its loggers)
    from tilepy.include.MapManagement import SkyMap, create_map_reader
    from tilepy.include.TilingDetermination import PGWinFoV

    with tempfile.TemporaryDirectory() as tmp, _quiet():
        obspar = _tilepy_obspar(skymap_path, start, cfg, Path(tmp))
        raw = create_map_reader(obspar)
        sky = SkyMap(obspar, raw)
        out, _ = PGWinFoV(sky, raw.name_event, obspar, tmp)
    t_start = Time([str(s) for s in out["Time[UTC]"]], format="iso", scale="utc") if len(out) else Time([], format="mjd")
    tab = make(np.asarray(out["RA[deg]"], float), np.asarray(out["DEC[deg]"], float), t_start,
               np.asarray(out["Duration"], float) * 60.0, "tilepy", t0=t0,
               pgw=np.asarray(out["PGW"], float))
    if stop is not None and len(tab):
        tab = tab[tab["t_start"] < stop]
        tab["obs_id"] = np.arange(1, len(tab) + 1)
    tab.meta["tilepy"] = {k: cfg[k] for k in ("FOV", "maxZenith", "duration", "maxRuns", "sunDown",
                                               "moonDown", "useGreytime", "minProbcut", "reducedNside",
                                               "HRnside")}
    tab.meta["window_start"] = start.utc.isot
    return tab


def tilepy_2d(skymap_path, t0, obs_window=None, settings=None, version: int = 1) -> Table:
    """Greedy 2D GW-probability tiling for LST-1 with tilepy (cached).

    tilepy walks through the dark-time run slots of the window (Sun below `sunDown`, Moon below
    `moonDown`), and for each slot points at the visible (zenith < `maxZenith`) candidate centre
    of the 90% region whose disc of radius `FOV` holds the most not-yet-covered probability.
    The slot is kept if that probability exceeds `minProbcut`. Up to `maxRuns` slots.

    tilepy needs a real calendar date: it schedules from `obs_window[0]` (default `t0`), and the
    merger time `t0` is taken as given (for simulated cases it is already shifted, see
    `uuls.eventtime`). Nothing about the skymap is rotated.

    In:  skymap_path (HEALPix FITS, flat or multi-order), t0 (Time, merger), obs_window
         (None, or (start, stop) Times; stop may be None; maxNights is raised to cover stop),
         settings (dict overriding TILEPY_LST1; defaults: FOV 2.5 deg radius, maxZenith 60 deg,
         duration 20 min, maxRuns 20, maxNights 1, sunDown -18 deg, moonDown -0.5 deg, no grey
         time, minProbcut 0.01, reducedNside 128, HRnside 512, percentageMOC 0.9).
    Out: Pointings table, strategy "tilepy", extra column `pgw`; meta["tilepy"] has the settings.
    """
    t0 = Time(t0).utc
    start, stop = (t0, None) if obs_window is None else obs_window
    start = Time(start).utc
    stop = None if stop is None else Time(stop).utc
    cfg = dict(TILEPY_LST1)
    cfg.update(settings or {})
    if stop is not None:
        cfg["maxNights"] = max(int(cfg["maxNights"]), math.ceil((stop - start).to_value(u.day)))
    params = {"skymap": Path(skymap_path), "t0": t0.isot, "start": start.isot,
              "stop": None if stop is None else stop.isot, "settings": cfg, "version": version}
    return cache.cached("tilepy_2d", params, lambda: _run_tilepy(Path(skymap_path), t0, start, stop, cfg))
