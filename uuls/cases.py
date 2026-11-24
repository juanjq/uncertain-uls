"""Mock cases: one YAML file per case under configs/cases/, parsed into a `Case`.

A case fixes the GW event, the merger time, the kind of data and the observation plan. It
does not simulate or reduce anything (uuls.simdata and uuls.reduce do that).

YAML layout (all keys shown; see configs/cases/*.yaml):

    name: sim_wobble
    gw:
      skymap: S240615dg.fits        # file in data/gw/skymaps/ (or an absolute path)
      pe: S240615dg                 # name for uuls.gwpe.load_samples, or null
    t0:
      mode: observable              # real: DATE-OBS of the skymap; observable: eventtime
      after: "2026-01-01T12:00:00"  # observable only: search starts here (UTC)
      max_zenith: 40                # observable only, deg
      target: peak                  # observable only: "peak" or {ra: .., dec: ..}
    data:
      kind: simulst                 # real | simulst
      nsb: 0.14                     # simulst only
      seed: 1                       # simulst only
    pointings:
      strategy: wobble              # on | wobble | grid | tilepy | real
      start_delay: 60               # s after t0 of the first run (on/wobble/grid/tilepy)
      ...                           # parameters of the strategy (see Case.pointings)
    analysis:
      center: auto                  # auto (mean pointing direction) or {ra: .., dec: ..}
      width: auto                   # auto, a number (deg, square) or [w_ra, w_dec]
      margin: 2.5                   # auto width only: deg added on each side

    load_case(path) -> Case
    case.t0() -> Time,  case.target() -> SkyCoord,  case.pointings() -> Pointings table,
    case.analysis_center() -> SkyCoord,  case.analysis_width() -> (w_ra, w_dec) deg
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import astropy.units as u
import numpy as np
import yaml
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from astropy.io import fits
from astropy.time import Time

from . import eventtime, paths
from . import pointings as pt

T0_MODES = ("real", "observable")
DATA_KINDS = ("real", "simulst")


def _path(p, base: Path) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else base / p


def skymap_time(path) -> Time:
    """Merger time from a skymap header (DATE-OBS, UTC). In: FITS path. Out: Time."""
    with fits.open(path) as h:
        for hdu in h:
            if "DATE-OBS" in hdu.header:
                return Time(hdu.header["DATE-OBS"], scale="utc")
    raise KeyError(f"no DATE-OBS in {path}")


def _coord(d) -> SkyCoord:
    return SkyCoord(float(d["ra"]) * u.deg, float(d["dec"]) * u.deg, frame="icrs")


@dataclass
class Case:
    """One mock case. Built by `load_case`; fields mirror the YAML sections."""
    name: str
    skymap: Path
    pe: str | None
    t0_mode: str
    t0_after: Time | None
    max_zenith: float
    target_spec: str | dict
    data_kind: str
    nsb: float | None
    seed: int | None
    pointing: dict
    analysis: dict
    source: Path | None = None
    raw: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.t0_mode not in T0_MODES:
            raise ValueError(f"{self.name}: t0.mode must be one of {T0_MODES}")
        if self.data_kind not in DATA_KINDS:
            raise ValueError(f"{self.name}: data.kind must be one of {DATA_KINDS}")
        if self.pointing.get("strategy") not in pt.STRATEGIES:
            raise ValueError(f"{self.name}: pointings.strategy must be one of {pt.STRATEGIES}")
        if self.t0_mode == "observable" and self.t0_after is None:
            raise ValueError(f"{self.name}: t0.after is required with mode 'observable'")
        if self.data_kind == "simulst" and (self.nsb is None or self.seed is None):
            raise ValueError(f"{self.name}: simulst data needs data.nsb and data.seed")

    def target(self) -> SkyCoord:
        """Direction the time and the fixed patterns refer to: skymap peak or a given (ra, dec)."""
        if self.target_spec == "peak":
            return self._peak
        return _coord(self.target_spec)

    @cached_property
    def _peak(self) -> SkyCoord:
        return eventtime.peak_direction(self.skymap)

    def t0(self) -> Time:
        """Merger time. real: from the skymap header; observable: eventtime.observable_time."""
        if self.t0_mode == "real":
            return skymap_time(self.skymap)
        return self._t0_observable

    @cached_property
    def _t0_observable(self) -> Time:
        return eventtime.observable_time(self.target(), self.t0_after, max_zenith=self.max_zenith)

    def pointings(self):
        """Pointings table of the case.

        on:     n_runs, run_duration (s)                      at the target
        wobble: n_runs, run_duration, offset (deg), angles    around the target
        grid:   n_side, spacing (deg), run_duration           around the target
        tilepy: settings (dict overriding pointings.TILEPY_LST1), window (h after the first
                run, optional)
        real:   file (ECSV, relative to the YAML). Until the real runs are converted, the file
                does not exist and this raises FileNotFoundError.
        All but `real` start at t0 + start_delay (s, default 0).
        """
        p = dict(self.pointing)
        strategy = p.pop("strategy")
        t0 = self.t0()
        if strategy == "real":
            f = _path(p["file"], self.source.parent if self.source else paths.CONFIGS / "cases")
            if not f.exists():
                raise FileNotFoundError(
                    f"{self.name}: real pointings not available yet; expected {f} "
                    "(a Pointings ECSV built from the observation index of the real DL3 runs)")
            return pt.read(f)
        start = t0 + float(p.pop("start_delay", 0.0)) * u.s
        if strategy == "on":
            return pt.on(self.target(), start, int(p["n_runs"]), float(p["run_duration"]), t0=t0)
        if strategy == "wobble":
            return pt.wobble(self.target(), start, int(p["n_runs"]), float(p["run_duration"]),
                             offset=float(p.get("offset", 0.4)),
                             angles=tuple(p.get("angles", (0, 90, 180, 270))), t0=t0)
        if strategy == "grid":
            return pt.grid(self.target(), start, int(p["n_side"]), float(p["spacing"]),
                           float(p["run_duration"]), t0=t0)
        window = None
        if p.get("window") is not None:
            window = (start, start + float(p["window"]) * u.h)
        elif start != t0:
            window = (start, None)
        return pt.tilepy_2d(self.skymap, t0, window, p.get("settings") or {})

    def analysis_center(self, pointings=None) -> SkyCoord:
        """analysis.center, or the mean direction of the pointings when 'auto'."""
        c = self.analysis.get("center", "auto")
        if c != "auto":
            return _coord(c)
        tab = self.pointings() if pointings is None else pointings
        xyz = pt.coords(tab).cartesian.xyz.value.mean(axis=1)
        return SkyCoord(np.degrees(np.arctan2(xyz[1], xyz[0])) % 360 * u.deg,
                        np.degrees(np.arctan2(xyz[2], np.hypot(xyz[0], xyz[1]))) * u.deg, frame="icrs")

    def analysis_width(self, pointings=None) -> tuple[float, float]:
        """(width along RA, width along Dec) in deg, as WcsGeom.create(width=...) takes it.

        analysis.width given: a number (square) or [w_ra, w_dec]. 'auto': on the tangent plane
        at analysis_center, 2 (largest offset of a pointing + margin) per axis, rounded up.
        """
        w = self.analysis.get("width", "auto")
        if w != "auto":
            w = np.broadcast_to(np.asarray(w, dtype=float), (2,))
            return float(w[0]), float(w[1])
        tab = self.pointings() if pointings is None else pointings
        off = pt.coords(tab).transform_to(SkyOffsetFrame(origin=self.analysis_center(tab)))
        m = float(self.analysis.get("margin", 2.5))
        return (float(np.ceil(2 * (np.abs(off.lon.wrap_at(180 * u.deg).deg).max() + m))),
                float(np.ceil(2 * (np.abs(off.lat.deg).max() + m))))

    def summary(self) -> dict:
        """Plain dict for printing: name, event, t0 rule, data, strategy."""
        return {"name": self.name, "skymap": self.skymap.name, "pe": self.pe,
                "t0": f"{self.t0_mode}" + (f" after {self.t0_after.isot}" if self.t0_after is not None else ""),
                "data": self.data_kind + (f" (nsb {self.nsb}, seed {self.seed})" if self.data_kind == "simulst" else ""),
                "strategy": self.pointing["strategy"]}


def load_case(path) -> Case:
    """Parse a case YAML. In: path (or a case name in configs/cases/). Out: Case."""
    path = Path(path)
    if not path.suffix:
        path = paths.CONFIGS / "cases" / f"{path}.yaml"
    d = yaml.safe_load(path.read_text())
    gw, t0, data = d["gw"], d.get("t0", {}), d.get("data", {})
    after = t0.get("after")
    return Case(
        name=d["name"],
        skymap=_path(gw["skymap"], paths.GW_SKYMAPS),
        pe=gw.get("pe"),
        t0_mode=t0.get("mode", "real"),
        t0_after=None if after is None else Time(str(after), scale="utc"),
        max_zenith=float(t0.get("max_zenith", 40.0)),
        target_spec=t0.get("target", "peak"),
        data_kind=data.get("kind", "simulst"),
        nsb=data.get("nsb"),
        seed=data.get("seed"),
        pointing=dict(d["pointings"]),
        analysis=dict(d.get("analysis") or {"center": "auto", "width": "auto"}),
        source=path,
        raw=d,
    )


def all_cases() -> list[Case]:
    """Every case in configs/cases/, sorted by name."""
    return [load_case(p) for p in sorted((paths.CONFIGS / "cases").glob("*.yaml"))]
