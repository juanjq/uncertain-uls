"""Simulated LST-1 DL3 from a Pointings table, by simulst in its own environment.

    dl3_dir = simulate(pointings, nsb=0.14, source=PowerLawSource(ra, dec, 1e-11), seed=0)
    DataStore.from_dir(dl3_dir)

simulst needs gammapy 1.3 and cannot share a process with this package (gammapy 2.1). `simulate` writes
a job file and runs `scripts/simulst_job.py` with the simulst python (`paths.SIMULST_PYTHON`) in offline
mode (`paths.SIMULST_IRF_ROOT`, `paths.SIMULST_LIBRARY_ROOT`). The result is cached: a second call with the
same pointings, NSB, source and seed returns the same folder without running simulst.

What the folder holds (per run): `dl3_LST-1.Run<obs_id>.fits` (events, GTI, pointing, full-enclosure IRFs
padded to 0-1.2 deg offset; events out to 2.5 deg), `bkg_Run<obs_id>.fits` (the background model simulst
used, registered in hdu-index as the run's `bkg` IRF), `hdu-index.fits.gz`, `obs-index.fits.gz`, and
`runs.json` (what was simulated per run: event count, whether the source was included, header keywords).
Run k is drawn by simulst from `SeedSequence([seed, obs_id_k])`: runs with different obs_id are independent,
and the same (seed, obs_id) gives the same background events with or without a source.
"""
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from astropy.table import Table
from astropy.time import Time

from . import cache, paths

VERSION = "1"  # bump when the job script or its output changes
JOB_SCRIPT = paths.REPO / "scripts" / "simulst_job.py"
NSB_LEVELS = (0.07, 0.14, 0.22, 0.38, 0.50)


@dataclass(frozen=True)
class PowerLawSource:
    """Point source with dN/dE = amplitude (E / 1 TeV)^-index.

    ra, dec: deg (ICRS); amplitude: cm-2 s-1 TeV-1 at 1 TeV; index: positive photon index.
    """
    ra: float
    dec: float
    amplitude: float
    index: float = 2.0


class SimulstRefusal(ValueError):
    """simulst refused a run (OutOfCoverageError: zenith >= 60 deg, NSB level, library bin, source offset ...)."""

    def __init__(self, obs_id: int, kind: str, message: str, run: dict):
        self.obs_id, self.kind, self.reason, self.run = obs_id, kind, message, run
        super().__init__(f"simulst refused run obs_id={obs_id} (t_start {run.get('t_start')}, "
                         f"ra {run.get('ra'):.3f}, dec {run.get('dec'):.3f}, {run.get('duration'):.0f} s): "
                         f"{kind}: {message}")


def _runs(pointings: Table) -> list[dict]:
    """Pointings rows as plain dicts (obs_id int, ra/dec deg, t_start isot UTC, duration s)."""
    for c in ("obs_id", "ra", "dec", "t_start", "duration"):
        if c not in pointings.colnames:
            raise ValueError(f"pointings has no column {c!r} (columns: {pointings.colnames})")
    t_start = pointings["t_start"]
    t_start = t_start if isinstance(t_start, Time) else Time(list(t_start), scale="utc")
    runs = []
    for row, t in zip(pointings, t_start):
        def val(c, unit):
            v = row[c]
            return float(v.to_value(unit)) if hasattr(v, "to_value") else float(v)
        runs.append({"obs_id": int(row["obs_id"]), "ra": val("ra", "deg"), "dec": val("dec", "deg"),
                     "t_start": t.utc.isot, "duration": val("duration", "s")})
    ids = [r["obs_id"] for r in runs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"obs_id must be unique within the pointings, got {ids}")
    if not runs:
        raise ValueError("pointings is empty")
    return runs


def _env() -> dict:
    env = dict(os.environ)
    env.update(SIMULST_IRF_ROOT=str(paths.SIMULST_IRF_ROOT), SIMULST_LIBRARY_ROOT=str(paths.SIMULST_LIBRARY_ROOT),
               PYTHONPATH=str(paths.SIMULST_REPO), OMP_NUM_THREADS=env.get("OMP_NUM_THREADS", "1"))
    env.pop("PYTHONHOME", None)
    return env


def _check_output(dl3_dir: Path, runs: list[dict]) -> None:
    for name in ("hdu-index.fits.gz", "obs-index.fits.gz", "runs.json"):
        if not (dl3_dir / name).is_file():
            raise RuntimeError(f"simulst job finished but {dl3_dir / name} is missing")
    obs = Table.read(dl3_dir / "obs-index.fits.gz")
    want, got = sorted(r["obs_id"] for r in runs), sorted(int(i) for i in obs["OBS_ID"])
    if want != got:
        raise RuntimeError(f"obs-index lists runs {got}, the pointings ask for {want}")
    for r in runs:
        if not (dl3_dir / f"dl3_LST-1.Run{r['obs_id']:05d}.fits").is_file():
            raise RuntimeError(f"run {r['obs_id']}: DL3 file missing in {dl3_dir}")


def simulate(pointings: Table, nsb: float = 0.14, source: PowerLawSource | None = None, seed: int = 0,
             far_source: str = "skip", verbose: bool = True) -> Path:
    """Simulate every run of `pointings` with simulst and return the DL3 folder.

    pointings: Pointings table (obs_id, ra, dec deg, t_start Time UTC, duration s), see docs/interfaces.md.
    nsb: NSB level of the IRFs and background library, one of 0.07, 0.14, 0.22, 0.38, 0.50.
    source: optional point source; None gives background only.
    seed: run k uses simulst's streams SeedSequence([seed, obs_id_k]).
    far_source: a run whose pointing is farther than simulst's IRF reach (1.1 deg) from the source is
        simulated without it ("skip", the source gives essentially no events there) or refused ("refuse").
    Returns: Path of the DL3 folder (hdu-index, obs-index, one DL3 and one background file per run). Cached.
    Raises SimulstRefusal (a ValueError) naming the run when simulst refuses one; nothing is cached then.
    """
    if not any(np.isclose(nsb, lv) for lv in NSB_LEVELS):
        raise ValueError(f"nsb {nsb} is not one of the IRF levels {NSB_LEVELS}")
    if far_source not in ("skip", "refuse"):
        raise ValueError(f"far_source must be 'skip' or 'refuse', got {far_source!r}")
    runs = _runs(pointings)
    src = None if source is None else {k: float(v) for k, v in asdict(source).items()}
    params = {"version": VERSION, "runs": runs, "nsb": float(nsb), "source": src, "seed": int(seed),
              "far_source": far_source if src is not None else None, "production": "2025",
              "library": paths.SIMULST_LIBRARY_ROOT.name}
    folder = cache.cache_dir("simulst_dl3", params)
    dl3_dir = folder / "dl3"
    if cache.is_done(folder):
        if verbose:
            print(f"[cache] simulst_dl3/{folder.name}: loaded ({len(runs)} runs)")
        return dl3_dir

    if dl3_dir.exists():  # left by an interrupted or refused attempt
        shutil.rmtree(dl3_dir)
    error_file = folder / "error.json"
    error_file.unlink(missing_ok=True)
    job = {"out_dir": str(dl3_dir), "error_file": str(error_file), "nsb": float(nsb), "seed": int(seed),
           "production": "2025", "source": src, "far_source": far_source, "runs": runs}
    job_file = folder / "job.json"
    job_file.write_text(json.dumps(job, indent=2))
    log_file = folder / "simulst.log"
    if not paths.SIMULST_PYTHON.is_file():
        raise FileNotFoundError(f"simulst python not found at {paths.SIMULST_PYTHON} (set UULS_SIMULST_PYTHON)")

    t = time.perf_counter()
    with open(log_file, "w") as log:
        proc = subprocess.run([str(paths.SIMULST_PYTHON), str(JOB_SCRIPT), str(job_file)], env=_env(),
                              stdout=log, stderr=subprocess.STDOUT, cwd=str(folder))
    if proc.returncode != 0:
        shutil.rmtree(dl3_dir, ignore_errors=True)
        if error_file.is_file():
            err = json.loads(error_file.read_text())
            raise SimulstRefusal(err["obs_id"], err["type"], err["message"], err["run"])
        tail = "".join(log_file.read_text().splitlines(keepends=True)[-30:])
        raise RuntimeError(f"simulst job failed (status {proc.returncode}), log {log_file}:\n{tail}")
    _check_output(dl3_dir, runs)
    cache.mark_done(folder)
    if verbose:
        print(f"[cache] simulst_dl3/{folder.name}: simulated {len(runs)} runs in {time.perf_counter() - t:.0f} s")
    return dl3_dir


def run_summary(dl3_dir: Path) -> Table:
    """Per-run record written by the job (obs_id, n_events, source simulated or not, simulst header keywords)."""
    rows = json.loads((Path(dl3_dir) / "runs.json").read_text())
    return Table(rows=[{k: v for k, v in r.items() if k != "IRFSRC"} for r in rows])
