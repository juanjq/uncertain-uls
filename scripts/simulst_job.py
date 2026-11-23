"""Simulate a list of LST-1 runs with simulst into one DL3 folder.

Run by the simulst python (gammapy 1.3), never by the main env:

    SIMULST_IRF_ROOT=... SIMULST_LIBRARY_ROOT=... PYTHONPATH=<simulst repo> \
        <simulst env>/bin/python scripts/simulst_job.py job.json

`uuls.simdata.simulate` writes the job file and runs this. The script does not import uuls.

Job file (JSON):

    {
      "out_dir": "/path/to/dl3",          # created; must not already hold the runs
      "error_file": "/path/to/error.json", # written only when a run is refused
      "nsb": 0.14,
      "seed": 0,                           # simulst draws run k from SeedSequence([seed, obs_id_k])
      "production": "2025",
      "source": null | {"ra": deg, "dec": deg, "amplitude": cm-2 s-1 TeV-1 at 1 TeV, "index": 2.0},
      "far_source": "skip" | "refuse",     # source beyond simulst's IRF reach (1.1 deg) from a pointing
      "runs": [{"obs_id": 1, "ra": deg, "dec": deg, "t_start": "2025-01-01T00:00:00", "duration": 1200.0}, ...]
    }

Output, in out_dir: dl3_LST-1.Run<obs_id>.fits per run, bkg_Run<obs_id>.fits (the background model of the
run as simulst used it, mid-run slice, run norm applied), hdu-index.fits.gz (with the background rows added,
HDU_TYPE "bkg", so gammapy reads it as the run's background IRF), obs-index.fits.gz and runs.json (one entry
per run: file, whether the source was simulated, the simulst header keywords).

Exit status: 0 on success; 3 when simulst refuses a run (OutOfCoverageError, or a ValueError raised while
checking the run), after writing error_file with the run's obs_id and the message; anything else is a crash.
"""
import json
import sys
import time
import traceback
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from gammapy.modeling.models import PointSpatialModel, PowerLawSpectralModel, SkyModel

from simulst.background.library import BackgroundLibrary
from simulst.errors import OutOfCoverageError
from simulst.sample import MAX_SOURCE_OFFSET
from simulst.simulate import simulate_pointing

REFUSED = 3
HEADER_KEYS = ("SEED", "BKGNORM", "NSB", "LIBVERS", "BKGPRED", "BKGEXTRP", "BKGLOWER", "BKGDROP", "BKGX2",
               "IRFPAD", "IRFOFFHI", "IRFSRC", "OFFMAX", "LIVETIME", "ONTIME")


def source_model(src: dict) -> SkyModel:
    spectral = PowerLawSpectralModel(index=src["index"], amplitude=src["amplitude"] * u.Unit("cm-2 s-1 TeV-1"),
                                     reference=1 * u.TeV)
    spatial = PointSpatialModel(lon_0=src["ra"] * u.deg, lat_0=src["dec"] * u.deg, frame="icrs")
    return SkyModel(spectral, spatial, name="source")


def add_background_rows(out_dir: Path, obs_ids: list[int]) -> None:
    """Append one HDU_TYPE "bkg" row per run to hdu-index.fits.gz, pointing to bkg_Run<obs_id>.fits."""
    path = out_dir / "hdu-index.fits.gz"
    with fits.open(path) as hdul:
        header = hdul[1].header.copy()
    table = Table.read(path, hdu=1)
    for obs_id in obs_ids:
        name = f"bkg_Run{obs_id:05d}.fits"
        with fits.open(out_dir / name) as h:
            hdu_name = h[1].name
            size = h[1].filebytes()
        row = {c: table[c][0] for c in table.colnames}  # copy the column types of the EVENTS row
        row.update(OBS_ID=obs_id, HDU_TYPE="bkg", HDU_CLASS="bkg_3d", FILE_DIR=".", FILE_NAME=name,
                   HDU_NAME=hdu_name)
        if "SIZE" in row:
            row["SIZE"] = size
        table.add_row([row[c] for c in table.colnames])
    hdu = fits.table_to_hdu(table)
    for k in ("HDUCLASS", "HDUDOC", "HDUVERS", "HDUCLAS1", "HDUCLAS2", "BASE_DIR"):
        if k in header:
            hdu.header[k] = header[k]
    hdu.name = header.get("EXTNAME", "HDU_INDEX")
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)


def main(job_path: str) -> int:
    job = json.loads(Path(job_path).read_text())
    out_dir = Path(job["out_dir"])
    error_file = Path(job["error_file"])
    nsb, seed, production = float(job["nsb"]), int(job["seed"]), job.get("production", "2025")
    src = job.get("source")
    far_source = job.get("far_source", "skip")
    library = BackgroundLibrary.read(production=production)
    print(f"library {library.version}, {len(job['runs'])} runs, nsb {nsb}, seed {seed}", flush=True)

    summary = []
    for run in job["runs"]:
        obs_id = int(run["obs_id"])
        pointing = SkyCoord(run["ra"] * u.deg, run["dec"] * u.deg, frame="icrs")
        t_start = Time(run["t_start"], scale="utc")
        model = None
        if src is not None:
            sep = pointing.separation(SkyCoord(src["ra"] * u.deg, src["dec"] * u.deg, frame="icrs"))
            if sep <= MAX_SOURCE_OFFSET or far_source == "refuse":
                model = source_model(src)  # simulst refuses it beyond MAX_SOURCE_OFFSET
            else:
                print(f"run {obs_id}: source {sep.deg:.2f} deg from the pointing, beyond {MAX_SOURCE_OFFSET}: "
                      "simulated without source", flush=True)
        t = time.perf_counter()
        try:
            paths = simulate_pointing(t_start, float(run["duration"]) * u.s, out_dir, pointing_radec=pointing,
                                      nsb=nsb, source_model=model, production=production, library=library,
                                      seed=seed, obs_id=obs_id, write_bkg_model=True, product="events")
        except (OutOfCoverageError, ValueError) as e:  # simulst's refusals, raised before the run is written
            error_file.write_text(json.dumps({"obs_id": obs_id, "type": type(e).__name__, "message": str(e),
                                              "run": run, "traceback": traceback.format_exc()}, indent=2))
            print(f"run {obs_id} refused: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            return REFUSED
        h = fits.getheader(paths[0], "EVENTS")
        n_events = fits.getheader(paths[0], "EVENTS")["NAXIS2"]
        summary.append({"obs_id": obs_id, "file": paths[0].name, "bkg_file": f"bkg_Run{obs_id:05d}.fits",
                        "source": model is not None, "n_events": int(n_events),
                        "seconds": round(time.perf_counter() - t, 1),
                        **{k: h.get(k) for k in HEADER_KEYS}})
        print(f"run {obs_id}: {n_events} events, {summary[-1]['seconds']} s", flush=True)

    add_background_rows(out_dir, [r["obs_id"] for r in summary])
    (out_dir / "runs.json").write_text(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: simulst_job.py job.json")
    sys.exit(main(sys.argv[1]))
