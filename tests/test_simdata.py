"""uuls.simdata: simulst runs in its own env through the job script. Each simulation is cached in the
project cache, so only the first session pays for it (two 5-minute runs, about 30 s)."""
import json
import time

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

from uuls import paths, simdata

pytestmark = pytest.mark.simulst

TARGET = SkyCoord(7.53, 45.81, unit="deg")       # culminates at 17 deg zenith from La Palma
T_START = Time("2025-10-20T23:00:00", scale="utc")  # dark, Moon down, target at 71 deg altitude
SOURCE = simdata.PowerLawSource(ra=7.53, dec=45.81, amplitude=5e-11, index=2.0)  # about 1.5 Crab at 1 TeV


def one_run(obs_id=1, t_start=T_START, dec_offset=0.4, duration=300.0):
    return Table({"obs_id": [obs_id], "ra": [TARGET.ra.deg], "dec": [TARGET.dec.deg + dec_offset],
                  "t_start": Time([t_start.isot], scale="utc"), "duration": [duration]})


def _have_simulst():
    try:
        simdata.simulst_python()
    except FileNotFoundError:
        return False
    return paths.SIMULST_IRF_ROOT.is_dir() and paths.SIMULST_LIBRARY_ROOT.is_dir()


needs_simulst = pytest.mark.skipif(not _have_simulst(), reason="simulst env or offline IRFs missing")


@needs_simulst
def test_background_only_run_is_a_readable_dl3_and_cached():
    dl3 = simdata.simulate(one_run(), nsb=0.14, seed=3)
    hdu = Table.read(dl3 / "hdu-index.fits.gz")
    assert set(hdu["HDU_TYPE"]) >= {"events", "gti", "aeff", "edisp", "psf", "bkg"}
    with fits.open(dl3 / "dl3_LST-1.Run00001.fits") as f:
        ev, h = f["EVENTS"].data, f["EVENTS"].header
        gti = f["GTI"].data
    assert h["OBS_ID"] == 1 and h["SRCMODEL"] == "none" and len(ev) > 1000
    assert np.isclose(gti["STOP"][0] - gti["START"][0], 300.0)
    pnt = SkyCoord(h["RA_PNT"], h["DEC_PNT"], unit="deg")
    assert SkyCoord(ev["RA"], ev["DEC"], unit="deg").separation(pnt).deg.max() <= 2.5
    summary = simdata.run_summary(dl3)
    assert list(summary["obs_id"]) == [1] and not summary["source"][0]
    assert summary["production"][0] == "2025_2p5" and summary["MCPROD"][0] == "2025_2p5"
    assert np.isclose(summary["IRFORGHI"][0], 2.5)
    t = time.perf_counter()
    assert simdata.simulate(one_run(), nsb=0.14, seed=3) == dl3  # second call: from the cache
    assert time.perf_counter() - t < 2


@needs_simulst
def test_source_adds_events_on_top_of_the_same_background():
    """Same seed and obs_id: simulst draws the same background, so the extra events are the source."""
    bkg = simdata.simulate(one_run(), nsb=0.14, seed=3)
    src = simdata.simulate(one_run(), nsb=0.14, seed=3, source=SOURCE)
    assert src != bkg
    assert simdata.run_summary(src)["source"][0]

    def near(d):
        ev = fits.getdata(d / "dl3_LST-1.Run00001.fits", "EVENTS")
        sel = ev["ENERGY"] > 0.3
        return np.sum(SkyCoord(ev["RA"][sel], ev["DEC"][sel], unit="deg").separation(TARGET).deg < 0.2)

    n_bkg, n_src = near(bkg), near(src)
    assert n_src - n_bkg > 5 * np.sqrt(max(n_bkg, 1))
    h = fits.getheader(src / "dl3_LST-1.Run00001.fits", "EVENTS")
    assert "PowerLawSpectralModel" in h["SRCMODEL"]


@needs_simulst
def test_refused_run_names_the_run():
    """At 16 UTC the pointing is at 4.5 deg altitude: simulst refuses it (zenith >= 60 deg)."""
    low = one_run(obs_id=7, t_start=Time("2025-10-20T16:00:00", scale="utc"))
    with pytest.raises(simdata.SimulstRefusal, match="obs_id=7"):
        simdata.simulate(low, nsb=0.14, seed=3)


def test_pointings_are_checked_before_running():
    two = one_run()
    two.add_row(two[0])
    with pytest.raises(ValueError, match="unique"):
        simdata.simulate(two)
    with pytest.raises(ValueError, match="NSB|nsb"):
        simdata.simulate(one_run(), nsb=0.1)
    with pytest.raises(ValueError, match="production"):
        simdata.simulate(one_run(), production="2024")
    with pytest.raises(ValueError, match="duration"):
        simdata.simulate(one_run()[["obs_id", "ra", "dec", "t_start"]])


def test_job_file_round_trip_of_runs():
    table = one_run(obs_id=12)
    table["duration"] = [10.0] * u.min  # a Quantity column is converted to seconds
    runs = simdata._runs(table)
    assert runs == [{"obs_id": 12, "ra": TARGET.ra.deg, "dec": TARGET.dec.deg + 0.4,
                     "t_start": "2025-10-20T23:00:00.000", "duration": 600.0}]
    json.dumps(runs)
