"""uuls.gwpe: reading the standardized S240615dg samples, and standardize on a fake raw release."""
import csv

import h5py
import numpy as np
import pytest

from uuls import gwpe, paths


@pytest.fixture(scope="module")
def samples():
    return gwpe.load_samples("S240615dg")


def _index_row(name):
    with open(paths.GW_PE / "index.csv", newline="") as fh:
        return next(r for r in csv.DictReader(fh) if r["superevent_id"] == name)


def test_columns_and_units(samples):
    for c in ("ra", "dec", "luminosity_distance", "theta_jn", "theta_v"):
        assert c in samples.colnames
    assert str(samples["ra"].unit) == "deg" and str(samples["luminosity_distance"].unit) == "Mpc"
    assert len(samples) == 20000 and len(samples.meta["analyses"]) == 4


def test_ranges(samples):
    assert np.all((samples["ra"] >= 0) & (samples["ra"] < 360))
    assert np.all(np.abs(samples["dec"]) <= 90)
    assert np.all((samples["theta_jn"] >= 0) & (samples["theta_jn"] <= 180))
    assert np.all((samples["theta_v"] >= 0) & (samples["theta_v"] <= 90))
    assert np.all(samples["luminosity_distance"] > 0)


def test_matches_index(samples):
    row = _index_row("S240615dg")
    assert np.median(samples["luminosity_distance"]) == pytest.approx(float(row["distance_Mpc_median"]), rel=1e-3)
    assert np.median(samples["theta_v"]) == pytest.approx(float(row["viewing_angle_deg_median"]), abs=0.06)
    assert samples.meta["trigger_time_gps"] == pytest.approx(float(row["trigger_time_gps"]), abs=1e-3)


def test_sky_position_near_skymap_maximum(samples):
    # the alert skymap peaks at (7.7, 45.7)
    assert np.median(samples["ra"]) == pytest.approx(7.7, abs=3)
    assert np.median(samples["dec"]) == pytest.approx(45.7, abs=3)


def test_theta_v_given_distance(samples):
    edges = np.percentile(samples["luminosity_distance"], [0, 25, 50, 75, 100])
    edges[-1] += 1
    parts = gwpe.theta_v_given_distance(samples, edges)
    assert len(parts) == 4 and sum(len(p) for p in parts) == len(samples)


def _fake_release(path, sizes):
    """A raw GWTC release with the group layout standardize reads, tiny and random."""
    rng = np.random.default_rng(1)
    dtype = [("ra", "f8"), ("dec", "f8"), ("luminosity_distance", "f8"), ("theta_jn", "f8"),
             ("mass_1_source", "f8")]
    with h5py.File(path, "w") as f:
        for label, n in sizes.items():
            ps = np.zeros(n, dtype=dtype)
            ps["ra"] = rng.uniform(0, 2 * np.pi, n)
            ps["dec"] = rng.uniform(-1, 1, n)
            ps["luminosity_distance"] = rng.uniform(100, 1000, n)
            ps["theta_jn"] = rng.uniform(0, np.pi, n)
            ps["mass_1_source"] = rng.uniform(1, 2, n)
            g = f.create_group(label)
            g["posterior_samples"] = ps
            g["priors/samples/luminosity_distance"] = rng.uniform(10, 2000, 50)
            g["priors/samples/theta_jn"] = rng.uniform(0, np.pi, 50)
            g["meta_data/meta_data/approximant"] = np.array([label.split(":")[1].encode()])
            g["config_file/config/trigger-time"] = np.array([b"1402486598.7"])
        f.create_group("version")
    return path


def test_standardize_structure(tmp_path):
    raw = _fake_release(tmp_path / "IGWN-GWTC5p0-x-GW240615_113620-combined_PEDataRelease.hdf5",
                        {"C00:A": 300, "C00:B": 200})
    out = gwpe.standardize(raw, "S000000x", None, n_per_analysis=250, out_dir=tmp_path / "std")
    assert out.name == "S000000x.h5"
    t = gwpe.load_samples(out)
    assert len(t) == 400                       # capped at the smaller analysis, 200 each
    assert np.bincount(t["analysis"]).tolist() == [200, 200]
    assert t.meta["gw_name"] == "GW240615_113620" and t.meta["catalog"] == "GWTC-5.0"
    assert t.meta["trigger_time_gps"] == pytest.approx(1402486598.7)
    assert np.all(t["theta_v"] <= 90) and t["ra"].max() > 6.3   # degrees, not radians
    with h5py.File(out) as f:
        assert f["prior/luminosity_distance"].shape == (100,)
        assert f["analyses/C00:A"].attrs["approximant"] == "A"
    assert (tmp_path / "std" / "index.csv").exists()


def test_standardize_one_label_and_errors(tmp_path):
    raw = _fake_release(tmp_path / "IGWN-GWTC5p0-x-GW240615_113620-combined_PEDataRelease.hdf5",
                        {"C00:A": 30, "C00:B": 20})
    out = gwpe.standardize(raw, "S1", "C00:B", out_dir=tmp_path)
    assert gwpe.load_samples(out).meta["analyses"] == ["C00:B"]
    mtime = out.stat().st_mtime_ns
    assert gwpe.standardize(raw, "S1", "C00:A", out_dir=tmp_path) == out
    assert out.stat().st_mtime_ns == mtime     # kept, not overwritten
    with pytest.raises(KeyError):
        gwpe.standardize(raw, "S2", "C00:Z", out_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        gwpe.load_samples(tmp_path / "missing.h5")


def test_standardize_gwtc3_mixed_takes_time_and_snr_from_other_analyses(tmp_path):
    """GWTC-3 `C01:Mixed` has no trigger-time and no SNR column: both come from the other
    analyses of the file (the case of S191216ap)."""
    raw = _fake_release(tmp_path / "IGWN-GWTC3p0-v2-GW191216_213338_PEDataRelease_mixed_cosmo.h5",
                        {"C01:Mixed": 40, "C01:IMRPhenomXPHM": 30})
    with h5py.File(raw, "a") as f:
        del f["C01:Mixed/config_file"]
        del f["C01:IMRPhenomXPHM/config_file/config/trigger-time"]
        f["C01:IMRPhenomXPHM/config_file/config/trigger-time"] = np.array([b"1260567236.471607"])
        ps = f["C01:IMRPhenomXPHM/posterior_samples"][()]
        snr = np.zeros(len(ps), dtype=ps.dtype.descr + [("network_matched_filter_snr", "f8")])
        for name in ps.dtype.names:
            snr[name] = ps[name]
        snr["network_matched_filter_snr"] = 18.0 + np.arange(len(ps)) % 3 - 1   # median 18
        del f["C01:IMRPhenomXPHM/posterior_samples"]
        f["C01:IMRPhenomXPHM/posterior_samples"] = snr
    out = gwpe.standardize(raw, "S191216ap", "C01:Mixed", out_dir=tmp_path / "std")
    t = gwpe.load_samples(out)
    assert t.meta["trigger_time_gps"] == pytest.approx(1260567236.471607)
    with h5py.File(out) as f:
        assert f.attrs["trigger_time_source"] == "C01:IMRPhenomXPHM/config_file/config/trigger-time"
        assert f.attrs["network_snr_median"] == pytest.approx(18.0)
    with open(tmp_path / "std" / "index.csv", newline="") as fh:
        row = next(csv.DictReader(fh))
    assert row["network_snr_median"] == "18.0" and row["trigger_time_gps"] == "1260567236.472"
