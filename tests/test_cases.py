import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time

from uuls import cases, eventtime, paths


def test_all_case_files_parse():
    names = {c.name for c in cases.all_cases()}
    assert {"real_S240615dg", "sim_wobble", "sim_tiling"} <= names
    for c in cases.all_cases():
        assert c.skymap.exists(), c.skymap


def test_real_case_keeps_real_time_and_waits_for_pointings():
    c = cases.load_case("real_S240615dg")
    assert c.data_kind == "real" and c.t0_mode == "real"
    assert abs((c.t0() - Time("2024-06-15T11:36:20.729")).to_value(u.s)) < 1
    with pytest.raises(FileNotFoundError, match="real pointings"):
        c.pointings()


def test_sim_wobble_pointings():
    c = cases.load_case(paths.CONFIGS / "cases" / "sim_wobble.yaml")
    t0 = c.t0()
    tab = c.pointings()
    assert len(tab) == 4 and tab.meta["strategy"] == "wobble" and tab.meta["t0"] == t0.isot
    assert abs((tab["t_start"][0] - t0).to_value(u.s) - 60) < 1e-3
    np.testing.assert_allclose(tab["duration"], 1200)
    assert eventtime.sun_altitude(t0) < -18 + 1e-3
    assert eventtime.altitude(c.target(), t0) > 50 - 1e-3
    assert c.analysis_center(tab).separation(c.target()).deg < 0.05
    w = c.analysis_width(tab)
    assert len(w) == 2 and min(w) >= 2 * (0.4 + 2.5)


def test_bad_config_is_rejected(tmp_path):
    text = (paths.CONFIGS / "cases" / "sim_wobble.yaml").read_text().replace("mode: observable", "mode: sometime")
    f = tmp_path / "bad.yaml"
    f.write_text(text)
    with pytest.raises(ValueError, match="t0.mode"):
        cases.load_case(f)


@pytest.mark.slow
def test_sim_tiling_pointings(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CACHE", tmp_path)
    c = cases.load_case("sim_tiling")
    tab = c.pointings()
    assert tab.meta["strategy"] == "tilepy" and len(tab) >= 5
    assert 0 < tab["pgw"].sum() <= 1
