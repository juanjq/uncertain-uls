"""uuls.flux_ul on the toy dataset: the whole 2D flux UL chain with a small budget.

Toy: one field (toy_reduced, flat instrument, 40 x 40 bins of 0.05 deg) with a Gaussian GW blob;
for the multi-field helpers, the three toy fields and HEALPix blob of test_fields.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import toy  # noqa: E402
from test_fields import blob_healpix, toy_fields  # noqa: E402,F401  (fixture)

from uuls import fields, flux_ul, lambda_stat, realisations as R, spectra  # noqa: E402

BUDGET = dict(n_bkg=150, n_scout=150, n_refine=150, max_sims=300, n_closure=100,
              x_lo=1e-13, x_hi=1e-10, seed=3)


@pytest.fixture(scope="module")
def one_field():
    reduced = toy.toy_reduced()
    return [SimpleNamespace(name="field0", reduced=reduced, gwmap=toy.toy_gwmap(reduced.geom))]


@pytest.fixture(scope="module")
def toy_run(one_field, tmp_path_factory):
    inj = flux_ul.make_injector(one_field)
    root = tmp_path_factory.mktemp("cache")
    summary, det = flux_ul.flux_ul_fields(one_field, inj, verbose=False, cache_root=root, **BUDGET)
    return summary, det, inj, root


def test_one_field_uses_the_roi_injector(one_field):
    inj = flux_ul.make_injector(one_field)
    assert isinstance(inj, R.PWLInjector2D)
    f = one_field[0]
    assert np.array_equal(inj.support, lambda_stat.roi(f.gwmap.mask, f.reduced.stacked))


def test_summary_is_complete_and_json(toy_run):
    s, det, _, _ = toy_run
    json.dumps(s)                                    # plain types only
    a = s["amplitude"]
    assert 1e-13 < a["lo_1sigma"] < a["ul"] < a["hi_1sigma"] < 1e-10
    assert a["lo_2sigma"] <= a["lo_1sigma"] and a["hi_2sigma"] >= a["hi_1sigma"]
    assert s["background"]["n"] == BUDGET["n_bkg"]
    assert 0 <= s["background"]["p_value"] <= 1
    assert s["fit"]["n_sim"] <= BUDGET["max_sims"]
    assert s["closure"]["n"] == BUDGET["n_closure"]
    assert det["ul"].x_ul == pytest.approx(a["ul"])


def test_fluxes_follow_the_amplitude(toy_run):
    s = toy_run[0]
    ph = spectra.pwl_photon_flux(s["amplitude"]["ul"], [0.6, 20] * spectra.u.TeV).to_value("cm-2 s-1")
    assert s["photon_flux"]["ul"] == pytest.approx(ph)
    # PWL index 2 at 1 TeV: energy flux = phi0 * (1 TeV)^2 * ln(20 / 0.6), in erg
    ef = s["amplitude"]["ul"] * np.log(20 / 0.6) * 1.602176634
    assert s["energy_flux"]["ul"] == pytest.approx(ef, rel=1e-6)


def test_closure_is_consistent(toy_run):
    c = toy_run[0]["closure"]
    assert abs(c["pull"]) < 4


def test_rerun_is_cached(toy_run, one_field):
    s, _, inj, root = toy_run
    s2, _ = flux_ul.flux_ul_fields(one_field, inj, verbose=False, cache_root=root, **BUDGET)
    assert s2["amplitude"] == s["amplitude"]
    assert s2["timing_s"]["background_s"] < 5


def test_multi_field_injector_stays_in_the_roi(toy_fields):  # noqa: F811
    sky = blob_healpix(nside=128)
    sup = flux_ul.roi_pixels(sky, toy_fields, credible=0.9)
    inj = flux_ul.make_injector(toy_fields, sky, credible=0.9)
    assert isinstance(inj, fields.PWLSkyInjector2D)
    assert "support" in inj.describe()
    assert 0 < flux_ul.roi_gw_prob(toy_fields, sky, credible=0.9) <= 0.9 + 1e-9
    rng = np.random.default_rng(0)
    for _ in range(50):
        d = inj(rng, 1e-12)
        assert sup[d.params["pix"]]
        assert any(R.field_contains(f, d.coord) for f in toy_fields)
