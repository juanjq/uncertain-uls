"""uuls.fields and the multi-field path of uuls.realisations, on toy fields.

Toy layout: three fields of 2 x 2 deg (toy_reduced, flat instrument) at RA 150, 151.5 and 154
(Dec 30). The first two overlap by about 0.7 deg in RA, the third is apart. One Gaussian GW blob
(sigma 0.5 deg) sits between the first two; each field's GW map is that blob on its own grid.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

sys.path.insert(0, str(Path(__file__).parent))
import toy  # noqa: E402

from uuls import cases, fields, gwmap as gwm, pointings as pt, realisations as R  # noqa: E402

BLOB = SkyCoord(150.75, 30.0, unit="deg")
SIGMA = 0.5
CENTRES = [(150.0, 30.0), (151.5, 30.0), (154.0, 30.0)]


def blob_gwmap(geom, credible=0.9):
    """GWMap-like: the global blob on `geom` (prob per bin, mask = global credible disc)."""
    image = geom.to_image()
    sep = image.get_coord().skycoord.separation(BLOB).deg
    prob = np.exp(-0.5 * (sep / SIGMA) ** 2) * image.solid_angle().to_value("deg2") / (2 * np.pi * SIGMA ** 2)
    mask = sep < SIGMA * np.sqrt(-2 * np.log(1 - credible))
    r = np.linspace(1.0, 1000.0, 50)
    cdf = np.broadcast_to(np.linspace(0, 1, r.size), prob.shape + r.shape).copy()
    return SimpleNamespace(geom=image, prob=prob, mask=mask, r_grid=r, cdf=cdf,
                           cdf_valid=np.ones(prob.shape, bool), name="blob")


def blob_healpix(nside=256):
    """The same blob as a flat HEALPix map with a distance ansatz (mean 400, sd 80 Mpc)."""
    from ligo.skymap.distance import moments_to_parameters
    import healpy as hp

    npix = hp.nside2npix(nside)
    ra, dec = hp.pix2ang(nside, np.arange(npix), nest=True, lonlat=True)
    sep = SkyCoord(ra, dec, unit="deg").separation(BLOB).deg
    prob = np.exp(-0.5 * (sep / SIGMA) ** 2)
    prob /= prob.sum()
    mu, sigma, norm = moments_to_parameters(np.full(npix, 400.0), np.full(npix, 80.0))
    return gwm.from_flat(prob, mu, sigma, norm, name="blob", t0=toy.T0)


@pytest.fixture(scope="module")
def toy_fields():
    out = []
    for k, c in enumerate(CENTRES):
        reduced = toy.toy_reduced(geom=toy.toy_geom(center=c))
        tab = pt.on(SkyCoord(*c, unit="deg"), toy.T0, 1, 1800)
        out.append(fields.Field(name=f"field{k}", pointings=tab, center=SkyCoord(*c, unit="deg"),
                                geom=reduced.geom, gwmap=blob_gwmap(reduced.geom), reduced=reduced))
    return out


class FixedInjector:
    """PWL of amplitude `strength` at a fixed position."""

    def __init__(self, coord):
        self.coord = coord

    def __call__(self, rng, strength):
        return R.Injection(self.coord, R._pwl(strength, 2.0, 1.0), {"amplitude": float(strength)})


# --- splitting -----------------------------------------------------------------------------

def test_split_wobble_is_one_field_grid_is_many():
    target = SkyCoord(150, 30, unit="deg")
    assert len(fields.split_fields(pt.wobble(target, toy.T0, 8, 1200, offset=0.4))) == 1
    groups = fields.split_fields(pt.grid(target, toy.T0, 3, 2.5, 1200))
    assert len(groups) == 9 and all(len(g) == 1 for g in groups)


def test_split_single_linkage_and_order():
    """A chain of pointings 0.8 deg apart is one field; a later wobble set far away is another;
    fields come in the order of their first run."""
    t = Time(["2026-01-01T02:00:00"] * 3 + ["2026-01-01T01:00:00"] * 4)
    ra = [150.0, 150.0, 150.0, 160.0, 160.0, 160.4, 159.6]
    dec = [30.0, 30.8, 31.6, 30.0, 30.4, 30.0, 30.0]
    tab = pt.make(ra, dec, t, 1200, "grid", obs_id=np.arange(10, 17))
    groups = fields.split_fields(tab, link_deg=1.0)
    assert [sorted(g["obs_id"].tolist()) for g in groups] == [[13, 14, 15, 16], [10, 11, 12]]
    assert len(fields.split_fields(tab, link_deg=0.5)) == 4     # the chain breaks, the wobbles stay


def test_field_geom_holds_every_run_to_offset_max():
    target = SkyCoord(150, 30, unit="deg")
    tab = pt.wobble(target, toy.T0, 4, 1200, offset=0.4)
    geom = fields.field_geom(tab, offset_max_deg=2.5, binsz=0.08)
    assert geom.pixel_scales[0].deg == pytest.approx(0.08)
    assert geom.center_skydir.separation(target).deg < 1e-6
    image = geom.to_image()
    for p in pt.coords(tab):
        edge = p.directional_offset_by(np.arange(0, 360, 15) * u.deg, 2.49 * u.deg)
        assert np.all(image.contains(edge))
    assert min(image.width.to_value(u.deg).ravel()) >= 2 * (0.4 + 2.5) - 1e-6


def test_build_fields_from_case_passes_settings(monkeypatch):
    """With a case: link_deg and the reduce settings come from its config, obs_ids per field."""
    calls = []
    monkeypatch.setattr(fields.gwm, "on_geom", lambda path, geom, credible, root: ("gw", geom))
    monkeypatch.setattr(fields.reduce, "from_dl3",
                        lambda d, geom, t0, s, verbose: calls.append(s) or ("red", geom))
    case = cases.load_case("sim_wobble")              # analysis section from the real config
    tab = pt.wobble(SkyCoord(150, 30, unit="deg"), toy.T0, 4, 1200)
    case.t0, case.pointings = (lambda: toy.T0), (lambda: tab)   # skip the slow eventtime search
    out = fields.build_fields(case, dl3_dir="dl3")
    assert len(out) == 1 and out[0].obs_ids == [1, 2, 3, 4]
    assert calls == [{"background": "fov", "obs_ids": [1, 2, 3, 4]}]
    assert out[0].gwmap[1] is out[0].geom and out[0].info["link_deg"] == 1.0
    # a table without skymap or DL3: geometry only
    geo = fields.build_fields(tab, link_deg=0.3)
    assert len(geo) == 4 and all(f.gwmap is None and f.reduced is None for f in geo)


# --- multi-field realisations ------------------------------------------------------------------

def test_injection_seen_by_the_fields_that_contain_it(toy_fields):
    pre = R.precompute_fields(toy_fields)
    overlap = SkyCoord(150.75, 30.0, unit="deg")         # inside fields 0 and 1
    alone = SkyCoord(154.0, 30.2, unit="deg")            # inside field 2 only
    assert [f.contains(overlap) for f in toy_fields] == [True, True, False]
    assert [R.field_contains(f, alone) for f in toy_fields] == [False, False, True]
    for coord, seen in ((overlap, [0, 1]), (alone, [2])):
        out = R.realisation_fields(toy_fields, pre, FixedInjector(coord), 3e-11, 5, 0, keep_maps=True)
        bkg = R.realisation_fields(toy_fields, pre, FixedInjector(coord), 0.0, 5, 0, keep_maps=True)
        assert out["n_fields_containing"] == len(seen) and out["n_signal"] > 20
        for k in range(3):
            excess = out["counts"][k].sum() - bkg["counts"][k].sum()
            assert (excess > 20) == (k in seen), (k, excess)


def test_lambda_is_max_over_fields(toy_fields):
    pre = R.precompute_fields(toy_fields)
    inj = FixedInjector(SkyCoord(151.4, 30.1, unit="deg"))
    out = R.realisation_fields(toy_fields, pre, inj, 1e-11, 9, 3)
    lam = []
    for k, (f, p) in enumerate(zip(toy_fields, pre)):
        seed_k = R.field_poisson_seed(9 + 3, k)
        if f.contains(inj.coord):
            counts, _ = R.fake_stacked(f.reduced.stacked, inj.coord, R._pwl(1e-11, 2.0, 1.0), seed_k)
        else:
            counts, _ = R.fake_stacked(f.reduced.stacked, inj.coord, None, seed_k)
        lam.append(R.evaluate_field(p, counts)["lambda"])
    np.testing.assert_array_equal(out["lambda_fields"], lam)
    assert lam[2] == -np.inf          # field 2 holds no bin of the GW region
    assert out["lambda"] == max(lam) and out["lambda_field"] == int(np.argmax(lam))


def test_fields_have_independent_counts(toy_fields):
    """Same background in every toy field, different Poisson draws."""
    pre = R.precompute_fields(toy_fields)
    out = R.realisation_fields(toy_fields, pre, FixedInjector(BLOB), 0.0, 1, 0, keep_maps=True)
    assert not np.array_equal(out["counts"][0], out["counts"][1])


def test_source_outside_every_field_is_background(toy_fields):
    far = FixedInjector(SkyCoord(200.0, -10.0, unit="deg"))
    a = R.simulate_fields(toy_fields, far, [1e-9, 1e-9], seed=4)
    b = R.simulate_fields(toy_fields, far, [0.0, 0.0], seed=4)
    assert np.all(a["n_fields_containing"] == 0) and np.all(a["n_signal"] == 0)
    for key in ("lambda", "lambda_fields", "tsmax", "lambda_field"):
        np.testing.assert_array_equal(a[key], b[key])


def test_one_field_equals_single_field_run(toy_fields):
    f = toy_fields[0]
    inj = R.PWLInjector2D(f.gwmap)
    s = np.r_[np.zeros(3), np.full(3, 5e-12)]
    single = R.simulate(f.reduced, f.gwmap, inj, s, seed=7)
    multi = R.simulate_fields([f], inj, s, seed=7)
    for key in single:
        np.testing.assert_array_equal(single[key], multi[key], err_msg=key)
    np.testing.assert_array_equal(multi["lambda_fields"][:, 0], single["lambda"])


def test_run_fields_is_cached(toy_fields, tmp_path):
    inj = FixedInjector(BLOB)
    inj.describe = lambda: {"kind": "fixed", "ra": BLOB.ra.deg, "dec": BLOB.dec.deg}
    kw = dict(seed=2, cache_root=tmp_path, verbose=False)
    a = R.run_fields(toy_fields, inj, [0.0, 1e-11], **kw)
    b = R.run_fields(toy_fields, inj, [0.0, 1e-11], **kw)
    assert len(list((tmp_path / "realisations_fields").iterdir())) == 1
    np.testing.assert_array_equal(a["lambda"], b["lambda"])
    R.run_fields(toy_fields[:2], inj, [0.0, 1e-11], **kw)
    assert len(list((tmp_path / "realisations_fields").iterdir())) == 2


def test_observed_lambda(toy_fields):
    for k, f in enumerate(toy_fields):
        f.reduced.stacked.counts.data = R.fake_stacked(f.reduced.stacked, BLOB, None, 100 + k)[0]
    obs = fields.observed_lambda(toy_fields)
    assert obs["lambda"] == obs["lambda_fields"].max() and obs["field"] == int(np.argmax(obs["lambda_fields"]))
    for f in toy_fields:
        f.reduced.stacked.counts.data[:] = 0


# --- injection on the whole sky ---------------------------------------------------------------

@pytest.fixture(scope="module")
def sky():
    return blob_healpix()


def test_sky_sampler_counts_overlap_probability_once(toy_fields, sky):
    """Draws land in a field with the blob probability of the field's area, and in the union of
    two overlapping fields with the probability of the union, not of the sum."""
    d = fields.sample_healpix(sky, 20000, np.random.default_rng(0))
    c = SkyCoord(d["ra"], d["dec"], unit="deg")
    inside = [f.geom.to_image().contains(c) for f in toy_fields]
    p_f = [f.gwmap.prob.sum() for f in toy_fields]          # blob integrated on each field's bins
    image0 = toy_fields[0].geom.to_image()
    in_both = toy_fields[1].geom.to_image().contains(image0.get_coord().skycoord)
    p_union = p_f[0] + p_f[1] - toy_fields[0].gwmap.prob[in_both].sum()
    assert p_f[0] + p_f[1] > p_union + 0.2                 # the overlap holds real probability
    for k in range(3):
        assert inside[k].mean() == pytest.approx(p_f[k], abs=0.015)
    assert (inside[0] | inside[1]).mean() == pytest.approx(p_union, abs=0.015)


def test_sky_sampler_uniform_inside_pixel():
    """One pixel holding all the probability: draws spread over it, mean at its centre."""
    import healpy as hp

    nside = 16
    prob = np.zeros(hp.nside2npix(nside))
    prob[1000] = 1.0
    one = gwm.from_flat(prob, np.full(prob.size, np.inf), np.ones(prob.size), np.zeros(prob.size))
    d = fields.sample_healpix(one, 5000, np.random.default_rng(1))
    assert np.all(hp.ang2pix(nside, d["ra"], d["dec"], nest=True, lonlat=True) == 1000)
    centre = SkyCoord(*hp.pix2ang(nside, 1000, nest=True, lonlat=True), unit="deg")
    c = SkyCoord(d["ra"], d["dec"], unit="deg")
    assert np.median(c.separation(centre).deg) > 0.3 * hp.nside2resol(nside, arcmin=True) / 60
    assert mean_offset(c, centre) < 0.05 * hp.nside2resol(nside, arcmin=True) / 60


def mean_offset(c, centre):
    xyz = c.cartesian.xyz.value.mean(axis=1)
    return SkyCoord(*xyz, representation_type="cartesian").separation(centre).deg


def test_sky_injectors(sky):
    inj2 = fields.PWLSkyInjector2D(sky, credible=0.9)
    seps = [inj2(np.random.default_rng([1, i]), 1e-12).coord.separation(BLOB).deg for i in range(300)]
    assert max(seps) < SIGMA * np.sqrt(-2 * np.log(0.1)) + 0.1
    inj3 = fields.PWLSkyInjector3D(sky)
    a = inj3(np.random.default_rng([5, 1]), 1e44)
    b = inj3(np.random.default_rng([5, 1]), 4e44)
    assert a.params["distance"] == b.params["distance"] and a.coord == b.coord
    assert b.params["amplitude"] == pytest.approx(4 * a.params["amplitude"], rel=1e-12)
    d = [inj3(np.random.default_rng([6, i]), 1e44).params["distance"] for i in range(400)]
    assert np.mean(d) == pytest.approx(400, rel=0.05) and np.std(d) == pytest.approx(80, rel=0.15)
    assert inj2.describe()["sky"] == inj3.describe()["sky"]


def test_sky_injection_outside_fields_counts_as_no_signal(toy_fields, sky):
    """With the sky injector, realisations whose source misses every field have no signal and the
    background Lambda of their seed; the others carry signal."""
    inj = fields.PWLSkyInjector2D(sky, credible=0.99)
    s = np.full(12, 3e-11)
    out = R.simulate_fields(toy_fields, inj, s, seed=3)
    bkg = R.simulate_fields(toy_fields, inj, np.zeros(12), seed=3)
    miss = out["n_fields_containing"] == 0
    np.testing.assert_array_equal(out["lambda"][miss], bkg["lambda"][miss])
    assert np.all(out["n_signal"][~miss] > 0) and (~miss).sum() >= 6


def test_fields_use_the_roi_for_data_and_monte_carlo(toy_fields):
    """A field whose eastern half has no exposure: its maximum is never taken there, in the
    realisations or in observed_lambda, and roi="gw_mask" brings the old region back."""
    from uuls import lambda_stat

    f0 = toy_fields[0]
    red = toy.toy_reduced(geom=toy.toy_geom(center=CENTRES[0]))
    for ds in [red.stacked, *red.runs]:
        ds.exposure.data[..., 20:] = 0
        ds.background.data[..., 20:] = 0
        ds.mask_safe.data[..., 20:] = False
    half = fields.Field("half", f0.pointings, f0.center, red.geom, f0.gwmap, red)
    fs = [half, toy_fields[1]]
    roi = lambda_stat.roi(half.gwmap.mask, red.stacked)
    assert roi.sum() < half.gwmap.mask.sum()
    pres = R.precompute_fields(fs)
    np.testing.assert_array_equal(pres[0].mask, roi)
    np.testing.assert_array_equal(R.precompute_fields(fs, roi="gw_mask")[0].mask, half.gwmap.mask)
    out = R.simulate_fields(fs, FixedInjector(BLOB), np.zeros(20), seed=6)
    in_half = out["lambda_field"] == 0
    assert np.all(roi[out["lambda_iy"][in_half], out["lambda_ix"][in_half]])
    red.stacked.counts.data = R.fake_stacked(red.stacked, BLOB, None, 3)[0]
    obs = fields.observed_lambda(fs)
    assert obs["lambda_fields"][0] == R.evaluate_field(pres[0], red.stacked.counts.data)["lambda"]
