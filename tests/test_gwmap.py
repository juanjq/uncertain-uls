"""uuls.gwmap on the S240615dg alert skymap (multi-order, 95% area 6.6 deg2 around RA 7.7, Dec 45.7)."""
import sys
import warnings

import astropy.units as u
import astropy_healpix as ah
import healpy as hp
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from gammapy.maps import MapAxis, WcsGeom
from ligo.skymap.bayestar import rasterize
from ligo.skymap.io import read_sky_map

from uuls import gwmap, paths

SKYMAP = paths.GW_SKYMAPS / "S240615dg.fits"
OLD_REPO = paths.REPO.parent / "gw-gamma-global-uls"
CENTER = (7.7, 45.7)          # deg, the densest pixel of the map


def small_geom(binsz=0.2, width=6.0):
    """6 x 6 deg AIR geom around the maximum, it contains the whole 95% region."""
    return WcsGeom.create(skydir=CENTER, width=width, binsz=binsz, frame="icrs", proj="AIR")


@pytest.fixture(scope="module")
def sky():
    return gwmap.read_healpix(SKYMAP)


@pytest.fixture(scope="module")
def gw(tmp_path_factory):
    return gwmap.on_geom(SKYMAP, small_geom(), root=tmp_path_factory.mktemp("cache"))


def test_read_healpix(sky):
    assert sky.name == "S240615dg"
    assert sky.t0.isot.startswith("2024-06-15T11:36:20.7")
    assert sky.prob.sum() == pytest.approx(1.0, abs=1e-9)
    i = sky.lookup(*CENTER)
    assert sky.density[i] == pytest.approx(sky.density.max(), rel=0.2)


# ligo.skymap.postprocess.crossmatch(read_sky_map(SKYMAP, moc=True), contours=...).contour_areas,
# computed once (it takes 7 s) with the ligo.skymap of env/uncertain-uls.yml.
LIGO_AREAS_DEG2 = {0.5: 1.49215, 0.9: 5.11055, 0.95: 6.64067}


def test_credible_area_matches_ligo():
    for credible, area in LIGO_AREAS_DEG2.items():
        assert gwmap.credible_area_deg2(SKYMAP, credible) == pytest.approx(area, rel=5e-3)


def test_shapes(gw):
    ny, nx = gw.geom.data_shape
    assert (ny, nx) == (30, 30)
    assert gw.prob.shape == gw.mask.shape == gw.cdf_valid.shape == (ny, nx)
    assert gw.cdf.shape == (ny, nx, len(gw.r_grid))
    assert gw.r_grid[0] == 1.0 and gw.mask.dtype == bool
    assert gw.t0.isot.startswith("2024-06-15")


def test_mask_probability_close_to_credible(gw):
    # The mask keeps or drops whole 0.2 deg bins by their centre, so the bins cut by the 95%
    # contour enter entirely or not at all. About 50 bins lie on the boundary with ~0.1% each;
    # the imbalance between those in and out moves the sum by up to ~1%.
    assert gw.prob[gw.mask].sum() == pytest.approx(0.95, abs=0.01)


def test_cdf_monotone_and_bounded(gw):
    c = gw.cdf[gw.cdf_valid]
    assert np.all(np.diff(c, axis=-1) >= -1e-6)
    assert c.min() >= 0 and c.max() <= 1 + 1e-6
    assert np.all(c[:, -1] > 0.999)          # the event is at ~1.4 Gpc, far below 10 Gpc
    assert gw.cdf_valid[gw.mask].all()


def _hp_prob_inside(sky, geom):
    """HEALPix probability of the pixels whose centre falls inside geom."""
    level, ipix = ah.uniq_to_level_ipix(sky.uniq)
    lon, lat = np.empty(len(ipix)), np.empty(len(ipix))
    for lv in np.unique(level):
        sel = level == lv
        lo, la = ah.healpix_to_lonlat(ipix[sel], ah.level_to_nside(lv), order="nested")
        lon[sel], lat[sel] = lo.deg, la.deg
    return sky.prob[geom.contains(SkyCoord(lon, lat, unit="deg"))].sum()


def test_total_probability_against_healpix(sky, gw):
    # The geom contains the 95% region, so both sums are close to 1; they differ only by the
    # integration over bins (sub-bin points vs whole pixels at the geom edge).
    assert gw.prob.sum() == pytest.approx(_hp_prob_inside(sky, gw.geom), abs=2e-3)
    assert gw.prob.sum() > 0.95


def test_cache_round_trip(tmp_path, capsys):
    geom = small_geom(binsz=0.5, width=4.0)
    a = gwmap.on_geom(SKYMAP, geom, r_grid=np.linspace(1, 5000, 50), root=tmp_path)
    b = gwmap.on_geom(SKYMAP, geom, r_grid=np.linspace(1, 5000, 50), root=tmp_path)
    assert "loaded" in capsys.readouterr().out
    assert np.array_equal(a.prob, b.prob) and np.array_equal(a.cdf, b.cdf)
    assert np.array_equal(a.mask, b.mask) and a.t0 == b.t0
    assert len(list((tmp_path / "gwmap").glob("*/params.json"))) == 1


def test_r_grid_must_start_above_zero():
    with pytest.raises(ValueError):
        gwmap.on_geom(SKYMAP, small_geom(), r_grid=np.linspace(0, 100, 10))


# --- regression against the old gwuls functions --------------------------------------------

@pytest.fixture(scope="module")
def flat9():
    """The skymap rasterised to nside 512 (RING arrays), the input both codes accept."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f = rasterize(read_sky_map(str(SKYMAP), moc=True), order=9)
    ring = hp.ring2nest(512, np.arange(hp.nside2npix(512)))
    return tuple(np.asarray(f[c])[ring] for c in ("PROB", "DISTMU", "DISTSIGMA", "DISTNORM"))


def _old_outputs(flat, geom3d, r_grid):
    """Bin probability, 95% mask and distance CDF from gwuls.utils, as 3_upper_limits does."""
    sys.path.insert(0, str(OLD_REPO))
    try:
        from gwuls import utils
    finally:
        sys.path.remove(str(OLD_REPO))
    prob, mu, sig, norm = flat
    nside = hp.npix2nside(len(prob))
    dec_hp, ra_hp = utils.IndexToDeclRa(np.arange(len(prob)), nside)
    keep = np.abs(dec_hp - CENTER[1]) < 6        # only speeds up; the old code cuts to the geom too
    idx = np.nonzero(keep)[0]
    c_ra, c_dec, e_ra, e_dec, area, *_ = utils.extract_geom_coords(geom3d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p_old, _ = utils.integrate_hp_on_wcs(prob[keep], dec_hp[keep], ra_hp[keep],
                                             hp.nside2pixarea(nside), e_ra, e_dec, area)
    bins = utils.map_hp_pixels_to_wcs_bins(dec_hp[keep], ra_hp[keep], e_ra, e_dec, fill_empty=False)
    bins = np.vectorize(lambda a: idx[a], otypes=[object])(bins)
    cdf_old, valid_old = utils.compute_distance_cdf_table(bins, prob, mu, sig, norm, r_grid)
    ra_b, dec_b = np.linspace(-180, 180, 1024), np.linspace(-90, 90, 1024)
    ra_g, dec_g = np.deg2rad(np.meshgrid(ra_b, dec_b))
    image = utils.healpix2map(prob, ra_b, dec_b)
    level = utils.get_hp_map_thresholds(prob, [0.95])
    mask_old = utils.compute_threshold_masks(level, ra_g, dec_g, image, c_ra, c_dec, p_old)[0]
    return p_old, mask_old, cdf_old, valid_old


def _exact_bin_prob(sky, geom):
    """Reference: nside-4096 sub-pixels of the map, each summed into the WCS bin it falls in."""
    fine = hp.query_disc(4096, hp.ang2vec(*CENTER, lonlat=True), np.radians(6), nest=True)
    lon, lat = ah.healpix_to_lonlat(fine, 4096, order="nested")
    p = sky.density[sky.lookup(lon.deg, lat.deg)] * hp.nside2pixarea(4096)
    x, y = geom.coord_to_idx(SkyCoord(lon, lat))
    ok = x >= 0
    ref = np.zeros(geom.data_shape)
    np.add.at(ref, (y[ok], x[ok]), p[ok])
    return ref


@pytest.mark.skipif(not (OLD_REPO / "gwuls" / "utils.py").exists(), reason="old repo not present")
def test_regression_against_gwuls(flat9):
    """Same input map, same geom: the new and old projections agree within the old method's
    discretisation error, and the new one is closer to the exact bin integral."""
    energy = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=1)
    geom3d = WcsGeom.create(skydir=CENTER, width=6.0, binsz=0.2, frame="icrs", proj="AIR",
                            axes=[energy])
    r = np.linspace(1, 10000, 1000)
    p_old, mask_old, cdf_old, valid_old = _old_outputs(flat9, geom3d, r)
    sky = gwmap.from_flat(*flat9, nest=False)
    new = gwmap.project(sky, geom3d, credible=0.95, r_grid=r)
    ref = _exact_bin_prob(sky, new.geom)

    sel = ref > 1e-3
    assert new.prob.sum() == pytest.approx(p_old.sum(), abs=5e-3)
    # old: mean of the ~3 pixels assigned to a bin by a (ra, dec) box, scaled to the bin area;
    # its per-bin error is ~5%. New: mean over 64 points per bin, error ~1.5%.
    assert np.median(np.abs(new.prob[sel] / p_old[sel] - 1)) < 0.08
    assert np.abs(new.prob - p_old).sum() < 0.07
    err_new = np.abs(new.prob - ref).sum()
    err_old = np.abs(p_old - ref).sum()
    assert err_new < 0.5 * err_old
    # mask: the same contour, sampled at bin centres; only boundary bins may differ
    assert (new.mask != mask_old).sum() <= 12
    assert abs(new.prob[new.mask].sum() - p_old[mask_old].sum()) < 0.01
    # distance: median distance per bin
    both = valid_old & new.cdf_valid & sel
    med = lambda c: np.array([np.interp(0.5, x, r) for x in c])
    assert np.max(np.abs(med(new.cdf[both]) / med(cdf_old[both]) - 1)) < 0.03
    assert np.max(np.abs(new.cdf[both] - cdf_old[both])) < 0.05
