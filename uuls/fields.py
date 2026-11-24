"""Fields: groups of runs analysed on their own small geometry (tiling cases).

    groups = split_fields(pointings, link_deg=1.0)       # list of Pointings tables, one per field
    fields = build_fields(case, dl3_dir)                   # list[Field]; GWMap and ReducedData cached
    inj = PWLSkyInjector2D(gwmap.read_healpix(case.skymap))   # source drawn on the whole sky
    out = realisations.run_fields(fields, inj, strengths, seed)
    obs = observed_lambda(fields)

A field is a set of runs whose pointings are close enough to share one geometry: pointings are
grouped by single linkage, two pointings closer than `link_deg` belonging to the same field. A
wobble set (offsets up to about 0.5 deg) is one field; tiles one field radius apart are separate
fields. Each field has its own WCS geometry, centred on the mean direction of its pointings and
wide enough to hold every run out to the reduction's offset-max, with the common bin size, its
own GWMap on that geometry and its own ReducedData made from its runs only.

The global statistic is

    Lambda = max over fields f of  max over the ROI bins b of f of  (TS_b + 2 ln p_b),

with TS_b and p_b computed on field f's own grid and data, and the ROI of f the GW 95% mask
intersected with f's observed region (exposure > 0 inside the safe mask; lambda_stat.roi).

Overlaps. Neighbouring tiles of a tiling overlap (tile centres a few degrees apart, each run
exposed to 2.5 deg), so a sky position can lie in two fields. The rule is: overlaps are left as
they are, and the position is a trial in every field that contains it. Merging overlapping fields
instead would chain the whole arc of a tiling into one field (every tile overlaps its neighbour),
which is the single large geometry this module exists to avoid. Leaving overlaps is correct
because
  - the fields partition the runs, so the counts of two fields are independent data: in an
    overlap the two TS values are two independent looks at the same position, not the same
    counts counted twice;
  - Lambda is a maximum, not a sum, so a bin present in two fields does not add GW probability:
    2 ln p_b enters once per look and the larger look wins;
  - the Monte Carlo (`realisations.run_fields`) applies exactly the same rule: one injection,
    every field faked from its own runs, the source in every field whose geometry contains it,
    Lambda the maximum over fields. The distribution of Lambda, and so the UL, is calibrated on
    the statistic actually used on the data. The extra trials in overlaps raise the background
    Lambda a little in data and Monte Carlo alike.
What it costs: in an overlap the runs of the two fields are not combined, so the sensitivity
there is that of the better field rather than of their sum. That makes the UL slightly higher
(conservative), never wrong. Bins of a field where that field has no exposure are outside its
ROI, so they put no data-independent floor 2 ln p_b under Lambda; a position covered by two
fields but exposed in only one is a trial only in that one.

Injection over several fields: `PWLSkyInjector2D` / `PWLSkyInjector3D` draw the position on the
HEALPix map itself (a pixel ~ its probability, then uniform inside the pixel), so the GW
probability of a position is counted once whatever the number of fields covering it. The source
is then injected into every field containing it. A position outside every field is a realisation
with no signal; it stays in the Monte Carlo, and is what makes the detection fraction saturate
below 1 when the fields do not cover the region.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import astropy.units as u
import astropy_healpix as ah
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time

from . import gwmap as gwm
from . import pointings as pt
from . import realisations, reduce, spectra

LINK_DEG = 1.0          # default linking length between pointings of one field
_MAX_LEVEL = 29         # HEALPix order of the uniform position inside a pixel


@dataclass(eq=False)
class Field:
    """One field: its runs, geometry, GW map and reduced data.

    name: "field<k>", k from 0 in order of the first run. pointings: the Pointings rows of the
    field. center: SkyCoord, mean pointing direction. geom: 3D WcsGeom (reco energy axis).
    gwmap: GWMap on geom (None if no skymap was given). reduced: ReducedData of the field's runs
    on geom (None if no DL3 folder was given).
    """
    name: str
    pointings: Table
    center: SkyCoord
    geom: object
    gwmap: object = None
    reduced: object = None
    info: dict = field(default_factory=dict)

    @property
    def obs_ids(self) -> list[int]:
        return [int(i) for i in self.pointings["obs_id"]]

    def contains(self, coord: SkyCoord) -> bool:
        """True if coord lies inside the image of geom (and within 90 deg of its centre)."""
        image = self.geom.to_image()
        return bool(coord.separation(image.center_skydir).deg < 90 and np.all(image.contains(coord)))


def mean_direction(coords: SkyCoord) -> SkyCoord:
    """Mean of unit vectors of `coords` (ICRS), as cases.Case.analysis_center."""
    xyz = coords.icrs.cartesian.xyz.value.reshape(3, -1).mean(axis=1)
    return SkyCoord(np.degrees(np.arctan2(xyz[1], xyz[0])) % 360 * u.deg,
                    np.degrees(np.arctan2(xyz[2], np.hypot(xyz[0], xyz[1]))) * u.deg, frame="icrs")


def split_fields(pointings: Table, link_deg: float = LINK_DEG) -> list[Table]:
    """Group the runs of a Pointings table into fields (single linkage).

    Two runs are in the same field if a chain of pointings, each closer than `link_deg` (deg) to
    the next, joins them. In: Pointings table, link_deg. Out: list of Pointings tables (rows of
    each field, meta copied), ordered by the start of their first run.
    """
    c = pt.coords(pointings)
    n = len(c)
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n > 1:
        sep = c[:, None].separation(c[None, :]).deg
        for i, j in zip(*np.nonzero(np.triu(sep < float(link_deg), k=1))):
            parent[root(i)] = root(j)
    labels = np.array([root(i) for i in range(n)])
    t = np.asarray(pointings["t_start"].mjd)
    groups = sorted({lab: np.flatnonzero(labels == lab) for lab in labels}.values(),
                    key=lambda rows: t[rows].min())
    out = []
    for rows in groups:
        sub = pointings[np.sort(rows)]
        sub.meta = copy.deepcopy(pointings.meta)
        out.append(sub)
    return out


def field_geom(pointings: Table, offset_max_deg: float = 2.5, binsz: float = 0.08,
               e_reco=(0.6, 20) * u.TeV, bins_per_decade: float = 4.5, pad_deg: float = 0.0):
    """Square analysis geometry of one field (reduce.analysis_geom, AIR projection).

    Centre: mean pointing direction. Width: 2 (largest pointing offset from the centre +
    offset_max_deg + pad_deg), rounded up to a whole number of bins of `binsz`.
    In: Pointings rows of the field, deg values, e_reco Quantity. Out: 3D WcsGeom.
    """
    c = pt.coords(pointings)
    center = mean_direction(c)
    r = float(np.max(c.separation(center).deg))
    npix = int(np.ceil(2 * (r + offset_max_deg + pad_deg) / binsz - 1e-9))
    return reduce.analysis_geom(center, width=npix * binsz, binsz=binsz, e_reco=e_reco,
                                bins_per_decade=bins_per_decade)


def build_fields(source, dl3_dir=None, settings: dict | None = None, *, skymap=None, t0=None,
                 link_deg: float | None = None, binsz: float = 0.08, e_reco=(0.6, 20) * u.TeV,
                 bins_per_decade: float = 4.5, credible: float = 0.95, pad_deg: float = 0.0,
                 gwmap_root=None, verbose: bool = False) -> list[Field]:
    """The fields of a case (or of a Pointings table), with their GW maps and reduced data.

    source: a cases.Case (pointings, skymap, t0, analysis.reduce and analysis.fields.link_deg are
        taken from it) or a Pointings table (then give skymap; t0 defaults to meta["t0"]).
    dl3_dir: DL3 folder holding every run (simulated or real), or None for geometry and GW map only.
    settings: reduce settings (default: the case's analysis.reduce); obs_ids is set per field and
        offset_max_deg sets the field width.
    skymap, t0: override the case's. link_deg: default the case's, else LINK_DEG.
    binsz, e_reco, bins_per_decade: common to every field. credible: GW region of each GWMap.
    Out: list[Field]. GWMap (gwmap.on_geom) and ReducedData (reduce.from_dl3) are cached per field.
    """
    if isinstance(source, Table):
        tab, case_settings, case_link = source, {}, None
    else:
        tab = source.pointings()
        skymap = source.skymap if skymap is None else skymap
        t0 = source.t0() if t0 is None else t0
        case_settings = dict(source.analysis.get("reduce") or {})
        case_link = (source.analysis.get("fields") or {}).get("link_deg")
    t0 = Time(tab.meta["t0"], scale="utc") if t0 is None else Time(t0)
    settings = copy.deepcopy(case_settings if settings is None else settings)
    link = float(link_deg if link_deg is not None else (case_link if case_link is not None else LINK_DEG))
    offset_max = float(settings.get("offset_max_deg", reduce.DEFAULT_SETTINGS["offset_max_deg"]))
    out = []
    for k, rows in enumerate(split_fields(tab, link)):
        geom = field_geom(rows, offset_max, binsz, e_reco, bins_per_decade, pad_deg)
        f = Field(name=f"field{k}", pointings=rows, center=geom.center_skydir, geom=geom,
                  info={"link_deg": link, "offset_max_deg": offset_max})
        if skymap is not None:
            f.gwmap = gwm.on_geom(skymap, geom, credible=credible, root=gwmap_root)
        if dl3_dir is not None:
            f.reduced = reduce.from_dl3(dl3_dir, geom, t0, {**settings, "obs_ids": f.obs_ids},
                                        verbose=verbose)
        out.append(f)
    return out


def observed_lambda(fields, correlation_radius_deg: float = 0.1, roi: str = "observed") -> dict:
    """Lambda of the data: each field's Lambda on its stacked counts, and the max over fields,
    over each field's ROI (lambda_stat.roi, the same as realisations.run_fields with this roi).

    In: fields with .reduced and .gwmap. Out: dict lambda, field (index reaching it), lambda_iy,
    lambda_ix (bin in that field), tsmax, lambda_fields (n_fields,).
    """
    pres = realisations.precompute_fields(fields, correlation_radius_deg, roi)
    ev = [realisations.evaluate_field(p, f.reduced.stacked.counts.data) for f, p in zip(fields, pres)]
    lam = np.array([e["lambda"] for e in ev])
    k = int(np.argmax(lam))
    return {"lambda": float(lam[k]), "field": k, "lambda_iy": ev[k]["lambda_iy"],
            "lambda_ix": ev[k]["lambda_ix"], "tsmax": float(max(e["tsmax"] for e in ev)),
            "lambda_fields": lam}


def coverage(fields, where) -> np.ndarray:
    """Number of field geometries containing each position.

    where: a WcsGeom (its image bin centres; out (ny, nx)) or a SkyCoord array (out, same shape). -> int array.
    """
    c = where if isinstance(where, SkyCoord) else where.to_image().get_coord(mode="center").skycoord
    n = np.zeros(c.shape, dtype=int)
    for f in fields:
        g = f.geom.to_image()
        n += (c.separation(g.center_skydir).deg < 90) & g.contains(c)
    return n


# --- injection on the whole sky ------------------------------------------------------------

def healpix_cdf(sky, support=None) -> np.ndarray:
    """Cumulative pixel probability (inside `support`), normalised to 1: what sample_healpix draws from."""
    w = np.where(support, sky.prob, 0.0) if support is not None else np.asarray(sky.prob, float)
    c = np.cumsum(w)
    if c[-1] <= 0:
        raise ValueError("No HEALPix pixel has probability > 0 inside the support.")
    return c / c[-1]


def sample_healpix(sky, n, rng, support=None, cdf=None) -> dict:
    """n positions drawn from a HEALPix map: pixel ~ prob (inside `support` if given, renormalised),
    then uniform inside the pixel (a random descendant at order 29, equal-area).

    sky: gwmap.HealpixSkymap (multi-order). support: bool per pixel or None; cdf: healpix_cdf(sky,
    support) computed once (faster for many calls). rng: Generator (n uniforms for the pixels,
    then n integers for the position inside).
    -> dict ra, dec (deg, (n,)), pix (index into the sky arrays, (n,)).
    """
    cdf = healpix_cdf(sky, support) if cdf is None else cdf
    pix = np.minimum(np.searchsorted(cdf, rng.random(n), side="right"), cdf.size - 1)
    level, ipix = ah.uniq_to_level_ipix(sky.uniq[pix])
    level, ipix = np.asarray(level, np.int64), np.asarray(ipix, np.int64)
    k = _MAX_LEVEL - level
    fine = (ipix << (2 * k)) + rng.integers(0, np.int64(1) << (2 * k))
    lon, lat = ah.healpix_to_lonlat(fine, nside=np.int64(1) << _MAX_LEVEL, order="nested")
    return {"ra": lon.to_value(u.deg), "dec": lat.to_value(u.deg), "pix": pix}


def _sky(sky):
    return gwm.read_healpix(sky) if isinstance(sky, (str, Path)) else sky


def _sky_id(sky) -> dict:
    return {"name": str(sky.name), "prob_sha1": hashlib.sha1(np.ascontiguousarray(sky.prob)).hexdigest()}


@dataclass
class PWLSkyInjector2D:
    """Strength = phi0 [cm-2 s-1 TeV-1]; position drawn on the HEALPix map inside its greedy
    `credible` region (the 2D analogue of realisations.PWLInjector2D, for several fields).
    sky: gwmap.HealpixSkymap or a skymap path. support: optional bool per pixel, intersected
    with the credible region (e.g. the pixels inside the fields' ROI, uuls.flux_ul.roi_pixels)."""
    sky: object
    credible: float = 0.95
    index: float = 2.0
    e_ref_TeV: float = 1.0
    support: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        self.sky = _sky(self.sky)
        inside = gwm.credible_levels(self.sky) <= self.credible
        if self.support is not None:
            self.support = np.asarray(self.support, dtype=bool)
            inside = inside & self.support
        self._cdf = healpix_cdf(self.sky, inside)

    def __call__(self, rng, strength) -> realisations.Injection:
        d = sample_healpix(self.sky, 1, rng, cdf=self._cdf)
        ra, dec = float(d["ra"][0]), float(d["dec"][0])
        return realisations.Injection(SkyCoord(ra * u.deg, dec * u.deg, frame="icrs"),
                                      realisations._pwl(strength, self.index, self.e_ref_TeV),
                                      {"ra": ra, "dec": dec, "pix": int(d["pix"][0]),
                                       "amplitude": float(strength)})

    def describe(self) -> dict:
        d = {"kind": "pwl_sky_2d", "sky": _sky_id(self.sky), "credible": self.credible,
             "index": self.index, "e_ref_TeV": self.e_ref_TeV}
        if self.support is not None:     # absent without support: earlier cache keys unchanged
            d["support"] = hashlib.sha1(np.packbits(self.support)).hexdigest()
        return d


@dataclass
class PWLSkyInjector3D:
    """Strength = band luminosity L0 [erg/s] in band_TeV. Position drawn on the HEALPix map
    (pixels with distance information; inside the `restrict_to_credible` region if given), then
    d from that pixel's distance ansatz (ligo.skymap conditional_ppf), floored at d_min_Mpc as the
    1 Mpc start of the GWMap distance grid. Amplitude as realisations.PWLInjector3D."""
    sky: object
    band_TeV: tuple = (0.6, 20.0)
    index: float = 2.0
    e_ref_TeV: float = 1.0
    restrict_to_credible: float | None = None
    apply_k_correction: bool = False
    d_min_Mpc: float = 1.0

    def __post_init__(self):
        self.sky = _sky(self.sky)
        s = self.sky
        support = np.isfinite(s.distmu) & np.isfinite(s.distsigma) & (s.distnorm > 0)
        if self.restrict_to_credible is not None:
            support &= gwm.credible_levels(s) <= self.restrict_to_credible
        self._cdf = healpix_cdf(s, support)

    def __call__(self, rng, strength) -> realisations.Injection:
        from ligo.skymap.distance import conditional_ppf

        d = sample_healpix(self.sky, 1, rng, cdf=self._cdf)
        p = int(d["pix"][0])
        s = self.sky
        dist = float(conditional_ppf(rng.uniform(), s.distmu[p], s.distsigma[p], s.distnorm[p]))
        dist = max(dist, self.d_min_Mpc)
        amp = spectra.amplitude_from_luminosity(
            strength, dist * u.Mpc, np.asarray(self.band_TeV) * u.TeV, index=self.index,
            e_ref=self.e_ref_TeV * u.TeV, redshift="auto" if self.apply_k_correction else None,
            apply_k_correction=self.apply_k_correction)
        ra, dec = float(d["ra"][0]), float(d["dec"][0])
        return realisations.Injection(SkyCoord(ra * u.deg, dec * u.deg, frame="icrs"),
                                      realisations._pwl(amp, self.index, self.e_ref_TeV),
                                      {"ra": ra, "dec": dec, "pix": p, "distance": dist,
                                       "amplitude": float(amp)})

    def describe(self) -> dict:
        return {"kind": "pwl_sky_3d", "sky": _sky_id(self.sky), "band_TeV": list(self.band_TeV),
                "index": self.index, "e_ref_TeV": self.e_ref_TeV,
                "restrict_to_credible": self.restrict_to_credible,
                "apply_k_correction": self.apply_k_correction, "d_min_Mpc": self.d_min_Mpc}
