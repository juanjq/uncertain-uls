"""Monte Carlo realisations: inject a source, Poisson-fake the counts, compute Lambda.

One realisation (index i, base seed s):
    1. the injector draws where and what to inject: injector(rng_i, strength_i), with
       rng_i = np.random.default_rng([s, i]);
    2. counts ~ Poisson(background + source folded through the IRFs), drawn with
       np.random.RandomState(s + i), as `MapDataset.fake(s + i)` in gwuls;
    3. Lambda and the max TS from `lambda_stat.TSPrecomputed` (the source model is removed,
       the background is treated as known).

Both random streams depend only on (s, i), so the output does not depend on `n_jobs` or on
how a batch is split. `run` stores each batch through `uuls.cache.cached`.

2D and 3D injectors place the source uniformly inside the sampled WCS bin (pixel coordinates
of the bin centre plus U(-0.5, 0.5) on each axis, drawn after the bin and the distance), not at
its centre (plan §10.6).

Several fields (tiling, `uuls.fields`): `run_fields` / `simulate_fields` take a list of fields,
each with its own `.reduced` and `.gwmap` on its own geometry. A realisation draws one injection
(the injector's rng as above), fakes the counts of every field (the source only in the fields
whose geometry contains it), and Lambda = max over fields of the field's Lambda. Field k draws
its Poisson counts with RandomState(s + i) for k = 0 and RandomState([s + i, k]) for k > 0, so
the fields are independent and a one-field list gives exactly `run` on that field.

Inputs are duck-typed:
    reduced: .stacked (MapDataset), .runs (list of MapDataset, same geom), .run_gti_s
    gwmap:   .prob, .mask, .cdf, .cdf_valid, .r_grid, .geom (2D WcsGeom)
    injector(rng, strength) -> Injection(coord, spectrum, params)
        spectrum: a gammapy SpectralModel (injected into the stacked dataset), or
        RunSpectra (one tabulated spectrum per run, injected run by run, for light curves).
        params: dict of scalars recorded per realisation (bin, distance, amplitude, ...).
    An injector used with the cache must also have describe() -> dict of plain values.

Ported from gwuls.simulate Sections 3b-5 (`_simulate_realisations`, `_fake_from_runs`,
`_sample_bin_indices`, `_sample_distances_batch`, `sample_sky_and_distance`).
"""
from __future__ import annotations

import hashlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from . import cache, lambda_stat, spectra

VERSION = 2      # bump when the output of `run` changes for the same parameters


# --- what an injector returns ------------------------------------------------------

@dataclass
class RunSpectra:
    """dN/dE per run at `energy_TeV`: values (n_runs, n_e) in cm-2 s-1 TeV-1.

    obs_ids: the run of each row. Needed with several fields (rows are picked by each field's
    runs); without it the rows follow the order of `reduced.runs`."""
    energy_TeV: np.ndarray
    values: np.ndarray
    obs_ids: list | None = None


@dataclass
class Injection:
    coord: SkyCoord
    spectrum: object             # gammapy SpectralModel or RunSpectra
    params: dict = field(default_factory=dict)


# --- samplers ------------------------------------------------------------------------

def bin_centres(geom):
    """(ra, dec) of the bin centres of a WcsGeom's image, deg, each (ny, nx)."""
    c = geom.to_image().get_coord(mode="center")
    return np.asarray(c.lon.to_value(u.deg)), np.asarray(c.lat.to_value(u.deg))


def sample_bins(prob, support, n, rng):
    """n bins drawn with probability ~ prob inside `support` (renormalised).

    prob, support: (ny, nx); rng: np.random.Generator. -> (iy, ix), each (n,) int.
    """
    weights = np.where(support, prob, 0.0).astype(float)
    total = weights.sum()
    if total <= 0:
        raise ValueError("No bin has GW probability > 0 inside the support.")
    flat_idx = rng.choice(weights.size, size=n, p=weights.ravel() / total)
    return np.unravel_index(flat_idx, np.shape(prob))


def sample_distances(r_grid, cdf_rows, rng):
    """One distance per row of `cdf_rows` (n, n_r) by inverse CDF, linear inside the
    bracketing grid interval; a flat CDF there gives its lower edge. -> (n,) Mpc."""
    n_sim, n_r = cdf_rows.shape
    x = rng.uniform(0.0, 1.0, size=n_sim)
    idx = np.clip(np.sum(cdf_rows < x[:, None], axis=1), 1, n_r - 1)
    rows = np.arange(n_sim)
    c_lo, c_hi = cdf_rows[rows, idx - 1], cdf_rows[rows, idx]
    r_lo, r_hi = r_grid[idx - 1], r_grid[idx]
    denom = np.where((c_hi - c_lo) > 0, c_hi - c_lo, 1.0)
    frac = np.clip((x - c_lo) / denom, 0.0, 1.0)
    return r_lo + frac * (r_hi - r_lo)


def sample_sky_2d(gwmap, n, rng):
    """2D draw: bins ~ prob inside the credible mask. -> (iy, ix)."""
    return sample_bins(gwmap.prob, np.asarray(gwmap.mask, bool), n, rng)


def sample_sky_and_distance(gwmap, n, rng, sky_mask=None):
    """3D draw: bin ~ prob over bins with a valid distance CDF (and inside `sky_mask` if
    given), then d ~ p(d | bin). -> dict iy, ix (int arrays), distance (Mpc)."""
    support = np.asarray(gwmap.cdf_valid, dtype=bool)
    if sky_mask is not None:
        support = support & np.asarray(sky_mask, dtype=bool)
    iy, ix = sample_bins(gwmap.prob, support, n, rng)
    d = sample_distances(np.asarray(gwmap.r_grid, float), np.asarray(gwmap.cdf)[iy, ix], rng)
    return {"iy": iy, "ix": ix, "distance": d}


# --- injectors -------------------------------------------------------------------------

def _pwl(amplitude, index, e_ref_TeV):
    from gammapy.modeling.models import PowerLawSpectralModel

    return PowerLawSpectralModel(index=index, amplitude=float(amplitude) * spectra.AMPLITUDE_UNIT,
                                 reference=e_ref_TeV * u.TeV)


def _coord(gwmap, iy, ix):
    ra, dec = bin_centres(gwmap.geom)
    return SkyCoord(ra=ra[iy, ix] * u.deg, dec=dec[iy, ix] * u.deg, frame="icrs")


def jitter_in_bin(geom, iy, ix, rng) -> SkyCoord:
    """A position drawn uniformly inside bin (iy, ix) of `geom`'s image: pixel coordinates
    (ix + U(-0.5, 0.5), iy + U(-0.5, 0.5)), two draws from `rng` (x first). -> SkyCoord (ICRS)."""
    dx, dy = rng.uniform(-0.5, 0.5, size=2)
    lon, lat = geom.to_image().pix_to_coord((np.array([ix + dx]), np.array([iy + dy])))
    ra, dec = (float(np.ravel(u.Quantity(c, u.deg).value)[0]) for c in (lon, lat))
    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")


def _position(gwmap, iy, ix, rng, jitter) -> tuple[SkyCoord, dict]:
    coord = jitter_in_bin(gwmap.geom, iy, ix, rng) if jitter else _coord(gwmap, iy, ix)
    return coord, {"ra": float(coord.ra.deg), "dec": float(coord.dec.deg)}


@dataclass
class PWLInjector2D:
    """Strength = phi0 [cm-2 s-1 TeV-1]; position drawn uniformly inside a bin drawn ~ prob in
    the mask (at the bin centre with jitter=False)."""
    gwmap: object
    index: float = 2.0
    e_ref_TeV: float = 1.0
    jitter: bool = True

    def __call__(self, rng, strength) -> Injection:
        iy, ix = (int(v[0]) for v in sample_sky_2d(self.gwmap, 1, rng))
        coord, pos = _position(self.gwmap, iy, ix, rng, self.jitter)
        return Injection(coord, _pwl(strength, self.index, self.e_ref_TeV),
                         {"iy": iy, "ix": ix, **pos, "amplitude": float(strength)})

    def describe(self) -> dict:
        return {"kind": "pwl_2d", "index": self.index, "e_ref_TeV": self.e_ref_TeV,
                "jitter": self.jitter}


@dataclass
class PWLInjector3D:
    """Strength = band luminosity L0 [erg/s] in `band_TeV`; (bin, d) drawn from the full map
    (or inside the mask), amplitude phi0(L0, d) from `spectra.amplitude_from_luminosity`.
    Position uniform inside the bin (bin centre with jitter=False)."""
    gwmap: object
    band_TeV: tuple = (0.6, 20.0)
    index: float = 2.0
    e_ref_TeV: float = 1.0
    restrict_to_mask: bool = False
    apply_k_correction: bool = False
    jitter: bool = True

    def __call__(self, rng, strength) -> Injection:
        mask = self.gwmap.mask if self.restrict_to_mask else None
        draw = sample_sky_and_distance(self.gwmap, 1, rng, sky_mask=mask)
        iy, ix, d = int(draw["iy"][0]), int(draw["ix"][0]), float(draw["distance"][0])
        if d <= 0:
            raise ValueError("Sampled distance <= 0: r_grid must start above 0.")
        amp = spectra.amplitude_from_luminosity(
            strength, d * u.Mpc, np.asarray(self.band_TeV) * u.TeV, index=self.index,
            e_ref=self.e_ref_TeV * u.TeV, redshift="auto" if self.apply_k_correction else None,
            apply_k_correction=self.apply_k_correction)
        coord, pos = _position(self.gwmap, iy, ix, rng, self.jitter)
        return Injection(coord, _pwl(amp, self.index, self.e_ref_TeV),
                         {"iy": iy, "ix": ix, **pos, "distance": d, "amplitude": float(amp)})

    def describe(self) -> dict:
        return {"kind": "pwl_3d", "band_TeV": list(self.band_TeV), "index": self.index,
                "e_ref_TeV": self.e_ref_TeV, "restrict_to_mask": self.restrict_to_mask,
                "apply_k_correction": self.apply_k_correction, "jitter": self.jitter}


# --- Poisson fakes ---------------------------------------------------------------------

def _sky_model(coord, spectral_model, name="model-simulated"):
    from gammapy.modeling.models import (ConstantTemporalModel, Models, PointSpatialModel,
                                         SkyModel)

    return Models([SkyModel(spatial_model=PointSpatialModel.from_position(coord),
                            spectral_model=spectral_model, temporal_model=ConstantTemporalModel(),
                            name=name)])


def fake_stacked(dataset, coord, spectral_model, seed):
    """Counts ~ Poisson(dataset.npred() with the source), as MapDataset.fake(seed).

    spectral_model None means no source. The dataset's models are left as None.
    -> (counts (n_e, ny, nx) float, expected signal counts).
    """
    if spectral_model is None:
        npred = dataset.npred().data
        n_signal = 0.0
    else:
        dataset.models = _sky_model(coord, spectral_model)
        npred = dataset.npred().data
        n_signal = float(np.nansum(dataset.npred_signal().data))
        dataset.models = None
    data = np.nan_to_num(npred, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    return np.random.RandomState(seed).poisson(data).astype(float), n_signal


def fake_from_runs(dataset, runs, coord, run_spectra, seed, name="model-simulated"):
    """Counts in the stacked `dataset`: its background plus, per run j, the point source with
    spectrum run_spectra.values[j] folded through run j's exposure, PSF and energy dispersion
    and masked by run j's safe mask. With no signal it equals `dataset.fake(seed)` draw for
    draw. -> (counts (n_e, ny, nx), expected signal counts)."""
    from gammapy.modeling.models import (Models, PointSpatialModel, SkyModel,
                                         TemplateSpectralModel)

    data = np.nan_to_num(dataset.npred_background().data.copy(), nan=0.0, posinf=0.0, neginf=0.0)
    spatial = PointSpatialModel.from_position(coord)
    energy = np.asarray(run_spectra.energy_TeV, dtype=float) * u.TeV
    n_signal = 0.0
    for run, values in zip(runs, run_spectra.values):
        if not np.any(values > 0):
            continue
        values = np.maximum(values, values.max() * 1e-30)   # log interpolation needs > 0
        run.models = Models([SkyModel(
            spatial_model=spatial,
            spectral_model=TemplateSpectralModel(energy=energy,
                                                 values=values * spectra.AMPLITUDE_UNIT),
            name=name)])
        signal = run.npred_signal().data
        run.models = None
        if run.mask_safe is not None:
            signal = signal * run.mask_safe.data
        # FFT PSF convolution leaves ~1e-9 of the peak below zero; a Poisson mean must be >= 0.
        signal = np.clip(np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        data = data + signal
        n_signal += float(signal.sum())
    data = np.clip(data, 0.0, None)      # as MapDataset.npred() before fake()
    return np.random.RandomState(seed).poisson(data).astype(float), n_signal


def fake_counts(reduced, injection, strength, seed):
    """Dispatch on the injected spectrum; strength 0 means background only.
    -> (counts, expected signal counts)."""
    if isinstance(injection.spectrum, RunSpectra):
        return fake_from_runs(reduced.stacked, reduced.runs, injection.coord,
                              injection.spectrum, seed)
    model = None if strength == 0 else injection.spectrum
    return fake_stacked(reduced.stacked, injection.coord, model, seed)


# --- one realisation, a batch, the cache ------------------------------------------------

def precompute(reduced, gwmap, correlation_radius_deg=0.1) -> lambda_stat.TSPrecomputed:
    """Kernel, safe mask, correlated background and 2 ln p for `reduced.stacked`."""
    ds = reduced.stacked
    return lambda_stat.TSPrecomputed.build(
        lambda_stat.background_cube(ds), gwmap.prob, gwmap.mask,
        lambda_stat.correlation_radius_bins(ds.counts.geom, correlation_radius_deg),
        safe_mask=lambda_stat.safe_mask_cube(ds))


def realisation(reduced, pre, injector, strength, seed, index, keep_maps=False) -> dict:
    """Realisation `index`: injector rng default_rng([seed, index]), Poisson seed seed + index.

    -> dict: lambda, tsmax, their bins, n_signal, fake_seed, the injector's params, and with
    keep_maps also counts (n_e, ny, nx) and ts (ny, nx).
    """
    rng = np.random.default_rng([int(seed), int(index)])
    inj = injector(rng, float(strength))
    fake_seed = int(seed) + int(index)
    counts, n_signal = fake_counts(reduced, inj, float(strength), fake_seed)
    out = pre.evaluate(counts)
    out.update(inj.params, n_signal=n_signal, fake_seed=fake_seed)
    if keep_maps:
        out.update(counts=counts, ts=pre.ts(counts), coord=inj.coord)
    return out


def _run_indices(reduced, pre, injector, strengths, seed, indices) -> list[dict]:
    """One row per (strength, index). `pre` a list: `reduced` is a list of fields."""
    one = realisation_fields if isinstance(pre, list) else realisation
    return [one(reduced, pre, injector, s, seed, i) for s, i in zip(strengths, indices)]


_WORKER = {}


def _init_worker(reduced, pre, injector):
    _WORKER.update(reduced=reduced, pre=pre, injector=injector)


def _worker_chunk(payload):
    strengths, seed, indices = payload
    return _run_indices(_WORKER["reduced"], _WORKER["pre"], _WORKER["injector"],
                        strengths, seed, indices)


def _run_parallel(reduced, pre, injector, strengths, seed, indices, n_jobs) -> list[dict]:
    """Contiguous chunks over `n_jobs` forkserver workers, merged back in index order."""
    n_chunks = max(n_jobs, min(len(strengths), 4 * n_jobs))
    size = max(1, -(-len(strengths) // n_chunks))
    starts = range(0, len(strengths), size)
    payloads = [(strengths[a:a + size], seed, indices[a:a + size]) for a in starts]
    ctx = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx, initializer=_init_worker,
                             initargs=(reduced, pre, injector)) as pool:
        chunks = list(pool.map(_worker_chunk, payloads))
    return [row for chunk in chunks for row in chunk]


def _stack_rows(rows, strengths) -> dict:
    out = {"strength": np.asarray(strengths, dtype=float)}
    for key in rows[0]:
        out[key] = np.asarray([r[key] for r in rows])
    return out


def _simulate_rows(reduced, pre, injector, strengths, seed, n_jobs, first_index) -> dict:
    strengths = np.atleast_1d(np.asarray(strengths, dtype=float))
    if np.any(strengths < 0):
        raise ValueError("strengths must be >= 0")
    indices = np.arange(first_index, first_index + strengths.size)
    if n_jobs is None or n_jobs <= 1 or strengths.size < 2:
        rows = _run_indices(reduced, pre, injector, strengths, seed, indices)
    else:
        rows = _run_parallel(reduced, pre, injector, strengths, seed, indices, int(n_jobs))
    return _stack_rows(rows, strengths)


def simulate(reduced, gwmap, injector, strengths, seed, n_jobs=1, first_index=0,
             correlation_radius_deg=0.1) -> dict:
    """Realisations first_index ... first_index + n - 1, one per strength, no cache.

    -> dict of (n,) arrays: strength, lambda, tsmax, lambda_iy/ix, tsmax_iy/ix, n_signal,
    fake_seed and the injector's params.
    """
    pre = precompute(reduced, gwmap, correlation_radius_deg)
    return _simulate_rows(reduced, pre, injector, strengths, seed, n_jobs, first_index)


def _sha1(*arrays) -> str:
    h = hashlib.sha1()
    for a in arrays:
        if a is not None:
            h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()


def dataset_fingerprint(ds) -> str:
    """SHA-1 of what a fake depends on: background, exposure, PSF, energy dispersion, masks, WCS."""
    edisp = getattr(ds.edisp, "edisp_map", None)
    return _sha1(ds.npred_background().data, ds.exposure.data,
                 None if ds.psf is None else ds.psf.psf_map.data,
                 None if edisp is None else edisp.data,
                 None if ds.mask_safe is None else ds.mask_safe.data,
                 np.frombuffer(ds.counts.geom.wcs.to_header_string().encode(), dtype=np.uint8))


def gwmap_fingerprint(gwmap) -> str:
    """SHA-1 of prob, mask, distance CDF and r_grid."""
    return _sha1(gwmap.prob, gwmap.mask, gwmap.cdf, gwmap.cdf_valid, gwmap.r_grid)


def cache_params(reduced, gwmap, injector, strengths, seed, first_index,
                 correlation_radius_deg) -> dict:
    """The cache key of a batch. The injector must have describe()."""
    if not hasattr(injector, "describe"):
        raise TypeError("realisations.run with cache=True needs injector.describe() -> dict")
    return {"version": VERSION,
            "stacked": dataset_fingerprint(reduced.stacked),
            "runs": [dataset_fingerprint(r) for r in (getattr(reduced, "runs", None) or [])],
            "gwmap": gwmap_fingerprint(gwmap), "injector": injector.describe(),
            "strengths": np.asarray(strengths, dtype=float), "seed": int(seed),
            "first_index": int(first_index), "correlation_radius_deg": float(correlation_radius_deg)}


def run(reduced, gwmap, injector, strengths, seed, n_jobs=1, first_index=0,
        correlation_radius_deg=0.1, use_cache=True, cache_root=None, verbose=True) -> dict:
    """A batch of realisations, stored in the cache under "realisations".

    reduced, gwmap, injector: see the module docstring. strengths: (n,) >= 0 (0 = background
    only). seed: base seed; realisation k of the batch has index first_index + k. n_jobs does
    not change the result and is not part of the key.
    -> dict of (n,) arrays (see `simulate`).
    """
    def compute():
        return simulate(reduced, gwmap, injector, strengths, seed, n_jobs=n_jobs,
                        first_index=first_index, correlation_radius_deg=correlation_radius_deg)

    if not use_cache:
        return compute()
    params = cache_params(reduced, gwmap, injector, strengths, seed, first_index,
                          correlation_radius_deg)
    return cache.cached("realisations", params, compute, root=cache_root, verbose=verbose)


def batch_function(reduced, gwmap, injector, seed, **run_kwargs):
    """The `simulate_batch(x, round_index, first_index) -> Lambda` that `ulfit.adaptive_fit`
    calls: realisations first_index ... of strengths x, through `run` (so cached).
    run_kwargs go to `run` (n_jobs, correlation_radius_deg, use_cache, cache_root, verbose)."""
    def simulate_batch(x, round_index, first_index):
        return run(reduced, gwmap, injector, x, seed, first_index=first_index, **run_kwargs)["lambda"]

    return simulate_batch


# --- several fields: one injection, every field faked, Lambda = max over fields --------------

def field_contains(field, coord) -> bool:
    """True if `coord` falls inside the image of `field.reduced.geom` (and within 90 deg of its
    centre, so the far side of the sky never projects into the map)."""
    geom = field.reduced.geom.to_image()
    if coord.separation(geom.center_skydir).deg >= 90:
        return False
    return bool(np.all(geom.contains(coord)))


def field_poisson_seed(fake_seed, k):
    """Poisson seed of field k: fake_seed for k = 0 (a one-field list is `run`), else [fake_seed, k]."""
    return int(fake_seed) if k == 0 else [int(fake_seed), int(k)]


def _field_injection(inj, field) -> Injection:
    """The injection as seen by one field: RunSpectra rows picked by the field's obs_ids."""
    spec = inj.spectrum
    if isinstance(spec, RunSpectra):
        if spec.obs_ids is None:
            raise ValueError("RunSpectra with several fields needs obs_ids")
        row = {int(o): j for j, o in enumerate(spec.obs_ids)}
        ids = [int(o) for o in field.reduced.obs_ids]
        spec = RunSpectra(spec.energy_TeV, np.asarray(spec.values)[[row[o] for o in ids]], ids)
    return Injection(inj.coord, spec, inj.params)


def evaluate_field(pre, counts) -> dict:
    """pre.evaluate(counts), or Lambda = TS max = -inf (bins -1) for a field whose geometry has
    no bin of the GW region (it can never hold the maximum)."""
    if not np.any(pre.mask):
        return {"lambda": -np.inf, "lambda_iy": -1, "lambda_ix": -1,
                "tsmax": -np.inf, "tsmax_iy": -1, "tsmax_ix": -1}
    return pre.evaluate(counts)


def precompute_fields(fields, correlation_radius_deg=0.1) -> list:
    """`precompute` of every field (its stacked dataset and its own GW map). -> list."""
    return [precompute(f.reduced, f.gwmap, correlation_radius_deg) for f in fields]


def realisation_fields(fields, pres, injector, strength, seed, index, keep_maps=False) -> dict:
    """Realisation `index` over several fields.

    One injection from default_rng([seed, index]). Field k: counts faked with
    `field_poisson_seed(seed + index, k)`, with the source if the field's geometry contains it,
    background only otherwise; then that field's Lambda (its own GW region and p_b).
    -> dict: lambda, tsmax (max over fields), lambda_field / tsmax_field (index of the field
    reaching it), lambda_iy/ix and tsmax_iy/ix (bins in that field), lambda_fields (n_fields,),
    n_signal (sum over fields), n_fields_containing, fake_seed, the injector's params; with
    keep_maps also counts and ts (lists over fields) and coord.
    """
    rng = np.random.default_rng([int(seed), int(index)])
    inj = injector(rng, float(strength))
    fake_seed = int(seed) + int(index)
    evals, counts_k, n_signal, n_in = [], [], 0.0, 0
    for k, (f, pre) in enumerate(zip(fields, pres)):
        pseed = field_poisson_seed(fake_seed, k)
        if field_contains(f, inj.coord):
            counts, n_k = fake_counts(f.reduced, _field_injection(inj, f), float(strength), pseed)
            n_in += 1
        else:
            counts, n_k = fake_stacked(f.reduced.stacked, inj.coord, None, pseed)
        evals.append(evaluate_field(pre, counts))
        counts_k.append(counts)
        n_signal += n_k
    lam = np.array([e["lambda"] for e in evals])
    k_lam = int(np.argmax(lam))
    k_ts = int(np.argmax([e["tsmax"] for e in evals]))
    out = {"lambda": evals[k_lam]["lambda"], "lambda_iy": evals[k_lam]["lambda_iy"],
           "lambda_ix": evals[k_lam]["lambda_ix"], "lambda_field": k_lam,
           "tsmax": evals[k_ts]["tsmax"], "tsmax_iy": evals[k_ts]["tsmax_iy"],
           "tsmax_ix": evals[k_ts]["tsmax_ix"], "tsmax_field": k_ts, "lambda_fields": lam}
    out.update(inj.params, n_signal=n_signal, n_fields_containing=n_in, fake_seed=fake_seed)
    if keep_maps:
        out.update(counts=counts_k, ts=[p.ts(c) for p, c in zip(pres, counts_k)], coord=inj.coord)
    return out


def simulate_fields(fields, injector, strengths, seed, n_jobs=1, first_index=0,
                    correlation_radius_deg=0.1) -> dict:
    """`simulate` over several fields, no cache. -> dict of (n,) arrays (see
    `realisation_fields`; lambda_fields is (n, n_fields))."""
    pres = precompute_fields(fields, correlation_radius_deg)
    return _simulate_rows(list(fields), pres, injector, strengths, seed, n_jobs, first_index)


def cache_params_fields(fields, injector, strengths, seed, first_index,
                        correlation_radius_deg) -> dict:
    """The cache key of a multi-field batch: per field the fingerprints of `cache_params`."""
    if not hasattr(injector, "describe"):
        raise TypeError("realisations.run_fields with cache=True needs injector.describe() -> dict")
    return {"version": VERSION,
            "fields": [{"stacked": dataset_fingerprint(f.reduced.stacked),
                        "runs": [dataset_fingerprint(r) for r in (getattr(f.reduced, "runs", None) or [])],
                        "gwmap": gwmap_fingerprint(f.gwmap)} for f in fields],
            "injector": injector.describe(), "strengths": np.asarray(strengths, dtype=float),
            "seed": int(seed), "first_index": int(first_index),
            "correlation_radius_deg": float(correlation_radius_deg)}


def run_fields(fields, injector, strengths, seed, n_jobs=1, first_index=0,
               correlation_radius_deg=0.1, use_cache=True, cache_root=None, verbose=True) -> dict:
    """`run` over several fields (list of objects with .reduced and .gwmap, e.g. uuls.fields.Field),
    stored in the cache under "realisations_fields". -> dict of arrays (see `simulate_fields`)."""
    def compute():
        return simulate_fields(fields, injector, strengths, seed, n_jobs=n_jobs,
                               first_index=first_index, correlation_radius_deg=correlation_radius_deg)

    if not use_cache:
        return compute()
    params = cache_params_fields(fields, injector, strengths, seed, first_index, correlation_radius_deg)
    return cache.cached("realisations_fields", params, compute, root=cache_root, verbose=verbose)


def batch_function_fields(fields, injector, seed, **run_kwargs):
    """`batch_function` for several fields, through `run_fields`."""
    def simulate_batch(x, round_index, first_index):
        return run_fields(fields, injector, x, seed, first_index=first_index, **run_kwargs)["lambda"]

    return simulate_batch
