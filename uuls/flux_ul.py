"""Study 1: the frequentist 2D flux upper limit of a case, end to end.

    res = flux_ul("sim_wobble")                 # dict, also written to outputs/results/1_flux_ul/<case>.json

Steps (each expensive one cached by the tool it calls):
    1. data: simulated DL3 (simdata.simulate, cache "simulst_dl3") or the real DL3 folder;
    2. fields (fields.build_fields): GW map and reduced data per field (caches "gwmap", "reduce");
    3. observed Lambda (fields.observed_lambda), Lambda = max over fields of the max over each
       field's ROI = GW 95% mask & observed region;
    4. background-only Lambda: realisations at zero strength (cache "realisations_fields");
    5. adaptive probit fit of P(Lambda > Lambda*) vs phi0 (ulfit.adaptive_fit; every batch
       cached, so a rerun only redoes the fits);
    6. closure: an independent sample at phi0_UL, whose detection fraction must be CL.

Single- and multi-field cases go through the same path (`realisations.run_fields`; a one-field
list is `realisations.run`). The injector is what differs:
    one field:  realisations.PWLInjector2D(gwmap, reduced=...), a bin ~ prob inside the ROI;
    several:    fields.PWLSkyInjector2D(sky, support=roi_pixels(...)), a HEALPix pixel ~ prob
                inside the credible region and inside the ROI of at least one field.
Both make the limit one for a source inside the observed part of the GW region (plan §10.9).
The GW probability of that part (`roi_gw_prob`) is reported with the limit.

Seeds: background, fit and closure realisations use base seeds 10^7 apart (seed + i is the
Poisson seed of realisation i, so close base seeds would share counts between the samples).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import astropy.units as u
import astropy_healpix as ah
import numpy as np
from astropy.coordinates import SkyCoord

from . import cases, fields as fl, gwmap as gwm, lambda_stat, paths, realisations, simdata, spectra, ulfit

STUDY = "1_flux_ul"
BAND_TEV = (0.6, 20.0)
SEED_STRIDE = 10_000_000          # offsets between the background, fit and closure seeds
DEFAULTS = {"cl": 0.95, "index": 2.0, "x_lo": 1e-13, "x_hi": 3e-10, "n_bkg": 1000,
            "n_scout": 300, "n_refine": 500, "max_sims": 1500, "precision": 0.03,
            "n_closure": 300, "seed": 0, "correlation_radius_deg": 0.1, "roi": "observed",
            "production": "2025_2p5"}


def dl3_dir(case, production="2025_2p5", verbose=False) -> Path:
    """DL3 folder of a case. simulst: simdata.simulate of the case pointings (cached); real:
    paths.REAL_DL3 (FileNotFoundError if absent)."""
    if case.data_kind == "simulst":
        return simdata.simulate(case.pointings(), nsb=case.nsb, seed=case.seed,
                                production=production, verbose=verbose)
    if not paths.REAL_DL3.exists():
        raise FileNotFoundError(f"real DL3 folder {paths.REAL_DL3} does not exist")
    return paths.REAL_DL3


def roi_pixels(sky, fields, credible=0.95, roi="observed") -> np.ndarray:
    """HEALPix pixels (bool per pixel of `sky`) inside the greedy `credible` region whose centre is
    in the ROI of at least one field (lambda_stat.roi on that field's grid)."""
    out = gwm.credible_levels(sky) <= credible
    level, ipix = ah.uniq_to_level_ipix(sky.uniq[out])
    lon, lat = ah.healpix_to_lonlat(ipix, nside=ah.level_to_nside(level), order="nested")
    c = SkyCoord(lon, lat, frame="icrs")
    hit = np.zeros(c.shape, bool)
    for f in fields:
        g = f.gwmap.geom.to_image() if hasattr(f.gwmap, "geom") else f.reduced.geom.to_image()
        m = lambda_stat.roi(f.gwmap.mask, f.reduced.stacked, roi)
        sel = (c.separation(g.center_skydir).deg < 90) & g.contains(c)
        if sel.any():
            ix, iy = (np.asarray(v, int) for v in g.coord_to_idx(c[sel]))
            hit[np.flatnonzero(sel)] |= m[iy, ix]
    out[np.flatnonzero(out)] = hit
    return out


def make_injector(fields, sky=None, index=2.0, roi="observed", credible=0.95):
    """One field: PWLInjector2D on its ROI. Several: PWLSkyInjector2D restricted to roi_pixels."""
    if len(fields) == 1:
        f = fields[0]
        return realisations.PWLInjector2D(f.gwmap, index=index, reduced=f.reduced, roi=roi)
    return fl.PWLSkyInjector2D(sky, credible=credible, index=index,
                               support=roi_pixels(sky, fields, credible, roi))


def roi_gw_prob(fields, sky=None, roi="observed", credible=0.95) -> float:
    """GW probability inside the ROI: HEALPix sum over roi_pixels (overlaps counted once); with
    one field and no sky, the sum of the field's binned prob over its ROI."""
    if sky is None:
        f = fields[0]
        return float(np.sum(f.gwmap.prob[lambda_stat.roi(f.gwmap.mask, f.reduced.stacked, roi)]))
    return float(np.sum(sky.prob[roi_pixels(sky, fields, credible, roi)]))


def prepare(case, production="2025_2p5", index=2.0, roi="observed", verbose=False) -> dict:
    """Data, fields and injector of a case. -> dict case, dl3_dir, fields, sky, injector."""
    case = cases.load_case(case) if isinstance(case, (str, Path)) else case
    d = dl3_dir(case, production, verbose)
    fields = fl.build_fields(case, d, verbose=verbose)
    sky = gwm.read_healpix(case.skymap)
    return {"case": case, "dl3_dir": d, "fields": fields, "sky": sky,
            "injector": make_injector(fields, sky, index, roi)}


def background_lambda(fields, injector, n, seed, n_jobs=1, **kw) -> np.ndarray:
    """Lambda of n background-only realisations (strength 0; cached). -> (n,)."""
    return realisations.run_fields(fields, injector, np.zeros(int(n)), seed + SEED_STRIDE,
                                   n_jobs=n_jobs, **kw)["lambda"]


def fit_ul(fields, injector, lambda_obs, lambda_bkg, seed, n_jobs=1, cl=0.95, x_lo=1e-13,
           x_hi=3e-10, n_scout=300, n_refine=500, max_sims=1500, precision=0.03, verbose=True,
           **kw) -> ulfit.ULResult:
    """ulfit.adaptive_fit on phi0 [cm-2 s-1 TeV-1] with cached batches of run_fields."""
    batch = realisations.batch_function_fields(fields, injector, seed + 2 * SEED_STRIDE,
                                               n_jobs=n_jobs, verbose=False, **kw)
    return ulfit.adaptive_fit(batch, x_lo, x_hi, lambda_obs, lambda_bkg, cl=cl, n_scout=n_scout,
                              n_refine=n_refine, max_sims=max_sims, precision=precision,
                              design_seed=seed, label="phi0", unit="cm-2 s-1 TeV-1",
                              verbose=verbose)


def closure_lambda(fields, injector, x_ul, n, seed, n_jobs=1, **kw) -> np.ndarray:
    """Lambda of n realisations at strength x_ul, independent of the fit sample (cached)."""
    return realisations.run_fields(fields, injector, np.full(int(n), float(x_ul)),
                                   seed + 3 * SEED_STRIDE, n_jobs=n_jobs, **kw)["lambda"]


def flux_values(amplitudes, band_tev=BAND_TEV, index=2.0) -> dict:
    """Photon (cm-2 s-1) and energy (erg cm-2 s-1) flux in the band of PWL amplitudes."""
    edges = np.asarray(band_tev, float) * u.TeV
    a = np.asarray(amplitudes, float)
    return {"photon_flux": spectra.pwl_photon_flux(a, edges, index).to_value("cm-2 s-1"),
            "energy_flux": spectra.pwl_energy_flux(a, edges, index).to_value("erg cm-2 s-1")}


def flux_ul_fields(fields, injector, *, cl=0.95, index=2.0, x_lo=1e-13, x_hi=3e-10, n_bkg=1000,
                   n_scout=300, n_refine=500, max_sims=1500, precision=0.03, n_closure=300, seed=0,
                   correlation_radius_deg=0.1, roi="observed", n_jobs=1, verbose=True,
                   cache_root=None) -> tuple:
    """Observed Lambda, background, fit and closure on given fields. -> (summary dict, details)
    with details = {"observed": dict, "lambda_bkg": array, "ul": ULResult, "lambda_closure": array}.
    cache_root: cache folder of the realisations (tests)."""
    kw = {"correlation_radius_deg": correlation_radius_deg, "roi": roi}
    rkw = {**kw, "cache_root": cache_root}
    timing = {}
    t = time.time()
    obs = fl.observed_lambda(fields, **kw)
    timing["observed_s"] = time.time() - t
    t = time.time()
    lam_bkg = background_lambda(fields, injector, n_bkg, seed, n_jobs, verbose=verbose, **rkw)
    timing["background_s"] = time.time() - t
    target, p, sig = ulfit.target_lambda(obs["lambda"], lam_bkg)
    t = time.time()
    res = fit_ul(fields, injector, obs["lambda"], lam_bkg, seed, n_jobs, cl, x_lo, x_hi, n_scout,
                 n_refine, max_sims, precision, verbose, **rkw)
    timing["fit_s"] = time.time() - t
    t = time.time()
    lam_clo = closure_lambda(fields, injector, res.x_ul, n_closure, seed, n_jobs, verbose=verbose, **rkw)
    clo = ulfit.closure_pull(res, lam_clo)
    timing["closure_s"] = time.time() - t

    amps = [res.x_ul, res.x_ul_lo_1sigma, res.x_ul_hi_1sigma, res.x_ul_lo_2sigma, res.x_ul_hi_2sigma]
    fx = flux_values(amps, BAND_TEV, index)
    names = ["ul", "lo_1sigma", "hi_1sigma", "lo_2sigma", "hi_2sigma"]
    summary = {
        "n_fields": len(fields),
        "obs_ids": [int(o) for f in fields for o in f.reduced.obs_ids] if hasattr(fields[0].reduced, "obs_ids") else None,
        "lambda_obs": float(obs["lambda"]), "lambda_obs_field": int(obs["field"]),
        "tsmax_obs": float(obs["tsmax"]),
        "lambda_obs_fields": [float(v) for v in obs["lambda_fields"]],
        "background": {"n": int(np.isfinite(lam_bkg).sum()), "median": float(np.median(lam_bkg)),
                       "p_value": p, "significance": sig},
        "target": float(target), "cl": cl, "index": index, "band_TeV": list(BAND_TEV),
        "amplitude": dict(zip(names, map(float, amps)), unit="cm-2 s-1 TeV-1 at 1 TeV"),
        "photon_flux": dict(zip(names, map(float, fx["photon_flux"])), unit="cm-2 s-1"),
        "energy_flux": dict(zip(names, map(float, fx["energy_flux"])), unit="erg cm-2 s-1"),
        "log10_ul_err_dex": float(res.log10_x_ul_err),
        "fit": {"width_dex": float(res.width_dex), "f0": float(res.f0), "f0_bkg": float(res.f0_bkg),
                "ceiling": float(res.ceiling), "gof_chi2": float(res.gof["chi2"]),
                "gof_ndof": int(res.gof["ndof"]), "gof_pvalue": float(res.gof["pvalue"]),
                "converged": bool(res.converged), "at_bracket_edge": bool(res.at_bracket_edge),
                "bracket_warning": res.bracket_warning, "n_sim": int(res.n_sim_total),
                "n_rounds": len(res.rounds)},
        "closure": {k: (int(v) if k == "n" else float(v)) for k, v in clo.items()},
        "settings": {"x_lo": x_lo, "x_hi": x_hi, "n_bkg": n_bkg, "n_scout": n_scout,
                     "n_refine": n_refine, "max_sims": max_sims, "precision": precision,
                     "n_closure": n_closure, "seed": seed, **kw},
        "timing_s": {k: round(v, 1) for k, v in timing.items()},
    }
    return summary, {"observed": obs, "lambda_bkg": lam_bkg, "ul": res, "lambda_closure": lam_clo}


def result_path(name) -> Path:
    return paths.RESULTS / STUDY / f"{name}.json"


def flux_ul(case, n_jobs=1, write=True, verbose=True, details=False, **options):
    """The whole 2D flux UL of a case (name, path or cases.Case).

    options: overrides of DEFAULTS (cl, index, x_lo, x_hi, n_bkg, n_scout, n_refine, max_sims,
    precision, n_closure, seed, correlation_radius_deg, roi, production).
    Out: plain dict (case, data, roi_gw_prob, Lambda, background, UL as amplitude / photon flux /
    energy flux in 0.6-20 TeV with 1 and 2 sigma MC intervals, fit diagnostics, closure, timing),
    written to outputs/results/1_flux_ul/<case>.json. With details=True, (dict, details) where
    details adds the prepared objects (`prepare`) to those of `flux_ul_fields`.
    """
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown options {sorted(unknown)}")
    opt = {**DEFAULTS, **options}
    t = time.time()
    prep = prepare(case, opt["production"], opt["index"], opt["roi"], verbose=False)
    t_prep = time.time() - t
    c, f = prep["case"], prep["fields"]
    fopt = {k: v for k, v in opt.items() if k != "production"}
    summary, det = flux_ul_fields(f, prep["injector"], n_jobs=n_jobs, verbose=verbose, **fopt)
    out = {"case": c.name, "skymap": c.skymap.name, "t0": c.t0().isot, "data": c.data_kind,
           "production": opt["production"] if c.data_kind == "simulst" else None,
           "dl3_dir": str(prep["dl3_dir"]),
           "background_method": (c.analysis.get("reduce") or {}).get("background", "ring"),
           "roi_gw_prob": roi_gw_prob(f, prep["sky"], opt["roi"]), **summary}
    out["timing_s"] = {"prepare_s": round(t_prep, 1), **summary["timing_s"]}
    if write:
        p = result_path(c.name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
    return (out, {**prep, **det}) if details else out


def load_result(name) -> dict:
    """The JSON written by flux_ul for case `name`."""
    return json.loads(result_path(name).read_text())
