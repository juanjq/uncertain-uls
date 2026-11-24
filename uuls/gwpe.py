"""GW parameter-estimation samples: a raw GWTC release reduced to one light file per alert.

    path = standardize(raw_path, "S240615dg", None)   # data/gw/pe_standardized/S240615dg.h5
    samples = load_samples("S240615dg")               # astropy Table, angles in degrees

Ported from gwuls.gw_pe (same file format, `FORMAT_VERSION = 1`). A standardized file keeps the
same number of posterior samples from each waveform analysis (`posterior/analysis` says which),
so the concatenation is the equal-weight mixture the catalog papers quote. It also keeps the PE
prior samples of (d_L, theta_jn) and, per analysis, the approximant and prior definitions. Angles
are stored in radians; `load_samples` returns degrees and the jet viewing angle
theta_v = min(theta_jn, 180 - theta_jn) (the jet is bipolar and follows J).

Trigger time and network SNR. GWTC-5 analyses carry `config_file/config/trigger-time` and a
`network_matched_filter_snr` posterior column. The combined analysis of a GWTC-2.1/3 release
(`C01:Mixed`, a mixture of two waveform runs) has neither: its `config_file` group is empty and
its posterior has no SNR column. For those, the trigger time is taken from the other analysis
groups of the same file (same event, same trigger), then from `skymap/meta_data/gps_time`, and as
a last resort the median of the `geocent_time` posterior. The network SNR median is taken from
the analyses of the file that have the column. Both are written as file attributes with the
analysis they came from (`trigger_time_source`, `network_snr_median`, `network_snr_source`).
"""
import csv
import datetime as _dt
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np
from astropy.table import Table

from . import paths

FORMAT_VERSION = 1
COLUMNS = ("ra", "dec", "luminosity_distance", "redshift", "theta_jn", "iota",
           "mass_1_source", "mass_2_source", "chirp_mass_source", "chi_eff",
           "lambda_1", "lambda_2", "network_matched_filter_snr")
PRIOR_COLUMNS = ("luminosity_distance", "theta_jn")
_NON_ANALYSIS_KEYS = {"history", "version"}
_GW_NAME_RE = re.compile(r"(GW\d{6}_\d{6})")
_CATALOG_RE = re.compile(r"IGWN-(GWTC)(\d+)p(\d+)")
_META_PATHS = {
    "approximant": "meta_data/meta_data/approximant",
    "f_ref": "meta_data/meta_data/f_ref",
    "cosmology": "meta_data/meta_data/cosmology",
    "ifos": "meta_data/meta_data/IFOs",
    "trigger_time": "config_file/config/trigger-time",
    "trigger_time_meta": "meta_data/other/config_file/trigger_time",
    "prior_luminosity_distance": "priors/analytic/luminosity_distance",
    "prior_theta_jn": "priors/analytic/theta_jn",
}


def fold_viewing_angle(theta_jn_deg):
    """Jet viewing angle (deg, 0-90) from theta_jn (deg, 0-180): angle to the nearer jet axis."""
    theta = np.asarray(theta_jn_deg, dtype=float)
    return np.minimum(theta, 180.0 - theta)


def _decode(value):
    """h5py scalar or one-element dataset -> plain str / number."""
    if isinstance(value, np.ndarray):
        value = value.ravel()
        if value.size != 1:
            return [_decode(v) for v in value]
        value = value[0]
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode(errors="replace")
    return value.item() if isinstance(value, np.generic) else value


def _read_optional(group, path):
    try:
        return _decode(group[path][()])
    except (KeyError, TypeError, ValueError):
        return None


def _analysis_groups(f) -> list:
    return [k for k in f.keys() if k not in _NON_ANALYSIS_KEYS and "posterior_samples" in f[k]]


def _choose_analyses(available, waveform_label) -> list:
    """All analyses (None), one label (str) or a list of labels, checked against the file."""
    chosen = available if waveform_label is None else (
        [waveform_label] if isinstance(waveform_label, str) else list(waveform_label))
    missing = [a for a in chosen if a not in available]
    if missing:
        raise KeyError(f"analyses {missing} not in the release; available: {available}")
    return chosen


def _read_analysis(group, take, columns):
    """Posterior columns (subset `take`), prior samples and metadata of one analysis group."""
    ps = group["posterior_samples"]
    post = {c: np.asarray(ps[c], dtype=float)[take] for c in columns if c in (ps.dtype.names or ())}
    prior_group = group.get("priors/samples")
    prior = {}
    if prior_group is not None and all(c in prior_group for c in PRIOR_COLUMNS):
        prior = {c: np.asarray(prior_group[c][()], dtype=float) for c in PRIOR_COLUMNS}
    meta = {k: _read_optional(group, p) for k, p in _META_PATHS.items()}
    return post, prior, meta


def _collect(raw_path, chosen, n_per_analysis, seed):
    """Equal-size random subsets of every chosen analysis, and their priors and metadata."""
    rng = np.random.default_rng(seed)
    with h5py.File(raw_path, "r") as f:
        n_avail = {a: int(f[a]["posterior_samples"].shape[0]) for a in chosen}
        n_keep = int(min(n_per_analysis, *n_avail.values()))
        runs = []
        for a in chosen:
            take = np.sort(rng.choice(n_avail[a], size=n_keep, replace=False))
            post, prior, meta = _read_analysis(f[a], take, COLUMNS)
            meta.update(n_available=n_avail[a], n_kept=n_keep,
                        n_prior=len(next(iter(prior.values()))) if prior else 0)
            runs.append((post, prior, meta))
    return runs, n_keep


def _write(out_path, attrs, runs, chosen):
    """Write the standardized file atomically (tmp file, then rename)."""
    columns = [c for c in COLUMNS if all(c in post for post, _, _ in runs)]
    attrs = dict(attrs, columns_json=json.dumps(columns))
    kw = dict(compression="gzip", compression_opts=9, shuffle=True, track_times=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as out:
        out.attrs.update(attrs)
        g = out.create_group("posterior")
        for c in columns:
            g.create_dataset(c, data=np.concatenate([p[c] for p, _, _ in runs]).astype(np.float32), **kw)
        g.create_dataset("analysis", data=np.concatenate(
            [np.full(len(p[columns[0]]), k, dtype=np.uint8) for k, (p, _, _) in enumerate(runs)]), **kw)
        with_prior = [(k, pr) for k, (_, pr, _) in enumerate(runs) if pr]
        if with_prior:
            gp = out.create_group("prior")
            for c in PRIOR_COLUMNS:
                gp.create_dataset(c, data=np.concatenate([pr[c] for _, pr in with_prior]).astype(np.float32), **kw)
            gp.create_dataset("analysis", data=np.concatenate(
                [np.full(len(pr[PRIOR_COLUMNS[0]]), k, dtype=np.uint8) for k, pr in with_prior]), **kw)
        ga = out.create_group("analyses")
        for a, (_, _, m) in zip(chosen, runs):
            ga.create_group(a).attrs.update({k: ("" if v is None else v) for k, v in m.items()})
    os.replace(tmp, out_path)


_TRIGGER_PATHS = ("config_file/config/trigger-time", "meta_data/other/config_file/trigger_time",
                  "skymap/meta_data/gps_time")
_SNR = "network_matched_filter_snr"


def _finite(value):
    """float(value) if it is a finite number, else None."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _event_times_and_snr(raw_path, chosen) -> dict:
    """Trigger time (GPS) and network SNR median of the event, and where each was found.

    Trigger time: the median over the chosen analyses of `config_file/config/trigger-time`; if
    none has it, the first of `_TRIGGER_PATHS` found in any analysis of the file (chosen ones
    first); else the median `geocent_time` of the chosen posteriors; else NaN.
    SNR: NaN when every chosen posterior has the column (the index takes it from the kept
    samples); otherwise the median over the analyses of the file that have it, of their medians.
    Out: dict trigger_time_gps, trigger_time_source, network_snr_median, network_snr_source.
    """
    out = {"trigger_time_gps": np.nan, "trigger_time_source": "",
           "network_snr_median": np.nan, "network_snr_source": ""}
    with h5py.File(raw_path, "r") as f:
        groups = list(chosen) + [a for a in _analysis_groups(f) if a not in chosen]
        trig = [t for t in (_finite(_read_optional(f[a], _TRIGGER_PATHS[0])) for a in chosen) if t is not None]
        if trig:
            out.update(trigger_time_gps=float(np.median(trig)), trigger_time_source=f"{_TRIGGER_PATHS[0]} (chosen)")
        else:
            found = ((a, p, _finite(_read_optional(f[a], p))) for p in _TRIGGER_PATHS for a in groups)
            hit = next(((a, p, t) for a, p, t in found if t is not None), None)
            if hit is not None:
                out.update(trigger_time_gps=hit[2], trigger_time_source=f"{hit[0]}/{hit[1]}")
            else:
                geo = [np.asarray(f[a]["posterior_samples"]["geocent_time"], float) for a in chosen
                       if "geocent_time" in (f[a]["posterior_samples"].dtype.names or ())]
                if geo:
                    out.update(trigger_time_gps=float(np.median(np.concatenate(geo))),
                               trigger_time_source="median posterior geocent_time")
        has = [a for a in groups if _SNR in (f[a]["posterior_samples"].dtype.names or ())]
        if has and not all(a in has for a in chosen):
            meds = [float(np.median(f[a]["posterior_samples"][_SNR])) for a in has]
            out.update(network_snr_median=float(np.median(meds)), network_snr_source=";".join(has))
    return out


def standardize(raw_release_path, superevent, waveform_label=None, n_per_analysis=5000,
                seed=0, out_dir=None, overwrite=False) -> Path:
    """Raw GWTC "*-combined_PEDataRelease.hdf5" -> <out_dir>/<superevent>.h5.

    In: raw release path; superevent (alert name, e.g. "S240615dg"); waveform_label: None for
    every analysis in the file, one label ("C00:IMRPhenomXPHM-SpinTaylor") or a list of labels;
    n_per_analysis (capped at the smallest analysis); seed of the random subset; out_dir
    (default paths.GW_PE). An existing file is kept unless overwrite. Also rewrites index.csv.
    Out: path of the standardized file.
    """
    raw_path = Path(raw_release_path)
    out_path = Path(out_dir or paths.GW_PE) / f"{superevent}.h5"
    if out_path.exists() and not overwrite:
        return out_path
    with h5py.File(raw_path, "r") as f:
        chosen = _choose_analyses(_analysis_groups(f), waveform_label)
    runs, n_keep = _collect(raw_path, chosen, n_per_analysis, seed)
    gw = _GW_NAME_RE.search(raw_path.name)
    cat = _CATALOG_RE.search(raw_path.name)
    event = _event_times_and_snr(raw_path, chosen)
    attrs = {
        "format_version": FORMAT_VERSION, "superevent_id": superevent,
        "gw_name": gw.group(1) if gw else raw_path.stem,
        "catalog": f"{cat.group(1)}-{cat.group(2)}.{cat.group(3)}" if cat else "unknown",
        "source_file": raw_path.name, "analyses_json": json.dumps(chosen),
        "n_per_analysis": n_keep, "seed": seed,
        **event,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": ("Equal-weight mixture: the same number of posterior samples from each "
                 "analysis in analyses_json; posterior/analysis says which. Angles in rad."),
    }
    _write(out_path, attrs, runs, chosen)
    rebuild_index(out_path.parent)
    return out_path


def _resolve(name) -> Path:
    p = Path(name)
    return p if p.suffix == ".h5" else paths.GW_PE / f"{name}.h5"


def load_samples(name) -> Table:
    """Posterior samples of one alert, by name ("S240615dg") or path to a standardized file.

    Out: Table with ra, dec (deg), luminosity_distance (Mpc), theta_jn (deg, 0-180), theta_v
    (deg, 0-90) and analysis (index into meta["analyses"]). meta also holds superevent_id,
    gw_name, catalog, trigger_time_gps and path.
    """
    p = _resolve(name)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run uuls.gwpe.standardize on the raw release first")
    with h5py.File(p, "r") as f:
        g, a = f["posterior"], dict(f.attrs)
        t = Table({
            "ra": np.degrees(g["ra"][()].astype(float)),
            "dec": np.degrees(g["dec"][()].astype(float)),
            "luminosity_distance": g["luminosity_distance"][()].astype(float),
            "theta_jn": np.degrees(g["theta_jn"][()].astype(float)),
            "analysis": g["analysis"][()],
        })
    t["theta_v"] = fold_viewing_angle(t["theta_jn"])
    for c, unit in (("ra", "deg"), ("dec", "deg"), ("luminosity_distance", "Mpc"),
                    ("theta_jn", "deg"), ("theta_v", "deg")):
        t[c].unit = unit
    t.meta.update({k: _decode(a[k]) for k in ("superevent_id", "gw_name", "catalog", "trigger_time_gps")})
    t.meta.update(analyses=json.loads(a["analyses_json"]), path=str(p))
    return t


def theta_v_given_distance(samples: Table, d_edges) -> list:
    """theta_v samples (deg) in each distance bin [d_edges[k], d_edges[k+1]) (Mpc)."""
    k = np.digitize(samples["luminosity_distance"], d_edges) - 1
    return [np.asarray(samples["theta_v"][k == i]) for i in range(len(d_edges) - 1)]


_INDEX_FIELDS = ("superevent_id", "gw_name", "catalog", "n_analyses", "n_per_analysis",
                 "distance_Mpc_median", "distance_Mpc_q05", "distance_Mpc_q95",
                 "viewing_angle_deg_median", "viewing_angle_deg_q05", "viewing_angle_deg_q95",
                 "p_viewing_angle_below_30deg", "mass_1_source_median", "mass_2_source_median",
                 "p_m2_below_3Msun", "network_snr_median", "trigger_time_gps", "analyses",
                 "source_file", "created_utc")


def _median(post, col, fmt):
    return fmt.format(np.median(post[col])) if col in post else ""


def _snr_attr(attrs) -> str:
    """Network SNR median stored as a file attribute (analyses without the column), or ""."""
    x = _finite(attrs.get("network_snr_median"))
    return "" if x is None else f"{x:.1f}"


def _index_row(p: Path) -> dict:
    """One index.csv row, same fields and formats as the gwuls index."""
    with h5py.File(p, "r") as f:
        a = {k: _decode(v) for k, v in f.attrs.items()}
        post = {c: f["posterior"][c][()].astype(float) for c in f["posterior"]}
    theta_v = fold_viewing_angle(np.degrees(post["theta_jn"]))
    d = np.percentile(post["luminosity_distance"], [50, 5, 95])
    v = np.percentile(theta_v, [50, 5, 95])
    analyses = json.loads(a["analyses_json"])
    m2 = f"{np.mean(post['mass_2_source'] < 3.0):.3f}" if "mass_2_source" in post else ""
    return {"superevent_id": a["superevent_id"], "gw_name": a["gw_name"], "catalog": a["catalog"],
            "n_analyses": len(analyses), "n_per_analysis": a.get("n_per_analysis", ""),
            "distance_Mpc_median": f"{d[0]:.4g}", "distance_Mpc_q05": f"{d[1]:.4g}",
            "distance_Mpc_q95": f"{d[2]:.4g}", "viewing_angle_deg_median": f"{v[0]:.1f}",
            "viewing_angle_deg_q05": f"{v[1]:.1f}", "viewing_angle_deg_q95": f"{v[2]:.1f}",
            "p_viewing_angle_below_30deg": f"{np.mean(theta_v < 30):.3f}",
            "mass_1_source_median": _median(post, "mass_1_source", "{:.3g}"),
            "mass_2_source_median": _median(post, "mass_2_source", "{:.3g}"),
            "p_m2_below_3Msun": m2,
            "network_snr_median": _median(post, _SNR, "{:.1f}") or _snr_attr(a),
            "trigger_time_gps": f"{a.get('trigger_time_gps', np.nan):.3f}",
            "analyses": ";".join(analyses), "source_file": a.get("source_file", ""),
            "created_utc": a.get("created_utc", "")}


def rebuild_index(out_dir=None) -> Path:
    """Rewrite <out_dir>/index.csv, one row per standardized file, so it cannot drift."""
    out_dir = Path(out_dir or paths.GW_PE)
    rows = [_index_row(p) for p in sorted(out_dir.glob("*.h5"))]
    path = out_dir / "index.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_INDEX_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path
