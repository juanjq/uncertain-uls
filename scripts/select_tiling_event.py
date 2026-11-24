"""Rank confident GWTC events as candidates for the `sim_tiling` case (LST-1 tiling).

Steps
1. Confident event names from the GWOSC event API (GWTC-2.1-confident, GWTC-3-confident,
   GWTC-4.1 (O4a, supersedes 4.0), GWTC-5.0).
2. PE skymap tarballs from the Zenodo PE releases (tens to hundreds of MB, kept under
   cache/gwtc_skymaps/, not under data/). One skymap per event: the "Mixed" analysis when it
   exists, otherwise the first available in ANALYSIS_PREFERENCE.
3. Per skymap: 90% credible area, probability and area of the 90% region at dec > DEC_MIN.
   Cached per release with uuls.cache.
4. Number of LST fields n = A90 / field area, with the field a disc of radius FOV_DEG
   (tilepy's LST default, 2.5 deg, 19.6 deg2). Candidates have N_MIN <= n <= N_MAX.
   Score = area_fit * p_vis, area_fit = exp(-0.5 (ln(n / n_mid) / ln(N_MAX / n_mid))^2)
   with n_mid = sqrt(N_MIN N_MAX), and p_vis the largest fraction of the 90% region's
   probability seen at zenith < VIS_ZENITH from LST-1 at one instant (best sidereal time).
   P(dec > DEC_MIN | 90% region) is listed as well.
5. The top candidates' skymaps are copied to data/gw/skymaps/<superevent>.<ext> and the
   ranked table is printed as Markdown (docs/tiling_event_candidates.md is written by hand
   from it). PE releases are not downloaded; their Zenodo record is listed.

Usage:  python scripts/select_tiling_event.py [--top 3] [--copy 2] [--out table.md]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from uuls import cache, paths

FOV_DEG = 2.5
FIELD_DEG2 = math.pi * FOV_DEG**2
N_MIN, N_MAX = 10, 50
DEC_MIN = -25.0
VIS_ZENITH = 60.0
LST1_LAT = 28.7616

GWOSC = "https://gwosc.org/api/v2/catalogs/{}/events?format=json&pagesize=100&page={}"
RELEASES = {
    # name: (GWOSC catalog, Zenodo PE records, skymap tarball file name, tarball record)
    "GWTC-2.1": ("GWTC-2.1-confident", [6513631], "IGWN-GWTC2p1-v2-PESkyMaps.tar.gz", 6513631),
    "GWTC-3": ("GWTC-3-confident", [8177023], "IGWN-GWTC3p0-v2-PESkyLocalizations.tar.gz", 8177023),
    "GWTC-4.1": ("GWTC-4.1", [20275769], "IGWN-GWTC4p1-18965dda8_5-Archived_Skymaps.tar.gz", 20275769),
    "GWTC-5.0": ("GWTC-5.0", [20348005, 20348006], "IGWN-GWTC5p0-29ebe06b7_25-Archived_Skymaps.tar.gz",
                 20348005),
}
ANALYSIS_PREFERENCE = ["Mixed", "Mixed+XO4a", "IMRPhenomXPHM_SpinTaylor", "IMRPhenomXPHM",
                       "IMRPhenomXPNR", "SEOBNRv5PHM", "SEOBNRv4PHM", "NRSur7dq4"]
ARCHIVE = paths.CACHE / "gwtc_skymaps"
GW_RE = re.compile(r"GW\d{6}_\d{6}")


def _get_json(url, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def confident_events(catalog: str) -> dict[str, dict]:
    """GWOSC catalog name -> {GW name: {gps, detail_url}}."""
    out, page = {}, 1
    while True:
        d = _get_json(GWOSC.format(catalog, page))
        for e in d["results"]:
            out[e["name"]] = {"gps": e["gps"], "detail_url": e["detail_url"]}
        if not d.get("next"):
            return out
        page += 1


def superevent_id(detail_url: str) -> str | None:
    return _get_json(detail_url).get("grace_id")


def zenodo_pe_files(record: int) -> dict[str, str]:
    """GW name -> PE file name in a Zenodo record (only names, nothing downloaded)."""
    d = _get_json(f"https://zenodo.org/api/records/{record}")
    out = {}
    for f in d["files"]:
        k = f["key"]
        m = GW_RE.search(k)
        if m and "PEDataRelease" in k and k.endswith((".h5", ".hdf5")):
            preferred = "combined" in k or ("_cosmo" in k and "nocosmo" not in k)
            if preferred or m.group() not in out:
                out[m.group()] = k
    return out


def tarball(release: str) -> Path:
    _, _, fname, rec = RELEASES[release]
    p = ARCHIVE / release / fname
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://zenodo.org/api/records/{rec}/files/{fname}/content"
        print(f"downloading {url}")
        tmp = p.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(p)
    return p


def _gw_and_analysis(member: str):
    base = Path(member).name
    m = GW_RE.search(base)
    if not m or not base.endswith((".fits", ".fits.gz")):
        return None
    gw = m.group()
    if "_Skymap_" in base:                      # GWTC-4.1 / 5.0
        analysis = base.split(gw + "-", 1)[1].split("_Skymap_")[0]
    else:                                       # GWTC-2.1 / 3: ..._cosmo_reweight_C01:<label>.fits
        analysis = base.split("C01:", 1)[1].removesuffix(".fits") if "C01:" in base else "?"
    return gw, analysis


def choose_members(tar_path: Path) -> dict[str, tuple[str, str]]:
    """GW name -> (tar member, analysis label), one per event."""
    by_gw: dict[str, dict[str, str]] = {}
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            ga = _gw_and_analysis(m.name)
            if ga:
                by_gw.setdefault(ga[0], {})[ga[1]] = m.name
    out = {}
    for gw, opts in by_gw.items():
        label = next((a for a in ANALYSIS_PREFERENCE if a in opts), sorted(opts)[0])
        out[gw] = (opts[label], label)
    return out


def skymap_stats(path: Path) -> dict:
    """90% area (deg2), P(dec > DEC_MIN) inside the 90% region, its area there, peak (ra, dec)."""
    import astropy_healpix as ah
    from ligo.skymap.io import read_sky_map

    sky = read_sky_map(str(path), moc=True)
    level, ipix = ah.uniq_to_level_ipix(sky["UNIQ"])
    nside = ah.level_to_nside(level)
    area = ah.nside_to_pixel_area(nside).to_value("sr")
    prob = np.asarray(sky["PROBDENSITY"]) * area
    order = np.argsort(-np.asarray(sky["PROBDENSITY"]))
    cum = np.cumsum(prob[order])
    inside = np.zeros(len(prob), bool)
    inside[order[: np.searchsorted(cum, 0.9) + 1]] = True
    lon, lat = ah.healpix_to_lonlat(ipix, nside, order="nested")
    dec = lat.deg
    sr2deg = (180 / np.pi) ** 2
    north = inside & (dec > DEC_MIN)
    ipk = int(np.argmax(sky["PROBDENSITY"]))
    return {
        "area90": float(area[inside].sum() * sr2deg),
        "p90_north": float(prob[north].sum() / prob[inside].sum()),
        "area90_north": float(area[north].sum() * sr2deg),
        "peak_ra": float(lon.deg[ipk]), "peak_dec": float(dec[ipk]),
        "date_obs": str(sky.meta.get("gps_time", "")),
    }


def best_visible_fraction(path: Path) -> float:
    """max over sidereal time of P(zenith < VIS_ZENITH | 90% region), from LST-1."""
    import astropy_healpix as ah
    from ligo.skymap.io import read_sky_map

    sky = read_sky_map(str(path), moc=True)
    level, ipix = ah.uniq_to_level_ipix(sky["UNIQ"])
    nside = ah.level_to_nside(level)
    rho = np.asarray(sky["PROBDENSITY"])
    prob = rho * ah.nside_to_pixel_area(nside).to_value("sr")
    order = np.argsort(-rho)
    inside = np.zeros(len(prob), bool)
    inside[order[: np.searchsorted(np.cumsum(prob[order]), 0.9) + 1]] = True
    lon, lat = ah.healpix_to_lonlat(ipix[inside], nside[inside], order="nested")
    ra, dec, w = lon.rad, lat.rad, prob[inside] / prob[inside].sum()
    phi = np.radians(LST1_LAT)
    lst = np.radians(np.arange(0, 360, 2.0))[:, None]
    cosz = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(lst - ra)
    return float(((cosz > np.cos(np.radians(VIS_ZENITH))) * w).sum(axis=1).max())


def release_stats(release: str) -> list[dict]:
    """Stats of every event in a release's skymap tarball (cached)."""
    tpath = tarball(release)

    def compute():
        rows = []
        members = choose_members(tpath)
        work = ARCHIVE / release / "extracted"
        work.mkdir(exist_ok=True)
        with tarfile.open(tpath) as tf:
            for gw, (member, label) in sorted(members.items()):
                dest = work / Path(member).name
                if not dest.exists() or dest.stat().st_size != tf.getmember(member).size:
                    part = dest.with_name(dest.name + ".part")
                    with tf.extractfile(member) as src, open(part, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    part.replace(dest)
                st = skymap_stats(dest)
                rows.append({"gw": gw, "release": release, "analysis": label,
                             "member": member, "file": str(dest), **st})
                print(f"  {release} {gw} {label:28s} A90 = {st['area90']:8.1f} deg2")
        return rows

    return cache.cached("gwtc_skymap_stats", {"tarball": tpath, "dec_min": DEC_MIN, "version": 1}, compute)


def rank(rows: list[dict]) -> list[dict]:
    """Add n_fields, area_fit, p_vis (for events in the field range) and score; sort."""
    for r in rows:
        n = r["area90"] / FIELD_DEG2
        if N_MIN <= n <= N_MAX and "p_vis" not in r:
            r["p_vis"] = best_visible_fraction(Path(r["file"]))
    n_mid = math.sqrt(N_MIN * N_MAX)
    width = math.log(N_MAX / n_mid)
    for r in rows:
        n = r["area90"] / FIELD_DEG2
        r["n_fields"] = n
        r["area_fit"] = math.exp(-0.5 * (math.log(n / n_mid) / width) ** 2)
        r["score"] = r["area_fit"] * r.get("p_vis", r["p90_north"])
        r["in_range"] = N_MIN <= n <= N_MAX
    return sorted(rows, key=lambda r: (not r["in_range"], -r["score"]))


def markdown(rows: list[dict], n: int) -> str:
    head = ("| rank | event | superevent | release | skymap analysis | A90 (deg2) | LST fields | "
            "P(dec > -25 deg) in 90% | best P(z < 60 deg) in 90% | peak (ra, dec) | distance (Mpc) | score | PE Zenodo record |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
    lines = []
    for i, r in enumerate(rows[:n], 1):
        lines.append(f"| {i} | {r['gw']} | {r.get('superevent') or '-'} | {r['release']} | {r['analysis']} | "
                     f"{r['area90']:.0f} | {r['n_fields']:.1f} | {r['p90_north']:.2f} | {r.get('p_vis', float('nan')):.2f} | "
                     f"({r['peak_ra']:.1f}, {r['peak_dec']:+.1f}) | {r['dist'][0]:.0f} +- {r['dist'][1]:.0f} | {r['score']:.2f} | "
                     f"{r.get('pe_record', '-')} `{r.get('pe_file', '-')}` |")
    return head + "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--top", type=int, default=10, help="rows in the printed table")
    ap.add_argument("--copy", type=int, default=3, help="skymaps copied to data/gw/skymaps")
    ap.add_argument("--events", nargs="*", default=None,
                    help="GW names whose skymaps are copied (default: the top --copy rows)")
    ap.add_argument("--out", type=Path, default=None, help="write the Markdown table here")
    args = ap.parse_args()

    rows = []
    for release, (catalog, records, _, _) in RELEASES.items():
        conf = confident_events(catalog)
        stats = release_stats(release)
        gps = {k: float(v["gps"]) for k, v in conf.items()}
        kept = []
        for r in stats:                 # GWOSC drops the time suffix for some names: match on GPS
            name = r["gw"] if r["gw"] in conf else next(
                (k for k, g in gps.items() if r["date_obs"] and abs(g - float(r["date_obs"])) < 2), None)
            if name is not None:
                r["gwosc_name"] = name
                kept.append(r)
        print(f"{release}: {len(conf)} confident events in GWOSC, {len(stats)} skymaps, {len(kept)} kept")
        pe = {}
        for rec in records:
            pe.update({k: (rec, v) for k, v in zenodo_pe_files(rec).items()})
        for r in kept:
            r["detail_url"] = conf[r["gwosc_name"]]["detail_url"]
            r["gps"] = conf[r["gwosc_name"]]["gps"]
            if r["gw"] in pe:
                r["pe_record"], r["pe_file"] = pe[r["gw"]]
        rows.extend(kept)

    ranked = rank(rows)
    n_in = sum(r["in_range"] for r in ranked)
    print(f"\n{len(ranked)} events; {n_in} with {N_MIN} <= A90 / {FIELD_DEG2:.1f} deg2 <= {N_MAX}")
    chosen = ranked[: args.copy] if args.events is None else [r for r in ranked if r["gw"] in args.events]
    for r in ranked[: args.top] + chosen:
        if not r.get("superevent"):
            r["superevent"] = superevent_id(r["detail_url"])
        h = __import__("astropy.io.fits", fromlist=["getheader"]).getheader(r["file"], 1)
        r["dist"] = (h.get("DISTMEAN", float("nan")), h.get("DISTSTD", float("nan")))
    table = markdown(ranked, args.top)
    print(table)
    if args.out:
        args.out.write_text(table)

    for r in chosen:
        src = Path(r["file"])
        ext = ".fits.gz" if src.name.endswith(".gz") else ".fits"
        sid = r["superevent"] if (r["superevent"] or "").startswith("S") else r["gw"]   # O2: no S-id
        dest = paths.GW_SKYMAPS / f"{sid}{ext}"
        if not dest.exists():
            shutil.copy2(src, dest)
        print(f"copied {r['gw']} ({r['analysis']}) -> {dest}  [{dest.stat().st_size / 1e6:.1f} MB]")


if __name__ == "__main__":
    main()
