"""Study 1 batch entry point: the 2D flux UL of one case (uuls.flux_ul.flux_ul).

    python scripts/run_flux_ul.py sim_wobble
    python scripts/run_flux_ul.py sim_tiling --max-sims 6000 --n-bkg 5000 --n-closure 2000 --n-jobs 8

Every step is cached (simulst DL3, reduction, GW maps, each batch of realisations), so a rerun
with the same options only redoes the fits, and a larger budget reuses the batches already done
up to the round where the designs differ. Writes outputs/results/1_flux_ul/<case>.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uuls import flux_ul  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("case", help="case name in configs/cases/ or a case YAML")
    d = flux_ul.DEFAULTS
    p.add_argument("--max-sims", type=int, default=d["max_sims"], help="realisations of the probit fit")
    p.add_argument("--n-scout", type=int, default=d["n_scout"])
    p.add_argument("--n-refine", type=int, default=d["n_refine"])
    p.add_argument("--n-bkg", type=int, default=d["n_bkg"], help="background-only realisations")
    p.add_argument("--n-closure", type=int, default=d["n_closure"])
    p.add_argument("--precision", type=float, default=d["precision"], help="target 1 sigma of log10 UL, dex")
    p.add_argument("--x-lo", type=float, default=d["x_lo"])
    p.add_argument("--x-hi", type=float, default=d["x_hi"])
    p.add_argument("--seed", type=int, default=d["seed"])
    p.add_argument("--roi", default=d["roi"], choices=("observed", "gw_mask"))
    p.add_argument("--production", default=d["production"])
    p.add_argument("--n-jobs", type=int, default=1)
    a = p.parse_args(argv)
    opts = {k: getattr(a, k) for k in ("max_sims", "n_scout", "n_refine", "n_bkg", "n_closure",
                                       "precision", "x_lo", "x_hi", "seed", "roi", "production")}
    t = time.time()
    res = flux_ul.flux_ul(a.case, n_jobs=a.n_jobs, **opts)
    amp, fx = res["amplitude"], res["energy_flux"]
    print(json.dumps({k: res[k] for k in ("case", "roi_gw_prob", "lambda_obs", "target", "closure",
                                          "timing_s")}, indent=1))
    print(f"phi0_UL = {amp['ul']:.3e} [{amp['lo_1sigma']:.3e}, {amp['hi_1sigma']:.3e}] {amp['unit']}")
    print(f"energy flux UL 0.6-20 TeV = {fx['ul']:.3e} [{fx['lo_1sigma']:.3e}, {fx['hi_1sigma']:.3e}] {fx['unit']}")
    print(f"written {flux_ul.result_path(res['case'])} in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
