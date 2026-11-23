# uncertain-uls

Gamma-ray upper limits for gravitational-wave events observed by LST-1, when the source position,
distance and orientation are known only through the GW posterior.

The project is a set of small tools (`uuls/`) and four studies that use them:

| Study | Question |
| --- | --- |
| `studies/0_tools` | What each tool does, with checks. Run once, in order. |
| `studies/1_flux_ul` | Frequentist flux upper limit over the GW region, on real or simulated data. |
| `studies/2_model_ul` | Constraints on GRB model parameters, the other parameters marginalised over priors. |
| `studies/3_bayesian` | The same constraints from the likelihood of no detection, compared with study 2. |
| `studies/4_tiling` | Standard tiling against tiling that uses the expected constraining power. |

Shared mock cases live in `configs/cases/`. Interfaces between tools: [`docs/interfaces.md`](docs/interfaces.md).

## Setup

Two conda environments. simulst pins an older gammapy, so it is only called as a subprocess.

```bash
# main environment
mamba env create -f env/uncertain-uls.yml
conda activate uncertain-uls
pip install --no-deps -e .
python -m ipykernel install --user --name uncertain-uls --display-name "Python (uncertain-uls)"

# simulst (simulated LST-1 data), offline mode
git clone <simulst repo> ../../lstreco/simulst        # or set UULS_SIMULST_REPO
cd ../../lstreco/simulst
mamba env create -n simulst -f environment.yml
# IRFs: copy irf_*.fits.gz of the 2025 all-sky production (NSB 0.07-0.50) keeping the tree, into data/irf/
#   source at PIC: /pnfs/pic.es/data/cta/LST/Data/MC/IRF/AllSky/20250212_v0.10.17_allsky_interp_dl2_irfs_nsb_<nsb>/TestingDataset/
SIMULST_IRF_ROOT=$PWD/data/irf PYTHONPATH=. python scripts/rebuild_library_from_wiki.py data/bkg_library_wiki
SIMULST_IRF_ROOT=$PWD/data/irf SIMULST_LIBRARY_ROOT=$PWD/data/bkg_library_wiki PYTHONPATH=. python scripts/check_offline.py
```

`uuls.paths` finds both by default when the two repos sit as `gw/uncertain-uls` and
`lstreco/simulst` under the same parent. Otherwise set `UULS_SIMULST_REPO` and
`UULS_SIMULST_PYTHON`. Real DL3 goes in `data/dl3_real/` or `$UULS_REAL_DL3`.

Check: `pytest -q -m "not slow"`.

## Caching

Long steps are stored under `cache/<step>/<key>/`, where the key is a hash of the parameters
(and of the input files' size and date). Each entry has a readable `params.json`. Rerunning a
notebook with the same parameters loads the results. Delete a folder to force a recompute.
