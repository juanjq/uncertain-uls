"""Every directory the project reads or writes, in one place.

Each path can be overridden with the environment variable named next to it.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _env_path(var, default):
    return Path(os.environ.get(var, default)).expanduser().resolve()


DATA = _env_path("UULS_DATA", REPO / "data")
CACHE = _env_path("UULS_CACHE", REPO / "cache")
OUTPUTS = _env_path("UULS_OUTPUTS", REPO / "outputs")
FIGURES = OUTPUTS / "figures"
RESULTS = OUTPUTS / "results"
CONFIGS = REPO / "configs"

GW_SKYMAPS = DATA / "gw" / "skymaps"
GW_PE_RAW = DATA / "gw" / "pe_raw"
GW_PE = DATA / "gw" / "pe_standardized"
EBL = DATA / "ebl"
REAL_DL3 = _env_path("UULS_REAL_DL3", DATA / "dl3_real")

# simulst: separate repo and conda env, called only as a subprocess (see uuls.simdata).
SIMULST_REPO = _env_path("UULS_SIMULST_REPO", REPO.parent / "simulst")
SIMULST_PYTHON = _env_path(
    "UULS_SIMULST_PYTHON",
    Path(os.environ.get("CONDA_ENVS", REPO.parents[1] / "softs" / "conda" / "envs")) / "simulst" / "bin" / "python",
)
SIMULST_IRF_ROOT = _env_path("SIMULST_IRF_ROOT", SIMULST_REPO / "data" / "irf")
SIMULST_LIBRARY_ROOT = _env_path("SIMULST_LIBRARY_ROOT", SIMULST_REPO / "data" / "bkg_library_wiki")
