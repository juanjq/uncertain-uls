# Tool interfaces

The contract between the tools in `uuls/`. Each tool owns its type; other tools use it only
through what is written here. Change this file first if an interface has to change.

Conventions: angles in degrees, times as `astropy.time.Time` (UTC) or seconds since the merger
`t0` when stored as numbers, energies with units, ICRS frame. Distances in Mpc.

## `uuls.gwmap`

```python
@dataclass(frozen=True)
class GWMap:
    name: str                 # superevent, e.g. "S240615dg"
    t0: Time                  # merger time read from the skymap header (may be overridden later)
    geom: WcsGeom             # 2D image geometry the map lives on
    prob: np.ndarray          # (ny, nx) GW probability integrated over each bin, sums to <= 1
    mask: np.ndarray          # (ny, nx) bool, bins inside the `credible` region
    credible: float           # e.g. 0.95
    r_grid: np.ndarray        # (n_r,) Mpc, starts at 1, never 0
    cdf: np.ndarray           # (ny, nx, n_r) distance CDF per bin
    cdf_valid: np.ndarray     # (ny, nx) bool, bins with a usable distance CDF

def read_healpix(path) -> HealpixSkymap      # prob, distmu, distsigma, distnorm, nside, t0, name
def on_geom(path, geom, credible=0.95, r_grid=None) -> GWMap     # cached
def credible_area_deg2(path, credible=0.9) -> float
```

## `uuls.gwpe`

```python
def standardize(raw_release_path, superevent, waveform_label) -> Path   # data/gw/pe_standardized/<name>.h5
def load_samples(name) -> astropy.table.Table   # columns ra, dec, luminosity_distance (Mpc), theta_jn, theta_v (deg, 0-90)
```

## `uuls.pointings`

A `Pointings` table is an `astropy.table.Table` with one row per observation run:

| column | type | meaning |
| --- | --- | --- |
| `obs_id` | int | unique within the table |
| `ra`, `dec` | float, deg | telescope pointing |
| `t_start` | Time | UTC start |
| `duration` | float, s | live time asked for |

Metadata: `meta["t0"]` (merger time, isot), `meta["strategy"]` (`"on"`, `"wobble"`, `"grid"`, `"tilepy"`).
Saved as ECSV with `write(table, path)` / `read(path)`.

```python
def on(target, t_start, n_runs, run_duration) -> Pointings
def wobble(target, t_start, n_runs, run_duration, offset=0.4, angles=(0, 90, 180, 270)) -> Pointings
def grid(center, t_start, n_side, spacing, run_duration) -> Pointings
def tilepy_2d(skymap_path, t0, obs_window, settings) -> Pointings       # cached
```

## `uuls.eventtime`

```python
def observable_time(target: SkyCoord, after: Time, max_zenith=40, site="LST-1",
                    need_dark=True) -> Time
```

## `uuls.simdata`

Runs simulst in its own environment (subprocess, `paths.SIMULST_PYTHON`).

```python
@dataclass(frozen=True)
class PowerLawSource:
    ra: float; dec: float; amplitude: float  # cm-2 s-1 TeV-1 at 1 TeV
    index: float = 2.0

def simulate(pointings, nsb=0.14, source=None, seed=0) -> Path   # DL3 folder with hdu/obs index; cached
```

## `uuls.reduce`

```python
@dataclass
class ReducedData:
    geom: WcsGeom             # 3D, reco energy axis
    stacked: MapDataset       # counts, background (ring or model), exposure, psf, edisp
    runs: list[MapDataset]    # one per observation, same geom, for time-resolved injection
    run_gti_s: list[np.ndarray]   # per run, (n_intervals, 2) seconds since t0
    t0: Time
    settings: dict            # analysis settings used (binsz, energy edges, offset_max, ...)

def analysis_geom(center, width, binsz=0.08, e_reco=(0.6, 20) TeV, bins_per_decade=4.5) -> WcsGeom
def from_dl3(dl3_dir, geom, t0, settings=None) -> ReducedData   # cached
```

## `uuls.lambda_stat`, `uuls.realisations`, `uuls.ulfit`, `uuls.spectra`

```python
# lambda_stat
def ts_map(counts, background, correlation_radius_bins) -> np.ndarray   # signed Cash TS, (ny, nx)
def global_lambda(ts, prob, mask) -> float     # max over mask of TS + 2 ln prob

# realisations
def run(reduced, gwmap, injector, strengths, seed, n_jobs=1) -> dict   # {"lambda": (n,), "strength": (n,), plus injected parameters}
# injector: callable(rng, strength) -> (SkyCoord, spectral model or per-run spectra)

# ulfit
def fit(strength, lambda_sim, lambda_obs, lambda_bkg, cl=0.95) -> ULResult   # UL, interval, probit params, diagnostics

# spectra
def pwl_photon_flux / pwl_energy_flux / amplitude_from_luminosity / luminosity_from_amplitude ...
```
