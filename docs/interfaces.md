# Tool interfaces

The contract between the tools in `uuls/`. Each tool owns its type; other tools use it only
through what is written here. Change this file first if an interface has to change.
Only the main entry points are listed; each module's docstrings list its helpers.

Conventions: angles in degrees, times as `astropy.time.Time` (UTC) or seconds since the merger
`t0` when stored as numbers, energies with units, ICRS frame. Distances in Mpc.
Cached functions take an optional `root=` / `cache_root=` (cache folder, used by tests).

## `uuls.gwmap`

```python
@dataclass(frozen=True, eq=False)   # holds arrays: no == or hashing
class GWMap:
    name: str                 # superevent, e.g. "S240615dg"
    t0: Time                  # merger time from the skymap header (cases may override it)
    geom: WcsGeom             # 2D image geometry the map lives on
    prob: np.ndarray          # (ny, nx) GW probability integrated over each bin (sum ~1 on a covering geom, error ~1e-5)
    mask: np.ndarray          # (ny, nx) bool, bins whose centre is inside the `credible` region
    credible: float           # e.g. 0.95
    r_grid: np.ndarray        # (n_r,) Mpc, starts at 1, never 0 (default 1-10000 Mpc, 1000 points)
    cdf: np.ndarray           # (ny, nx, n_r) distance CDF per bin
    cdf_valid: np.ndarray     # (ny, nx) bool, bins with a usable distance CDF

def read_healpix(path) -> HealpixSkymap      # uniq, prob, distmu, distsigma, distnorm, nside (per pixel, multi-order kept), t0, name
def on_geom(path, geom, credible=0.95, r_grid=None, root=None) -> GWMap     # cached ("gwmap"); energy axis of geom dropped
def credible_area_deg2(path, credible=0.9) -> float
```

## `uuls.gwpe`

```python
def standardize(raw_release_path, superevent, waveform_label=None, n_per_analysis=5000,
                seed=0, out_dir=None, overwrite=False) -> Path   # data/gw/pe_standardized/<name>.h5, rewrites index.csv
def load_samples(name_or_path) -> Table   # ra, dec (deg), luminosity_distance (Mpc), theta_jn, theta_v (deg, 0-90), analysis
def theta_v_given_distance(samples, d_edges) -> list[np.ndarray]
```

File attributes of a standardized file include `trigger_time_gps` and `trigger_time_source`, and, when a chosen
analysis has no `network_matched_filter_snr` column (GWTC-2.1/3 `C01:Mixed`), `network_snr_median` and
`network_snr_source` taken from the other analyses of the same release.

## `uuls.pointings`

A `Pointings` table is an `astropy.table.Table` with one row per observation run:

| column | type | meaning |
| --- | --- | --- |
| `obs_id` | int | unique within the table |
| `ra`, `dec` | float, deg | telescope pointing |
| `t_start` | Time | UTC start |
| `duration` | float, s | live time asked for |
| `wobble_angle` | float, deg | wobble only, position angle north through east |
| `pgw` | float | tilepy only, GW probability of the tile |

Metadata: `meta["t0"]` (merger time, isot), `meta["strategy"]` (`"on"`, `"wobble"`, `"grid"`, `"tilepy"`, `"real"`),
`meta["tilepy"]` (settings used, tilepy only). Saved as ECSV with `write(table, path)` / `read(path)`.

```python
def on(target, t_start, n_runs, run_duration, t0=None) -> Pointings
def wobble(target, t_start, n_runs, run_duration, offset=0.4, angles=(0, 90, 180, 270), t0=None) -> Pointings
def grid(center, t_start, n_side, spacing, run_duration, t0=None) -> Pointings
def tilepy_2d(skymap_path, t0, obs_window=None, settings=None) -> Pointings   # cached; settings override TILEPY_LST1
```

## `uuls.eventtime`

```python
def observable_time(target: SkyCoord, after, max_zenith=40, site="LST-1", need_dark=True) -> Time
    # first time with zenith < max_zenith and Sun < -18 deg; Moon ignored
def peak_direction(skymap_path, nside=128) -> SkyCoord
```

## `uuls.cases`

```python
def load_case(path_or_name) -> Case      # configs/cases/<name>.yaml
class Case:
    def t0(self) -> Time                 # real: skymap time; observable: eventtime.observable_time
    def target(self) -> SkyCoord
    def pointings(self) -> Pointings
    def analysis_center(self, pointings=None) -> SkyCoord
    def analysis_width(self, pointings=None) -> tuple[float, float]   # (along RA, along Dec), deg
    def summary(self) -> dict
```

## `uuls.simdata`

Runs simulst in its own environment (subprocess, `paths.SIMULST_PYTHON`, offline mode).

```python
@dataclass(frozen=True)
class PowerLawSource:
    ra: float; dec: float; amplitude: float  # cm-2 s-1 TeV-1 at 1 TeV
    index: float = 2.0

class SimulstRefusal(ValueError)             # names the refused run (obs_id, time, pointing, reason)

def simulate(pointings, nsb=0.14, source=None, seed=0, far_source="skip",
             production="2025_2p5") -> Path
    # cached ("simulst_dl3", key includes the production); returns a DL3 folder:
    # dl3_LST-1.Run<obs_id>.fits, bkg_Run<obs_id>.fits (simulst's background model, registered as
    # the run's bkg), hdu/obs index, runs.json (per run: production, source simulated or not, keywords).
    # Run k is seeded SeedSequence([seed, obs_id]).
    # production: simulst IRF production. "2025_2p5" (default): IRFs, exposure and sources out to
    # 2.5 deg offset; "2025": out to 1.1 deg (padded to 1.2). Same cuts and background library.
    # far_source: runs farther from the source than the production's max_source_offset (2.5 / 1.1 deg)
    # are simulated without it ("skip", flagged in runs.json) or refused ("refuse").
def run_summary(dl3_dir) -> Table
```

## `uuls.reduce`

```python
@dataclass
class ReducedData:
    geom: WcsGeom             # 3D, reco energy axis
    stacked: MapDataset       # counts, background, exposure, psf, edisp, mask_safe
    runs: list[MapDataset]    # one per run ("run<obs_id>"), zero outside its own safe mask
    run_gti_s: list[np.ndarray]   # per run, (n_intervals, 2) seconds since t0
    t0: Time
    settings: dict            # DEFAULT_SETTINGS as used, plus binsz, energy edges, obs_ids_used
    obs_ids: list[int]        # property

def analysis_geom(center, width=3.575, binsz=0.08, e_reco=(0.6, 20) TeV, bins_per_decade=4.5,
                  proj="AIR") -> WcsGeom          # width: float or (along RA, along Dec)
def from_dl3(dl3_dir, geom, t0, settings=None) -> ReducedData   # cached ("reduce")
```

`settings` keys (unknown keys are refused): `background` (`"ring"` | `"fov"` | `"model"`), `ring_r_in_deg`,
`ring_width_deg`, `ring_within_safe_mask`, `exclusion` ([[ra, dec, r], ...]), `offset_max_deg`, `safe_mask`,
`e_true_tev`, `true_bins_per_decade`, `obs_ids`, `bkg_pattern` (background file per run for real DL3 without
a bkg HDU), `correlation_radius_deg`.

## `uuls.lambda_stat`

```python
def ts_map(counts, background, correlation_radius_bins, mask=None) -> np.ndarray   # signed Cash TS, (ny, nx)
def global_lambda(ts, prob, mask) -> float          # max over mask of TS + 2 ln prob
class TSPrecomputed:                                 # fast path: background correlated once
    @classmethod
    def build(cls, background, prob, mask, correlation_radius_bins, safe_mask=None)
    def evaluate(self, counts) -> dict              # lambda, tsmax and their bins
def observed_region(dataset) -> np.ndarray          # (ny, nx) bool: exposure summed over true energy > 0 & any safe-mask bin
def roi(gw_mask, dataset, mode="observed") -> np.ndarray   # GW mask & observed_region ("observed") or GW mask ("gw_mask")
```

Region of interest (owner decision, 2026-11-24): Λ is maximised over ROI = GW 95% mask ∩ observed region, for data
and Monte Carlo alike, single- and multi-field. `roi="gw_mask"` reproduces the reference code.

## `uuls.realisations`

```python
@dataclass
class Injection:              # what an injector returns
    coord: SkyCoord
    spectrum: SpectralModel | RunSpectra   # RunSpectra: one spectrum per run (time-resolved models)
    params: dict              # recorded per realisation (distance, theta_v, ...)

# An injector is an object with __call__(rng, strength) -> Injection and describe() -> dict (cache key).
class PWLInjector2D(gwmap, index=2.0, jitter=True, reduced=None, roi="observed")
                                                 # strength = phi0 [cm-2 s-1 TeV-1], bin ~ prob in the ROI of `reduced`
                                                 # (without reduced: the GW mask, recorded as roi "gw_mask"); .support
class PWLInjector3D(gwmap, band_TeV, index=2.0, restrict_to_mask=False, apply_k_correction=False, jitter=True)
                                                 # strength = L0 [erg/s], sky + distance from the full map
# jitter: position uniform inside the drawn bin (two uniforms drawn after the bin and distance);
# params gain ra, dec. jitter=False is the bin centre (the reference behaviour).

def run(reduced, gwmap, injector, strengths, seed, n_jobs=1, first_index=0,
        correlation_radius_deg=0.1, use_cache=True, roi="observed") -> dict
    # cached ("realisations"); dict of (n,) arrays: strength, lambda, tsmax, bins, n_signal, fake_seed, + injector params.
    # Realisation i: injection rng default_rng([seed, i]), Poisson RandomState(seed + i). Independent of n_jobs.
def batch_function(reduced, gwmap, injector, seed, **run_kwargs) -> callable   # what ulfit.adaptive_fit calls
def precompute(reduced, gwmap, correlation_radius_deg=0.1, roi="observed") -> TSPrecomputed   # mask = lambda_stat.roi
def observed_lambda(reduced, gwmap, correlation_radius_deg=0.1, roi="observed") -> dict      # data, same ROI as `run`
# simulate/run warn if a 2D injector's support leaves the ROI.
def sample_sky_2d(gwmap, n, rng, support=None) / sample_sky_and_distance(gwmap, n, rng, sky_mask=None)
def jitter_in_bin(geom, iy, ix, rng) -> SkyCoord

# Several fields (uuls.fields). A field is any object with .reduced and .gwmap on its own geometry.
def run_fields(fields, injector, strengths, seed, n_jobs=1, first_index=0, correlation_radius_deg=0.1,
               use_cache=True, roi="observed") -> dict
    # cached ("realisations_fields"). One injection per realisation (rng default_rng([seed, i])); field k
    # faked with Poisson seed field_poisson_seed(seed + i, k) = seed + i (k = 0) or [seed + i, k], with the
    # source only if field_contains(field, coord); Lambda = max over fields of the max over each field's ROI. Keys as `run` plus
    # lambda_field, tsmax_field, lambda_fields (n, n_fields), n_fields_containing. A one-field list gives `run`.
def simulate_fields(...) / batch_function_fields(fields, injector, seed, **run_kwargs)
def evaluate_field(pre, counts) -> dict          # pre.evaluate, or -inf for a field with an empty ROI
```

`RunSpectra(energy_TeV, values, obs_ids=None)`: with several fields, `obs_ids` (the run of each row) is required.

## `uuls.fields`

One small geometry per group of runs (tiling). Global statistic Λ = max over fields of max over the field's
ROI bins (GW 95% mask ∩ the field's observed region) of (TS_b + 2 ln p_b). Overlapping fields are kept as they are (no merging): fields partition the
runs, so the two looks at an overlap are independent data; Λ is a max, so p_b is never summed; the Monte Carlo
applies the same rule. See the module docstring.

```python
@dataclass(eq=False)
class Field:
    name: str                 # "field<k>", k in order of the first run
    pointings: Table          # Pointings rows of the field
    center: SkyCoord          # mean pointing direction
    geom: WcsGeom             # 3D, AIR, width 2 (max offset + offset_max_deg), common binsz
    gwmap: GWMap | None       # gwmap.on_geom(skymap, geom), cached
    reduced: ReducedData | None   # reduce.from_dl3(dl3_dir, geom, t0, settings + obs_ids), cached
    obs_ids: list[int]        # property
    def contains(self, coord) -> bool

def split_fields(pointings, link_deg=1.0) -> list[Table]    # single linkage: pointings < link_deg apart share a field
def field_geom(pointings, offset_max_deg=2.5, binsz=0.08, e_reco=(0.6, 20) TeV, bins_per_decade=4.5, pad_deg=0) -> WcsGeom
def build_fields(case_or_pointings, dl3_dir=None, settings=None, *, skymap=None, t0=None, link_deg=None,
                 binsz=0.08, credible=0.95, ...) -> list[Field]
    # from a Case: pointings, skymap, t0, analysis.reduce (reduce settings) and analysis.fields.link_deg
def observed_lambda(fields, correlation_radius_deg=0.1, roi="observed") -> dict   # lambda, field, lambda_iy/ix, tsmax, lambda_fields
def coverage(fields, geom_or_coords) -> np.ndarray   # number of field geometries containing each bin centre / position
def sample_healpix(sky, n, rng, support=None) -> dict    # ra, dec, pix: pixel ~ prob, uniform inside the pixel
class PWLSkyInjector2D(sky, credible=0.95, index=2.0)    # position on the HEALPix map, in its credible region
class PWLSkyInjector3D(sky, band_TeV, index=2.0, restrict_to_credible=None, apply_k_correction=False)
                                                        # + d from the pixel's distance ansatz (>= 1 Mpc)
```

Case YAML (read by `build_fields`, ignored by `uuls.cases`): `analysis.fields.link_deg`, and `analysis.reduce`
(overrides of `reduce.DEFAULT_SETTINGS`; the simulated cases set `background: fov`).

## `uuls.ulfit`

```python
@dataclass
class ULResult:               # ul, interval, probit params and covariance, gof, n_sims, converged, ...
    def report(self, label="x", unit="") -> str
    def to_dict(self) -> dict

def fit(strength, lambda_sim, lambda_obs, lambda_bkg, cl=0.95, bracket=None, target=None) -> ULResult
def adaptive_fit(simulate_batch, x_lo, x_hi, lambda_obs, lambda_bkg, cl=0.95, ...) -> ULResult   # scout + refine rounds
def target_lambda(lambda_obs, lambda_bkg) -> (lambda_star, p, S)
def closure_pull(result, lambda_closure) -> dict
```

## `uuls.spectra`

```python
pwl_photon_flux, pwl_energy_flux, pwl_e2dnde (amplitude, energy_edges, index=2.0, e_ref=1 TeV)
amplitude_from_photon_flux, amplitude_from_energy_flux
redshift_from_distance(d), k_correction(index, z)
luminosity_from_energy_flux, energy_flux_from_luminosity, amplitude_from_luminosity, luminosity_from_amplitude
flux_point(amplitude, energy_edges) -> FluxPoint
check_conversions() -> list      # raises if any check fails
```
