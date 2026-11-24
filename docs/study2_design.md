# Study 2 design: constraining afterglow parameters with an LST-1 non-detection

Status: design, not implemented (2026-11-23). Old reference code: `../gw-gamma-global-uls/gwuls/`
(`ph` = `grb_phenomenological.py`, `gm` = `grb_model.py`, `sim` = `simulate.py`). Line numbers refer to that repo.

## 0. Decisions in one place

1. Study 2 computes, per grid point ψ_k, P_det(ψ_k) = P(Λ > Λ* | ψ_k), with (Ω, d, θ_v) drawn jointly from one PE
   sample, nuisances η from their priors, Λ* from `ulfit.target_lambda`. A point is excluded where P_det ≥ CL, with
   Monte Carlo uncertainty (Clopper-Pearson on the grid, or the probit profile when one axis is solved for).
2. Scans: baseline (E_γ,iso, θ_c); second scan (L_TeV,11h, θ_c) with L_TeV,11h as a continuous `solve_for` axis.
   The TeV/X-ray ratio R is not scanned (§1.3).
3. Normalisation: the jet's own on-axis anchor (old `"anchor"` mode, `sim:1373-1377, 1611-1614`), not `"gti_mean"`,
   whose amplitude cancels and so constrains no model parameter.
4. Cost: two exact factorisations (amplitude, time scale, §3.1); everything else in a precomputed rest-frame
   "atlas" over (θ_v, θ_c, Γ_c, β₂), case-independent, shared by studies 2, 3, 4. Production maps use the fast
   response (plan §5); the full Monte Carlo is the reference (spot checks, `solve_for` contours).
5. Strength: one injector per grid point. `strength` = value of the `solve_for` parameter (L_TeV,11h, or L0 for
   the power law) if there is one; otherwise a dimensionless amplitude multiplier (1 = model, 0 = background).

## 1. Parameters of the recipe

Event parameters (Ω, d_L, θ_v) are drawn jointly from the PE (port `sim:1268-1307`); z from d_L with Planck18.

| config name | symbol | role | population distribution (source; old code) | unit | acts on L(E′,t′;θ_v) | baseline |
|---|---|---|---|---|---|---|
| `E_peak` | E_peak | gives E_γ,iso through Amati | broken PL, break 1.4 MeV, −0.8/−2.6, [0.1, 1e5] keV (Ghirlanda+16 Eq. 13, Abe+26 Eq. 3; `ph:110-114, 185-215`) | keV | only behind `population` of E_γ,iso | – |
| `E_gamma_iso` | E_γ,iso | amplitude (Berger) and time scale (E_k,iso) | Amati log(E/1e51) = 0.036 + 1.1 log(E_pk/670 keV), no scatter (`ph:121-123, 218-222`); median 2.0e50, 16–84% 3.5e48–2.0e51; median moves with `EPEAK_MIN_KEV` (`ph:566-570`) | erg | amplitude ∝ E^0.83 + time stretch | scan |
| `eta_gamma` | η_γ | E_k,iso = E_γ,iso(1−η)/η | fixed 0.2 (Nava §5.1, Abe §3.1; `ph:145, 549-559`) | – | time stretch | fixed |
| `n_ism` | n | t_dec ∝ (E_k,iso/n)^{1/3} | fixed 0.1 in both sources (Abe §3.2; `ph:146, 313-322`); a prior is a new assumption | cm⁻³ | time stretch | fixed (or prior) |
| `theta_c` | θ_core | Gaussian width, ε ∝ exp(−θ²/θ_c²) (Abe Eqs. 1-2; `ph:268-294`) | lognormal, median 14.1°, 0.2 dex (Fong+15; `ph:136-137`) | deg | shape | scan |
| `Gamma_c` | Γ_core | t_dec ∝ Γ^{-8/3}, beaming, θ_max where Γ₀ = 5 (`ph:297-310`) | lognormal, median 200, 0.2 dex (`ph:142-143`) | – | shape | prior |
| `beta1` | β₁ | rise index | normal 2.0 ± 0.05 (`ph:150`) | – | negligible | fixed 2.0 |
| `beta2` | β₂ | decay index; carries the anchor from 11 h to the observation | normal −1.45 ± 0.48 (`ph:151`) | – | shape | prior, truncated [−3, −0.5] |
| `p` | p | photon index | normal 2.2 ± 0.1 (`ph:167-168`) | – | spectrum exactly, light curve weakly (a^{2+p}) | prior |
| `dlogLX` | δ_X | scatter of L_X,11h about Berger (2014) | normal 0 ± 0.5 dex (Abe Eq. 5; `ph:156-158, 325-331`) | dex | amplitude | prior |
| `log10_R` | log R | L_TeV/L_X at 11 h | normal 0 ± 0.3 dex (`ph:162, 334-339`) | dex | amplitude | prior |
| `L_TeV_11h` | L_TeV,11h | on-axis 0.3–1 TeV luminosity at t′ = 11 h | derived: 8.5e43 (E/1e51)^0.83 × 10^(δ_X + log R) (`ph:542-547, 580`) | erg s⁻¹ | linear amplitude | derived; scan/solve_for in scan 2 |
| settings | – | structure, Γ₀ floor 5, t_anchor = 11 h rest frame (`ph:170-175`), anchor band 0.3–1 TeV, EBL franceschini17 | – | – | – | fixed |

Derived-parameter rule: `L_TeV_11h` is computed from the Berger chain unless it appears in `scan`, `priors` or
`fixed`; then the chain is cut (`dlogLX`, `log10_R` refused) and E_γ,iso acts only on the time scale.

### What each scan constrains

- **L_TeV,11h (recommended second scan).** Counts are exactly linear in it (P_det monotone, `ulfit.adaptive_fit`
  can solve for it); the 0.58 dex amplitude scatter drops out. Analogue of Ronchini's L₀. It constrains the on-axis
  luminosity at the observation epoch, carried to 11 h through β₂: for `sim_wobble` (1–81 min) that is ±0.65 dex of
  leverage. Make `anchor_time_h` a setting and also quote the limit at an anchor near the GTI mean (exact shift).
- **R (not scanned).** Well defined but prior-dominated: the data constrain only R·L_X, and the low tail of L_X is
  set by the E_iso population (~2.8 dex 16–84%), the arbitrary `EPEAK_MIN_KEV` cut and the 0.5 dex Berger scatter.
  R becomes meaningful with a measured X-ray afterglow: R_UL = L_TeV,UL / L_X,measured.
- **E_γ,iso (baseline).** Amplitude rises as E^0.83 but the peak is delayed as E^{1/3}; for off-axis observers
  before their peak P_det can decrease with E_iso. Grid axis only. Its exclusion is the L_TeV one widened by the
  0.58 dex scatter (~1.15 dex in E at 95%).
- **n.** Only rescales time, degenerate with E_k,iso, matters only near off-axis peaks. Nuisance.
- **Expected reach.** Benchmark L_TeV,11h = 2.2e43 erg/s (3.9e45 at E_iso = 1e53); the old S240615dg model limit
  was 2.0e47. S240615dg will probably exclude nothing in the E_iso scan; contours need `sim_wobble` (early) or
  `sim_tiling` (S191216ap, 343 Mpc).

## 2. Config format (`studies/2_model_ul/configs/<name>.yaml`)

```yaml
model: grb_afterglow            # grb_afterglow | grb_apparent | pwl | pwl_flux
structure: gaussian             # gaussian | tophat | powerlaw
case: sim_wobble
others: population              # REQUIRED: unlisted recipe inputs -> "population" or "benchmark"

scan:
  E_gamma_iso: {log: [1.0e49, 1.0e53], n: 12, unit: erg}
  theta_c:     {log: [3, 30], n: 10, unit: deg}       # or {lin: [..], n: ..} or {values: [..]}
# solve_for: L_TeV_11h

priors:
  Gamma_c:  {dist: population}
  beta2:    {dist: normal, mean: -1.45, sigma: 0.48, min: -3.0, max: -0.5, source: "Abe+26 Sec. 3.3, truncated"}
  p:        {dist: population}
  dlogLX:   {dist: population}
  log10_R:  {dist: population}
  n_ism:    {dist: loguniform, min: 1.0e-4, max: 1.0, unit: cm-3, source: "Fong+15 sGRB densities (not in the recipe)"}

fixed:
  eta_gamma: 0.2
  beta1: 2.0

settings:
  anchor_time_h: 11
  anchor_band_TeV: [0.3, 1.0]
  ebl: franceschini17
  sky_distance: pe              # pe | skymap
  outside_geometry: background  # background (undetected) | drop (condition on the field)
  light_curve: atlas            # atlas | direct

mc:
  method: fast                  # fast | full
  n_per_point: 20000            # full: ~1000-2000
  seed: 2000000                 # same seed at every point (common random numbers)
  cl: [0.5, 0.9, 0.95]
```

Second scan: `scan: {L_TeV_11h: {log: [1e42, 1e50], unit: erg/s}, theta_c: {...}}`, `solve_for: L_TeV_11h`, and
`dlogLX`, `log10_R` removed. Power law: `model: pwl`, `scan: {L0: {log: [1e43, 1e48], unit: erg/s}}`,
`solve_for: L0`, `fixed: {index: 2.0}`.

Distributions (all accept `min`/`max` truncation): `uniform`, `loguniform`, `normal`, `lognormal` (median,
sigma_dex), `broken_powerlaw`, `population` (the recipe's own, cited; `via: direct` for Nava's E_iso shortcut).

Validation (`scan.load` raises one `ConfigError` listing every problem): unknown names (with suggestion); each
parameter in one section; `others` required; units convertible; grids increasing, log > 0, 1–3 axes; `solve_for`
in `scan`, amplitude-type, no `n`; the derived-parameter rule; every non-`population` prior has `source:`; atlas
parameters and their prior supports inside the atlas range; `model`/`structure`/case compatible.
`scan.resolved_table(cfg)` prints every parameter with role, value/prior/grid, unit, source; saved with the results
and part of the cache key. Moving a parameter between sections is moving its line.

## 3. Computational strategy

### Exact factorisations
1. Amplitude: L = L_TeV,11h × rel(θ_v, t′)/rel(0, 11 h) × E′^{-p} (`ph:781-789`). δ_X, log R, Berger, the
   amplitude role of E_iso are free.
2. Time scale: E_k,iso and n enter only through t_dec ∝ (E_k,iso/n)^{1/3} f(θ) (`ph:731`). With
   s = [(E_k,iso/n)/ref]^{1/3}: L(θ_v, t′; s) = L_TeV,11h × rel_ref(θ_v, t′/s)/rel_ref(0, 11 h/s). Exact for the
   continuous integral (≤0.002 dex with the quadrature). E_iso, η, n need no atlas axis; the anchor shift is exact.
3. Γ_c partly: tabulate in τ = t′/t_dec,core to align peaks; interpolate the rest.
4. Observer projection: distance, z, EBL on a precomputed z × E table (port `gm:128-151`).
5. p: exact in the spectrum; its a^{2+p} light-curve effect neglected (≤0.01 dex to 45°; check at 60–90°).
6. β₁ fixed.

### Atlas
log rel(θ_v, τ; θ_c, Γ_c, β₂) at p = 2.2. Nodes: θ_v 0–60° every 1° then to 90° every 2.5° (73); θ_c 2–40° in
0.05 dex (27, scan values on nodes); log Γ_c 1.7–2.9 step 0.2 (7); β₂ −2.9 to −0.5 step 0.4 (7); τ 1e-2–1e8 at
24/decade (241). 96.6k element integrals at 0.3–1.1 s: 8–30 CPU-h, once, on the cluster (`scripts/build_atlas.py`,
one cached job per node), ~93 MB float32. Slim first version: θ_c only at the 10 scan values (3–11 CPU-h).
Multilinear interpolation, target ≤0.05 dex.

### Cost of a 12 × 10 scan

| strategy | realisations | CPU | gives |
|---|---|---|---|
| plan §4: 120 × 6000 | 7.2e5 | 100–400 h | overkill |
| full MC, N = 1000, atlas | 1.2e5 | 17–67 h | P̂ ± 0.007 at 0.95 |
| `solve_for` L_TeV, 10 θ_c columns × ≤6000 | ≤6e4 | 8–33 h | CL contour ± 0.03 dex, ceiling c(θ_c) |
| fast response, 120 × 1e5, atlas | 0 | minutes | full P maps; 50 × 50 grids free |
| fast-response spot checks, 10 × 2000 | 2e4 | 3–11 h | validation |

Approximate pieces: atlas interpolation, p in the light curve, element-integral convergence (≤0.02 dex at 80°),
P_det(s|b) (no neighbour leakage), binomial noise. Everything else exact.

## 4. Tools

- `uuls/grbmodel/recipe.py`: SOURCES, PARAMETERS (symbol, unit, kind, population prior, benchmark, source, atlas
  range), `population_prior`, vectorised `derive` (E_k,iso, L_TeV_11h, t_dec, θ_max), BENCHMARK. Port
  `ph:89-175, 185-261, 313-339, 525-575`.
- `uuls/grbmodel/structure.py`: gaussian, tophat, powerlaw; `theta_max_deg`. Port `ph:268-310`.
- `uuls/grbmodel/offaxis.py`: `relative_light_curve(shape, theta_v, t_rest, structure)`, bit-for-bit for gaussian.
  Port `ph:629-756`.
- `uuls/grbmodel/atlas.py`: `LightCurveModel` protocol (`anchor_ratio`, `describe`); `Atlas` (cached per node),
  `Direct` (reference), `Apparent` (Ronchini Eqs. 11-13).
- `uuls/grbmodel/observer.py`: `EBLTable`, `gti_time_samples` (`sim:1550-1568`), `run_spectra` (`sim:1577-1631`,
  anchor mode).
- `uuls/grbmodel/injector.py`: `GRBInjector(model, point, priors, pe, geom, run_gti_s, e_TeV, strength_param,
  settings)`; `rng.spawn(2)` for sky and nuisances, priors drawn in sorted name order; zeros outside the geometry.
- `uuls/priors.py`: `Prior` (Uniform, LogUniform, Normal, LogNormal, BrokenPowerLaw, Fixed, Population) with
  `sample`, `logpdf` (study 3), `quantile`, `describe`; `from_spec`; `plot`.
- `uuls/scan.py`: `load`, `resolved_table`, `context`, `points`, `injector_for`, `run_grid` (full or fast),
  `run_solve` (`ulfit.adaptive_fit` per column), `exclusion` (excluded / robust / uncertain), `contours`.
  Outputs in `outputs/results/2_model_ul/<case>/<config>/`. Cluster: `scripts/run_scan.py`.
- Needed changes elsewhere: `realisations.fake_counts` ignores `strength` for `RunSpectra`
  (`realisations.py:231-236`); `PWLInjector3D(sky_distance="skymap"|"pe")`.
- Response contract needed from phase 4a: `expected_counts(iy, ix, spectra) -> s`, `p_det(s, iy, ix)`.
- `pwl_flux` with `solve_for: phi0` must reproduce study 1's 2D limit bit for bit (test).

## 5. Notebooks (`studies/2_model_ul/`)

1. `1_recipe_and_priors`: parameters, one cell per prior with source, resolved table, regression of samplers.
2. `2_offaxis_jet`: port vs old at BENCHMARK (θ_v 0–90), light curves/spectra vs θ_v, time-stretch exactness,
   p-dependence at 60–90°, structures.
3. `3_atlas`: coverage, interpolation error at off-node points, node refinement.
4. `4_injection` (sim_wobble): PE draws, per-run spectra, EBL, expected counts vs θ_v, regression vs old, timing.
5. `5_benchmark_limit`: fixed benchmark, `solve_for` L_TeV,11h, regression vs old `_perform_3d_model` in anchor mode,
   closure; then the marginalised 1D limit.
6. `6_scan_Eiso_thetac`: P_det map, 50/90/95% contours with bands, ceiling, P_det by θ_v, spot checks.
7. `7_scan_LTeV_thetac`: `solve_for` contour with bands and ceiling; consistency with 6; anchor-time comparison.
8. `8_tophat_vs_ronchini`: apparent-structure models with Ronchini's priors; analytic ceiling check; Bayesian
   credible regions from the same grid overlaid (full treatment in study 3).
9. `9_pwl_2d_vs_3d`: power law Γ = 2 through the scan; 3D (skymap and PE routes) vs 2D at 5/50/95% distances.
10. `10_cases`: both scans on the three cases.

## 6. Risks and open questions (ranked)

1. Empty 95% exclusions are likely: PE θ_v tails cap P_det (S240615dg 17% beyond benchmark θ_max; S191216ap
   θ_v q95 = 68°) and recipe luminosities sit 2–4 dex below LST sensitivity at 1.5 Gpc. Decide the headline CL
   (90% like Ronchini?), scan ranges per case, and consider a BNS mock case (GW170817 PE/skymap, 40 Mpc).
2. θ_c convention: Abe Eq. 1 exp(−θ²/θ_c²) vs Nava and Ronchini exp(−θ²/2θ_c²): up to √2 on the axis. Check the
   typeset Abe paper.
3. Far off-axis physics (no lateral spreading, Γ floor, element normalisation) sets the 90–95% contours; report
   P_det by θ_v bin and test cutting emission beyond θ_max + X.
4. If the fast response fails, full MC with `solve_for` contours (8–33 CPU-h per scan per case) on the cluster.
5. E_iso not monotone: grid only. Confirm (L_TeV,11h, θ_c) and no R scan.
6. PE samples outside the geometry: count as undetected (proposed) or drop (old behaviour). Big effect for tiling.
7. Priors not in the recipe: n fixed in both sources; β₂ truncation; `EPEAK_MIN_KEV`; Amati without scatter.
8. "11 h" read as rest frame (factor ~1.46 at z = 0.3); t_dec prefactor assumed.
9. E_k convention for t_dec inconsistent in the old code (collimated `ph:441-450` vs isotropic `ph:549-563`);
   port the isotropic one.
10. The statement is a hybrid exclusion with power-constrained Λ*, not a credible region.
11. Atlas build and spot checks need the cluster.

## 7. Changes to the plan

- §4 cost overstated 6–12×; the YAML uses `{log: [1e49, 1e53], n:, unit:}`, truncated β₂, required `others:`,
  `source:` on every prior; exclusions carry MC intervals and an explicit "no exclusion possible" result; the 2D/3D
  comparison needs `PWLInjector3D` with the PE route; layout adds `uuls/scan.py` and `uuls/grbmodel/`.
- The old "one fixed jet loses little" argument holds only under `gti_mean`, not for study 2.
