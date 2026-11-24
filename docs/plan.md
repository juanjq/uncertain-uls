# From a notebooks folder to a project: plan

Status: draft for review (2026-11-23). Nothing has been moved yet.

## 0. What exists and what blocks us

- `gwuls/` (7.5k lines) already implements most of the physics: GW map on a WCS grid, the global
  statistic Λ = max_b (TS_b + 2 ln p_b), Monte Carlo realisations, the probit UL fit, flux ↔ luminosity,
  the benchmark GRB afterglow, PE handling, p(θ_v | d). `simulate.py` (3k lines) mixes all of these.
- `lstreco/simulst` simulates LST-1 DL3 runs (`simulate_pointing`) and map datasets (`simulate_maps`)
  from a background library and the all-sky IRFs.
- Blockers found:
  1. `notebooks/1_grb_model.ipynb` was committed with an unresolved merge conflict (`<<<<<<< HEAD` at line 298). It does not open.
  2. No conda env on this machine has `ligo.skymap` or `tilepy`. The `gw-gamma-global-uls` env is not here.
  3. simulst pins gammapy 1.3; gwuls uses gammapy 2.1. They cannot share one Python process.
  4. simulst offline mode needs `data/irf` (79 MB from PIC) and the rebuilt library; neither is on this machine.
  5. The real S240615dg `.pkl`/DL3 are not here (you will copy them).

Consequence of 3: simulst and this project talk through files on disk (DL3 FITS or MapDataset FITS),
and simulst runs in its own env via its CLI or a subprocess. That is also the cleanest interface.

## 1. Layout

One repo, one package of small tools, one folder per study. simulst and tilepy stay external dependencies.

```
gw-gamma-global-uls/
├── uuls/                     # shared tools (new package; physics ported from ../gw-gamma-global-uls/gwuls)
│   ├── cache.py                 # cached(name, params, func) -> result; params.json next to output
│   ├── figures.py               # save_fig, plot style
│   ├── gwmap.py                 # skymap -> GWMap on a WCS geom (p_b, mask, distance CDF, t_merger)
│   ├── gwpe.py                  # PE release -> standardized samples (RA, Dec, d, θ_v)
│   ├── pointings.py             # on / wobble / grid patterns + tilepy wrapper -> Pointings table
│   ├── eventtime.py             # choose a merger time at which a region is observable from La Palma
│   ├── simdata.py               # Pointings + source -> simulst DL3 (subprocess, cached)
│   ├── reduce.py                # DL3 dir (real or simulated) + GWMap -> ReducedData (.pkl contract)
│   ├── lambda_stat.py           # TS map, Λ, fast Poisson-only path
│   ├── realisations.py          # inject source, Poisson fake, Λ, parallel and seeded
│   ├── ulfit.py                 # probit fit of P(Λ > Λ*) vs strength -> UL ± MC error
│   ├── spectra.py               # PWL conversions, flux ↔ luminosity, k-correction, EBL
│   ├── grbmodel/                # afterglow recipe, jet structure, off-axis integral, parameter space
│   ├── priors.py                # Prior objects: sample(n, rng), logpdf(x), describe()
│   └── response.py              # P_det(s | b): detection probability vs expected signal counts per bin
├── studies/
│   ├── 1_flux_ul/               # notebooks + configs/*.yaml + scripts/
│   ├── 2_model_ul/
│   ├── 3_bayesian/
│   └── 4_tiling/
├── configs/cases/               # the shared mock cases (see §2)
├── tests/
├── data/  outputs/  cache/      # ignored, except tracked small inputs as today
└── docs/
```

Each tool has one input type and one output type, written in its docstring, and files on disk in
plain formats (ECSV for tables, FITS for maps, `.npz`/`.pkl` for arrays). Example:

| Tool | In | Out |
| --- | --- | --- |
| `gwmap.load` | skymap path, WCS geom | `GWMap` |
| `pointings.wobble / grid / tilepy` | centre or skymap, time window, settings | `Pointings` table (ra, dec, t_start, duration, obs_id) |
| `eventtime.observable_time` | sky position, date, site, max zenith | merger time (UTC) |
| `simdata.simulate` | `Pointings`, optional source model, seed | DL3 folder |
| `reduce.from_dl3` | DL3 folder, `GWMap`, analysis settings | `ReducedData` |
| `realisations.run` | `ReducedData`, injection sampler, n, seed | Λ array (+ what was injected) |
| `ulfit.fit` | strengths, Λ, Λ*, CL | UL, interval, diagnostics |
| `response.build` | `ReducedData` | per-bin table of P_det(s) |

## 2. The mock cases (shared by studies 1–3)

| Case | GW | Data | Pointings |
| --- | --- | --- | --- |
| `real_S240615dg` | S240615dg skymap + PE | real DL3 (when copied) | as observed |
| `sim_wobble` | S240615dg (or any) | simulst | 4 wobbles at 0.4° around the 50% peak |
| `sim_tiling` | smallest-area event in GWTC with public PE (to be chosen, §7 Q4) | simulst | tilepy tiling |

The merger time of simulated cases is a parameter. Default: `eventtime.observable_time` returns the
first night after a reference date when the skymap's highest-probability point culminates in
darkness below 40° zenith. The skymap is in equatorial coordinates, so only the time changes; nothing
is rotated. The real case keeps the real time.

## 3. Study 1: 2D flux UL (frequentist)

What it is: the existing method. A realisation draws a sky bin from the 95% region, injects a PWL
(Γ = 2) at amplitude φ₀, Poisson-fakes counts, computes Λ. The UL is the φ₀ at which P(Λ > Λ_obs) = CL.

Notebooks:
1. `1_case.ipynb`: load a case config, GW map, pointings, plots of coverage.
2. `2_data.ipynb`: simulate (or point to real) DL3, reduce, show counts/background/exposure, observed Λ.
3. `3_background.ipynb`: background-only realisations, Λ distribution, p-value of the data.
4. `4_flux_ul.ipynb`: the probit fit and UL, fit diagnostics, closure test.
5. `5_cases.ipynb`: the three cases side by side.

Questions to settle here:
- **Full DL3 or maps.** Proposal: the observed dataset of a simulated case is built from simulst DL3,
  so the real and simulated paths share `reduce.from_dl3` and the ring background exactly. The Monte
  Carlo realisations already fake counts from the reduced maps, so they never need DL3. A maps-only
  shortcut (`simulst.simulate_maps`) is used only where many cases are needed (study 4).
- **Regression.** Before deleting anything in `simulate.py`, the extracted tools must reproduce the
  S240615dg numbers in the README (φ₀ = 2.34e-12) with the same seed. This needs the S240615dg `.pkl`.
- Open item from the README: the global UL 17% below the largest per-pixel UL in M95. Worth one
  notebook cell in `4_flux_ul` once the response tool exists (§5), because it predicts this directly.

## 4. Study 2: 3D UL constraining model parameters

What changes from today: today one fixed benchmark jet is scaled by a luminosity. The new version
picks a few **parameters of interest** ψ (e.g. E_γ,iso or E_k, θ_core) and treats everything else
as nuisance η with a prior. For each grid point ψ_k, realisations draw (sky, d, θ_v) from the GW PE
and η from its prior, and the detection fraction P(Λ > Λ_obs | ψ_k) is computed. Points where it
exceeds CL are excluded. This is a hybrid (Cousins–Highland) frequentist test: nuisances are
marginalised, the parameters of interest are not.

How the parameter split is written (intuitive and explicit):

```yaml
model: grb_afterglow
scan:                       # parameters of interest: a grid
  E_iso:   {log10: [49, 53], n: 12, unit: erg}
  theta_c: {lin: [2, 20], n: 10, unit: deg}
priors:                     # nuisances: sampled per realisation
  n_ism:   {loguniform: [1e-3, 1], unit: cm-3}
  p:       {normal: [2.2, 0.1]}
  beta2:   {normal: [-1.45, 0.48]}
fixed:
  eta_gamma: 0.2
```

Any parameter can move between `scan`, `priors` and `fixed` without code changes. The PWL Γ = 2
case is the same machinery with `scan: {L0: ...}` and no priors.

Priors: start from the population distributions in Nava (2020) / Abe et al. (2026), already cited in
`grb_phenomenological.SOURCES`. Each prior gets one cell in the notebook that plots it and cites it.

Cost: today one UL takes up to 6000 realisations. A 12 × 10 grid with the full Monte Carlo is
120 × that. This is why §5 matters.

Comparison with study 1 "where possible": for the PWL case, the 3D luminosity UL converted to flux
at the distance posterior, against the 2D flux UL (already in `compare_2d_3d`).

## 5. Study 3: Bayesian likelihood of no detection, and a shared fast response

Ronchini et al. (2024) use, for each sample (Ω, d, θ_v) of the GW posterior, the step function
P(F < UL(Ω)) with a per-pixel flux UL, and the likelihood P(no detection | ψ) is the fraction of
samples below the UL. The posterior is that likelihood times priors on (L₀, θ_c).

Points to question before copying it:

1. **The likelihood is the same integral as our detection fraction.** Their P(no det | ψ) is
   1 − P(Λ > Λ* | ψ) with a cruder detector model (a step at a 5σ per-pixel UL instead of the Poisson
   probability of the actual Λ). So studies 2 and 3 share one quantity; they differ in how it is
   turned into a statement (exclusion at fixed ψ vs. posterior over ψ).
2. **Prior dependence of a Bayesian UL.** When the jet can be off-axis, P(no det | L₀) does not go
   to zero as L₀ → ∞ (their Fig. 2 flattens above 1e50). The posterior then has a tail whose mass is
   set by the prior's upper edge, so a "90% UL on L₀" depends on choosing 1e54 vs 1e56. Their Table 1
   shows this (90th percentile 4.7e53 for a top-hat). The frequentist exclusion has no such
   dependence. The comparison notebook should show both, and vary the prior edge.
3. **Their threshold is 5σ, ours is Λ_obs.** Use both definitions in the comparison so the
   difference between methods is not confused with a difference of threshold.

The shared fast tool, `response.py`: Λ is built from counts n_b and background μ_b correlated over
r_c and summed over energy and runs. So, for a point source at bin b, detection depends on the
source only through its expected correlated signal counts s_b (plus leakage to neighbour bins).
`response.build` tabulates P_det(s | b) once per dataset (analytically from Poisson, checked against
the full Monte Carlo). Then for any model, any parameter point, any prior:

    s_b = Σ_runs Σ_E exposure_run(b, E) × F_model(E, t ∈ run; ψ, η, d, θ_v) × containment
    P(no det | ψ) = mean over GW samples and η draws of 1 − P_det(s_b | b)

This turns a grid scan from hours into seconds. It must be validated against the full Monte Carlo
on all three cases (that validation is a notebook in study 3, and the full MC stays the reference).
If it fails for some regime (very extended PSF leakage, many overlapping tiles), we learn where.

Notebooks: `1_response.ipynb` (build and validate P_det), `2_likelihood.ipynb` (P(no det | ψ) on the
grid), `3_posterior.ipynb` (priors, sampler, corner plots), `4_vs_frequentist.ipynb` (same case,
same model, both statements, prior-edge test), `5_hypotheses.ipynb` (top-hat, Gaussian, PWL, our
afterglow, as in Ronchini §6).

## 6. Study 4: tiling

Standard: tilepy 2D (maximise GW probability per tile) and 3D (galaxy-weighted).

Model-aware alternative. The idea "further sources are more likely seen face-on" is already in the
GW posterior: the Malmquist selection makes p(θ_v | d) favour small θ_v at large d (our
`angle_distribution.py` builds it). So no extra assumption is needed. For each sky bin, the
expected detection probability under the model prior is

    W(Ω) = p_GW(Ω) × E_{d, θ_v | Ω} E_{ψ, η ∼ prior} [ P_det(s(Ω, d, θ_v, ψ, η, t) | Ω) ]

which replaces p_GW(Ω) as the tile score. It also depends on time (afterglow fades, so an early
tile is worth more), which tilepy does not see.

Is maximising W the same as maximising constraining power? For a non-detection, the expected
prior mass removed is ∫ π(ψ) P_det(ψ) dψ, which is the prior average in W. So greedily maximising
the sum of W over chosen tiles is a defensible single objective, not just a heuristic. That needs to
be checked numerically: compare, per strategy, (a) P(detect), (b) excluded area in the (E_iso, θ_c)
plane from study 2, (c) Bayesian posterior shrinkage from study 3.

Cost: thousands of strategies × realisations is only feasible with `response.py` and maps-only data
(exposure from IRFs, no events).

Notebooks: `1_standard_tiling.ipynb`, `2_expectation_maps.ipynb` (W(Ω) and its pieces, per time),
`3_model_aware_tiling.ipynb` (greedy scheduler on W), `4_compare.ipynb`.

## 7. Open decisions (owner)

1. Restructure this repo in place (keeps history, `gwuls` → `gwgamma` with a compatibility shim), or start a new repo and keep this one frozen?
2. Environment: may I create the env here (`conda env create`, ~3 GB), adding `tilepy` and `emcee`?
3. `1_grb_model.ipynb` merge conflict: keep HEAD, keep the other side, or re-run it after the move?
4. `sim_tiling` event: smallest 90% area among GWTC events with public PE, or restrict to events with a NS (BNS/NSBH), even if larger?
5. Parameters of interest for study 2: (E_iso, θ_core) as in Ronchini's (L₀, θ_c), or something else?

## 8. Phases and agents

Each phase ends with a check I run myself (tests pass, notebooks execute, figures saved, JSON valid).
No agent commits. "Effort" is the reasoning effort the agent runs at.

| Phase | Work | Agent | Effort | Depends on |
| --- | --- | --- | --- | --- |
| 0 | Env, skeleton, `cache.py`, `figures.py`, CLAUDE.md, fix notebook 1 | me | high | decisions 1–3 |
| 1a | `gwmap`, `gwpe` extracted, tests | general-purpose | medium | 0 |
| 1b | `lambda_stat`, `realisations`, `ulfit`, `spectra` extracted, regression vs. old `simulate.py` | general-purpose | high | 0, S240615dg `.pkl` |
| 1c | `pointings`, `eventtime`, tilepy wrapper | general-purpose | medium | 0 |
| 1d | `simdata` (simulst subprocess) + `reduce.from_dl3` | general-purpose | high | 0, simulst IRFs |
| 2 | Study 1 notebooks on the three cases | general-purpose | high | 1a–1d |
| 3a | `priors.py`, `grbmodel` parameter space, scan config | Plan, then general-purpose | xhigh | 1b |
| 3b | Study 2 notebooks | general-purpose | high | 3a, 4a |
| 4a | `response.py` and its validation against the full MC | general-purpose | xhigh | 1b, 2 |
| 4b | Study 3 notebooks | general-purpose | high | 3a, 4a |
| 5 | Study 4 notebooks | Plan, then general-purpose | high | 4a |
| every phase | style and clarity review of new notebooks against CLAUDE.md | Explore | low | |

1a, 1c and 1d run in parallel; 1b needs the regression data. 4a comes before 3b on purpose: if the
fast response holds, study 2's grid scan uses it with full-MC spot checks; if not, study 2 runs on
the full MC with a coarser grid.

## 9. Decisions (2026-11-23)

1. New repo `uncertain-uls`, package `uuls`, built from scratch. `gw-gamma-global-uls` is the read-only reference.
2. Envs created here; recipes in `env/` and the README.
3. The GRB model notebook is rebuilt in the new structure (the old one has a committed merge conflict).
4. Tiling case: an event whose 90% area makes tiling non-trivial (tens of LST fields), not the best localised. Not restricted to NS events.
5. Study 2 parameters of interest: (E_iso, θ_core) to start, open to a better choice.
6. simulst through its offline mode (IRFs copied from PIC, library rebuilt from the wiki data).
7. Tool notebooks live in `studies/0_tools/`.

## 10. Decisions after phase 1 (2026-11-23)

1. simulst must give exposure to 2.5 deg offset (offline mode included). The 2.5 deg IRFs exist on the cluster
   (simulst `docs/irf-2p5-report.md`) but the simulator does not use them yet. Being solved by an agent.
2. Background: default to the run background model with a fitted normalisation; ring kept for comparison.
   Open problem to understand deeply before real data: on real data the background must be estimated from the
   data, and the ring absorbs part of any source, which the Monte Carlo does not reproduce (UL too low).
3. Tiling cases: one small geometry per field, Λ = max over all fields. Overlaps need care.
4. Tiling max zenith 55 deg (simulst refuses >= 60).
5. Keep all simulated events (ULs may be wanted at low energy); accept the cache size.
6. 2D injections jittered uniformly inside their bin.
7. S191216ap PE downloaded (Zenodo 8177023, C01:Mixed).
8. Real S240615dg pointings built from the real obs-index when the DL3 is copied.
9. ROI = GW 95% mask ∩ observed region (exposure > 0, safe mask), for data and Monte Carlo, single- and multi-field
   (2026-11-24; `lambda_stat.roi`, `roi="gw_mask"` keeps the reference behaviour).
