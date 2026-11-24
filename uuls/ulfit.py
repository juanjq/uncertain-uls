"""Upper limit from a maximum-likelihood fit of the detection probability.

Every realisation i has its own strength x_i (phi0 in 2D, L0 in 3D) and gives
y_i = 1[Lambda_i > Lambda*], a Bernoulli draw with probability

    f(x) = f0 + (c - f0) Phi(z0 + (log10 x - log10 x_UL) / s),   z0 = Phi^-1((CL - f0) / (c - f0)),

so f(x_UL) = CL and the limit is a fit parameter. Nuisances: f0 (no signal; the
background-only sample, k of N above Lambda*, enters the same likelihood), c (ceiling for a
bright source, below 1 with a coverage cap), s (transition width in dex). The Monte Carlo
uncertainty of log10 x_UL is the profile likelihood interval (-2 Delta ln L = 1 and 4).

Lambda* is Lambda_obs, or the background median when the data under-fluctuate (S < 0).

`fit` fits a given sample once. `adaptive_fit` runs the scout + refine rounds of gwuls
(`_fit_ul`): a scout batch log-uniform over the bracket, then refine batches where the
curve carries the information on x_UL, until the 1 sigma half-width reaches `precision`.

Ported from gwuls.simulate Section 6; the fitting math is unchanged. The old `cache_path`
of rounds is gone: each batch is cached by `realisations.run` instead, and a rerun with
the same design seed asks for the same strengths, so it recomputes nothing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_ndtr, logit, ndtr, ndtri, xlogy
from scipy.stats import chi2 as chi2_dist
from scipy.stats import norm

FIT_ETA_WINDOW = (-2.5, 1.5)      # refine window: eta from -2.5 up to z0 + 1.5
FIT_TOP_FRACTION = 0.1            # share of each refine round above the window, for c


# --- target ------------------------------------------------------------------------

def target_lambda(lambda_obs, lambda_bkg):
    """Lambda* and the observed significance.

    p = P(Lambda_bkg > Lambda_obs), S = Phi^-1(1 - p); Lambda* = Lambda_obs if S >= 0,
    else median(Lambda_bkg). -> (target, p, S).
    """
    lam_bkg = np.asarray(lambda_bkg, dtype=float)
    p = float(np.mean(lam_bkg > lambda_obs))
    s = float(norm.ppf(1 - p))
    return (float(np.median(lam_bkg)) if s < 0 else float(lambda_obs)), p, s


def background_counts(lambda_bkg, target, cl):
    """(k, n): background realisations above `target`, out of the finite ones. Raises if
    k / n >= cl (no upper limit exists)."""
    lam = np.asarray(lambda_bkg, dtype=float)
    lam = lam[np.isfinite(lam)]
    n, k = lam.size, int(np.sum(lam > target))
    if n == 0 or k / n >= cl:
        raise ValueError(
            f"{k} of {n} background-only realisations exceed the target: with no signal the "
            f"fraction is already >= CL = {cl}, so no upper limit exists.")
    return k, n


# --- the probit model ---------------------------------------------------------------

def binomial_error(frac, n, floor=1e-6):
    """Standard error on a fraction, with a floor so 0 and 1 stay usable."""
    frac = np.clip(np.asarray(frac, dtype=float), 0.0, 1.0)
    n = np.maximum(np.asarray(n, dtype=float), 1.0)
    return np.maximum(np.sqrt(frac * (1.0 - frac) / n), floor)


def probit_params(theta, cl):
    """(log10 x_UL, ln s, logit(f0 / cl), logit((c - cl) / (1 - cl))) -> (t_ul, s, f0, c)."""
    th = np.asarray(theta, dtype=float)
    return (th[..., 0], np.exp(th[..., 1]), cl * expit(th[..., 2]),
            cl + (1.0 - cl) * expit(th[..., 3]))


def probit_eta(t, t_ul, s, f0, c, cl):
    """Linear predictor at t = log10 x."""
    return ndtri((cl - f0) / (c - f0)) + (t - t_ul) / s


def detection_probability(t, t_ul, s, f0, c, cl):
    """f(x) = f0 + (c - f0) Phi(eta) at t = log10 x; equal to `cl` at t = t_ul."""
    return f0 + (c - f0) * ndtr(probit_eta(t, t_ul, s, f0, c, cl))


def probit_nll(theta, t, y, k_bkg, n_bkg, cl):
    """-ln L of the realisations (t = log10 x, y bool) plus k_bkg of n_bkg background above."""
    t_ul, s, f0, c = probit_params(theta, cl)
    eta = probit_eta(t, t_ul, s, f0, c, cl)
    log_p = np.logaddexp(np.log(f0), np.log(c - f0) + log_ndtr(eta))
    log_q = np.logaddexp(np.log1p(-c), np.log(c - f0) + log_ndtr(-eta))
    return -(np.sum(np.where(y, log_p, log_q))
             + xlogy(k_bkg, f0) + xlogy(n_bkg - k_bkg, 1.0 - f0))


def param_bounds(t_lo, t_hi):
    """Bounds of theta for a bracket [t_lo, t_hi] in log10 x."""
    span = t_hi - t_lo
    return [(t_lo - span, t_hi + span), (np.log(1e-3), np.log(span)), (-30.0, 30.0), (-30.0, 30.0)]


def scout_starts(t_lo, t_hi, k_bkg, n_bkg, cl):
    """The five generic starting points spread over the bracket."""
    span = t_hi - t_lo
    v0 = float(logit(np.clip(k_bkg / n_bkg / cl, 1e-4, 1.0 - 1e-4)))
    return [(t_lo + q * span, np.log(span / 10.0), v0, 5.0) for q in (0.1, 0.3, 0.5, 0.7, 0.9)]


def probit_fit(t, y, k_bkg, n_bkg, cl, bounds, starts):
    """Best of several L-BFGS-B starts -> (theta (4,), nll)."""
    best = None
    for start in starts:
        start = np.clip(start, [b[0] for b in bounds], [b[1] for b in bounds])
        r = minimize(probit_nll, start, args=(t, y, k_bkg, n_bkg, cl),
                     method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    return np.asarray(best.x, dtype=float), float(best.fun)


def probit_cov(theta, t, y, k_bkg, n_bkg, cl, bounds, h=1e-3):
    """Covariance from the numerical Hessian at the MLE, (4, 4).

    Nuisances on a bound or saturated (|logit| > 7) are held fixed (zero rows). All NaN if
    log10 x_UL is on a bound or the Hessian is not positive definite.
    """
    n = len(theta)
    free = [i for i, (lo, hi) in enumerate(bounds) if lo + 10 * h < theta[i] < hi - 10 * h
            and not (i >= 2 and abs(theta[i]) > 7.0)]           # expit(7) = 0.9991
    if 0 not in free:
        return np.full((n, n), np.nan)

    def nll_shifted(i, si, j, sj):
        p = np.array(theta, dtype=float)
        p[i] += si * h
        p[j] += sj * h
        return probit_nll(p, t, y, k_bkg, n_bkg, cl)

    hess = np.empty((len(free), len(free)))
    for a, i in enumerate(free):
        for b in range(a, len(free)):
            j = free[b]
            hess[a, b] = hess[b, a] = (
                nll_shifted(i, 1, j, 1) - nll_shifted(i, 1, j, -1)
                - nll_shifted(i, -1, j, 1) + nll_shifted(i, -1, j, -1)) / (4.0 * h * h)
    if not np.all(np.linalg.eigvalsh(hess) > 0):
        return np.full((n, n), np.nan)
    cov = np.zeros((n, n))
    cov[np.ix_(free, free)] = np.linalg.inv(hess)
    return cov


def probit_profile(theta, nll_min, t, y, k_bkg, n_bkg, cl, bounds, half_width, n_grid=31):
    """Profile of log10 x_UL: -> (grid, -2 Delta ln L), nuisances refitted at each point
    (warm-started outwards); the grid is widened until it passes 4 on both sides."""
    for _ in range(5):
        grid = np.clip(theta[0] + np.linspace(-half_width, half_width, n_grid), *bounds[0])
        prof = np.empty(n_grid)
        mid = n_grid // 2
        for sweep in (range(mid, n_grid), range(mid, -1, -1)):
            start = theta[1:]
            for k in sweep:
                r = minimize(lambda p, tk=grid[k]: probit_nll((tk, *p), t, y, k_bkg, n_bkg, cl),
                             start, method="L-BFGS-B", bounds=bounds[1:])
                prof[k], start = r.fun, r.x
        d2 = 2.0 * (prof - min(nll_min, prof.min()))
        if min(d2[0], d2[-1]) > 4.0:
            break
        half_width *= 2.0
    return grid, d2


def profile_interval(grid, d2, level):
    """Where the profile crosses `level` either side of its minimum -> (lo, hi), NaN if never."""
    i0 = int(np.argmin(d2))

    def crossing(idx):
        for a, b in zip(idx[:-1], idx[1:]):
            if d2[b] >= level:
                return grid[a] + (level - d2[a]) * (grid[b] - grid[a]) / (d2[b] - d2[a])
        return np.nan

    return crossing(np.arange(i0, -1, -1)), crossing(np.arange(i0, len(grid)))


def probit_band(theta, cov, t_eval, cl, bounds, n_draws=500, seed=0):
    """Pointwise 1 and 2 sigma band of the curve from Gaussian parameter draws; None
    without a covariance."""
    if not np.all(np.isfinite(cov)):
        return None
    draws = np.random.default_rng(seed).multivariate_normal(theta, cov, n_draws, check_valid="ignore")
    draws = np.clip(draws, [b[0] for b in bounds], [b[1] for b in bounds])
    params = [q[:, None] for q in probit_params(draws, cl)]
    p = detection_probability(t_eval[None, :], *params, cl)
    q = np.percentile(p, [2.275, 15.866, 84.134, 97.725], axis=0)
    return {"p_lo_2sigma": q[0], "p_lo_1sigma": q[1], "p_hi_1sigma": q[2], "p_hi_2sigma": q[3]}


# --- diagnostics -------------------------------------------------------------------------

def equal_count_groups(t, n_groups):
    """Indices of t in `n_groups` groups of (nearly) equal size, in order of t."""
    return [g for g in np.array_split(np.argsort(t), max(1, min(n_groups, len(t)))) if len(g)]


def binned_fraction(t, y, n_groups):
    """Fraction above target in equal-count groups -> (mean t, frac, err, n)."""
    groups = equal_count_groups(t, n_groups)
    n = np.array([len(g) for g in groups])
    frac = np.array([y[g].mean() for g in groups])
    return np.array([t[g].mean() for g in groups]), frac, binomial_error(frac, n), n


def probit_gof(t, y, p, n_groups):
    """Hosmer-Lemeshow test in equal-count groups -> dict t, pulls, chi2, ndof, pvalue."""
    groups = equal_count_groups(t, n_groups)
    obs = np.array([y[g].sum() for g in groups], dtype=float)
    expected = np.array([p[g].sum() for g in groups])
    var = np.array([(p[g] * (1.0 - p[g])).sum() for g in groups])
    pulls = (obs - expected) / np.sqrt(np.maximum(var, 1e-12))
    chi2 = float(np.sum(pulls ** 2))
    ndof = max(len(groups) - 2, 1)
    return {"t": np.array([t[g].mean() for g in groups]), "pulls": pulls,
            "chi2": chi2, "ndof": ndof, "pvalue": float(chi2_dist.sf(chi2, ndof))}


def bracket_warning(t, y, t_ul, t_lo, t_hi, cl, names=("x_lo", "x_hi")):
    """After the scout: a message if [t_lo, t_hi] does not contain the crossing, else None."""
    n_edge = max(20, len(t) // 20)
    order = np.argsort(t)
    f_low, f_high = y[order[:n_edge]].mean(), y[order[-n_edge:]].mean()
    tol = 2.0 * float(binomial_error(cl, n_edge))
    if t_ul > t_hi or f_high < cl - tol:
        return (f"{f_high:.3f} of the realisations at the top of the bracket exceed the target "
                f"(CL = {cl}): the limit is ABOVE the bracket -- raise `{names[1]}` -- or the curve "
                f"saturates below the CL (3D with full-map sampling: the coverage cap).")
    if t_ul < t_lo or f_low > cl + tol:
        return (f"{f_low:.3f} of the realisations at the bottom of the bracket already exceed the "
                f"target (CL = {cl}): the limit is BELOW the bracket. Lower `{names[0]}`.")
    return None


# --- design of the rounds ------------------------------------------------------------------

def stratified_log_design(n, t_lo, t_hi, rng):
    """n values of log10 x, one in each of n equal slices of [t_lo, t_hi], in random order."""
    return t_lo + (t_hi - t_lo) * (rng.permutation(n) + rng.random(n)) / n


def refine_window(theta, cl, se, min_width):
    """Next window (eta from -2.5 to z0 + 1.5, never narrower than +-3 SE) and the range
    above it where the curve has saturated. -> ((lo, hi), (hi, hi + max(3 s, 0.3)))."""
    t_ul, s, f0, c = (float(q) for q in probit_params(theta, cl))
    s = max(s, min_width)
    z0 = ndtri((cl - f0) / (c - f0))
    pad = 3.0 * se if np.isfinite(se) else 0.0
    lo = min(t_ul + (FIT_ETA_WINDOW[0] - z0) * s, t_ul - pad)
    hi = max(t_ul + FIT_ETA_WINDOW[1] * s, t_ul + pad)
    return (lo, hi), (hi, hi + max(3.0 * s, 0.3))


def round_design(n_r, window, top, rng):
    """log10 x of one round: stratified over `window`, plus a tenth over `top` if given."""
    n_top = 0 if top is None else int(round(FIT_TOP_FRACTION * n_r))
    t_r = stratified_log_design(n_r - n_top, *window, rng)
    if n_top:
        t_r = rng.permutation(np.concatenate([t_r, stratified_log_design(n_top, *top, rng)]))
    return t_r


# --- result ------------------------------------------------------------------------------

@dataclass
class ULResult:
    """An upper limit, its MC uncertainty and the fit diagnostics. Strengths in the unit of
    the input x; `params`/`cov` refer to theta = (log10 x_UL, ln s, logit f0/cl, logit (c-cl)/(1-cl))."""
    x_ul: float
    x_ul_lo_1sigma: float
    x_ul_hi_1sigma: float
    x_ul_lo_2sigma: float
    x_ul_hi_2sigma: float
    log10_x_ul: float
    log10_x_ul_err: float            # 1 sigma half-width, dex
    width_dex: float
    width_dex_err: float
    f0: float
    f0_err: float
    ceiling: float
    ceiling_err: float
    f0_bkg: float
    n_bkg: int
    k_bkg: int
    target: float
    cl: float
    params: np.ndarray
    cov: np.ndarray
    fit_window: tuple
    profile: dict                    # x, delta_2nll
    curve: dict                      # x, p, and the band if a covariance exists
    gof: dict                        # x, pulls, chi2, ndof, pvalue
    binned: dict                     # x, frac, err, n
    samples: dict                    # x, lambda, round
    rounds: list = field(default_factory=list)
    converged: bool = False
    at_bracket_edge: bool = False
    bracket_warning: str | None = None
    n_sim_total: int = 0
    precision: float = np.nan

    def to_dict(self) -> dict:
        return asdict(self)

    def report(self, label="x", unit="") -> str:
        """Plain-text summary: limit, 1 and 2 sigma MC intervals, fit diagnostics."""
        x = self.x_ul
        lines = [f"{label}_UL = {x:.4e} {unit}  (CL {self.cl}, target Lambda {self.target:.3f})"]
        for tag in ("1sigma", "2sigma"):
            lo, hi = getattr(self, f"x_ul_lo_{tag}"), getattr(self, f"x_ul_hi_{tag}")
            lines.append(f"  {tag[0]} sigma  {(lo / x - 1) * 100:+6.1f}% / {(hi / x - 1) * 100:+6.1f}%"
                         f"   [{lo:.3e}, {hi:.3e}] {unit}")
        lines.append(f"  width {self.width_dex:.3f} +- {self.width_dex_err:.3f} dex, f0 {self.f0:.3f}"
                     f" +- {self.f0_err:.3f} (background only {self.f0_bkg:.3f}), ceiling "
                     f"{self.ceiling:.4f} +- {self.ceiling_err:.4f}")
        lines.append(f"  goodness of fit chi2/ndof = {self.gof['chi2']:.1f}/{self.gof['ndof']} "
                     f"(p = {self.gof['pvalue']:.2f}); {len(self.rounds)} round(s), "
                     f"{self.n_sim_total} realisations, converged {self.converged}")
        if self.at_bracket_edge:
            lines.append("  WARNING: the limit or its 1 sigma interval reaches the edge of the bracket.")
        if self.bracket_warning:
            lines.append(f"  BRACKET PROBLEM: {self.bracket_warning}")
        return "\n".join(lines)


def _round_entry(r, x_r, n_used, n_fit, cached, theta, lo1, hi1, half, cl):
    _, s, f0, c = (float(q) for q in probit_params(theta, cl))
    return {"round": r, "n": len(x_r), "n_total": n_used, "n_fit": n_fit,
            "x_min": float(x_r.min()), "x_max": float(x_r.max()), "cached": cached,
            "x_ul": 10.0 ** theta[0], "x_ul_lo_1sigma": 10.0 ** lo1,
            "x_ul_hi_1sigma": 10.0 ** hi1, "half_width_dex": half,
            "width_dex": s, "f0": f0, "ceiling": c}


def _result(theta, cov, grid, d2, ts, ys, t_all, lam_all, r_all, ok, y_all, bounds, t_lo, t_hi,
            fit_window, k_bkg, n_bkg, cl, target, history, converged, warning, n_used,
            precision) -> ULResult:
    """Final curve, band, goodness of fit, binned data -> ULResult (as the end of _fit_ul)."""
    lo1, hi1 = profile_interval(grid, d2, 1.0)
    half = 0.5 * (hi1 - lo1)
    lo2, hi2 = profile_interval(grid, d2, 4.0)
    t_eval = np.linspace(t_all.min(), t_all.max(), 400)
    params = probit_params(theta, cl)
    band = probit_band(theta, cov, t_eval, cl, bounds)
    gof = probit_gof(ts, ys, detection_probability(ts, *params, cl),
                     int(np.clip(len(ts) // 100, 4, 15)))
    gof["x"] = 10.0 ** gof.pop("t")
    t_b, f_b, e_b, n_b = binned_fraction(t_all[ok], y_all[ok], int(np.clip(ok.sum() // 150, 8, 30)))
    at_edge = bool(not (t_lo < theta[0] < t_hi)
                   or (np.isfinite(lo1) and not (t_lo < lo1 and hi1 < t_hi)))
    ev, ew = expit(theta[2]), expit(theta[3])
    return ULResult(
        x_ul=float(10.0 ** theta[0]),
        x_ul_lo_1sigma=float(10.0 ** lo1), x_ul_hi_1sigma=float(10.0 ** hi1),
        x_ul_lo_2sigma=float(10.0 ** lo2), x_ul_hi_2sigma=float(10.0 ** hi2),
        log10_x_ul=float(theta[0]), log10_x_ul_err=float(half),
        width_dex=float(params[1]), width_dex_err=float(params[1] * np.sqrt(cov[1, 1])),
        f0=float(params[2]), f0_err=float(cl * ev * (1 - ev) * np.sqrt(cov[2, 2])),
        ceiling=float(params[3]), ceiling_err=float((1 - cl) * ew * (1 - ew) * np.sqrt(cov[3, 3])),
        f0_bkg=k_bkg / n_bkg, n_bkg=n_bkg, k_bkg=k_bkg, target=float(target), cl=float(cl),
        params=theta, cov=cov,
        fit_window=(float(10.0 ** fit_window[0]), float(10.0 ** fit_window[1])),
        profile={"x": 10.0 ** grid, "delta_2nll": d2},
        curve={"x": 10.0 ** t_eval, "p": detection_probability(t_eval, *params, cl), **(band or {})},
        gof=gof, binned={"x": 10.0 ** t_b, "frac": f_b, "err": e_b, "n": n_b},
        samples={"x": 10.0 ** t_all, "lambda": lam_all, "round": r_all},
        rounds=history, converged=bool(converged), at_bracket_edge=at_edge,
        bracket_warning=warning, n_sim_total=int(n_used), precision=float(precision))


def _fit_and_profile(ts, ys, k_bkg, n_bkg, cl, bounds, starts, span):
    """Fit, covariance, profile -> (theta, nll, cov, se, grid, d2)."""
    theta, nll = probit_fit(ts, ys, k_bkg, n_bkg, cl, bounds, starts)
    cov = probit_cov(theta, ts, ys, k_bkg, n_bkg, cl, bounds)
    se = float(np.sqrt(cov[0, 0]))
    grid, d2 = probit_profile(theta, nll, ts, ys, k_bkg, n_bkg, cl, bounds,
                              half_width=4.0 * se if np.isfinite(se) and se > 0 else 0.1 * span)
    return theta, nll, cov, se, grid, d2


# --- one fit of a given sample -------------------------------------------------------------

def fit(strength, lambda_sim, lambda_obs, lambda_bkg, cl=0.95, bracket=None, target=None) -> ULResult:
    """Probit fit of one sample of realisations.

    strength: (n,) > 0, the injected x_i; lambda_sim: (n,) Lambda_i; lambda_obs: observed
    Lambda; lambda_bkg: background-only Lambdas; bracket: (x_lo, x_hi) for the parameter
    bounds (default: the range of `strength`); target: overrides Lambda* (default
    `target_lambda`). Same fit as the scout round of `adaptive_fit`.
    """
    x = np.asarray(strength, dtype=float)
    if np.any(x <= 0):
        raise ValueError("strengths must be > 0 (log10 x is fitted)")
    if target is None:
        target = target_lambda(lambda_obs, lambda_bkg)[0]
    k_bkg, n_bkg = background_counts(lambda_bkg, target, cl)
    x_lo, x_hi = bracket if bracket is not None else (x.min(), x.max())
    t_lo, t_hi = float(np.log10(x_lo)), float(np.log10(x_hi))
    bounds = param_bounds(t_lo, t_hi)
    t_all, lam_all = np.log10(x), np.asarray(lambda_sim, dtype=float)
    ok = np.isfinite(lam_all)
    y_all = lam_all > target
    ts, ys = t_all[ok], y_all[ok]
    theta, nll, cov, se, grid, d2 = _fit_and_profile(
        ts, ys, k_bkg, n_bkg, cl, bounds, scout_starts(t_lo, t_hi, k_bkg, n_bkg, cl), t_hi - t_lo)
    lo1, hi1 = profile_interval(grid, d2, 1.0)
    history = [_round_entry(0, x, x.size, int(ok.sum()), False, theta, lo1, hi1,
                            0.5 * (hi1 - lo1), cl)]
    warning = bracket_warning(ts, ys, theta[0], t_lo, t_hi, cl)
    return _result(theta, cov, grid, d2, ts, ys, t_all, lam_all, np.zeros(x.size, int), ok, y_all,
                   bounds, t_lo, t_hi, (t_lo, t_hi), k_bkg, n_bkg, cl, target, history, False,
                   warning, x.size, np.nan)


# --- the adaptive rounds -----------------------------------------------------------------------

def adaptive_fit(simulate_batch, x_lo, x_hi, lambda_obs, lambda_bkg, cl=0.95, target=None,
                 n_scout=500, n_refine=1000, precision=0.03, max_sims=6000, max_rounds=6,
                 design_seed=0, label="x", unit="", bracket_names=("x_lo", "x_hi"),
                 verbose=True) -> ULResult:
    """Upper limit from scout + refine rounds of realisations.

    simulate_batch(x, round_index, first_index) -> Lambda (len(x),): realisation i of the
    batch must use global index first_index + i (see `realisations.batch_function`).
    Round r draws its strengths with np.random.default_rng([design_seed, r]).
    Stops when the 1 sigma profile half-width of log10 x_UL is <= `precision` dex (after at
    least one refine round, with a resolved transition), or at `max_rounds` / `max_sims`.
    """
    if target is None:
        target = target_lambda(lambda_obs, lambda_bkg)[0]
    k_bkg, n_bkg = background_counts(lambda_bkg, target, cl)
    t_lo, t_hi = float(np.log10(x_lo)), float(np.log10(x_hi))
    span = t_hi - t_lo
    bounds = param_bounds(t_lo, t_hi)
    starts0 = scout_starts(t_lo, t_hi, k_bkg, n_bkg, cl)
    v0, w0 = starts0[0][2], starts0[0][3]

    t_all, lam_all, r_all = np.empty(0), np.empty(0), np.empty(0, dtype=int)
    history, theta, window, top = [], None, (t_lo, t_hi), None
    converged, warning, n_used, n_next = False, None, 0, n_scout

    for r in range(max_rounds):
        n_r = int(min(n_next, max_sims - n_used))
        if n_r <= 0:
            break
        x_r = 10.0 ** round_design(n_r, window, top, np.random.default_rng([int(design_seed), r]))
        lam_r = np.asarray(simulate_batch(x_r, r, n_used), dtype=float)
        n_used += len(x_r)
        t_all = np.concatenate([t_all, np.log10(x_r)])
        lam_all = np.concatenate([lam_all, lam_r])
        r_all = np.concatenate([r_all, np.full(len(x_r), r)])

        # fit the realisations from the bottom of the window up
        ok = np.isfinite(lam_all)
        y_all = lam_all > target
        sel = ok & (t_all >= window[0])
        if sel.sum() < 50:
            sel = ok
        ts, ys = t_all[sel], y_all[sel]
        if theta is None:
            starts = starts0
        else:     # the last estimate, plus generic starts in case it was a near-step
            w = np.log((window[1] - window[0]) / 4.0)
            starts = [theta, (theta[0], max(theta[1], w), theta[2], theta[3]),
                      (0.5 * (window[0] + window[1]), w, v0, w0)]
        theta, nll = probit_fit(ts, ys, k_bkg, n_bkg, cl, bounds, starts)
        inside = r == 0 or window[0] < theta[0] < window[1]
        if not inside:   # crossing outside this window: locate it again with every realisation
            sel = ok
            ts, ys = t_all[sel], y_all[sel]
            theta, nll = probit_fit(ts, ys, k_bkg, n_bkg, cl, bounds, starts + starts0)
        cov = probit_cov(theta, ts, ys, k_bkg, n_bkg, cl, bounds)
        se = float(np.sqrt(cov[0, 0]))
        grid, d2 = probit_profile(theta, nll, ts, ys, k_bkg, n_bkg, cl, bounds,
                                  half_width=4.0 * se if np.isfinite(se) and se > 0 else 0.1 * span)
        lo1, hi1 = profile_interval(grid, d2, 1.0)
        half = 0.5 * (hi1 - lo1)
        fit_window = window if inside else (t_all[ok].min(), t_all[ok].max())
        s = float(probit_params(theta, cl)[1])
        # a transition narrower than a few point spacings is not resolved
        n_win = int(np.sum(ok & (t_all >= fit_window[0]) & (t_all <= fit_window[1])))
        spacing = (fit_window[1] - fit_window[0]) / max(n_win, 1)
        resolved = s >= 3.0 * spacing
        history.append(_round_entry(r, x_r, n_used, int(sel.sum()), False, theta, lo1, hi1, half, cl))
        if verbose:
            h = history[-1]
            print(f"[{'scout' if r == 0 else f'round {r}':>7}] N = {len(x_r):5d}, total {n_used}, "
                  f"fit on {int(sel.sum())}{'' if inside else ' (refitted on all)'}: {label}_UL = "
                  f"{h['x_ul']:.4e} {unit} [{h['x_ul_lo_1sigma']:.4e}, {h['x_ul_hi_1sigma']:.4e}] "
                  f"(+-{half:.4f} dex), width {s:.3f} dex, f0 {h['f0']:.3f}, c {h['ceiling']:.4f}"
                  f"{'' if resolved else ' (transition not resolved yet)'}")

        if r == 0:
            warning = bracket_warning(t_all[ok], y_all[ok], theta[0], t_lo, t_hi, cl, bracket_names)
            if warning:
                if verbose:
                    print(f"  ERROR: {warning}")
                break
        if r > 0 and inside and resolved and np.isfinite(half) and half <= precision:
            converged = True
            break

        window, top = refine_window(theta, cl, se, min_width=10.0 * spacing)
        n_in = int(np.sum(ok & (t_all >= window[0]) & (t_all <= window[1])))
        if r == 0 or not np.isfinite(half):
            n_next = n_refine
        else:     # the interval shrinks as 1/sqrt(N) of the realisations in the window
            n_next = int(np.clip(np.ceil(n_in * ((half / precision) ** 2 - 1.0)),
                                 max(100, n_refine // 10), n_refine))

    if theta is None:
        raise RuntimeError("No realisation was simulated (max_sims = 0?).")
    if verbose and not converged and not warning:
        print(f"  stopped after {len(history)} round(s) and {n_used} realisations with a 1 sigma "
              f"half-width of {half:.4f} dex > precision {precision} dex")
    return _result(theta, cov, grid, d2, ts, ys, t_all, lam_all, r_all, ok, y_all, bounds, t_lo,
                   t_hi, fit_window, k_bkg, n_bkg, cl, target, history, converged, warning,
                   n_used, precision)


# --- closure -----------------------------------------------------------------------------------

def closure_pull(result: ULResult, lambda_closure) -> dict:
    """Fraction of an independent Lambda sample simulated AT x_UL above the target, vs CL.

    The error combines the binomial error with the limit's own 1 sigma MC uncertainty mapped
    through the fitted curve. -> dict frac, err, err_ul, err_tot, pull, n.
    """
    lam = np.asarray(lambda_closure, dtype=float)
    lam = lam[np.isfinite(lam)]
    frac = float(np.mean(lam > result.target))
    err = float(binomial_error(frac, lam.size))
    err_ul = 0.0
    curve = result.curve
    if curve is not None and np.isfinite(result.x_ul_lo_1sigma):
        p_lo, p_hi = np.interp(np.log10([result.x_ul_lo_1sigma, result.x_ul_hi_1sigma]),
                               np.log10(curve["x"]), curve["p"])
        err_ul = 0.5 * abs(p_hi - p_lo)
    err_tot = float(np.hypot(err, err_ul))
    return {"frac": frac, "err": err, "err_ul": err_ul, "err_tot": err_tot,
            "pull": (frac - result.cl) / err_tot, "n": int(lam.size)}
