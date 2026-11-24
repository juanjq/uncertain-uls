"""The per-bin Cash TS map and the global statistic Lambda.

Counts and background are summed over energy and correlated with a top-hat of radius r_c,
giving per bin n_b ON counts and mu_b expected background counts, then

    TS_b   = sign(n_b - mu_b) * max(0, C(n_b, mu_b) - C(n_b, n_b)),   C(n, mu) = 2 (mu - n ln mu)
    Lambda = max over the mask of  TS_b + 2 ln p_b .

This is what gwuls.simulate._TSEngine computes through gammapy's ExcessMapEstimator
internals (`_get_convolved_maps` + `CashCountsStatistic`), written on plain arrays. The
correlation reproduces gammapy's: each energy plane times the safe mask, convolved with
`scipy.signal.convolve(method="fft", mode="same")` and the `Tophat2DKernel(r_c / binsz)`
scaled to a peak of 1, in the input's precision, result stored as float32; ON counts
rounded with `np.rint`.

Region of interest (ROI). The maximum is taken over the bins of the GW 95% credible mask that
are also observed: exposure summed over true energy > 0 and inside the safe mask in at least one
reco-energy bin (`roi`, mode "observed", the default since 2026-11-24). A bin of the GW region
without data has TS = 0 and would otherwise put a data-independent floor 2 ln p_b under Lambda.
Mode "gw_mask" is the GW mask alone, as in the reference code.

Fast path: the background does not change between realisations, so `TSPrecomputed` holds
its correlated sum once, and the counts are summed over energy before one 2D convolution.
For integer counts and a 0/1 kernel that equals the sum of per-plane rounded convolutions
exactly (the FFT noise is far below 0.5).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import convolve

CASH_TRUNCATION = 1e-25      # gammapy's TRUNCATION_VALUE for mu <= 0
PROB_FLOOR = 1e-20           # guards 2 ln p against log(0)


# --- correlation --------------------------------------------------------------------

def correlation_radius_bins(geom, radius_deg: float) -> float:
    """Top-hat radius in pixels: radius_deg / mean |CDELT| of `geom` (as gammapy)."""
    return float(radius_deg) / float(np.mean(np.abs(geom.wcs.wcs.cdelt)))


def tophat_kernel(radius_bins: float) -> np.ndarray:
    """astropy Tophat2DKernel(radius_bins) scaled to a peak of 1 (0/1 values), 2D float64."""
    from astropy.convolution import Tophat2DKernel

    k = Tophat2DKernel(float(radius_bins)).array
    return k / k.max()


def _as_cube(a) -> np.ndarray:
    a = np.asarray(a)
    return a[None] if a.ndim == 2 else a


def _masked(cube, mask):
    cube = _as_cube(cube)
    return cube if mask is None else cube * _as_cube(mask)


def _convolve_plane(image, kernel) -> np.ndarray:
    """One plane, in the input's own precision, stored as float32 (as WcsNDMap.convolve)."""
    return convolve(np.asarray(image), kernel, method="fft", mode="same").astype(np.float32)


def correlate_planes(cube, kernel) -> np.ndarray:
    """Each plane of a (n_e, ny, nx) cube convolved with `kernel`, float32 (as Map.convolve)."""
    cube = _as_cube(cube)
    out = np.empty(cube.shape, dtype=np.float32)
    for i, plane in enumerate(cube):
        out[i] = _convolve_plane(plane, kernel)
    return out


def correlated_background(background, kernel, mask=None) -> np.ndarray:
    """mu_b: sum over energy of the correlated (background x mask) planes. (ny, nx) float32."""
    return correlate_planes(_masked(background, mask), kernel).sum(axis=0)


def correlated_counts(counts, kernel, mask=None) -> np.ndarray:
    """n_b, exact gammapy order: rint of each correlated plane, then sum. (ny, nx) float32."""
    return np.rint(correlate_planes(_masked(counts, mask), kernel)).sum(axis=0)


def correlated_counts_fast(counts, kernel, mask=None) -> np.ndarray:
    """n_b with one 2D convolution of the energy-summed counts. Equal to `correlated_counts`
    for integer counts and a 0/1 kernel. (ny, nx) float32."""
    image = _masked(counts, mask).sum(axis=0)
    return np.rint(_convolve_plane(image, kernel))


# --- TS and Lambda ---------------------------------------------------------------------

def cash(n_on, mu):
    """Cash statistic 2 (mu - n ln mu), mu truncated at 1e-25 (as gammapy.stats.cash)."""
    n_on, mu = np.asanyarray(n_on), np.asanyarray(mu)
    mu = np.where(mu <= np.asarray(CASH_TRUNCATION), np.asarray(CASH_TRUNCATION), mu)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2 * (mu - n_on * np.log(mu))


def signed_ts(n_on, mu_bkg) -> np.ndarray:
    """Signed Cash TS per bin from correlated ON counts and background, same shape."""
    lik_alt = cash(n_on, n_on)
    lik_null = cash(n_on, mu_bkg)
    sign = np.where((n_on - mu_bkg) >= 0.0, +1.0, -1.0)
    return np.where((lik_null - lik_alt) < 0.0, 0.0, lik_null - lik_alt) * sign


def ts_map(counts, background, correlation_radius_bins, mask=None) -> np.ndarray:
    """Signed Cash TS map.

    counts, background: (n_e, ny, nx) or (ny, nx) arrays; mask: optional bool array of the
    same shape (the safe mask). correlation_radius_bins: top-hat radius in pixels.
    Returns (ny, nx) float64.
    """
    kernel = tophat_kernel(correlation_radius_bins)
    return signed_ts(correlated_counts(counts, kernel, mask),
                     correlated_background(background, kernel, mask))


def log_prob_term(prob) -> np.ndarray:
    """2 ln p_b with p floored at 1e-20, (ny, nx)."""
    return 2.0 * np.log(np.clip(np.asarray(prob, dtype=float), PROB_FLOOR, None))


def argmax_in_mask(values, mask):
    """Largest value of `values` inside `mask` and its (iy, ix). -> (float, (int, int))."""
    masked = np.where(np.asarray(mask, dtype=bool), values, np.nan)
    idx = np.unravel_index(np.nanargmax(masked), masked.shape)
    return float(masked[idx]), tuple(int(i) for i in idx)


def global_lambda(ts, prob, mask) -> float:
    """Lambda = max over `mask` of TS + 2 ln p. ts, prob, mask: (ny, nx)."""
    return argmax_in_mask(np.asarray(ts) + log_prob_term(prob), mask)[0]


# --- precomputed pieces for many realisations on one dataset ----------------------------

@dataclass
class TSPrecomputed:
    """What stays fixed across realisations: kernel, safe mask, correlated background,
    2 ln p and the region. Build it with `TSPrecomputed.build`."""
    kernel: np.ndarray           # (k, k) 0/1
    safe_mask: np.ndarray | None  # (n_e, ny, nx) bool
    mu_bkg: np.ndarray           # (ny, nx) correlated background
    log_prob: np.ndarray         # (ny, nx) 2 ln p
    mask: np.ndarray             # (ny, nx) bool, region of the maximum

    @classmethod
    def build(cls, background, prob, mask, correlation_radius_bins, safe_mask=None):
        """background: (n_e, ny, nx) expected counts; prob, mask: (ny, nx);
        safe_mask: optional (n_e, ny, nx) bool."""
        kernel = tophat_kernel(correlation_radius_bins)
        safe = None if safe_mask is None else np.asarray(safe_mask, dtype=bool)
        return cls(kernel=kernel, safe_mask=safe,
                   mu_bkg=correlated_background(background, kernel, safe),
                   log_prob=log_prob_term(prob), mask=np.asarray(mask, dtype=bool))

    def ts(self, counts) -> np.ndarray:
        """Signed TS map of a counts cube (n_e, ny, nx), fast path. (ny, nx)."""
        return signed_ts(correlated_counts_fast(counts, self.kernel, self.safe_mask), self.mu_bkg)

    def evaluate(self, counts) -> dict:
        """Lambda, the max TS in the mask, and the bins where each is reached."""
        ts = self.ts(counts)
        lam, i_lam = argmax_in_mask(ts + self.log_prob, self.mask)
        tsmax, i_ts = argmax_in_mask(ts, self.mask)
        return {"lambda": lam, "lambda_iy": i_lam[0], "lambda_ix": i_lam[1],
                "tsmax": tsmax, "tsmax_iy": i_ts[0], "tsmax_ix": i_ts[1]}


def background_cube(dataset) -> np.ndarray:
    """Expected background counts of a MapDataset with no source models, as its npred():
    npred_background with negatives set to 0. (n_e, ny, nx) float."""
    data = np.array(dataset.npred_background().data, dtype=float)
    data[data < 0.0] = 0
    return data


def safe_mask_cube(dataset):
    """The mask gammapy's ExcessMapEstimator uses: mask_fit & mask_safe, else mask_safe,
    else None (all bins)."""
    if dataset.mask_fit:
        return np.asarray(dataset.mask.data, dtype=bool)
    if dataset.mask_safe:
        return np.asarray(dataset.mask_safe.data, dtype=bool)
    return None


# --- region of interest ------------------------------------------------------------------

ROI_MODES = ("observed", "gw_mask")


def observed_region(dataset) -> np.ndarray:
    """Bins with data: exposure summed over true energy > 0 and, if the dataset has a safe mask,
    inside it in at least one reco-energy bin. (ny, nx) bool."""
    expo = np.nan_to_num(np.asarray(dataset.exposure.data, dtype=float)).sum(axis=0) > 0
    if dataset.mask_safe is not None:
        expo &= np.asarray(dataset.mask_safe.data, dtype=bool).any(axis=0)
    return expo


def roi(gw_mask, dataset, mode: str = "observed") -> np.ndarray:
    """The bins over which Lambda is maximised: GW mask & observed_region(dataset) ("observed"),
    or the GW mask alone ("gw_mask", reference behaviour). gw_mask: (ny, nx). -> (ny, nx) bool."""
    if mode not in ROI_MODES:
        raise ValueError(f"roi must be one of {ROI_MODES}, got {mode!r}")
    mask = np.asarray(gw_mask, dtype=bool)
    return mask & observed_region(dataset) if mode == "observed" else mask.copy()
