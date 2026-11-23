"""Power-law spectra: band-integrated fluxes, amplitude <-> flux, flux <-> luminosity.

Ported unchanged from gwuls.simulate, Section 1.

Convention: dN/dE = phi0 (E / E0)^(-Gamma), with `index` = Gamma > 0 (the "PWL -2" spectrum
is index = 2.0) and `amplitude` = phi0 in cm-2 s-1 TeV-1 at E0 = `e_ref` (1 TeV). With
x = E / E0,

    photon flux  F_ph = phi0 E0   int_x1^x2 x^-G dx       (ln branch at G = 1)
    energy flux  F_E  = phi0 E0^2 int_x1^x2 x^(1-G) dx    (ln branch at G = 2)
    luminosity   L    = 4 pi d_L^2 F_E (1+z)^(G-2)

All luminosities are isotropic-equivalent and integrated over the same band as the flux
(rest frame); nothing is bolometric. Gamma = 2 is the singular case of the energy-flux
integral and is handled with the logarithm. At Gamma = 2 the K-correction is exactly 1.

Floats without units are read as: amplitude cm-2 s-1 TeV-1, photon flux cm-2 s-1, energy
flux erg cm-2 s-1, luminosity erg/s, distance Mpc, energies TeV.
"""
from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np

E_REF = 1.0 * u.TeV
AMPLITUDE_UNIT = u.Unit("cm-2 s-1 TeV-1")
_SINGULAR_TOL = 1e-8          # |1-G| or |2-G| below this -> use the log branch


# --- band integrals -----------------------------------------------------------

def _as_energy(value, default_unit=u.TeV):
    """Scalar, Quantity or sequence -> Quantity in TeV."""
    q = u.Quantity(value)
    if q.unit.is_unity():
        q = q * default_unit
    return q.to(u.TeV)


def _as_quantity(value, unit):
    """Float -> Quantity in `unit`; a Quantity is returned as is."""
    q = u.Quantity(value)
    return q * u.Unit(unit) if q.unit.is_unity() else q


def band(energy_edges):
    """(e_min, e_max) in TeV from a sequence of edges (first and last are used), validated."""
    edges = _as_energy(np.atleast_1d(u.Quantity(energy_edges)))
    if edges.size < 2:
        raise ValueError("`energy_edges` must contain at least two energies.")
    e_lo, e_hi = edges[0], edges[-1]
    if not np.isfinite(e_lo.value) or not np.isfinite(e_hi.value):
        raise ValueError(f"Non-finite energy edges: {edges}")
    if e_lo.value <= 0:
        raise ValueError(f"Energy edges must be > 0, got e_min = {e_lo}")
    if e_hi <= e_lo:
        raise ValueError(f"Need e_max > e_min, got [{e_lo}, {e_hi}]")
    return e_lo, e_hi


def _power_integral(x_lo, x_hi, exponent):
    """int_{x_lo}^{x_hi} x^exponent dx, with the exponent = -1 (log) branch explicit."""
    p = exponent + 1.0
    if abs(p) < _SINGULAR_TOL:
        return np.log(x_hi / x_lo)
    return (x_hi ** p - x_lo ** p) / p


def _band_x(energy_edges, e_ref):
    e_lo, e_hi = band(energy_edges)
    e0 = _as_energy(e_ref)
    return (e_lo / e0).to_value(""), (e_hi / e0).to_value(""), e0


def pwl_photon_flux(amplitude, energy_edges, index=2.0, e_ref=E_REF):
    """Photon flux int dN/dE dE over the band. amplitude: phi0 -> Quantity cm-2 s-1."""
    amp = _as_quantity(amplitude, AMPLITUDE_UNIT)
    x_lo, x_hi, e0 = _band_x(energy_edges, e_ref)
    return (amp * e0 * _power_integral(x_lo, x_hi, -float(index))).to(u.Unit("cm-2 s-1"))


def pwl_energy_flux(amplitude, energy_edges, index=2.0, e_ref=E_REF):
    """Energy flux int E dN/dE dE over the band. amplitude: phi0 -> Quantity erg cm-2 s-1."""
    amp = _as_quantity(amplitude, AMPLITUDE_UNIT)
    x_lo, x_hi, e0 = _band_x(energy_edges, e_ref)
    return (amp * e0 ** 2 * _power_integral(x_lo, x_hi, 1.0 - float(index))).to(
        u.Unit("erg cm-2 s-1"))


def pwl_e2dnde(amplitude, energy, index=2.0, e_ref=E_REF):
    """SED point E^2 dN/dE at `energy`. -> Quantity erg cm-2 s-1."""
    amp = _as_quantity(amplitude, AMPLITUDE_UNIT)
    e, e0 = _as_energy(energy), _as_energy(e_ref)
    return (amp * e ** 2 * (e / e0) ** (-float(index))).to(u.Unit("erg cm-2 s-1"))


def amplitude_from_photon_flux(photon_flux, energy_edges, index=2.0, e_ref=E_REF):
    """Inverse of `pwl_photon_flux`. photon_flux (cm-2 s-1) -> phi0 (float, cm-2 s-1 TeV-1)."""
    ref = pwl_photon_flux(1.0, energy_edges, index=index, e_ref=e_ref)
    return float((_as_quantity(photon_flux, "cm-2 s-1") / ref).to_value(""))


def amplitude_from_energy_flux(energy_flux, energy_edges, index=2.0, e_ref=E_REF):
    """Inverse of `pwl_energy_flux`. energy_flux (erg cm-2 s-1) -> phi0 (float)."""
    ref = pwl_energy_flux(1.0, energy_edges, index=index, e_ref=e_ref)
    return float((_as_quantity(energy_flux, "erg cm-2 s-1") / ref).to_value(""))


# --- distance, redshift, K-correction -------------------------------------------

_Z_INTERP = None


def redshift_from_distance(distance, cosmology=None):
    """Redshift for a luminosity distance (Planck18 by default).

    distance: float (Mpc), array or Quantity -> float or ndarray, same shape. Uses a
    cached monotone interpolation over z in [0, 5] (600 log-spaced nodes).
    """
    global _Z_INTERP
    d_mpc = np.atleast_1d(_as_quantity(distance, u.Mpc).to_value(u.Mpc)).astype(float)
    if cosmology is None:
        from astropy.cosmology import Planck18 as cosmology
    if _Z_INTERP is None or _Z_INTERP[2] is not cosmology:
        z_tab = np.concatenate([[0.0], np.logspace(-6, np.log10(5.0), 600)])
        d_tab = cosmology.luminosity_distance(z_tab).to_value(u.Mpc)
        _Z_INTERP = (d_tab, z_tab, cosmology)
    d_tab, z_tab, _ = _Z_INTERP
    z = np.interp(d_mpc, d_tab, z_tab)
    return z if np.ndim(distance) else float(z[0])


def k_correction(index, redshift):
    """(1+z)^(Gamma-2): L_band,rest = 4 pi d_L^2 F_E,band,obs * this. 1 if redshift is None."""
    if redshift is None:
        return 1.0
    return (1.0 + np.asarray(redshift, dtype=float)) ** (float(index) - 2.0)


def _redshift_for(distance, redshift, apply_k_correction):
    """None (no K-correction), the given z, or z from d_L when redshift == "auto"."""
    if not apply_k_correction:
        return None
    if isinstance(redshift, str) and redshift == "auto":
        return redshift_from_distance(distance)
    return redshift


# --- flux <-> luminosity -------------------------------------------------------------

def luminosity_from_energy_flux(energy_flux, distance, index=2.0, redshift=None,
                                apply_k_correction=True):
    """L = 4 pi d_L^2 F_E (1+z)^(G-2). -> Quantity erg/s.

    redshift: None (no K-correction), "auto" (z from d_L, Planck18) or a value.
    """
    f_e = _as_quantity(energy_flux, "erg cm-2 s-1")
    d = _as_quantity(distance, u.Mpc)
    z = _redshift_for(d, redshift, apply_k_correction)
    return ((4.0 * np.pi * d.to(u.cm) ** 2 * f_e) * k_correction(index, z)).to(u.erg / u.s)


def energy_flux_from_luminosity(luminosity, distance, index=2.0, redshift=None,
                                apply_k_correction=True):
    """Inverse of `luminosity_from_energy_flux`. -> Quantity erg cm-2 s-1."""
    lum = _as_quantity(luminosity, u.erg / u.s)
    d = _as_quantity(distance, u.Mpc)
    z = _redshift_for(d, redshift, apply_k_correction)
    return (lum / (4.0 * np.pi * d.to(u.cm) ** 2 * k_correction(index, z))).to(
        u.Unit("erg cm-2 s-1"))


def amplitude_from_luminosity(luminosity, distance, energy_edges, index=2.0, e_ref=E_REF,
                              redshift=None, apply_k_correction=True):
    """Band luminosity L0 (erg/s, in `energy_edges`, rest frame) and d_L -> phi0.

    Returns a float, or an ndarray when `distance` / `luminosity` are arrays (broadcast).
    """
    f_e = energy_flux_from_luminosity(luminosity, distance, index=index, redshift=redshift,
                                      apply_k_correction=apply_k_correction)
    amp = (f_e / pwl_energy_flux(1.0, energy_edges, index=index, e_ref=e_ref)).to_value("")
    return float(amp) if np.ndim(amp) == 0 else np.asarray(amp, dtype=float)


def luminosity_from_amplitude(amplitude, distance, energy_edges, index=2.0, e_ref=E_REF,
                              redshift=None, apply_k_correction=True):
    """Inverse of `amplitude_from_luminosity`. -> Quantity erg/s."""
    f_e = pwl_energy_flux(amplitude, energy_edges, index=index, e_ref=e_ref)
    return luminosity_from_energy_flux(f_e, distance, index=index, redshift=redshift,
                                       apply_k_correction=apply_k_correction)


@dataclass
class FluxPoint:
    """Every flux-like quantity implied by one PWL amplitude."""
    amplitude: float                 # phi0 [cm-2 s-1 TeV-1] at e_ref
    index: float
    e_ref: u.Quantity
    e_min: u.Quantity
    e_max: u.Quantity
    photon_flux: u.Quantity          # cm-2 s-1
    energy_flux: u.Quantity          # erg cm-2 s-1
    e_dec: u.Quantity                # sqrt(e_min e_max)
    e2dnde: u.Quantity               # erg cm-2 s-1 at e_dec

    def __str__(self):
        return (
            f"phi0        = {self.amplitude:.3e} cm-2 s-1 TeV-1 @ {self.e_ref:.2f}\n"
            f"photon flux = {self.photon_flux.value:.3e} cm-2 s-1        "
            f"[{self.e_min:.2f}-{self.e_max:.2f}]\n"
            f"energy flux = {self.energy_flux.value:.3e} erg cm-2 s-1    "
            f"[{self.e_min:.2f}-{self.e_max:.2f}]\n"
            f"E2 dN/dE    = {self.e2dnde.value:.3e} erg cm-2 s-1 @ {self.e_dec:.2f}"
        )


def flux_point(amplitude, energy_edges, index=2.0, e_ref=E_REF) -> FluxPoint:
    """phi0 -> FluxPoint (photon flux, energy flux, SED point at the log-centre of the band)."""
    e_lo, e_hi = band(energy_edges)
    e_dec = np.sqrt(e_lo * e_hi)
    return FluxPoint(
        amplitude=float(amplitude), index=float(index), e_ref=_as_energy(e_ref),
        e_min=e_lo, e_max=e_hi,
        photon_flux=pwl_photon_flux(amplitude, (e_lo, e_hi), index, e_ref),
        energy_flux=pwl_energy_flux(amplitude, (e_lo, e_hi), index, e_ref),
        e_dec=e_dec, e2dnde=pwl_e2dnde(amplitude, e_dec, index, e_ref),
    )


# --- self-check ---------------------------------------------------------------------

def check_conversions(energy_edges=(0.6 * u.TeV, 20 * u.TeV), indices=(1.0, 1.5, 2.0, 2.5, 3.0),
                      amplitude=1e-12, distance=400 * u.Mpc, rtol=1e-9, verbose=True) -> list:
    """Check the conversions against independent computations.

    (a) band integrals vs numerical quadrature, (b) vs gammapy's PowerLawSpectralModel,
    (c) phi0 -> L -> phi0 and phi0 -> F_ph -> phi0 round trips, (d) the Gamma = 2 log branch
    vs the limit from both sides, (e) K-correction = 1 at Gamma = 2, (1+z)^0.5 at 2.5, L ~ d^2.

    Returns a list of (name, value, reference, rel_error, passed); raises AssertionError if
    any check fails. Prints one line per check if `verbose`.
    """
    from gammapy.modeling.models import PowerLawSpectralModel
    from scipy.integrate import quad

    e_lo, e_hi = band(energy_edges)
    e0 = E_REF
    rows = []

    def report(name, a, b, tol=rtol):
        rel = abs(a - b) / max(abs(b), 1e-300)
        rows.append((name, float(a), float(b), float(rel), bool(rel < tol)))
        if verbose:
            print(f"{'PASS' if rel < tol else 'FAIL'}  {name:<48s} {a:.8e} vs {b:.8e}  (rel {rel:.1e})")

    edges = (e_lo, e_hi)
    for g in indices:
        dnde = lambda e: amplitude * (e / e0.value) ** (-g)       # noqa: E731
        qkw = dict(limit=400, epsabs=0.0, epsrel=1e-13)
        num_ph = quad(dnde, e_lo.value, e_hi.value, **qkw)[0]
        num_en = (quad(lambda e: e * dnde(e), e_lo.value, e_hi.value, **qkw)[0] * u.TeV).to_value(u.erg)
        ana_ph = pwl_photon_flux(amplitude, edges, g).value
        ana_en = pwl_energy_flux(amplitude, edges, g).value
        report(f"G={g}: photon flux vs quad", ana_ph, num_ph)
        report(f"G={g}: energy flux vs quad", ana_en, num_en)
        pwl = PowerLawSpectralModel(amplitude=amplitude * AMPLITUDE_UNIT, index=g, reference=e0)
        report(f"G={g}: photon flux vs gammapy", ana_ph, pwl.integral(e_lo, e_hi).to_value("cm-2 s-1"), 1e-5)
        report(f"G={g}: energy flux vs gammapy", ana_en,
               pwl.energy_flux(e_lo, e_hi).to_value("erg cm-2 s-1"), 1e-5)
        lum = luminosity_from_amplitude(amplitude, distance, edges, index=g)
        report(f"G={g}: phi0 -> L -> phi0", amplitude_from_luminosity(lum, distance, edges, index=g), amplitude)
        report(f"G={g}: phi0 -> F_ph -> phi0", amplitude_from_photon_flux(ana_ph, edges, g), amplitude)

    eps = 1e-6
    f2 = pwl_energy_flux(amplitude, edges, 2.0).value
    fm = pwl_energy_flux(amplitude, edges, 2.0 - eps).value
    fp = pwl_energy_flux(amplitude, edges, 2.0 + eps).value
    report("G=2 log branch == limit from both sides", f2, 0.5 * (fm + fp), 1e-8)
    closed = (amplitude * e0.value ** 2 * np.log((e_hi / e_lo).to_value("")) * u.TeV).to_value(u.erg)
    report("G=2 log branch == phi0 E0^2 ln(E2/E1)", f2, closed, 1e-10)

    z = 0.09
    lum = lambda g, **kw: luminosity_from_amplitude(amplitude, distance, edges, index=g, **kw).value  # noqa: E731
    report("K-correction is 1 at G=2", lum(2.0, redshift=z), lum(2.0, apply_k_correction=False), 1e-12)
    report("K-correction = (1+z)^0.5 at G=2.5", lum(2.5, redshift=z) / lum(2.5, apply_k_correction=False),
           (1 + z) ** 0.5, 1e-10)
    l1 = luminosity_from_amplitude(amplitude, 100 * u.Mpc, edges).value
    l2 = luminosity_from_amplitude(amplitude, 200 * u.Mpc, edges).value
    report("L scales as d^2", l2 / l1, 4.0, 1e-10)

    if not all(r[-1] for r in rows):
        raise AssertionError("spectra.check_conversions: at least one check failed.")
    return rows
