import sys
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import toy  # noqa: E402

from uuls import spectra  # noqa: E402

REF = toy.reference_simulate()
needs_ref = pytest.mark.skipif(REF is None, reason="reference repo gw-gamma-global-uls not found")
BAND = (0.6 * u.TeV, 20 * u.TeV)


def test_self_check_passes():
    rows = spectra.check_conversions(verbose=False)
    assert all(r[-1] for r in rows)


def test_gamma2_energy_flux_closed_form():
    f = spectra.pwl_energy_flux(1e-12, BAND, 2.0)
    expected = (1e-12 * np.log(20 / 0.6) * u.TeV).to_value(u.erg)
    assert f.unit == u.Unit("erg cm-2 s-1")
    assert f.value == pytest.approx(expected, rel=1e-14)


def test_redshift_planck18():
    from astropy.cosmology import Planck18

    z = spectra.redshift_from_distance(np.array([100.0, 400.0, 1500.0]))
    d_back = Planck18.luminosity_distance(z).to_value(u.Mpc)
    np.testing.assert_allclose(d_back, [100.0, 400.0, 1500.0], rtol=1e-4)


@needs_ref
@pytest.mark.parametrize("index", [1.0, 1.7, 2.0, 2.5, 3.0])
def test_regression_fluxes(index):
    amp = 2.34e-12
    for new, old in [(spectra.pwl_photon_flux, REF.pwl_photon_flux),
                     (spectra.pwl_energy_flux, REF.pwl_energy_flux)]:
        assert new(amp, BAND, index) == old(amp, BAND, index)
    assert spectra.pwl_e2dnde(amp, 3 * u.TeV, index) == REF.pwl_e2dnde(amp, 3 * u.TeV, index)
    assert (spectra.amplitude_from_photon_flux(1e-12, BAND, index)
            == REF.amplitude_from_photon_flux(1e-12, BAND, index))
    assert (spectra.amplitude_from_energy_flux(1e-12, BAND, index)
            == REF.amplitude_from_energy_flux(1e-12, BAND, index))


@needs_ref
@pytest.mark.parametrize("index,z,k", [(2.0, None, True), (2.5, 0.09, True), (2.5, "auto", True),
                                        (2.5, "auto", False)])
def test_regression_luminosity(index, z, k):
    d = np.array([40.0, 400.0, 1400.0]) * u.Mpc
    kw = dict(index=index, redshift=z, apply_k_correction=k)
    lum_new = spectra.luminosity_from_amplitude(1e-12, d, BAND, **kw)
    lum_old = REF.luminosity_from_amplitude(1e-12, d, BAND, **kw)
    np.testing.assert_array_equal(lum_new.value, lum_old.value)
    np.testing.assert_array_equal(spectra.amplitude_from_luminosity(1e45, d, BAND, **kw),
                                  REF.amplitude_from_luminosity(1e45, d, BAND, **kw))
    np.testing.assert_array_equal(spectra.redshift_from_distance(d),
                                  REF.redshift_from_luminosity_distance(d))
    assert spectra.k_correction(index, 0.3) == REF._k_correction(index, 0.3)


@needs_ref
def test_regression_flux_point():
    new, old = spectra.flux_point(3e-12, BAND), REF.flux_point_from_amplitude(3e-12, BAND)
    assert str(new) == str(old)
