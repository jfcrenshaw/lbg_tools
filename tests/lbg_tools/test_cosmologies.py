"""Test some calculations with different cosmologies."""

import numpy as np
import pytest

from lbg_tools.cosmo_utils import (
    check_cosmology,
    diff_comoving_volume,
    luminosity_distance,
)

# Protected import for optional dependency
try:
    import pyccl as ccl
except ImportError:
    ccl = None

import astropy.cosmology

from lbg_tools import SED, LuminosityFunction, TomographicBin

# Set-up cosmologies for testing

# Default astropy cosmology
cosmo_astropy = astropy.cosmology.Planck18

# Default CCL cosmology constructed to match default astropy
if ccl is None:
    cosmo_ccl = None
else:
    cosmo_ccl = ccl.Cosmology(
        Omega_c=cosmo_astropy.Odm0,
        Omega_b=cosmo_astropy.Ob0,
        h=cosmo_astropy.h,
        T_CMB=cosmo_astropy.Tcmb0.value,
        Neff=cosmo_astropy.Neff,
        m_nu=cosmo_astropy.m_nu.value,
        sigma8=0.81,
        n_s=0.96,
    )

# A different cosmology that should yield different calculations
cosmo_diff = astropy.cosmology.WMAP1


def test_check_cosmology() -> None:
    """Test that the check cosmology function works."""
    check_cosmology(cosmo_astropy)
    check_cosmology(cosmo_diff)
    if cosmo_ccl is not None:
        check_cosmology(cosmo_ccl)
    with pytest.raises(TypeError):
        check_cosmology(None)
    with pytest.raises(TypeError):
        check_cosmology("LCDM")


def test_luminosity_distance() -> None:
    """Test that luminosity distances agree."""
    # Shared params
    z = np.linspace(0, 6, 100)

    # First check astropy and ccl calculations agree
    if cosmo_ccl is not None:
        assert np.allclose(
            luminosity_distance(cosmo_astropy, z),
            luminosity_distance(cosmo_ccl, z),
        )

    # Now check that different cosmologies impact calculations
    with pytest.raises(AssertionError):
        assert np.allclose(
            luminosity_distance(cosmo_astropy, z),
            luminosity_distance(cosmo_diff, z),
        )


def test_diff_comoving_volume() -> None:
    """Test if differential comoving volumes agree."""
    # Shared params
    z = np.linspace(0, 6, 2)

    # First check astropy and ccl calculations agree
    if cosmo_ccl is not None:
        assert np.allclose(
            diff_comoving_volume(cosmo_astropy, z),
            diff_comoving_volume(cosmo_ccl, z),
            rtol=1e-4,
        )

    # Now check that different cosmologies impact calculations
    with pytest.raises(AssertionError):
        assert np.allclose(
            diff_comoving_volume(cosmo_astropy, z),
            diff_comoving_volume(cosmo_diff, z),
            rtol=1e-4,
        )


def test_luminosity_function() -> None:
    """Test if luminosity functions agree"""
    # Shared params
    M = np.linspace(-30, -10)
    z = np.linspace(2, 6)

    # First check astropy and ccl calculations agree
    if cosmo_ccl is not None:
        assert np.allclose(
            LuminosityFunction(cosmology=cosmo_astropy)(M, z),
            LuminosityFunction(cosmology=cosmo_ccl)(M, z),
        )


def test_sed() -> None:
    """Test SED observed values agree."""
    # Shared params
    z = 3.2
    M = -22

    # First check astropy and ccl calculations agree
    if cosmo_ccl is not None:
        assert np.allclose(
            SED(z, M, cosmology=cosmo_astropy).observed[1],
            SED(z, M, cosmology=cosmo_ccl).observed[1],
        )

    # Now check that different cosmologies impact calculations
    with pytest.raises(AssertionError):
        assert np.allclose(
            SED(z, M, cosmology=cosmo_astropy).observed[1],
            SED(z, M, cosmology=cosmo_diff).observed[1],
            atol=1e-30,
        )


def test_tomography() -> None:
    """Test tomographic number densities."""
    # Shared params
    band = "u"
    mag_cut = 25

    # First check astropy and ccl calculations agree
    if cosmo_ccl is not None:
        assert np.allclose(
            TomographicBin(band, mag_cut, cosmology=cosmo_astropy).nz[1],
            TomographicBin(band, mag_cut, cosmology=cosmo_ccl).nz[1],
            rtol=1e-4,
        )

    # Now check that different cosmologies impact calculations
    with pytest.raises(AssertionError):
        assert np.allclose(
            TomographicBin(band, mag_cut, cosmology=cosmo_astropy).nz[1],
            TomographicBin(band, mag_cut, cosmology=cosmo_diff).nz[1],
            rtol=1e-4,
        )
