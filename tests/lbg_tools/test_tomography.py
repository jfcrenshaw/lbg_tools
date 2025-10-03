"""Test Tomographic Bin class"""

import numpy as np
import pytest

from lbg_tools import TomographicBin, library


def test_cant_set_properties() -> None:
    """Make sure we can't set properties after creation"""
    # Create tomographic bin object
    tbin = TomographicBin(library.get_bands()[0], 26)

    # Check that changing properties throws errors
    with pytest.raises(AttributeError):
        tbin.band = "fake"  # type: ignore
    with pytest.raises(AttributeError):
        tbin.mag_cut = -99  # type: ignore
    with pytest.raises(AttributeError):
        tbin.m5_det = -99  # type: ignore
    with pytest.raises(AttributeError):
        tbin.dz = -99  # type: ignore
    with pytest.raises(AttributeError):
        tbin.f_interlopers = -99  # type: ignore


def test_properties() -> None:
    """Test that bin properties run successfully"""
    tbin = TomographicBin(library.get_bands()[0], 26)
    tbin.nz
    tbin.number_density
    tbin.pz
    tbin.g_bias
    tbin.mag_bias


def test_min_redshift_zero() -> None:
    """Make sure negative redshifts aren't returned."""
    for band in library.get_bands():
        tbin = TomographicBin(band, 26, f_interlopers=0.2)
        z, _ = tbin.pz
        assert z.min() >= 0


def test_reasonable_mag_bias() -> None:
    """Test that the mag bias is reasonable."""
    u0 = TomographicBin("u", 24.5, 24.5, f_interlopers=0)
    assert np.isclose(u0.mag_bias, 2.7, atol=0.1)

    u0 = TomographicBin("u", 24.5, 24.5, f_interlopers=0.1)
    assert np.isclose(u0.mag_bias, 2.7, atol=0.1)


def test_dz() -> None:
    """Test that setting dz works."""
    u0 = TomographicBin("u", 24.5, 24.5, f_interlopers=0)
    u1 = TomographicBin("u", 24.5, 24.5, dz=0.3, f_interlopers=0)

    # Compare mean of redshift distributions
    z0, pz0 = u0.pz
    z1, pz1 = u1.pz
    mean0 = np.trapezoid(z0 * pz0, z0) / np.trapezoid(pz0, z0)
    mean1 = np.trapezoid(z1 * pz1, z1) / np.trapezoid(pz1, z1)
    assert mean1 > mean0


def test_stretch() -> None:
    """Test that stretching the p(z) works."""
    u0 = TomographicBin("u", 24.5, 24.5, f_interlopers=0)
    u1 = TomographicBin("u", 24.5, 24.5, stretch=2.0, f_interlopers=0)

    # Compare mean of redshift distributions
    z0, pz0 = u0.pz
    z1, pz1 = u1.pz
    mean0 = np.trapezoid(z0 * pz0, z0) / np.trapezoid(pz0, z0)
    mean1 = np.trapezoid(z1 * pz1, z1) / np.trapezoid(pz1, z1)
    assert mean1 > mean0

    # Check that number density is the same
    assert np.isclose(u1.number_density, u0.number_density)

    # Check that variance is greater
    var0 = np.trapezoid((z0 - mean0) ** 2 * pz0, z0) / np.trapezoid(pz0, z0)
    var1 = np.trapezoid((z1 - mean1) ** 2 * pz1, z1) / np.trapezoid(pz1, z1)
    assert var1 > var0
