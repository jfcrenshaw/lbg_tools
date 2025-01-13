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


def test_redshift_grids() -> None:
    """Make sure all the redshift grids are properly concatenated."""
    # First make sure that no interlopers means no low-z grid concatenated
    tbin = TomographicBin(library.get_bands()[0], 26)
    zi, zt = tbin._get_z_grids()
    z = zt

    z_, _ = tbin.nz
    assert np.allclose(z_, z)

    z_, _ = tbin.pz
    assert np.allclose(z_, z)

    z_, _ = tbin.pz
    assert np.allclose(z_, z)

    z_, _ = tbin.g_bias
    assert np.allclose(z_, z)

    # Now add interlopers and make sure the low-z grid WAS concatentated
    tbin = TomographicBin(library.get_bands()[0], 26, f_interlopers=0.1)
    zi, zt = tbin._get_z_grids()
    z = np.concatenate((zi, zt))

    z_, _ = tbin.nz
    assert np.allclose(z_, z)

    z_, _ = tbin.pz
    assert np.allclose(z_, z)

    z_, _ = tbin.pz
    assert np.allclose(z_, z)

    z_, _ = tbin.g_bias
    assert np.allclose(z_, z)


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
