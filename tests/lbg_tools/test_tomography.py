"""Test Tomographic Bin class"""

import pytest

from lbg_tools import TomographicBin, data


def test_cant_set_properties() -> None:
    """Make sure we can't set properties after creation"""
    # Create tomographic bin object
    tbin = TomographicBin(data.bands[0], 26)

    # Check that changing properties throws errors
    with pytest.raises(AttributeError):
        tbin.band = "fake"  # type: ignore
    with pytest.raises(AttributeError):
        tbin.mag_cut = -99  # type: ignore
    with pytest.raises(AttributeError):
        tbin.m5_det = -99  # type: ignore


def test_properties() -> None:
    """Test that bin properties run successfully"""
    tbin = TomographicBin(data.bands[0], 26)
    tbin.nz
    tbin.number_density
    tbin.pz
