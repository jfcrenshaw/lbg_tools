"""Test the completeness class"""

import numpy as np
import pytest

from lbg_tools import Completeness, library


def test_extrapolation() -> None:
    """Test expected behavior for extrapolating completeness"""
    for band in library.get_bands():
        completeness = Completeness(band, 0)

        # Extrapolate to wide regimes and ensure vals always between (0, 1)
        vals = completeness(-100, np.linspace(-1, 10))
        assert np.all((vals >= 0) & (vals <= 1))

        vals = completeness(+100, np.linspace(-1, 10))
        assert np.all((vals >= 0) & (vals <= 1))

        # Furthermore, check that deep magnitudes are all zero
        assert np.allclose(vals, 0)

        # And that bright magnitudes are all the same
        vals0 = completeness(-100, np.linspace(-1, 10))
        vals1 = completeness(-90, np.linspace(-1, 10))
        assert np.allclose(vals0, vals1)


def test_cant_set_properties() -> None:
    """Make sure we can't set completeness properties after creation"""
    # Create object
    completeness = Completeness(library.get_bands()[0], 26)

    # Check that changing properties throws errors
    with pytest.raises(AttributeError):
        completeness.band = "fake"  # type: ignore
    with pytest.raises(AttributeError):
        completeness.m5_det = -99  # type: ignore


def test_load_missing_band() -> None:
    """Loading a missing band should throw error."""
    with pytest.raises(ValueError):
        Completeness("fake", 25)
