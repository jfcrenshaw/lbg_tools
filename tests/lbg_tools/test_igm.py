"""Test the IGM class."""

import numpy as np
import pytest

from lbg_tools import IGM


def test_bad_model() -> None:
    """Test that bad model name raises error."""
    with pytest.raises(ValueError):
        IGM("fake")


def test_inoue_less_than_madau() -> None:
    """The Inoue model should have less extinction than Madau."""
    wavelen = np.linspace(0, 10_000, 10_000)
    z = 4
    inoue = IGM().transmission(wavelen, z)
    madau = IGM("madau").transmission(wavelen, z)
    assert np.all(madau <= inoue)


def test_limits() -> None:
    """Test values in appropriate limits."""
    for model in ["inoue", "madau"]:
        igm = IGM(model)
        for z in np.arange(3, 7, 0.5):
            # No transmission at very low wavelengths
            assert igm.transmission(0.0, z) < 0.1  # type: ignore

            # Full transmission at very high wavelengths
            assert np.isclose(igm.transmission(1e9, z), 1)  # type: ignore


def test_decrease_with_redshift() -> None:
    """Test that transmission decreases with redshift."""
    for model in ["inoue", "madau"]:
        igm = IGM(model)
        wavelen = np.linspace(0, 10_000)

        # Calculate transmission at different redshifts
        trans = []
        for z in np.arange(3, 7, 0.5):
            trans.append(igm.transmission(wavelen, z))  # type: ignore

        # Make sure they are decreasing
        diffs = np.diff(trans, axis=0)
        assert np.all(diffs <= 0)
